# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import time
import sys
import os
import threading
import logging
from queue import Queue
from PIL import Image

import gradio as gr
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from karlo.sampler.template import CKPT_PATH, BaseSampler
from karlo.sampler.t2i import T2ISampler
from karlo.sampler.i2i import I2ISampler
from karlo.utils.util import set_seed


def tensor_to_images(tensor: torch.Tensor, output_res=(1024, 1024)):
    assert tensor.ndim == 4
    tensor = torch.clone(tensor)
    # NCHW -> NHWC
    images = torch.permute(tensor * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy()
    concat_image = np.concatenate(images, axis=1)
    target_size = (output_res[1] * tensor.shape[0], output_res[0])
    concat_image = Image.fromarray(concat_image).resize(
        target_size, resample=Image.NEAREST
    )
    return images, concat_image


class GradioSampler:
    def __init__(
        self,
        root_dir,
        max_bsz,
        progressive,
        sampling_type: str,
    ):
        self._root_dir = root_dir
        self._max_bsz = max_bsz
        self._progressive = progressive
        self._sampling_type = sampling_type

        self.load_ckpt()
        self.set_options_from_sampler()

        self.result_queue = Queue()

    def load_ckpt(self):
        base_sampler = BaseSampler(root_dir=self._root_dir)
        base_sampler.load_clip(clip_path="ViT-L-14.pt")
        base_sampler.load_prior(
            f"{CKPT_PATH['prior']}",
            clip_stat_path="ViT-L-14_stats.th",
        )
        base_sampler.load_decoder(f"{CKPT_PATH['decoder']}")
        base_sampler.load_sr_64_256(f"{CKPT_PATH['sr_256']}")

        self.t2i_sampler = T2ISampler(
            root_dir=self._root_dir, sampling_type=self._sampling_type
        )
        self.i2i_sampler = I2ISampler(
            root_dir=self._root_dir, sampling_type=self._sampling_type
        )

        self.t2i_sampler._clip = base_sampler._clip
        self.t2i_sampler._tokenizer = base_sampler._tokenizer
        self.t2i_sampler._prior = base_sampler._prior
        self.t2i_sampler._decoder = base_sampler._decoder
        self.t2i_sampler._sr_64_256 = base_sampler._sr_64_256

        self.i2i_sampler._clip = base_sampler._clip
        self.i2i_sampler._tokenizer = base_sampler._tokenizer
        self.i2i_sampler._prior = base_sampler._prior
        self.i2i_sampler._decoder = base_sampler._decoder
        self.i2i_sampler._sr_64_256 = base_sampler._sr_64_256

        self.ckpt_info = f"""
        * **prior**: `{self._root_dir}/{CKPT_PATH['prior']}`
        * **decoder**: `{self._root_dir}/{CKPT_PATH['decoder']}`
        * **sr_64_256**: `{self._root_dir}/{CKPT_PATH['sr_256']}`
        """

    def set_options_from_sampler(self):
        self.global_options = {"seed": 0, "max_bsz": self._max_bsz}

        self.prior_options = {
            "sm": self.t2i_sampler._prior_sm,
            "cf_scale": self.t2i_sampler._prior_cf_scale,
        }
        self.decoder_options = {
            "sm": self.t2i_sampler._decoder_sm,
            "cf_scale": self.t2i_sampler._decoder_cf_scale,
        }
        self.sr_64_256_options = {
            "sm": self.t2i_sampler._sr_sm,
        }

    def make_global_options(self):
        gr.Markdown("Global Options")
        with gr.Row():
            return [
                gr.Slider(
                    label="seed",
                    value=self.global_options["seed"],
                    minimum=np.iinfo(np.uint32).min,
                    maximum=np.iinfo(np.uint32).max,
                    step=1,
                ),
                gr.Slider(
                    label="maximum batch size",
                    value=self.global_options["max_bsz"],
                    minimum=1,
                    maximum=4,
                    step=1,
                ),
            ]

    def make_prior_options(self):
        gr.Markdown("Prior Options")
        return [
            gr.Textbox(
                label="sampling method",
                value=self.prior_options["sm"],
            ),
            gr.Slider(
                label="Classifier-free guidance scales",
                value=self.prior_options["cf_scale"],
                minimum=0.1,
                maximum=24,
            ),
        ]

    def make_decoder_options(self):
        gr.Markdown("Decoder Options")
        with gr.Row():
            return [
                gr.Textbox(
                    label="sampling method",
                    value=self.decoder_options["sm"],
                ),
                gr.Slider(
                    label="Classifier-free guidance scales",
                    value=self.decoder_options["cf_scale"],
                    minimum=0.1,
                    maximum=24,
                ),
            ]

    def make_sr_64_256_options(self):
        return [gr.Variable(self.sr_64_256_options["sm"])]

    def make_basic_options(self):
        self.global_options_gr = self.make_global_options()
        self.prior_optios_gr = self.make_prior_options()
        self.decoder_options_gr = self.make_decoder_options()
        self.sr_64_256_options_gr = self.make_sr_64_256_options()

    def seed(self, seed):
        set_seed(seed)

    def _sample(self, output_generator):
        for k, out in enumerate(output_generator):
            self.result_queue.put((out, False))
        self.result_queue.put((None, True))

    def t2i_sample(
        self,
        text_input,
        prior_sm,
        prior_cf_scale,
        decoder_sm,
        decoder_cf_scale,
        sr_sm,
        seed,
        max_bsz,
    ):
        t0 = time.time()
        assert hasattr(self.t2i_sampler, "_prior_sm")
        assert hasattr(self.t2i_sampler, "_prior_cf_scale")
        assert hasattr(self.t2i_sampler, "_decoder_sm")
        assert hasattr(self.t2i_sampler, "_decoder_cf_scale")
        assert hasattr(self.t2i_sampler, "_sr_sm")

        print("-" * 100)
        print(f"text_input: {text_input}")
        print(f"prior_sm: {prior_sm}")
        print(f"prior_cf_scale: {prior_cf_scale}")
        print(f"decoder_sm: {decoder_sm}")
        print(f"decoder_cf_scale: {decoder_cf_scale}")
        print(f"sr_sm: {sr_sm}")
        print(f"seed: {seed}")
        print(f"max_bsz: {max_bsz}")

        self.t2i_sampler._prior_sm = prior_sm
        self.t2i_sampler._prior_cf_scale = prior_cf_scale

        self.t2i_sampler._decoder_sm = decoder_sm
        self.t2i_sampler._decoder_cf_scale = decoder_cf_scale

        self.t2i_sampler._sr_sm = sr_sm

        self.seed(seed)

        output_generator = self.t2i_sampler(
            prompt=text_input,
            bsz=max_bsz,
            progressive_mode=self._progressive,
        )

        thread = threading.Thread(target=self._sample, args=(output_generator,))
        thread.start()
        done = False

        while not done:
            if self.result_queue.empty():
                time.sleep(0.1)
            else:
                while not self.result_queue.empty():
                    _out, done = self.result_queue.get(0)  # get last item to display
                    if not done:
                        out = _out
                images, concat_image = tensor_to_images(out, (256, 256))
                yield (text_input, images), concat_image

        thread.join()
        yield (text_input, images), concat_image

        t1 = time.time()
        execution_time = t1 - t0
        logging.info(f"Generation done. {text_input} -- {execution_time:.6f}secs")
        print("-" * 100)

    def i2i_sample(
        self,
        image_input,
        decoder_sm,
        decoder_cf_scale,
        sr_sm,
        seed,
        max_bsz,
    ):
        t0 = time.time()
        assert hasattr(self.i2i_sampler, "_decoder_sm")
        assert hasattr(self.i2i_sampler, "_decoder_cf_scale")
        assert hasattr(self.i2i_sampler, "_sr_sm")

        print("-" * 100)
        print(f"decoder_sm: {decoder_sm}")
        print(f"decoder_cf_scale: {decoder_cf_scale}")
        print(f"sr_sm: {sr_sm}")
        print(f"seed: {seed}")
        print(f"max_bsz: {max_bsz}")

        self.i2i_sampler._decoder_sm = decoder_sm
        self.i2i_sampler._decoder_cf_scale = decoder_cf_scale

        self.i2i_sampler._sr_sm = sr_sm

        self.seed(seed)

        output_generator = self.i2i_sampler(
            image=image_input,
            bsz=max_bsz,
            progressive_mode=self._progressive,
        )

        thread = threading.Thread(target=self._sample, args=(output_generator,))
        thread.start()
        done = False

        while not done:
            if self.result_queue.empty():
                time.sleep(0.1)
            else:
                while not self.result_queue.empty():
                    _out, done = self.result_queue.get(0)  # get last item to display
                    if not done:
                        out = _out
                images, concat_image = tensor_to_images(out, (256, 256))
                yield ("", images), concat_image

        thread.join()
        yield ("", images), concat_image

        t1 = time.time()
        execution_time = t1 - t0
        logging.info(f"Variation done. {execution_time:.6f}secs")
        print("-" * 100)


class ImageSelecter:
    @classmethod
    def make_basic_ui(cls, max_bsz):
        with gr.Box():
            i2i_select_idx = gr.Radio(
                choices=[str(i) for i in range(0, max_bsz)],
                value="0",
                label="Image index",
            )
            i2i_select_button = gr.Button(
                "Select for Image Variation", variant="primary"
            )
        return {
            "i2i_select_idx": i2i_select_idx,
            "i2i_select_button": i2i_select_button,
        }

    @classmethod
    def select_fn(cls, stash, idx):
        if stash is not None:
            return Image.fromarray(stash[1][int(idx)].copy())

    @classmethod
    def setup_button_click(
        cls,
        selector_ui,
        stash,
        i2i_input_images,
    ):
        selector_ui["i2i_select_button"].click(
            fn=cls.select_fn,
            inputs=[stash, selector_ui["i2i_select_idx"]],
            outputs=[i2i_input_images],
        )
