# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import argparse
import logging
import gradio as gr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from karlo import __version__ as karlo_ver
from demo.components import GradioSampler, ImageSelecter


class GradioDemo:
    def __init__(
        self,
        root_dir: str,
        max_bsz: int,
        progressive: str,
        sampling_type: str,
    ):
        sampler = GradioSampler(
            root_dir=root_dir,
            max_bsz=max_bsz,
            progressive=progressive,
            sampling_type=sampling_type,
        )

        demo = gr.Blocks()
        with demo:
            gr.Markdown(f"# Karlo Demo {karlo_ver}")
            with gr.Box():
                gr.Markdown("## Generate 64px images + Upscaling to 256px")

                with gr.Tabs():
                    with gr.TabItem("Image Generation"):
                        t2i_text_input = gr.Textbox(
                            lines=1,
                            placeholder="Type text prompt...",
                            label="Text prompts",
                        )
                        t2i_button = gr.Button("Generate", variant="primary")
                    with gr.TabItem("Image Variation"):
                        i2i_img_input = gr.Image(label="Image input", type="pil")
                        i2i_button = gr.Button("Generate", variant="primary")

            with gr.Box():
                outputs = gr.Image(label="Generated", type="pil")
                stash = gr.Variable()
                with gr.Row():
                    selector_ui = ImageSelecter.make_basic_ui(max_bsz=max_bsz)

            with gr.Box():
                with gr.Accordion(label="Advanced Options", open=False):
                    sampler.make_basic_options()

            with gr.Box():
                with gr.Accordion(label="Checkpoint Information", open=False):
                    gr.Markdown(sampler.ckpt_info)

            t2i_button.click(
                fn=sampler.t2i_sample,
                inputs=[t2i_text_input]
                + sampler.prior_optios_gr
                + sampler.decoder_options_gr
                + sampler.sr_64_256_options_gr
                + sampler.global_options_gr,
                outputs=[stash, outputs],
            )
            i2i_button.click(
                fn=sampler.i2i_sample,
                inputs=[i2i_img_input]
                + sampler.decoder_options_gr
                + sampler.sr_64_256_options_gr
                + sampler.global_options_gr,
                outputs=[stash, outputs],
            )

            ImageSelecter.setup_button_click(selector_ui, stash, i2i_img_input)

        demo.queue()
        self.demo = demo


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--max_bsz", type=int, default=4)
    parser.add_argument(
        "--progressive", type=str, default="loop", choices=("loop", "stage", "final")
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6006)

    parser.add_argument(
        "--sampling-type",
        type=str,
        default="fast",
        choices=("fast", "default"),
    )

    return parser


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    assert (
        args.root_dir is not None
    ), "--root-dir argument should be specified to load the pretrained ckpt"

    """Making Gradio"""
    gradio_demo = GradioDemo(
        root_dir=args.root_dir,
        max_bsz=args.max_bsz,
        progressive=args.progressive,
        sampling_type=args.sampling_type,
    )
    gradio_demo.demo.launch(server_name=args.host, server_port=args.port)
