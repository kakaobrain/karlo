# ------------------------------------------------------------------------------------
# Karlo-v1.0.alpha
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import logging
import time
from datetime import datetime

import torch
from PIL import Image

from karlo.sampler.t2i import T2ISampler
from karlo.utils.util import set_seed


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", type=str, required=True, help="path for model checkpoints"
    )
    parser.add_argument("--max-bsz", type=int, default=1, help="#images to generate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="output path for generated images",
    )
    parser.add_argument(
        "--sampling-type",
        type=str,
        default="fast",
        choices=("fast", "default"),
    )
    parser.add_argument(
        "--prompt", type=str, default="A photo of a baby puppy waiting for her mom."
    )
    parser.add_argument("--seed", type=int, default=0)

    return parser


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    logging.getLogger().setLevel(logging.INFO)

    save_dir = os.path.join(args.output_dir, datetime.now().strftime("%d%m%Y_%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = T2ISampler.from_pretrained(
        root_dir=args.root_dir,
        clip_model_path="ViT-L-14.pt",
        clip_stat_path="ViT-L-14_stats.th",
        sampling_type=args.sampling_type,
    )

    for i in range(5):
        t1 = time.time()

        images = iter(
            model(
                prompt=args.prompt,
                bsz=args.max_bsz,
                progressive_mode="final",
            )
        ).__next__()

        # NCHW, [0, 1], float32 -> NHWC, [0, 255], uint8
        images = (
            torch.permute(images * 255.0, [0, 2, 3, 1]).type(torch.uint8).cpu().numpy()
        )

        t2 = time.time()
        execution_time = t2 - t1
        logging.info(f"Iteration {i} -- {execution_time:.6f}secs")

        # Select the first one
        image = Image.fromarray(images[0])
        image_name = "_".join(args.prompt.split(" "))
        image.save(f"{save_dir}/{image_name}_{i:02d}.jpg")
