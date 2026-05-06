#!/usr/bin/env python3
"""
Generate ZoDiac watermarked images.

This is the *watermark embedding* counterpart to ../attack/forge_zodiac.py.
Both use the same ZoDiac (Zhuang et al. 2024) optimisation procedure, but:

* generate_zodiac.py     embeds ZoDiac into clean / un-watermarked images
                         using a per-image seed = int(filename) (default
                         offset = 0).
* attack/forge_zodiac.py forges ZoDiac on top of an already-watermarked
                         image. To avoid colliding with a possible existing
                         ZoDiac watermark on the victim, the forgery uses
                         seed = int(filename) + seed-offset (typically a
                         large value such as 2_000_000).

Both modes write filename-derived seeds, so no metadata file is required for
later detection: pass `--rand --seed-offset N` to ../decode/decode_zodiac.py.

Usage
-----

    # Embed ZoDiac with rand-per-image seeds (default offset = 0):
    python generate_zodiac.py --input ./data/real --output ./data/zodiac --mode rand

    # Embed ZoDiac with fixed seed (single shared watermark for all images):
    python generate_zodiac.py --input ./data/real --output ./data/zodiac --mode fixed

Usage:
    python generate_zodiac.py --input <clean_dir> --output <out_dir> --mode rand --limit 500
    python generate_zodiac.py --input <clean_dir> --output <out_dir> --mode fixed
"""

import os
import sys
import glob
import argparse
import logging
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from diffusers import DDIMScheduler

# Add ZoDiac to path
sys.path.insert(0, os.environ.get("ZODIAC_PATH", "./external/ZoDiac"))

from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark
from main.utils import save_img
from loss.loss import LossProvider

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    return pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )


class ZoDiacEmbedder:
    def __init__(self, device_id=0, mode="fixed", seed_offset=0, fixed_seed=10):
        self.mode = mode
        self.seed_offset = seed_offset
        self.fixed_seed = fixed_seed
        self.cfgs = {
            "model_id": "stabilityai/stable-diffusion-2-1-base",
            "w_channel": 3,
            "w_radius": 10,
            "iters": 100,
            "loss_weights": [10.0, 0.1, 1.0, 0.0],
        }
        self.device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )

        zodiac_root = os.environ.get("ZODIAC_PATH", "./external/ZoDiac")
        cwd = os.getcwd()
        os.chdir(zodiac_root)

        scheduler = DDIMScheduler.from_pretrained(
            self.cfgs["model_id"], subfolder="scheduler"
        )
        self.pipe = WMDetectStableDiffusionPipeline.from_pretrained(
            self.cfgs["model_id"], scheduler=scheduler
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.totalLoss = LossProvider(self.cfgs["loss_weights"], self.device)
        self.text_embeddings = self.pipe.get_text_embedding("")

        if self.mode == "fixed":
            self.wm_pipe = GTWatermark(
                self.device,
                w_channel=self.cfgs["w_channel"],
                w_radius=self.cfgs["w_radius"],
                generator=torch.Generator(self.device).manual_seed(self.fixed_seed),
            )
        else:
            self.wm_pipe = None

        os.chdir(cwd)

    def embed(self, in_path, out_path):
        if self.mode == "rand":
            seed = (
                int(os.path.splitext(os.path.basename(in_path))[0])
                + self.seed_offset
            )
            wm_pipe = GTWatermark(
                self.device,
                w_channel=self.cfgs["w_channel"],
                w_radius=self.cfgs["w_radius"],
                generator=torch.Generator(self.device).manual_seed(seed),
            )
        else:
            wm_pipe = self.wm_pipe

        img = Image.open(in_path).convert("RGB")
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.LANCZOS)
        gt = (pil_to_tensor(img) / 255.0).unsqueeze(0).to(self.device)

        latents = get_init_latent(gt, self.pipe, self.text_embeddings).detach().clone()
        latents.requires_grad = True
        opt = optim.Adam([latents], lr=0.01)
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 70], gamma=0.3)

        for i in range(self.cfgs["iters"]):
            wm_lat = wm_pipe.inject_watermark(latents)
            pred = self.pipe(
                "",
                guidance_scale=1.0,
                num_inference_steps=50,
                output_type="tensor",
                use_trainable_latents=True,
                init_latents=wm_lat,
            ).images
            loss = self.totalLoss(pred, gt, wm_lat, wm_pipe)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            if (i + 1) % 25 == 0:
                logger.info(f"  iter {i + 1}: loss {loss.item():.4f}")

        save_img(out_path, pred, self.pipe)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Embed ZoDiac watermark on clean images")
    parser.add_argument("--input", required=True, help="Input directory with clean images")
    parser.add_argument("--output", required=True, help="Output directory for watermarked images")
    parser.add_argument(
        "--mode",
        choices=["fixed", "rand"],
        default="rand",
        help="Watermark mode: 'fixed' uses one shared seed; 'rand' uses seed=int(filename)+offset",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to filename-derived seed in --mode rand (default 0).",
    )
    parser.add_argument(
        "--fixed-seed",
        type=int,
        default=10,
        help="Seed used in --mode fixed (default 10).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--start", type=int, default=None, help="Start at image with this numeric index")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(args.input, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if args.start is not None:
        files = [
            f for f in files if int(os.path.splitext(os.path.basename(f))[0]) >= args.start
        ]
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} images. Mode: {args.mode}  Seed-offset: {args.seed_offset}")

    embedder = ZoDiacEmbedder(
        mode=args.mode,
        seed_offset=args.seed_offset,
        fixed_seed=args.fixed_seed,
    )

    n_done = 0
    for i, fp in enumerate(files):
        out = os.path.join(args.output, os.path.basename(fp))
        if os.path.exists(out):
            n_done += 1
            continue
        print(f"[{i + 1}/{len(files)}] {os.path.basename(fp)}", flush=True)
        try:
            embedder.embed(fp, out)
            n_done += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone: {n_done}/{len(files)} successful")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
