#!/usr/bin/env python3
"""
ZoDiac watermark decode script for WAVES analysis pipeline.
Uses DDIM inversion and frequency-domain analysis to detect ZoDiac watermarks.
Generates results compatible with WAVES analysis pipeline.

Detection Process:
1. Load image and resize to 512x512
2. Encode to latent space using VAE
3. Perform DDIM inversion to recover initial noise
4. Compute p-value comparing reversed latents against expected watermark pattern
5. Higher p-value = stronger watermark presence

Usage:
    python decode_zodiac.py --path <image_dir> --num-images 500 --device cuda:0
    python decode_zodiac.py --path <image_dir> --num-images 500 --rand --seed-offset 0           # rand-mode embed
    python decode_zodiac.py --path <image_dir> --num-images 500 --rand --seed-offset 2000000     # forgery
"""

import os
import sys
import click
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import warnings

# Add ZoDiac to path
sys.path.insert(0, os.environ.get("ZODIAC_PATH", "./external/ZoDiac"))

from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark
from diffusers import DDIMScheduler

import json as _json

# Standalone equivalents of WAVES dev helpers (avoids orjson dependency)
LIMIT = 5000
SUBSET_LIMIT = 500

def check_file_existence(path, name_pattern, limit):
    found = set(os.listdir(path))
    return [name_pattern.format(i) in found for i in range(limit)]

def existence_to_indices(existences, limit):
    return [i for i, e in enumerate(existences[:limit]) if e]

def load_json(filepath):
    try:
        with open(filepath, "r") as f:
            return _json.load(f)
    except Exception:
        return None

def save_json(data, filepath):
    with open(filepath, "w") as f:
        _json.dump(data, f)

warnings.filterwarnings("ignore")

RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")

# ZoDiac configuration (must match embedding parameters from zodiac_optimized.py)
MODEL_ID = 'Manojb/stable-diffusion-2-1-base'
W_CHANNEL = 3
W_RADIUS = 10
W_SEED = 10  # Fixed seed mode default


class GTWatermarkFast(GTWatermark):
    """GTWatermark that skips the expensive watermark_stat() computation.
    
    The stat computation (1000 random latent evaluations) is only needed for
    one_minus_p_value(). Since we use tree_ring_p_value() which uses a
    non-central chi-squared test directly, we can skip it entirely.
    This saves ~1-2 seconds per unique seed, critical for rand mode with many seeds.
    """
    def __init__(self, device, shape=(1, 4, 64, 64), dtype=torch.float32,
                 w_channel=3, w_radius=10, generator=None):
        self.device = device
        self.shape = shape
        self.dtype = dtype
        self.w_channel = w_channel
        self.w_radius = w_radius
        self.gt_patch, self.watermarking_mask = self._gen_gt(generator=generator)
        # Skip watermark_stat() — not needed for tree_ring_p_value
        self.mu = None
        self.sigma = None


def init_zodiac_detector(device, rand_mode=False):
    """
    Initialize the ZoDiac watermark detector.
    
    Returns:
        pipe: WMDetectStableDiffusionPipeline
        wm_pipe: GTWatermarkFast (None if rand_mode, created per-image)
        text_embeddings: empty prompt embeddings
    """
    print("Loading ZoDiac watermark detector...")

    # Change to ZoDiac directory for model loading
    original_dir = os.getcwd()
    os.chdir(os.environ.get("ZODIAC_PATH", "./external/ZoDiac"))

    # Load diffusion pipeline
    print(f"  Loading SD pipeline: {MODEL_ID}")
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = WMDetectStableDiffusionPipeline.from_pretrained(
        MODEL_ID, scheduler=scheduler
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Get text embeddings (reuse for all images)
    text_embeddings = pipe.get_text_embedding('')

    # Initialize watermark pattern
    wm_pipe = None
    if not rand_mode:
        print(f"  Fixed seed mode: w_seed={W_SEED}")
        wm_pipe = GTWatermarkFast(
            device,
            w_channel=W_CHANNEL,
            w_radius=W_RADIUS,
            generator=torch.Generator(device).manual_seed(W_SEED)
        )
    else:
        print("  Random seed mode: seed derived from filename")

    os.chdir(original_dir)

    print("✅ ZoDiac detector loaded")
    print(f"   Channel: {W_CHANNEL}, Radius: {W_RADIUS}")

    return pipe, wm_pipe, text_embeddings


def detect_watermark_single(image_path, pipe, wm_pipe, text_embeddings, device):
    """
    Detect ZoDiac watermark in a single image using DDIM inversion.
    
    Process:
    1. Load and resize image to 512x512 (ZoDiac requirement)
    2. Encode to VAE latent space
    3. DDIM inversion to recover initial noise
    4. Compute tree-ring-style p-value against expected watermark pattern
    
    Returns:
        p_value (float): Detection p-value (higher = more watermark present)
    """
    from torchvision.transforms.functional import pil_to_tensor

    # Load image — ZoDiac operates at 512x512
    img = Image.open(image_path).convert('RGB')
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.LANCZOS)

    img_tensor = (pil_to_tensor(img) / 255.0).unsqueeze(0).to(device)

    # Encode to latent space
    img_latents = pipe.get_image_latents(img_tensor, sample=False)

    # DDIM inversion to recover initial noise
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )

    # Compute p-value using non-central chi-squared test
    p_value = wm_pipe.tree_ring_p_value(reversed_latents)

    return p_value


def get_indices(path, limit, subset, subset_limit, num_images=None, key="zodiac"):
    """Get indices of images to decode."""
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]

    json_path = os.path.join(RESULT_DIR, dataset_name, f"{filename}-decode.json")

    if num_images is not None:
        target_count = num_images
    elif subset:
        target_count = subset_limit
    else:
        target_count = limit

    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    all_existing_indices = existence_to_indices(image_existences, limit=limit)
    print(f"Found {len(all_existing_indices)} images in {path} (searched indices 0-{limit-1})")

    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        decoded_count = 0
        for idx in all_existing_indices[:target_count]:
            if data.get(str(idx), {}).get(key) is not None:
                decoded_count += 1

        if decoded_count >= target_count:
            print(f"✅ Already fully decoded {target_count} images: {json_path}")
            return [], all_existing_indices[:target_count]

    if not os.path.exists(json_path):
        indices = all_existing_indices[:target_count]
    else:
        data = load_json(json_path)
        target_indices = all_existing_indices[:target_count]
        indices = [
            idx for idx in target_indices
            if data.get(str(idx), {}).get(key) is None
        ]

    return indices, all_existing_indices[:target_count]


@click.command()
@click.option("--path", type=str, required=True, help="Path to image directory")
@click.option("--subset", is_flag=True, help="Only process subset of images")
@click.option("--limit", type=int, default=LIMIT, help="Maximum index to search")
@click.option("--subset-limit", type=int, default=SUBSET_LIMIT, help="Subset limit")
@click.option("--device", type=str, default="cuda:0", help="Device to use")
@click.option("--num-images", type=int, default=None, help="Number of images to decode")
@click.option("--key", type=str, default="zodiac", help="Key name in output JSON")
@click.option("--rand", is_flag=True, help="Random seed mode (seed = image filename index)")
@click.option(
    "--seed-offset",
    type=int,
    default=0,
    help=(
        "Integer added to per-image seed in --rand mode. Use to detect a "
        "watermark embedded with a non-zero seed offset (e.g. ddb2 zodiac "
        "forgery used --seed-offset 2000000 at embed time)."
    ),
)
def main(path, subset, limit, subset_limit, device, num_images, key, rand, seed_offset):
    """
    Decode ZoDiac watermarks using DDIM inversion + frequency-domain p-value.
    
    Fixed seed mode (default):
        python decode_zodiac.py --path .../data/main/mscoco/zodiac
        python decode_zodiac.py --path .../data/attacked/mscoco/blurring-0.1-zodiac
    
    Random seed mode (seed derived from filename):
        python decode_zodiac.py --path .../data/main/mscoco/zodiac_rand --rand
        python decode_zodiac.py --path .../data/attacked/mscoco/blurring-0.1-zodiac_rand --rand
    """
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU (will be very slow)")

    print(f"Using device: {torch_device}")
    print(f"Mode: {'random seed' if rand else 'fixed seed'}")
    if rand and seed_offset:
        print(f"Seed offset: {seed_offset} (seed = int(filename) + {seed_offset})")

    indices, all_target_indices = get_indices(path, limit, subset, subset_limit, num_images, key=key)

    if len(indices) == 0:
        print("Nothing to decode!")
        return

    print(f"Will decode {len(indices)} images")

    # Initialize detector
    pipe, wm_pipe, text_embeddings = init_zodiac_detector(torch_device, rand_mode=rand)

    # Prepare output
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    json_path = os.path.join(RESULT_DIR, dataset_name, f"{filename}-decode.json")

    if os.path.exists(json_path):
        results = load_json(json_path)
        for idx in all_target_indices:
            if str(idx) not in results:
                results[str(idx)] = {key: None}
    else:
        results = {str(idx): {key: None} for idx in all_target_indices}

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Cache for per-image GTWatermark instances in rand mode
    _wm_cache = {}

    print("Decoding ZoDiac watermarks...")
    for idx in tqdm(indices, desc="Processing images"):
        image_path = os.path.join(path, f"{idx}.png")

        try:
            cur_wm_pipe = wm_pipe
            if rand:
                # In rand mode, seed = image index + seed_offset.
                # seed_offset != 0 lets us detect a watermark embedded with a
                # non-zero offset (e.g. ddb2 zodiac forgery used 2000000),
                # without colliding with the victim's seed-0 watermark.
                seed = idx + seed_offset
                if seed not in _wm_cache:
                    _wm_cache[seed] = GTWatermarkFast(
                        torch_device,
                        w_channel=W_CHANNEL,
                        w_radius=W_RADIUS,
                        generator=torch.Generator(torch_device).manual_seed(seed)
                    )
                cur_wm_pipe = _wm_cache[seed]

            p_value = detect_watermark_single(
                image_path, pipe, cur_wm_pipe, text_embeddings, torch_device
            )
            results[str(idx)][key] = str(p_value)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            import traceback
            traceback.print_exc()
            results[str(idx)][key] = None

    save_json(results, json_path)
    print(f"✅ Saved results to: {json_path}")


if __name__ == "__main__":
    main()
