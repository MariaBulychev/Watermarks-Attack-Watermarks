#!/usr/bin/env python3
"""
Tree-Ring watermark decode script for WAVES analysis pipeline.
Uses DDIM inversion and FFT analysis to detect tree-ring watermarks.
Generates results compatible with WAVES analysis pipeline.

Detection Process:
1. Load image
2. Encode to latent space using VAE
3. Perform DDIM inversion to recover initial noise
4. FFT to frequency domain
5. Compute L1 distance to expected watermark pattern in masked region
6. Lower metric = stronger watermark presence

Usage:
    python decode_tree_ring.py --path <image_dir> --num-images 500 --device cuda:0
    python decode_tree_ring.py --path <image_dir> --num-images 500 --seeds-json <path/to/seeds.json>   # rand-mode embed
"""

import os
import sys
import json as _json
import click
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import warnings

# Add the tree-ring-watermark directory to the path
sys.path.append(os.environ.get("TREE_RING_PATH", "./external/tree-ring-watermark"))

# Import from Tree-Ring repo
from optim_utils import (
    get_watermarking_pattern, 
    get_watermarking_mask, 
    eval_watermark,
    transform_img,
    set_random_seed
)
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

# Import from WAVES dev module
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    check_file_existence,
    existence_to_indices,
    save_json,
    load_json,
)

warnings.filterwarnings("ignore")

# Use result directory
CORRECTED_RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")


class TreeRingArgs:
    """Tree-ring watermark configuration - must match embedding parameters from generate_tree_ring.py"""
    def __init__(self):
        self.w_seed = 999999
        self.w_channel = 3
        self.w_pattern = 'ring'
        self.w_radius = 10
        self.w_mask_shape = 'circle'
        self.w_measurement = 'l1_complex'
        self.w_injection = 'complex'
        self.w_pattern_const = 0
        self.test_num_inference_steps = 50
        # Use the local cached Stable Diffusion 2.1 model
        self.model_id = 'stabilityai/stable-diffusion-2-1'


def init_tree_ring_detector(device, seeds_map=None):
    """
    Initialize the Tree-Ring watermark detector.
    Loads the Stable Diffusion pipeline for DDIM inversion.
    
    Args:
        device: torch device
        seeds_map: If provided, per-image seed dict; gt_patch is created per-image instead.
    
    Returns:
        pipe: InversableStableDiffusionPipeline for inversion
        args: TreeRingArgs with watermark parameters
        gt_patch: Ground truth watermark pattern (None if seeds_map is used)
        watermarking_mask: Mask for watermark region
    """
    print("Loading Tree-Ring watermark detector...")
    
    args = TreeRingArgs()
    
    # Load Stable Diffusion pipeline
    print("  Loading Stable Diffusion 2.1 pipeline...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        revision='fp16' if device.type == 'cuda' else None,
    )
    pipe = pipe.to(device)
    
    # Create watermarking mask (must match encoding, does not depend on seed)
    print("  Creating watermarking mask...")
    dummy_latents = pipe.get_random_latents()
    watermarking_mask = get_watermarking_mask(dummy_latents, args, device)
    
    gt_patch = None
    if seeds_map is None:
        # Fixed seed mode: create one gt_patch for all images
        print("  Creating watermark pattern...")
        gt_patch = get_watermarking_pattern(pipe, args, device)
        print(f"   Watermark seed: {args.w_seed}")
    else:
        print(f"  Per-image seeds mode: {len(seeds_map)} seeds loaded")
    
    print("✅ Tree-Ring detector loaded")
    print(f"   Pattern: {args.w_pattern}, Radius: {args.w_radius}, Channel: {args.w_channel}")
    
    return pipe, args, gt_patch, watermarking_mask


def detect_watermark_single(image_path, pipe, args, gt_patch, watermarking_mask, device):
    """
    Detect tree-ring watermark in a single image using DDIM inversion.
    
    Process:
    1. Load and transform image
    2. Encode to latent space
    3. DDIM inversion to recover initial noise
    4. Compute L1 distance to watermark pattern in frequency domain
    
    Args:
        image_path: Path to image file
        pipe: InversableStableDiffusionPipeline
        args: TreeRingArgs
        gt_patch: Ground truth watermark pattern
        watermarking_mask: Mask for watermark region
        device: torch device
    
    Returns:
        metric (float): Detection metric (lower = more watermark present)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Setup for detection (unknown prompt scenario)
    tester_prompt = ''  # At detection time, original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    
    # Transform and encode image to latent space
    img = transform_img(image).unsqueeze(0).to(text_embeddings.dtype).to(device)
    image_latents = pipe.get_image_latents(img, sample=False)
    
    # DDIM inversion (reverse diffusion to recover initial noise)
    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.test_num_inference_steps,
    )
    
    # Evaluate watermark - compute L1 distance to watermark pattern in frequency domain
    # Pass the same latents twice since eval_watermark expects two latents to compare
    metric, _ = eval_watermark(reversed_latents, reversed_latents, watermarking_mask, gt_patch, args)
    
    return metric


def get_indices(path, limit, subset, subset_limit, num_images=None, key="tree_ring"):
    """Get indices of images to decode.
    
    Args:
        path: Directory containing images
        limit: Maximum index to search (e.g., 5000 means check indices 0-4999)
        subset: Whether to use subset mode
        subset_limit: Limit for subset mode
        num_images: Number of actual images to decode (if None, use limit/subset_limit)
        key: Key name in output JSON
    """
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    
    json_path = os.path.join(CORRECTED_RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
    # Determine target number of images to decode
    if num_images is not None:
        target_count = num_images
    elif subset:
        target_count = subset_limit
    else:
        target_count = limit
    
    # Check which images exist in the full range
    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    all_existing_indices = existence_to_indices(image_existences, limit=limit)
    print(f"Found {len(all_existing_indices)} images in {path} (searched indices 0-{limit-1})")
    
    # Check if already decoded
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        # Count how many of the first target_count existing images are already decoded
        decoded_count = 0
        for idx in all_existing_indices[:target_count]:
            if data.get(str(idx), {}).get(key) is not None:
                decoded_count += 1
        
        if decoded_count >= target_count:
            print(f"✅ Already fully decoded {target_count} images: {json_path}")
            return [], all_existing_indices[:target_count]
    
    if not os.path.exists(json_path):
        # Take first target_count existing images
        indices = all_existing_indices[:target_count]
    else:
        # Only decode missing ones from the first target_count existing images
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
@click.option("--limit", type=int, default=LIMIT, help="Maximum index to search (searches 0 to limit-1)")
@click.option("--subset-limit", type=int, default=SUBSET_LIMIT, help="Subset limit")
@click.option("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cuda:1, etc.)")
@click.option("--num-images", type=int, default=None, help="Number of actual images to decode (overrides limit for counting)")
@click.option("--key", type=str, default="tree_ring", help="Key name in output JSON")
@click.option("--seeds-json", type=str, default=None,
              help="Path to seeds.json with per-image w_seed values (for _rand mode)")
def main(path, subset, limit, subset_limit, device, num_images, key, seeds_json):
    """
    Decode Tree-Ring watermarks using DDIM inversion and FFT analysis.
    
    Example:
        python decode_tree_ring.py --path ./data/tree_ring
        python decode_tree_ring.py --path ./data/attacked/mscoco/blurring-0.1-tree_ring
    """
    # Setup device
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU (will be very slow)")
    
    print(f"Using device: {torch_device}")
    
    # Load per-image seeds if provided
    seeds_map = None
    if seeds_json:
        with open(seeds_json, 'r') as f:
            raw_seeds = _json.load(f)
        # Convert keys like "0.png" -> int index
        seeds_map = {}
        for k, v in raw_seeds.items():
            try:
                seeds_map[int(k.replace('.png', ''))] = int(v)
            except (ValueError, TypeError):
                pass
        print(f"Loaded {len(seeds_map)} per-image seeds from {seeds_json}")
    
    # Get indices to process
    indices, all_target_indices = get_indices(path, limit, subset, subset_limit, num_images, key=key)
    
    if len(indices) == 0:
        print("Nothing to decode!")
        return
    
    print(f"Will decode {len(indices)} images")
    
    # Initialize detector
    pipe, args, gt_patch, watermarking_mask = init_tree_ring_detector(torch_device, seeds_map)
    
    # Prepare output
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    json_path = os.path.join(CORRECTED_RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
    # Load existing data or create new with proper structure for target indices
    if os.path.exists(json_path):
        results = load_json(json_path)
        # Extend results to include all target indices
        for idx in all_target_indices:
            if str(idx) not in results:
                results[str(idx)] = {key: None}
    else:
        results = {str(idx): {key: None} for idx in all_target_indices}
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Cache for per-image gt_patches to avoid regenerating the same pattern
    _gt_patch_cache = {}
    
    # Process images one at a time (DDIM inversion is sequential)
    print("Decoding tree-ring watermarks...")
    for idx in tqdm(indices, desc="Processing images"):
        image_path = os.path.join(path, f"{idx}.png")
        
        try:
            # Per-image seed: generate a unique gt_patch for this image
            cur_gt_patch = gt_patch
            if seeds_map is not None and idx in seeds_map:
                seed = seeds_map[idx]
                if seed not in _gt_patch_cache:
                    args.w_seed = seed
                    _gt_patch_cache[seed] = get_watermarking_pattern(pipe, args, torch_device)
                cur_gt_patch = _gt_patch_cache[seed]
            
            metric = detect_watermark_single(
                image_path, pipe, args, cur_gt_patch, watermarking_mask, torch_device
            )
            # Store metric as float string (for consistency with WAVES format)
            results[str(idx)][key] = str(metric)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            results[str(idx)][key] = None
    
    # Save results
    save_json(results, json_path)
    print(f"✅ Saved results to: {json_path}")


if __name__ == "__main__":
    main()
