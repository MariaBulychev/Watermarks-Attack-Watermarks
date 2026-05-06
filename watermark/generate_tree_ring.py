#!/usr/bin/env python3
"""
Generate Tree-Ring watermarked images for all COCO prompts.
Uses seed = 42 + image_index to match stable_diffusion baseline.
Saves only watermarked images to main/mscoco/tree_ring/

Usage:
    python generate_tree_ring.py --prompts_file <prompts.json> --output <out_dir> --mode rand --limit 500
    python generate_tree_ring.py --prompts_file <prompts.json> --output <out_dir> --mode fixed
"""

import os
import sys
import json
import copy
import random as pyrandom
import argparse
import torch
from tqdm import tqdm

# Add tree-ring-watermark to path
sys.path.append(os.environ.get("TREE_RING_PATH", "./external/tree-ring-watermark"))
sys.path.append(os.environ.get("TREE_RING_IMPL_PATH", "./external/tree_ring_implementation"))

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils_no_datasets import (
    get_watermarking_pattern,
    get_watermarking_mask,
    inject_watermark,
    set_random_seed
)

class Args:
    """Tree-ring watermark configuration"""
    def __init__(self):
        self.image_length = 512
        self.model_id = 'stabilityai/stable-diffusion-2-1'
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        
        # Watermark parameters (constant for all images)
        self.w_seed = 999999
        self.w_channel = 3
        self.w_pattern = 'ring'
        self.w_mask_shape = 'circle'
        self.w_radius = 10
        self.w_measurement = 'l1_complex'
        self.w_injection = 'complex'
        self.w_pattern_const = 0

def main():
    parser = argparse.ArgumentParser(description='Tree-Ring Watermark Generation')
    parser.add_argument('--prompts_file',
                        default='./data/prompts.json',
                        help='Path to prompts JSON file')
    parser.add_argument('--output', '--output_dir', dest='output_dir',
                        default='./data/tree_ring',
                        help='Output directory for watermarked images')
    parser.add_argument('--random', action='store_true',
                        help='Use a random w_seed per image (saves seeds.json)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Max images to generate (-1 for all)')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    cli_args = parser.parse_args()
    if hasattr(cli_args, 'mode') and cli_args.mode == 'rand':
        cli_args.random = True


    # Configuration
    BASE_SEED = 42
    
    # Paths
    PROMPTS_FILE = cli_args.prompts_file
    OUTPUT_DIR = cli_args.output_dir
    
    # Setup
    args = Args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load prompts
    print(f"Loading prompts from {PROMPTS_FILE}...")
    with open(PROMPTS_FILE, 'r') as f:
        prompts = json.load(f)
    
    num_prompts = len(prompts)
    if cli_args.limit > 0:
        num_prompts = min(num_prompts, cli_args.limit)
    print(f"Loaded {num_prompts} prompts")
    print()
    
    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion 2.1 base pipeline...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        revision='fp16' if device == 'cuda' else None,
    )
    pipe = pipe.to(device)
    print("Pipeline loaded successfully!")
    print()
    
    # Create watermark pattern
    if cli_args.random:
        print("Mode: RANDOM w_seed per image")
        gt_patch = None  # Will generate per image
    else:
        print("Creating Tree-Ring watermark pattern (fixed seed)...")
        gt_patch = get_watermarking_pattern(pipe, args, device)
        print(f"Watermark pattern shape: {gt_patch.shape}")
        print(f"Watermark seed: {args.w_seed}")
    print(f"Pattern: {args.w_pattern}, Radius: {args.w_radius}, Channel: {args.w_channel}")
    print()
    
    # Generate watermarked images
    print(f"Generating {num_prompts} Tree-Ring watermarked images...")
    print(f"Base seed: {BASE_SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load existing seeds map if resuming random mode
    seeds_path = os.path.join(OUTPUT_DIR, 'seeds.json')
    if cli_args.random:
        if os.path.exists(seeds_path):
            with open(seeds_path, 'r') as f:
                seeds_map = json.load(f)
            print(f"Resuming: loaded {len(seeds_map)} existing seeds")
        else:
            seeds_map = {}
    else:
        seeds_map = None
    
    for idx in tqdm(range(num_prompts), desc="Generating watermarked images"):
        fname = f"{idx}.png"
        output_path = os.path.join(OUTPUT_DIR, fname)
        
        # Skip if already processed
        if os.path.exists(output_path):
            if cli_args.random and fname in seeds_map:
                continue
            elif not cli_args.random:
                continue
        
        prompt = prompts[str(idx)]
        seed = BASE_SEED + idx
        
        # Set seed for reproducibility (matches stable_diff baseline)
        set_random_seed(seed)
        
        # Get initial latents
        init_latents = pipe.get_random_latents()
        
        # Per-image random w_seed if requested
        if cli_args.random:
            cur_w_seed = pyrandom.randint(0, 999999999)
            args.w_seed = cur_w_seed
            cur_gt_patch = get_watermarking_pattern(pipe, args, device)
        else:
            cur_gt_patch = gt_patch
        
        # In random mode, cast to float32 for FFT (ComplexHalf not fully supported)
        if cli_args.random:
            init_latents = init_latents.float()
            cur_gt_patch = cur_gt_patch.to(torch.complex64)

        # Inject watermark
        watermarking_mask = get_watermarking_mask(init_latents, args, device)
        init_latents_w = inject_watermark(
            copy.deepcopy(init_latents), 
            watermarking_mask, 
            cur_gt_patch, 
            args
        )

        # Cast back to fp16 for generation if we upcast
        if cli_args.random:
            init_latents_w = init_latents_w.half()
        
        # Generate watermarked image
        output = pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        
        image = output.images[0]
        image.save(output_path)
        
        # Record seed
        if cli_args.random:
            seeds_map[fname] = cur_w_seed
    
    # Save seeds
    if cli_args.random:
        with open(seeds_path, 'w') as f:
            json.dump(seeds_map, f, indent=2)
        print(f"Per-image seeds saved to: {seeds_path}")
    
    print()
    print("="*70)
    print("TREE-RING WATERMARK GENERATION COMPLETE!")
    print("="*70)
    print(f"Generated {num_prompts} watermarked images")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"Seed range: {BASE_SEED} to {BASE_SEED + num_prompts - 1}")
    print(f"Watermark parameters:")
    print(f"  - Pattern: {args.w_pattern}")
    if not cli_args.random:
        print(f"  - Seed: {args.w_seed}")
    else:
        print(f"  - Seed: RANDOM per image (see seeds.json)")
    print(f"  - Radius: {args.w_radius}")
    print(f"  - Channel: {args.w_channel}")
    print("="*70)

if __name__ == "__main__":
    main()
