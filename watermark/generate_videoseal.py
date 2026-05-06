#!/usr/bin/env python3
"""
Generate VideoSeal watermarked images for all images in a source directory.
Uses a deterministic per-image message: seed = base_seed + image_index.
Message is recoverable from filename, so no file saving is needed.

Usage:
    python generate_videoseal.py --input <clean_dir> --output <out_dir> --mode rand --limit 500
    python generate_videoseal.py --input <clean_dir> --output <out_dir> --mode fixed
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# Add videoseal to path
VIDEOSEAL_PATH = os.environ.get("VIDEOSEAL_PATH", "./external/videoseal")
sys.path.insert(0, VIDEOSEAL_PATH)

# Set cache directory for model downloads
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "./cache/torch")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./cache/hf")

# Change to videoseal directory for relative path resolution
os.chdir(VIDEOSEAL_PATH)

import videoseal
from videoseal.evals.metrics import bit_accuracy


def generate_fixed_message(nbits: int, seed: int = 42) -> torch.Tensor:
    """
    Generate a fixed binary message for watermarking.
    Uses a seed for reproducibility.
    """
    torch.manual_seed(seed)
    return torch.randint(0, 2, (1, nbits)).float()


def msg_to_str(msg: torch.Tensor) -> str:
    """Convert message tensor to binary string."""
    return ''.join([str(int(b > 0.5)) for b in msg.flatten().tolist()])


def main():
    parser = argparse.ArgumentParser(description='Generate VideoSeal watermarked images')
    parser.add_argument('--input', '--input_dir', dest='input_dir', type=str,
                        default='./data/real',
                        help='Directory containing source images')
    parser.add_argument('--output', '--output_dir', dest='output_dir', type=str,
                        default='./data/videoseal',
                        help='Directory to save watermarked images')
    parser.add_argument('--message_seed', type=int, default=42,
                        help='Base seed for generating watermark messages (per-image seed = base + image_index)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify watermark detection after embedding (slower)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for processing (for resuming)')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='End index for processing (-1 for all)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Maximum number of images to process (-1 for all)')
    args = parser.parse_args()

    # Configuration
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MESSAGE_SEED = args.message_seed

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load VideoSeal model
    print("=" * 70)
    print("Loading VideoSeal model...")
    print("=" * 70)

    model = videoseal.load("videoseal")
    model = model.to(device)
    model.eval()

    # Get model configuration
    nbits = model.embedder.msg_processor.nbits
    print(f"✅ Model loaded successfully!")
    print(f"   Number of bits: {nbits}")
    print(f"   Scaling W: {model.blender.scaling_w}")
    print(f"   Scaling I: {model.blender.scaling_i}")
    print(f"   Image size: {model.img_size}")
    print()

    # Message mode: deterministic per-image (seed = base_seed + image_index)
    print(f"Mode: Deterministic per-image message (seed = {MESSAGE_SEED} + image_index)")
    print(f"Message is recoverable from filename, no file saving needed.")
    print()

    # Get list of input images (sort numerically by filename)
    image_files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else f
    )

    # Apply start/end indices
    if args.end_idx > 0:
        image_files = image_files[args.start_idx:args.end_idx]
    else:
        image_files = image_files[args.start_idx:]

    # Apply limit if specified
    if args.limit > 0:
        image_files = image_files[:args.limit]

    num_images = len(image_files)
    print(f"Found {num_images} images to process")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Transforms
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    # Statistics
    total_psnr = 0.0
    total_bit_acc = 0.0
    num_verified = 0
    num_processed = 0

    # Process images
    print("=" * 70)
    print("Generating VideoSeal watermarked images...")
    print("=" * 70)

    for filename in tqdm(image_files, desc="Watermarking images"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Skip if already processed
        if os.path.exists(output_path):
            continue

        try:
            # Deterministic per-image message: seed = base_seed + image_index
            img_index = int(os.path.splitext(filename)[0]) if os.path.splitext(filename)[0].isdigit() else hash(filename)
            img_seed = MESSAGE_SEED + img_index
            torch.manual_seed(img_seed)
            cur_message = torch.randint(0, 2, (1, nbits)).float().to(device)

            # Load image
            img_pil = Image.open(input_path).convert('RGB')
            img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)

            # Embed watermark (is_video=False for single images)
            with torch.no_grad():
                outputs = model.embed(img_tensor, msgs=cur_message, is_video=False)

            watermarked_tensor = outputs['imgs_w']

            # Compute PSNR
            mse = ((img_tensor - watermarked_tensor) ** 2).mean().item()
            if mse > 1e-10:
                psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
            else:
                psnr = float('inf')
            total_psnr += psnr
            num_processed += 1

            # Verify watermark if requested
            if args.verify:
                with torch.no_grad():
                    detection = model.detect(watermarked_tensor, is_video=False)
                bit_preds = detection['preds'][:, 1:]
                ba = bit_accuracy(bit_preds, cur_message).item()
                total_bit_acc += ba
                num_verified += 1

            # Save watermarked image
            watermarked_pil = to_pil(watermarked_tensor[0].cpu().clamp(0, 1))
            watermarked_pil.save(output_path)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue

    # Print summary
    print()
    print("=" * 70)
    print("VIDEOSEAL WATERMARK GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Processed {num_processed} images")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"Message length: {nbits} bits")
    print(f"Base seed: {MESSAGE_SEED} (per-image seed = base + image_index)")
    if num_processed > 0:
        print(f"Average PSNR: {total_psnr / num_processed:.2f} dB")

    if args.verify and num_verified > 0:
        print(f"Average bit accuracy: {total_bit_acc / num_verified * 100:.2f}%")

    print("=" * 70)

    print(f"Messages are deterministic: recover with torch.manual_seed({MESSAGE_SEED} + image_index)")


if __name__ == "__main__":
    main()
