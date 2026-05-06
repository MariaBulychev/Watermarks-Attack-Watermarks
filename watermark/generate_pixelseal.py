#!/usr/bin/env python3
"""
Generate PixelSeal watermarked images for all images in a source directory.
Uses a consistent message for all images (256 bits).
Saves watermarked images to the output directory.

Usage:
    python generate_pixelseal.py --input <clean_dir> --output <out_dir> --mode rand --limit 500
    python generate_pixelseal.py --input <clean_dir> --output <out_dir> --mode fixed
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
    parser = argparse.ArgumentParser(description='Generate PixelSeal watermarked images')
    parser.add_argument('--input', '--input_dir', dest='input_dir', type=str, 
                        default='./data/real',
                        help='Directory containing source images')
    parser.add_argument('--output', '--output_dir', dest='output_dir', type=str,
                        default='./data/pixelseal',
                        help='Directory to save watermarked images')
    parser.add_argument('--message_seed', type=int, default=42,
                        help='Seed for generating the watermark message')
    parser.add_argument('--verify', action='store_true',
                        help='Verify watermark detection after embedding (slower)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for processing (for resuming)')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='End index for processing (-1 for all)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Maximum number of images to process (-1 for all)')
    parser.add_argument('--random', action='store_true',
                        help='Use a random message per image (saves messages.json)')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    args = parser.parse_args()
    # Unified-CLI: --mode {fixed,rand} as alias for --random
    if hasattr(args, 'mode') and args.mode == 'rand':
        args.random = True

    
    # Configuration
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MESSAGE_SEED = args.message_seed
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load PixelSeal model
    print("=" * 70)
    print("Loading PixelSeal model...")
    print("=" * 70)
    
    model = videoseal.load("pixelseal")
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
    
    # Generate message
    if args.random:
        print("Mode: RANDOM message per image")
        # Load existing messages if resuming
        messages_path = os.path.join(OUTPUT_DIR, 'messages.json')
        if os.path.exists(messages_path):
            with open(messages_path, 'r') as f:
                messages_map = json.load(f)
            print(f"Loaded {len(messages_map)} existing messages (resuming)")
        else:
            messages_map = {}
        message = None  # Will generate per image
    else:
        message = generate_fixed_message(nbits, MESSAGE_SEED).to(device)
        print(f"Watermark message (first 48 bits): {msg_to_str(message[:, :48])}")
        print(f"Message seed: {MESSAGE_SEED}")
        messages_map = None
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
    
    # Process images
    print("=" * 70)
    print("Generating PixelSeal watermarked images...")
    print("=" * 70)
    
    for filename in tqdm(image_files, desc="Watermarking images"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            if args.random and filename in messages_map:
                continue
            elif not args.random:
                continue
        
        try:
            # Generate per-image random message if needed
            if args.random:
                cur_message = torch.randint(0, 2, (1, nbits)).float().to(device)
            else:
                cur_message = message
            
            # Load image
            img_pil = Image.open(input_path).convert('RGB')
            img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)
            
            # Embed watermark
            with torch.no_grad():
                outputs = model.embed(img_tensor, msgs=cur_message, lowres_attenuation=True)
            
            watermarked_tensor = outputs['imgs_w']
            
            # Compute PSNR
            mse = ((img_tensor - watermarked_tensor) ** 2).mean().item()
            if mse > 1e-10:
                psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
            else:
                psnr = float('inf')
            total_psnr += psnr
            
            # Verify watermark if requested
            if args.verify:
                with torch.no_grad():
                    detection = model.detect(watermarked_tensor)
                bit_preds = detection['preds'][:, 1:]
                ba = bit_accuracy(bit_preds, cur_message).item()
                total_bit_acc += ba
                num_verified += 1
            
            # Save watermarked image
            watermarked_pil = to_pil(watermarked_tensor[0].cpu().clamp(0, 1))
            watermarked_pil.save(output_path)
            
            # Record random message
            if args.random:
                messages_map[filename] = msg_to_str(cur_message)
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    # Print summary
    print()
    print("=" * 70)
    print("PIXELSEAL WATERMARK GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Processed {num_images} images")
    print(f"Saved to: {OUTPUT_DIR}")
    if not args.random:
        print(f"Watermark message (first 48 bits): {msg_to_str(message[:, :48])}")
    print(f"Message length: {nbits} bits")
    print(f"Average PSNR: {total_psnr / max(num_images, 1):.2f} dB")
    
    if args.verify and num_verified > 0:
        print(f"Average bit accuracy: {total_bit_acc / num_verified * 100:.2f}%")
    
    print("=" * 70)
    
    # Save message(s)
    if args.random:
        messages_path = os.path.join(OUTPUT_DIR, 'messages.json')
        with open(messages_path, 'w') as f:
            json.dump(messages_map, f, indent=2)
        print(f"Per-image messages saved to: {messages_path}")
    else:
        message_file = os.path.join(OUTPUT_DIR, 'watermark_message.txt')
        with open(message_file, 'w') as f:
            f.write(f"Message seed: {MESSAGE_SEED}\n")
            f.write(f"Message length: {nbits} bits\n")
            f.write(f"Message (binary): {msg_to_str(message)}\n")
        print(f"Message saved to: {message_file}")


if __name__ == "__main__":
    main()
