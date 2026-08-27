#!/usr/bin/env python3
"""
PixelSeal forgery attack with a RANDOM message per image.
Applies PixelSeal watermark on top of existing watermarked images
using a different random message for each image.
Saves per-image messages to a JSON file in the output directory.

Usage:
    python forge_pixelseal.py --method rosteals --dataset mscoco --mode rand --limit 500
    python forge_pixelseal.py --method stable_sig --dataset diffusiondb --mode rand
"""

import os
import sys
import json
import random
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# Add videoseal to path
VIDEOSEAL_PATH = os.environ.get("VIDEOSEAL_PATH", "./external/videoseal")
sys.path.insert(0, VIDEOSEAL_PATH)

# Set cache directories
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "./cache/torch")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./cache/hf")

# Change to videoseal directory for relative path resolution
os.chdir(VIDEOSEAL_PATH)

import videoseal


def generate_random_message(nbits: int) -> torch.Tensor:
    """Generate a random binary message (no fixed seed)."""
    return torch.randint(0, 2, (1, nbits)).float()


def msg_to_str(msg: torch.Tensor) -> str:
    """Convert message tensor to binary string."""
    return ''.join([str(int(b > 0.5)) for b in msg.flatten().tolist()])


def main():
    parser = argparse.ArgumentParser(
        description='PixelSeal forgery with random per-image messages')
    parser.add_argument('--method', type=str, required=True,
                        help='Watermark method whose images to forge (e.g. pixelseal, rosteals_rand)')
    parser.add_argument('--dataset', type=str, default='mscoco',
                        choices=['mscoco', 'diffusiondb', 'diffusiondb_2'],
                        help='Dataset name (default: mscoco)')
    parser.add_argument('--limit', type=int, default=500,
                        help='Max images to process (-1 = all, default: 500)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global random seed for reproducibility (default: None = truly random)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Paths
    BASE_DATA_DIR = os.environ.get('DATA_DIR', './data')
    INPUT_DIR = os.path.join(BASE_DATA_DIR, 'main', args.dataset, args.method)
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'attacked', args.dataset,
                              f'forgery_pixelseal_corrected-1.0-{args.method}')

    print("=" * 70)
    print("PIXELSEAL FORGERY — RANDOM MESSAGE PER IMAGE")
    print("=" * 70)
    print(f"Dataset:       {args.dataset}")
    print(f"Method:        {args.method}")
    print(f"Input dir:     {INPUT_DIR}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Limit:         {args.limit}")
    print(f"Global seed:   {args.seed}")
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load PixelSeal model
    print("Loading PixelSeal model...")
    model = videoseal.load("pixelseal")
    model = model.to(device)
    model.eval()

    nbits = model.embedder.msg_processor.nbits
    print(f"Model loaded — {nbits} bits per message\n")

    # Get image list (sort numerically)
    image_files = sorted(
        [f for f in os.listdir(INPUT_DIR)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else f
    )
    if args.limit > 0:
        image_files = image_files[:args.limit]

    if not image_files:
        print(f"ERROR: No images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    # Load existing messages if resuming
    messages_path = os.path.join(OUTPUT_DIR, 'messages.json')
    if os.path.exists(messages_path):
        with open(messages_path, 'r') as f:
            messages_map = json.load(f)
        print(f"Loaded {len(messages_map)} existing messages (resuming)")
    else:
        messages_map = {}

    # Filter already-processed
    files_to_process = []
    skipped = 0
    for fname in image_files:
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path) and fname in messages_map:
            skipped += 1
        else:
            files_to_process.append(fname)

    print(f"Already done:  {skipped}")
    print(f"To process:    {len(files_to_process)}\n")

    if not files_to_process:
        print("All images already processed. Nothing to do.")
        return

    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    success = 0

    for fname in tqdm(files_to_process, desc=f"Forging {args.method}"):
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)
        try:
            # Random message for this image
            msg = generate_random_message(nbits).to(device)

            # Load image
            img_pil = Image.open(input_path).convert('RGB')
            img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)

            # Embed watermark
            with torch.no_grad():
                outputs = model.embed(img_tensor, msgs=msg,
                                      lowres_attenuation=True)
            watermarked = outputs['imgs_w']

            # Save
            wm_pil = to_pil(watermarked[0].cpu().clamp(0, 1))
            wm_pil.save(output_path)

            # Record message
            messages_map[fname] = msg_to_str(msg)
            success += 1

        except Exception as e:
            print(f"\nError processing {fname}: {e}")
            continue

    # Save messages
    with open(messages_path, 'w') as f:
        json.dump(messages_map, f, indent=2)

    print(f"\nProcessed {success}/{len(files_to_process)} images")
    print(f"Messages saved to: {messages_path}")

    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "method": args.method,
        "forgery_watermark": "pixelseal",
        "message_type": "random_per_image",
        "nbits": nbits,
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "total_images": len(image_files),
        "already_existed": skipped,
        "newly_processed": success,
        "total_in_output": skipped + success,
        "global_seed": args.seed
    }
    meta_path = os.path.join(OUTPUT_DIR, 'forgery_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
