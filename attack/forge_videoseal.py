#!/usr/bin/env python3
"""
VideoSeal forgery attack with deterministic per-image messages.
Applies VideoSeal watermark on top of existing watermarked images
using seed = FORGERY_BASE_SEED + image_index (different from watermark seed).

Usage:
    python forge_videoseal.py --method pixelseal --dataset mscoco --mode rand --limit 500
    python forge_videoseal.py --method stable_sig --dataset diffusiondb --mode rand
"""

import os
import sys
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

# Forgery uses a different base seed than the watermark (42) to avoid same messages
FORGERY_BASE_SEED = 1337


def main():
    parser = argparse.ArgumentParser(
        description='VideoSeal forgery with deterministic per-image messages')
    parser.add_argument('--method', type=str, required=True,
                        help='Watermark method whose images to forge (e.g. pixelseal, rosteals)')
    parser.add_argument('--dataset', type=str, default='mscoco',
                        choices=['mscoco', 'diffusiondb'],
                        help='Dataset name (default: mscoco)')
    parser.add_argument('--limit', type=int, default=500,
                        help='Max images to process (-1 = all, default: 500)')
    args = parser.parse_args()

    # Paths
    BASE_DATA_DIR = os.environ.get('DATA_DIR', './data')
    INPUT_DIR = os.path.join(BASE_DATA_DIR, 'main', args.dataset, args.method)
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'attacked', args.dataset,
                              f'videoseal_forgery-1.0-{args.method}')

    print("=" * 70)
    print("VIDEOSEAL FORGERY — DETERMINISTIC PER-IMAGE MESSAGE")
    print("=" * 70)
    print(f"Dataset:       {args.dataset}")
    print(f"Method:        {args.method}")
    print(f"Input dir:     {INPUT_DIR}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Limit:         {args.limit}")
    print(f"Base seed:     {FORGERY_BASE_SEED} (per-image seed = {FORGERY_BASE_SEED} + image_index)")
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load VideoSeal model
    print("Loading VideoSeal model...")
    model = videoseal.load("videoseal")
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

    # Filter already-processed
    files_to_process = []
    skipped = 0
    for fname in image_files:
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
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
            # Deterministic message: FORGERY_BASE_SEED + image_index
            img_index = int(os.path.splitext(fname)[0]) if os.path.splitext(fname)[0].isdigit() else hash(fname)
            torch.manual_seed(FORGERY_BASE_SEED + img_index)
            msg = torch.randint(0, 2, (1, nbits)).float().to(device)

            # Load image
            img_pil = Image.open(input_path).convert('RGB')
            img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)

            # Embed watermark (is_video=False for images)
            with torch.no_grad():
                outputs = model.embed(img_tensor, msgs=msg, is_video=False)
            watermarked = outputs['imgs_w']

            # Save
            wm_pil = to_pil(watermarked[0].cpu().clamp(0, 1))
            wm_pil.save(output_path)

            success += 1

        except Exception as e:
            print(f"\nError processing {fname}: {e}")
            continue

    print(f"\nProcessed {success}/{len(files_to_process)} images")
    print(f"Messages recoverable via: torch.manual_seed({FORGERY_BASE_SEED} + image_index)")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
