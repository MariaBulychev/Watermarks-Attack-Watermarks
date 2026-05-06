#!/usr/bin/env python3
"""
Watermark images with WAM (Watermark Anything).
Supports fixed message (seed=42) or --random per-image messages.

Usage:
    python generate_wam.py --input <clean_dir> --output <out_dir> --mode rand --limit 500
    python generate_wam.py --input <clean_dir> --output <out_dir> --mode fixed
"""

import sys
import os
import json
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# WAM repo (configs use relative paths, so we chdir there)
WAM_REPO = os.environ.get("WAM_PATH", "./external/watermark-anything")
sys.path.insert(0, WAM_REPO)
os.chdir(WAM_REPO)

from watermark_anything.data.transforms import default_transform, unnormalize_img
from notebooks.inference_utils import load_model_from_checkpoint, create_random_mask, msg2str


def main():
    parser = argparse.ArgumentParser(description='WAM Watermark Generation')
    parser.add_argument('--input', '--input_dir', dest='input_dir', type=str,
                        default='./data/real',
                        help='Input image directory')
    parser.add_argument('--output', '--output_dir', dest='output_dir', type=str,
                        default='./data/wam',
                        help='Output directory for watermarked images')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Max images to process (-1 for all)')
    parser.add_argument('--random', action='store_true',
                        help='Use a random message per image (saves messages.json)')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    args = parser.parse_args()
    # Unified-CLI: --mode {fixed,rand} as alias for --random
    if hasattr(args, 'mode') and args.mode == 'rand':
        args.random = True


    OUTPUT_DIR = args.output_dir

    CKPT_DIR  = os.path.join(WAM_REPO, "checkpoints")
    JSON_PATH = os.path.join(CKPT_DIR, "params.json")
    CKPT_PATH = os.path.join(CKPT_DIR, "wam_mit.pth")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading WAM model...")
    wam = load_model_from_checkpoint(JSON_PATH, CKPT_PATH).to(device).eval()
    print("WAM model loaded.\n")

    # Message setup
    if args.random:
        print("Mode: RANDOM message per image")
        wm_msg = None  # Will generate per image
        messages_file = os.path.join(OUTPUT_DIR, 'messages.json')
        if os.path.exists(messages_file):
            with open(messages_file, 'r') as f:
                messages_dict = json.load(f)
            print(f"Resuming: loaded {len(messages_dict)} existing messages")
        else:
            messages_dict = {}
    else:
        torch.manual_seed(42)
        wm_msg = wam.get_random_msg(1).to(device)  # [1, 32]
        msg_str = msg2str(wm_msg[0])
        print(f"Watermark message: {msg_str}  ({wm_msg.shape[1]} bits)")
        with open(os.path.join(OUTPUT_DIR, "wam_message.txt"), "w") as f:
            f.write(msg_str)
        messages_dict = None

    # Collect images
    all_pngs = sorted(glob.glob(os.path.join(args.input_dir, "*.png")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))
    if args.limit > 0:
        all_pngs = all_pngs[:args.limit]
    print(f"Images to watermark: {len(all_pngs)}")

    processed = 0
    failed = 0

    for src_path in tqdm(all_pngs, desc="WAM watermarking"):
        fname = os.path.basename(src_path)
        out_path = os.path.join(OUTPUT_DIR, fname)

        # Skip already processed
        if os.path.exists(out_path):
            if args.random and fname in messages_dict:
                processed += 1
                continue
            elif not args.random:
                processed += 1
                continue

        try:
            img_pil = Image.open(src_path).convert("RGB")
            img_tensor = default_transform(img_pil).unsqueeze(0).to(device)

            # Per-image random message or fixed
            cur_msg = wam.get_random_msg(1).to(device) if args.random else wm_msg

            with torch.no_grad():
                outputs = wam.embed(img_tensor, cur_msg)
            wm_tensor = outputs["imgs_w"]

            # Localised mask (30 %)
            mask = create_random_mask(img_tensor, num_masks=1, mask_percentage=0.3)
            final_wm = wm_tensor * mask + img_tensor * (1 - mask)

            # Convert to PIL and save
            wm_pil = Image.fromarray(
                (unnormalize_img(final_wm).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255).astype(np.uint8)
            )
            wm_pil.save(out_path)

            if args.random:
                messages_dict[fname] = msg2str(cur_msg[0])

            processed += 1

            # Periodic save for random mode
            if args.random and processed % 500 == 0:
                with open(messages_file, 'w') as f:
                    json.dump(messages_dict, f, indent=2)

        except Exception as e:
            print(f"\nError processing {fname}: {e}")
            failed += 1

    # Final save
    if args.random:
        with open(messages_file, 'w') as f:
            json.dump(messages_dict, f, indent=2)
        print(f"Per-image messages saved to: {messages_file}")

    print(f"\nDone: {processed} processed, {failed} failed")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
