#!/usr/bin/env python3
"""
StegaStamp forgery attack — apply StegaStamp watermark on top of
already-watermarked images. Supports fixed or random per-image secrets.

Usage:
    python forge_stegastamp.py --method stable_sig --dataset mscoco --mode rand --limit 500
    python forge_stegastamp.py --method tree_ring --dataset diffusiondb --mode rand
"""

import os
import sys
import glob
import json
import string
import random as pyrandom
import argparse
import bchlib
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
from tqdm import tqdm

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def generate_random_secret(length=7):
    """Generate a random 7-character ASCII secret."""
    chars = string.ascii_letters + string.digits + string.punctuation
    return ''.join(pyrandom.choice(chars) for _ in range(length))


def encode_secret(bch, secret_str):
    """Encode a secret string into binary format for StegaStamp."""
    secret_str = secret_str[:7]
    data = bytearray(secret_str + ' ' * (7 - len(secret_str)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])  # 4-zero footer
    return np.array([secret], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Apply StegaStamp forgery watermark')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mscoco', 'diffusiondb', 'diffusiondb_2'],
                        help='Dataset name')
    parser.add_argument('--method', type=str, required=True,
                        help='Watermark method to attack (e.g. pixelseal, rosteals_rand)')
    parser.add_argument('--secret', type=str, default='Forge42',
                        help='Fixed forgery secret (max 7 chars, default: Forge42)')
    parser.add_argument('--random', action='store_true',
                        help='Use a different random secret per image')
    parser.add_argument('--limit', type=int, default=500,
                        help='Max images to process (-1 = all, default: 500)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    args = parser.parse_args()
    # Unified-CLI: --mode {fixed,rand} as alias for --random
    if hasattr(args, 'mode') and args.mode == 'rand':
        args.random = True


    if args.seed is not None:
        pyrandom.seed(args.seed)
        np.random.seed(args.seed)

    # Paths
    MODEL_PATH = os.path.join(os.environ.get("WAVES_PATH", "./external/WAVES"), "decoders/stega_stamp.onnx")
    BASE_DATA_DIR = os.environ.get('DATA_DIR', './data')
    INPUT_DIR = os.path.join(BASE_DATA_DIR, 'main', args.dataset, args.method)
    OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'attacked', args.dataset,
                              f'stegastamp_forgery-1.0-{args.method}')

    WIDTH, HEIGHT = 400, 400
    size = (WIDTH, HEIGHT)

    print("=" * 70)
    print("STEGASTAMP FORGERY ATTACK")
    print("=" * 70)
    print(f"Dataset:   {args.dataset}")
    print(f"Method:    {args.method}")
    print(f"Input:     {INPUT_DIR}")
    print(f"Output:    {OUTPUT_DIR}")
    print(f"Random:    {args.random}")
    print(f"Limit:     {args.limit}")
    print()

    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ONNX model
    print("Loading StegaStamp ONNX model...")
    sess = ort.InferenceSession(MODEL_PATH)
    print("Model loaded.\n")

    # Setup BCH
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # Fixed secret setup (used when --random is not set)
    if not args.random:
        fixed_secret = args.secret[:7]
        fixed_secret_input = encode_secret(bch, fixed_secret)
        print(f"Mode: FIXED secret '{fixed_secret}'")
    else:
        fixed_secret = None
        fixed_secret_input = None
        print("Mode: RANDOM secret per image")
    print()

    # Get image list sorted numerically
    files_list = sorted(
        glob.glob(os.path.join(INPUT_DIR, '*.png')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))
    if args.limit > 0:
        files_list = files_list[:args.limit]

    if not files_list:
        print(f"ERROR: No images found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(files_list)} images (after limit)")

    # Load existing secrets if resuming
    secrets_path = os.path.join(OUTPUT_DIR, 'secrets.json')
    if args.random:
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                secrets_map = json.load(f)
            print(f"Resuming: loaded {len(secrets_map)} existing secrets")
        else:
            secrets_map = {}
    else:
        secrets_map = None

    # Filter already-processed
    files_to_process = []
    skipped = 0
    for fpath in files_list:
        fname = os.path.basename(fpath)
        out_path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(out_path):
            if args.random and fname in secrets_map:
                skipped += 1
            elif not args.random:
                skipped += 1
            else:
                files_to_process.append(fpath)
        else:
            files_to_process.append(fpath)

    print(f"Already done: {skipped}")
    print(f"To process:   {len(files_to_process)}\n")

    if not files_to_process:
        print("All images already processed. Nothing to do.")
        return

    success_count = 0

    for fpath in tqdm(files_to_process, desc=f"Forging {args.method}"):
        fname = os.path.basename(fpath)
        try:
            # Determine secret
            if args.random:
                cur_secret = generate_random_secret(7)
                cur_secret_input = encode_secret(bch, cur_secret)
            else:
                cur_secret = fixed_secret
                cur_secret_input = fixed_secret_input

            # Load and preprocess (resize to 400x400 for StegaStamp)
            image = Image.open(fpath).convert("RGB")
            image = np.array(ImageOps.fit(image, size), dtype=np.float32)
            image /= 255.0
            image_input = np.array([image], dtype=np.float32)

            # Run inference
            outputs = sess.run(None, {
                'secret': cur_secret_input,
                'image': image_input
            })

            stegastamp = outputs[0]

            # Convert to uint8 and save
            rescaled = (stegastamp[0] * 255).astype(np.uint8)
            im = Image.fromarray(rescaled)
            im.save(os.path.join(OUTPUT_DIR, fname))

            # Record secret
            if args.random:
                secrets_map[fname] = cur_secret

            success_count += 1

        except Exception as e:
            print(f"\nError processing {fpath}: {e}")
            continue

    # Save secrets
    if args.random:
        with open(secrets_path, 'w') as f:
            json.dump(secrets_map, f, indent=2)
        print(f"Secrets saved to: {secrets_path}")

    print(f"\nProcessed {success_count}/{len(files_to_process)} images")
    print(f"Output: {OUTPUT_DIR}")

    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "method": args.method,
        "forgery_watermark": "stegastamp",
        "secret_type": "random_per_image" if args.random else "fixed",
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "image_size": f"{WIDTH}x{HEIGHT}",
        "total_images": len(files_list),
        "already_existed": skipped,
        "newly_processed": success_count,
        "total_in_output": skipped + success_count,
    }
    metadata_path = os.path.join(OUTPUT_DIR, 'forgery_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
