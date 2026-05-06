#!/usr/bin/env python3
"""
Generate StegaStamp watermarked images from stable_diff baseline images.
Applies StegaStamp watermark to all images in stable_diff directory.
Uses ONNX model and follows the exact procedure from encode_image_onnx.py

Usage:
    python generate_stegastamp.py --input <clean_dir> --output <out_dir> --mode rand --limit 500
    python generate_stegastamp.py --input <clean_dir> --output <out_dir> --mode fixed
"""

import os
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

def encode_secret(bch, secret_str):
    """BCH-encode a 7-char secret string into model input array."""
    data = bytearray(secret_str + ' '*(7-len(secret_str)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])  # 4 zeros footer
    return np.array([secret], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='StegaStamp Watermark Generation')
    parser.add_argument('--input', '--input_dir', dest='input_dir',
                        default='./data/real/',
                        help='Input image directory')
    parser.add_argument('--output', '--output_dir', dest='output_dir',
                        default='./data/stegastamp/',
                        help='Output directory for watermarked images')
    parser.add_argument('--secret', default='Stega!!',
                        help='Secret text to embed (max 7 characters)')
    parser.add_argument('--random', action='store_true',
                        help='Use a random 7-char secret per image (saves secrets.json)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Max images to process (-1 for all)')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    args = parser.parse_args()
    # Unified-CLI: --mode {fixed,rand} as alias for --random
    if hasattr(args, 'mode') and args.mode == 'rand':
        args.random = True


    # Configuration
    MODEL_PATH = os.path.join(os.environ.get("WAVES_PATH", "./external/WAVES"), "decoders/stega_stamp.onnx")
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    
    # StegaStamp parameters
    SECRET = args.secret[:7]
    WIDTH = 400
    HEIGHT = 400
    
    print("="*70)
    print("STEGASTAMP WATERMARK GENERATION")
    print("="*70)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model:            {MODEL_PATH}")
    if args.random:
        print(f"Mode:             RANDOM secret per image")
    else:
        print(f"Secret message:   {SECRET}")
    print()
    
    # Load ONNX model
    print("Loading StegaStamp ONNX model...")
    sess = ort.InferenceSession(MODEL_PATH)
    
    print("Model inputs:")
    for input_info in sess.get_inputs():
        print(f"  {input_info.name}: {input_info.shape}")
    print("Model outputs:")
    for output_info in sess.get_outputs():
        print(f"  {output_info.name}: {output_info.shape}")
    print()
    
    # Setup BCH encoder
    print(f"BCH_POLYNOMIAL: {BCH_POLYNOMIAL}")
    print(f"BCH_BITS: {BCH_BITS}")
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    
    # Prepare secret for fixed mode
    if not args.random:
        secret_str = SECRET
        secret_input = encode_secret(bch, secret_str)
        print(f"Encoding secret: '{secret_str}'")
        print(f"Secret binary length: {secret_input.shape[1]}")
    else:
        secret_str = None
        secret_input = None
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of images
    files_list = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')),
                         key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))
    if args.limit > 0:
        files_list = files_list[:args.limit]
    num_images = len(files_list)
    
    if num_images == 0:
        print(f"Error: No images found in {INPUT_DIR}")
        return
    
    print(f"Found {num_images} images to process")
    print()
    
    # Character pool for random secrets
    CHARSET = string.ascii_letters + string.digits + string.punctuation
    
    # Load existing secrets map if resuming random mode
    secrets_path = os.path.join(OUTPUT_DIR, 'secrets.json')
    if args.random:
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                secrets_map = json.load(f)
            print(f"Loaded {len(secrets_map)} existing secrets (resuming)")
        else:
            secrets_map = {}
    else:
        secrets_map = None
    
    # Process each image
    size = (WIDTH, HEIGHT)
    success_count = 0
    
    for filename in tqdm(files_list, desc="Watermarking images"):
        try:
            save_name = os.path.basename(filename)
            out_path = os.path.join(OUTPUT_DIR, save_name)
            
            # Skip if already processed
            if os.path.exists(out_path):
                if args.random and save_name in secrets_map:
                    continue
                elif not args.random:
                    continue
            
            # Determine secret for this image
            if args.random:
                cur_secret_str = ''.join(pyrandom.choices(CHARSET, k=7))
                cur_secret_input = encode_secret(bch, cur_secret_str)
            else:
                cur_secret_str = secret_str
                cur_secret_input = secret_input
            
            # Load and preprocess image
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image, size), dtype=np.float32)
            image /= 255.
            
            # Prepare inputs for ONNX model
            image_input = np.array([image], dtype=np.float32)
            
            # Run inference
            outputs = sess.run(None, {
                'secret': cur_secret_input,
                'image': image_input
            })
            
            # Extract outputs
            stegastamp = outputs[0]  # Watermarked image
            residual = outputs[1]    # Residual
            decoded = outputs[2]     # Decoded secret (for verification)
            
            # Verify decoding (same as original)
            decoded_bits = decoded[0]
            packet_binary_decoded = "".join([str(int(bit > 0.5)) for bit in decoded_bits[:96]])
            packet_decoded = bytes(int(packet_binary_decoded[i : i + 8], 2) for i in range(0, len(packet_binary_decoded), 8))
            packet_decoded = bytearray(packet_decoded)
            data_decoded, ecc_decoded = packet_decoded[:-bch.ecc_bytes], packet_decoded[-bch.ecc_bytes:]
            bitflips = bch.decode_inplace(data_decoded, ecc_decoded)
            
            # Convert to uint8 for saving
            rescaled = (stegastamp[0] * 255).astype(np.uint8)
            
            # Save watermarked image
            im = Image.fromarray(rescaled)
            im.save(out_path)
            
            # Record secret
            if args.random:
                secrets_map[save_name] = cur_secret_str
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    # Save secrets
    if args.random:
        with open(secrets_path, 'w') as f:
            json.dump(secrets_map, f, indent=2)
        print(f"Per-image secrets saved to: {secrets_path}")
    
    print()
    print("="*70)
    print("STEGASTAMP WATERMARK GENERATION COMPLETE!")
    print("="*70)
    print(f"Successfully processed: {success_count}/{num_images} images")
    print(f"Saved to: {OUTPUT_DIR}")
    if args.random:
        print(f"Mode: RANDOM secret per image")
    else:
        print(f"Secret used: '{SECRET}' (same for all images)")
    print("="*70)

if __name__ == "__main__":
    main()
