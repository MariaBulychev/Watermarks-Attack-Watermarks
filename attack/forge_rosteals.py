#!/usr/bin/env python3
"""
Apply RoSteALS forgery watermark on top of existing watermarked images.
This simulates a forgery attack where someone tries to overwrite existing watermarks with RoSteALS.

Usage:
    python forge_rosteals.py --method stable_sig --dataset mscoco --mode rand --limit 500
    python forge_rosteals.py --method tree_ring --dataset diffusiondb --mode rand
"""

import os
import sys
import glob
import json
import argparse
import random
import string
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add RoSteALS to path
ROSTEALS_PATH = os.environ.get("ROSTEALS_PATH", "./external/RoSteALS")
sys.path.append(ROSTEALS_PATH)

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

# Import ECC directly
sys.path.insert(0, os.path.join(ROSTEALS_PATH, 'tools'))
from ecc import ECC

def load_rosteals_model(config_path, weight_path):
    """Load the RoSteALS model"""
    print(f"Loading RoSteALS model from {weight_path}")
    
    config = OmegaConf.load(config_path).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    
    model = instantiate_from_config(config)
    state_dict = torch.load(weight_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if 'global_step' in state_dict:
        print(f'Model checkpoint - Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'Missing keys: {missing}')
    if unexpected:
        print(f'Unexpected keys: {unexpected}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, device

def embed_watermark_batch(model, device, cover_image_path, secret_text, output_path, image_size=256):
    """Embed secret text into cover image using RoSteALS - simplified for batch processing"""
    
    # Load and preprocess cover image
    cover_org = Image.open(cover_image_path).convert('RGB')
    original_w, original_h = cover_org.size
    
    # Transform for model input
    tform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    cover = tform(cover_org).unsqueeze(0).to(device)  # 1, 3, 256, 256
    
    # Encode secret text
    ecc = ECC()
    secret = ecc.encode_text([secret_text])  # 1, 100
    secret = torch.from_numpy(secret).to(device).float()  # 1, 100
    
    # Perform watermark embedding
    with torch.no_grad():
        # Encode cover image to latent space
        z = model.encode_first_stage(cover)
        
        # Embed secret in latent space
        z_embed, _ = model(z, None, secret)
        
        # Decode back to image space
        stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
        
        # Calculate residual and resize to original dimensions
        res = stego.clamp(-1, 1) - cover  # (1,3,256,256) residual
        res = torch.nn.functional.interpolate(res, (original_h, original_w), mode='bilinear')
        res = res.permute(0, 2, 3, 1).cpu().numpy()  # (1,h,w,3)
        
        # Convert back to uint8 image
        stego_uint8 = np.clip(res[0] + np.array(cover_org)/127.5-1., -1, 1) * 127.5 + 127.5  
        stego_uint8 = stego_uint8.astype(np.uint8)  # (h,w, 3), ndarray, uint8
        
        # Save watermarked image
        stego_image = Image.fromarray(stego_uint8)
        stego_image.save(output_path)
        
        return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply RoSteALS forgery watermark to images')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mscoco', 'diffusiondb', 'diffusiondb_2'],
                        help='Dataset name (mscoco or diffusiondb)')
    parser.add_argument('--method', type=str, required=True,
                        help='Watermark method to process (e.g. pixelseal, rosteals_rand)')
    parser.add_argument('--secret', type=str, default='Forge42',
                        help='Forgery secret (max 7 characters, default: F0rg342)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Model input image size (default: 256)')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to generate (default: all images)')
    parser.add_argument('--random-secret', action='store_true',
                        help='Use a different random secret for each image')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Custom input directory (default: auto from dataset/method)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory (default: auto from dataset/method)')
    args = parser.parse_args()
    
    # Configuration
    BASE_DATA_DIR = os.environ.get('DATA_DIR', './data')
    MODEL_PATH = os.path.join(ROSTEALS_PATH, "models", "RoSteALS", "epoch=000017-step=000449999.ckpt")
    CONFIG_PATH = os.path.join(ROSTEALS_PATH, "models", "VQ4_mir_inference.yaml")
    
    # Build input and output paths (use custom if provided, otherwise auto)
    if args.input_dir:
        INPUT_DIR = args.input_dir
    else:
        INPUT_DIR = os.path.join(BASE_DATA_DIR, 'main', args.dataset, args.method)
    
    # Use "7.0" for random mode, "1.0" for fixed mode
    strength_suffix = "7.0" if args.random_secret else "1.0"
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'attacked', args.dataset, f'rosteals_forgery-{strength_suffix}-{args.method}')
    
    # Forgery watermark secret (different from original)
    FORGERY_SECRET = args.secret[:7]  # Limit to 7 characters
    USE_RANDOM_SECRET = args.random_secret
    
    def generate_random_secret(length=7):
        """Generate a random 7-character secret"""
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choice(chars) for _ in range(length))
    
    print("="*70)
    print("ROSTEALS FORGERY WATERMARK GENERATION")
    print("="*70)
    print(f"Dataset:              {args.dataset}")
    print(f"Method:               {args.method}")
    print(f"Input directory:      {INPUT_DIR}")
    print(f"Output directory:     {OUTPUT_DIR}")
    print(f"Model:                {MODEL_PATH}")
    if USE_RANDOM_SECRET:
        print(f"Secret mode:          RANDOM (different per image)")
    else:
        print(f"Forgery secret:       {FORGERY_SECRET}")
    print(f"Device:               {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory created/verified: {OUTPUT_DIR}")
    print()
    
    # Get list of images from input directory (sort numerically)
    files_list = sorted(
        glob.glob(os.path.join(INPUT_DIR, '*.png')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))
    total_available = len(files_list)
    
    if total_available == 0:
        print(f"ERROR: No images found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Found {total_available} images in input folder.")
    
    # Limit to num_images if specified
    if args.num_images is not None:
        files_list = files_list[:args.num_images]
        num_images = len(files_list)
        print(f"Limiting to first {num_images} images (--num-images={args.num_images})")
    else:
        num_images = total_available
    
    # Check which images already exist
    files_to_process = []
    skipped_count = 0
    for filename in files_list:
        save_name = os.path.basename(filename)
        output_path = os.path.join(OUTPUT_DIR, save_name)
        if os.path.exists(output_path):
            skipped_count += 1
        else:
            files_to_process.append(filename)
    
    print(f"Already processed: {skipped_count} images")
    print(f"To process: {len(files_to_process)} images")
    print()
    
    if len(files_to_process) == 0:
        print("All images already processed. Nothing to do.")
        return
    
    # Load RoSteALS model
    model, device = load_rosteals_model(CONFIG_PATH, MODEL_PATH)
    print()
    
    # Process images
    success_count = 0
    secrets_used = {}  # Track secrets for metadata when using random mode
    
    for filename in tqdm(files_to_process, desc=f"Forging {args.method}"):
        try:
            save_name = os.path.basename(filename)
            output_path = os.path.join(OUTPUT_DIR, save_name)
            
            # Generate secret for this image
            if USE_RANDOM_SECRET:
                current_secret = generate_random_secret()
                secrets_used[save_name] = current_secret
            else:
                current_secret = FORGERY_SECRET
            
            success = embed_watermark_batch(
                model, device, filename, current_secret, 
                output_path, args.image_size
            )
            
            if success:
                success_count += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    print(f"\nSuccessfully processed {success_count}/{len(files_to_process)} new images")
    
    print(f"\n{'='*70}")
    print("FORGERY GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Dataset:              {args.dataset}")
    print(f"Method:               {args.method}")
    print(f"Already existed:      {skipped_count}")
    print(f"Newly processed:      {success_count}")
    print(f"Total in output:      {skipped_count + success_count}/{num_images}")
    if USE_RANDOM_SECRET:
        print(f"Secret mode:          RANDOM (different per image)")
    else:
        print(f"Forgery secret:       {FORGERY_SECRET}")
    print(f"Output directory:     {OUTPUT_DIR}")
    
    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "method": args.method,
        "forgery_watermark": "rosteals",
        "forgery_secret": "RANDOM" if USE_RANDOM_SECRET else FORGERY_SECRET,
        "random_secret_mode": USE_RANDOM_SECRET,
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "model_path": MODEL_PATH,
        "image_size": args.image_size,
        "total_images": num_images,
        "already_existed": skipped_count,
        "newly_processed": success_count,
        "total_in_output": skipped_count + success_count
    }
    
    # Save per-image secrets if using random mode
    if USE_RANDOM_SECRET and secrets_used:
        secrets_path = os.path.join(OUTPUT_DIR, 'random_secrets.json')
        with open(secrets_path, 'w') as f:
            json.dump(secrets_used, f, indent=2)
        print(f"\nRandom secrets saved to: {secrets_path}")
    
    metadata_path = os.path.join(OUTPUT_DIR, 'forgery_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
