#!/usr/bin/env python3
"""
Generate RoSteALS watermarked images from existing Stable Diffusion images.
Takes images from main/mscoco/stable_diff/ and embeds RoSteALS watermarks.
Saves watermarked images to main/mscoco/rosteals/

Usage:
    python generate_rosteals.py --input <clean_dir> --output <out_dir> --mode rand --limit 500
    python generate_rosteals.py --input <clean_dir> --output <out_dir> --mode fixed
"""

import os
import sys
import json
import string
import random as pyrandom
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path
from tqdm import tqdm
import glob

# Add RoSteALS to path
ROSTEALS_PATH = os.environ.get("ROSTEALS_PATH", "./external/RoSteALS")
sys.path.append(ROSTEALS_PATH)

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Import ECC directly
sys.path.insert(0, os.path.join(ROSTEALS_PATH, 'tools'))
from ecc import ECC

def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def compute_ssim(img1, img2):
    """Compute SSIM between two images"""
    return structural_similarity(img1[0], img2[0], channel_axis=2, data_range=255)

def compute_mse(img1, img2):
    """Compute MSE between two images"""
    return mean_squared_error(img1, img2)

def load_rosteals_model(config_path, weight_path, device):
    """Load the RoSteALS model"""
    print(f"Loading RoSteALS model from {weight_path}")
    
    config = OmegaConf.load(config_path).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    
    model = instantiate_from_config(config)
    state_dict = torch.load(weight_path, map_location=device)
    
    if 'global_step' in state_dict:
        print(f'Model checkpoint - Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'Missing keys: {missing}')
    if unexpected:
        print(f'Unexpected keys: {unexpected}')
    
    model = model.to(device)
    model.eval()
    
    return model

def embed_watermark(model, device, cover_image_path, secret_text, output_path, image_size=256):
    """Embed secret text into cover image using RoSteALS"""
    
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
        
        # Compute quality metrics
        cover_array = np.array(cover_org)
        mse = compute_mse(cover_array[None,...], stego_uint8[None,...])
        psnr = compute_psnr(cover_array[None,...], stego_uint8[None,...])
        ssim = compute_ssim(cover_array[None,...], stego_uint8[None,...])
        
        # Test secret recovery for validation
        stego_tensor = torch.from_numpy(stego_uint8[None,...]/127.5-1.).permute(0,3,1,2).float().to(device)
        stego_resized = torch.nn.functional.interpolate(stego_tensor, (image_size, image_size), mode='bilinear')
        
        secret_pred = (model.decoder(stego_resized) > 0).cpu().numpy()  # 1, 100
        bit_accuracy = np.mean(secret_pred == secret.cpu().numpy())
        
        # Save watermarked image
        stego_image = Image.fromarray(stego_uint8)
        stego_image.save(output_path)
        
        return {
            'mse': mse,
            'psnr': psnr, 
            'ssim': ssim,
            'bit_accuracy': bit_accuracy
        }

def main():
    parser = argparse.ArgumentParser(description='RoSteALS Batch Watermarking')
    parser.add_argument('--input', '--input_dir', dest='input_dir', 
                       default='./data/real',
                       help='Input directory with images to watermark')
    parser.add_argument('--output', '--output_dir', dest='output_dir', 
                       default='./data/rosteals',
                       help='Output directory for watermarked images')
    parser.add_argument('--secret', 
                       default='RoSteAL',
                       help='Secret text to embed (max 7 characters)')
    parser.add_argument('--image_size', 
                       type=int, default=256,
                       help='Model input image size')
    parser.add_argument('--limit', 
                       type=int, default=None,
                       help='Limit number of images to process (for testing)')
    parser.add_argument('--random', action='store_true',
                       help='Use a random 7-char secret per image (saves secrets.json)')
    
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    args = parser.parse_args()
    # Unified-CLI: --mode {fixed,rand} as alias for --random
    if hasattr(args, 'mode') and args.mode == 'rand':
        args.random = True

    
    print("=== RoSteALS Batch Watermarking ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.random:
        print(f"Mode: RANDOM secret per image")
    else:
        print(f"Secret text: '{args.secret}' ({len(args.secret)} characters)")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Validate secret length (only for fixed mode)
    if not args.random and len(args.secret) > 7:
        print(f"Warning: Secret text '{args.secret}' is {len(args.secret)} characters long.")
        print("RoSteALS supports maximum 7 characters due to BCH error correction.")
        args.secret = args.secret[:7]
        print(f"Truncated to: '{args.secret}'")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration and model paths
    config_path = os.path.join(ROSTEALS_PATH, "models", "VQ4_mir_inference.yaml")
    model_path = os.path.join(ROSTEALS_PATH, "models", "RoSteALS", "epoch=000017-step=000449999.ckpt")
    
    # Check if models exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run download_models.sh in RoSteALS directory first")
        return
    
    # Load model
    model = load_rosteals_model(config_path, model_path, device)
    
    # Find all images in input directory
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    image_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))  # Sort numerically
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print(f"No images found in {args.input_dir}")
        return
    
    # Load existing secrets map if resuming random mode
    secrets_path = os.path.join(args.output_dir, 'secrets.json')
    if args.random:
        if os.path.exists(secrets_path):
            with open(secrets_path, 'r') as f:
                secrets_map = json.load(f)
            print(f"Loaded {len(secrets_map)} existing secrets (resuming)")
        else:
            secrets_map = {}
    else:
        secrets_map = None
    
    # Character pool for random secrets
    CHARSET = string.ascii_letters + string.digits + string.punctuation
    
    # Process images
    total_metrics = {'mse': [], 'psnr': [], 'ssim': [], 'bit_accuracy': []}
    failed_images = []
    
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Get output filename
            image_name = os.path.basename(image_path)
            output_path = os.path.join(args.output_dir, image_name)
            
            # Skip if already exists
            if os.path.exists(output_path):
                if args.random and image_name in secrets_map:
                    continue
                elif not args.random:
                    print(f"Skipping {image_name} (already exists)")
                    continue
            
            # Determine secret for this image
            if args.random:
                cur_secret = ''.join(pyrandom.choices(CHARSET, k=7))
            else:
                cur_secret = args.secret
            
            # Embed watermark
            metrics = embed_watermark(
                model, device, image_path, cur_secret, output_path, args.image_size
            )
            
            # Record secret
            if args.random:
                secrets_map[image_name] = cur_secret
            
            # Collect metrics
            for key in total_metrics:
                total_metrics[key].append(metrics[key])
            
            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Failed to process {image_path}: {str(e)}")
            failed_images.append(image_path)
            continue
    
    # Print final statistics
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {len(total_metrics['psnr'])} images")
    print(f"Failed: {len(failed_images)} images")
    
    if total_metrics['psnr']:
        print(f"\n=== Quality Metrics (Average) ===")
        print(f"MSE: {np.mean(total_metrics['mse']):.4f} ± {np.std(total_metrics['mse']):.4f}")
        print(f"PSNR: {np.mean(total_metrics['psnr']):.4f} ± {np.std(total_metrics['psnr']):.4f}")
        print(f"SSIM: {np.mean(total_metrics['ssim']):.4f} ± {np.std(total_metrics['ssim']):.4f}")
        print(f"Bit Accuracy: {np.mean(total_metrics['bit_accuracy']):.4f} ± {np.std(total_metrics['bit_accuracy']):.4f}")
    
    if failed_images:
        print(f"\nFailed images:")
        for failed in failed_images:
            print(f"  {failed}")
    
    # Save secrets
    if args.random:
        with open(secrets_path, 'w') as f:
            json.dump(secrets_map, f, indent=2)
        print(f"Per-image secrets saved to: {secrets_path}")
    
    print(f"\nWatermarked images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()