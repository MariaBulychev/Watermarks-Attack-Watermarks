#!/usr/bin/env python3
"""
Generate Stable Signature watermarked images.
Supports fixed 48-bit message or --random per-image messages.

Usage:
    python generate_stable_signature.py --prompts_file <prompts.json> --output <out_dir> --mode rand --limit 500
    python generate_stable_signature.py --prompts_file <prompts.json> --output <out_dir> --mode fixed
"""

import os
import sys
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# Add paths
sys.path.append(os.environ.get("TREE_RING_PATH", "./external/tree-ring-watermark"))
sys.path.append(os.environ.get("STABLE_SIG_PATH", "./external/stable_signature/hidden"))

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from models import HiddenEncoder, HiddenDecoder, EncoderWithJND
from attenuations import JND

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def msg2str(msg):
    """Convert message array to binary string."""
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str_msg):
    """Convert binary string to message array."""
    return [True if el=='1' else False for el in str_msg]

class Params():
    """Stable Signature encoder parameters"""
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int):
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        self.attenuation = "jnd"
        self.scale_channels = False
        self.scaling_i = 1
        self.scaling_w = 1.5

def main():
    parser = argparse.ArgumentParser(description='Stable Signature Watermark Generation')
    parser.add_argument('--prompts_file',
                        default='./data/prompts.json',
                        help='Path to prompts JSON file (used when --input_dir not set)')
    parser.add_argument('--input', '--input_dir', dest='input_dir', type=str, default=None,
                        help='Watermark existing images from this dir instead of generating new ones')
    parser.add_argument('--output', '--output_dir', dest='output_dir',
                        default='./data/stable_sig',
                        help='Output directory for watermarked images')
    parser.add_argument('--random', action='store_true',
                        help='Use a random 48-bit message per image (saves messages.json)')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Max images to process (-1 for all)')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    cli_args = parser.parse_args()
    if hasattr(cli_args, 'mode') and cli_args.mode == 'rand':
        cli_args.random = True


    # Configuration
    BASE_SEED = 42
    MODEL_ID = 'stabilityai/stable-diffusion-2-1'
    IMAGE_SIZE = 512
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 7.5
    NUM_BITS = 48
    
    # Stable Signature parameters
    MESSAGE = "111010110101000001010111010011010100010000100111"  # Fixed 48-bit message
    ENCODER_CKPT = os.environ.get("STABLE_SIG_CKPT", "./checkpoints/stable_signature/hidden_replicate.pth")
    
    # Paths
    PROMPTS_FILE = cli_args.prompts_file
    OUTPUT_DIR = cli_args.output_dir
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Determine mode: watermark existing images (--input_dir) or generate new ones (--prompts_file)
    use_existing_images = cli_args.input_dir is not None
    
    if use_existing_images:
        import glob
        image_files = sorted(glob.glob(os.path.join(cli_args.input_dir, '*.png')),
                              key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else os.path.basename(p))
        if cli_args.limit > 0:
            image_files = image_files[:cli_args.limit]
        num_images = len(image_files)
        print(f"Mode: Watermark existing images from {cli_args.input_dir}")
        print(f"Found {num_images} images")
        pipe = None
    else:
        # Load prompts
        print(f"Loading prompts from {PROMPTS_FILE}...")
        with open(PROMPTS_FILE, 'r') as f:
            prompts = json.load(f)
        num_images = len(prompts)
        if cli_args.limit > 0:
            num_images = min(num_images, cli_args.limit)
        print(f"Loaded {num_images} prompts")
        print()
        
        # Load Stable Diffusion pipeline
        print("Loading Stable Diffusion 2.1 base pipeline...")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            scheduler=scheduler,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            revision='fp16' if device == 'cuda' else None,
        )
        pipe = pipe.to(device)
        print("Pipeline loaded successfully!")
    print()
    
    # Load Stable Signature encoder with JND
    print("Loading Stable Signature encoder...")
    params = Params(encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=NUM_BITS)
    
    # Setup transforms
    NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    decoder = HiddenDecoder(
        num_blocks=params.decoder_depth, 
        num_bits=params.num_bits, 
        channels=params.decoder_channels
    )
    encoder = HiddenEncoder(
        num_blocks=params.encoder_depth,
        num_bits=params.num_bits,
        channels=params.encoder_channels
    )
    attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
    encoder_with_jnd = EncoderWithJND(
        encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
    )
    
    state_dict = torch.load(ENCODER_CKPT, map_location='cpu', weights_only=False)['encoder_decoder']
    encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
    decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
    
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    
    encoder_with_jnd = encoder_with_jnd.to(device).eval()
    decoder = decoder.to(device).eval()
    print("Encoder with JND loaded successfully!")
    
    # Message setup
    if cli_args.random:
        print("Mode: RANDOM 48-bit message per image")
        msg = None  # Will generate per image
    else:
        print(f"Message: {MESSAGE}")
        print(f"Message length: {len(MESSAGE)} bits")
        msg_ori = torch.Tensor(str2msg(MESSAGE)).unsqueeze(0).to(device)
        msg = 2 * msg_ori.type(torch.float) - 1  # Convert to -1, 1 range
    print()
    
    # Generate watermarked images
    print(f"Watermarking {num_images} images...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load existing messages if resuming random mode
    messages_file = os.path.join(OUTPUT_DIR, 'messages.json')
    if cli_args.random:
        if os.path.exists(messages_file):
            with open(messages_file, 'r') as f:
                messages_dict = json.load(f)
            print(f"Resuming: loaded {len(messages_dict)} existing messages")
        else:
            messages_dict = {}
    else:
        messages_dict = None
    
    transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    
    for idx in tqdm(range(num_images), desc="Watermarking images"):
        if use_existing_images:
            src_path = image_files[idx]
            fname = os.path.basename(src_path)
        else:
            fname = f"{idx}.png"
        output_path = os.path.join(OUTPUT_DIR, fname)
        
        # Skip if already processed
        if os.path.exists(output_path):
            if cli_args.random and fname in messages_dict:
                continue
            elif not cli_args.random:
                continue
        
        if use_existing_images:
            # Load existing image
            from PIL import Image
            image = Image.open(src_path).convert('RGB')
        else:
            prompt = prompts[str(idx)]
            seed = BASE_SEED + idx
            set_random_seed(seed)
            output = pipe(
                prompt,
                num_images_per_prompt=1,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            )
            image = output.images[0]
        
        # Per-image random message or fixed
        if cli_args.random:
            random_bits = [random.choice([True, False]) for _ in range(NUM_BITS)]
            msg_str = msg2str(random_bits)
            cur_msg_ori = torch.Tensor(random_bits).unsqueeze(0).to(device)
            cur_msg = 2 * cur_msg_ori.type(torch.float) - 1
        else:
            cur_msg = msg
        
        # Convert to tensor for watermarking
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Encode watermark with JND
        with torch.no_grad():
            img_w = encoder_with_jnd(img_tensor, cur_msg)
        
        # Process output
        clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
        clip_img = torch.round(255 * clip_img) / 255 
        watermarked_pil = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
        
        # Save watermarked image
        watermarked_pil.save(output_path)
        
        # Record message
        if cli_args.random:
            messages_dict[fname] = msg_str
            
            # Periodic save
            if (idx + 1) % 500 == 0:
                with open(messages_file, 'w') as f:
                    json.dump(messages_dict, f, indent=2)
    
    # Final save
    if cli_args.random:
        with open(messages_file, 'w') as f:
            json.dump(messages_dict, f, indent=2)
        print(f"Per-image messages saved to: {messages_file}")
    
    print()
    print("="*70)
    print("STABLE SIGNATURE WATERMARK GENERATION COMPLETE!")
    print("="*70)
    print(f"Watermarked {num_images} images")
    print(f"Saved to: {OUTPUT_DIR}")
    if cli_args.random:
        print(f"Mode: RANDOM 48-bit message per image (see messages.json)")
    else:
        print(f"Watermark message: {MESSAGE}")
    print(f"Message length: {NUM_BITS} bits")
    print("="*70)

if __name__ == "__main__":
    main()
    if cli_args.random:
        print(f"Mode: RANDOM 48-bit message per image (see messages.json)")
    else:
        print(f"Watermark message: {MESSAGE}")
    print(f"Message length: {NUM_BITS} bits")
    print("="*70)

if __name__ == "__main__":
    main()
