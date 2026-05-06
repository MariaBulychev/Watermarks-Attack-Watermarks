#!/usr/bin/env python3
"""
ZoDiac forgery attack — re-embed a ZoDiac watermark on top of an already-
watermarked image. Multi-GPU enabled.

Usage:
    python forge_zodiac.py --method pixelseal --dataset mscoco --mode rand
    python forge_zodiac.py --method rosteals  --dataset diffusiondb --mode rand --gpus 2
    python forge_zodiac.py --input <dir> --output <dir> --mode rand     # explicit dirs
"""

import os
import sys
import glob
import json
import argparse
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
import logging
import numpy as np
from PIL import Image 
import torch.optim as optim
import torchvision.transforms as transforms
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

# Add ZoDiac to path
sys.path.insert(0, os.environ.get("ZODIAC_PATH", "./external/ZoDiac"))

from main.wmdiffusion import WMDetectStableDiffusionPipeline
from main.wmpatch import GTWatermark, GTWatermarkMulti
from main.utils import save_img, get_img_tensor, watermark_prob
from loss.loss import LossProvider
from loss.pytorch_ssim import ssim

# Setup logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    """Get init noise using DDIM inversion (from example.ipynb)"""
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    return reversed_latents

class ZoDiacBatchProcessor:
    """Optimized ZoDiac processor with batch pipeline loading"""
    
    def __init__(self, device_ids=None, random_mode=False, seed_offset=0):
        self.random_mode = random_mode
        self.seed_offset = seed_offset
        # Configuration (matching config.yaml with paper defaults)
        self.cfgs = {
            # IMPORTANT: Must use -base model (512x512), not the 768x768 variant!
            'model_id': 'Manojb/stable-diffusion-2-1-base',
            'w_type': 'single',
            'w_channel': 3,
            'w_radius': 10,
            'w_seed': 10,
            'empty_prompt': True,
            'iters': 100,  # Paper default
            'save_iters': [100],
            'loss_weights': [10.0, 0.1, 1.0, 0.0],  # L2, watson-vgg, SSIM, watermark L1
            'ssim_threshold': 0.92, # this could be our "strength" parameter
            'detect_threshold': 0.9
        }
        
        # Setup devices
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
        else:
            self.device_ids = device_ids
            
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using devices: {self.device_ids}")
        logger.info(f"Primary device: {self.primary_device}")
        
        # Initialize components once
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize pipeline components once - major speedup"""
        logger.info('===== Initializing Pipeline (One Time) =====')
        
        # Change to ZoDiac directory for loss model loading
        original_dir = os.getcwd()
        os.chdir(os.environ.get("ZODIAC_PATH", "./external/ZoDiac"))
        
        # Initialize watermark (will be re-created per image in random mode)
        if not self.random_mode:
            self.wm_pipe = GTWatermark(
                self.primary_device, 
                w_channel=self.cfgs['w_channel'], 
                w_radius=self.cfgs['w_radius'], 
                generator=torch.Generator(self.primary_device).manual_seed(self.cfgs['w_seed'])
            )
        else:
            self.wm_pipe = None  # will be set per image
        
        # Initialize diffusion pipeline (load once!)
        scheduler = DDIMScheduler.from_pretrained(self.cfgs['model_id'], subfolder="scheduler")
        self.pipe = WMDetectStableDiffusionPipeline.from_pretrained(self.cfgs['model_id'], scheduler=scheduler).to(self.primary_device)
        self.pipe.set_progress_bar_config(disable=True)
        
        # Initialize loss function (load once!)
        self.totalLoss = LossProvider(self.cfgs['loss_weights'], self.primary_device)
        
        # Multi-GPU setup if available
        if len(self.device_ids) > 1 and torch.cuda.is_available():
            logger.info(f"Setting up multi-GPU processing on devices: {self.device_ids}")
            # Note: DataParallel for diffusion models can be tricky, so we'll use manual distribution
            
        # Get text embeddings once (reuse for all images)
        self.empty_text_embeddings = self.pipe.get_text_embedding('')
        
        os.chdir(original_dir)
        logger.info('===== Pipeline Initialized =====')
    
    def watermark_single_image(self, input_image, output_image):
        """Process single image with pre-loaded pipeline. Returns (success, seed_used)."""
        
        if not os.path.exists(input_image):
            logger.error(f"Input image not found: {input_image}")
            return False, None
        
        seed_used = None
        try:
            # In random mode, derive seed from image filename (e.g. 196.png -> seed 196)
            # This makes seeds deterministic and recoverable without saving to a file
            if self.random_mode:
                seed_used = int(os.path.splitext(os.path.basename(input_image))[0]) + self.seed_offset
                self.wm_pipe = GTWatermark(
                    self.primary_device,
                    w_channel=self.cfgs['w_channel'],
                    w_radius=self.cfgs['w_radius'],
                    generator=torch.Generator(self.primary_device).manual_seed(seed_used)
                )
                logger.info(f"Random mode: using seed {seed_used} (from filename)")

            # Load image and resize to 512x512 if needed (ZoDiac requires 512x512)
            from PIL import Image as PILImage
            from torchvision.transforms.functional import pil_to_tensor
            img_pil = PILImage.open(input_image).convert("RGB")
            if img_pil.size != (512, 512):
                img_pil = img_pil.resize((512, 512), PILImage.LANCZOS)
            gt_img_tensor = (pil_to_tensor(img_pil) / 255.0).unsqueeze(0).to(self.primary_device)
            
            # Step 1: Get init noise (reuse text embeddings)
            init_latents_approx = get_init_latent(gt_img_tensor, self.pipe, self.empty_text_embeddings)
            
            # Step 2: Prepare training
            init_latents = init_latents_approx.detach().clone()
            init_latents.requires_grad = True
            optimizer = optim.Adam([init_latents], lr=0.01)
            scheduler_opt = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70], gamma=0.3)
            
            # Step 3: Train the init latents
            for i in range(self.cfgs['iters']):
                init_latents_wm = self.wm_pipe.inject_watermark(init_latents)
                
                if self.cfgs['empty_prompt']:
                    pred_img_tensor = self.pipe('', guidance_scale=1.0, num_inference_steps=50, 
                                               output_type='tensor', use_trainable_latents=True, 
                                               init_latents=init_latents_wm).images
                
                loss = self.totalLoss(pred_img_tensor, gt_img_tensor, init_latents_wm, self.wm_pipe)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler_opt.step()
                
                # Less frequent logging for batch processing
                if (i + 1) % 25 == 0:
                    logger.info(f'iter {i+1}: Loss {loss.item():.4f}')
            
            # Save watermarked image
            save_img(output_image, pred_img_tensor, self.pipe)
            
            # Clear memory but keep pipeline loaded
            del init_latents, init_latents_wm, pred_img_tensor, loss, optimizer, scheduler_opt
            torch.cuda.empty_cache()
            
            return True, seed_used
            
        except Exception as e:
            logger.error(f"Error watermarking {input_image}: {e}")
            import traceback
            traceback.print_exc()
            return False, None

def process_gpu_batch(gpu_id, image_batch, output_dir, start_idx, random_mode=False, seeds_dict=None, seed_offset=0):
    """Process a batch of images on specific GPU"""
    torch.cuda.set_device(gpu_id)
    
    processor = ZoDiacBatchProcessor(device_ids=[gpu_id], random_mode=random_mode, seed_offset=seed_offset)
    
    success_count = 0
    for i, img_path in enumerate(image_batch):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        
        # Skip if exists
        if os.path.exists(output_path):
            logger.info(f"GPU {gpu_id}: Skipping {img_name} - already exists")
            success_count += 1
            continue
            
        logger.info(f"GPU {gpu_id}: Processing {img_name} ({start_idx + i + 1})")
        
        success, seed_used = processor.watermark_single_image(img_path, output_path)
        if success:
            logger.info(f"GPU {gpu_id}: ✓ Success: {img_name}")
            success_count += 1
            if seeds_dict is not None and seed_used is not None:
                seeds_dict[img_name] = seed_used
        else:
            logger.error(f"GPU {gpu_id}: ✗ Failed: {img_name}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description='ZoDiac forgery — re-watermark already-watermarked images')
    parser.add_argument('--method', type=str, default=None,
                        help='Watermark method whose images to forge (e.g. pixelseal, rosteals_rand). '
                             'Used together with --dataset to compute default --input/--output.')
    parser.add_argument('--dataset', type=str, default='mscoco',
                        choices=['mscoco', 'diffusiondb'],
                        help='Dataset name (default: mscoco)')
    parser.add_argument('--input', dest='input_dir', default=None,
                        help='Input directory (already-watermarked images). '
                             'Defaults to $DATA_DIR/main/<dataset>/<method> when --method is given.')
    parser.add_argument('--output', dest='output_dir', default=None,
                        help='Output directory. '
                             'Defaults to $DATA_DIR/attacked/<dataset>/forgery_zodiac_corrected-1.0-<method>.')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs to use')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    parser.add_argument('--start', type=int, default=None, help='Start from image with this numeric index (inclusive)')
    parser.add_argument('--random', action='store_true', help='Use random watermark seed per image (alias of --mode rand)')
    parser.add_argument('--seed-offset', type=int, default=2_000_000,
                        help='Offset added to filename-derived seed in rand mode. Default 2_000_000 to avoid collision with a victim watermark embedded at offset 0.')
    parser.add_argument('--mode', choices=['fixed', 'rand'], default='fixed',
                        help='Watermark mode (alias for --random)')
    args = parser.parse_args()
    # Unified-CLI: --mode {fixed,rand} as alias for --random
    if hasattr(args, 'mode') and args.mode == 'rand':
        args.random = True

    # Resolve --input/--output from --method/--dataset if not given
    if args.input_dir is None or args.output_dir is None:
        if args.method is None:
            parser.error('Either --input/--output or --method (with --dataset) must be provided.')
        base = os.environ.get('DATA_DIR', './data')
        if args.input_dir is None:
            args.input_dir = os.path.join(base, 'main', args.dataset, args.method)
        if args.output_dir is None:
            args.output_dir = os.path.join(base, 'attacked', args.dataset,
                                           f'forgery_zodiac_corrected-1.0-{args.method}')

    print(f"=== ZoDiac forgery — batch watermarking ===")
    print(f"Dataset: {args.dataset}    Method: {args.method}")
    print(f"Input:   {args.input_dir}")
    print(f"Output:  {args.output_dir}")
    
    # Setup multi-GPU
    num_gpus = torch.cuda.device_count() if args.gpus is None else min(args.gpus, torch.cuda.device_count())
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Using GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 1
        print("CUDA not available, using CPU")
    
    # Find all images (numeric sort by filename)
    image_files = glob.glob(os.path.join(args.input_dir, "*.png"))
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # Filter by start index if specified
    if args.start is not None:
        image_files = [f for f in image_files if int(os.path.splitext(os.path.basename(f))[0]) >= args.start]
        print(f"Starting from index {args.start}")
    
    # Apply limit if specified
    if args.limit is not None:
        image_files = image_files[:args.limit]
        print(f"Limiting to {args.limit} images")
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    seeds_record = {}  # track random seeds per image

    if num_gpus == 1:
        # Single GPU processing with optimized pipeline
        processor = ZoDiacBatchProcessor(random_mode=args.random, seed_offset=args.seed_offset)
        
        success_count = 0
        for i, img_path in enumerate(image_files):
            img_name = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, img_name)
            
            # Skip if exists
            if os.path.exists(output_path):
                print(f"Skipping {img_name} - already exists", flush=True)
                success_count += 1
                continue
                
            print(f"\n=== Processing {img_name} ({i+1}/{len(image_files)}) ===", flush=True)
            
            success, seed_used = processor.watermark_single_image(img_path, output_path)
            if success:
                print(f"✓ Success: {img_name}", flush=True)
                success_count += 1
                if seed_used is not None:
                    seeds_record[img_name] = seed_used
            else:
                print(f"✗ Failed: {img_name}", flush=True)
        
        print(f"\n=== Complete: {success_count}/{len(image_files)} successful ===", flush=True)
    
    else:
        # Multi-GPU processing
        print(f"Distributing {len(image_files)} images across {num_gpus} GPUs", flush=True)
        
        # Split images across GPUs
        images_per_gpu = len(image_files) // num_gpus
        remainder = len(image_files) % num_gpus
        
        processes = []
        start_idx = 0
        
        # Use a multiprocessing Manager dict so subprocesses can write seeds
        manager = mp.Manager()
        shared_seeds = manager.dict()

        for gpu_id in range(num_gpus):
            # Distribute remainder across first few GPUs
            batch_size = images_per_gpu + (1 if gpu_id < remainder else 0)
            image_batch = image_files[start_idx:start_idx + batch_size]
            
            print(f"GPU {gpu_id}: Processing {len(image_batch)} images (indices {start_idx}-{start_idx + len(image_batch) - 1})", flush=True)
            
            # Start process for this GPU
            p = mp.Process(target=process_gpu_batch, 
                         args=(gpu_id, image_batch, args.output_dir, start_idx, args.random, shared_seeds, args.seed_offset))
            p.start()
            processes.append(p)
            
            start_idx += batch_size
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        seeds_record.update(dict(shared_seeds))
        print(f"\n=== Multi-GPU Processing Complete ===", flush=True)

    # Seeds are derived from filenames (e.g. 196.png -> seed 196), no file needed
    if args.random:
        print(f"Used filename-derived seeds for {len(seeds_record)} images", flush=True)

if __name__ == "__main__":
    # Enable multiprocessing
    mp.set_start_method('spawn', force=True)
    main()