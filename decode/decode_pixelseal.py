#!/usr/bin/env python3
"""
PixelSeal watermark decode script for WAVES analysis pipeline.
Uses the VideoSeal/PixelSeal model to detect watermarks.
Generates results compatible with WAVES analysis pipeline.

Detection Process:
1. Load image
2. Run through PixelSeal detector
3. Extract bit predictions
4. Compute bit accuracy against ground truth message
5. Store raw bit predictions for later analysis

Usage:
    python decode_pixelseal.py --path <image_dir> --num-images 500 --device cuda:0
"""

import os
import sys
import click
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import warnings
import base64

# Add videoseal to path
VIDEOSEAL_PATH = os.environ.get("VIDEOSEAL_PATH", "./external/videoseal")
sys.path.insert(0, VIDEOSEAL_PATH)

# Add WAVES to path so `from dev import ...` works regardless of cwd
WAVES_PATH = os.environ.get("WAVES_PATH", "./external/WAVES")
sys.path.insert(0, WAVES_PATH)

# Set cache directories
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "./cache/torch")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./cache/hf")

import torchvision.transforms as T

# Import from WAVES dev module
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    check_file_existence,
    existence_to_indices,
    save_json,
    load_json,
)

warnings.filterwarnings("ignore")

# Result directory
RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")

# PixelSeal ground truth message (same pattern used during embedding)
# This should match the message used in generate_pixelseal.py
PIXELSEAL_NBITS = 96  # PixelSeal uses 96 bits


def encode_bits_to_string(bits):
    """Encode binary bits array to base64 string for storage."""
    # Convert to numpy array of uint8 (0 or 1)
    bits_array = np.array(bits, dtype=np.uint8)
    # Pack bits into bytes
    packed = np.packbits(bits_array)
    # Encode to base64
    return base64.b64encode(packed).decode('utf-8')


def init_pixelseal_detector(device):
    """
    Initialize the PixelSeal watermark detector.
    
    Returns:
        model: PixelSeal model
        nbits: Number of bits in the watermark
    """
    print("Loading PixelSeal watermark detector...")
    
    # Change to videoseal directory so relative paths work
    original_dir = os.getcwd()
    os.chdir(VIDEOSEAL_PATH)
    
    import videoseal
    
    model = videoseal.load("pixelseal")
    model = model.to(device)
    model.eval()
    
    nbits = model.embedder.msg_processor.nbits
    
    # Change back to original directory
    os.chdir(original_dir)
    
    print(f"✅ PixelSeal detector loaded")
    print(f"   Number of bits: {nbits}")
    
    return model, nbits


def detect_watermark_single(image_path, model, device):
    """
    Detect PixelSeal watermark in a single image.
    
    Args:
        image_path: Path to image file
        model: PixelSeal model
        device: torch device
    
    Returns:
        bits: List of detected bits (0 or 1)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(image).unsqueeze(0).to(device)  # [1, C, H, W]
    
    # Detect watermark
    with torch.no_grad():
        detection = model.detect(img_tensor)
    
    # Extract bit predictions (skip index 0 which is mask)
    # The detection output contains:
    # - 'preds': tensor of shape [B, 1+nbits] 
    #   - First value (index 0) is a mask/presence prediction
    #   - Remaining values (index 1:) are bit predictions as logits (threshold at 0)
    bit_preds = detection["preds"][:, 1:]  # Shape: [B, nbits]
    
    # Convert to binary
    detected_bits = (bit_preds[0] > 0).cpu().numpy().astype(int).tolist()
    
    return detected_bits


def get_indices(path, limit, subset, subset_limit, num_images=None, key="pixelseal"):
    """Get indices of images to decode.
    
    Args:
        path: Directory containing images
        limit: Maximum index to search (e.g., 5000 means check indices 0-4999)
        subset: Whether to use subset mode
        subset_limit: Limit for subset mode
        num_images: Number of actual images to decode (if None, use limit/subset_limit)
        key: Key name in output JSON
    """
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    
    json_path = os.path.join(RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
    # Determine target number of images to decode
    if num_images is not None:
        target_count = num_images
    elif subset:
        target_count = subset_limit
    else:
        target_count = limit
    
    # Check which images exist in the full range
    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    all_existing_indices = existence_to_indices(image_existences, limit=limit)
    print(f"Found {len(all_existing_indices)} images in {path} (searched indices 0-{limit-1})")
    
    # Check if already decoded
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        # Count how many of the first target_count existing images are already decoded
        decoded_count = 0
        for idx in all_existing_indices[:target_count]:
            if data.get(str(idx), {}).get(key) is not None:
                decoded_count += 1
        
        if decoded_count >= target_count:
            print(f"✅ Already fully decoded {target_count} images: {json_path}")
            return [], all_existing_indices[:target_count]
    
    if not os.path.exists(json_path):
        # Take first target_count existing images
        indices = all_existing_indices[:target_count]
    else:
        # Only decode missing ones from the first target_count existing images
        data = load_json(json_path)
        target_indices = all_existing_indices[:target_count]
        indices = [
            idx for idx in target_indices
            if data.get(str(idx), {}).get(key) is None
        ]
    
    return indices, all_existing_indices[:target_count]


@click.command()
@click.option("--path", type=str, required=True, help="Path to image directory")
@click.option("--subset", is_flag=True, help="Only process subset of images")
@click.option("--limit", type=int, default=LIMIT, help="Maximum index to search (searches 0 to limit-1)")
@click.option("--subset-limit", type=int, default=SUBSET_LIMIT, help="Subset limit")
@click.option("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cuda:1, etc.)")
@click.option("--batch-size", type=int, default=1, help="Batch size for processing")
@click.option("--num-images", type=int, default=None, help="Number of actual images to decode (overrides limit for counting)")
@click.option("--key", type=str, default="pixelseal", help="Key name in output JSON")
def main(path, subset, limit, subset_limit, device, batch_size, num_images, key):
    """
    Decode PixelSeal watermarks from images.
    
    Example:
        python decode_pixelseal.py --path ./data/pixelseal
        python decode_pixelseal.py --path ./data/attacked/mscoco/blurring-0.1-pixelseal
    """
    print("PixelSeal Decoder")
    print(f"Path: {path}")
    print("")
    
    # Setup device
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU (will be slower)")
    
    print(f"Using device: {torch_device}")
    
    # Get indices to process
    indices, all_target_indices = get_indices(path, limit, subset, subset_limit, num_images, key=key)
    
    if len(indices) == 0:
        print("Nothing to decode!")
        return
    
    print(f"Will decode {len(indices)} images")
    
    # Initialize detector
    model, nbits = init_pixelseal_detector(torch_device)
    
    # Prepare output
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    json_path = os.path.join(RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
    # Load existing data or create new with proper structure for target indices
    if os.path.exists(json_path):
        results = load_json(json_path)
        # Extend results to include all target indices
        for idx in all_target_indices:
            if str(idx) not in results:
                results[str(idx)] = {key: None}
            elif key not in results[str(idx)]:
                results[str(idx)][key] = None
    else:
        results = {str(idx): {key: None} for idx in all_target_indices}
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Process images
    print("Decoding PixelSeal watermarks...")
    for idx in tqdm(indices, desc="Processing images"):
        image_path = os.path.join(path, f"{idx}.png")
        
        try:
            bits = detect_watermark_single(image_path, model, torch_device)
            # Store as base64 encoded bits
            results[str(idx)][key] = encode_bits_to_string(bits)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            results[str(idx)][key] = None
    
    # Save results
    save_json(results, json_path)
    print(f"✅ Saved results to: {json_path}")


if __name__ == "__main__":
    main()
