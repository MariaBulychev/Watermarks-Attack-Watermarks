#!/usr/bin/env python3
"""
WAM (Watermark Anything Model) decode script for WAVES analysis pipeline.
Decodes watermarks from images and computes bit error rate.
Generates results compatible with WAVES analysis pipeline.

Detection Process:
1. Load image
2. Pass through WAM detector network
3. Extract predicted message bits using mask-weighted averaging
4. Compare against ground truth message
5. Compute bit error rate (lower = watermark detected)

Usage:
    python decode_wam.py --path <image_dir> --num-images 500 --device cuda:0
"""

import os
import sys
import click
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import warnings

# Add WAM to path
WAM_REPO = os.environ.get("WAM_PATH", "./external/watermark-anything")
sys.path.insert(0, WAM_REPO)

# WAM imports
from watermark_anything.data.transforms import default_transform
from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import load_model_from_checkpoint

# Import from WAVES dev module - only import what we need to avoid plotly dependency
sys.path.append(os.environ.get("WAVES_PATH", "./external/WAVES"))
from dev.constants import LIMIT, SUBSET_LIMIT
from dev.find import check_file_existence, existence_to_indices
from dev.io import save_json, load_json, encode_array_to_string

warnings.filterwarnings("ignore")

# Use result directory
RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")

# Ground truth message (generated with torch.manual_seed(42))
# Binary string: "01000100010000101110101111111100"
GROUND_TRUTH_MSG = torch.tensor([
    0, 1, 0, 0, 0, 1, 0, 0,  # 01000100
    0, 1, 0, 0, 0, 0, 1, 0,  # 01000010
    1, 1, 1, 0, 1, 0, 1, 1,  # 11101011
    1, 1, 1, 1, 1, 1, 0, 0   # 11111100
], dtype=torch.bool)


def init_wam_detector(device):
    """
    Initialize the WAM watermark detector.
    Loads the model.
    
    Returns:
        wam: WAM model
    """
    print("Loading WAM watermark detector...")
    
    # Save current directory and change to WAM repo (required for config paths)
    original_dir = os.getcwd()
    os.chdir(WAM_REPO)
    
    exp_dir = os.path.join(WAM_REPO, "checkpoints")
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, "wam_mit.pth")
    
    if not os.path.exists(ckpt_path):
        # Try alternative checkpoint
        ckpt_path = os.path.join(exp_dir, "wam_large_adversarial.pth")
    
    print(f"  Loading from {ckpt_path}")
    
    wam = load_model_from_checkpoint(json_path, ckpt_path)
    wam = wam.to(device)
    wam.eval()
    
    # Restore original directory
    os.chdir(original_dir)
    
    print("✅ WAM detector loaded")
    
    return wam


def detect_watermark_single(image_path, wam, device, gt_msg):
    """
    Detect WAM watermark in a single image and compute bit error rate.
    
    Args:
        image_path: Path to image file
        wam: WAM model
        device: torch device
        gt_msg: Ground truth message tensor (32 bits) - unused, kept for API compatibility
    
    Returns:
        decoded_bits: numpy array of decoded bits (32 bits)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Transform for model input
    img_tensor = default_transform(img).unsqueeze(0).to(device)
    
    # Detect watermark
    with torch.no_grad():
        outputs = wam.detect(img_tensor)
        preds = outputs["preds"]  # [1, 33, H, W] - first channel is mask, rest are bits
        
        # Extract mask and bit predictions
        mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, H, W] predicted watermark mask
        bit_preds = preds[:, 1:, :, :]  # [1, 32, H, W] predicted bits per pixel
        
        # Predict message by averaging over masked pixels
        pred_message = msg_predict_inference(bit_preds, mask_preds.unsqueeze(1), method='semihard')  # [1, 32]
        pred_message = pred_message[0].cpu()  # [32]
    
    # Return decoded bits as numpy bool array
    return pred_message.bool().numpy()


def get_indices(path, limit, subset, subset_limit, num_images=None, output_json=None, key="wam"):
    """Get indices of images to decode.
    
    Args:
        path: Directory containing images
        limit: Maximum index to search (e.g., 5000 means check indices 0-4999)
        subset: Whether to use subset mode
        subset_limit: Limit for subset mode
        num_images: Number of actual images to decode (if None, use limit/subset_limit)
        output_json: Custom output JSON path (if None, auto-generate)
        key: Key name in output JSON
    """
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    
    if output_json:
        json_path = output_json
    else:
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
    
    def needs_decode(val):
        """Check if value needs re-decoding (None or old numeric format instead of bits string)"""
        if val is None:
            return True
        # Old format was a number (bit_error_rate), new format is base64 encoded bits string
        if isinstance(val, (int, float)):
            return True
        # JSON loads numbers as strings - check if it's a string that looks like a number
        if isinstance(val, str):
            try:
                float(val)
                return True  # It's an old numeric value stored as string
            except ValueError:
                pass
        return False
    
    # Check if already decoded
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        # Count how many of the first target_count existing images are already decoded with NEW format
        decoded_count = 0
        for idx in all_existing_indices[:target_count]:
            val = data.get(str(idx), {}).get(key)
            if not needs_decode(val):
                decoded_count += 1
        
        if decoded_count >= target_count:
            print(f"✅ Already fully decoded {target_count} images: {json_path}")
            return [], all_existing_indices[:target_count]
    
    if not os.path.exists(json_path):
        # Take first target_count existing images
        indices = all_existing_indices[:target_count]
    else:
        # Only decode missing ones or old format from the first target_count existing images
        data = load_json(json_path)
        target_indices = all_existing_indices[:target_count]
        indices = [
            idx for idx in target_indices
            if needs_decode(data.get(str(idx), {}).get(key))
        ]
    
    return indices, all_existing_indices[:target_count]


@click.command()
@click.option("--path", type=str, required=True, help="Path to image directory")
@click.option("--subset", is_flag=True, help="Only process subset of images")
@click.option("--limit", type=int, default=LIMIT, help="Maximum index to search (searches 0 to limit-1)")
@click.option("--subset-limit", type=int, default=SUBSET_LIMIT, help="Subset limit")
@click.option("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cuda:1, etc.)")
@click.option("--num-images", type=int, default=None, help="Number of actual images to decode (overrides limit for counting)")
@click.option("--output-json", type=str, default=None, help="Custom output JSON path (overrides auto-generated path)")
@click.option("--key", type=str, default="wam", help="Key name in output JSON")
def main(path, subset, limit, subset_limit, device, num_images, output_json, key):
    """
    Decode WAM watermarks.
    
    Example:
        python decode_wam.py --path ./data/wam
        python decode_wam.py --path ./data/attacked/mscoco/blurring-0.1-wam
    """
    # Setup device
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU (will be very slow)")
    
    print(f"Using device: {torch_device}")
    
    # Get indices to process
    indices, all_target_indices = get_indices(path, limit, subset, subset_limit, num_images, output_json, key=key)
    
    if len(indices) == 0:
        print("Nothing to decode!")
        return
    
    print(f"Will decode {len(indices)} images")
    
    # Initialize detector
    wam = init_wam_detector(torch_device)
    
    # Prepare output
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    if output_json:
        json_path = output_json
    else:
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
    
    # Process images one at a time
    print("Decoding WAM watermarks...")
    gt_msg = GROUND_TRUTH_MSG  # Keep for reference but not used in decoding
    
    for idx in tqdm(indices, desc="Processing images"):
        image_path = os.path.join(path, f"{idx}.png")
        
        try:
            decoded_bits = detect_watermark_single(
                image_path, wam, torch_device, gt_msg
            )
            # Store decoded bits as encoded string (WAVES framework computes BER later)
            results[str(idx)][key] = encode_array_to_string(decoded_bits)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            results[str(idx)][key] = None
    
    # Save results
    save_json(results, json_path)
    print(f"✅ Saved results to: {json_path}")


if __name__ == "__main__":
    main()
