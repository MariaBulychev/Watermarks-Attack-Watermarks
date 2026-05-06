#!/usr/bin/env python3
"""
Corrected decode script using the same PyTorch decoder as generation.
Generates results compatible with WAVES analysis pipeline.

Usage:
    python decode_stable_sig.py --path <image_dir> --num-images 500 --device cuda:0
"""

import os
import sys
import click
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import dotenv
import warnings
import glob

# Add the stable_signature/hidden directory to the path
sys.path.append(os.environ.get("STABLE_SIG_PATH", "./external/stable_signature/hidden"))
from models import HiddenDecoder

# Import from WAVES dev module
from dev import (
    LIMIT,
    SUBSET_LIMIT,
    check_file_existence,
    existence_to_indices,
    save_json,
    load_json,
    encode_array_to_string,
)

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")

# Use result directory
CORRECTED_RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")


def init_stable_sig_decoder(device):
    """Initialize the PyTorch Stable Signature decoder (same as generation script)."""
    print("Loading Stable Signature decoder...")
    decoder = HiddenDecoder(num_blocks=8, num_bits=48, channels=64)
    
    state_dict = torch.load(
        os.environ.get("STABLE_SIG_CKPT", "./checkpoints/stable_signature/hidden_replicate.pth"),
        map_location='cpu',
        weights_only=False
    )['encoder_decoder']
    
    encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
    
    decoder.load_state_dict(decoder_state_dict)
    decoder = decoder.to(device).eval()
    
    print("✅ Decoder loaded")
    return decoder


def decode_stable_sig_batch(decoder, images, device):
    """
    Decode watermarks from a batch of images using PyTorch decoder.
    Same method as detect_stable_signature.py
    """
    NORMALIZE_IMAGENET = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE_IMAGENET
    ])
    
    # Transform images
    image_tensors = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img_tensor = default_transform(img)
        image_tensors.append(img_tensor)
    
    # Batch tensors
    batch = torch.stack(image_tensors).to(device)
    
    # Decode
    with torch.no_grad():
        decoded = decoder(batch)
        decoded_msgs = (decoded > 0).cpu().numpy()  # Convert to boolean
    
    return decoded_msgs


def get_indices(path, limit, subset, subset_limit, num_images=None, key="stable_sig"):
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
    
    json_path = os.path.join(CORRECTED_RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
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
@click.option("--batch-size", type=int, default=16, help="Batch size for processing")
@click.option("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cuda:1, etc.)")
@click.option("--num-images", type=int, default=None, help="Number of actual images to decode (overrides limit for counting)")
@click.option("--key", type=str, default="stable_sig", help="Key name in output JSON")
def main(path, subset, limit, subset_limit, batch_size, device, num_images, key):
    """
    Decode Stable Signature watermarks using the corrected PyTorch decoder.

    Example:
        python decode_corrected.py --path ./data/images/mscoco/stable_sig
        python decode_corrected.py --path ./data/images/mscoco/blurring-0.1-stable_sig
    """
    # Setup device
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU (will be slow)")

    print(f"Using device: {torch_device}")
    
    # Get indices to process
    indices, all_target_indices = get_indices(path, limit, subset, subset_limit, num_images, key=key)
    
    if len(indices) == 0:
        print("Nothing to decode!")
        return
    
    print(f"Will decode {len(indices)} images")
    
    # Initialize decoder
    decoder = init_stable_sig_decoder(torch_device)
    
    # Prepare output
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    json_path = os.path.join(CORRECTED_RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
    # Load existing data or create new with proper structure for target indices
    if os.path.exists(json_path):
        results = load_json(json_path)
        # Extend results to include all target indices
        for idx in all_target_indices:
            if str(idx) not in results:
                results[str(idx)] = {key: None}
    else:
        results = {str(idx): {key: None} for idx in all_target_indices}
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Process in batches
    print("Decoding watermarks...")
    for i in tqdm(range(0, len(indices), batch_size)):
        batch_indices = indices[i:min(i + batch_size, len(indices))]
        image_paths = [os.path.join(path, f"{idx}.png") for idx in batch_indices]
        
        # Decode batch
        messages = decode_stable_sig_batch(decoder, image_paths, torch_device)
        
        # Store results
        for idx, message in zip(batch_indices, messages):
            results[str(idx)][key] = encode_array_to_string(message)
    
    # Save results
    save_json(results, json_path)
    print(f"✅ Saved results to: {json_path}")


if __name__ == "__main__":
    main()
