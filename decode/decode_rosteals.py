#!/usr/bin/env python3
"""
RoSteALS watermark decode script for WAVES analysis pipeline.
Decodes watermarks from images and computes bit accuracy.
Generates results compatible with WAVES analysis pipeline.

Detection Process:
1. Load image
2. Pass through RoSteALS decoder network
3. Extract predicted secret bits
4. Compute bit accuracy against expected secret

Usage:
    python decode_rosteals.py --path <image_dir> --num-images 500 --device cuda:0
"""

import os
import sys
import click
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import warnings

# Add RoSteALS to path
ROSTEALS_PATH = os.environ.get("ROSTEALS_PATH", "./external/RoSteALS")
sys.path.append(ROSTEALS_PATH)

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

# Import ECC directly
sys.path.insert(0, os.path.join(ROSTEALS_PATH, 'tools'))
from ecc import ECC

# Define constants directly to avoid importing from dev package (which has plotly dependency)
LIMIT = 5000
SUBSET_LIMIT = 500

# Use corrected result directory
CORRECTED_RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")


def check_file_existence(path, name_pattern, limit):
    """Check which files exist in a directory"""
    found_filenames = set(os.listdir(path))
    return [name_pattern.format(i) in found_filenames for i in range(limit)]


def existence_to_indices(existences, limit):
    """Convert existence list to list of indices"""
    indices = []
    for i in range(min(len(existences), limit)):
        if existences[i]:
            indices.append(i)
    return indices


def save_json(data, path):
    """Save data to JSON file"""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Load data from JSON file"""
    import json
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def encode_bits_to_string(bits):
    """Encode binary bits array to base64 string for storage."""
    import base64
    # Convert to numpy array of uint8 (0 or 1)
    bits_array = np.array(bits, dtype=np.uint8)
    # Pack bits into bytes
    packed = np.packbits(bits_array)
    # Encode to base64
    return base64.b64encode(packed).decode('utf-8')


warnings.filterwarnings("ignore")

# Expected secret for RoSteALS watermarked images
EXPECTED_SECRET = "RoSteAL"


def load_rosteals_model(config_path, weight_path, device):
    """Load the RoSteALS model"""
    print(f"Loading RoSteALS model from {weight_path}")
    
    config = OmegaConf.load(config_path).model
    secret_len = config.params.control_config.params.secret_len
    config.params.decoder_config.params.secret_len = secret_len
    
    model = instantiate_from_config(config)
    state_dict = torch.load(weight_path, map_location=device)
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def decode_single_image(model, device, image_path, ecc, image_size=256):
    """Decode secret from a single image"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Transform for model input
        tform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        img_tensor = tform(img).unsqueeze(0).to(device)
        
        # Decode secret
        with torch.no_grad():
            secret_pred = (model.decoder(img_tensor) > 0).cpu().numpy()  # 1, 100
        
        # Compute bit accuracy against expected secret
        expected_bits = ecc.encode_text([EXPECTED_SECRET])
        bit_accuracy = float(np.mean(secret_pred == expected_bits))
        
        # Decode text
        decoded_text = ecc.decode_text(secret_pred)[0]
        
        return {
            "bit_accuracy": bit_accuracy,
            "decoded_text": decoded_text,
            "match": decoded_text == EXPECTED_SECRET
        }
    except Exception as e:
        return {
            "bit_accuracy": 0.0,
            "decoded_text": "",
            "match": False,
            "error": str(e)
        }


def parse_path(path):
    """Parse image directory path to extract dataset info"""
    parts = str(path).split("/")
    
    # Check if attacked or main
    if "attacked" in parts:
        idx = parts.index("attacked")
        dataset = parts[idx + 1]
        dirname = parts[idx + 2]
        attack_parts = dirname.split("-")
        if len(attack_parts) == 3:
            attack_name, attack_strength, source_name = attack_parts
            return dataset, attack_name, attack_strength, source_name
    elif "main" in parts:
        idx = parts.index("main")
        dataset = parts[idx + 1]
        source_name = parts[idx + 2]
        return dataset, None, None, source_name
    
    return None, None, None, None


def init_rosteals_detector(device):
    """
    Initialize the RoSteALS watermark detector.
    Loads the model and ECC decoder.
    
    Returns:
        model: RoSteALS model
        ecc: ECC decoder for text recovery
    """
    print("Loading RoSteALS watermark detector...")
    
    config_path = os.path.join(ROSTEALS_PATH, "models", "VQ4_mir_inference.yaml")
    weight_path = os.path.join(ROSTEALS_PATH, "models", "RoSteALS", "epoch=000017-step=000449999.ckpt")
    
    model = load_rosteals_model(config_path, weight_path, device)
    ecc = ECC()
    
    print("✅ RoSteALS detector loaded")
    
    return model, ecc


def detect_watermark_single(image_path, model, ecc, device, image_size=256):
    """
    Detect RoSteALS watermark in a single image.
    
    Args:
        image_path: Path to image file
        model: RoSteALS model
        ecc: ECC decoder
        device: torch device
        image_size: Model input size
    
    Returns:
        decoded_bits: numpy array of decoded bits (100 bits)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Transform for model input
    tform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    img_tensor = tform(img).unsqueeze(0).to(device)
    
    # Decode secret
    with torch.no_grad():
        secret_pred = (model.decoder(img_tensor) > 0).cpu().numpy()  # 1, 100
    
    # Return the raw decoded bits (100 bits)
    return secret_pred[0].astype(bool)


def get_indices(path, limit, subset, subset_limit, num_images=None, output_json=None, key="rosteals"):
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
@click.option("--key", type=str, default="rosteals", help="Key name in output JSON")
def main(path, subset, limit, subset_limit, device, num_images, output_json, key):
    """
    Decode RoSteALS watermarks.
    
    Example:
        python decode_rosteals.py --path ./data/rosteals
        python decode_rosteals.py --path ./data/attacked/mscoco/blurring-0.1-rosteals
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
    model, ecc = init_rosteals_detector(torch_device)
    
    # Prepare output
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    if output_json:
        json_path = output_json
    else:
        json_path = os.path.join(CORRECTED_RESULT_DIR, dataset_name, f"{filename}-decode.json")
    
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
    print("Decoding RoSteALS watermarks...")
    for idx in tqdm(indices, desc="Processing images"):
        image_path = os.path.join(path, f"{idx}.png")
        
        try:
            decoded_bits = detect_watermark_single(
                image_path, model, ecc, torch_device
            )
            # Store decoded bits as base64 encoded string (WAVES framework computes BER later)
            results[str(idx)][key] = encode_bits_to_string(decoded_bits)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            results[str(idx)][key] = None
    
    # Save results
    save_json(results, json_path)
    print(f"✅ Saved results to: {json_path}")


if __name__ == "__main__":
    main()
