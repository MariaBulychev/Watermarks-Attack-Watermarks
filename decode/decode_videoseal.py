#!/usr/bin/env python3
"""
VideoSeal watermark decode script for WAVES analysis pipeline.
Uses the VideoSeal model to detect watermarks.
Generates results compatible with WAVES analysis pipeline.

Detection Process:
1. Load image
2. Run through VideoSeal detector
3. Extract bit predictions
4. Compute bit accuracy against ground truth message
5. Store raw bit predictions for later analysis

Usage:
    python decode_videoseal.py --path <image_dir> --num-images 500 --device cuda:0
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

# VideoSeal uses 256 bits, ground truth message: torch.manual_seed(42 + image_index)
VIDEOSEAL_NBITS = 256
VIDEOSEAL_MESSAGE_SEED = 42


def encode_bits_to_string(bits):
    """Encode binary bits array to base64 string for storage."""
    bits_array = np.array(bits, dtype=np.uint8)
    packed = np.packbits(bits_array)
    return base64.b64encode(packed).decode('utf-8')


def init_videoseal_detector(device):
    """Initialize the VideoSeal watermark detector."""
    print("Loading VideoSeal watermark detector...")

    original_dir = os.getcwd()
    os.chdir(VIDEOSEAL_PATH)

    import videoseal

    model = videoseal.load("videoseal")
    model = model.to(device)
    model.eval()

    nbits = model.embedder.msg_processor.nbits

    os.chdir(original_dir)

    print(f"VideoSeal detector loaded")
    print(f"   Number of bits: {nbits}")

    return model, nbits


def detect_watermark_single(image_path, model, device):
    """
    Detect VideoSeal watermark in a single image.

    Returns:
        bits: List of detected bits (0 or 1)
    """
    image = Image.open(image_path).convert('RGB')

    to_tensor = T.ToTensor()
    img_tensor = to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        detection = model.detect(img_tensor, is_video=False)

    # detection["preds"]: [B, 1+nbits] — first is mask/presence, rest are bit logits
    bit_preds = detection["preds"][:, 1:]
    detected_bits = (bit_preds[0] > 0).cpu().numpy().astype(int).tolist()

    return detected_bits


def get_indices(path, limit, subset, subset_limit, num_images=None, key="videoseal"):
    """Get indices of images to decode."""
    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]

    json_path = os.path.join(RESULT_DIR, dataset_name, f"{filename}-decode.json")

    if num_images is not None:
        target_count = num_images
    elif subset:
        target_count = subset_limit
    else:
        target_count = limit

    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    all_existing_indices = existence_to_indices(image_existences, limit=limit)
    print(f"Found {len(all_existing_indices)} images in {path} (searched indices 0-{limit-1})")

    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        decoded_count = 0
        for idx in all_existing_indices[:target_count]:
            if data.get(str(idx), {}).get(key) is not None:
                decoded_count += 1

        if decoded_count >= target_count:
            print(f"Already fully decoded {target_count} images: {json_path}")
            return [], all_existing_indices[:target_count]

    if not os.path.exists(json_path):
        indices = all_existing_indices[:target_count]
    else:
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
@click.option("--device", type=str, default="cuda:0", help="Device to use")
@click.option("--batch-size", type=int, default=1, help="Batch size for processing")
@click.option("--num-images", type=int, default=None, help="Number of actual images to decode (overrides limit)")
@click.option("--key", type=str, default="videoseal", help="Key name in output JSON")
def main(path, subset, limit, subset_limit, device, batch_size, num_images, key):
    """
    Decode VideoSeal watermarks from images.

    Example:
        python decode_videoseal.py --path /data/.../data/main/mscoco/videoseal
        python decode_videoseal.py --path /data/.../data/attacked/mscoco/videoseal_forgery-1.0-stable_sig
    """
    print("VideoSeal Decoder")
    print(f"Path: {path}")
    print("")

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        print("CUDA not available, using CPU (will be slower)")

    print(f"Using device: {torch_device}")

    indices, all_target_indices = get_indices(path, limit, subset, subset_limit, num_images, key=key)

    if len(indices) == 0:
        print("Nothing to decode!")
        return

    print(f"Will decode {len(indices)} images")

    model, nbits = init_videoseal_detector(torch_device)

    dataset_name = str(path).split("/")[-2]
    filename = str(path).split("/")[-1]
    json_path = os.path.join(RESULT_DIR, dataset_name, f"{filename}-decode.json")

    if os.path.exists(json_path):
        results = load_json(json_path)
        for idx in all_target_indices:
            if str(idx) not in results:
                results[str(idx)] = {key: None}
            elif key not in results[str(idx)]:
                results[str(idx)][key] = None
    else:
        results = {str(idx): {key: None} for idx in all_target_indices}

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    print("Decoding VideoSeal watermarks...")
    for idx in tqdm(indices, desc="Processing images"):
        image_path = os.path.join(path, f"{idx}.png")

        try:
            bits = detect_watermark_single(image_path, model, torch_device)
            results[str(idx)][key] = encode_bits_to_string(bits)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            results[str(idx)][key] = None

    save_json(results, json_path)
    print(f"Saved results to: {json_path}")


if __name__ == "__main__":
    main()
