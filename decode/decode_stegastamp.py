#!/usr/bin/env python3
"""
StegaStamp decode script compatible with WAVES analysis pipeline.
Uses ONNX model for decoding stegastamp watermarks.

Environment: stegastamp_tf115 conda environment
Expected secret: "Stega!!"

Usage:
    python decode_stegastamp.py --path <image_dir> --num-images 500
"""

import os
import sys
import argparse
import json
import base64
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
import onnxruntime as ort
import bchlib

warnings.filterwarnings("ignore")

# Constants
LIMIT = 5000
SUBSET_LIMIT = 100
RESULT_DIR = os.environ.get("RESULTS_DIR", "./results")

# StegaStamp settings
STEGASTAMP_MODEL_PATH = os.path.join(
    os.environ.get("WAVES_PATH", "./external/WAVES"), "decoders/stega_stamp.onnx"
)
EXPECTED_SECRET = "Stega!!"
BCH_POLYNOMIAL = 137
BCH_BITS = 5


# Helper functions (replaces dev module)
def encode_array_to_string(array):
    """Encode numpy boolean array to base64 string."""
    return base64.b64encode(np.packbits(array)).decode('utf-8')


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def check_file_existence(path, name_pattern="{}.png", limit=5000):
    """Check which image files exist."""
    existences = []
    for i in range(limit):
        filepath = os.path.join(path, name_pattern.format(i))
        existences.append(os.path.exists(filepath))
    return existences


def existence_to_indices(existences, limit=None):
    """Convert existence list to list of indices where files exist."""
    indices = [i for i, exists in enumerate(existences) if exists]
    if limit:
        indices = indices[:limit]
    return indices


def init_stegastamp_decoder():
    """Initialize the ONNX StegaStamp decoder and BCH codec."""
    print("Loading StegaStamp ONNX decoder...")
    
    # Setup ONNX session
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 4
    session_options.log_severity_level = 3
    
    sess = ort.InferenceSession(
        STEGASTAMP_MODEL_PATH,
        providers=["CPUExecutionProvider"],
        sess_options=session_options,
    )
    
    # Setup BCH codec
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    
    print("✅ StegaStamp decoder loaded")
    return sess, bch


def decode_stegastamp_single(image_path, sess, bch):
    """
    Decode StegaStamp watermark from a single image.
    
    Returns:
        tuple: (raw_bits as numpy array, decoded_secret, success)
    """
    try:
        # Load and preprocess image (resize to 400x400)
        # Convert to RGB to handle RGBA images (e.g., from zodiac forgery)
        img = Image.open(image_path).convert('RGB')
        
        # Use resize() instead of ImageOps.fit() to avoid cropping
        # BILINEAR is often used in neural network preprocessing and preserves 
        # watermark information better than BICUBIC which can introduce ringing artifacts
        if img.size != (400, 400):
            img = img.resize((400, 400), Image.BILINEAR)
        
        image = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_batch = image.reshape(1, 400, 400, 3)
        
        # Run inference
        outputs = sess.run(
            None,
            {
                "image": image_batch,
                "secret": np.zeros((1, 100), dtype=np.float32),
            },
        )
        
        # Get decoded bits from outputs[2] - these are the raw decoded values
        decoded_bits = outputs[2][0]
        
        # Convert first 96 bits to binary for BCH decoding
        message_bits = (decoded_bits[:96] > 0.5).astype(np.uint8)
        
        # BCH decode to get the secret
        packed_bits = np.packbits(message_bits)
        packet = bytearray(packed_bits)
        
        # Split packet into data and ECC parts
        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        
        bitflips = bch.decode_inplace(data, ecc)
        
        decoded_secret = None
        success = False
        
        if bitflips >= 0:
            try:
                decoded_secret = data.decode('utf-8').strip()
                success = (decoded_secret == EXPECTED_SECRET)
            except UnicodeDecodeError:
                pass
        
        # Return the raw decoded bits (100 bits) as boolean array for storage
        raw_bits_bool = (decoded_bits > 0.5).astype(bool)
        
        return raw_bits_bool, decoded_secret, success
        
    except Exception as e:
        print(f"Error decoding {image_path}: {e}")
        return None, None, False


def decode_stegastamp_batch(image_paths, sess, bch):
    """
    Decode StegaStamp watermarks from a batch of images.
    Note: ONNX model processes one image at a time, so we loop.
    
    Returns:
        list of numpy arrays (raw bits for each image)
    """
    results = []
    for img_path in image_paths:
        raw_bits, _, _ = decode_stegastamp_single(img_path, sess, bch)
        results.append(raw_bits)
    return results


def get_indices(path, limit, subset, subset_limit, num_images=None, force=False, key="stegastamp"):
    """Get indices of images to decode.
    
    Args:
        path: Directory containing images
        limit: Maximum index to search (e.g., 5000 means check indices 0-4999)
        subset: Whether to use subset mode
        subset_limit: Limit for subset mode
        num_images: Number of actual images to decode (if None, use limit/subset_limit)
        force: If True, decode all images even if already decoded
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
    
    # If force mode, decode all target images regardless of existing data
    if force:
        indices = all_existing_indices[:target_count]
        print(f"Force mode: will decode {len(indices)} images (ignoring existing data)")
        return indices, all_existing_indices[:target_count]
    
    # Check if already decoded
    if os.path.exists(json_path):
        data = load_json(json_path)
        if data is not None:
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


def main():
    """
    Decode StegaStamp watermarks using ONNX decoder.
    
    Environment: stegastamp_tf115 conda environment
    Expected secret: "Stega!!"
    """
    parser = argparse.ArgumentParser(description='Decode StegaStamp watermarks')
    parser.add_argument('--path', type=str, required=True, help='Path to image directory')
    parser.add_argument('--subset', action='store_true', help='Only process subset of images')
    parser.add_argument('--limit', type=int, default=LIMIT, help='Maximum index to search (searches 0 to limit-1)')
    parser.add_argument('--subset-limit', type=int, default=SUBSET_LIMIT, help='Subset limit')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--num-images', type=int, default=None, help='Number of actual images to decode (overrides limit for counting)')
    parser.add_argument('--force', action='store_true', help='Force decode even if already decoded')
    parser.add_argument('--key', type=str, default='stegastamp', help='Key name in output JSON')
    
    args = parser.parse_args()
    
    print("StegaStamp Decoder")
    print("Expected secret: {}".format(EXPECTED_SECRET))
    print("Model: {}".format(STEGASTAMP_MODEL_PATH))
    print()
    
    # Get indices to process
    indices, all_target_indices = get_indices(args.path, args.limit, args.subset, args.subset_limit, args.num_images, args.force, key=args.key)
    
    if len(indices) == 0:
        print("Nothing to decode!")
        return
    
    print("Will decode {} images".format(len(indices)))
    
    # Initialize decoder
    sess, bch = init_stegastamp_decoder()
    
    # Prepare output
    dataset_name = str(args.path).split("/")[-2]
    filename = str(args.path).split("/")[-1]
    json_path = os.path.join(RESULT_DIR, dataset_name, "{}-decode.json".format(filename))
    
    key = args.key
    
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
    
    # Track success rate
    successful_decodes = 0
    matched_secrets = 0
    
    # Process in batches
    print("Decoding watermarks...")
    for i in tqdm(range(0, len(indices), args.batch_size)):
        batch_indices = indices[i:min(i + args.batch_size, len(indices))]
        image_paths = [os.path.join(args.path, "{}.png".format(idx)) for idx in batch_indices]
        
        # Decode batch
        for idx, img_path in zip(batch_indices, image_paths):
            raw_bits, decoded_secret, success = decode_stegastamp_single(img_path, sess, bch)
            
            if raw_bits is not None:
                results[str(idx)][key] = encode_array_to_string(raw_bits)
                successful_decodes += 1
                if success:
                    matched_secrets += 1
            else:
                results[str(idx)][key] = None
    
    # Save results
    save_json(results, json_path)
    print("✅ Saved results to: {}".format(json_path))
    print()
    print("Summary:")
    print("  Total processed: {}".format(len(indices)))
    print("  Successful decodes: {}".format(successful_decodes))
    print("  Matched expected secret '{}': {}".format(EXPECTED_SECRET, matched_secrets))


if __name__ == "__main__":
    main()
