"""
Dataset for watermark classifier v3 (9 classes).

Supports mapping class names to different subdirectory names via config.method_dirs.
This allows e.g. zodiac class to read from zodiac_rand/ directory.
"""

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter

from config import DataConfig, get_label_maps


class WatermarkDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            inputs = self.transform(image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        else:
            pixel_values = image
        return {"pixel_values": pixel_values, "labels": self.labels[idx]}


def gather_image_paths_and_labels(data_config: DataConfig) -> Tuple[List[str], List[int]]:
    label2id, _ = get_label_maps(data_config.methods)
    all_paths, all_labels = [], []

    cap = data_config.max_per_class

    for method, dirname in data_config.method_dirs.items():
        method_dir = os.path.join(data_config.data_root, dirname)
        if not os.path.isdir(method_dir):
            raise FileNotFoundError(f"Directory not found for class '{method}': {method_dir}")
        label = label2id[method]
        fnames = sorted(f for f in os.listdir(method_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        total = len(fnames)
        if cap is not None and total > cap:
            fnames = fnames[:cap]
        for fname in fnames:
            all_paths.append(os.path.join(method_dir, fname))
            all_labels.append(label)
        print(f"  {method} ({dirname}): {len(fnames)}/{total} images")

    return all_paths, all_labels


def create_splits(data_config: DataConfig) -> Dict[str, Tuple[List[str], List[int]]]:
    print("Gathering images...")
    all_paths, all_labels = gather_image_paths_and_labels(data_config)

    val_test_ratio = data_config.val_ratio + data_config.test_ratio
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        all_paths, all_labels,
        test_size=val_test_ratio, random_state=data_config.seed, stratify=all_labels,
    )

    relative_test_ratio = data_config.test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=relative_test_ratio, random_state=data_config.seed, stratify=valtest_labels,
    )

    _, id2label = get_label_maps(data_config.methods)
    print(f"\nDataset splits — Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        counts = Counter(labels)
        counts_str = ", ".join(f"{id2label[k]}: {v}" for k, v in sorted(counts.items()))
        print(f"  {split_name}: {counts_str}")

    return {
        "train": (train_paths, train_labels),
        "val":   (val_paths,   val_labels),
        "test":  (test_paths,  test_labels),
    }


if __name__ == "__main__":
    cfg = DataConfig()
    create_splits(cfg)
