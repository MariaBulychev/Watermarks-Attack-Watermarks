"""
Dataset for the two-stage watermark classifier.

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

from config import DataConfig, REAL_CLASS_NAME, get_two_stage_label_maps


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage variants
# Each item carries two labels:
#   stage1_labels : 0/1  (not-watermarked / watermarked)  — always present
#   stage2_labels : 0..7 for the known scheme, or -100 for real
#                   (-100 is ignored by nn.CrossEntropyLoss)
# ─────────────────────────────────────────────────────────────────────────────

STAGE2_IGNORE_INDEX: int = -100


class TwoStageWatermarkDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 stage1_labels: List[int],
                 stage2_labels: List[int],
                 transform=None,
                 augment=None):
        """
        Args:
            transform: HF image processor (e.g. ConvNextImageProcessor). Applied
                after `augment`. Handles resize + centre-crop + normalise.
            augment: optional callable PIL.Image -> PIL.Image (typically a
                torchvision Compose). Applied BEFORE `transform`. Only pass
                this for the training split — leave None for val/test.
        """
        assert len(image_paths) == len(stage1_labels) == len(stage2_labels)
        self.image_paths = image_paths
        self.stage1_labels = stage1_labels
        self.stage2_labels = stage2_labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        if self.transform is not None:
            inputs = self.transform(image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        else:
            pixel_values = image
        return {
            "pixel_values":  pixel_values,
            "stage1_labels": self.stage1_labels[idx],
            "stage2_labels": self.stage2_labels[idx],
        }


def _gather_two_stage(data_config: DataConfig) -> Tuple[List[str], List[int], List[int]]:
    """
    Walk the same directory layout as the 9-class version, but emit two labels
    per sample.
    """
    stage1_l2i, _, stage2_l2i, _ = get_two_stage_label_maps(data_config.methods)
    all_paths, s1_labels, s2_labels = [], [], []

    for method, dirname in data_config.method_dirs.items():
        method_dir = os.path.join(data_config.data_root, dirname)
        if not os.path.isdir(method_dir):
            raise FileNotFoundError(f"Directory not found for class '{method}': {method_dir}")

        is_real = (method == REAL_CLASS_NAME)
        s1_label = stage1_l2i["not_watermarked"] if is_real else stage1_l2i["watermarked"]
        s2_label = STAGE2_IGNORE_INDEX if is_real else stage2_l2i[method]

        cap = data_config.cap_for(method)
        fnames = sorted(f for f in os.listdir(method_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        total = len(fnames)
        if cap is not None and total > cap:
            fnames = fnames[:cap]
        for fname in fnames:
            all_paths.append(os.path.join(method_dir, fname))
            s1_labels.append(s1_label)
            s2_labels.append(s2_label)
        print(f"  {method} ({dirname}): {len(fnames)}/{total} images "
              f"[stage1={s1_label}, stage2={s2_label}, cap={cap}]")

    return all_paths, s1_labels, s2_labels


def create_two_stage_splits(data_config: DataConfig) -> Dict[
    str, Tuple[List[str], List[int], List[int]]
]:
    print("Gathering images (two-stage) ...")
    paths, s1, s2 = _gather_two_stage(data_config)

    # Stratify on the 9-class equivalent = stage2 if watermarked, else "real"
    stratify_key = [
        s2_l if s2_l != STAGE2_IGNORE_INDEX else -1
        for s2_l in s2
    ]

    val_test_ratio = data_config.val_ratio + data_config.test_ratio
    train_idx, valtest_idx = train_test_split(
        list(range(len(paths))),
        test_size=val_test_ratio,
        random_state=data_config.seed,
        stratify=stratify_key,
    )
    valtest_strat = [stratify_key[i] for i in valtest_idx]
    rel_test = data_config.test_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=rel_test,
        random_state=data_config.seed,
        stratify=valtest_strat,
    )

    def _slice(indices):
        return (
            [paths[i] for i in indices],
            [s1[i]    for i in indices],
            [s2[i]    for i in indices],
        )

    splits = {
        "train": _slice(train_idx),
        "val":   _slice(val_idx),
        "test":  _slice(test_idx),
    }

    _, _, stage2_l2i, stage2_i2l = get_two_stage_label_maps(data_config.methods)
    print(f"\nDataset splits — Train: {len(splits['train'][0])}, "
          f"Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")
    for name in ("train", "val", "test"):
        s1_list = splits[name][1]
        s2_list = splits[name][2]
        n_real = sum(1 for x in s1_list if x == 0)
        n_wm   = sum(1 for x in s1_list if x == 1)
        per_scheme = Counter(x for x in s2_list if x != STAGE2_IGNORE_INDEX)
        per_scheme_str = ", ".join(
            f"{stage2_i2l[k]}: {v}" for k, v in sorted(per_scheme.items())
        )
        print(f"  {name:<5}  real={n_real}  watermarked={n_wm}  ({per_scheme_str})")

    return splits


if __name__ == "__main__":
    cfg = DataConfig()
    create_two_stage_splits(cfg)
