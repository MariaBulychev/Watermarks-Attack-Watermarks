"""
Configuration for watermark method classifier v3.
9 classes: 7 watermark methods (pixelseal, rosteals, stable_sig, stegastamp,
tree_ring, wam, zodiac, videoseal) + real (no watermark).

zodiac uses zodiac_rand/ (500 images), videoseal uses videoseal/ (500 images).
Other methods have 5000 images each.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


import os


@dataclass
class DataConfig:
    data_root: str = os.environ.get("DATA_DIR", "./data")
    # Maps class name -> subdirectory name under data_root
    # This allows zodiac class to read from zodiac_rand/
    method_dirs: Dict[str, str] = field(default_factory=lambda: {
        "pixelseal":  "pixelseal",
        "rosteals":   "rosteals",
        "stable_sig": "stable_sig",
        "stegastamp": "stegastamp",
        "tree_ring":  "tree_ring",
        "wam":        "wam",
        "zodiac":     "zodiac_rand",
        "videoseal":  "videoseal",
        "real":       "real",
    })
    max_per_class: int = 500
    image_size: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 4

    @property
    def methods(self) -> List[str]:
        return list(self.method_dirs.keys())


@dataclass
class ModelConfig:
    model_name: str = "facebook/convnextv2-large-22k-224"
    num_labels: int = 9
    cache_dir: str = os.environ.get("HF_HOME", "./cache/hf")


@dataclass
class TrainConfig:
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"

    num_epochs: int = 20
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    fp16: bool = True

    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True

    logging_steps: int = 50
    report_to: str = "tensorboard"

    seed: int = 42
    early_stopping_patience: int = 5
    label_smoothing_factor: float = 0.1


def get_label_maps(methods: List[str]):
    label2id = {m: i for i, m in enumerate(methods)}
    id2label = {i: m for i, m in enumerate(methods)}
    return label2id, id2label
