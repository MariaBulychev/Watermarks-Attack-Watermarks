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
    # Optional per-class override of `max_per_class`. Any class not present
    # in this dict falls back to `max_per_class`. Used to address stage-1
    # imbalance (real is capped much lower than what's actually available on
    # disk). Empty by default => existing behaviour is preserved.
    per_class_max: Dict[str, int] = field(default_factory=dict)
    image_size: int = 512
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 4

    @property
    def methods(self) -> List[str]:
        return list(self.method_dirs.keys())

    def cap_for(self, method: str) -> int:
        return self.per_class_max.get(method, self.max_per_class)


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


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage classifier configuration
# Stage 1 : binary  watermarked (1)  vs  not watermarked / real (0)
# Stage 2 : 8-class over the known watermarking schemes (real excluded)
# ─────────────────────────────────────────────────────────────────────────────

REAL_CLASS_NAME: str = "real"


@dataclass
class TwoStageModelConfig:
    model_name: str = "facebook/convnextv2-large-22k-224"
    num_stage1_labels: int = 2                  # not-watermarked / watermarked
    num_stage2_labels: int = 8                  # 8 known schemes
    head_dropout: float = 0.1
    cache_dir: str = os.environ.get("HF_HOME", "./cache/hf")


@dataclass
class TwoStageTrainConfig:
    output_dir: str = "./outputs_2stage"
    logging_dir: str = "./logs_2stage"

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
    metric_for_best_model: str = "overall_accuracy"
    greater_is_better: bool = True

    logging_steps: int = 50
    report_to: str = "tensorboard"

    seed: int = 42
    early_stopping_patience: int = 5
    label_smoothing_factor: float = 0.1         # applied manually in each head loss

    # Inference-time only: max stage-2 logit below tau => flagged as unknown.
    # Default -inf => nothing ever flagged unknown (sanity-check mode).
    tau: float = float("-inf")


def get_two_stage_label_maps(methods: List[str]) -> Tuple[
    Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]
]:
    """
    Build label maps for the 2-stage classifier.

    Returns:
        stage1_label2id, stage1_id2label : {"not_watermarked": 0, "watermarked": 1}
        stage2_label2id, stage2_id2label : 0..7 over the 8 known schemes,
                                            in the order they appear in `methods`
                                            (with "real" removed).
    """
    stage1_label2id = {"not_watermarked": 0, "watermarked": 1}
    stage1_id2label = {v: k for k, v in stage1_label2id.items()}

    scheme_methods = [m for m in methods if m != REAL_CLASS_NAME]
    stage2_label2id = {m: i for i, m in enumerate(scheme_methods)}
    stage2_id2label = {i: m for m, i in stage2_label2id.items()}

    return stage1_label2id, stage1_id2label, stage2_label2id, stage2_id2label
