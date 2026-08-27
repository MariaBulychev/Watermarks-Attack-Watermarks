"""
Training script for the 2-stage watermark classifier.

Architecture:
    - Shared backbone : ConvNeXt-V2 Large (loaded via ConvNextV2Model, no
      classification head from HF; pretrained weights only).
    - Stage-1 head    : Linear(hidden -> 2)  — binary  watermarked / not
    - Stage-2 head    : Linear(hidden -> 8)  — one logit per known scheme
    - Joint loss      : L = CE_stage1 + CE_stage2
                        Stage-2 CE ignores samples with label = -100 (real).

The two heads are jointly trained *from scratch* (heads randomly initialised);
we do NOT load weights from any existing finetuned 9-class checkpoint.

Unknown-scheme detection (inference-time only):
    is_unknown = max(stage2_logits) < tau
tau defaults to -inf (nothing ever flagged) so the pipeline behaves like a
standard closed-set 8-scheme classifier for sanity checking.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    ConvNextV2Model,
    ConvNextImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import ModelOutput
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torchvision import transforms as T

from config import (
    DataConfig,
    TwoStageModelConfig,
    TwoStageTrainConfig,
    get_two_stage_label_maps,
)
from dataset import (
    TwoStageWatermarkDataset,
    create_two_stage_splits,
    STAGE2_IGNORE_INDEX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TwoStageOutput(ModelOutput):
    # NOTE: only `loss` and the two logit tensors are exposed here.
    # HF Trainer collects every non-`loss` field of the model output as a
    # prediction and stacks it — adding auxiliary sub-loss fields here would
    # make `eval_pred.predictions` a 4-tuple and break compute_metrics.
    loss: Optional[torch.FloatTensor] = None
    stage1_logits: Optional[torch.FloatTensor] = None
    stage2_logits: Optional[torch.FloatTensor] = None


class TwoStageWatermarkClassifier(nn.Module):
    """
    Shared ConvNeXt-V2 backbone with two independent linear heads.

    Loss:
        L = CE(stage1_logits, stage1_labels)
          + CE(stage2_logits, stage2_labels)   # ignore_index = -100 for real
    """

    def __init__(
        self,
        model_name: str,
        num_stage1_labels: int = 2,
        num_stage2_labels: int = 8,
        head_dropout: float = 0.1,
        label_smoothing: float = 0.0,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        # Backbone: pretrained weights, no HF classification head.
        self.backbone = ConvNextV2Model.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        hidden = self.backbone.config.hidden_sizes[-1]

        self.dropout = nn.Dropout(head_dropout)
        self.stage1_head = nn.Linear(hidden, num_stage1_labels)
        self.stage2_head = nn.Linear(hidden, num_stage2_labels)

        # Randomly initialise the two heads (from-scratch heads).
        for head in (self.stage1_head, self.stage2_head):
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

        self.loss_stage1_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # ignore_index = -100 is the CE default; being explicit for clarity.
        self.loss_stage2_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=STAGE2_IGNORE_INDEX
        )

        # Sizes exposed so training code / metric fns can use them.
        self.num_stage1_labels = num_stage1_labels
        self.num_stage2_labels = num_stage2_labels

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        stage1_labels: Optional[torch.LongTensor] = None,
        stage2_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> TwoStageOutput:
        outputs = self.backbone(pixel_values=pixel_values)
        pooled = outputs.pooler_output               # (B, hidden)
        pooled = self.dropout(pooled)

        stage1_logits = self.stage1_head(pooled)     # (B, 2)
        stage2_logits = self.stage2_head(pooled)     # (B, 8)

        loss = loss_s1 = loss_s2 = None
        if stage1_labels is not None:
            loss_s1 = self.loss_stage1_fn(stage1_logits, stage1_labels)
        if stage2_labels is not None:
            # If every sample in a batch is real (-100), CE returns NaN. Guard.
            valid = (stage2_labels != STAGE2_IGNORE_INDEX)
            if valid.any():
                loss_s2 = self.loss_stage2_fn(stage2_logits, stage2_labels)
            else:
                loss_s2 = stage2_logits.new_zeros(())
        if loss_s1 is not None and loss_s2 is not None:
            loss = loss_s1 + loss_s2

        return TwoStageOutput(
            loss=loss,
            stage1_logits=stage1_logits,
            stage2_logits=stage2_logits,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unknown-scheme detection helper (inference-time only)
# ─────────────────────────────────────────────────────────────────────────────
def detect_unknown(stage2_logits: np.ndarray, tau: float) -> np.ndarray:
    """
    Given raw stage-2 logits (NOT softmax probabilities) of shape (N, 8),
    return a boolean mask of length N where True => flagged as unknown.

    is_unknown = max(stage2_logits) < tau

    tau = -inf  =>  never flags anything as unknown (sanity check).
    """
    return stage2_logits.max(axis=-1) < tau


# ─────────────────────────────────────────────────────────────────────────────
# Training-time augmentation (applied ONLY to train_dataset)
# ─────────────────────────────────────────────────────────────────────────────
def build_train_augmentation() -> T.Compose:
    """
    Conservative PIL -> PIL augmentation pipeline. Runs BEFORE the HF image
    processor, so the processor still handles resize + centre-crop + normalise.

    Choices are deliberately watermark-safe for a first augmentation pass:
      - RandomHorizontalFlip / RandomVerticalFlip:  symmetries most schemes tolerate.
      - ColorJitter (mild):                          benign photo-style
                                                     perturbations.
      - Occasional light GaussianBlur:               small denoising-like noise.
    We intentionally SKIP JPEG / heavy crops / rotations in this pass — those
    directly overlap with our downstream attack evaluations (waves / regen /
    diffusion attacks) and would leak test-time attacks into training.
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.3,
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def make_compute_metrics(tau: float, num_scheme_classes: int):
    """
    Trainer passes predictions as (stage1_logits, stage2_logits) and label_ids
    as (stage1_labels, stage2_labels), because model output has two logit
    fields and TrainingArguments.label_names contains two entries.
    """

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        stage1_logits, stage2_logits = predictions
        stage1_labels, stage2_labels = label_ids

        s1_preds = np.argmax(stage1_logits, axis=-1)
        s2_preds = np.argmax(stage2_logits, axis=-1)

        # Stage-1 metrics on every example.
        stage1_acc = accuracy_score(stage1_labels, s1_preds)
        stage1_f1  = f1_score(stage1_labels, s1_preds, average="binary")

        # Stage-2 metrics only over truly watermarked examples.
        wm_mask = stage2_labels != STAGE2_IGNORE_INDEX
        if wm_mask.any():
            stage2_acc = accuracy_score(stage2_labels[wm_mask], s2_preds[wm_mask])
            stage2_f1  = f1_score(
                stage2_labels[wm_mask], s2_preds[wm_mask],
                average="macro", labels=list(range(num_scheme_classes)),
                zero_division=0,
            )
        else:
            stage2_acc = 0.0
            stage2_f1  = 0.0

        # Combined 9-way decision (matches the single-head 9-class classifier).
        # If stage-1 predicts 'not_watermarked' -> class = 'real'
        # Else -> class = stage-2 argmax (optionally overridden as 'unknown').
        unknown_mask = detect_unknown(stage2_logits, tau)

        # Map ground truth into a 9-way label space: 0..7 are schemes, 8 is real.
        # (unknown class is inference-only; not in ground truth.)
        REAL_ID = num_scheme_classes
        gt_9way = np.where(wm_mask, stage2_labels, REAL_ID)

        pred_9way = np.where(
            s1_preds == 0,        # predicted not-watermarked
            REAL_ID,
            s2_preds,
        )
        # unknown-flagged samples get their own class (num_scheme_classes + 1);
        # they will not match any of the 9 real classes, which is the intended
        # behaviour: gated by tau at inference time only.
        UNKNOWN_ID = num_scheme_classes + 1
        pred_9way = np.where(unknown_mask & (s1_preds == 1), UNKNOWN_ID, pred_9way)

        overall_acc = float(np.mean(pred_9way == gt_9way))

        return {
            "stage1_accuracy": float(stage1_acc),
            "stage1_f1":       float(stage1_f1),
            "stage2_accuracy": float(stage2_acc),
            "stage2_f1_macro": float(stage2_f1),
            "overall_accuracy": overall_acc,
            "unknown_flag_rate": float(unknown_mask.mean()),
        }

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    data_cfg  = DataConfig()
    model_cfg = TwoStageModelConfig()
    train_cfg = TwoStageTrainConfig()

    if args.epochs     is not None: train_cfg.num_epochs = args.epochs
    if args.batch_size is not None: train_cfg.per_device_train_batch_size = args.batch_size
    if args.lr         is not None: train_cfg.learning_rate = args.lr
    if args.image_size is not None: data_cfg.image_size = args.image_size
    if args.output_dir is not None: train_cfg.output_dir = args.output_dir
    if args.tau        is not None: train_cfg.tau = args.tau
    if args.real_max   is not None:
        data_cfg.per_class_max = {**data_cfg.per_class_max, "real": args.real_max}
        print(f"[data] Overriding real-class cap: {args.real_max}")

    stage1_l2i, stage1_i2l, stage2_l2i, stage2_i2l = get_two_stage_label_maps(
        data_cfg.methods
    )
    scheme_names = [stage2_i2l[i] for i in range(len(stage2_i2l))]
    print(f"Stage-1 labels: {stage1_l2i}")
    print(f"Stage-2 labels: {stage2_l2i}")

    # ── Image processor ──────────────────────────────────────────────────
    image_processor = ConvNextImageProcessor.from_pretrained(
        model_cfg.model_name,
        cache_dir=model_cfg.cache_dir,
        size={"shortest_edge": data_cfg.image_size},
        crop_size={"height": data_cfg.image_size, "width": data_cfg.image_size},
    )
    print(f"Image processor size: {image_processor.size}")
    print(f"Image processor crop_size: {image_processor.crop_size}")

    # ── Data ─────────────────────────────────────────────────────────────
    splits = create_two_stage_splits(data_cfg)

    train_augment = build_train_augmentation() if args.augment else None
    if train_augment is not None:
        print(f"\n[augment] Training augmentation ENABLED:")
        print(f"          {train_augment}")
    else:
        print("\n[augment] Training augmentation disabled (raw image processor only).")

    train_dataset = TwoStageWatermarkDataset(
        *splits["train"], transform=image_processor, augment=train_augment,
    )
    val_dataset   = TwoStageWatermarkDataset(*splits["val"],   transform=image_processor)

    print(f"\nTrain: {len(train_dataset)}  Val: {len(val_dataset)}")
    sample = train_dataset[0]
    print(f"Sample shape: {sample['pixel_values'].shape}, "
          f"stage1={sample['stage1_labels']}, stage2={sample['stage2_labels']}")

    # ── Model ─────────────────────────────────────────────────────────────
    # Use HF label smoothing manually inside the model heads; keep
    # TrainingArguments.label_smoothing_factor at 0 to avoid double-application.
    model = TwoStageWatermarkClassifier(
        model_name=model_cfg.model_name,
        num_stage1_labels=model_cfg.num_stage1_labels,
        num_stage2_labels=model_cfg.num_stage2_labels,
        head_dropout=model_cfg.head_dropout,
        label_smoothing=train_cfg.label_smoothing_factor,
        cache_dir=model_cfg.cache_dir,
    )

    if args.freeze_backbone_stages > 0:
        print(f"\nFreezing first {args.freeze_backbone_stages} backbone stages")
        for i, stage in enumerate(model.backbone.encoder.stages):
            if i < args.freeze_backbone_stages:
                for param in stage.parameters():
                    param.requires_grad = False
        for param in model.backbone.embeddings.parameters():
            param.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_p:,}")
    print(f"Trainable params: {trainable_p:,}")

    # ── Training args ─────────────────────────────────────────────────────
    total_steps = (
        len(train_dataset)
        // (train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps)
    ) * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    os.environ["TENSORBOARD_LOGGING_DIR"] = train_cfg.logging_dir
    os.makedirs(train_cfg.logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        fp16=train_cfg.fp16,
        eval_strategy=train_cfg.eval_strategy,
        save_strategy=train_cfg.save_strategy,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        logging_steps=train_cfg.logging_steps,
        report_to=train_cfg.report_to,
        seed=train_cfg.seed,
        dataloader_num_workers=data_cfg.num_workers,
        # We apply label smoothing manually inside the head CE losses; disable
        # the Trainer-side smoothing (it operates on `labels`, which we don't
        # emit) to avoid confusion.
        label_smoothing_factor=0.0,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        # Tell Trainer that our dataset has two label fields, not one — so it
        # forwards both into the model and returns both in label_ids during
        # eval/predict.
        label_names=["stage1_labels", "stage2_labels"],
    )

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=train_cfg.early_stopping_patience
    )]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=make_compute_metrics(
            tau=train_cfg.tau,
            num_scheme_classes=model_cfg.num_stage2_labels,
        ),
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training  (2-stage: binary WM detection + 8-class scheme ID)")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    best_model_dir = os.path.join(train_cfg.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    # Save weights of the custom model as a plain state_dict, plus a small
    # config json so it can be loaded back cleanly at inference time.
    torch.save(model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
    with open(os.path.join(best_model_dir, "two_stage_config.json"), "w") as f:
        json.dump({
            "model_name":         model_cfg.model_name,
            "num_stage1_labels":  model_cfg.num_stage1_labels,
            "num_stage2_labels":  model_cfg.num_stage2_labels,
            "head_dropout":       model_cfg.head_dropout,
            "stage1_id2label":    {int(k): v for k, v in stage1_i2l.items()},
            "stage2_id2label":    {int(k): v for k, v in stage2_i2l.items()},
            "tau":                train_cfg.tau,
        }, f, indent=2)
    image_processor.save_pretrained(best_model_dir)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # ── Val ───────────────────────────────────────────────────────────────
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ── Test ──────────────────────────────────────────────────────────────
    test_dataset = TwoStageWatermarkDataset(*splits["test"], transform=image_processor)
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    test_preds = trainer.predict(test_dataset)
    stage1_logits, stage2_logits = test_preds.predictions
    stage1_labels, stage2_labels = test_preds.label_ids

    s1_pred = np.argmax(stage1_logits, axis=-1)
    s2_pred = np.argmax(stage2_logits, axis=-1)

    print("\nStage-1 classification report (binary):")
    print(classification_report(
        stage1_labels, s1_pred,
        target_names=["not_watermarked", "watermarked"], digits=4,
    ))

    wm_mask = stage2_labels != STAGE2_IGNORE_INDEX
    if wm_mask.any():
        report = classification_report(
            stage2_labels[wm_mask], s2_pred[wm_mask],
            labels=list(range(model_cfg.num_stage2_labels)),
            target_names=scheme_names, digits=4, zero_division=0,
        )
        print("\nStage-2 classification report (8 schemes, watermarked only):")
        print(report)
        with open(os.path.join(train_cfg.output_dir, "test_stage2_report.txt"), "w") as f:
            f.write(report)

    # Unknown-flag stats at the configured tau (default -inf => none flagged).
    unknown_mask = detect_unknown(stage2_logits, train_cfg.tau)
    print(f"\nUnknown flag rate at tau={train_cfg.tau}: "
          f"{unknown_mask.mean():.4f} ({int(unknown_mask.sum())}/{len(unknown_mask)})")

    with open(os.path.join(train_cfg.output_dir, "test_predictions.json"), "w") as f:
        json.dump({
            "image_paths":    splits["test"][0],
            "stage1_true":    [int(x) for x in stage1_labels],
            "stage1_pred":    [int(x) for x in s1_pred],
            "stage2_true":    [int(x) for x in stage2_labels],  # -100 for real
            "stage2_pred":    [int(x) for x in s2_pred],
            "unknown_flag":   [bool(x) for x in unknown_mask],
            "tau":            train_cfg.tau,
            "stage1_id2label": {int(k): v for k, v in stage1_i2l.items()},
            "stage2_id2label": {int(k): v for k, v in stage2_i2l.items()},
        }, f, indent=2)

    print(f"\nOutputs saved to: {train_cfg.output_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2-stage watermark classifier")
    parser.add_argument("--epochs",                 type=int,   default=None)
    parser.add_argument("--batch-size",             type=int,   default=None)
    parser.add_argument("--lr",                     type=float, default=None)
    parser.add_argument("--image-size",             type=int,   default=None)
    parser.add_argument("--output-dir",             type=str,   default=None)
    parser.add_argument("--freeze-backbone-stages", type=int,   default=0)
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Unknown-scheme threshold (inference only). "
             "is_unknown = max(stage2_logits) < tau. "
             "Default = -inf => never flag anything as unknown.",
    )
    parser.add_argument(
        "--real-max", type=int, default=None,
        help="Override the max number of real (unwatermarked) images used. "
             "Lets you address stage-1 class imbalance without touching the "
             "other class caps. Falls back to DataConfig.max_per_class when "
             "unset.",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable training-time data augmentation "
             "(random flips + mild colour jitter + occasional light blur). "
             "Only applied to the training split — val/test are untouched.",
    )
    main(parser.parse_args())
