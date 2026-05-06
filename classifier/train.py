"""
Training script for watermark classifier v3 (9 classes: 8 methods + real).
Fine-tunes ConvNeXt-V2 Large at 512x512 resolution.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import (
    ConvNextV2ForImageClassification,
    ConvNextImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import DataConfig, ModelConfig, TrainConfig, get_label_maps
from dataset import WatermarkDataset, create_splits


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(args):
    data_cfg  = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.epochs     is not None: train_cfg.num_epochs = args.epochs
    if args.batch_size is not None: train_cfg.per_device_train_batch_size = args.batch_size
    if args.lr         is not None: train_cfg.learning_rate = args.lr
    if args.image_size is not None: data_cfg.image_size = args.image_size
    if args.output_dir is not None: train_cfg.output_dir = args.output_dir

    label2id, id2label = get_label_maps(data_cfg.methods)

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
    splits = create_splits(data_cfg)

    train_dataset = WatermarkDataset(splits["train"][0], splits["train"][1], image_processor)
    val_dataset   = WatermarkDataset(splits["val"][0],   splits["val"][1],   image_processor)

    print(f"\nTrain: {len(train_dataset)}  Val: {len(val_dataset)}")
    sample = train_dataset[0]
    print(f"Sample shape: {sample['pixel_values'].shape}, label: {id2label[sample['labels']]}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = ConvNextV2ForImageClassification.from_pretrained(
        model_cfg.model_name,
        num_labels=model_cfg.num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        cache_dir=model_cfg.cache_dir,
    )

    if args.freeze_backbone_stages > 0:
        print(f"\nFreezing first {args.freeze_backbone_stages} backbone stages")
        for i, stage in enumerate(model.convnextv2.encoder.stages):
            if i < args.freeze_backbone_stages:
                for param in stage.parameters():
                    param.requires_grad = False
        for param in model.convnextv2.embeddings.parameters():
            param.requires_grad = False

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_p:,}")
    print(f"Trainable params: {trainable_p:,}")

    # ── Training args ─────────────────────────────────────────────────────
    total_steps   = (len(train_dataset) // (train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps)) * train_cfg.num_epochs
    warmup_steps  = int(total_steps * train_cfg.warmup_ratio)
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
        label_smoothing_factor=train_cfg.label_smoothing_factor,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=train_cfg.early_stopping_patience
    )]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting training  (9 classes: 8 watermark methods + real)")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    best_model_dir = os.path.join(train_cfg.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    image_processor.save_pretrained(best_model_dir)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # ── Val ───────────────────────────────────────────────────────────────
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ── Test ──────────────────────────────────────────────────────────────
    test_dataset = WatermarkDataset(splits["test"][0], splits["test"][1], image_processor)
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    test_preds   = trainer.predict(test_dataset)
    pred_labels  = np.argmax(test_preds.predictions, axis=-1)
    true_labels  = test_preds.label_ids
    report = classification_report(true_labels, pred_labels, target_names=data_cfg.methods, digits=4)
    print("\nTest Classification Report:")
    print(report)

    with open(os.path.join(train_cfg.output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(train_cfg.output_dir, "test_predictions.json"), "w") as f:
        json.dump({
            "image_paths": splits["test"][0],
            "true_labels": [int(l) for l in true_labels],
            "pred_labels": [int(l) for l in pred_labels],
            "id2label": id2label,
        }, f, indent=2)

    print(f"\nOutputs saved to: {train_cfg.output_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 9-class watermark classifier")
    parser.add_argument("--epochs",                type=int,   default=None)
    parser.add_argument("--batch-size",            type=int,   default=None)
    parser.add_argument("--lr",                    type=float, default=None)
    parser.add_argument("--image-size",            type=int,   default=None)
    parser.add_argument("--output-dir",            type=str,   default=None)
    parser.add_argument("--freeze-backbone-stages",type=int,   default=0)
    main(parser.parse_args())
