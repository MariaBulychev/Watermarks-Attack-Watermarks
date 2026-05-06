"""
Test classifier_3 on MSCOCO SD3.5 watermarked images.
Each subfolder under mscoco_SD3.5/ is treated as the ground-truth class.
Only folders whose name matches one of the 9 trained classes are evaluated.
"""

import argparse
import os
import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import ConvNextV2ForImageClassification, ConvNextImageProcessor
from sklearn.metrics import classification_report, confusion_matrix


def load_model(model_dir, device):
    processor = ConvNextImageProcessor.from_pretrained(model_dir)
    model = ConvNextV2ForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, processor


def classify_images(model, processor, image_paths, device, batch_size=32):
    preds = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.environ.get("TEST_DATA_DIR", "./data"),
    )
    parser.add_argument(
        "--model-dir",
        default="./outputs/best_model",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-file", default=None,
                        help="Optional JSON file to save results")
    parser.add_argument("--dir-map", nargs="*", default=None,
                        help="Override dir->class mappings, e.g. zodiac_rand=zodiac wam_rand=wam")
    parser.add_argument("--max-per-dir", type=int, default=None,
                        help="Max images to evaluate per directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, processor = load_model(args.model_dir, device)

    # Get trained class names from model config
    # Normalize id2label keys to int (HF config may serialize as str or int)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    trained_classes = set(label2id.keys())
    print(f"Trained classes: {sorted(trained_classes)}")

    # Build dir->class name mapping
    dir_to_class = {}
    if args.dir_map:
        for mapping in args.dir_map:
            dirname, classname = mapping.split("=")
            dir_to_class[dirname] = classname

    # Scan subdirectories
    subdirs = sorted(
        d
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    )
    print(f"Found subdirs: {subdirs}")
    if dir_to_class:
        print(f"Dir->class overrides: {dir_to_class}")

    all_true = []
    all_pred = []
    all_paths = []
    per_class_results = {}

    for subdir in subdirs:
        # Map subdir name to class via override or direct match
        class_name = dir_to_class.get(subdir, subdir)
        if class_name not in trained_classes:
            print(f"  Skipping '{subdir}' — not in trained classes")
            continue

        true_label_id = label2id[class_name]
        dir_path = os.path.join(args.data_root, subdir)
        fnames = sorted(
            f
            for f in os.listdir(dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
        if args.max_per_dir and len(fnames) > args.max_per_dir:
            fnames = fnames[: args.max_per_dir]
        if not fnames:
            print(f"  Skipping '{subdir}' — no images")
            continue

        image_paths = [os.path.join(dir_path, fn) for fn in fnames]
        print(f"\n  Classifying {class_name}: {len(image_paths)} images ...", end=" ", flush=True)

        preds = classify_images(model, processor, image_paths, device, args.batch_size)
        pred_labels = [id2label[p] for p in preds]

        correct = sum(1 for p in preds if p == true_label_id)
        acc = correct / len(preds)
        print(f"accuracy = {acc:.4f} ({correct}/{len(preds)})")

        # Show misclassification breakdown
        misclass = defaultdict(int)
        for p in preds:
            if p != true_label_id:
                misclass[id2label[p]] += 1
        if misclass:
            mis_str = ", ".join(f"{k}: {v}" for k, v in sorted(misclass.items(), key=lambda x: -x[1]))
            print(f"    Misclassified as: {mis_str}")

        per_class_results[class_name] = {
            "total": len(preds),
            "correct": correct,
            "accuracy": acc,
            "misclassifications": dict(misclass),
        }

        true_ids = [true_label_id] * len(preds)
        all_true.extend(true_ids)
        all_pred.extend(preds)
        all_paths.extend(image_paths)

    # Overall report
    evaluated_labels = sorted(set(all_true + all_pred))
    target_names = [id2label[i] for i in evaluated_labels]

    print("\n" + "=" * 70)
    print("OVERALL CLASSIFICATION REPORT")
    print("=" * 70)
    report = classification_report(
        all_true, all_pred, labels=evaluated_labels, target_names=target_names, digits=4
    )
    print(report)

    cm = confusion_matrix(all_true, all_pred, labels=evaluated_labels)
    print("Confusion Matrix:")
    header = "".join(f"{n:>12s}" for n in target_names)
    print(f"{'':>12s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12d}" for v in row)
        print(f"{target_names[i]:>12s}{row_str}")

    # Save results
    if args.output_file:
        results = {
            "data_root": args.data_root,
            "model_dir": args.model_dir,
            "per_class": per_class_results,
            "overall_accuracy": sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true) if all_true else 0,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": target_names,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
