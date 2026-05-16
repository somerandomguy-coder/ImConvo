"""Evaluate isolated-word GRID checkpoints on all hard splits."""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.isolated_word_level.common import (
    NUM_WORD_CLASSES,
    detect_preprocessed_roi_backend,
    infer_frontend_from_checkpoint_path,
    infer_variant_from_checkpoint_path,
    word_idx_to_text,
)
from experiments.isolated_word_level.dataset import (
    DEFAULT_TARGET_FRAMES,
    build_word_segments_from_ids,
    create_isolated_word_dataset,
)
from experiments.isolated_word_level.model import (
    FRONTEND_MODELS,
    MODEL_VARIANTS,
    build_lipreading_isolated_word_classifier,
)
from src.dataset import load_split_ids

CONFIG = {
    "model_variant": "bigru",
    "frontend_model": "flatten",
    "batch_size": 16,
    "target_frames": DEFAULT_TARGET_FRAMES,
    "pad_mode": "repeat_last",
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
    "report_dir": "./reports/eval_result_isolated_word",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate isolated-word GRID model")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default=None)
    parser.add_argument("--frontend-model", choices=FRONTEND_MODELS, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-frames", type=int, default=None)
    parser.add_argument("--pad-mode", choices=["repeat_last", "zeros"], default=None)
    parser.add_argument("--force-center-crop", action="store_true", default=False)
    return parser.parse_args()


def evaluate_split(
    model,
    split_name: str,
    dataset,
    num_steps: int,
    report_file,
):
    total_top1 = 0.0
    total_top5 = 0.0
    num_samples = 0

    report_file.write(f"\n{'='*60}\n")
    report_file.write(f"SPLIT: {split_name}\n")
    report_file.write(f"{'='*60}\n")

    for batch_idx, batch in enumerate(dataset.take(num_steps)):
        x, y = batch
        labels = y["label"].numpy().astype(np.int32)
        logits = model(x, training=False).numpy()

        top1_pred = np.argmax(logits, axis=1)
        top5_pred = np.argsort(logits, axis=1)[:, ::-1][:, :5]

        for i in range(len(labels)):
            gt = int(labels[i])
            pred1 = int(top1_pred[i])
            pred5 = top5_pred[i].tolist()

            is_top1 = float(pred1 == gt)
            is_top5 = float(gt in pred5)

            total_top1 += is_top1
            total_top5 += is_top5
            num_samples += 1

            gt_word = word_idx_to_text(gt)
            pred1_word = word_idx_to_text(pred1)
            pred5_words = [word_idx_to_text(idx) for idx in pred5]
            report_file.write(
                f"{split_name} sample {num_samples} | "
                f"top1={is_top1:.0f} top5={is_top5:.0f} | "
                f"GT='{gt_word}' | top1='{pred1_word}' | top5={pred5_words}\n"
            )

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{split_name}] processed {num_samples} samples...")

    avg_top1 = total_top1 / max(num_samples, 1)
    avg_top5 = total_top5 / max(num_samples, 1)

    summary = (
        f"\n{split_name} SUMMARY\n"
        f"Total Samples: {num_samples}\n"
        f"Top-1 Accuracy: {avg_top1:.4f} ({avg_top1*100:.1f}%)\n"
        f"Top-5 Accuracy: {avg_top5:.4f} ({avg_top5*100:.1f}%)\n"
    )
    report_file.write(summary)
    print(summary)

    return avg_top1, avg_top5, num_samples


def run_evaluation():
    args = parse_args()
    checkpoint_path = args.checkpoint_path

    model_variant = (
        args.model_variant
        or infer_variant_from_checkpoint_path(checkpoint_path)
        or CONFIG["model_variant"]
    ).lower()
    frontend_model = (
        args.frontend_model
        or infer_frontend_from_checkpoint_path(checkpoint_path)
        or CONFIG["frontend_model"]
    ).lower()

    batch_size = int(args.batch_size or CONFIG["batch_size"])
    target_frames = int(args.target_frames or CONFIG["target_frames"])
    pad_mode = (args.pad_mode or CONFIG["pad_mode"]).lower()

    if frontend_model not in FRONTEND_MODELS:
        raise ValueError(f"Unsupported frontend_model='{frontend_model}'")
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant='{model_variant}'")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not os.path.isdir(CONFIG["split_dir"]):
        raise FileNotFoundError(f"Split dir not found: {CONFIG['split_dir']}")

    os.makedirs(CONFIG["report_dir"], exist_ok=True)
    report_path = os.path.join(
        CONFIG["report_dir"],
        f"eval_report_isolated_word_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    backend_name, roi_centered = detect_preprocessed_roi_backend(project_root=".")
    enforce_center_crop = bool(args.force_center_crop) or (not roi_centered)

    print("Loading isolated-word model...")
    print(f"  model_variant={model_variant}")
    print(f"  frontend_model={frontend_model}")
    print(f"  target_frames={target_frames} pad_mode={pad_mode}")
    print(f"  preprocess_backend={backend_name} roi_centered={roi_centered}")
    print(f"  enforce_center_crop={enforce_center_crop}")

    model = build_lipreading_isolated_word_classifier(
        model_variant=model_variant,
        frontend_model=frontend_model,
        num_word_classes=NUM_WORD_CLASSES,
    )

    _ = model(np.random.randn(1, target_frames, 80, 120, 1).astype(np.float32))
    model.load_weights(checkpoint_path)

    split_names = ["val_oos", "val_is", "test_oos", "test_is"]
    aggregate = {}

    with open(report_path, "w", buffering=1024 * 1024, encoding="utf-8") as f:
        f.write(f"GRID Isolated-Word Evaluation Report - {datetime.now()}\n")
        f.write(f"Model Variant: {model_variant}\n")
        f.write(f"Frontend Model: {frontend_model}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Target Frames: {target_frames} | pad_mode={pad_mode}\n")
        f.write("Loss: sparse cross-entropy\n")
        f.write(f"{'='*60}\n\n")

        for split_name in split_names:
            sample_ids = load_split_ids(split_dir=CONFIG["split_dir"], split_name=split_name)
            segments, stats = build_word_segments_from_ids(
                preprocessed_dir=CONFIG["preprocessed_dir"],
                sample_ids=sample_ids,
            )
            if not segments:
                raise ValueError(f"No segments found for split={split_name}")

            paths = [seg.npy_path for seg in segments]
            starts = np.array([seg.start_frame for seg in segments], dtype=np.int32)
            ends = np.array([seg.end_frame for seg in segments], dtype=np.int32)
            labels = np.array([seg.class_idx for seg in segments], dtype=np.int32)

            split_ds = create_isolated_word_dataset(
                paths,
                starts,
                ends,
                labels,
                batch_size=batch_size,
                target_frames=target_frames,
                pad_mode=pad_mode,
                shuffle=False,
                training=False,
                augmentation_profile="off",
                enforce_center_crop=enforce_center_crop,
            )
            num_steps = math.ceil(len(segments) / batch_size)

            top1, top5, count = evaluate_split(
                model=model,
                split_name=split_name,
                dataset=split_ds,
                num_steps=num_steps,
                report_file=f,
            )

            aggregate[split_name] = {
                "top1_acc": top1,
                "top5_acc": top5,
                "count": count,
                "segment_stats": {
                    "missing_align": stats.missing_align,
                    "missing_npy": stats.missing_npy,
                    "bad_align": stats.bad_align,
                    "unknown_word": stats.unknown_word,
                    "out_of_bounds_fixed": stats.out_of_bounds_fixed,
                    "empty_segment_fixed": stats.empty_segment_fixed,
                },
            }

        f.write(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}\n")
        for split_name in split_names:
            row = aggregate[split_name]
            f.write(
                f"{split_name}: count={row['count']}, "
                f"TOP1={row['top1_acc']:.4f}, TOP5={row['top5_acc']:.4f}\n"
            )

    print("\nFinal summary:")
    for split_name in split_names:
        row = aggregate[split_name]
        print(
            f"  {split_name}: count={row['count']}, "
            f"TOP1={row['top1_acc']:.4f}, TOP5={row['top5_acc']:.4f}"
        )

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run_evaluation()
