"""Evaluate GRID word-level checkpoints on val/test splits."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
import sys

import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd())


from experiments.word_level_grid.common import (
    SLOT_VOCAB_SIZES,
    infer_variant_from_checkpoint_path,
    indices_to_sentence,
    infer_frontend_from_checkpoint_path,
)
from experiments.word_level_grid.dataset import build_split_word_arrays, create_word_dataset
from experiments.word_level_grid.metrics import compute_cer, compute_wer
from experiments.word_level_grid.model import (
    FRONTEND_MODELS,
    MODEL_VARIANTS,
    build_lipreading_word_classifier,
)

CONFIG = {
    "model_variant": "bigru",
    "frontend_model": "flatten",
    "batch_size": 8,
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
    "report_dir": "./reports/eval_result_word",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRID word-level lipreading model")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default=None)
    parser.add_argument("--frontend-model", choices=FRONTEND_MODELS, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def evaluate_split(
    model,
    split_name: str,
    dataset: tf.data.Dataset,
    num_steps: int,
    report_file,
):
    total_wer = 0.0
    total_cer = 0.0
    total_sentence_acc = 0.0
    total_slot_acc = 0.0
    num_samples = 0

    report_file.write(f"\n{'='*60}\n")
    report_file.write(f"SPLIT: {split_name}\n")
    report_file.write(f"{'='*60}\n")

    for batch_idx, batch in enumerate(dataset.take(num_steps)):
        x, y = batch
        logits_list = model(x, training=False)
        pred = model.decode_greedy_slots(logits_list).numpy()
        labels = y["word_labels"].numpy()

        for i in range(len(labels)):
            gt = labels[i]
            pd = pred[i]

            gt_sentence = indices_to_sentence(gt)
            pd_sentence = indices_to_sentence(pd)

            wer = compute_wer(gt_sentence, pd_sentence)
            cer = compute_cer(gt_sentence, pd_sentence)
            sentence_acc = float(np.array_equal(gt, pd))
            slot_acc = float(np.mean(gt == pd))

            total_wer += wer
            total_cer += cer
            total_sentence_acc += sentence_acc
            total_slot_acc += slot_acc
            num_samples += 1

            report_file.write(
                f"{split_name} sample {num_samples} | "
                f"WER: {wer:.2f} | CER: {cer:.2f} | "
                f"SentAcc: {sentence_acc:.1f} | SlotAcc: {slot_acc:.2f} | "
                f"GT: '{gt_sentence}' | Pred: '{pd_sentence}'\n"
            )

        if (batch_idx + 1) % 5 == 0:
            print(f"  [{split_name}] processed {num_samples} samples...")

    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)
    avg_sentence_acc = total_sentence_acc / max(num_samples, 1)
    avg_slot_acc = total_slot_acc / max(num_samples, 1)

    summary = (
        f"\n{split_name} SUMMARY\n"
        f"Total Samples: {num_samples}\n"
        f"Average WER:      {avg_wer:.4f} ({avg_wer*100:.1f}%)\n"
        f"Average CER:      {avg_cer:.4f} ({avg_cer*100:.1f}%)\n"
        f"Sentence Accuracy:{avg_sentence_acc:.4f} ({avg_sentence_acc*100:.1f}%)\n"
        f"Slot Accuracy:    {avg_slot_acc:.4f} ({avg_slot_acc*100:.1f}%)\n"
    )
    report_file.write(summary)
    print(summary)

    return avg_wer, avg_cer, num_samples, avg_sentence_acc, avg_slot_acc


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
    batch_size = args.batch_size or CONFIG["batch_size"]

    if frontend_model not in FRONTEND_MODELS:
        raise ValueError(f"Unsupported frontend_model='{frontend_model}'")

    preprocessed_dir = CONFIG["preprocessed_dir"]
    split_dir = CONFIG["split_dir"]
    os.makedirs(CONFIG["report_dir"], exist_ok=True)
    report_path = os.path.join(
        CONFIG["report_dir"],
        f"eval_report_word_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Split manifests not found at {split_dir}. "
            "Run scripts/build_split_manifests.py"
        )

    print("Loading model weights...")
    print(f"  Variant: {model_variant}")
    print(f"  Frontend: {frontend_model}")

    model = build_lipreading_word_classifier(
        model_variant=model_variant,
        frontend_model=frontend_model,
        slot_vocab_sizes=SLOT_VOCAB_SIZES,
    )

    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
    model.load_weights(checkpoint_path)

    split_names = ["val_oos", "val_is", "test_oos", "test_is"]
    print(f"Evaluating splits {split_names}. Writing to {report_path}")

    with open(report_path, "w", buffering=1024 * 1024, encoding="utf-8") as f:
        f.write(f"GRID Word-level Evaluation Report - {datetime.now()}\n")
        f.write(f"Model Variant: {model_variant}\n")
        f.write(f"Frontend Model: {frontend_model}\n")
        f.write("Loss: slot-wise cross-entropy (6 heads)\n")
        f.write(f"{'='*60}\n\n")

        aggregate = {}

        for split_name in split_names:
            split_file = os.path.join(split_dir, f"{split_name}.txt")
            with open(split_file, encoding="utf-8") as split_f:
                sample_ids = [line.strip() for line in split_f if line.strip()]

            paths, labels = build_split_word_arrays(preprocessed_dir, sample_ids)
            split_ds = create_word_dataset(
                paths,
                labels,
                batch_size=batch_size,
                shuffle=False,
                training=False,
                augmentation_profile="off",
            )
            num_steps = math.ceil(len(paths) / batch_size)
            wer, cer, count, sentence_acc, slot_acc = evaluate_split(
                model=model,
                split_name=split_name,
                dataset=split_ds,
                num_steps=num_steps,
                report_file=f,
            )
            aggregate[split_name] = {
                "wer": wer,
                "cer": cer,
                "count": count,
                "sentence_acc": sentence_acc,
                "slot_acc": slot_acc,
            }

        f.write(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}\n")
        for split_name in split_names:
            row = aggregate[split_name]
            f.write(
                f"{split_name}: count={row['count']}, "
                f"WER={row['wer']:.4f}, CER={row['cer']:.4f}, "
                f"SENT_ACC={row['sentence_acc']:.4f}, "
                f"SLOT_ACC={row['slot_acc']:.4f}\n"
            )

        print("\nFinal summary:")
        for split_name in split_names:
            row = aggregate[split_name]
            print(
                f"  {split_name}: count={row['count']}, "
                f"WER={row['wer']:.4f}, CER={row['cer']:.4f}, "
                f"SENT_ACC={row['sentence_acc']:.4f}, "
                f"SLOT_ACC={row['slot_acc']:.4f}"
            )

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    run_evaluation()
