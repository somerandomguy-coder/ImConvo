import os
from datetime import datetime
import math

import numpy as np
import tensorflow as tf
from src import NUM_CHARS, LipReadingCTC, char_indices_to_text
from src.dataset import build_split_arrays, create_ctc_dataset

# Re-use your compute functions from train.py
def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between two strings."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Levenshtein distance at word level
    r = len(ref_words)
    h = len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,     # deletion
                    d[i][j - 1] + 1,     # insertion
                    d[i - 1][j - 1] + 1  # substitution
                )

    return d[r][h] / max(r, 1)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between two strings."""
    r = len(reference)
    h = len(hypothesis)
    d = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1
                )

    return d[r][h] / max(r, 1)

def evaluate_split(
    model: LipReadingCTC,
    split_name: str,
    dataset: tf.data.Dataset,
    num_steps: int,
    report_file,
):
    total_wer, total_cer, num_samples = 0.0, 0.0, 0
    report_file.write(f"\n{'='*60}\n")
    report_file.write(f"SPLIT: {split_name}\n")
    report_file.write(f"{'='*60}\n")

    for batch_idx, batch in enumerate(dataset.take(num_steps)):
        x, y = batch
        logits = model(x, training=False)
        decoded_batch = model.decode_greedy(logits)

        labels = y["labels"].numpy()
        lengths = y["label_length"].numpy()

        for i in range(len(labels)):
            gt_text = char_indices_to_text(labels[i][: lengths[i]].tolist())
            pred_indices = decoded_batch[i]
            pred_indices = pred_indices[pred_indices >= 0]
            pred_text = char_indices_to_text(pred_indices.tolist())

            wer = compute_wer(gt_text, pred_text)
            cer = compute_cer(gt_text, pred_text)

            total_wer += wer
            total_cer += cer
            num_samples += 1
            report_file.write(
                f"{split_name} sample {num_samples} | "
                f"WER: {wer:.2f} | GT: '{gt_text}' | Pred: '{pred_text}'\n"
            )

        if (batch_idx + 1) % 5 == 0:
            print(f"  [{split_name}] processed {num_samples} samples...")

    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)
    summary = (
        f"\n{split_name} SUMMARY\n"
        f"Total Samples: {num_samples}\n"
        f"Average WER:   {avg_wer:.4f} ({avg_wer*100:.1f}%)\n"
        f"Average CER:   {avg_cer:.4f} ({avg_cer*100:.1f}%)\n"
    )
    report_file.write(summary)
    print(summary)
    return avg_wer, avg_cer, num_samples


def run_evaluation():
    BATCH_SIZE = 8

    # 1. Setup paths
    checkpoint_path = "./checkpoints/best_ctc_model.keras"
    preprocessed_dir = "./data/preprocessed/"
    split_dir = "./splits/grid_v1"
    report_path = (
        f"./reports/eval_result/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    if not os.path.isdir(split_dir):
        print(
            f"❌ Error: split manifests not found at {split_dir}. "
            "Run: python scripts/build_split_manifests.py"
        )
        return

    # 3. Load Model
    print("🤖 Loading model weights...")
    model = LipReadingCTC(num_chars=NUM_CHARS)
    # Build model with dummy input
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
    model.load_weights(checkpoint_path)

    split_names = ["val_oos", "val_is", "test_oos", "test_is"]
    print(f"📝 Evaluating {split_names}. Writing results to {report_path}...")

    with open(report_path, "w", buffering=1024 * 1024) as f:
        f.write(f"LipNet Evaluation Report - {datetime.now()}\n")
        f.write(f"{'='*60}\n\n")
        aggregate = {}

        for split_name in split_names:
            split_file = os.path.join(split_dir, f"{split_name}.txt")
            with open(split_file, encoding="utf-8") as split_f:
                sample_ids = [line.strip() for line in split_f if line.strip()]

            paths, labels, lengths = build_split_arrays(preprocessed_dir, sample_ids)
            split_ds = create_ctc_dataset(
                paths, labels, lengths, batch_size=BATCH_SIZE, shuffle=False
            )
            num_steps = math.ceil(len(paths) / BATCH_SIZE)
            wer, cer, count = evaluate_split(
                model=model,
                split_name=split_name,
                dataset=split_ds,
                num_steps=num_steps,
                report_file=f,
            )
            aggregate[split_name] = {"wer": wer, "cer": cer, "count": count}

        f.write(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}\n")
        for split_name in split_names:
            row = aggregate[split_name]
            f.write(
                f"{split_name}: count={row['count']}, "
                f"WER={row['wer']:.4f}, CER={row['cer']:.4f}\n"
            )

        print("\nFinal summary:")
        for split_name in split_names:
            row = aggregate[split_name]
            print(
                f"  {split_name}: count={row['count']}, "
                f"WER={row['wer']:.4f}, CER={row['cer']:.4f}"
            )
    from src.visualization import save_loss_plot

    save_loss_plot("./checkpoints/training_history.json")


if __name__ == "__main__":
    run_evaluation()
