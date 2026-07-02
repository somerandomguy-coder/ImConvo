import os
from datetime import datetime
import math
import argparse

import numpy as np
import tensorflow as tf
from src import (
    FRONTEND_MODELS,
    MODEL_VARIANTS,
    NUM_CHARS,
    LipReadingCTC,
    build_lipreading_ctc,
    char_indices_to_text,
)
from src.dataset import build_split_arrays, create_ctc_dataset
from src.llm_postprocessor import LLMPostprocessor

CONFIG = {
    "model_variant": "bigru",
    "frontend_model": "flatten",
    "batch_size": 8,
    "preprocessed_dir": "./data/preprocessed/",
    "split_dir": "./splits/grid_v1",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lipreading model on hard splits")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to model checkpoint file (.keras/.weights)",
    )
    parser.add_argument(
        "--model-variant",
        choices=MODEL_VARIANTS,
        default=None,
        help="Temporal backbone variant override",
    )
    parser.add_argument(
        "--frontend-model",
        choices=FRONTEND_MODELS,
        default=None,
        help="Visual frontend override. If omitted, infer from checkpoint name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size override",
    )
    parser.add_argument(
        "--llm-postprocess",
        action="store_true",
        default=False,
        help="Apply Gemini LLM post-processing to correct CTC predictions.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=LLMPostprocessor.DEFAULT_MODEL,
        help="Gemini model name for post-processing.",
    )
    return parser.parse_args()


def infer_frontend_from_checkpoint_path(checkpoint_path: str) -> str:
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0].lower()
    if stem.endswith("_resnet18"):
        return "resnet18"
    if stem.endswith("_gap_proj"):
        return "gap_proj"
    return "flatten"

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
    postprocessor: LLMPostprocessor | None = None,
):
    total_wer, total_cer = 0.0, 0.0
    total_wer_llm, total_cer_llm = 0.0, 0.0
    num_samples = 0

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

            if postprocessor is not None:
                corrected_text = postprocessor.correct(pred_text)
                wer_llm = compute_wer(gt_text, corrected_text)
                cer_llm = compute_cer(gt_text, corrected_text)
                total_wer_llm += wer_llm
                total_cer_llm += cer_llm
                report_file.write(
                    f"{split_name} sample {num_samples} | "
                    f"WER: {wer:.2f} -> {wer_llm:.2f} | "
                    f"GT: '{gt_text}' | Pred: '{pred_text}' | LLM: '{corrected_text}'\n"
                )
            else:
                report_file.write(
                    f"{split_name} sample {num_samples} | "
                    f"WER: {wer:.2f} | GT: '{gt_text}' | Pred: '{pred_text}'\n"
                )

        if (batch_idx + 1) % 5 == 0:
            print(f"  [{split_name}] processed {num_samples} samples...")

    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)

    if postprocessor is not None:
        avg_wer_llm = total_wer_llm / max(num_samples, 1)
        avg_cer_llm = total_cer_llm / max(num_samples, 1)
        summary = (
            f"\n{split_name} SUMMARY\n"
            f"Total Samples: {num_samples}\n"
            f"Average WER (raw):  {avg_wer:.4f} ({avg_wer*100:.1f}%)\n"
            f"Average WER (LLM):  {avg_wer_llm:.4f} ({avg_wer_llm*100:.1f}%)\n"
            f"Average CER (raw):  {avg_cer:.4f} ({avg_cer*100:.1f}%)\n"
            f"Average CER (LLM):  {avg_cer_llm:.4f} ({avg_cer_llm*100:.1f}%)\n"
            f"LLM cache hits: {postprocessor.cache_size} unique predictions cached\n"
        )
        report_file.write(summary)
        print(summary)
        return avg_wer, avg_cer, num_samples, avg_wer_llm, avg_cer_llm
    else:
        summary = (
            f"\n{split_name} SUMMARY\n"
            f"Total Samples: {num_samples}\n"
            f"Average WER:   {avg_wer:.4f} ({avg_wer*100:.1f}%)\n"
            f"Average CER:   {avg_cer:.4f} ({avg_cer*100:.1f}%)\n"
        )
        report_file.write(summary)
        print(summary)
        return avg_wer, avg_cer, num_samples, None, None


def run_evaluation():
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    model_variant = (args.model_variant or CONFIG["model_variant"]).lower()
    frontend_model = (
        args.frontend_model
        or infer_frontend_from_checkpoint_path(checkpoint_path)
        or CONFIG["frontend_model"]
    ).lower()
    batch_size = args.batch_size or CONFIG["batch_size"]

    if frontend_model not in FRONTEND_MODELS:
        print(f"❌ Error: Unsupported frontend_model='{frontend_model}'")
        return

    # 1. Setup paths
    preprocessed_dir = CONFIG["preprocessed_dir"]
    split_dir = CONFIG["split_dir"]
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
    print(f"   Variant: {model_variant}")
    print(f"   Frontend: {frontend_model}")
    model = build_lipreading_ctc(
        model_variant=model_variant,
        frontend_model=frontend_model,
        num_chars=NUM_CHARS,
    )
    # Build model with dummy input
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
    model.load_weights(checkpoint_path)

    # 4. Optionally initialise LLM post-processor
    postprocessor = None
    if args.llm_postprocess:
        print(f"🔮 LLM post-processing enabled (model: {args.llm_model})")
        postprocessor = LLMPostprocessor(
            api_key=args.gemini_api_key,
            model=args.llm_model,
        )

    split_names = ["val_oos", "val_is", "test_oos", "test_is"]
    print(f"📝 Evaluating {split_names}. Writing results to {report_path}...")

    with open(report_path, "w", buffering=1024 * 1024) as f:
        f.write(f"LipNet Evaluation Report - {datetime.now()}\n")
        f.write(f"Model Variant: {model_variant}\n")
        f.write(f"Frontend Model: {frontend_model}\n")
        if postprocessor is not None:
            f.write(f"LLM Post-processing: {args.llm_model}\n")
        f.write(f"{'='*60}\n\n")
        aggregate = {}

        for split_name in split_names:
            split_file = os.path.join(split_dir, f"{split_name}.txt")
            with open(split_file, encoding="utf-8") as split_f:
                sample_ids = [line.strip() for line in split_f if line.strip()]

            paths, labels, lengths = build_split_arrays(preprocessed_dir, sample_ids)
            split_ds = create_ctc_dataset(
                paths,
                labels,
                lengths,
                batch_size=batch_size,
                shuffle=False,
                training=False,
                augmentation_profile="off",
            )
            num_steps = math.ceil(len(paths) / batch_size)
            wer, cer, count, wer_llm, cer_llm = evaluate_split(
                model=model,
                split_name=split_name,
                dataset=split_ds,
                num_steps=num_steps,
                report_file=f,
                postprocessor=postprocessor,
            )
            aggregate[split_name] = {
                "wer": wer, "cer": cer, "count": count,
                "wer_llm": wer_llm, "cer_llm": cer_llm,
            }

        f.write(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}\n")
        for split_name in split_names:
            row = aggregate[split_name]
            if row["wer_llm"] is not None:
                f.write(
                    f"{split_name}: count={row['count']}, "
                    f"WER={row['wer']:.4f} -> {row['wer_llm']:.4f} (LLM), "
                    f"CER={row['cer']:.4f} -> {row['cer_llm']:.4f} (LLM)\n"
                )
            else:
                f.write(
                    f"{split_name}: count={row['count']}, "
                    f"WER={row['wer']:.4f}, CER={row['cer']:.4f}\n"
                )

        print("\nFinal summary:")
        for split_name in split_names:
            row = aggregate[split_name]
            if row["wer_llm"] is not None:
                print(
                    f"  {split_name}: count={row['count']}, "
                    f"WER {row['wer']:.4f} -> {row['wer_llm']:.4f} (LLM), "
                    f"CER {row['cer']:.4f} -> {row['cer_llm']:.4f} (LLM)"
                )
            else:
                print(
                    f"  {split_name}: count={row['count']}, "
                    f"WER={row['wer']:.4f}, CER={row['cer']:.4f}"
                )
    from src.visualization import save_loss_plot

    save_loss_plot("./checkpoints/training_history.json")


if __name__ == "__main__":
    run_evaluation()
