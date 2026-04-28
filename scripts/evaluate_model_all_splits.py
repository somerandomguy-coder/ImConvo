#!/usr/bin/env python3
"""
Evaluate one lip-reading model on train/valid/test splits and export metrics as JSON.

Outputs:
- Accuracy (exact sentence match)
- Word Error Rate (WER)
- Character Error Rate (CER)

Usage example:
python scripts/evaluate_model_all_splits.py \
  --model-path ./model_1.keras \
  --preprocessed-dir ./data/preprocessed_model_1 \
  --model-name model_1 \
  --output-json ./temp_reports/model_1_eval_splits.json \
  --save-split-file ./temp_reports/shared_split.json
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src import (FRAME_HEIGHT, FRAME_WIDTH, MAX_FRAMES, NUM_CHARS, LipReadingCTC,
                 char_indices_to_text, parse_alignment_text)


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    r, h = len(ref_words), len(hyp_words)
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
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[r][h] / max(r, 1)


def compute_cer(reference: str, hypothesis: str) -> float:
    r, h = len(reference), len(hypothesis)
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
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[r][h] / max(r, 1)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


@dataclass
class Sample:
    name: str
    npy_path: Path
    align_path: Path
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one model on train/valid/test splits and export accuracy, WER, CER as JSON."
        )
    )
    parser.add_argument("--model-path", required=True, help="Path to model weights (.keras).")
    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help="Directory containing .npy files and manifest.txt.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest path (default: <preprocessed-dir>/manifest.txt).",
    )
    parser.add_argument(
        "--align-dir",
        default=None,
        help=(
            "Optional align directory override. If omitted, tries "
            "<preprocessed-dir>/align then <preprocessed-parent>/align."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Friendly model label for output JSON (default: stem of --model-path).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split generation when --split-file is not provided.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70).",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional JSON file with train/valid/test name lists to force identical splits.",
    )
    parser.add_argument(
        "--save-split-file",
        default=None,
        help="Optional path to save generated split JSON for reuse with other models.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="For quick smoke tests only. 0 means use all samples.",
    )
    parser.add_argument(
        "--output-json",
        default="temp_reports/model_eval_splits.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=250,
        help="Print progress every N samples per split.",
    )
    return parser.parse_args()


def get_align_candidates(preprocessed_dir: Path, align_dir_override: str | None) -> List[Path]:
    candidates: List[Path] = []
    if align_dir_override:
        candidates.append(Path(align_dir_override).resolve())
    candidates.append((preprocessed_dir / "align").resolve())
    candidates.append((preprocessed_dir.parent / "align").resolve())
    # Keep order but remove duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        if str(c) in seen:
            continue
        seen.add(str(c))
        unique_candidates.append(c)
    return unique_candidates


def find_align_path(sample_name: str, align_candidates: List[Path]) -> Path | None:
    for align_dir in align_candidates:
        p = align_dir / f"{sample_name}.align"
        if p.exists():
            return p
    return None


def load_samples(
    preprocessed_dir: Path,
    manifest_path: Path,
    align_candidates: List[Path],
) -> Dict[str, Sample]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    sample_map: Dict[str, Sample] = {}
    missing_npy = 0
    missing_align = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]

    for name in names:
        npy_path = preprocessed_dir / f"{name}.npy"
        if not npy_path.exists():
            missing_npy += 1
            continue

        align_path = find_align_path(name, align_candidates)
        if align_path is None:
            missing_align += 1
            continue

        text = normalize_text(parse_alignment_text(str(align_path)))
        sample_map[name] = Sample(
            name=name,
            npy_path=npy_path,
            align_path=align_path,
            text=text,
        )

    print(
        f"[Samples] manifest={len(names)}, usable={len(sample_map)}, "
        f"missing_npy={missing_npy}, missing_align={missing_align}"
    )
    return sample_map


def generate_split_names(
    names: List[str],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if train_ratio <= 0 or valid_ratio <= 0:
        raise ValueError("Both --train-ratio and --valid-ratio must be > 0.")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("--train-ratio + --valid-ratio must be < 1.0.")

    sorted_names = sorted(names)
    rng = random.Random(seed)
    rng.shuffle(sorted_names)

    n = len(sorted_names)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = n - n_train - n_valid

    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError(
            f"Split produced empty subset: train={n_train}, valid={n_valid}, test={n_test}. "
            "Adjust ratios or provide more samples."
        )

    return {
        "train": sorted_names[:n_train],
        "valid": sorted_names[n_train:n_train + n_valid],
        "test": sorted_names[n_train + n_valid:],
    }


def load_split_file(split_path: Path) -> Dict[str, List[str]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not all(k in payload for k in ("train", "valid", "test")):
        raise ValueError(f"Split file missing one of train/valid/test keys: {split_path}")
    return {
        "train": list(payload["train"]),
        "valid": list(payload["valid"]),
        "test": list(payload["test"]),
    }


def save_split_file(
    split_path: Path,
    split_names: Dict[str, List[str]],
    source_manifest: Path,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "source_manifest": str(source_manifest),
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": round(1.0 - train_ratio - valid_ratio, 6),
        "train": split_names["train"],
        "valid": split_names["valid"],
        "test": split_names["test"],
    }
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Split] Saved split file: {split_path}")


def to_frames_tensor(npy_path: Path) -> np.ndarray:
    frames = np.load(str(npy_path))
    if frames.ndim == 3:
        frames = frames[..., np.newaxis]
    if frames.ndim != 4 or frames.shape[-1] != 1:
        raise ValueError(f"Unexpected frame shape for {npy_path}: {frames.shape}")
    if frames.shape[0] != MAX_FRAMES or frames.shape[1] != FRAME_HEIGHT or frames.shape[2] != FRAME_WIDTH:
        raise ValueError(
            f"Unexpected frame size for {npy_path}: {frames.shape}. "
            f"Expected ({MAX_FRAMES}, {FRAME_HEIGHT}, {FRAME_WIDTH}, 1)."
        )
    return frames.astype(np.float32, copy=False)


def batched(items: List[Sample], batch_size: int) -> Iterable[List[Sample]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def evaluate_split(
    model: LipReadingCTC,
    split_name: str,
    samples: List[Sample],
    batch_size: int,
    verbose_every: int,
) -> Dict[str, float | int]:
    n = len(samples)
    if n == 0:
        return {
            "samples": 0,
            "correct": 0,
            "accuracy": 0.0,
            "wer": 0.0,
            "cer": 0.0,
        }

    total_wer = 0.0
    total_cer = 0.0
    correct = 0
    processed = 0

    for chunk in batched(samples, batch_size):
        batch_frames = np.stack([to_frames_tensor(s.npy_path) for s in chunk], axis=0)
        logits = model(batch_frames, training=False)
        decoded_batch = model.decode_greedy(logits)

        for i, sample in enumerate(chunk):
            pred_indices = decoded_batch[i]
            pred_indices = pred_indices[pred_indices >= 0]
            pred_text = normalize_text(char_indices_to_text(pred_indices.tolist()))
            gt_text = sample.text

            if pred_text == gt_text:
                correct += 1

            total_wer += compute_wer(gt_text, pred_text)
            total_cer += compute_cer(gt_text, pred_text)
            processed += 1

            if verbose_every > 0 and processed % verbose_every == 0:
                print(
                    f"[{split_name}] {processed}/{n} "
                    f"(acc={correct / processed:.4f}, wer={total_wer / processed:.4f}, cer={total_cer / processed:.4f})"
                )

    return {
        "samples": processed,
        "correct": correct,
        "accuracy": correct / max(processed, 1),
        "wer": total_wer / max(processed, 1),
        "cer": total_cer / max(processed, 1),
    }


def aggregate_metrics(metrics_by_split: Dict[str, Dict[str, float | int]]) -> Dict[str, float | int]:
    total_samples = 0
    total_correct = 0
    weighted_wer = 0.0
    weighted_cer = 0.0

    for split_name in ("train", "valid", "test"):
        m = metrics_by_split[split_name]
        samples = int(m["samples"])
        total_samples += samples
        total_correct += int(m["correct"])
        weighted_wer += float(m["wer"]) * samples
        weighted_cer += float(m["cer"]) * samples

    return {
        "samples": total_samples,
        "correct": total_correct,
        "accuracy": total_correct / max(total_samples, 1),
        "wer": weighted_wer / max(total_samples, 1),
        "cer": weighted_cer / max(total_samples, 1),
    }


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path).resolve()
    preprocessed_dir = Path(args.preprocessed_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else preprocessed_dir / "manifest.txt"
    model_name = args.model_name or model_path.stem

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

    align_candidates = get_align_candidates(preprocessed_dir, args.align_dir)
    print("[Align] Candidates:")
    for c in align_candidates:
        print(f"  - {c}")

    sample_map = load_samples(preprocessed_dir, manifest_path, align_candidates)
    if len(sample_map) < 3:
        raise RuntimeError("Not enough usable samples after manifest/align checks.")

    split_source = "generated"
    missing_from_split: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}

    if args.split_file:
        split_file = Path(args.split_file).resolve()
        if split_file.exists():
            split_names = load_split_file(split_file)
            split_source = str(split_file)
            print(f"[Split] Using existing split file: {split_file}")
        else:
            print(f"[Split] Split file not found: {split_file}")
            print("[Split] Auto-generating split from seed/ratios and saving it for reuse.")
            split_names = generate_split_names(
                names=list(sample_map.keys()),
                train_ratio=args.train_ratio,
                valid_ratio=args.valid_ratio,
                seed=args.seed,
            )
            save_split_file(
                split_path=split_file,
                split_names=split_names,
                source_manifest=manifest_path,
                seed=args.seed,
                train_ratio=args.train_ratio,
                valid_ratio=args.valid_ratio,
            )
            split_source = f"generated_then_saved:{split_file}"
    else:
        split_names = generate_split_names(
            names=list(sample_map.keys()),
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print(
            "[Split] Generated split sizes: "
            f"train={len(split_names['train'])}, valid={len(split_names['valid'])}, test={len(split_names['test'])}"
        )

    split_samples: Dict[str, List[Sample]] = {}
    for split_name in ("train", "valid", "test"):
        kept = []
        for name in split_names[split_name]:
            sample = sample_map.get(name)
            if sample is None:
                missing_from_split[split_name] += 1
                continue
            kept.append(sample)
        if args.max_samples_per_split > 0:
            kept = kept[: args.max_samples_per_split]
        split_samples[split_name] = kept
        print(
            f"[Split] {split_name}: usable={len(kept)}, "
            f"missing_from_split_file={missing_from_split[split_name]}"
        )

    if args.save_split_file:
        save_split_file(
            split_path=Path(args.save_split_file).resolve(),
            split_names=split_names,
            source_manifest=manifest_path,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )

    print("[Model] Building and loading weights...")
    model = LipReadingCTC(num_chars=NUM_CHARS)
    dummy = np.zeros((1, MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.float32)
    _ = model(dummy, training=False)
    model.load_weights(str(model_path))

    metrics_by_split: Dict[str, Dict[str, float | int]] = {}
    for split_name in ("train", "valid", "test"):
        print(f"[Eval] {split_name}...")
        metrics_by_split[split_name] = evaluate_split(
            model=model,
            split_name=split_name,
            samples=split_samples[split_name],
            batch_size=args.batch_size,
            verbose_every=args.verbose_every,
        )
        m = metrics_by_split[split_name]
        print(
            f"[Result] {split_name}: samples={m['samples']}, acc={m['accuracy']:.4f}, "
            f"wer={m['wer']:.4f}, cer={m['cer']:.4f}"
        )

    overall = aggregate_metrics(metrics_by_split)
    print(
        f"[Result] overall: samples={overall['samples']}, acc={overall['accuracy']:.4f}, "
        f"wer={overall['wer']:.4f}, cer={overall['cer']:.4f}"
    )

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "model_name": model_name,
        "model_path": str(model_path),
        "preprocessed_dir": str(preprocessed_dir),
        "manifest_path": str(manifest_path),
        "split_source": split_source,
        "split_ratios": {
            "train": args.train_ratio,
            "valid": args.valid_ratio,
            "test": round(1.0 - args.train_ratio - args.valid_ratio, 6),
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "missing_from_split_file": missing_from_split,
        "metrics": {
            "train": metrics_by_split["train"],
            "valid": metrics_by_split["valid"],
            "test": metrics_by_split["test"],
            "overall": overall,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Output] Saved metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
