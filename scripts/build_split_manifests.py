"""
Build deterministic, leakage-proof split manifests for GRID.

This script creates:
  - train.txt
  - val_oos.txt
  - val_is.txt
  - test_oos.txt
  - test_is.txt
  - summary.json

Default policy (grid_v1):
  - Out-of-sample validation speaker: s5
  - Out-of-sample test speaker: s15
  - Pool 330 clips from remaining speakers with balanced sampling
  - Split pooled clips into 165/165 for val_is/test_is
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    val_oos_speaker: str = "s5"
    test_oos_speaker: str = "s15"
    pooled_total: int = 330
    val_is_count: int = 165
    test_is_count: int = 165


def _read_manifest(manifest_path: str) -> list[str]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return names


def _speaker_of(sample_id: str) -> str:
    # Expected format: sX_processed_<clip_id>
    if "_processed_" not in sample_id:
        raise ValueError(f"Unexpected sample ID format: {sample_id}")
    return sample_id.split("_processed_")[0]


def _group_by_speaker(sample_ids: list[str]) -> Dict[str, List[str]]:
    by_speaker: Dict[str, List[str]] = defaultdict(list)
    for sample_id in sample_ids:
        by_speaker[_speaker_of(sample_id)].append(sample_id)
    for speaker in by_speaker:
        by_speaker[speaker] = sorted(by_speaker[speaker])
    return dict(by_speaker)


def _sample_balanced_pool(
    by_speaker: Dict[str, List[str]],
    train_speakers: list[str],
    pooled_total: int,
    rng: np.random.RandomState,
) -> list[str]:
    if pooled_total <= 0:
        raise ValueError("pooled_total must be > 0")
    if pooled_total > sum(len(by_speaker[s]) for s in train_speakers):
        raise ValueError("pooled_total exceeds available train samples")

    base = pooled_total // len(train_speakers)
    remainder = pooled_total % len(train_speakers)

    selected: list[str] = []
    for idx, speaker in enumerate(train_speakers):
        needed = base + (1 if idx < remainder else 0)
        speaker_ids = list(by_speaker[speaker])
        if len(speaker_ids) < needed:
            raise ValueError(
                f"Speaker {speaker} has {len(speaker_ids)} samples, needs {needed}"
            )
        chosen_indices = rng.choice(len(speaker_ids), size=needed, replace=False)
        chosen_ids = [speaker_ids[i] for i in sorted(chosen_indices.tolist())]
        selected.extend(chosen_ids)

    if len(set(selected)) != pooled_total:
        raise ValueError("Balanced pool sampling created duplicate selections")
    return sorted(selected)


def _validate_splits(
    all_ids: list[str],
    splits: dict[str, list[str]],
    val_oos_speaker: str,
    test_oos_speaker: str,
):
    required = {"train", "val_oos", "val_is", "test_oos", "test_is"}
    if set(splits) != required:
        raise ValueError(f"Split keys mismatch: got {sorted(splits.keys())}")

    # Pairwise disjointness
    names = sorted(required)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left = names[i]
            right = names[j]
            overlap = set(splits[left]).intersection(splits[right])
            if overlap:
                raise ValueError(
                    f"Overlap between {left} and {right}: {len(overlap)} samples"
                )

    # Full coverage
    union_ids = set().union(*[set(v) for v in splits.values()])
    if union_ids != set(all_ids):
        missing = len(set(all_ids) - union_ids)
        extra = len(union_ids - set(all_ids))
        raise ValueError(
            f"Coverage mismatch: missing={missing}, extra={extra}, "
            f"union={len(union_ids)}, source={len(all_ids)}"
        )

    # Speaker purity checks for full OOS splits
    if any(_speaker_of(sid) != val_oos_speaker for sid in splits["val_oos"]):
        raise ValueError("val_oos contains non-val speaker samples")
    if any(_speaker_of(sid) != test_oos_speaker for sid in splits["test_oos"]):
        raise ValueError("test_oos contains non-test speaker samples")

    # Ensure no held-out speaker appears in train / in-sample splits
    forbidden = {val_oos_speaker, test_oos_speaker}
    for split_name in ("train", "val_is", "test_is"):
        speakers = {_speaker_of(sid) for sid in splits[split_name]}
        bad = speakers.intersection(forbidden)
        if bad:
            raise ValueError(
                f"{split_name} includes held-out speakers: {sorted(bad)}"
            )


def _write_list(path: str, values: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(value + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build deterministic GRID split manifests.")
    parser.add_argument(
        "--manifest",
        default="./data/preprocessed/manifest.txt",
        help="Path to source manifest.txt",
    )
    parser.add_argument(
        "--out_dir",
        default="./splits/grid_v1",
        help="Output directory for split files",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SplitConfig(seed=args.seed)
    if cfg.val_is_count + cfg.test_is_count != cfg.pooled_total:
        raise ValueError("val_is_count + test_is_count must equal pooled_total")

    all_ids = _read_manifest(args.manifest)
    by_speaker = _group_by_speaker(all_ids)

    if cfg.val_oos_speaker not in by_speaker:
        raise ValueError(f"Val speaker not found: {cfg.val_oos_speaker}")
    if cfg.test_oos_speaker not in by_speaker:
        raise ValueError(f"Test speaker not found: {cfg.test_oos_speaker}")
    if cfg.val_oos_speaker == cfg.test_oos_speaker:
        raise ValueError("Val and test OOS speakers must differ")

    all_speakers = sorted(by_speaker.keys(), key=lambda s: int(s[1:]))
    train_speakers = [
        s
        for s in all_speakers
        if s not in {cfg.val_oos_speaker, cfg.test_oos_speaker}
    ]

    rng = np.random.RandomState(cfg.seed)

    val_oos = sorted(by_speaker[cfg.val_oos_speaker])
    test_oos = sorted(by_speaker[cfg.test_oos_speaker])

    pooled_subset = _sample_balanced_pool(
        by_speaker=by_speaker,
        train_speakers=train_speakers,
        pooled_total=cfg.pooled_total,
        rng=rng,
    )
    pooled_subset = list(pooled_subset)
    rng.shuffle(pooled_subset)

    val_is = sorted(pooled_subset[: cfg.val_is_count])
    test_is = sorted(pooled_subset[cfg.val_is_count :])
    pooled_set = set(pooled_subset)

    train_candidates = []
    for speaker in train_speakers:
        train_candidates.extend(by_speaker[speaker])
    train = sorted([sid for sid in train_candidates if sid not in pooled_set])

    splits = {
        "train": train,
        "val_oos": val_oos,
        "val_is": val_is,
        "test_oos": test_oos,
        "test_is": test_is,
    }
    _validate_splits(
        all_ids=all_ids,
        splits=splits,
        val_oos_speaker=cfg.val_oos_speaker,
        test_oos_speaker=cfg.test_oos_speaker,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, split_ids in splits.items():
        _write_list(os.path.join(args.out_dir, f"{split_name}.txt"), split_ids)

    summary = {
        "version": "grid_v1",
        "seed": cfg.seed,
        "source_manifest": os.path.abspath(args.manifest),
        "val_oos_speaker": cfg.val_oos_speaker,
        "test_oos_speaker": cfg.test_oos_speaker,
        "train_speaker_count": len(train_speakers),
        "train_speakers": train_speakers,
        "counts": {k: len(v) for k, v in splits.items()},
        "total_samples": len(all_ids),
    }

    with open(
        os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("[Split] Successfully generated deterministic manifests.")
    for k, v in summary["counts"].items():
        print(f"  {k}: {v}")
    print(f"  total: {summary['total_samples']}")


if __name__ == "__main__":
    main()
