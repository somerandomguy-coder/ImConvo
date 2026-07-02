# Isolated Word-Level GRID Experiment

This experiment is fully isolated from CTC and sentence-level multi-head paths.

## Goal

Train on **word segments** (not whole sentence clips) by slicing existing preprocessed `.npy` clips using `.align` timestamps.

## What changes

- Input: isolated word clips sliced from full sentence clips
- Label: single word class over flattened GRID vocabulary (51 classes)
- Head: single classifier head after temporal global average pooling
- Loss: sparse cross-entropy
- Metrics: top-1 and top-5 accuracy

## Scripts

- Train:
  - `python experiments/isolated_word_level/train_isolated.py --model-variant bigru --frontend-model flatten --target-frames 25 --pad-mode repeat_last`
- Evaluate:
  - `python experiments/isolated_word_level/test_isolated.py --checkpoint-path <path_to_weights.h5>`
- Log result row:
  - `python experiments/isolated_word_level/log_isolated.py`

## Output locations

- Checkpoints/history:
  - `checkpoints/isolated_word_level/`
- Eval reports:
  - `reports/eval_result_isolated_word/`
- Experiment log:
  - `log_isolated.jsonl`

## Robustness notes

- Missing `.align` or `.npy` files are skipped and counted.
- Out-of-bounds alignment indices are clamped and tracked.
- Empty segments are repaired to at least 1 frame then padded.
- Word clips are fixed-length via temporal padding (`repeat_last` or `zeros`).
