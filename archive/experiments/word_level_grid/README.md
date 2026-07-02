# Word-Level GRID Experiment

This folder is an isolated experiment track that keeps the current CTC/character flow untouched.

## What is different

- Label unit: `word` (6 fixed GRID slots) instead of per-frame characters.
- Loss: slot-wise cross-entropy (6 heads) instead of CTC loss.
- Decoding: per-slot argmax (`command color preposition letter digit adverb`).

## Scripts

- Train:
  - `python experiments/word_level_grid/train_word.py --model-variant bigru --frontend-model flatten --augmentation-profile off`
- Evaluate:
  - `python experiments/word_level_grid/test_word.py --checkpoint-path <path_to_weights.h5> --model-variant bigru`
- Log experiment row (same style as `experiments.jsonl`, separate file):
  - `python experiments/word_level_grid/log_experiment_word.py`

## Output locations

- Checkpoints/history:
  - `checkpoints/word_level_grid/`
- Eval reports:
  - `reports/eval_result_word/`
- JSONL records:
  - `experiments_word_level.jsonl`
