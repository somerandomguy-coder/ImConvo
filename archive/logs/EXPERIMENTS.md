# Experiment Tracker

Use this file as the quick visual summary, and keep detailed machine-readable records in `experiments.jsonl`.

## How To Log
1. Create/switch to your experiment branch (example: `exp/bilstm-backbone`).
2. Run training and evaluation.
3. Append one JSON object line to `experiments.jsonl`.
4. Add one row to the table below.

## Results Table

| run_id | date | branch | model_variant | base_checkpoint | split_version | batch_size | lr | val_oos_wer | test_oos_wer | test_is_wer | eval_report_path | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-30_baseline_resume_gridv1 | 2026-04-30 | main | LipReadingCTC-BiGRU-baseline | checkpoints/best_ctc_model.keras | grid_v1 | 48 | 3e-4 | TBC | TBC | TBC | reports/eval_result/eval_report_20260430_083646.txt | Template baseline row; replace TBC after metric extraction. |

## Suggested Run ID Format

`YYYY-MM-DD_<model>_<key_setting>_<split>`

Examples:
- `2026-05-01_bilstm_bs36_gridv1`
- `2026-05-02_transformer_lr1e4_gridv1`
