# GRID Word-Level TCN Adapter (Latency First)

## 1) Device Fallback
`main.py` now supports:
- `--device auto` (default)
- `--device cuda`
- `--device cpu`

This removes hard CUDA-only behavior and enables CPU fallback.

## 2) Build GRID Word-Level Clips
Source expected:
- `/home/nam/ImConvo/data/preprocessed/*.npy`
- `/home/nam/ImConvo/data/preprocessed/align/*.align`

Run:
```bash
python /home/nam/ImConvo/scripts/dataset/convert_grid_preprocessed_to_tcn_words.py \
  --input-dir /home/nam/ImConvo/data/preprocessed \
  --output-dir /home/nam/ImConvo/Lipreading_using_Temporal_Convolutional_Networks/datasets/visual_data_grid51
```

Outputs:
- `datasets/visual_data_grid51/<word>/<train|val|test>/*.npz`
- `datasets/visual_data_grid51/grid51_words.txt`
- `datasets/visual_data_grid51/metadata.csv`

`metadata.csv` columns:
`clip_id,speaker,word,split,frame_start,frame_end,path`

Split policy:
- `train`: `s1..s28`
- `val`: `s29..s32`
- `test`: `s33..s34`

## 3) Verify Dataset Artifacts
```bash
python /home/nam/ImConvo/scripts/dataset/verify_grid_tcn_word_dataset.py \
  --dataset-dir /home/nam/ImConvo/Lipreading_using_Temporal_Convolutional_Networks/datasets/visual_data_grid51 \
  --source-preprocessed-dir /home/nam/ImConvo/data/preprocessed
```

## 4) Live Latency Benchmark (Tiny Model)
```bash
cd /home/nam/ImConvo/Lipreading_using_Temporal_Convolutional_Networks
python live_latency_benchmark.py \
  --device auto \
  --model-path ./models/tiny_lrw_snv05x_tcn1x.pth \
  --config-path ./configs/lrw_snv05x_tcn1x.json \
  --buffer-size 29 \
  --infer-every 5 \
  --max-windows 200
```

Optional synthetic benchmark (no camera):
```bash
python live_latency_benchmark.py --synthetic --max-windows 200 --device auto --no-display
```
