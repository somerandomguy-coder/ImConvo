# ImConvo
**Lip-reading AI developed by the Convobeo Team.**

ImConvo is a LipNet-based spatio-temporal model designed to interpret speech from visual lip movements. The project includes a full pipeline: preprocessing, training, evaluation, a FastAPI inference server, real-time inference, and a Next.js web UI.

---

## Quick Start (Linux / macOS)

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

The script creates a virtual environment, installs dependencies, optionally downloads and preprocesses data, sets up the frontend, and presents a menu to launch any component — including **Start All** which boots the API server in the background and the frontend in the foreground together (Ctrl+C stops both).

---

## Project Structure

```
ImConvo/
├── api/                    # FastAPI inference server (port 8001)
│   ├── main.py             # Routes: /analyze, /analyze-word, /health, …
│   ├── _models.py          # Model cache & loading helpers
│   └── _utils.py           # Pure helpers (metrics, preview, encoding)
├── checkpoints/            # Trained model weights (.keras / .h5)
├── data/                   # Raw GRID dataset + preprocessed .npy files
├── experiments/
│   ├── word_level_grid/    # Slot-based sentence classifier (word-level)
│   └── isolated_word_level/# Isolated word classifier & backbone pre-training
├── frontend/               # Next.js web application (port 3000)
├── notebooks/              # Exploratory analysis notebooks
├── reports/
│   ├── decoder_artifacts/  # N-gram LM artifacts (grid_word_trigram.json)
│   ├── eval_result/        # CTC eval logs (WER/CER)
│   └── plots/              # Loss curves and performance summaries
├── scripts/                # Data & utility scripts (see below)
├── splits/grid_v1/         # Train/val/test split manifests
├── src/                    # Core library
│   ├── dataset.py          # Data pipeline
│   ├── decoding.py         # Greedy, beam, and n-gram decoders
│   ├── llm_postprocessor.py# Gemini LLM post-correction
│   ├── model.py            # CTC model + all backbone variants
│   ├── utils.py            # Constants, preprocessing, alignment parsing
│   └── visualization.py
├── inference.py            # Real-time webcam inference
├── train.py                # CTC training loop
├── test.py                 # CTC evaluation
└── transplant.py           # Transfer backbone weights from isolated model
```

---

## Model Variants

The CTC model supports interchangeable temporal backbones and visual front-ends.

| Backbone | Flag |
| :--- | :--- |
| BiGRU (default) | `bigru` |
| GRU | `gru` |
| BiLSTM | `bilstm` |
| Transformer (small) | `transformer` |
| Transformer (medium) | `transformer_medium` |
| TCN | `tcn` |
| Conformer Lite | `conformer_lite` |

| Visual front-end | Flag |
| :--- | :--- |
| Flatten (default) | `flatten` |
| GAP + Projection | `gap_proj` |
| ResNet-18 | `resnet18` |

Checkpoint filenames encode the variant (e.g. `best_ctc_model_conformer_lite.keras`). The inference scripts infer the variant automatically from the filename.

---

## Usage

### API Server (primary interface)

```bash
python -m api.main                    # default: http://0.0.0.0:8001
python -m api.main --port 8002        # custom port
```

Key endpoints:

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Status, loaded model, TF version |
| `GET` | `/checkpoints` | List available checkpoints |
| `GET` | `/decoders` | List decoder modes |
| `GET` | `/examples` | List example videos |
| `POST` | `/analyze` | CTC inference on uploaded video |
| `POST` | `/analyze-example` | CTC inference on a server-side example |
| `POST` | `/analyze-word` | Word-level slot inference on uploaded video |
| `POST` | `/analyze-word-example` | Word-level inference on a server-side example |

### Decoder Modes

Pass `decoder_mode` to `/analyze`:

| Mode | Description |
| :--- | :--- |
| `greedy_ctc` (default) | Fast argmax CTC decoding |
| `beam_ctc` | Multi-hypothesis CTC beam search |
| `beam_ngram_grid` | Beam search reranked with GRID trigram LM |

### LLM Post-processing

Set `GEMINI_API_KEY` in a `.env` file at the project root, then pass `llm_postprocess=true` (API) or `--llm-postprocess` (inference.py).

```
# .env
GEMINI_API_KEY=your_key_here
```

### Training & Evaluation

```bash
python train.py                        # CTC training (bigru default)
python train.py --model-variant conformer_lite

python test.py                         # CTC evaluation → reports/eval_result/
python test.py --model-variant conformer_lite --checkpoint-path checkpoints/best_ctc_model_conformer_lite.keras

# Word-level experiment
python experiments/word_level_grid/train_word.py
python experiments/word_level_grid/test_word.py

# Isolated word experiment
python experiments/isolated_word_level/train_isolated.py
python experiments/isolated_word_level/test_isolated.py
```

### Real-time Inference

```bash
python inference.py                                          # laptop webcam
python inference.py --ip http://192.168.0.69:8080           # IP Webcam (Android/iOS)
python inference.py --ip 1                                   # camera index 1
python inference.py --checkpoint-path checkpoints/best_ctc_model_conformer_lite.keras
python inference.py --llm-postprocess --gemini-api-key <KEY>
```

Press **q** (with the video window focused) to stop. Keep lips inside the yellow bounding box; the bottom progress bar shows the 75-frame buffer fill.

### Data Scripts

| Script | Description |
| :--- | :--- |
| `scripts/download_dataset.py` | Download GRID corpus |
| `scripts/preprocess.py` | Convert `.mpg` → `.npy` (single-core) |
| `scripts/preprocess_multi_cores.py` | Convert `.mpg` → `.npy` (multi-core) |
| `scripts/build_split_manifests.py` | Generate train/val/test split files |
| `scripts/build_grid_ngram_lm.py` | Build GRID trigram LM for beam decoder |
| `scripts/check_data.py` | Validate preprocessed data integrity |

### Backbone Transplant

To initialize a CTC model with weights pre-trained on the isolated word task:

```bash
python transplant.py
```

---

## Frontend (Web UI)

The `frontend/` directory is a Next.js application that calls the API server.

### Prerequisites
- Node.js v18+
- pnpm

### Setup & Run

```bash
cd frontend
pnpm install
pnpm dev    # http://localhost:3000
```

The frontend expects the API server to be running on port 8001.

| Command | Description |
| :--- | :--- |
| `pnpm dev` | Development server with hot reload |
| `pnpm build` | Production build |
| `pnpm start` | Start production server |

---

## Camera Setup

### Option A: Smartphone (useful for WSL2)
WSL2 cannot easily access internal webcams. Use an IP Webcam app instead.

1. Install [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam) or iVCam (iOS).
2. Set resolution to `352 x 288` and frame rate to `25 FPS`.
3. Run: `python inference.py --ip http://<phone-ip>:8080`

### Option B: Direct USB / Integrated Camera

```bash
python inference.py --ip 0    # or 1, 2, … for other indices
```
