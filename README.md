# ImConvo
**Production-Grade Spatio-Temporal Lip-Reading System — Convobeo Team**

ImConvo is a deep learning-based spatio-temporal lip-reading application designed to interpret English speech from visual mouth movements. The system features a custom visual front-end (Flatten/GAP) paired with a deep temporal backbone (Conformer/BiGRU) trained on the GRID corpus. 

This repository has been productionized and optimized for high-performance serverless deployment:
- **Frontend:** Next.js (TailwindCSS v4, React 19) ready to deploy on **Vercel**.
- **Backend:** FastAPI inference API ready to deploy on **Google Cloud Run** using a decoupled weights architecture with **Cloud Storage FUSE**.

---

## 📺 Project Demo
<video src="demo.mp4" width="100%" controls muted></video>

---

## 🏗️ Production Architecture (Decoupled Weights)

To keep the container footprint minimal and speed up deployment pipelines, model weights are decoupled from the Docker image. 

```
  ┌─────────────────────────────────────────────────────────┐
  │                   GOOGLE CLOUD PLATFORM                 │
  │                                                         │
  │  ┌─────────────────────────┐                            │
  │  │  Cloud Storage Bucket   │                            │
  │  │ (imconvo-model-weights) │                            │
  │  └────────────┬────────────┘                            │
  │               │ Natively Mounts via FUSE                │
  │               ▼ (Exposed as /code/checkpoints/)         │
  │  ┌───────────────────────────────────────────────────┐  │
  │  │ Cloud Run Serverless Instance                     │  │
  │  │ - 4GB RAM Allocated (Prevents TensorFlow OOM)     │  │
  │  │ - Pure Python App Code (Tiny Image Size)         │  │
  │  └───────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────┘
```

1. **Storage Setup:** Weights are stored in a Google Cloud Storage bucket (e.g., `imconvo-model-weights`).
2. **FUSE Volume Mount:** Cloud Run dynamically mounts the bucket onto `/code/checkpoints`. The FastAPI app reads `.keras` / `.h5` weights directly from the virtual folder as if it were a local disk, keeping the container image size under 1GB.

---

## 🚀 Quick Start (Local Run)

You can launch both the backend API and Next.js frontend concurrently using a single shell command:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Select **Option 3 (Start All)** to boot:
- The FastAPI API server on `http://127.0.0.1:8001`
- The Next.js Web UI on `http://localhost:3000`

---

## 📂 Project Structure

```
ImConvo/
├── api/                    # FastAPI inference server (port 8001)
│   ├── main.py             # HTTP Routes & Websocket Game server
│   ├── _game.py            # Word game backend evaluation logic
│   ├── _models.py          # Model lazy-loading cache helpers
│   └── _utils.py           # Preprocessing & encoding helpers
├── checkpoints/            # Model weight directories (FUSE mounted in production)
│   ├── best_ctc_model_conformer_lite_gap_proj.keras   # CTC Sentence Model
│   ├── word_level_grid/    # Grid Word Classifier Model
│   └── isolated_word_level/# Word Game Classifier Model
├── data/
│   └── s3_processed/       # Example videos & alignments for backend demo
├── frontend/               # Next.js web application (port 3000)
├── src/                    # Core lip-reading python package
│   ├── isolated_word/      # Isolated word model definitions (Game backbone)
│   ├── word_level/         # Word-level slot model definitions (Word API)
│   ├── model.py            # Sentence CTC model & frontends (Flatten, GAP)
│   ├── decoding.py         # CTC decoding (Greedy, Beam, N-gram Beam)
│   ├── llm_postprocessor.py# Gemini LLM post-correction
│   └── utils.py            # Lip cropping, bbox extraction, preprocessing helpers
├── archive/                # Archived training loops, data prep scripts, & eval logs
├── Dockerfile              # Lean backend Dockerfile for Cloud Run
└── setup_and_run.sh        # Developer bootstrap script
```

---

## 🔌 API Endpoints

The backend server exposes the following routes for video analysis and status checks:

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Status, loaded models, and TF version |
| `GET` | `/checkpoints` | List active weight checkpoints |
| `GET` | `/decoders` | List supported CTC decoder modes |
| `GET` | `/examples` | List example videos |
| `POST` | `/analyze` | CTC inference on uploaded video |
| `POST` | `/analyze-example` | CTC inference on a server-side example |
| `POST` | `/analyze-word` | Word-level slot inference on uploaded video |
| `POST` | `/analyze-word-example` | Word-level inference on a server-side example |
| `WS` | `/ws/game` | WebSocket endpoint driving the interactive sandbox game |

### Decoder Modes
Pass `decoder_mode` to `/analyze`:
- `greedy_ctc` (default) — Fast argmax CTC decoding.
- `beam_ctc` — Multi-hypothesis CTC beam search.
- `beam_ngram_grid` — Beam search reranked with GRID trigram language model.

### LLM Post-processing
Export your key in a `.env` file at the project root to enable Gemini-based error correction:
```env
GEMINI_API_KEY=your_gemini_api_key
```

---

## 🌐 Frontend (Next.js)

The `frontend/` directory houses the web UI client built with React 19, TailwindCSS v4, and Next.js 16.2.

### Local Development
```bash
cd frontend
pnpm install
pnpm dev
```

### Vercel Deployment
To deploy the frontend to Vercel, set the **Root Directory** to `frontend` and configure:
- `NEXT_PUBLIC_API_URL`: Your deployed FastAPI backend URL.
- `NEXT_PUBLIC_DEMO_API_URL`: Your deployed FastAPI backend URL.

---

## 🗄️ Historical Experiments Archive
All experimental scripts, splits, training routines, and performance reports are archived under `archive/` to keep the codebase clean and dedicated to the production application.
