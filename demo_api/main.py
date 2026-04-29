"""Simple offline inference API for demo purposes.

This module is intentionally separate from inference.py (live inference).
It exposes:
  - GET /health
  - POST /analyze
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import (
    BLANK_IDX,
    CHAR_LIST,
    NUM_CHARS,
    SPACE_IDX,
    LipReadingCTC,
    char_indices_to_text,
    extract_lip_frames,
    parse_alignment_text,
)

DEFAULT_MODEL_PATH = ROOT_DIR / "checkpoints" / "best_ctc_model.keras"
PREVIEW_DIR = ROOT_DIR / "demo_api" / "preview_cache"
EXAMPLE_DIR = ROOT_DIR / "data" / "s3_processed"
LOCAL_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3100",
    "http://127.0.0.1:3100",
]

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _get_or_load_model(str(DEFAULT_MODEL_PATH.resolve()))
    except Exception:
        # Health endpoint will still respond with model_loaded=false.
        pass
    yield


app = FastAPI(title="ImConvo Demo Inference API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_CACHE: dict[str, LipReadingCTC] = {}
_ACTIVE_MODEL_PATH: str | None = None

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/preview", StaticFiles(directory=str(PREVIEW_DIR)), name="preview")


def _compute_levenshtein(reference: list[str], hypothesis: list[str]) -> int:
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
                    d[i - 1][j - 1] + 1,
                )
    return d[r][h]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    edits = _compute_levenshtein(ref_words, hyp_words)
    return edits / max(len(ref_words), 1)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    edits = _compute_levenshtein(ref_chars, hyp_chars)
    return edits / max(len(ref_chars), 1)


def _normalize_model_path(model_path: str | None) -> str:
    if not model_path or not model_path.strip():
        return str(DEFAULT_MODEL_PATH.resolve())

    p = Path(model_path.strip())
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    return str(p)


def _get_or_load_model(model_path: str) -> LipReadingCTC:
    global _ACTIVE_MODEL_PATH

    if model_path in _MODEL_CACHE:
        _ACTIVE_MODEL_PATH = model_path
        return _MODEL_CACHE[model_path]

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")

    model = LipReadingCTC(num_chars=NUM_CHARS)
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
    model.load_weights(model_path)

    _MODEL_CACHE[model_path] = model
    _ACTIVE_MODEL_PATH = model_path
    return model


def _resolve_reference_text(file_name: str, expected_text: str | None) -> tuple[str | None, str]:
    if expected_text and expected_text.strip():
        return expected_text.strip().lower(), "manual"

    stem = Path(file_name).stem
    search_patterns = [
        ROOT_DIR / "data" / "preprocessed" / "align" / f"{stem}.align",
        ROOT_DIR / "data" / "align" / f"{stem}.align",
        ROOT_DIR / "data" / "**" / "align" / f"{stem}.align",
        ROOT_DIR / "data" / "**" / "align" / f"*_{stem}.align",
    ]

    candidates: list[str] = []
    for pattern in search_patterns:
        matches = glob.glob(str(pattern), recursive=True)
        candidates.extend(matches)

    unique_candidates = sorted(set(candidates))
    if not unique_candidates:
        return None, "none"

    try:
        return parse_alignment_text(unique_candidates[0]).lower(), "align_auto"
    except Exception:
        return None, "none"


def _resolve_example_path(example_name: str) -> Path:
    if not example_name or not example_name.strip():
        raise HTTPException(status_code=400, detail="Missing example_name.")

    safe_name = Path(example_name).name
    candidate = (EXAMPLE_DIR / safe_name).resolve()
    try:
        candidate.relative_to(EXAMPLE_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid example path.") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Example not found: {safe_name}")

    return candidate


def _get_video_metadata(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {
            "width": None,
            "height": None,
            "fps": None,
            "frame_count": None,
            "duration_sec": None,
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_raw) if fps_raw and fps_raw > 0 else None
    frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_raw) if frame_count_raw and frame_count_raw > 0 else None
    cap.release()

    duration_sec = None
    if fps and frame_count:
        duration_sec = frame_count / fps

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def _build_preview_file(temp_path: str, original_name: str | None) -> str | None:
    """Create a browser-playable mp4 preview and return its relative URL path."""
    suffix = Path(original_name or "").suffix.lower()
    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = PREVIEW_DIR / out_name

    # If already mp4, copy for stable serving URL
    if suffix == ".mp4":
        try:
            shutil.copyfile(temp_path, out_path)
            return f"/preview/{out_name}"
        except Exception:
            return None

    # Convert other formats (e.g. mpg) to mp4 via ffmpeg
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            temp_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out_path),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        if out.returncode == 0 and out_path.exists():
            return f"/preview/{out_name}"
    except Exception:
        return None
    return None


def _get_cpu_model() -> str | None:
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _get_gpu_specs() -> tuple[list[str], int | None]:
    gpu_names = []
    gpu_mem_total_mb: int | None = None

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name") or gpu.name)
        except Exception:
            gpu_names.append(gpu.name)

    # Best-effort fallback query from nvidia-smi.
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            parsed_names = []
            parsed_mem = []
            for line in out.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    parsed_names.append(parts[0])
                    if parts[1].isdigit():
                        parsed_mem.append(int(parts[1]))

            if parsed_names and not gpu_names:
                gpu_names = parsed_names
            if parsed_mem:
                gpu_mem_total_mb = sum(parsed_mem)
    except Exception:
        gpu_mem_total_mb = None

    return gpu_names, gpu_mem_total_mb


def get_device_specs() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    gpu_names, gpu_mem_total_mb = _get_gpu_specs()

    return {
        "cpu_model": _get_cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "gpu_names": gpu_names,
        "gpu_memory_total_mb": gpu_mem_total_mb,
        "tf_version": tf.__version__,
        "device_used": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


def _token_from_index(idx: int) -> str:
    if idx == BLANK_IDX:
        return "<blank>"
    if idx == SPACE_IDX:
        return "<space>"
    if 0 <= idx < len(CHAR_LIST):
        return CHAR_LIST[idx]
    return f"<unk:{idx}>"


def _run_inference_from_video_path(
    video_path: str,
    file_name: str,
    content_size_bytes: int,
    model_path: str | None,
    expected_text: str | None,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    normalized_model_path = _normalize_model_path(model_path)
    try:
        model = _get_or_load_model(normalized_model_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    video_meta = _get_video_metadata(video_path)
    if video_meta["frame_count"] is None or video_meta["width"] is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable video stream.")

    preview_path = _build_preview_file(video_path, file_name)

    preprocess_start = time.perf_counter()
    frames = extract_lip_frames(video_path)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    if frames is None:
        raise HTTPException(status_code=400, detail="Could not detect mouth/face region from uploaded video.")
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        raise HTTPException(status_code=400, detail="Invalid preprocessed frame tensor.")

    input_tensor = np.expand_dims(frames, axis=(0, -1)).astype(np.float32)

    inference_start = time.perf_counter()
    logits = model(input_tensor, training=False)
    decoded = model.decode_greedy(logits)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    raw_timestep_indices = np.argmax(logits.numpy()[0], axis=-1).astype(int).tolist()
    raw_timestep_tokens = [_token_from_index(i) for i in raw_timestep_indices]

    pred_indices = decoded[0]
    pred_indices = pred_indices[pred_indices >= 0]
    predicted_text = char_indices_to_text(pred_indices.tolist()).lower()

    reference_text, reference_source = _resolve_reference_text(file_name, expected_text)
    wer = compute_wer(reference_text, predicted_text) if reference_text is not None else None
    cer = compute_cer(reference_text, predicted_text) if reference_text is not None else None

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "predicted_text": predicted_text,
        "reference_text": reference_text,
        "reference_source": reference_source,
        "wer": wer,
        "cer": cer,
        "model_path_used": normalized_model_path,
        "latency_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "total": round(total_ms, 2),
        },
        "video_stats": {
            "filename": file_name,
            "size_bytes": content_size_bytes,
            "width": video_meta["width"],
            "height": video_meta["height"],
            "fps": video_meta["fps"],
            "frame_count": video_meta["frame_count"],
            "duration_sec": video_meta["duration_sec"],
            "processed_shape": list(frames.shape),
        },
        "preview_url": preview_path,
        "debug": {
            "raw_timestep_indices": raw_timestep_indices,
            "raw_timestep_tokens": raw_timestep_tokens,
            "raw_timestep_text": " ".join(raw_timestep_tokens),
        },
        "device_specs": get_device_specs(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model_path = str(DEFAULT_MODEL_PATH.resolve())
    model_loaded = model_path in _MODEL_CACHE
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "active_model_path": _ACTIVE_MODEL_PATH,
        "tf_version": tf.__version__,
        "device_used": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


@app.get("/examples")
def list_examples(limit: int = 100) -> dict[str, Any]:
    if limit <= 0:
        limit = 1
    if limit > 1000:
        limit = 1000

    if not EXAMPLE_DIR.exists():
        return {"base_dir": str(EXAMPLE_DIR), "count": 0, "examples": []}

    files = sorted(
        [
            p.name
            for p in EXAMPLE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in {".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".webm"}
        ]
    )
    selected = files[:limit]
    return {
        "base_dir": str(EXAMPLE_DIR),
        "count": len(selected),
        "examples": selected,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_path: str | None = Form(default=None),
    expected_text: str | None = Form(default=None),
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        return _run_inference_from_video_path(
            video_path=temp_path,
            file_name=file.filename or "upload.bin",
            content_size_bytes=len(content),
            model_path=model_path,
            expected_text=expected_text,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/analyze-example")
def analyze_example(
    example_name: str = Form(...),
    model_path: str | None = Form(default=None),
    expected_text: str | None = Form(default=None),
) -> dict[str, Any]:
    example_path = _resolve_example_path(example_name)
    return _run_inference_from_video_path(
        video_path=str(example_path),
        file_name=example_path.name,
        content_size_bytes=example_path.stat().st_size,
        model_path=model_path,
        expected_text=expected_text,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImConvo Demo Inference API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run("demo_api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
