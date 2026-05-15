"""Simple offline inference API for demo purposes.

This module is intentionally separate from inference.py (live inference).
It exposes:
  - GET /health
  - POST /analyze
"""

from __future__ import annotations

import argparse
import base64
from contextlib import asynccontextmanager
import glob
import io
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

# Load .env so GEMINI_API_KEY is available without manual export
_env_file = ROOT_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from src import (
    BLANK_IDX,
    CHAR_LIST,
    DEFAULT_BEAM_WIDTH,
    FRONTEND_MODELS,
    MODEL_VARIANTS,
    DEFAULT_DEBUG_TOP_K,
    DEFAULT_DECODER_MODE,
    DEFAULT_NGRAM_ALPHA,
    NUM_CHARS,
    SPACE_IDX,
    LipReadingCTC,
    build_lipreading_ctc,
    char_indices_to_text,
    decode_logits,
    extract_lip_frames,
    list_decoder_specs,
    parse_alignment_text,
)
from src.model import LegacyLipReadingCTC
from src.llm_postprocessor import LLMPostprocessor
from experiments.word_level_grid.model import (
    LipReadingWordClassifier,
    build_lipreading_word_classifier,
)
from experiments.word_level_grid.common import (
    SLOT_NAMES,
    SLOT_VOCAB_SIZES,
    decode_word_indices,
    indices_to_sentence,
)

DEFAULT_MODEL_PATH = ROOT_DIR / "checkpoints" / "best_ctc_model_conformer_lite_gap_proj.keras"
DEFAULT_WORD_MODEL_PATH = ROOT_DIR / "checkpoints" / "word_level_grid"
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
_WORD_MODEL_CACHE: dict[str, LipReadingWordClassifier] = {}
_ACTIVE_MODEL_PATH: str | None = None
_LLM_CACHE: dict[tuple[str, str], LLMPostprocessor] = {}


def _get_or_create_postprocessor(api_key: str | None, model: str) -> LLMPostprocessor:
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    cache_key = (resolved_key, model)
    if cache_key not in _LLM_CACHE:
        _LLM_CACHE[cache_key] = LLMPostprocessor(api_key=api_key, model=model)
    return _LLM_CACHE[cache_key]

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

    def infer_variant_and_frontend(path: str) -> tuple[str, str]:
        """Return (model_variant, frontend_model) inferred from checkpoint filename."""
        stem = Path(path).stem.lower()
        # Check combined variant+frontend suffixes first (most specific).
        for variant in sorted(MODEL_VARIANTS, key=len, reverse=True):
            for frontend in sorted(FRONTEND_MODELS, key=len, reverse=True):
                if frontend == "flatten":
                    # flatten is the default — only match it if no other frontend matches
                    continue
                if stem.endswith(f"_{variant}_{frontend}"):
                    return variant, frontend
        # Check variant-only suffixes.
        for variant in sorted(MODEL_VARIANTS, key=len, reverse=True):
            if stem.endswith(f"_{variant}"):
                return variant, "flatten"
        # Legacy default.
        return "bigru", "flatten"

    inferred_variant, inferred_frontend = infer_variant_and_frontend(model_path)
    # Try inferred combination first, then all (variant, frontend) combos as fallback.
    candidates: list[tuple[str, str]] = [(inferred_variant, inferred_frontend)]
    for v in MODEL_VARIANTS:
        for f in FRONTEND_MODELS:
            if (v, f) not in candidates:
                candidates.append((v, f))

    load_errors: list[str] = []
    model = None
    for variant, frontend in candidates:
        try:
            candidate_model = build_lipreading_ctc(
                model_variant=variant, frontend_model=frontend, num_chars=NUM_CHARS
            )
            _ = candidate_model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
            candidate_model.load_weights(model_path)
            model = candidate_model
            print(f"[DemoAPI] Loaded model '{model_path}' with variant='{variant}' frontend='{frontend}'.")
            break
        except Exception as exc:
            load_errors.append(f"{variant}/{frontend}: {type(exc).__name__}: {exc}")

    if model is None:
        # Legacy fallback for checkpoints saved before backbone refactor.
        try:
            legacy_model = LegacyLipReadingCTC(num_chars=NUM_CHARS)
            _ = legacy_model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
            legacy_model.load_weights(model_path)
            model = legacy_model
            print(f"[DemoAPI] Loaded legacy model '{model_path}' with LegacyLipReadingCTC.")
        except Exception as legacy_exc:
            load_errors.append(f"legacy_bigru: {type(legacy_exc).__name__}: {legacy_exc}")
            joined = " | ".join(load_errors[:5])
            tried = [f"{v}/{f}" for v, f in candidates] + ["legacy_bigru"]
            raise RuntimeError(
                "Could not load checkpoint with any known variant. "
                f"Tried {tried}. Errors: {joined}"
            )

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


def _encode_sample_frames(frames: np.ndarray, n: int = 6) -> list[str]:
    """Return n evenly-spaced frames as base64-encoded PNG data URIs."""
    T = frames.shape[0]
    indices = [int(i * (T - 1) / (n - 1)) for i in range(n)]
    result: list[str] = []
    for idx in indices:
        gray = (frames[idx] * 255).astype(np.uint8)
        # scale up 4x for visibility (120x80 → 480x320)
        display = cv2.resize(gray, (480, 320), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode(".png", display)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        result.append(f"data:image/png;base64,{b64}")
    return result


def _run_inference_from_video_path(
    video_path: str,
    file_name: str,
    content_size_bytes: int,
    model_path: str | None,
    expected_text: str | None,
    decoder_mode: str,
    beam_width: int,
    debug_top_k: int,
    llm_postprocess: bool = False,
    gemini_api_key: str | None = None,
    llm_model: str = LLMPostprocessor.DEFAULT_MODEL,
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

    crop_samples = _encode_sample_frames(frames, n=6)
    input_tensor = np.expand_dims(frames, axis=(0, -1)).astype(np.float32)

    inference_start = time.perf_counter()
    logits = model(input_tensor, training=False)
    decode_result = decode_logits(
        logits,
        mode=decoder_mode,
        beam_width=beam_width,
        debug_top_k=debug_top_k,
        ngram_alpha=DEFAULT_NGRAM_ALPHA,
    )
    inference_ms = (time.perf_counter() - inference_start) * 1000

    raw_timestep_indices = np.argmax(logits.numpy()[0], axis=-1).astype(int).tolist()
    raw_timestep_tokens = [_token_from_index(i) for i in raw_timestep_indices]
    predicted_text = decode_result.final_text

    reference_text, reference_source = _resolve_reference_text(file_name, expected_text)
    wer = compute_wer(reference_text, predicted_text) if reference_text is not None else None
    cer = compute_cer(reference_text, predicted_text) if reference_text is not None else None

    # Optional LLM post-processing
    llm_corrected_text: str | None = None
    llm_wer: float | None = None
    llm_cer: float | None = None
    if llm_postprocess:
        try:
            postprocessor = _get_or_create_postprocessor(gemini_api_key, llm_model)
            llm_corrected_text = postprocessor.correct(predicted_text)
            if reference_text is not None:
                llm_wer = compute_wer(reference_text, llm_corrected_text)
                llm_cer = compute_cer(reference_text, llm_corrected_text)
        except Exception as llm_exc:
            print(f"[LLM] Post-processing failed: {llm_exc}")

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
        "decoder": {
            "mode": decode_result.mode,
            "label": decode_result.label,
            "beam_width": decode_result.beam_width,
            "final_text": decode_result.final_text,
            "hypotheses": decode_result.hypotheses,
            "metadata": decode_result.metadata,
        },
        "debug": {
            "raw_timestep_indices": raw_timestep_indices,
            "raw_timestep_tokens": raw_timestep_tokens,
            "raw_timestep_text": " ".join(raw_timestep_tokens),
            "collapsed_indices": decode_result.collapsed_indices,
            "collapsed_text": decode_result.final_text,
            "decoder_top_k": decode_result.debug_top_k,
        },
        "crop_samples": crop_samples,
        "device_specs": get_device_specs(),
        "llm": {
            "corrected_text": llm_corrected_text,
            "wer": llm_wer,
            "cer": llm_cer,
            "model": llm_model if llm_postprocess else None,
        },
    }


def _get_or_load_word_model(model_path: str) -> LipReadingWordClassifier:
    if model_path in _WORD_MODEL_CACHE:
        return _WORD_MODEL_CACHE[model_path]

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Word model path not found: {model_path}")

    def _infer_word_variant_frontend(path: str) -> tuple[str, str]:
        stem = Path(path).stem.lower()
        variant = "conformer_lite"
        frontend = "flatten"
        for v in sorted(MODEL_VARIANTS, key=len, reverse=True):
            if f"mv-{v.replace('_', '-')}" in stem or f"_{v}" in stem:
                variant = v
                break
        for f in sorted(FRONTEND_MODELS, key=len, reverse=True):
            if f"fe-{f.replace('_', '-')}" in stem or f"_{f}" in stem:
                frontend = f
                break
        return variant, frontend

    variant, frontend = _infer_word_variant_frontend(model_path)
    candidates: list[tuple[str, str]] = [(variant, frontend)]
    for v in MODEL_VARIANTS:
        for f in FRONTEND_MODELS:
            if (v, f) not in candidates:
                candidates.append((v, f))

    for v, f in candidates:
        try:
            m = build_lipreading_word_classifier(model_variant=v, frontend_model=f)
            _ = m(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
            m.load_weights(model_path)
            _WORD_MODEL_CACHE[model_path] = m
            print(f"[DemoAPI] Loaded word model '{model_path}' variant='{v}' frontend='{f}'.")
            return m
        except Exception:
            continue

    raise HTTPException(status_code=500, detail=f"Could not load word model: {model_path}")


def _resolve_default_word_checkpoint() -> str:
    if DEFAULT_WORD_MODEL_PATH.is_dir():
        h5_files = sorted(DEFAULT_WORD_MODEL_PATH.glob("*.weights.h5"))
        if h5_files:
            return str(h5_files[-1])
    raise HTTPException(status_code=404, detail="No word-level checkpoint found in checkpoints/word_level_grid/")


def _run_word_inference_from_video_path(
    video_path: str,
    file_name: str,
    content_size_bytes: int,
    model_path: str | None,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    resolved_path = model_path.strip() if model_path and model_path.strip() else None
    if not resolved_path:
        resolved_path = _resolve_default_word_checkpoint()
    if not Path(resolved_path).is_absolute():
        resolved_path = str((ROOT_DIR / resolved_path).resolve())

    model = _get_or_load_word_model(resolved_path)

    video_meta = _get_video_metadata(video_path)
    if video_meta["frame_count"] is None or video_meta["width"] is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable video stream.")

    preview_path = _build_preview_file(video_path, file_name)

    preprocess_start = time.perf_counter()
    frames = extract_lip_frames(video_path)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    if frames is None:
        raise HTTPException(status_code=400, detail="Could not detect mouth/face region.")
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        raise HTTPException(status_code=400, detail="Invalid preprocessed frame tensor.")

    crop_samples = _encode_sample_frames(frames, n=6)
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(frames, axis=(0, -1)).astype(np.float32)
    )

    inference_start = time.perf_counter()
    logits_list = model(input_tensor, training=False)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    slot_indices = [int(tf.argmax(slot_logits, axis=-1).numpy()[0]) for slot_logits in logits_list]
    slot_confidences = [float(tf.reduce_max(tf.nn.softmax(slot_logits, axis=-1)).numpy()) for slot_logits in logits_list]
    slot_words = decode_word_indices(slot_indices)
    predicted_sentence = " ".join(slot_words)

    reference_text, reference_source = _resolve_reference_text(file_name, None)
    wer = compute_wer(reference_text, predicted_sentence) if reference_text else None
    cer = compute_cer(reference_text, predicted_sentence) if reference_text else None

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "predicted_sentence": predicted_sentence,
        "slot_predictions": [
            {"slot": SLOT_NAMES[i], "word": slot_words[i], "confidence": slot_confidences[i]}
            for i in range(len(SLOT_NAMES))
        ],
        "reference_text": reference_text,
        "reference_source": reference_source,
        "wer": wer,
        "cer": cer,
        "model_path_used": resolved_path,
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
        "crop_samples": crop_samples,
        "device_specs": get_device_specs(),
    }


@app.post("/analyze-word")
async def analyze_word(
    file: UploadFile = File(...),
    model_path: str | None = Form(default=None),
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        content = await file.read()
        tmp.write(content)
    try:
        return _run_word_inference_from_video_path(
            video_path=temp_path,
            file_name=file.filename or "upload.bin",
            content_size_bytes=len(content),
            model_path=model_path,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@app.post("/analyze-word-example")
def analyze_word_example(
    example_name: str = Form(...),
    model_path: str | None = Form(default=None),
) -> dict[str, Any]:
    example_path = _resolve_example_path(example_name)
    return _run_word_inference_from_video_path(
        video_path=str(example_path),
        file_name=example_path.name,
        content_size_bytes=example_path.stat().st_size,
        model_path=model_path,
    )


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


@app.get("/decoders")
def list_decoders() -> dict[str, Any]:
    return {"default_mode": DEFAULT_DECODER_MODE, "decoders": list_decoder_specs()}


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
    decoder_mode: str = Form(default=DEFAULT_DECODER_MODE),
    beam_width: int = Form(default=DEFAULT_BEAM_WIDTH),
    debug_top_k: int = Form(default=DEFAULT_DEBUG_TOP_K),
    llm_postprocess: bool = Form(default=False),
    gemini_api_key: str | None = Form(default=None),
    llm_model: str = Form(default=LLMPostprocessor.DEFAULT_MODEL),
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
            decoder_mode=decoder_mode,
            beam_width=beam_width,
            debug_top_k=debug_top_k,
            llm_postprocess=llm_postprocess,
            gemini_api_key=gemini_api_key,
            llm_model=llm_model,
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
    decoder_mode: str = Form(default=DEFAULT_DECODER_MODE),
    beam_width: int = Form(default=DEFAULT_BEAM_WIDTH),
    debug_top_k: int = Form(default=DEFAULT_DEBUG_TOP_K),
    llm_postprocess: bool = Form(default=False),
    gemini_api_key: str | None = Form(default=None),
    llm_model: str = Form(default=LLMPostprocessor.DEFAULT_MODEL),
) -> dict[str, Any]:
    example_path = _resolve_example_path(example_name)
    return _run_inference_from_video_path(
        video_path=str(example_path),
        file_name=example_path.name,
        content_size_bytes=example_path.stat().st_size,
        model_path=model_path,
        expected_text=expected_text,
        decoder_mode=decoder_mode,
        beam_width=beam_width,
        debug_top_k=debug_top_k,
        llm_postprocess=llm_postprocess,
        gemini_api_key=gemini_api_key,
        llm_model=llm_model,
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
