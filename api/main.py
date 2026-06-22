"""ImConvo inference API — routes only.

Helpers live in _utils.py (pure functions) and _models.py (model cache).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import api._game as game
from experiments.isolated_word_level.common import SLOT_VOCABS

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
    DEFAULT_BEAM_WIDTH,
    DEFAULT_DEBUG_TOP_K,
    DEFAULT_DECODER_MODE,
    DEFAULT_NGRAM_ALPHA,
    extract_lip_frames,
    list_decoder_specs,
    parse_alignment_text,
    decode_logits,
)
from src.llm_postprocessor import LLMPostprocessor
from experiments.word_level_grid.common import SLOT_NAMES, decode_word_indices

from api._utils import (
    PREVIEW_DIR,
    compute_wer,
    compute_cer,
    get_device_specs,
    get_video_metadata,
    build_preview,
    encode_sample_frames,
    token_from_index,
)
from api._models import (
    DEFAULT_CTC_PATH,
    normalize_ctc_path,
    load_ctc_model,
    load_word_model,
    get_or_create_postprocessor,
    ctc_model_loaded,
    get_active_ctc_path,
)

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
        load_ctc_model(str(DEFAULT_CTC_PATH.resolve()))
    except Exception:
        pass
    yield


app = FastAPI(title="ImConvo Inference API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/preview", StaticFiles(directory=str(PREVIEW_DIR)), name="preview")


# Shared helpers (route-level logic not belonging in _utils or _models)

def _resolve_reference_text(file_name: str, expected_text: str | None) -> tuple[str | None, str]:
    if expected_text and expected_text.strip():
        return expected_text.strip().lower(), "manual"

    stem = Path(file_name).stem
    patterns = [
        ROOT_DIR / "data" / "preprocessed" / "align" / f"{stem}.align",
        ROOT_DIR / "data" / "align" / f"{stem}.align",
        ROOT_DIR / "data" / "**" / "align" / f"{stem}.align",
        ROOT_DIR / "data" / "**" / "align" / f"*_{stem}.align",
    ]
    candidates = sorted(set(m for p in patterns for m in glob.glob(str(p), recursive=True)))
    if not candidates:
        return None, "none"
    try:
        return parse_alignment_text(candidates[0]).lower(), "align_auto"
    except Exception:
        return None, "none"


def _resolve_example_path(example_name: str) -> Path:
    if not example_name or not example_name.strip():
        raise HTTPException(status_code=400, detail="Missing example_name.")
    safe = Path(example_name).name
    candidate = (EXAMPLE_DIR / safe).resolve()
    try:
        candidate.relative_to(EXAMPLE_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid example path.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Example not found: {safe}")
    return candidate


def _preprocess_video(
    video_path: str,
    file_name: str,
) -> tuple[np.ndarray, dict[str, Any], str | None, float, list[str]]:
    video_meta = get_video_metadata(video_path)
    if video_meta["frame_count"] is None or video_meta["width"] is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable video stream.")

    preview_url = build_preview(video_path, file_name)

    t0 = time.perf_counter()
    frames = extract_lip_frames(video_path)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    if frames is None:
        raise HTTPException(status_code=400, detail="Could not detect mouth/face region.")
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        raise HTTPException(status_code=400, detail="Invalid preprocessed frame tensor.")

    return frames, video_meta, preview_url, preprocess_ms, encode_sample_frames(frames)


def _run_ctc_inference(
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

    normalized = normalize_ctc_path(model_path)
    try:
        model = load_ctc_model(normalized)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    frames, video_meta, preview_url, preprocess_ms, crop_samples = _preprocess_video(video_path, file_name)
    input_tensor = np.expand_dims(frames, axis=(0, -1)).astype(np.float32)

    t0 = time.perf_counter()
    logits = model(input_tensor, training=False)
    decode_result = decode_logits(
        logits,
        mode=decoder_mode,
        beam_width=beam_width,
        debug_top_k=debug_top_k,
        ngram_alpha=DEFAULT_NGRAM_ALPHA,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    raw_indices = np.argmax(logits.numpy()[0], axis=-1).astype(int).tolist()
    raw_tokens = [token_from_index(i) for i in raw_indices]
    predicted_text = decode_result.final_text

    ref_text, ref_source = _resolve_reference_text(file_name, expected_text)
    wer = compute_wer(ref_text, predicted_text) if ref_text is not None else None
    cer = compute_cer(ref_text, predicted_text) if ref_text is not None else None

    llm_corrected: str | None = None
    llm_wer: float | None = None
    llm_cer: float | None = None
    if llm_postprocess:
        try:
            pp = get_or_create_postprocessor(gemini_api_key, llm_model)
            llm_corrected = pp.correct(predicted_text)
            if ref_text is not None:
                llm_wer = compute_wer(ref_text, llm_corrected)
                llm_cer = compute_cer(ref_text, llm_corrected)
        except Exception as exc:
            print(f"[LLM] Post-processing failed: {exc}")

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "predicted_text": predicted_text,
        "reference_text": ref_text,
        "reference_source": ref_source,
        "wer": wer,
        "cer": cer,
        "model_path_used": normalized,
        "latency_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "total": round(total_ms, 2),
        },
        "video_stats": {
            "filename": file_name,
            "size_bytes": content_size_bytes,
            **video_meta,
            "processed_shape": list(frames.shape),
        },
        "preview_url": preview_url,
        "decoder": {
            "mode": decode_result.mode,
            "label": decode_result.label,
            "beam_width": decode_result.beam_width,
            "final_text": decode_result.final_text,
            "hypotheses": decode_result.hypotheses,
            "metadata": decode_result.metadata,
        },
        "debug": {
            "raw_timestep_indices": raw_indices,
            "raw_timestep_tokens": raw_tokens,
            "raw_timestep_text": " ".join(raw_tokens),
            "collapsed_indices": decode_result.collapsed_indices,
            "collapsed_text": decode_result.final_text,
            "decoder_top_k": decode_result.debug_top_k,
        },
        "crop_samples": crop_samples,
        "device_specs": get_device_specs(),
        "llm": {
            "corrected_text": llm_corrected,
            "wer": llm_wer,
            "cer": llm_cer,
            "model": llm_model if llm_postprocess else None,
        },
    }


def _run_word_inference(
    video_path: str,
    file_name: str,
    content_size_bytes: int,
    model_path: str | None,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    model = load_word_model(model_path)
    resolved = model_path  # load_word_model already resolved it; re-use path from response

    frames, video_meta, preview_url, preprocess_ms, crop_samples = _preprocess_video(video_path, file_name)
    input_tensor = tf.convert_to_tensor(np.expand_dims(frames, axis=(0, -1)).astype(np.float32))

    t0 = time.perf_counter()
    logits_list = model(input_tensor, training=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    slot_indices = [int(tf.argmax(sl, axis=-1).numpy()[0]) for sl in logits_list]
    slot_confidences = [float(tf.reduce_max(tf.nn.softmax(sl, axis=-1)).numpy()) for sl in logits_list]
    slot_words = decode_word_indices(slot_indices)
    predicted_sentence = " ".join(slot_words)

    ref_text, ref_source = _resolve_reference_text(file_name, None)
    wer = compute_wer(ref_text, predicted_sentence) if ref_text else None
    cer = compute_cer(ref_text, predicted_sentence) if ref_text else None

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "predicted_sentence": predicted_sentence,
        "slot_predictions": [
            {"slot": SLOT_NAMES[i], "word": slot_words[i], "confidence": slot_confidences[i]}
            for i in range(len(SLOT_NAMES))
        ],
        "reference_text": ref_text,
        "reference_source": ref_source,
        "wer": wer,
        "cer": cer,
        "model_path_used": str(model_path or ""),
        "latency_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "total": round(total_ms, 2),
        },
        "video_stats": {
            "filename": file_name,
            "size_bytes": content_size_bytes,
            **video_meta,
            "processed_shape": list(frames.shape),
        },
        "preview_url": preview_url,
        "crop_samples": crop_samples,
        "device_specs": get_device_specs(),
    }


# Routes

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": ctc_model_loaded(),
        "active_model_path": get_active_ctc_path(),
        "tf_version": tf.__version__,
        "device_used": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


@app.get("/decoders")
def list_decoders() -> dict[str, Any]:
    from src import DEFAULT_DECODER_MODE
    return {"default_mode": DEFAULT_DECODER_MODE, "decoders": list_decoder_specs()}


@app.get("/checkpoints")
def list_checkpoints() -> dict[str, Any]:
    ckpt_dir = ROOT_DIR / "checkpoints"
    files = sorted(
        p.name for p in ckpt_dir.iterdir()
        if p.is_file() and p.suffix in {".keras", ".h5", ".weights"}
    ) if ckpt_dir.exists() else []
    default = DEFAULT_CTC_PATH.name
    return {
        "default": default,
        "checkpoints": [
            {"name": f, "path": f"checkpoints/{f}", "is_default": f == default}
            for f in files
        ],
    }


@app.get("/examples")
def list_examples(limit: int = 100) -> dict[str, Any]:
    limit = max(1, min(limit, 1000))
    if not EXAMPLE_DIR.exists():
        return {"base_dir": str(EXAMPLE_DIR), "count": 0, "examples": []}
    files = sorted(
        p.name for p in EXAMPLE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".webm"}
    )
    selected = files[:limit]
    return {"base_dir": str(EXAMPLE_DIR), "count": len(selected), "examples": selected}


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
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name
    try:
        return _run_ctc_inference(
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
    path = _resolve_example_path(example_name)
    return _run_ctc_inference(
        video_path=str(path),
        file_name=path.name,
        content_size_bytes=path.stat().st_size,
        model_path=model_path,
        expected_text=expected_text,
        decoder_mode=decoder_mode,
        beam_width=beam_width,
        debug_top_k=debug_top_k,
        llm_postprocess=llm_postprocess,
        gemini_api_key=gemini_api_key,
        llm_model=llm_model,
    )


@app.post("/analyze-word")
async def analyze_word(
    file: UploadFile = File(...),
    model_path: str | None = Form(default=None),
) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name
    try:
        return _run_word_inference(
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
    path = _resolve_example_path(example_name)
    return _run_word_inference(
        video_path=str(path),
        file_name=path.name,
        content_size_bytes=path.stat().st_size,
        model_path=model_path,
    )


@app.websocket("/ws/game")
async def ws_game(ws: WebSocket) -> None:
    await ws.accept()
    buffer: deque = deque(maxlen=game.BUFFER_FRAMES)
    slot_index: int | None = None
    target: str | None = None
    frames_since_score = 0
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "ping":
                await ws.send_json({"type": "pong"})

            elif kind == "start_round":
                slot_index = int(msg.get("slot_index", -1))
                target = str(msg.get("target", ""))
                buffer.clear()
                frames_since_score = 0
                if not (0 <= slot_index < len(SLOT_VOCABS)) or target not in SLOT_VOCABS[slot_index]:
                    await ws.send_json({"type": "error", "message": "invalid slot or target"})
                    slot_index, target = None, None

            elif kind == "end_round":
                if slot_index is not None and target is not None and buffer:
                    await _score_and_emit(ws, list(buffer), slot_index, target, force_result=True)
                slot_index, target = None, None

            elif kind == "frame":
                if slot_index is None or target is None:
                    continue
                try:
                    buffer.append(game.decode_jpeg(msg.get("data", "")))
                except ValueError:
                    continue
                frames_since_score += 1
                if frames_since_score >= game.SCORE_EVERY_N_FRAMES:
                    frames_since_score = 0
                    passed = await _score_and_emit(ws, list(buffer), slot_index, target, force_result=False)
                    if passed:
                        slot_index, target = None, None
    except WebSocketDisconnect:
        return


async def _score_and_emit(ws: WebSocket, frames, slot_index: int, target: str, *, force_result: bool) -> bool:
    arr = game.preprocess_frames(frames)
    if arr is None:
        await ws.send_json({"type": "progress", "word": "", "confidence": 0.0, "topk": [], "face": False})
        if force_result:
            await ws.send_json({"type": "result", "pass": False, "word": "", "confidence": 0.0, "target": target})
        return False
    scored = game.score_window(arr, slot_index)
    passed = game.evaluate_pass(slot_index, target, scored["ranked_words"], scored["confidence"])
    if passed or force_result:
        await ws.send_json({
            "type": "result",
            "pass": passed,
            "word": scored["top_word"],
            "confidence": scored["confidence"],
            "target": target,
        })
    else:
        await ws.send_json({
            "type": "progress",
            "word": scored["top_word"],
            "confidence": scored["confidence"],
            "topk": scored["topk"],
            "face": True,
        })
    return passed


# Entry point

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImConvo Inference API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
