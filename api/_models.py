"""Model loading and caching for CTC and word-level models."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from src import FRONTEND_MODELS, MODEL_VARIANTS, NUM_CHARS, LipReadingCTC, build_lipreading_ctc
from src.model import LegacyLipReadingCTC
from src.llm_postprocessor import LLMPostprocessor
from experiments.word_level_grid.model import LipReadingWordClassifier, build_lipreading_word_classifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CTC_PATH = ROOT_DIR / "checkpoints" / "best_ctc_model_conformer_lite_gap_proj.keras"
DEFAULT_WORD_DIR = ROOT_DIR / "checkpoints" / "word_level_grid"

_CTC_CACHE: dict[str, LipReadingCTC] = {}
_WORD_CACHE: dict[str, LipReadingWordClassifier] = {}
_LLM_CACHE: dict[tuple[str, str], LLMPostprocessor] = {}
_active_ctc_path: str | None = None


def normalize_ctc_path(model_path: str | None) -> str:
    if not model_path or not model_path.strip():
        return str(DEFAULT_CTC_PATH.resolve())
    p = Path(model_path.strip())
    return str(p if p.is_absolute() else (ROOT_DIR / p).resolve())


def get_active_ctc_path() -> str | None:
    return _active_ctc_path


def load_ctc_model(model_path: str) -> LipReadingCTC:
    global _active_ctc_path
    if model_path in _CTC_CACHE:
        _active_ctc_path = model_path
        return _CTC_CACHE[model_path]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model not found: {model_path}")

    def _infer(path: str) -> tuple[str, str]:
        stem = Path(path).stem.lower()
        for v in sorted(MODEL_VARIANTS, key=len, reverse=True):
            for f in sorted(FRONTEND_MODELS, key=len, reverse=True):
                if f != "flatten" and stem.endswith(f"_{v}_{f}"): return v, f
        for v in sorted(MODEL_VARIANTS, key=len, reverse=True):
            if stem.endswith(f"_{v}"): return v, "flatten"
        return "bigru", "flatten"

    inferred = _infer(model_path)
    candidates = [inferred] + [(v, f) for v in MODEL_VARIANTS for f in FRONTEND_MODELS if (v, f) != inferred]

    for variant, frontend in candidates:
        try:
            m = build_lipreading_ctc(model_variant=variant, frontend_model=frontend, num_chars=NUM_CHARS)
            _ = m(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
            m.load_weights(model_path)
            _CTC_CACHE[model_path] = m
            _active_ctc_path = model_path
            return m
        except Exception:
            continue

    # Legacy fallback
    try:
        m = LegacyLipReadingCTC(num_chars=NUM_CHARS)
        _ = m(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
        m.load_weights(model_path)
        _CTC_CACHE[model_path] = m
        _active_ctc_path = model_path
        return m
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model: {exc}")


def load_word_model(model_path: str | None) -> LipReadingWordClassifier:
    resolved = model_path.strip() if model_path and model_path.strip() else None
    if not resolved:
        h5_files = sorted(DEFAULT_WORD_DIR.glob("*.weights.h5")) if DEFAULT_WORD_DIR.is_dir() else []
        if not h5_files:
            raise HTTPException(status_code=404, detail="No word checkpoint in checkpoints/word_level_grid/")
        resolved = str(h5_files[-1])
    if not Path(resolved).is_absolute():
        resolved = str((ROOT_DIR / resolved).resolve())

    if resolved in _WORD_CACHE:
        return _WORD_CACHE[resolved]
    if not os.path.exists(resolved):
        raise HTTPException(status_code=400, detail=f"Word model not found: {resolved}")

    def _infer(path: str) -> tuple[str, str]:
        stem = Path(path).stem.lower()
        variant, frontend = "conformer_lite", "flatten"
        for v in sorted(MODEL_VARIANTS, key=len, reverse=True):
            if f"mv-{v.replace('_','-')}" in stem or f"_{v}" in stem:
                variant = v; break
        for f in sorted(FRONTEND_MODELS, key=len, reverse=True):
            if f"fe-{f.replace('_','-')}" in stem or f"_{f}" in stem:
                frontend = f; break
        return variant, frontend

    variant, frontend = _infer(resolved)
    candidates = [(variant, frontend)] + [(v, f) for v in MODEL_VARIANTS for f in FRONTEND_MODELS if (v, f) != (variant, frontend)]

    for v, f in candidates:
        try:
            m = build_lipreading_word_classifier(model_variant=v, frontend_model=f)
            _ = m(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
            m.load_weights(resolved)
            _WORD_CACHE[resolved] = m
            return m
        except Exception:
            continue

    raise HTTPException(status_code=500, detail=f"Could not load word model: {resolved}")


def get_or_create_postprocessor(api_key: str | None, model: str) -> LLMPostprocessor:
    key = (api_key or os.environ.get("GEMINI_API_KEY", ""), model)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = LLMPostprocessor(api_key=api_key, model=model)
    return _LLM_CACHE[key]


def ctc_model_loaded() -> bool:
    return str(DEFAULT_CTC_PATH.resolve()) in _CTC_CACHE
