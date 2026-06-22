# GRID Lip Game Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live, per-word webcam game where the player silently mouths the six words of a random GRID-grammar sentence, one word per round, recognized in real time over a WebSocket using the existing isolated-word conformer-lite classifier.

**Architecture:** A new FastAPI WebSocket endpoint (`/ws/game`) streams webcam frames into a per-connection rolling buffer, lip-crops + classifies the latest ~25 frames with slot-constrained scoring, and emits live confidence/result messages. A new Next.js `/game` page captures the webcam, streams frames, and runs all game/session state (sentence, round, score, streak) client-side. No training code or existing endpoints change.

**Tech Stack:** Python 3.12, FastAPI 0.116, TensorFlow/Keras, OpenCV, MediaPipe (backend); Next.js 15 + TypeScript + Tailwind (frontend). Tests: pytest + httpx (backend), vitest + @testing-library/react (frontend).

## Global Constraints

- Branch: `feat/game` (already checked out).
- Reuse, do not duplicate: `SLOT_VOCABS`, `WORD_TO_IDX`, `IDX_TO_WORD` from `experiments.isolated_word_level.common`; `slice_and_pad_word_clip` from `experiments.isolated_word_level.temporal_slicer`; `_mediapipe_lip_bbox`, `FRAME_HEIGHT` (80), `FRAME_WIDTH` (120) from `src.utils`.
- Model: `build_lipreading_isolated_word_classifier(model_variant="conformer_lite", frontend_model="flatten", num_word_classes=51)`; weights at `checkpoints/isolated_word_level/best_mv-conformer-lite_fe-flatten_be-conformer-lite_bs-48_t-25_20260514T072743Z_conformer-lite_flatten_conformer-lite_isolated_word_v1.weights.h5`.
- Slot order (GRID sentence positions): `("command","color","preposition","letter","digit","adverb")`.
- Tunables (single source of truth in `api/_game.py`): `BUFFER_FRAMES=25`, `TARGET_FRAMES=25`, `SCORE_EVERY_N_FRAMES=3`, `PASS_THRESHOLD=0.6`, `LETTER_SLOT_INDEX=3`, `LETTER_TOPK_PASS=3`.
- Game rules: 6 rounds/sentence, **no lives / no lose state**, unlimited retries per word, win on 6/6. Letters slot passes if target ∈ top-3; all other slots pass if top-1 == target AND confidence ≥ `PASS_THRESHOLD`.
- Frames on the wire are base64-encoded JPEG; all control messages are JSON text frames.
- Do not modify existing endpoints (`/analyze`, `/analyze-word`, etc.) or training code.

---

## File Structure

**Backend**
- Create `api/_game.py` — vocab/slot maps, pass-rule, frame decode/preprocess, classifier singleton, `score_window`.
- Modify `api/main.py` — add `/ws/game` WebSocket endpoint.
- Create `tests/test_game_rules.py`, `tests/test_game_preprocess.py`, `tests/test_game_scoring.py`, `tests/test_game_ws.py`.
- Modify `requirements.txt` — add `pytest`, `httpx`.

**Frontend** (under `frontend/`)
- Create `src/lib/gridGrammar.ts` — slot vocabs + `generateSentence()`.
- Create `src/hooks/useGame.ts` — game reducer + types.
- Create `src/hooks/useWebcam.ts`, `src/hooks/useGameSocket.ts`.
- Create `src/components/game/{WordPrompt,ConfidenceMeter,Hud,RoundFeedback,ResultScreen}.tsx`.
- Create `src/app/game/page.tsx`; modify `src/components/Navbar.tsx` (add link).
- Test tooling: `vitest.config.ts`, `vitest.setup.ts`, modify `package.json`; tests `src/lib/__tests__/gridGrammar.test.ts`, `src/hooks/__tests__/useGame.test.ts`.

---

## Task 1: Backend test tooling + game rules/vocab helpers

**Files:**
- Modify: `requirements.txt`
- Create: `api/_game.py`
- Test: `tests/test_game_rules.py`

**Interfaces:**
- Consumes: `SLOT_VOCABS`, `WORD_TO_IDX` from `experiments.isolated_word_level.common`.
- Produces:
  - `SLOT_NAMES: tuple[str, ...]` (6 names)
  - constants `BUFFER_FRAMES, TARGET_FRAMES, SCORE_EVERY_N_FRAMES, PASS_THRESHOLD, LETTER_SLOT_INDEX, LETTER_TOPK_PASS`
  - `slot_candidate_indices(slot_index: int) -> list[int]`
  - `evaluate_pass(slot_index: int, target: str, ranked_words: list[str], confidence: float) -> bool` (`ranked_words` are the slot's candidate words sorted by probability descending; `ranked_words[0]` is the top word; `confidence` is the top word's probability)

- [ ] **Step 1: Add backend test dependencies**

Append to `requirements.txt`:

```
pytest==8.3.4
httpx==0.28.1
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_game_rules.py`:

```python
from api._game import (
    SLOT_NAMES,
    LETTER_SLOT_INDEX,
    slot_candidate_indices,
    evaluate_pass,
)
from experiments.isolated_word_level.common import SLOT_VOCABS, WORD_TO_IDX


def test_slot_names_match_grid_positions():
    assert SLOT_NAMES == ("command", "color", "preposition", "letter", "digit", "adverb")
    assert len(SLOT_NAMES) == len(SLOT_VOCABS)


def test_slot_candidate_indices_color():
    color_idx = SLOT_NAMES.index("color")
    expected = [WORD_TO_IDX[w] for w in SLOT_VOCABS[color_idx]]
    assert slot_candidate_indices(color_idx) == expected
    assert len(slot_candidate_indices(color_idx)) == 4


def test_pass_normal_slot_requires_top1_and_threshold():
    # color slot (index 1): top word must equal target AND clear threshold
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], 0.9) is True
    assert evaluate_pass(1, "blue", ["blue", "green", "red", "white"], 0.4) is False
    assert evaluate_pass(1, "blue", ["green", "blue", "red", "white"], 0.9) is False


def test_pass_letter_slot_allows_top3():
    ranked = ["c", "e", "b", "a", "d"]
    assert LETTER_SLOT_INDEX == 3
    assert evaluate_pass(3, "b", ranked, 0.2) is True   # b is in top-3
    assert evaluate_pass(3, "a", ranked, 0.2) is False  # a is 4th
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_game_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api._game'`

- [ ] **Step 4: Write minimal implementation**

Create `api/_game.py`:

```python
"""Live per-word game: vocab/slot maps, pass-rule, preprocessing, scoring."""
from __future__ import annotations

from experiments.isolated_word_level.common import SLOT_VOCABS, WORD_TO_IDX

SLOT_NAMES: tuple[str, ...] = (
    "command",
    "color",
    "preposition",
    "letter",
    "digit",
    "adverb",
)

# Tunables — single source of truth.
BUFFER_FRAMES = 25
TARGET_FRAMES = 25
SCORE_EVERY_N_FRAMES = 3
PASS_THRESHOLD = 0.6
LETTER_SLOT_INDEX = 3
LETTER_TOPK_PASS = 3


def slot_candidate_indices(slot_index: int) -> list[int]:
    """Return the FLAT_VOCAB class indices for a given GRID slot."""
    return [WORD_TO_IDX[w] for w in SLOT_VOCABS[slot_index]]


def evaluate_pass(
    slot_index: int,
    target: str,
    ranked_words: list[str],
    confidence: float,
) -> bool:
    """Decide whether a scored window passes the round.

    ranked_words: the slot's candidate words sorted by probability desc.
    confidence:   probability mass of ranked_words[0] (the top word).
    """
    if slot_index == LETTER_SLOT_INDEX:
        return target in ranked_words[:LETTER_TOPK_PASS]
    return bool(ranked_words and ranked_words[0] == target and confidence >= PASS_THRESHOLD)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_game_rules.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt api/_game.py tests/test_game_rules.py
git commit -m "feat(game): backend game rules, slot maps, test tooling"
```

---

## Task 2: Frame decode + lip-crop preprocessing

**Files:**
- Modify: `api/_game.py`
- Test: `tests/test_game_preprocess.py`

**Interfaces:**
- Consumes: `_mediapipe_lip_bbox`, `FRAME_HEIGHT`, `FRAME_WIDTH` from `src.utils`; `slice_and_pad_word_clip` from `experiments.isolated_word_level.temporal_slicer`; constant `TARGET_FRAMES`.
- Produces:
  - `decode_jpeg(b64: str) -> np.ndarray` — returns a BGR `uint8` `(H, W, 3)` frame; raises `ValueError` on undecodable input.
  - `preprocess_frames(bgr_frames: list[np.ndarray]) -> np.ndarray | None` — returns float32 `(1, TARGET_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1)` in [0,1], or `None` if no frame yielded a lip crop.

- [ ] **Step 1: Write the failing test**

Create `tests/test_game_preprocess.py`:

```python
import base64

import cv2
import numpy as np

import api._game as game


def _jpeg_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_decode_jpeg_roundtrip():
    bgr = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    out = game.decode_jpeg(_jpeg_b64(bgr))
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_decode_jpeg_bad_input_raises():
    import pytest

    with pytest.raises(ValueError):
        game.decode_jpeg("not-base64-jpeg")


def test_preprocess_frames_shapes_when_face_found(monkeypatch):
    # Force a detected lip bbox so we exercise the crop/resize/pad path.
    monkeypatch.setattr(game, "_mediapipe_lip_bbox", lambda frame: (0, 0, frame.shape[1], frame.shape[0]))
    frames = [np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8) for _ in range(10)]
    arr = game.preprocess_frames(frames)
    assert arr is not None
    assert arr.shape == (1, game.TARGET_FRAMES, game.FRAME_HEIGHT, game.FRAME_WIDTH, 1)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0


def test_preprocess_frames_returns_none_when_no_face(monkeypatch):
    monkeypatch.setattr(game, "_mediapipe_lip_bbox", lambda frame: None)
    frames = [np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8) for _ in range(5)]
    assert game.preprocess_frames(frames) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_game_preprocess.py -v`
Expected: FAIL with `AttributeError: module 'api._game' has no attribute 'decode_jpeg'`

- [ ] **Step 3: Write minimal implementation**

Add to `api/_game.py` (imports at top, functions below the existing code):

```python
import base64

import cv2
import numpy as np

from src.utils import FRAME_HEIGHT, FRAME_WIDTH, _mediapipe_lip_bbox
from experiments.isolated_word_level.temporal_slicer import slice_and_pad_word_clip


def decode_jpeg(b64: str) -> np.ndarray:
    """Decode a base64 JPEG string into a BGR uint8 frame."""
    try:
        raw = base64.b64decode(b64, validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not decode JPEG: {exc}") from exc
    if img is None:
        raise ValueError("Could not decode JPEG: empty image")
    return img


def preprocess_frames(bgr_frames: list[np.ndarray]) -> np.ndarray | None:
    """Lip-crop, grayscale, resize and pad frames to the model input tensor."""
    crops: list[np.ndarray] = []
    for frame in bgr_frames:
        bbox = _mediapipe_lip_bbox(frame)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        lip = frame[y0:y1, x0:x1]
        if lip.size == 0:
            continue
        gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
        crops.append(resized.astype(np.float32) / 255.0)

    if not crops:
        return None

    stacked = np.stack(crops, axis=0)  # (t, H, W)
    clip = slice_and_pad_word_clip(
        stacked, 0, stacked.shape[0], target_frames=TARGET_FRAMES, pad_mode="repeat_last"
    )  # (TARGET_FRAMES, H, W, 1)
    return clip[np.newaxis, ...].astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_game_preprocess.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add api/_game.py tests/test_game_preprocess.py
git commit -m "feat(game): per-frame JPEG decode and lip-crop preprocessing"
```

---

## Task 3: Isolated-word classifier singleton + score_window

**Files:**
- Modify: `api/_game.py`
- Test: `tests/test_game_scoring.py`

**Interfaces:**
- Consumes: `build_lipreading_isolated_word_classifier` from `experiments.isolated_word_level.model`; `IDX_TO_WORD` from `experiments.isolated_word_level.common`; `slot_candidate_indices`, `TARGET_FRAMES`, `FRAME_HEIGHT`, `FRAME_WIDTH`.
- Produces:
  - `get_classifier()` -> object exposing `predict_logits(frames: np.ndarray) -> np.ndarray` of shape `(1, 51)`.
  - `score_window(frames: np.ndarray, slot_index: int, *, classifier=None, top_k: int = 3) -> dict` returning `{"top_word": str, "confidence": float, "topk": [{"word": str, "confidence": float}, ...], "ranked_words": [str, ...]}` where probabilities are a softmax restricted to the slot's candidates.

- [ ] **Step 1: Write the failing test**

Create `tests/test_game_scoring.py`:

```python
import numpy as np

import api._game as game
from experiments.isolated_word_level.common import WORD_TO_IDX, NUM_WORD_CLASSES


class _FakeClassifier:
    """Returns fixed logits so scoring is deterministic."""

    def __init__(self, logits: np.ndarray):
        self._logits = logits

    def predict_logits(self, frames: np.ndarray) -> np.ndarray:
        return self._logits


def _logits_favoring(word: str) -> np.ndarray:
    logits = np.zeros((1, NUM_WORD_CLASSES), dtype=np.float32)
    logits[0, WORD_TO_IDX[word]] = 10.0
    return logits


def test_score_window_restricts_to_slot_and_picks_top():
    frames = np.zeros((1, game.TARGET_FRAMES, game.FRAME_HEIGHT, game.FRAME_WIDTH, 1), dtype=np.float32)
    fake = _FakeClassifier(_logits_favoring("blue"))
    out = game.score_window(frames, slot_index=1, classifier=fake)  # color slot
    assert out["top_word"] == "blue"
    assert out["confidence"] > 0.9
    assert set(out["ranked_words"]) == {"blue", "green", "red", "white"}
    assert out["ranked_words"][0] == "blue"


def test_score_window_confidence_is_softmax_over_slot_only():
    frames = np.zeros((1, game.TARGET_FRAMES, game.FRAME_HEIGHT, game.FRAME_WIDTH, 1), dtype=np.float32)
    # Equal logits across the 4 colors -> ~0.25 each.
    logits = np.full((1, NUM_WORD_CLASSES), -50.0, dtype=np.float32)
    for w in ("blue", "green", "red", "white"):
        logits[0, WORD_TO_IDX[w]] = 1.0
    out = game.score_window(frames, slot_index=1, classifier=_FakeClassifier(logits))
    assert abs(out["confidence"] - 0.25) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_game_scoring.py -v`
Expected: FAIL with `AttributeError: module 'api._game' has no attribute 'score_window'`

- [ ] **Step 3: Write minimal implementation**

Add to `api/_game.py`:

```python
import os
from pathlib import Path

from experiments.isolated_word_level.common import IDX_TO_WORD, NUM_WORD_CLASSES
from experiments.isolated_word_level.model import build_lipreading_isolated_word_classifier

_ROOT = Path(__file__).resolve().parent.parent
_ISOLATED_DIR = _ROOT / "checkpoints" / "isolated_word_level"

_classifier = None


class IsolatedWordClassifier:
    """Lazy-loaded singleton wrapper around the isolated-word model."""

    def __init__(self) -> None:
        self.model = build_lipreading_isolated_word_classifier(
            model_variant="conformer_lite",
            frontend_model="flatten",
            num_word_classes=NUM_WORD_CLASSES,
        )
        dummy = np.zeros((1, TARGET_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.float32)
        _ = self.model(dummy, training=False)
        self.model.load_weights(self._resolve_weights())

    @staticmethod
    def _resolve_weights() -> str:
        env = os.environ.get("IMCONVO_ISOLATED_WEIGHTS")
        if env:
            return env
        files = sorted(_ISOLATED_DIR.glob("*.weights.h5"))
        if not files:
            raise FileNotFoundError(f"No isolated-word weights in {_ISOLATED_DIR}")
        return str(files[-1])

    def predict_logits(self, frames: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(frames, training=False))


def get_classifier() -> IsolatedWordClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IsolatedWordClassifier()
    return _classifier


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def score_window(frames: np.ndarray, slot_index: int, *, classifier=None, top_k: int = 3) -> dict:
    clf = classifier if classifier is not None else get_classifier()
    logits = clf.predict_logits(frames)[0]  # (51,)
    cand = slot_candidate_indices(slot_index)
    cand_logits = np.array([logits[i] for i in cand], dtype=np.float32)
    probs = _softmax(cand_logits)
    order = np.argsort(-probs)
    ranked_words = [IDX_TO_WORD[cand[i]] for i in order]
    ranked_probs = [float(probs[i]) for i in order]
    topk = [{"word": w, "confidence": p} for w, p in zip(ranked_words[:top_k], ranked_probs[:top_k])]
    return {
        "top_word": ranked_words[0],
        "confidence": ranked_probs[0],
        "topk": topk,
        "ranked_words": ranked_words,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_game_scoring.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add api/_game.py tests/test_game_scoring.py
git commit -m "feat(game): isolated-word classifier singleton and slot-constrained scoring"
```

---

## Task 4: WebSocket endpoint `/ws/game`

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_game_ws.py`

**Interfaces:**
- Consumes: `decode_jpeg`, `preprocess_frames`, `score_window`, `evaluate_pass`, `SLOT_NAMES`, `BUFFER_FRAMES`, `SCORE_EVERY_N_FRAMES` from `api._game`; `SLOT_VOCABS` from `experiments.isolated_word_level.common`.
- Produces: WebSocket route `GET /ws/game`. Client→server messages: `start_round` `{type, slot_index, target}`, `frame` `{type, data}`, `end_round` `{type}`, `ping` `{type}`. Server→client: `progress` `{type, word, confidence, topk, face}`, `result` `{type, pass, word, confidence, target}`, `error` `{type, message}`, `pong` `{type}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_game_ws.py`:

```python
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api._game as game
from api.main import app


def _jpeg_b64() -> str:
    bgr = np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _patch_pipeline(monkeypatch, *, face=True, top_word="blue", confidence=0.95):
    monkeypatch.setattr(game, "preprocess_frames", lambda frames: (None if not face else np.zeros((1, 1), dtype=np.float32)))
    monkeypatch.setattr(
        game,
        "score_window",
        lambda frames, slot_index, **kw: {
            "top_word": top_word,
            "confidence": confidence,
            "topk": [{"word": top_word, "confidence": confidence}],
            "ranked_words": [top_word, "green", "red", "white"],
        },
    )


def test_ws_pass_on_correct_word(monkeypatch):
    _patch_pipeline(monkeypatch, top_word="blue", confidence=0.95)
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        result = None
        for _ in range(game.SCORE_EVERY_N_FRAMES + 1):
            ws.send_json({"type": "frame", "data": _jpeg_b64()})
            msg = ws.receive_json()
            if msg["type"] == "result":
                result = msg
                break
        assert result is not None
        assert result["pass"] is True
        assert result["word"] == "blue"


def test_ws_reports_no_face(monkeypatch):
    _patch_pipeline(monkeypatch, face=False)
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        for _ in range(game.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg_b64()})
        msg = ws.receive_json()
        assert msg["type"] == "progress"
        assert msg["face"] is False


def test_ws_rejects_bad_target(monkeypatch):
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "purple"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_game_ws.py -v`
Expected: FAIL (404 / WebSocket route not found)

- [ ] **Step 3: Write minimal implementation**

Add to the imports section of `api/main.py`:

```python
from collections import deque

from fastapi import WebSocket, WebSocketDisconnect

import api._game as game
from experiments.isolated_word_level.common import SLOT_VOCABS
```

Add the endpoint near the other routes in `api/main.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_game_ws.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the whole backend suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_game_ws.py
git commit -m "feat(game): /ws/game WebSocket endpoint with live scoring"
```

---

## Task 5: Frontend test tooling + GRID grammar generator

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`, `frontend/vitest.setup.ts`
- Create: `frontend/src/lib/gridGrammar.ts`
- Test: `frontend/src/lib/__tests__/gridGrammar.test.ts`

**Interfaces:**
- Produces:
  - `SLOT_NAMES: readonly string[]` (6 names) and `SLOT_VOCABS: readonly (readonly string[])[]`
  - `type Sentence = { words: string[]; slots: string[] }`
  - `generateSentence(rng?: () => number): Sentence`

- [ ] **Step 1: Add frontend test dependencies**

Run from `frontend/`:

```bash
npm install -D vitest@2 @vitejs/plugin-react@4 @testing-library/react@16 @testing-library/jest-dom@6 jsdom@25
```

Add to the `"scripts"` block in `frontend/package.json`:

```json
    "test": "vitest run",
    "test:watch": "vitest"
```

- [ ] **Step 2: Create the vitest config and setup**

Create `frontend/vitest.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
  },
});
```

Create `frontend/vitest.setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 3: Write the failing test**

Create `frontend/src/lib/__tests__/gridGrammar.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { SLOT_NAMES, SLOT_VOCABS, generateSentence } from "../gridGrammar";

describe("gridGrammar", () => {
  it("has six slots in GRID order", () => {
    expect(SLOT_NAMES).toEqual([
      "command",
      "color",
      "preposition",
      "letter",
      "digit",
      "adverb",
    ]);
    expect(SLOT_VOCABS).toHaveLength(6);
  });

  it("generates one valid word per slot", () => {
    const s = generateSentence();
    expect(s.words).toHaveLength(6);
    expect(s.slots).toEqual([...SLOT_NAMES]);
    s.words.forEach((w, i) => {
      expect(SLOT_VOCABS[i]).toContain(w);
    });
  });

  it("is deterministic given an rng", () => {
    const rng = () => 0; // always picks the first word in each slot
    const s = generateSentence(rng);
    expect(s.words).toEqual(["bin", "blue", "at", "a", "zero", "again"]);
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run (from `frontend/`): `npm run test -- src/lib/__tests__/gridGrammar.test.ts`
Expected: FAIL — cannot resolve `../gridGrammar`.

- [ ] **Step 5: Write minimal implementation**

Create `frontend/src/lib/gridGrammar.ts`:

```ts
export const SLOT_NAMES = [
  "command",
  "color",
  "preposition",
  "letter",
  "digit",
  "adverb",
] as const;

export const SLOT_VOCABS: readonly (readonly string[])[] = [
  ["bin", "lay", "place", "set"],
  ["blue", "green", "red", "white"],
  ["at", "by", "in", "with"],
  ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","x","y","z"],
  ["zero","one","two","three","four","five","six","seven","eight","nine"],
  ["again", "now", "please", "soon"],
];

export type Sentence = { words: string[]; slots: string[] };

export function generateSentence(rng: () => number = Math.random): Sentence {
  const words = SLOT_VOCABS.map((vocab) => vocab[Math.floor(rng() * vocab.length)]);
  return { words, slots: [...SLOT_NAMES] };
}
```

- [ ] **Step 6: Run test to verify it passes**

Run (from `frontend/`): `npm run test -- src/lib/__tests__/gridGrammar.test.ts`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts frontend/vitest.setup.ts frontend/src/lib/gridGrammar.ts frontend/src/lib/__tests__/gridGrammar.test.ts
git commit -m "feat(game): frontend test tooling and GRID grammar generator"
```

---

## Task 6: Game state reducer (`useGame`)

**Files:**
- Create: `frontend/src/hooks/useGame.ts`
- Test: `frontend/src/hooks/__tests__/useGame.test.ts`

**Interfaces:**
- Consumes: `generateSentence`, `Sentence` from `../lib/gridGrammar`.
- Produces:
  - `type GameStatus = "playing" | "won"`
  - `type GameState = { sentence: Sentence; roundIndex: number; status: GameStatus; score: number; streak: number; bestStreak: number; attempts: number; lastResult: "pass" | "fail" | null }`
  - `type GameAction = { type: "ROUND_PASS" } | { type: "ROUND_FAIL" } | { type: "NEW_SENTENCE"; sentence?: Sentence } | { type: "SKIP_WORD" }`
  - `initGameState(sentence?: Sentence): GameState`
  - `gameReducer(state: GameState, action: GameAction): GameState`
  - persistence keys: `localStorage["imconvo.game.bestStreak"]`, `localStorage["imconvo.game.score"]`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/hooks/__tests__/useGame.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { gameReducer, initGameState } from "../useGame";
import type { Sentence } from "../../lib/gridGrammar";

const SENTENCE: Sentence = {
  words: ["bin", "blue", "at", "a", "zero", "again"],
  slots: ["command", "color", "preposition", "letter", "digit", "adverb"],
};

describe("gameReducer", () => {
  beforeEach(() => localStorage.clear());

  it("advances round and scores on pass", () => {
    const s0 = initGameState(SENTENCE);
    const s1 = gameReducer(s0, { type: "ROUND_PASS" });
    expect(s1.roundIndex).toBe(1);
    expect(s1.score).toBe(1);
    expect(s1.status).toBe("playing");
    expect(s1.lastResult).toBe("pass");
  });

  it("stays on same round and counts attempt on fail (no lose state)", () => {
    const s0 = initGameState(SENTENCE);
    const s1 = gameReducer(s0, { type: "ROUND_FAIL" });
    expect(s1.roundIndex).toBe(0);
    expect(s1.score).toBe(0);
    expect(s1.attempts).toBe(1);
    expect(s1.status).toBe("playing");
  });

  it("wins after six passes and bumps streak", () => {
    let s = initGameState(SENTENCE);
    for (let i = 0; i < 6; i++) s = gameReducer(s, { type: "ROUND_PASS" });
    expect(s.status).toBe("won");
    expect(s.streak).toBe(1);
    expect(s.bestStreak).toBe(1);
    expect(localStorage.getItem("imconvo.game.bestStreak")).toBe("1");
  });

  it("starts a new sentence and resets round/attempts", () => {
    let s = initGameState(SENTENCE);
    s = gameReducer(s, { type: "ROUND_FAIL" });
    s = gameReducer(s, { type: "NEW_SENTENCE", sentence: SENTENCE });
    expect(s.roundIndex).toBe(0);
    expect(s.attempts).toBe(0);
    expect(s.status).toBe("playing");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `frontend/`): `npm run test -- src/hooks/__tests__/useGame.test.ts`
Expected: FAIL — cannot resolve `../useGame`.

- [ ] **Step 3: Write minimal implementation**

Create `frontend/src/hooks/useGame.ts`:

```ts
import { useReducer } from "react";
import { generateSentence, type Sentence } from "../lib/gridGrammar";

export type GameStatus = "playing" | "won";

export type GameState = {
  sentence: Sentence;
  roundIndex: number;
  status: GameStatus;
  score: number;
  streak: number;
  bestStreak: number;
  attempts: number;
  lastResult: "pass" | "fail" | null;
};

export type GameAction =
  | { type: "ROUND_PASS" }
  | { type: "ROUND_FAIL" }
  | { type: "NEW_SENTENCE"; sentence?: Sentence }
  | { type: "SKIP_WORD" };

const BEST_KEY = "imconvo.game.bestStreak";
const SCORE_KEY = "imconvo.game.score";

function readNumber(key: string): number {
  if (typeof localStorage === "undefined") return 0;
  const raw = localStorage.getItem(key);
  return raw ? Number(raw) || 0 : 0;
}

function write(key: string, value: number): void {
  if (typeof localStorage !== "undefined") localStorage.setItem(key, String(value));
}

export function initGameState(sentence: Sentence = generateSentence()): GameState {
  return {
    sentence,
    roundIndex: 0,
    status: "playing",
    score: readNumber(SCORE_KEY),
    streak: 0,
    bestStreak: readNumber(BEST_KEY),
    attempts: 0,
    lastResult: null,
  };
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "ROUND_PASS": {
      const score = state.score + 1;
      write(SCORE_KEY, score);
      const nextRound = state.roundIndex + 1;
      if (nextRound >= state.sentence.words.length) {
        const streak = state.streak + 1;
        const bestStreak = Math.max(state.bestStreak, streak);
        write(BEST_KEY, bestStreak);
        return { ...state, status: "won", score, streak, bestStreak, attempts: 0, lastResult: "pass" };
      }
      return { ...state, roundIndex: nextRound, score, attempts: 0, lastResult: "pass" };
    }
    case "ROUND_FAIL":
      return { ...state, attempts: state.attempts + 1, lastResult: "fail" };
    case "SKIP_WORD": {
      const nextRound = state.roundIndex + 1;
      if (nextRound >= state.sentence.words.length) {
        return { ...state, status: "won", attempts: 0, lastResult: null };
      }
      return { ...state, roundIndex: nextRound, attempts: 0, lastResult: null };
    }
    case "NEW_SENTENCE":
      return {
        ...state,
        sentence: action.sentence ?? generateSentence(),
        roundIndex: 0,
        status: "playing",
        streak: state.status === "won" ? state.streak : 0,
        attempts: 0,
        lastResult: null,
      };
    default:
      return state;
  }
}

export function useGame(initial?: Sentence) {
  return useReducer(gameReducer, initGameState(initial));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `frontend/`): `npm run test -- src/hooks/__tests__/useGame.test.ts`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useGame.ts frontend/src/hooks/__tests__/useGame.test.ts
git commit -m "feat(game): client game state reducer with score/streak persistence"
```

---

## Task 7: Webcam + WebSocket hooks

**Files:**
- Create: `frontend/src/hooks/useWebcam.ts`
- Create: `frontend/src/hooks/useGameSocket.ts`

**Interfaces:**
- Produces:
  - `useWebcam()` -> `{ videoRef: RefObject<HTMLVideoElement>, ready: boolean, error: string | null, grabFrame: () => string | null }` where `grabFrame` returns a base64 JPEG (no data-URL prefix) at a downscaled size, or `null` if not ready.
  - `wsUrlFromApi(apiBase: string): string` — converts an `http(s)://host` API base to a `ws(s)://host/ws/game` URL.
  - `useGameSocket(onMessage)` -> `{ connected: boolean, startRound: (slotIndex: number, target: string) => void, sendFrame: (b64: string) => void, endRound: () => void }`.

- [ ] **Step 1: Write the failing test (URL helper is the pure, testable part)**

Create `frontend/src/hooks/__tests__/useGameSocket.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { wsUrlFromApi } from "../useGameSocket";

describe("wsUrlFromApi", () => {
  it("maps http to ws and appends the game path", () => {
    expect(wsUrlFromApi("http://localhost:8001")).toBe("ws://localhost:8001/ws/game");
  });
  it("maps https to wss", () => {
    expect(wsUrlFromApi("https://api.example.com")).toBe("wss://api.example.com/ws/game");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `frontend/`): `npm run test -- src/hooks/__tests__/useGameSocket.test.ts`
Expected: FAIL — cannot resolve `../useGameSocket`.

- [ ] **Step 3: Implement `useGameSocket.ts`**

Create `frontend/src/hooks/useGameSocket.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export type ServerMessage =
  | { type: "progress"; word: string; confidence: number; topk: { word: string; confidence: number }[]; face: boolean }
  | { type: "result"; pass: boolean; word: string; confidence: number; target: string }
  | { type: "error"; message: string }
  | { type: "pong" };

export function wsUrlFromApi(apiBase: string): string {
  const proto = apiBase.startsWith("https") ? "wss" : "ws";
  const host = apiBase.replace(/^https?:\/\//, "").replace(/\/$/, "");
  return `${proto}://${host}/ws/game`;
}

export function useGameSocket(onMessage: (msg: ServerMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      const ws = new WebSocket(wsUrlFromApi(API_BASE));
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 1000);
      };
      ws.onmessage = (e) => {
        try {
          onMessageRef.current(JSON.parse(e.data) as ServerMessage);
        } catch {
          /* ignore malformed */
        }
      };
    };
    connect();

    return () => {
      closed = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((payload: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }, []);

  const startRound = useCallback((slotIndex: number, target: string) => send({ type: "start_round", slot_index: slotIndex, target }), [send]);
  const sendFrame = useCallback((b64: string) => send({ type: "frame", data: b64 }), [send]);
  const endRound = useCallback(() => send({ type: "end_round" }), [send]);

  return { connected, startRound, sendFrame, endRound };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `frontend/`): `npm run test -- src/hooks/__tests__/useGameSocket.test.ts`
Expected: PASS (2 passed)

- [ ] **Step 5: Implement `useWebcam.ts` (no separate unit test; verified via manual run in Task 9)**

Create `frontend/src/hooks/useWebcam.ts`:

```ts
import { useCallback, useEffect, useRef, useState } from "react";

const CAPTURE_W = 240;
const CAPTURE_H = 180;

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((s) => {
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      })
      .catch((e) => setError(e?.message || "Camera permission denied"));
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  const grabFrame = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || !ready) return null;
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
      canvasRef.current.width = CAPTURE_W;
      canvasRef.current.height = CAPTURE_H;
    }
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    return canvasRef.current.toDataURL("image/jpeg", 0.7).split(",")[1] ?? null;
  }, [ready]);

  return { videoRef, ready, error, grabFrame };
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useWebcam.ts frontend/src/hooks/useGameSocket.ts frontend/src/hooks/__tests__/useGameSocket.test.ts
git commit -m "feat(game): webcam capture and game WebSocket hooks"
```

---

## Task 8: UI components + game page + nav link

**Files:**
- Create: `frontend/src/components/game/WordPrompt.tsx`, `ConfidenceMeter.tsx`, `Hud.tsx`, `RoundFeedback.tsx`, `ResultScreen.tsx`
- Create: `frontend/src/app/game/page.tsx`
- Modify: `frontend/src/components/Navbar.tsx`
- Test: `frontend/src/components/game/__tests__/ConfidenceMeter.test.tsx`

**Interfaces:**
- Consumes: `useGame`, `useWebcam`, `useGameSocket`, `gridGrammar`.
- Produces: presentational components plus the wired `/game` page.
  - `ConfidenceMeter({ confidence, word, face }: { confidence: number; word: string; face: boolean })`
  - `WordPrompt({ word, slot }: { word: string; slot: string })`
  - `Hud({ score, streak, bestStreak, attempts }: { score: number; streak: number; bestStreak: number; attempts: number })`
  - `RoundFeedback({ result }: { result: "pass" | "fail" | null })`
  - `ResultScreen({ onNewSentence }: { onNewSentence: () => void })`

- [ ] **Step 1: Write the failing component test**

Create `frontend/src/components/game/__tests__/ConfidenceMeter.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceMeter } from "../ConfidenceMeter";

describe("ConfidenceMeter", () => {
  it("shows the face prompt when no face is detected", () => {
    render(<ConfidenceMeter confidence={0} word="" face={false} />);
    expect(screen.getByText(/center your face/i)).toBeInTheDocument();
  });

  it("renders the current guess and percentage when a face is present", () => {
    render(<ConfidenceMeter confidence={0.83} word="blue" face={true} />);
    expect(screen.getByText(/blue/)).toBeInTheDocument();
    expect(screen.getByText(/83%/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `frontend/`): `npm run test -- src/components/game/__tests__/ConfidenceMeter.test.tsx`
Expected: FAIL — cannot resolve `../ConfidenceMeter`.

- [ ] **Step 3: Implement the presentational components**

Create `frontend/src/components/game/ConfidenceMeter.tsx`:

```tsx
export function ConfidenceMeter({ confidence, word, face }: { confidence: number; word: string; face: boolean }) {
  if (!face) {
    return <p className="text-amber-500">Center your face in frame…</p>;
  }
  const pct = Math.round(confidence * 100);
  return (
    <div className="w-full">
      <div className="flex justify-between text-sm">
        <span>{word || "…"}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-3 w-full rounded bg-gray-200">
        <div className="h-3 rounded bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
```

Create `frontend/src/components/game/WordPrompt.tsx`:

```tsx
export function WordPrompt({ word, slot }: { word: string; slot: string }) {
  return (
    <div className="text-center">
      <p className="text-xs uppercase tracking-widest text-gray-500">{slot}</p>
      <p className="text-5xl font-bold">{word}</p>
    </div>
  );
}
```

Create `frontend/src/components/game/Hud.tsx`:

```tsx
export function Hud({ score, streak, bestStreak, attempts }: { score: number; streak: number; bestStreak: number; attempts: number }) {
  return (
    <div className="flex gap-6 text-sm">
      <span>Score: {score}</span>
      <span>Streak: {streak}</span>
      <span>Best: {bestStreak}</span>
      <span>Attempts: {attempts}</span>
    </div>
  );
}
```

Create `frontend/src/components/game/RoundFeedback.tsx`:

```tsx
export function RoundFeedback({ result }: { result: "pass" | "fail" | null }) {
  if (!result) return null;
  return result === "pass" ? (
    <p className="text-emerald-600 font-semibold">✓ Correct!</p>
  ) : (
    <p className="text-rose-600 font-semibold">✗ Try again</p>
  );
}
```

Create `frontend/src/components/game/ResultScreen.tsx`:

```tsx
export function ResultScreen({ onNewSentence }: { onNewSentence: () => void }) {
  return (
    <div className="text-center space-y-4">
      <p className="text-3xl font-bold">🎉 You read the whole sentence!</p>
      <button onClick={onNewSentence} className="rounded bg-emerald-600 px-4 py-2 text-white">
        New sentence
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `frontend/`): `npm run test -- src/components/game/__tests__/ConfidenceMeter.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire the game page**

Create `frontend/src/app/game/page.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGame } from "../../hooks/useGame";
import { useWebcam } from "../../hooks/useWebcam";
import { useGameSocket, type ServerMessage } from "../../hooks/useGameSocket";
import { WordPrompt } from "../../components/game/WordPrompt";
import { ConfidenceMeter } from "../../components/game/ConfidenceMeter";
import { Hud } from "../../components/game/Hud";
import { RoundFeedback } from "../../components/game/RoundFeedback";
import { ResultScreen } from "../../components/game/ResultScreen";

const ROUND_WINDOW_MS = 1500;
const FRAME_INTERVAL_MS = 66; // ~15 fps

export default function GamePage() {
  const [state, dispatch] = useGame();
  const { videoRef, ready, error, grabFrame } = useWebcam();
  const [meter, setMeter] = useState({ word: "", confidence: 0, face: true });
  const attemptingRef = useRef(false);

  const onMessage = useCallback((msg: ServerMessage) => {
    if (msg.type === "progress") {
      setMeter({ word: msg.word, confidence: msg.confidence, face: msg.face });
    } else if (msg.type === "result") {
      attemptingRef.current = false;
      dispatch({ type: msg.pass ? "ROUND_PASS" : "ROUND_FAIL" });
    }
  }, [dispatch]);

  const { connected, startRound, sendFrame, endRound } = useGameSocket(onMessage);

  const attempt = useCallback(() => {
    if (attemptingRef.current || !ready || state.status !== "playing") return;
    attemptingRef.current = true;
    setMeter({ word: "", confidence: 0, face: true });
    startRound(state.roundIndex, state.sentence.words[state.roundIndex]);
    const interval = setInterval(() => {
      const f = grabFrame();
      if (f) sendFrame(f);
    }, FRAME_INTERVAL_MS);
    setTimeout(() => {
      clearInterval(interval);
      if (attemptingRef.current) endRound();
    }, ROUND_WINDOW_MS);
  }, [ready, state.status, state.roundIndex, state.sentence.words, startRound, sendFrame, endRound, grabFrame]);

  useEffect(() => {
    attemptingRef.current = false;
  }, [state.roundIndex, state.status]);

  return (
    <main className="mx-auto max-w-xl space-y-6 p-6">
      <Hud score={state.score} streak={state.streak} bestStreak={state.bestStreak} attempts={state.attempts} />
      {error && <p className="text-rose-600">Camera error: {error}</p>}
      <video ref={videoRef} autoPlay playsInline muted className="w-full rounded bg-black" />
      {state.status === "won" ? (
        <ResultScreen onNewSentence={() => dispatch({ type: "NEW_SENTENCE" })} />
      ) : (
        <>
          <WordPrompt word={state.sentence.words[state.roundIndex]} slot={state.sentence.slots[state.roundIndex]} />
          <ConfidenceMeter confidence={meter.confidence} word={meter.word} face={meter.face} />
          <RoundFeedback result={state.lastResult} />
          <button
            onClick={attempt}
            disabled={!ready || !connected}
            className="w-full rounded bg-emerald-600 px-4 py-3 text-white disabled:opacity-50"
          >
            {connected ? "Ready… GO" : "Connecting…"}
          </button>
        </>
      )}
    </main>
  );
}
```

- [ ] **Step 6: Add the navbar link**

In `frontend/src/components/Navbar.tsx`, add a link to `/game` (labelled "Game") alongside the existing links, following the existing link markup pattern in that file.

- [ ] **Step 7: Run the full frontend suite + typecheck**

Run (from `frontend/`): `npm run test && npx tsc --noEmit`
Expected: all tests pass; no type errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/game frontend/src/app/game frontend/src/components/Navbar.tsx
git commit -m "feat(game): game UI components, /game page, and nav link"
```

---

## Task 9: Manual end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Start backend + frontend**

Run (from `frontend/`): `npm run dev:all`
Expected: API on `:8001`, web on `:3000`, no startup errors.

- [ ] **Step 2: Play a round**

Open `http://localhost:3000/game`, grant camera permission, press **Ready… GO**, and silently mouth the prompted word.
Expected: the confidence meter moves; a correct mouthing advances to the next word; six correct words reach the win screen; "New sentence" restarts.

- [ ] **Step 3: Verify no-face and reconnect handling**

Cover the camera → meter shows "Center your face…". Stop and restart the API → the GO button shows "Connecting…" then re-enables.

- [ ] **Step 4: Commit any fixes found during verification**

```bash
git add -A && git commit -m "fix(game): address issues found in manual verification"
```

---

## Self-Review Notes

- **Spec coverage:** WebSocket transport (Task 4), isolated-word conformer-lite model + slot-constrained scoring (Task 3), per-frame lip-crop reusing `_mediapipe_lip_bbox` (Task 2), grammar/slot maps (Tasks 1 & 5), `useGame` no-lives state machine with score/streak/localStorage (Task 6), webcam + WS hooks (Task 7), UI components + page + nav, including no-face and disconnect handling (Task 8), error handling and manual verification (Task 9). Testing strategy from spec §9 is realized as the per-task tests.
- **Deferred per spec (not in plan):** per-round timer, difficulty levels, mobile polish, leaderboard, TF.js inference, auto-segmentation. `SKIP_WORD` included as the spec's optional nice-to-have.
- **Type consistency:** server message shapes (`progress`/`result`/`error`) match between Task 4 (backend) and Task 7 (`ServerMessage`); `score_window` return keys (`top_word`, `confidence`, `topk`, `ranked_words`) are consistent across Tasks 3–4; `evaluate_pass` signature consistent Tasks 1 & 4; `GameAction` variants consistent across Tasks 6 & 8.
