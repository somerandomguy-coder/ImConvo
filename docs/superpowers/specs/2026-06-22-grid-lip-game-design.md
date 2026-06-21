# GRID Lip Game — Live Per-Word Lip-Reading Game

**Date:** 2026-06-22
**Branch:** `feat/game`
**Status:** Design approved, pending spec review

## 1. Summary

Add a new browser game to ImConvo where a player silently mouths (no sound) the
six words of a randomly generated GRID-grammar sentence — e.g. **"bin blue at f
zero now"**. Each word is one **round**: the player reads the word, a live webcam
stream is classified in real time, and reading it correctly advances to the next
round. Completing all six words wins the sentence. Inspired by the TikTok
"lip-reading challenge" format.

This is a new *game mode* layered on top of the existing lip-reading stack. It
reuses the already-trained **isolated-word conformer-lite classifier** and adds a
real-time WebSocket path plus a new frontend page. It does not modify the
existing upload/demo flows or any training code.

## 2. Goals & Non-Goals

### Goals
- Live, per-word gameplay over the webcam (no record-then-upload, no file picker).
- One word per round; six rounds per sentence following GRID grammar.
- Real-time confidence feedback as the player mouths each word.
- Unlimited retries per word (no lives / no lose state) — the player keeps
  attempting a word until it is recognized, then advances.
- Streak / score, persisted locally.
- Run on the existing web stack (Next.js frontend + FastAPI backend) and webcam.

### Non-Goals (v1)
- Per-round countdown timer (deferred).
- Selectable difficulty levels (deferred).
- Open-vocabulary recognition (the game only uses the 51 GRID words).
- Mobile-specific layout tuning (desktop webcam first; should not break on mobile).
- Multiplayer / online leaderboard / accounts.
- Any change to training, the CTC sentence model, or existing API endpoints.

## 3. Key Decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Capture mechanic | Live, **1 word per round** | Matches the TikTok format the user wants. |
| Transport | **WebSocket frame streaming** | Lowest latency, true live confidence meter. |
| Model | **Isolated-word conformer-lite classifier** | Purpose-built for single 25-frame words (top-1 ≈ 0.85 / top-5 ≈ 0.97 in-speaker); already trained; no OOD risk. It already uses the conformer-lite backbone. |
| Scoring | **Slot-constrained** | Each round targets a known slot; restrict the 51-way softmax to that slot's candidates (e.g. color = best of 4). Dramatically more reliable. |
| v1 scope | Core 6-round loop + **streak / score**, **no lives / no lose state** | No timer, no difficulty levels, no loss condition in v1. |

### Why not char-level CTC
Char-level CTC is trained on 75-frame *sentences*; a single padded word is
out-of-distribution and its sentence trigram LM is useless per word. Also, per
[experiments.jsonl](../../../experiments.jsonl), conformer-lite char
`test_oos_wer ≈ 0.23–0.27` is worse than the bigru baseline (0.10). The isolated
head is the correct tool and reuses conformer-lite anyway.

## 4. Existing Assets Reused

- **Model:** `build_lipreading_isolated_word_classifier(model_variant="conformer_lite", frontend_model="flatten", num_word_classes=51)` in [experiments/isolated_word_level/model.py](../../../experiments/isolated_word_level/model.py).
- **Checkpoint:** `checkpoints/isolated_word_level/best_mv-conformer-lite_fe-flatten_be-conformer-lite_bs-48_t-25_*_isolated_word_v1.weights.h5`.
- **Vocab & slots:** `SLOT_VOCABS`, `FLAT_VOCAB`, `WORD_TO_IDX`, `IDX_TO_WORD`, `word_idx_to_text` in [experiments/isolated_word_level/common.py](../../../experiments/isolated_word_level/common.py). `SLOT_VOCABS` is ordered exactly as the GRID sentence positions (command, color, prep, letter, digit, adverb).
- **Clip shaping:** `slice_and_pad_word_clip(frames, start, end, target_frames=25, pad_mode="repeat_last")` in [experiments/isolated_word_level/temporal_slicer.py](../../../experiments/isolated_word_level/temporal_slicer.py).
- **Lip extraction:** the existing per-frame mouth-ROI crop (grayscale 80×120) in [src/utils.py](../../../src/utils.py) (MediaPipe with Haar fallback). Frames must be preprocessed identically to training.
- **Frontend stack:** Next.js + TypeScript + Tailwind in [frontend/](../../../frontend), API helpers in [frontend/src/utils/api.ts](../../../frontend/src/utils/api.ts).

## 5. Architecture

```
Browser (/game)                         FastAPI backend
┌────────────────────────┐             ┌─────────────────────────────┐
│ useWebcam              │             │ /ws/game (WebSocket)        │
│  getUserMedia → <video>│  frames     │  per-connection:            │
│  canvas frame grabber  │ ──────────▶ │   rolling buffer (deque)    │
│ useGameSocket (WS)     │             │   throttled scoring         │
│  send frame / control  │ ◀────────── │   slot-constrained softmax  │
│  recv progress/result  │ progress    │   IsolatedWordClassifier    │
│ useGame (state machine)│  /result    │   (singleton, lazy-loaded)  │
│  sentence/round/score  │             └─────────────────────────────┘
│ UI components          │
└────────────────────────┘
```

- **Game/session state lives client-side** (sentence, current round, score,
  streak). The backend is a stateless-per-round live recognizer that only
  holds a short rolling frame buffer per connection.
- The backend loads the model **once** (singleton, lazy) and serves all
  connections.

## 6. Backend Design

### 6.1 New module `api/_game.py`
Single responsibility: turn a window of webcam frames + a target slot into a
classification result.

- **Vocab/slot maps:** import `SLOT_VOCABS`, `WORD_TO_IDX`, `IDX_TO_WORD` from
  `experiments.isolated_word_level.common`. Define `SLOT_NAMES = ("command",
  "color", "preposition", "letter", "digit", "adverb")` and a helper
  `slot_candidate_indices(slot_index) -> list[int]`.
- **`IsolatedWordClassifier` singleton:** lazy-build the model, load the
  checkpoint weights (resolve the newest matching file in
  `checkpoints/isolated_word_level/`), warm up with a dummy `(1, 25, 80, 120, 1)`
  tensor. Mirrors the existing CTC model-loader pattern in
  [api/_models.py](../../../api/_models.py)/[api/_utils.py](../../../api/_utils.py).
- **`preprocess_frames(raw_frames) -> np.ndarray`:** decode incoming frames to
  grayscale mouth-ROI 80×120 using the existing lip-extraction utility, stack,
  then `slice_and_pad_word_clip(..., target_frames=25, pad_mode="repeat_last")`.
  Returns `(1, 25, 80, 120, 1)`. If no face/mouth detected, returns `None`.
- **`score_window(frames, slot_index, *, top_k=3) -> ScoreResult`:**
  run the model → 51 logits → **restrict to `slot_candidate_indices(slot_index)`**
  → softmax over that subset → return `{top_word, confidence, topk:[{word,conf}...],
  face: True}`. If `preprocess_frames` returned `None`, return `{face: False}`.

### 6.2 WebSocket endpoint `/ws/game` in `api/main.py`
Per-connection rolling buffer (`collections.deque(maxlen=BUFFER_FRAMES)` where
`BUFFER_FRAMES = 25`). Inference is **throttled**: only run `score_window` every
`SCORE_EVERY_N_FRAMES` (default 3) while a round is active, and only lip-crop
those scored windows (not every incoming frame) to bound CPU cost.

**Message protocol (JSON text frames; image payloads base64 JPEG):**

Client → server:
- `{"type": "start_round", "slot_index": 0..5, "target": "<word>"}` — clears
  buffer, marks round active. The client sends both the slot and the target word
  so the backend stays grammar-agnostic; the server validates `target ∈ SLOT_VOCABS[slot_index]`.
- `{"type": "frame", "data": "<base64 jpeg>"}` — appended to buffer.
- `{"type": "end_round"}` — server scores the final window and emits a `result`.
- `{"type": "ping"}` — keepalive.

Server → client:
- `{"type": "progress", "word": str, "confidence": float, "topk": [...], "face": bool}` — throttled live feedback.
- `{"type": "result", "pass": bool, "word": str, "confidence": float, "target": str}` — round outcome (on early threshold-hit or `end_round`).
- `{"type": "error", "message": str}`.

**Round outcome logic (server-side, per scored window):**
- `target` is the word supplied in `start_round` (validated against the slot).
- **Pass rule:**
  - Letters slot (index 3, 25 candidates): pass if `target` is in the **top-3**.
  - All other slots: pass if `top_word == target` **and** `confidence ≥ PASS_THRESHOLD`.
- `PASS_THRESHOLD` default **0.6** (module constant, easily tunable).
- **Early success:** if a scored window satisfies the pass rule during streaming,
  the server immediately emits `result{pass:true}` and the client ends the round.
- **Timeout/`end_round`:** server emits `result` using the best window seen.

### 6.3 Constants (single source of truth in `api/_game.py`)
`BUFFER_FRAMES=25`, `TARGET_FRAMES=25`, `SCORE_EVERY_N_FRAMES=3`,
`PASS_THRESHOLD=0.6`, `LETTER_SLOT_INDEX=3`, `LETTER_TOPK_PASS=3`.

## 7. Frontend Design (`frontend/src/`)

### 7.1 Page & routing
- `app/game/page.tsx` — the game screen (client component).
- Add a "Game" link to [components/Navbar.tsx](../../../frontend/src/components/Navbar.tsx).

### 7.2 Grammar (`lib/gridGrammar.ts`)
- Mirror `SLOT_VOCABS` and slot names on the client.
- `generateSentence()` → `{ words: string[6], slots: SlotName[6] }` picking one
  random word per slot.
- Pure, no I/O — directly unit-testable.

### 7.3 Hooks
- `hooks/useWebcam.ts` — request `getUserMedia({video:true})`, expose the stream,
  a hidden `<canvas>` frame grabber `grabFrame(): string` (downscaled JPEG
  base64, e.g. 25fps cap), and permission/error state.
- `hooks/useGameSocket.ts` — open `ws://<api>/ws/game`, expose
  `startRound(slotIndex, target)`, `sendFrame(data)`, `endRound()`, and an
  `onMessage` stream of `progress`/`result`/`error`. Auto-reconnect with backoff;
  surfaces connection state so the UI can pause.

### 7.4 Game state (`hooks/useGame.ts`, a reducer)
State machine (no lose state):
```
idle → playing(round n, attempting) → roundResult(pass|fail)
     → (pass & n<6) playing(n+1)
     → (pass & n==6) won
     → (fail)       playing(n)             // re-attempt same word, unlimited
won → idle (new sentence)
```
- Tracks: `sentence`, `roundIndex (0..5)`, `attempts` (per word, informational),
  `score`, `streak`, `bestStreak`.
- **No lives / no lose state.** Fail just re-attempts the same word indefinitely;
  the only terminal state is `won`.
- `score` increments per word passed; `streak` increments per completed sentence;
  the player may choose to start a **new sentence** at any time (this resets the
  current `streak` only if they abandon mid-sentence). `bestStreak` and
  cumulative `score` persisted to `localStorage`.
- Optional **Skip word** control (advances without passing, does not add score) —
  nice-to-have, not required for v1.
- Pure reducer (frames/WS injected via dispatched events) → unit-testable.

### 7.5 UI components (`components/game/`)
- `GameStage` — webcam `<video>` with an overlay; orchestrates capture loop ↔ WS.
- `WordPrompt` — big target word + slot hint (e.g. "COLOR") + "Ready… GO".
- `ConfidenceMeter` — animated bar driven by `progress.confidence`; shows
  current top guess; "position your face" when `face:false`.
- `Hud` — score, streak / best streak, and current-word attempt count.
- `RoundFeedback` — pass ✓ / "try again" flash between attempts.
- `ResultScreen` — win summary (all 6 words) + **New sentence** button. No lose
  screen.

### 7.6 Round flow (client)
1. `useGame` picks a sentence on start; shows round `n` word + slot.
2. Player clicks **GO** → `useGameSocket.startRound(slotIndex, targetWord)`; begin
   capture loop (`grabFrame` → `sendFrame`) at the fps cap.
3. Render `progress` into `ConfidenceMeter`.
4. On `result.pass` (early or after a short ~1.5s window when the client calls
   `endRound()`), stop the loop and advance per the state machine.

## 8. Error Handling

| Condition | Behavior |
|---|---|
| Webcam permission denied / no camera | Friendly prompt with retry; game can't start until granted. |
| No face/mouth detected | `progress.face=false` → "center your face in frame"; window can't pass. |
| WebSocket disconnect mid-round | Pause game, auto-reconnect with backoff, resume current round. |
| Model load failure (backend) | Server emits `error`; UI shows non-fatal "recognizer unavailable" with retry. |
| Slow inference / high CPU | Throttle (`SCORE_EVERY_N_FRAMES`), fps cap, downscaled JPEG, lip-crop only scored windows. |
| Invalid client message / bad `target` | Server emits `error`, ignores the message; no crash. |

## 9. Testing Strategy

### Backend (pytest)
- **Grammar/slot maps:** `slot_candidate_indices` returns the correct GRID
  indices; `SLOT_VOCABS` order matches the six slots.
- **Scoring logic:** with a mocked model returning crafted logits, assert
  `score_window` restricts to the slot, computes confidence, and the pass rule
  (top-1+threshold for normal slots, top-3 for letters) is correct at boundaries.
- **`preprocess_frames`:** synthetic frames → `(1,25,80,120,1)`; no-face path
  returns `None`.
- **WS protocol:** FastAPI `TestClient` websocket — `start_round` → stream frames
  → receive `progress`; `end_round` → receive `result`; bad message → `error`.
- **Optional integration:** feed frames sliced from a known GRID example clip and
  assert the correct word passes.

### Frontend (vitest/jest + RTL, webcam/WS mocked)
- `generateSentence` always yields one valid word per slot, correct slot names.
- `useGame` reducer: round transitions, win at 6/6, re-attempt same word on fail
  (no lose state), score/streak/bestStreak, localStorage persistence.
- Pass mapping from `result` events drives state correctly.
- Component smoke tests for `ConfidenceMeter`, `Hud`, `ResultScreen` and the
  no-face / disconnected states.

## 10. Tunable Defaults (summary)
No lives (unlimited retries), `PASS_THRESHOLD = 0.6`, letters pass = top-3, normal slots = top-1,
`BUFFER_FRAMES = TARGET_FRAMES = 25`, `SCORE_EVERY_N_FRAMES = 3`, capture fps cap
~15–25, round window ~1.5s before client `endRound()`.

## 11. Out of Scope / Future Work
Per-round timer; difficulty levels (easy/hard thresholds); mobile layout polish;
online leaderboard; client-side (TF.js) inference for zero round-trip latency;
mouth-motion auto-segmentation (remove the explicit "GO").
