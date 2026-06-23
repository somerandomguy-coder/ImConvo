# GRID Lip Game v4 — Face-Dominant Layout + Rail Minimap + Pass Reliability

**Date:** 2026-06-23
**Branch:** `feat/game`
**Status:** Approved direction (user feedback on v3); writing spec → plan → build

## 1. Summary

Refine the v3 redesign in response to user testing:
1. **Bug: auto-pass before the user opens their mouth.** Easy-mode top-k scoring fires `result.pass` on idle frames because random ranking has a 50% prior in 4-option slots. Fix server-side with **warmup** + **sustained-signal** requirement.
2. **Layout: face full-bleed (mouth visible) + rail in a small corner.** v3's corridor occupied the whole frame and the bubble word sat in the middle of the face — both blocked the lip-reading subject. Restructure so the face fills the screen, the word sits above (never covering the mouth), and the corridor lives as a small **rail minimap** in the top-right showing progress.
3. **Per-round animation.** Add an "advance through the wall" animation in the minimap each pass: a candy car icon moves forward + the front wall scales out + next walls dolly forward.

Frontend-only **plus** a small server-side tightening of the pass rule. Recognizer model, WS protocol shape, and easy-mode top-k are unchanged.

## 2. Goals & Non-Goals

### Goals
- The webcam fills the play area; the user's face and mouth are always unobstructed.
- The corridor (rails + wall-boxes + advancing car) renders as a **small minimap** in the top-right (~180×140).
- The current word renders **above** the face (between top chrome and the face center) — never overlapping the mouth area.
- Round pass requires the model to agree across **≥2 consecutive scoring windows** AND occur **after a ~1.2 s warmup** of each round.
- Pass triggers a brief animation in the minimap (car advances + walls slide forward), plus the existing confetti.

### Non-Goals
- Re-introducing lives / game-over.
- Re-introducing the confidence bar (still removed per the user's earlier request).
- WebGL/real 3D corridor — minimap stays SVG/CSS.
- Re-training or changing the model / WS message shapes.

## 3. Key Decisions

| Decision | Choice |
|---|---|
| Layout | Face full-bleed; word top-center; rail minimap top-right; score bottom-left; mute top-left |
| Word position | Above the face center, below the top chrome — never over the mouth |
| Corridor | Lives in the minimap only (~180×140), with rails + receding walls + a candy car |
| Pass rule | warmup 1200 ms + 2 consecutive top-k windows (server-side) |
| Animation | Pass → car advances one wall in the minimap; current wall scales out; next walls dolly forward |

## 4. Architecture (changes only)

**Backend (`api/_game.py` + `api/main.py`):**
- Add tunables: `PASS_WARMUP_MS = 1200`, `PASS_STREAK_REQUIRED = 2`.
- `evaluate_pass` is purely top-k (unchanged signature).
- Move the "is this a pass for the connection?" decision into the WS handler, which tracks per-round state: `round_started_at`, `streak` (count of consecutive windows where `evaluate_pass` returned true), `last_match_word`. Only emit `result.pass=true` when `(now - round_started_at) ≥ PASS_WARMUP_MS` AND `streak ≥ PASS_STREAK_REQUIRED` AND the matched word is the current target. Reset these on every `start_round`. Below-bar scoring windows emit `progress` as today.

**Frontend (replace v3 corridor):**
- Delete `CorridorScene.tsx`, `WallWord.tsx`.
- New `FaceStage.tsx` — full-bleed `<video>` + an overlay slot for children. Owns the responsive frame.
- New `WordPrompt.tsx` — cream board + bubble word, positioned in the upper third (above the face center), never over the mouth area.
- New `RailMinimap.tsx` — small SVG corridor (~180×140), 6 wall slots; shows current/passed/upcoming; renders a small candy car between rails that animates to the next slot on `roundIndex` change. Honors `prefers-reduced-motion`.
- `page.tsx` swaps `CorridorScene` → `FaceStage`, places `WordPrompt` and `RailMinimap` as children.

## 5. Components

### `FaceStage({ videoRef, children })`
Replaces v3 `CorridorScene`. Renders:
- `<video ref={videoRef} autoPlay playsInline muted>` at `opacity: 1`, `object-fit: cover` filling a rounded card.
- A relative-positioned overlay containing `{children}` (timer, dots, word, minimap, score, mute, win screen, camera-error retry).
- No corridor SVG. The face is unobstructed.

### `WordPrompt({ slot, word })`
Cream candy board with uppercase slot label + bubble-font magenta word. Positioned via parent CSS at `top: 92px` so it sits above the face center.

### `RailMinimap({ total, roundIndex })`
- Container ~180×140, top-right, candy white border, semi-transparent dark background.
- SVG corridor: floor polygon, two red-and-white dashed rails converging to vanishing point, **6 wall frames** at depths z₀…z₅ (front → back).
- A small candy car (rounded rectangle + 2 wheels) sits between rails near the bottom; on `roundIndex` change, its x/y transforms to the next wall slot.
- Walls update state: passed = faded gold-green tick; current = bright wood; upcoming = muted wood.
- Animation: CSS transition on `transform` (~450 ms). `prefers-reduced-motion` → snap.

### Backend `_game.py` additions
```python
PASS_WARMUP_MS = 1200
PASS_STREAK_REQUIRED = 2
```
WS handler keeps `round_started_at: float | None`, `streak: int`, `last_match: str | None` in the per-connection state; updates on each scored window; resets on `start_round`.

## 6. Round Flow (changes)

1. `start_round{slot_index,target}` → server: clear buffer, set `round_started_at = monotonic()`, `streak = 0`, `last_match = None`.
2. Every scored window: compute `match = evaluate_pass(slot_index, target, ranked_words)`.
   - If `match` and `last_match == target`: `streak += 1`.
   - Else if `match`: `streak = 1`, `last_match = target`.
   - Else: `streak = 0`, `last_match = None`.
3. Emit `result.pass=true` only when `streak ≥ PASS_STREAK_REQUIRED` and `(now - round_started_at) ≥ PASS_WARMUP_MS / 1000`.
4. Frontend: existing `useRoundLoop` unchanged — it just receives `result` later than before (more reliable). On pass, fire confetti + `RailMinimap` advances + the wall scales out.

## 7. Tunables
`PASS_WARMUP_MS=1200`, `PASS_STREAK_REQUIRED=2`, minimap size 180×140, advance animation 450 ms, candy car path ≈ 22 px between adjacent walls.

## 8. Testing
- **Backend (pytest):** new tests in `tests/test_game_rules_v4.py`:
  - `evaluate_pass` unchanged (top-k membership).
  - WS handler: with a mocked classifier returning the correct word every window, the server emits `progress` during warmup, then `result.pass=true` only after the 2nd post-warmup matching window.
  - Wrong-word streaks never produce `result.pass`.
- **Frontend (vitest):** `RailMinimap` renders 6 walls + applies the right state per `roundIndex` (passed/current/upcoming) and the car offset advances per `roundIndex`. `FaceStage` renders the video and children. WordPrompt unchanged behavior.
- Existing tests stay green (deleted CorridorScene/WallWord tests removed; new components have their own).

## 9. Out of Scope
Mobile portrait; lives/lose state; music; difficulty selector; 3D minimap.
