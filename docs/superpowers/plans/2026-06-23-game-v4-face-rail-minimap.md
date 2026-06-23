# Game v4 Implementation Plan — Face-Dominant + Rail Minimap + Pass Reliability

> Execute task-by-task with TDD. Branch: `feat/game`. Steps use `- [ ]`.

**Goal:** Fix the auto-pass bug; restructure the play screen so the face fills the screen and the corridor lives in a small top-right minimap that animates the runner forward each round.

**Tech stack:** FastAPI/Python (backend), Next.js 16 + React 19 + TS (frontend), vitest, pytest.

## Global Constraints
- Branch `feat/game`. All work in worktree `/Users/maverick/Desktop/UTS/DeepLearning/ImConvo/.claude/worktrees/lucid-carson-d6bc40`. **Never commit to main.**
- Pass-rule tunables (single source in `api/_game.py`): `PASS_WARMUP_MS=1200`, `PASS_STREAK_REQUIRED=2`. Existing `BUFFER_FRAMES=25, SCORE_EVERY_N_FRAMES=3, TARGET_FRAMES=25, LETTER_SLOT_INDEX=3, DEFAULT_PASS_TOPK=2, PASS_TOPK={3:5}` stay as-is.
- `evaluate_pass(slot_index, target, ranked_words, confidence=None) -> bool` signature unchanged (still top-k membership only).
- WS protocol shape unchanged (`progress`/`result`/`error`/`pong`).
- Frontend: face full-bleed, mouth visible at all times. Word above face center, never over the mouth. Rail minimap top-right (~180×140).

---

## Task 1: Backend — warmup + sustained-signal pass rule

**Files:**
- Modify: `api/_game.py` (add 2 constants)
- Modify: `api/main.py` (per-connection streak/warmup; only emit `result.pass=true` when both thresholds met)
- Create: `tests/test_game_pass_v4.py`

**Interfaces produced:** `PASS_WARMUP_MS`, `PASS_STREAK_REQUIRED` exported from `api._game`.

- [ ] **Step 1: Write failing tests** in `tests/test_game_pass_v4.py`:
```python
import base64, time, cv2, numpy as np
from fastapi.testclient import TestClient
import api._game as game
from api.main import app

def _jpeg() -> str:
    bgr = np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr); assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _patch_pass(monkeypatch, top_word="blue"):
    monkeypatch.setattr(game, "preprocess_frames", lambda f: np.zeros((1,1), dtype=np.float32))
    monkeypatch.setattr(game, "score_window", lambda f, s, **k: {
        "top_word": top_word, "confidence": 0.99,
        "topk": [{"word": top_word, "confidence": 0.99}],
        "ranked_words": [top_word, "green", "red", "white"],
    })

def test_pass_requires_streak_and_warmup(monkeypatch):
    _patch_pass(monkeypatch, top_word="blue")
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        # First scoring window during warmup: must emit progress, not result
        for _ in range(game.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
        msg = ws.receive_json()
        assert msg["type"] == "progress", msg
        # Wait past warmup, then send enough to complete the streak
        time.sleep((game.PASS_WARMUP_MS / 1000.0) + 0.05)
        result = None
        for _ in range(game.PASS_STREAK_REQUIRED * game.SCORE_EVERY_N_FRAMES + game.SCORE_EVERY_N_FRAMES):
            ws.send_json({"type": "frame", "data": _jpeg()})
            m = ws.receive_json()
            if m["type"] == "result":
                result = m; break
        assert result is not None and result["pass"] is True and result["word"] == "blue"

def test_no_pass_when_wrong_word(monkeypatch):
    _patch_pass(monkeypatch, top_word="green")  # wrong target
    client = TestClient(app)
    with client.websocket_connect("/ws/game") as ws:
        ws.send_json({"type": "start_round", "slot_index": 1, "target": "blue"})
        time.sleep((game.PASS_WARMUP_MS / 1000.0) + 0.05)
        saw_pass = False
        for _ in range(20):
            ws.send_json({"type": "frame", "data": _jpeg()})
            m = ws.receive_json()
            if m["type"] == "result" and m["pass"]:
                saw_pass = True; break
        assert not saw_pass
```

- [ ] **Step 2: Run** `/Users/maverick/Desktop/UTS/DeepLearning/ImConvo/.venv/bin/pytest tests/test_game_pass_v4.py -v` → FAIL (constants/handler missing).

- [ ] **Step 3: Add constants** in `api/_game.py` (next to existing tunables block):
```python
PASS_WARMUP_MS = 1200
PASS_STREAK_REQUIRED = 2
```

- [ ] **Step 4: Update `ws_game` in `api/main.py`** to track per-connection streak + warmup:
- Add `import time` at the top with the others (skip if already present).
- Inside `ws_game`, alongside `buffer`, `slot_index`, `target`, `frames_since_score`, add:
```python
round_started_at: float | None = None
streak = 0
last_match_word: str | None = None
```
- In the `start_round` branch, after `frames_since_score = 0` (and BEFORE the validation check), add:
```python
round_started_at = time.monotonic()
streak = 0
last_match_word = None
```
- In the `frame` branch, replace the `if frames_since_score >= game.SCORE_EVERY_N_FRAMES:` block body with:
```python
                if frames_since_score >= game.SCORE_EVERY_N_FRAMES:
                    frames_since_score = 0
                    scored = await _score(list(buffer), slot_index)
                    if scored is None:
                        await ws.send_json({"type": "progress", "word": "", "confidence": 0.0, "topk": [], "face": False})
                    else:
                        match = game.evaluate_pass(slot_index, target, scored["ranked_words"], scored["confidence"])
                        if match and last_match_word == target:
                            streak += 1
                        elif match:
                            streak = 1
                            last_match_word = target
                        else:
                            streak = 0
                            last_match_word = None
                        elapsed_ms = (time.monotonic() - (round_started_at or time.monotonic())) * 1000.0
                        passed = (
                            match
                            and streak >= game.PASS_STREAK_REQUIRED
                            and elapsed_ms >= game.PASS_WARMUP_MS
                        )
                        if passed:
                            await ws.send_json({
                                "type": "result", "pass": True,
                                "word": scored["top_word"],
                                "confidence": scored["confidence"],
                                "target": target,
                            })
                            slot_index, target = None, None
                            round_started_at, streak, last_match_word = None, 0, None
                        else:
                            await ws.send_json({
                                "type": "progress",
                                "word": scored["top_word"],
                                "confidence": scored["confidence"],
                                "topk": scored["topk"],
                                "face": True,
                            })
```
- Add a new helper `_score` next to `_score_and_emit` that only computes (does not send):
```python
async def _score(frames, slot_index: int):
    try:
        arr = await run_in_threadpool(game.preprocess_frames, frames)
        if arr is None:
            return None
        return await run_in_threadpool(game.score_window, arr, slot_index)
    except Exception as exc:
        logger.exception("scoring failed")
        return None
```
- In the `end_round` branch, keep calling `_score_and_emit(..., force_result=True)` so timeout-driven endRound still emits a terminal `result.pass=False`. Reset `round_started_at, streak, last_match_word` there too.

- [ ] **Step 5: Run tests** → all backend tests + the new v4 file should pass:
`/Users/maverick/Desktop/UTS/DeepLearning/ImConvo/.venv/bin/pytest tests/ -v`

- [ ] **Step 6: Commit** `fix(game): warmup + sustained-signal — no auto-pass before mouth`

---

## Task 2: Frontend — FaceStage component (replaces CorridorScene)

**Files:**
- Delete: `frontend/src/components/game/CorridorScene.tsx`, `frontend/src/components/game/WallWord.tsx`, `frontend/src/components/game/__tests__/CorridorScene.test.tsx`
- Create: `frontend/src/components/game/FaceStage.tsx`, `frontend/src/components/game/WordPrompt.tsx`, `frontend/src/components/game/__tests__/FaceStage.test.tsx`, `frontend/src/components/game/__tests__/WordPrompt.test.tsx`

**Interfaces produced:**
- `FaceStage({ videoRef: RefObject<HTMLVideoElement | null>, children?: ReactNode })`
- `WordPrompt({ slot: string, word: string })`

- [ ] **Step 1: Failing test** `FaceStage.test.tsx`:
```tsx
import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { useRef } from "react";
import { FaceStage } from "../FaceStage";
function H() { const r = useRef<HTMLVideoElement | null>(null); return <FaceStage videoRef={r}><span data-testid="kid">k</span></FaceStage>; }
describe("FaceStage", () => {
  it("renders a video and overlay children", () => {
    const { container, getByTestId } = render(<H />);
    expect(container.querySelector("video")).not.toBeNull();
    expect(getByTestId("kid")).toBeInTheDocument();
  });
});
```
And `WordPrompt.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { WordPrompt } from "../WordPrompt";
describe("WordPrompt", () => {
  it("renders slot uppercase and the word", () => {
    render(<WordPrompt slot="command" word="place" />);
    expect(screen.getByText("place")).toBeInTheDocument();
    expect(screen.getByText("command")).toHaveStyle({ textTransform: "uppercase" });
  });
});
```

- [ ] **Step 2: Run** `cd frontend && npm run test -- src/components/game/__tests__/FaceStage.test.tsx src/components/game/__tests__/WordPrompt.test.tsx` → FAIL.

- [ ] **Step 3: Implement** `FaceStage.tsx`:
```tsx
import type { ReactNode, RefObject } from "react";
export function FaceStage({ videoRef, children }: { videoRef: RefObject<HTMLVideoElement | null>; children?: ReactNode }) {
  return (
    <div style={{ position: "relative", width: "100%", aspectRatio: "16/9", borderRadius: 24, overflow: "hidden", background: "#1d2030" }}>
      <video
        ref={videoRef}
        autoPlay playsInline muted
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", opacity: 1 }}
      />
      <div style={{ position: "absolute", inset: 0 }}>{children}</div>
    </div>
  );
}
```
`WordPrompt.tsx`:
```tsx
export function WordPrompt({ slot, word }: { slot: string; word: string }) {
  return (
    <div style={{ position: "absolute", top: 96, left: "50%", transform: "translateX(-50%)", background: "var(--candy-cream)", borderRadius: 16, border: "4px solid #fff", padding: "10px 30px", textAlign: "center" }}>
      <div style={{ fontSize: 12, letterSpacing: 4, color: "#b9803c", textTransform: "uppercase", fontWeight: 600 }}>{slot}</div>
      <div className="bubble" style={{ fontSize: 50, fontWeight: 700, color: "var(--candy-magenta)", lineHeight: 1.04 }}>{word}</div>
    </div>
  );
}
```

- [ ] **Step 4: Delete old corridor files** (and their test).

- [ ] **Step 5: Run** `cd frontend && npm run test -- src/components/game/__tests__/FaceStage.test.tsx src/components/game/__tests__/WordPrompt.test.tsx` → PASS. Then run the full suite — old CorridorScene test is gone so no stragglers; expect green.

- [ ] **Step 6: Commit** `feat(game): face-dominant FaceStage + WordPrompt`

---

## Task 3: Frontend — RailMinimap component

**Files:**
- Create: `frontend/src/components/game/RailMinimap.tsx`, `frontend/src/components/game/__tests__/RailMinimap.test.tsx`

**Interfaces produced:** `RailMinimap({ total: number; roundIndex: number })`

- [ ] **Step 1: Failing test** `RailMinimap.test.tsx`:
```tsx
import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RailMinimap } from "../RailMinimap";
describe("RailMinimap", () => {
  it("renders 6 walls and tags them passed/current/upcoming", () => {
    const { container } = render(<RailMinimap total={6} roundIndex={2} />);
    const walls = container.querySelectorAll("[data-wall]");
    expect(walls.length).toBe(6);
    expect(walls[0].getAttribute("data-state")).toBe("passed");
    expect(walls[1].getAttribute("data-state")).toBe("passed");
    expect(walls[2].getAttribute("data-state")).toBe("current");
    expect(walls[3].getAttribute("data-state")).toBe("upcoming");
  });
  it("includes a candy car", () => {
    const { container } = render(<RailMinimap total={6} roundIndex={0} />);
    expect(container.querySelector("[data-car]")).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run** `cd frontend && npm run test -- src/components/game/__tests__/RailMinimap.test.tsx` → FAIL.

- [ ] **Step 3: Implement** `RailMinimap.tsx`:
```tsx
export function RailMinimap({ total, roundIndex }: { total: number; roundIndex: number }) {
  const w = 180, h = 140;
  const carX = 90;
  const carYStart = 110;
  const carYEnd = 30;
  const carY = carYStart + ((carYEnd - carYStart) * Math.min(roundIndex, total - 1)) / Math.max(total - 1, 1);
  return (
    <div style={{ position: "absolute", top: 16, right: 16, width: w, height: h, borderRadius: 14, border: "3px solid #fff", overflow: "hidden", background: "rgba(38,41,54,0.85)" }}>
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label="rail minimap">
        <polygon points={`0,${h} ${w},${h} ${w*0.66},${h*0.42} ${w*0.34},${h*0.42}`} fill="#383b4a" />
        <line x1={6} y1={h - 5} x2={w*0.34} y2={h*0.42 + 3} stroke="#fff" strokeWidth={8} />
        <line x1={w - 6} y1={h - 5} x2={w*0.66} y2={h*0.42 + 3} stroke="#fff" strokeWidth={8} />
        <line x1={6} y1={h - 5} x2={w*0.34} y2={h*0.42 + 3} stroke="#E24B4A" strokeWidth={8} strokeDasharray="10 10" />
        <line x1={w - 6} y1={h - 5} x2={w*0.66} y2={h*0.42 + 3} stroke="#E24B4A" strokeWidth={8} strokeDasharray="10 10" />
        {Array.from({ length: total }, (_, i) => {
          const state = i < roundIndex ? "passed" : i === roundIndex ? "current" : "upcoming";
          const depth = 1 - i / (total - 1 || 1);
          const ww = 80 * depth + 20;
          const hh = 60 * depth + 16;
          const x = (w - ww) / 2;
          const y = 70 - i * 8;
          const stroke = state === "passed" ? "#5DE0A6" : state === "current" ? "#C9A06A" : "#9c7842";
          const opacity = state === "upcoming" ? 0.45 : 0.95;
          return (
            <g key={i} data-wall data-state={state}>
              <rect x={x} y={y} width={ww} height={hh} rx={3} fill="none" stroke={stroke} strokeWidth={4} opacity={opacity}
                    style={{ transition: "all 450ms ease" }} />
            </g>
          );
        })}
        <g data-car transform={`translate(${carX} ${carY})`} style={{ transition: "transform 450ms ease" }}>
          <rect x={-14} y={-10} width={28} height={14} rx={3} fill="#FF3D9A" stroke="#fff" strokeWidth={2} />
          <circle cx={-8} cy={6} r={3} fill="#1d2030" stroke="#fff" strokeWidth={1.5} />
          <circle cx={8} cy={6} r={3} fill="#1d2030" stroke="#fff" strokeWidth={1.5} />
        </g>
      </svg>
      <div style={{ position: "absolute", top: 6, left: "50%", transform: "translateX(-50%)", fontSize: 10, color: "#fff", fontWeight: 600, background: "rgba(0,0,0,0.5)", padding: "2px 8px", borderRadius: 8 }}>
        wall {Math.min(roundIndex + 1, total)} / {total}
      </div>
    </div>
  );
}
```
Add a `@media (prefers-reduced-motion: reduce) { svg [data-car], svg [data-wall] rect { transition: none !important; } }` rule inside `app/game/game.css`.

- [ ] **Step 4: Run** `cd frontend && npm run test -- src/components/game/__tests__/RailMinimap.test.tsx` → PASS.

- [ ] **Step 5: Commit** `feat(game): rail minimap with advancing car`

---

## Task 4: Wire the new layout into `/game` page

**Files:**
- Modify: `frontend/src/app/game/page.tsx`

- [ ] **Step 1:** Replace `CorridorScene` usage with `FaceStage`, mount `WordPrompt`, `RailMinimap`, and existing chrome as children:
```tsx
"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useGame } from "../../hooks/useGame";
import { useWebcam } from "../../hooks/useWebcam";
import { useGameSocket, type ServerMessage } from "../../hooks/useGameSocket";
import { useRoundLoop } from "../../hooks/useRoundLoop";
import { useSound } from "../../hooks/useSound";
import { FaceStage } from "../../components/game/FaceStage";
import { WordPrompt } from "../../components/game/WordPrompt";
import { RailMinimap } from "../../components/game/RailMinimap";
import { TimerPill } from "../../components/game/TimerPill";
import { ScoreBadge } from "../../components/game/ScoreBadge";
import { ProgressDots } from "../../components/game/ProgressDots";
import { MuteToggle } from "../../components/game/MuteToggle";
import { Confetti } from "../../components/game/Confetti";
import { ResultScreen } from "../../components/game/ResultScreen";

export default function GamePage() {
  const [state, dispatch] = useGame();
  const { videoRef, error, grabFrame, retry } = useWebcam();
  const { muted, toggleMuted, play } = useSound();
  const [confettiKey, setConfettiKey] = useState(0);
  const handleMessageRef = useRef<(m: ServerMessage) => void>(() => {});
  const { connected, startRound, sendFrame, endRound } = useGameSocket((m) => handleMessageRef.current(m));
  const onPass = useCallback(() => { play("pass"); setConfettiKey((k) => k + 1); dispatch({ type: "ROUND_PASS" }); }, [play, dispatch]);
  const loop = useRoundLoop({
    active: state.status === "playing" && connected && !error,
    roundIndex: state.roundIndex,
    target: state.sentence.words[state.roundIndex],
    startRound, endRound, sendFrame, grabFrame,
    onPass,
    onTimeout: () => play("timeout"),
  });
  handleMessageRef.current = loop.handleMessage;
  const wonRef = useRef(false);
  useEffect(() => {
    if (state.status === "won" && !wonRef.current) { wonRef.current = true; play("win"); setConfettiKey((k) => k + 1); }
    if (state.status === "playing") wonRef.current = false;
  }, [state.status, play]);

  const total = state.sentence.words.length;
  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: 16 }}>
      <FaceStage videoRef={videoRef}>
        <div style={{ position: "absolute", top: 14, left: 14, zIndex: 2 }}><MuteToggle muted={muted} onToggle={toggleMuted} /></div>
        <div style={{ position: "absolute", top: 16, left: "50%", transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, zIndex: 2 }}>
          {state.status === "playing" && <TimerPill secondsLeft={loop.secondsLeft} />}
          <ProgressDots total={total} roundIndex={Math.min(state.roundIndex, total - 1)} />
        </div>
        <RailMinimap total={total} roundIndex={state.status === "won" ? total : state.roundIndex} />
        {state.status === "playing" && (
          <WordPrompt slot={state.sentence.slots[state.roundIndex]} word={state.sentence.words[state.roundIndex]} />
        )}
        <div style={{ position: "absolute", bottom: 18, left: 18, zIndex: 2 }}><ScoreBadge score={state.score} /></div>
        {state.status === "won" && <ResultScreen onNewSentence={() => dispatch({ type: "NEW_SENTENCE" })} />}
        {error && (
          <div style={{ position: "absolute", left: 18, right: 18, top: "50%", transform: "translateY(-50%)", textAlign: "center", color: "#fff", background: "rgba(0,0,0,0.6)", borderRadius: 12, padding: 16, zIndex: 3 }}>
            <p style={{ margin: "0 0 10px" }}>{error}</p>
            <button onClick={retry} style={{ background: "var(--candy-magenta)", color: "#fff", border: "3px solid #fff", borderRadius: 999, padding: "8px 20px", fontWeight: 600, cursor: "pointer" }}>Enable camera</button>
          </div>
        )}
      </FaceStage>
      <Confetti fireKey={confettiKey} big={state.status === "won"} />
    </div>
  );
}
```

- [ ] **Step 2:** `cd frontend && npm run test && npx tsc --noEmit` — all green, no dangling imports.
- [ ] **Step 3:** Commit `feat(game): wire face-stage + rail minimap into /game`

---

## Task 5: Manual verification + restart

- [ ] **Step 1:** Restart API via Preview (the backend changed). Confirm `/health` 200; refresh `/game`; play a round — face should be fully visible, word above the face (not on the mouth), rail minimap top-right, car advances per pass, no instant pass before mouthing.
- [ ] **Step 2:** If recognition feels right, commit any tuning of `PASS_WARMUP_MS` / `PASS_STREAK_REQUIRED` you want.
