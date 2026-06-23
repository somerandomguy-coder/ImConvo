# GRID Lip Game — TikTok Hole-in-the-Wall UI/UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the `/game` page into the referenced TikTok "hole-in-the-wall" corridor runner — striped rails, wooden wall-boxes advancing toward the webcam, a bubble-font word on the front wall — with an auto-countdown (retry on timeout, no loss), confetti, score, six-word progress, and sound.

**Architecture:** Frontend-only change in the Next.js app. The backend recognizer, `/ws/game` protocol, and easy-mode scoring are untouched. The GO-button flow is replaced by a `useRoundLoop` hook that auto-starts each word, streams frames continuously, runs a per-word countdown, and retries on timeout. A `CorridorScene` renders the corridor (CSS + SVG); candy-themed chrome (timer pill, score, progress dots, confidence meter, confetti, win screen) sits on top. Fredoka font + candy CSS variables are scoped to the `/game` route via a route layout.

**Tech Stack:** Next.js 16 + React 19 + TypeScript, vitest + @testing-library/react, `canvas-confetti` (npm), Fredoka via `next/font/google`.

## Global Constraints

- Branch: `feat/game` (already checked out).
- Frontend only. Do NOT modify `api/`, the recognizer, `/ws/game`, `useGame.ts`, `useWebcam.ts`, `useGameSocket.ts`, `gridGrammar.ts`, or any backend test.
- Reuse existing hooks as-is: `useGame()` → `[state, dispatch]` (state: `sentence{words[],slots[]}, roundIndex, status('playing'|'won'), score, streak, bestStreak, attempts, lastResult`; actions `ROUND_PASS`, `ROUND_FAIL`, `NEW_SENTENCE`); `useWebcam()` → `{ videoRef, error, grabFrame, retry }`; `useGameSocket(onMessage)` → `{ connected, startRound(slotIndex,target), sendFrame(b64), endRound() }` with `ServerMessage` union `progress{word,confidence,topk,face} | result{pass,word,confidence,target} | error{message} | pong`.
- No game-over / no lives. Timeout re-arms the same word. Only terminal state is `won`.
- Tunables: `ROUND_SECONDS = 6`, `FRAME_INTERVAL_MS = 66` (~15 fps). SFX on / music off by default. Mute persisted to `localStorage` (`imconvo.game.muted`, `imconvo.game.music`).
- Candy tokens (exact): magenta `#FF3D9A`, periwinkle `#7C8CF8`, lime `#5DE0A6`, sun `#FFE25A`, wood `#C9A06A`, cream `#F4E3C6`, rail-red `#E24B4A`. Bubble text = colored fill + white `-webkit-text-stroke`.
- Run frontend tests from `frontend/`: `npm run test -- <path>`; typecheck: `npx tsc --noEmit`. Both must pass before each commit.
- Respect `prefers-reduced-motion`: skip corridor/confetti animation, keep instant state changes.

---

## File Structure

**Create:**
- `frontend/src/app/game/layout.tsx` — Fredoka font + `.game-root` wrapper.
- `frontend/src/app/game/game.css` — candy CSS variables + helper classes (scoped to `.game-root`).
- `frontend/src/hooks/useSound.ts` — SFX/music playback + mute state.
- `frontend/src/hooks/useRoundLoop.ts` — auto countdown + capture loop + timeout retry.
- `frontend/src/components/game/TimerPill.tsx`, `ScoreBadge.tsx`, `ProgressDots.tsx`, `MuteToggle.tsx`, `CorridorScene.tsx`, `WallWord.tsx`, `Confetti.tsx`.
- Tests under `frontend/src/hooks/__tests__/` and `frontend/src/components/game/__tests__/`.

**Modify:**
- `frontend/src/components/game/ResultScreen.tsx` — candy win screen (same props).
- `frontend/src/app/game/page.tsx` — wire the new flow + scene.
- `frontend/package.json` — add `canvas-confetti`.

**Removed by user request — no confidence bar:** `ConfidenceMeter.tsx` (and its test) is deleted; the play screen shows NO live meter. Feedback is the confetti/advance on pass only (matches the reference). The camera feed must show the user's face **clearly** (full opacity, minimal overlay tint).

**Deleted (now unused):** `Hud.tsx`, `RoundFeedback.tsx`, `WordPrompt.tsx`, `ConfidenceMeter.tsx` and `__tests__/ConfidenceMeter.test.tsx` — remove in Task 8.

---

## Task 1: Setup — deps, Fredoka font, candy theme, TimerPill

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/src/app/game/layout.tsx`, `frontend/src/app/game/game.css`
- Create: `frontend/src/components/game/TimerPill.tsx`
- Test: `frontend/src/components/game/__tests__/TimerPill.test.tsx`

**Interfaces:**
- Produces: `TimerPill({ secondsLeft }: { secondsLeft: number })` — renders `0:0X` (mm:ss with single-digit seconds zero-padded), pulses when `secondsLeft <= 2` (adds class `timer-low`).
- Produces: `.game-root` scope providing `--candy-*` CSS variables and `--font-fredoka`.

- [ ] **Step 1: Add canvas-confetti**

Run from `frontend/`:
```bash
npm install canvas-confetti@1.9.3
npm install -D @types/canvas-confetti@1.6.4
```

- [ ] **Step 2: Create the candy theme CSS**

Create `frontend/src/app/game/game.css`:
```css
.game-root {
  --candy-magenta: #FF3D9A;
  --candy-periwinkle: #7C8CF8;
  --candy-periwinkle-dark: #4b4fb5;
  --candy-lime: #5DE0A6;
  --candy-sun: #FFE25A;
  --candy-wood: #C9A06A;
  --candy-cream: #F4E3C6;
  --rail-red: #E24B4A;
  font-family: var(--font-fredoka), system-ui, sans-serif;
}
.bubble { -webkit-text-stroke: 3px #fff; font-weight: 700; }
.bubble-thin { -webkit-text-stroke: 1.5px var(--candy-periwinkle-dark); font-weight: 700; }
.timer-low { animation: timer-pulse 0.6s ease-in-out infinite; }
@keyframes timer-pulse { 0%,100% { transform: translateX(-50%) scale(1);} 50% { transform: translateX(-50%) scale(1.08);} }
@media (prefers-reduced-motion: reduce) { .timer-low { animation: none; } }
```

- [ ] **Step 3: Create the game route layout (Fredoka + scope)**

Create `frontend/src/app/game/layout.tsx`:
```tsx
import { Fredoka } from "next/font/google";
import "./game.css";

const fredoka = Fredoka({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-fredoka" });

export default function GameLayout({ children }: { children: React.ReactNode }) {
  return <div className={`${fredoka.variable} game-root`}>{children}</div>;
}
```

- [ ] **Step 4: Write the failing test**

Create `frontend/src/components/game/__tests__/TimerPill.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TimerPill } from "../TimerPill";

describe("TimerPill", () => {
  it("formats seconds as 0:0X", () => {
    render(<TimerPill secondsLeft={6} />);
    expect(screen.getByText("0:06")).toBeInTheDocument();
  });
  it("adds the low-time class under 2 seconds", () => {
    const { container } = render(<TimerPill secondsLeft={2} />);
    expect(container.querySelector(".timer-low")).not.toBeNull();
  });
  it("has no low-time class when there is time left", () => {
    const { container } = render(<TimerPill secondsLeft={5} />);
    expect(container.querySelector(".timer-low")).toBeNull();
  });
});
```

- [ ] **Step 5: Run test to verify it fails**

Run: `npm run test -- src/components/game/__tests__/TimerPill.test.tsx`
Expected: FAIL — cannot resolve `../TimerPill`.

- [ ] **Step 6: Implement TimerPill**

Create `frontend/src/components/game/TimerPill.tsx`:
```tsx
export function TimerPill({ secondsLeft }: { secondsLeft: number }) {
  const s = Math.max(0, Math.ceil(secondsLeft));
  const label = `0:${s.toString().padStart(2, "0")}`;
  const low = s <= 2;
  return (
    <div
      className={low ? "timer-low" : undefined}
      style={{
        position: "absolute", top: 18, left: "50%", transform: "translateX(-50%)",
        background: "var(--candy-periwinkle)", color: "#fff", padding: "6px 26px",
        borderRadius: 999, fontSize: 28, border: "3px solid #fff",
      }}
    >
      <span className="bubble-thin">{label}</span>
    </div>
  );
}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `npm run test -- src/components/game/__tests__/TimerPill.test.tsx`
Expected: PASS (3 passed)

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/app/game/layout.tsx frontend/src/app/game/game.css frontend/src/components/game/TimerPill.tsx frontend/src/components/game/__tests__/TimerPill.test.tsx
git commit -m "feat(game): candy theme, Fredoka font, canvas-confetti dep, TimerPill"
```

---

## Task 2: ScoreBadge + ProgressDots

**Files:**
- Create: `frontend/src/components/game/ScoreBadge.tsx`, `frontend/src/components/game/ProgressDots.tsx`
- Test: `frontend/src/components/game/__tests__/ScoreBadge.test.tsx`, `frontend/src/components/game/__tests__/ProgressDots.test.tsx`

**Interfaces:**
- Produces: `ScoreBadge({ score }: { score: number })` — star icon + score number, candy style.
- Produces: `ProgressDots({ total, roundIndex }: { total: number; roundIndex: number })` — `total` dots; index `< roundIndex` = passed (data-state="passed"), `=== roundIndex` = current (data-state="current"), `> roundIndex` = upcoming (data-state="upcoming").

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/components/game/__tests__/ScoreBadge.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScoreBadge } from "../ScoreBadge";

describe("ScoreBadge", () => {
  it("shows the score", () => {
    render(<ScoreBadge score={12} />);
    expect(screen.getByText("12")).toBeInTheDocument();
  });
});
```

Create `frontend/src/components/game/__tests__/ProgressDots.test.tsx`:
```tsx
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProgressDots } from "../ProgressDots";

describe("ProgressDots", () => {
  it("marks passed, current and upcoming dots", () => {
    const { container } = render(<ProgressDots total={6} roundIndex={2} />);
    const dots = container.querySelectorAll("[data-state]");
    expect(dots).toHaveLength(6);
    expect(dots[0].getAttribute("data-state")).toBe("passed");
    expect(dots[1].getAttribute("data-state")).toBe("passed");
    expect(dots[2].getAttribute("data-state")).toBe("current");
    expect(dots[3].getAttribute("data-state")).toBe("upcoming");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test -- src/components/game/__tests__/ScoreBadge.test.tsx src/components/game/__tests__/ProgressDots.test.tsx`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement ScoreBadge**

Create `frontend/src/components/game/ScoreBadge.tsx`:
```tsx
export function ScoreBadge({ score }: { score: number }) {
  return (
    <div style={{ position: "absolute", bottom: 56, left: 20, display: "flex", alignItems: "center", gap: 7, color: "var(--candy-sun)", fontSize: 38 }}>
      <i className="ti ti-star" aria-hidden="true" style={{ fontSize: 28 }} />
      <span className="bubble">{score}</span>
    </div>
  );
}
```

Note: the app already loads the Tabler icon webfont globally (used elsewhere). If `ti ti-star` does not render in this app, replace the `<i>` with an inline `★` span — verify by eye during Task 9.

- [ ] **Step 4: Implement ProgressDots**

Create `frontend/src/components/game/ProgressDots.tsx`:
```tsx
export function ProgressDots({ total, roundIndex }: { total: number; roundIndex: number }) {
  const dots = Array.from({ length: total }, (_, i) =>
    i < roundIndex ? "passed" : i === roundIndex ? "current" : "upcoming",
  );
  const base = { borderRadius: "50%", border: "2px solid #fff" } as const;
  const styleFor = (state: string) =>
    state === "passed"
      ? { ...base, width: 14, height: 14, background: "var(--candy-lime)" }
      : state === "current"
        ? { ...base, width: 17, height: 17, border: "3px solid #fff", background: "var(--candy-magenta)" }
        : { ...base, width: 14, height: 14, background: "rgba(255,255,255,0.35)" };
  return (
    <div style={{ position: "absolute", top: 74, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 8 }}>
      {dots.map((state, i) => (
        <span key={i} data-state={state} style={styleFor(state)} />
      ))}
    </div>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm run test -- src/components/game/__tests__/ScoreBadge.test.tsx src/components/game/__tests__/ProgressDots.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/game/ScoreBadge.tsx frontend/src/components/game/ProgressDots.tsx frontend/src/components/game/__tests__/ScoreBadge.test.tsx frontend/src/components/game/__tests__/ProgressDots.test.tsx
git commit -m "feat(game): candy ScoreBadge and ProgressDots"
```

---

## Task 3: MuteToggle

**Files:**
- Create: `frontend/src/components/game/MuteToggle.tsx`
- Test: `frontend/src/components/game/__tests__/MuteToggle.test.tsx`

**Interfaces:**
- Produces: `MuteToggle({ muted, onToggle }: { muted: boolean; onToggle: () => void })` — speaker icon button; `aria-label` = `muted ? "Unmute" : "Mute"`.

Note: there is NO ConfidenceMeter in this redesign (removed by user request). `ConfidenceMeter.tsx` and its test are deleted in Task 8.

- [ ] **Step 1: Write the failing MuteToggle test**

Create `frontend/src/components/game/__tests__/MuteToggle.test.tsx`:
```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MuteToggle } from "../MuteToggle";

describe("MuteToggle", () => {
  it("labels by state and fires onToggle", () => {
    const onToggle = vi.fn();
    render(<MuteToggle muted={false} onToggle={onToggle} />);
    const btn = screen.getByLabelText("Mute");
    fireEvent.click(btn);
    expect(onToggle).toHaveBeenCalledOnce();
  });
  it("shows Unmute when muted", () => {
    render(<MuteToggle muted={true} onToggle={() => {}} />);
    expect(screen.getByLabelText("Unmute")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm run test -- src/components/game/__tests__/MuteToggle.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement MuteToggle**

Create `frontend/src/components/game/MuteToggle.tsx`:
```tsx
export function MuteToggle({ muted, onToggle }: { muted: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      aria-label={muted ? "Unmute" : "Mute"}
      style={{ position: "absolute", top: 18, right: 18, width: 40, height: 40, borderRadius: "50%", border: "3px solid #fff", background: "var(--candy-periwinkle)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}
    >
      <i className={muted ? "ti ti-volume-off" : "ti ti-volume"} aria-hidden="true" style={{ fontSize: 20 }} />
    </button>
  );
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npm run test -- src/components/game/__tests__/MuteToggle.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/MuteToggle.tsx frontend/src/components/game/__tests__/MuteToggle.test.tsx
git commit -m "feat(game): MuteToggle control"
```

---

## Task 4: useSound hook

**Files:**
- Create: `frontend/src/hooks/useSound.ts`
- Test: `frontend/src/hooks/__tests__/useSound.test.ts`

**Interfaces:**
- Produces: `useSound()` → `{ muted: boolean, musicOn: boolean, toggleMuted: () => void, play: (name: "pass" | "timeout" | "win") => void }`. Reads/writes `localStorage["imconvo.game.muted"]` (default `false`) and `localStorage["imconvo.game.music"]` (default `false`). `play` is a no-op when `muted`. Audio files live at `/sounds/{name}.mp3`; if an Audio fails to play, swallow the error (game stays playable).

- [ ] **Step 1: Write the failing test**

Create `frontend/src/hooks/__tests__/useSound.test.ts`:
```ts
import { renderHook, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSound } from "../useSound";

describe("useSound", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.spyOn(window.HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  });

  it("defaults to unmuted and plays a cue", () => {
    const playSpy = window.HTMLMediaElement.prototype.play as unknown as ReturnType<typeof vi.fn>;
    const { result } = renderHook(() => useSound());
    expect(result.current.muted).toBe(false);
    act(() => result.current.play("pass"));
    expect(playSpy).toHaveBeenCalled();
  });

  it("does not play when muted, and persists the toggle", () => {
    const { result } = renderHook(() => useSound());
    act(() => result.current.toggleMuted());
    expect(result.current.muted).toBe(true);
    expect(localStorage.getItem("imconvo.game.muted")).toBe("true");
    const playSpy = window.HTMLMediaElement.prototype.play as unknown as ReturnType<typeof vi.fn>;
    playSpy.mockClear();
    act(() => result.current.play("win"));
    expect(playSpy).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm run test -- src/hooks/__tests__/useSound.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement useSound**

Create `frontend/src/hooks/useSound.ts`:
```ts
import { useCallback, useEffect, useRef, useState } from "react";

type Cue = "pass" | "timeout" | "win";
const MUTED_KEY = "imconvo.game.muted";
const MUSIC_KEY = "imconvo.game.music";

function readBool(key: string): boolean {
  if (typeof localStorage === "undefined") return false;
  return localStorage.getItem(key) === "true";
}

export function useSound() {
  const [muted, setMuted] = useState(false);
  const [musicOn, setMusicOn] = useState(false);
  const cache = useRef<Record<string, HTMLAudioElement>>({});

  useEffect(() => {
    setMuted(readBool(MUTED_KEY));
    setMusicOn(readBool(MUSIC_KEY));
  }, []);

  const play = useCallback((name: Cue) => {
    if (muted || typeof Audio === "undefined") return;
    try {
      let el = cache.current[name];
      if (!el) {
        el = new Audio(`/sounds/${name}.mp3`);
        cache.current[name] = el;
      }
      el.currentTime = 0;
      void el.play().catch(() => {});
    } catch {
      /* audio unavailable — ignore */
    }
  }, [muted]);

  const toggleMuted = useCallback(() => {
    setMuted((m) => {
      const next = !m;
      if (typeof localStorage !== "undefined") localStorage.setItem(MUTED_KEY, String(next));
      return next;
    });
  }, []);

  return { muted, musicOn, toggleMuted, play };
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npm run test -- src/hooks/__tests__/useSound.test.ts`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useSound.ts frontend/src/hooks/__tests__/useSound.test.ts
git commit -m "feat(game): useSound hook with mute persistence"
```

---

## Task 5: useRoundLoop hook (core flow)

**Files:**
- Create: `frontend/src/hooks/useRoundLoop.ts`
- Test: `frontend/src/hooks/__tests__/useRoundLoop.test.ts`

**Interfaces:**
- Consumes: `ServerMessage` from `./useGameSocket`.
- Produces:
  ```ts
  type RoundLoopParams = {
    active: boolean;            // playing && connected
    roundIndex: number;         // slot index AND restart key
    target: string;             // current word
    roundSeconds?: number;      // default 6
    frameIntervalMs?: number;   // default 66
    startRound: (slotIndex: number, target: string) => void;
    endRound: () => void;
    sendFrame: (b64: string) => void;
    grabFrame: () => string | null;
    onPass: () => void;
    onTimeout?: () => void;
  };
  type RoundLoop = {
    secondsLeft: number;
    meter: { word: string; confidence: number; face: boolean };
    handleMessage: (msg: ServerMessage) => void;
  };
  useRoundLoop(params: RoundLoopParams): RoundLoop
  ```
- Behavior: when `active` and `roundIndex`/`target` change, it `startRound(roundIndex, target)`, resets `secondsLeft` to `roundSeconds`, resets the meter, and begins a capture interval (`grabFrame → sendFrame`) plus a 1s countdown tick. On `handleMessage`: `progress` → update meter; `result` with `pass:true` → stop capture+timer and call `onPass`. On countdown reaching 0 without a pass → call `endRound`, `onTimeout?.()`, then re-`startRound` and reset `secondsLeft` (retry; never a loss). All timers cleared on unmount and on `active` going false.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/hooks/__tests__/useRoundLoop.test.ts`:
```ts
import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useRoundLoop } from "../useRoundLoop";

function makeParams(over = {}) {
  return {
    active: true,
    roundIndex: 0,
    target: "blue",
    roundSeconds: 6,
    frameIntervalMs: 66,
    startRound: vi.fn(),
    endRound: vi.fn(),
    sendFrame: vi.fn(),
    grabFrame: vi.fn(() => "JPEGDATA"),
    onPass: vi.fn(),
    onTimeout: vi.fn(),
    ...over,
  };
}

describe("useRoundLoop", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("auto-starts the round and streams frames while active", () => {
    const p = makeParams();
    renderHook(() => useRoundLoop(p));
    expect(p.startRound).toHaveBeenCalledWith(0, "blue");
    act(() => { vi.advanceTimersByTime(200); });
    expect(p.sendFrame).toHaveBeenCalledWith("JPEGDATA");
  });

  it("passes on a winning result and stops capturing", () => {
    const p = makeParams();
    const { result } = renderHook(() => useRoundLoop(p));
    act(() => result.current.handleMessage({ type: "result", pass: true, word: "blue", confidence: 0.9, target: "blue" }));
    expect(p.onPass).toHaveBeenCalledOnce();
    p.sendFrame.mockClear();
    act(() => { vi.advanceTimersByTime(300); });
    expect(p.sendFrame).not.toHaveBeenCalled();
  });

  it("re-arms (retries) on timeout without calling onPass", () => {
    const p = makeParams({ roundSeconds: 2 });
    renderHook(() => useRoundLoop(p));
    expect(p.startRound).toHaveBeenCalledTimes(1);
    act(() => { vi.advanceTimersByTime(2000); });
    expect(p.endRound).toHaveBeenCalled();
    expect(p.onTimeout).toHaveBeenCalled();
    expect(p.startRound).toHaveBeenCalledTimes(2);
    expect(p.onPass).not.toHaveBeenCalled();
  });

  it("updates the meter on progress", () => {
    const p = makeParams();
    const { result } = renderHook(() => useRoundLoop(p));
    act(() => result.current.handleMessage({ type: "progress", word: "blue", confidence: 0.5, topk: [], face: true }));
    expect(result.current.meter).toEqual({ word: "blue", confidence: 0.5, face: true });
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm run test -- src/hooks/__tests__/useRoundLoop.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement useRoundLoop**

Create `frontend/src/hooks/useRoundLoop.ts`:
```ts
import { useCallback, useEffect, useRef, useState } from "react";
import type { ServerMessage } from "./useGameSocket";

type RoundLoopParams = {
  active: boolean;
  roundIndex: number;
  target: string;
  roundSeconds?: number;
  frameIntervalMs?: number;
  startRound: (slotIndex: number, target: string) => void;
  endRound: () => void;
  sendFrame: (b64: string) => void;
  grabFrame: () => string | null;
  onPass: () => void;
  onTimeout?: () => void;
};

export function useRoundLoop(params: RoundLoopParams) {
  const { active, roundIndex, target, roundSeconds = 6, frameIntervalMs = 66 } = params;
  const [secondsLeft, setSecondsLeft] = useState(roundSeconds);
  const [meter, setMeter] = useState({ word: "", confidence: 0, face: true });

  const ref = useRef(params);
  ref.current = params;
  const captureRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const passedRef = useRef(false);

  const stop = useCallback(() => {
    if (captureRef.current !== null) { clearInterval(captureRef.current); captureRef.current = null; }
    if (tickRef.current !== null) { clearInterval(tickRef.current); tickRef.current = null; }
  }, []);

  const arm = useCallback(() => {
    const p = ref.current;
    stop();
    passedRef.current = false;
    setMeter({ word: "", confidence: 0, face: true });
    setSecondsLeft(p.roundSeconds ?? 6);
    p.startRound(p.roundIndex, p.target);
    captureRef.current = setInterval(() => {
      const f = p.grabFrame();
      if (f) p.sendFrame(f);
    }, p.frameIntervalMs ?? 66);
    tickRef.current = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          const pp = ref.current;
          pp.endRound();
          pp.onTimeout?.();
          passedRef.current = false;
          pp.startRound(pp.roundIndex, pp.target);
          return pp.roundSeconds ?? 6;
        }
        return s - 1;
      });
    }, 1000);
  }, [stop]);

  useEffect(() => {
    if (!active) { stop(); return; }
    arm();
    return stop;
  }, [active, roundIndex, target, arm, stop]);

  const handleMessage = useCallback((msg: ServerMessage) => {
    if (msg.type === "progress") {
      setMeter({ word: msg.word, confidence: msg.confidence, face: msg.face });
    } else if (msg.type === "result" && msg.pass && !passedRef.current) {
      passedRef.current = true;
      stop();
      ref.current.onPass();
    }
  }, [stop]);

  return { secondsLeft, meter, handleMessage };
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npm run test -- src/hooks/__tests__/useRoundLoop.test.ts`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useRoundLoop.ts frontend/src/hooks/__tests__/useRoundLoop.test.ts
git commit -m "feat(game): useRoundLoop — auto countdown, live capture, timeout retry"
```

---

## Task 6: CorridorScene + WallWord

**Files:**
- Create: `frontend/src/components/game/WallWord.tsx`, `frontend/src/components/game/CorridorScene.tsx`
- Test: `frontend/src/components/game/__tests__/CorridorScene.test.tsx`

**Interfaces:**
- Produces: `WallWord({ slot, word }: { slot: string; word: string })` — cream board with uppercase slot label + bubble-font magenta word.
- Produces: `CorridorScene({ videoRef, words, slots, roundIndex, children }: { videoRef: React.RefObject<HTMLVideoElement | null>; words: string[]; slots: string[]; roundIndex: number; children?: React.ReactNode })` — full-bleed `<video>` backdrop + SVG rails + up to 4 receding wooden wall-frames; the front wall renders `WallWord` for `words[roundIndex]`. `children` overlay (timer, dots, score, meter) render on top.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/game/__tests__/CorridorScene.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useRef } from "react";
import { CorridorScene } from "../CorridorScene";

function Harness() {
  const ref = useRef<HTMLVideoElement>(null);
  return (
    <CorridorScene
      videoRef={ref}
      words={["bin", "blue", "place", "at", "f", "now"]}
      slots={["command", "color", "preposition", "letter", "digit", "adverb"]}
      roundIndex={2}
    />
  );
}

describe("CorridorScene", () => {
  it("shows the current word and slot on the front wall", () => {
    render(<Harness />);
    expect(screen.getByText("place")).toBeInTheDocument();
    expect(screen.getByText("preposition")).toBeInTheDocument();
  });
  it("renders a video backdrop", () => {
    const { container } = render(<Harness />);
    expect(container.querySelector("video")).not.toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm run test -- src/components/game/__tests__/CorridorScene.test.tsx`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement WallWord**

Create `frontend/src/components/game/WallWord.tsx`:
```tsx
export function WallWord({ slot, word }: { slot: string; word: string }) {
  return (
    <div style={{ position: "absolute", top: 150, left: "50%", transform: "translateX(-50%)", background: "var(--candy-cream)", borderRadius: 16, border: "4px solid #fff", padding: "10px 30px", textAlign: "center" }}>
      <div style={{ fontSize: 12, letterSpacing: 4, color: "#b9803c", textTransform: "uppercase", fontWeight: 600 }}>{slot}</div>
      <div className="bubble" style={{ fontSize: 50, color: "var(--candy-magenta)", lineHeight: 1.04 }}>{word}</div>
    </div>
  );
}
```

- [ ] **Step 4: Implement CorridorScene**

Create `frontend/src/components/game/CorridorScene.tsx`:
```tsx
import type { RefObject, ReactNode } from "react";
import { WallWord } from "./WallWord";

const WALLS = [
  { x: 296, y: 172, w: 88, h: 84, sw: 9, color: "#9c7842", opacity: 0.7 },
  { x: 255, y: 150, w: 170, h: 150, sw: 15, color: "#b78d54", opacity: 0.88 },
  { x: 198, y: 118, w: 284, h: 234, sw: 26, color: "#C9A06A", opacity: 1 },
];

export function CorridorScene({
  videoRef, words, slots, roundIndex, children,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  words: string[]; slots: string[]; roundIndex: number; children?: ReactNode;
}) {
  return (
    <div style={{ position: "relative", width: "100%", height: 460, borderRadius: 24, overflow: "hidden", background: "#20222e" }}>
      <video ref={videoRef} autoPlay playsInline muted style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", opacity: 1 }} />
      <svg viewBox="0 0 680 460" width="100%" height="100%" style={{ position: "absolute", inset: 0 }} role="img" aria-label="corridor">
        <polygon points="0,460 680,460 430,250 250,250" fill="rgba(40,43,58,0.25)" />
        <line x1="12" y1="452" x2="250" y2="255" stroke="#ffffff" strokeWidth="16" />
        <line x1="668" y1="452" x2="430" y2="255" stroke="#ffffff" strokeWidth="16" />
        <line x1="12" y1="452" x2="250" y2="255" stroke="var(--rail-red)" strokeWidth="16" strokeDasharray="22 22" />
        <line x1="668" y1="452" x2="430" y2="255" stroke="var(--rail-red)" strokeWidth="16" strokeDasharray="22 22" />
        {WALLS.map((wll, i) => (
          <rect key={i} x={wll.x} y={wll.y} width={wll.w} height={wll.h} rx={8} fill="none" stroke={wll.color} strokeWidth={wll.sw} opacity={wll.opacity} />
        ))}
      </svg>
      <WallWord slot={slots[roundIndex]} word={words[roundIndex]} />
      {children}
    </div>
  );
}
```

Note (animation): the static walls above match the approved mockup. A follow-up polish step (Task 9 manual pass) may add a CSS `transition` keyed on `roundIndex` to animate the front wall opening and the deeper walls advancing; the static version is the correctness baseline and must render first.

- [ ] **Step 5: Run it to verify it passes**

Run: `npm run test -- src/components/game/__tests__/CorridorScene.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/game/WallWord.tsx frontend/src/components/game/CorridorScene.tsx frontend/src/components/game/__tests__/CorridorScene.test.tsx
git commit -m "feat(game): CorridorScene + WallWord (striped rails, wall-boxes, bubble word)"
```

---

## Task 7: Confetti wrapper

**Files:**
- Create: `frontend/src/components/game/Confetti.tsx`
- Test: `frontend/src/components/game/__tests__/Confetti.test.tsx`

**Interfaces:**
- Produces: `Confetti({ fireKey, big }: { fireKey: number; big?: boolean })` — fires a `canvas-confetti` burst whenever `fireKey` changes (and `fireKey > 0`); `big` = larger burst (win). Renders nothing visible (the lib draws to its own canvas). Must not throw if confetti is unavailable.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/game/__tests__/Confetti.test.tsx`:
```tsx
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const fire = vi.fn();
vi.mock("canvas-confetti", () => ({ default: (...args: unknown[]) => fire(...args) }));

import { Confetti } from "../Confetti";

describe("Confetti", () => {
  it("does not fire on initial key 0", () => {
    render(<Confetti fireKey={0} />);
    expect(fire).not.toHaveBeenCalled();
  });
  it("fires when fireKey increases", () => {
    const { rerender } = render(<Confetti fireKey={0} />);
    rerender(<Confetti fireKey={1} />);
    expect(fire).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm run test -- src/components/game/__tests__/Confetti.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement Confetti**

Create `frontend/src/components/game/Confetti.tsx`:
```tsx
import { useEffect, useRef } from "react";
import confetti from "canvas-confetti";

export function Confetti({ fireKey, big = false }: { fireKey: number; big?: boolean }) {
  const last = useRef(0);
  useEffect(() => {
    if (fireKey === last.current || fireKey <= 0) return;
    last.current = fireKey;
    try {
      confetti({
        particleCount: big ? 220 : 90,
        spread: big ? 120 : 70,
        startVelocity: big ? 55 : 40,
        origin: { y: 0.6 },
      });
    } catch {
      /* confetti unavailable — ignore */
    }
  }, [fireKey, big]);
  return null;
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npm run test -- src/components/game/__tests__/Confetti.test.tsx`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/Confetti.tsx frontend/src/components/game/__tests__/Confetti.test.tsx
git commit -m "feat(game): confetti burst wrapper"
```

---

## Task 8: WinScreen restyle + page integration + cleanup

**Files:**
- Modify: `frontend/src/components/game/ResultScreen.tsx`
- Rewrite: `frontend/src/app/game/page.tsx`
- Delete: `frontend/src/components/game/Hud.tsx`, `frontend/src/components/game/RoundFeedback.tsx`, `frontend/src/components/game/WordPrompt.tsx` (now unused)
- Test: existing `ConfidenceMeter`/`ScoreBadge`/etc. tests + full suite + `npx tsc --noEmit`

**Interfaces:**
- `ResultScreen({ onNewSentence }: { onNewSentence: () => void })` — unchanged prop; candy win styling; renders a button labelled `New sentence` (keep this exact text — any existing test relies on it).

- [ ] **Step 1: Restyle ResultScreen (keep prop + button text)**

Replace `frontend/src/components/game/ResultScreen.tsx` with:
```tsx
export function ResultScreen({ onNewSentence }: { onNewSentence: () => void }) {
  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 18, background: "rgba(32,34,46,0.82)", textAlign: "center", padding: 24 }}>
      <p className="bubble" style={{ fontSize: 44, color: "var(--candy-sun)", margin: 0 }}>You cleared the wall!</p>
      <button
        onClick={onNewSentence}
        style={{ background: "var(--candy-magenta)", color: "#fff", border: "3px solid #fff", borderRadius: 999, padding: "12px 28px", fontSize: 20, fontWeight: 600, cursor: "pointer" }}
      >
        New sentence
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Rewrite the page to wire the new flow**

Replace `frontend/src/app/game/page.tsx` with:
```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGame } from "../../hooks/useGame";
import { useWebcam } from "../../hooks/useWebcam";
import { useGameSocket, type ServerMessage } from "../../hooks/useGameSocket";
import { useRoundLoop } from "../../hooks/useRoundLoop";
import { useSound } from "../../hooks/useSound";
import { CorridorScene } from "../../components/game/CorridorScene";
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

  const onPass = useCallback(() => {
    play("pass");
    setConfettiKey((k) => k + 1);
    dispatch({ type: "ROUND_PASS" });
  }, [play, dispatch]);

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

  return (
    <main style={{ maxWidth: 720, margin: "0 auto", padding: 16 }}>
      <CorridorScene videoRef={videoRef} words={state.sentence.words} slots={state.sentence.slots} roundIndex={state.roundIndex}>
        <ScoreBadge score={state.score} />
        <ProgressDots total={state.sentence.words.length} roundIndex={state.roundIndex} />
        <MuteToggle muted={muted} onToggle={toggleMuted} />
        {state.status === "playing" && <TimerPill secondsLeft={loop.secondsLeft} />}
        {state.status === "won" && <ResultScreen onNewSentence={() => dispatch({ type: "NEW_SENTENCE" })} />}
        {error && (
          <div style={{ position: "absolute", left: 18, right: 18, top: "50%", transform: "translateY(-50%)", textAlign: "center", color: "#fff", background: "rgba(0,0,0,0.6)", borderRadius: 12, padding: 16 }}>
            <p style={{ margin: "0 0 10px" }}>{error}</p>
            <button onClick={retry} style={{ background: "var(--candy-magenta)", color: "#fff", border: "3px solid #fff", borderRadius: 999, padding: "8px 20px", fontWeight: 600, cursor: "pointer" }}>Enable camera</button>
          </div>
        )}
      </CorridorScene>
      <Confetti fireKey={confettiKey} big={state.status === "won"} />
    </main>
  );
}
```

- [ ] **Step 3: Delete now-unused components (incl. the removed ConfidenceMeter)**

```bash
git rm frontend/src/components/game/Hud.tsx frontend/src/components/game/RoundFeedback.tsx frontend/src/components/game/WordPrompt.tsx frontend/src/components/game/ConfidenceMeter.tsx frontend/src/components/game/__tests__/ConfidenceMeter.test.tsx
```
Then confirm nothing else imports them: `grep -rl "Hud\|RoundFeedback\|WordPrompt\|ConfidenceMeter" frontend/src` should return no source files (delete any stragglers).

- [ ] **Step 4: Run the full frontend suite + typecheck**

Run: `npm run test` then `npx tsc --noEmit`
Expected: all tests pass; no type errors. (If a deleted component had a test, remove it.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(game): wire corridor play screen — auto-loop, confetti, sound, win"
```

---

## Task 9: Manual end-to-end verification + motion polish

**Files:** optionally `CorridorScene.tsx` (add reduced-motion-aware advance transition), `frontend/public/sounds/` (drop in `pass.mp3`, `timeout.mp3`, `win.mp3` if available).

- [ ] **Step 1: Start servers**

Backend (from repo root): `/Users/maverick/Desktop/UTS/DeepLearning/ImConvo/.venv/bin/python -m api.main --host 127.0.0.1 --port 8001`
Frontend (from `frontend/`): `npm run dev`

- [ ] **Step 2: Play and verify**

Open `http://localhost:3000/game`. Verify: the user's face shows **clearly** (full-opacity camera) with the corridor + striped rails over it; there is **no confidence bar**; the front wall shows the current word in the bubble font; the countdown pill ticks and auto-starts each word (no GO button); mouthing a recognized word fires confetti + advances the progress dot + shows the next word; letting the timer hit 0 resets and retries the same word (no game-over); clearing all six shows the win screen + confetti; the mute toggle silences cues.

- [ ] **Step 3: (Optional polish) animate the wall advance**

If the advance feels static, add to the front wall `<rect>` and `WallWord` a `transition: transform 0.45s ease, opacity 0.45s ease` and key a brief scale/opacity change off `roundIndex` (gate behind a `prefers-reduced-motion` check). Re-run `npm run test` + `npx tsc --noEmit`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix(game): polish corridor advance + verification fixes"
```

---

## Self-Review Notes

- **Spec coverage:** corridor + rails + wall-boxes + bubble word (Task 6); auto-countdown + live scoring + timeout-retry, no GO (Task 5, wired Task 8); confetti on pass/win (Task 7, Task 8); score / 6-word progress (Task 2); sound + mute (Tasks 3–4, 8); candy theme + Fredoka (Task 1); camera-error retry preserved (Task 8); reduced-motion (Task 1 CSS, Task 9 polish). Backend untouched (Global Constraints).
- **Type consistency:** `ServerMessage` reused from `useGameSocket`; `useRoundLoop` params/returns match the page wiring (the page uses `loop.secondsLeft`; `loop.meter` is intentionally unused now that the meter is removed); `ResultScreen` keeps its existing prop + "New sentence" button text; `ProgressDots` `data-state` values (`passed`/`current`/`upcoming`) consistent between component and test.
- **Confidence bar removed (user request):** no `ConfidenceMeter` on the play screen; `ConfidenceMeter.tsx` + its test deleted in Task 8. Camera shows the face clearly (video opacity 1, light corridor tint).
- **No placeholders:** every component and hook has complete code; the only optional items (Task 9 motion polish, audio binaries) are explicitly optional and the game is fully playable without them.
