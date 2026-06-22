# GRID Lip Game — TikTok "Hole-in-the-Wall" UI/UX Redesign

**Date:** 2026-06-23
**Branch:** `feat/game`
**Status:** Design approved (mockup), pending spec review
**Builds on:** [2026-06-22-grid-lip-game-design.md](2026-06-22-grid-lip-game-design.md) — the working live per-word game. This spec changes only the **frontend presentation + round flow**; the backend recognizer, WebSocket protocol, and easy-mode scoring are unchanged.

## 1. Summary

Restyle the `/game` screen to look and feel like the referenced TikTok "read the word" game (a *hole-in-the-wall* runner): a 3D corridor with red/white **striped rails (đường ray)** down which wooden **wall-boxes (ô đục lỗ)** advance toward the player's webcam. Each wall carries one GRID word in a bubble font; the player silently mouths it to "pass through" the wall, triggering confetti, and the next wall slides forward. Adds a per-word **auto-countdown**, **confetti**, **sound**, **score**, and **6-word progress** — all in a bright candy aesthetic (bubble fonts, periwinkle pill timer, magenta word, lime accents).

The reference uses *voice* + a mic; this game keeps **silent lip-reading** and adds a live **confidence meter** as the feedback the original gets from speech.

## 2. Goals & Non-Goals

### Goals
- Visual match to the reference: corridor + striped rails + advancing wooden wall-boxes + bubble fonts + candy colors.
- **Auto-start countdown per word** (no GO button); continuous live scoring; **timeout just retries** (no game-over — keep no-lives).
- Celebration on pass: **confetti** + wall "opens"/passes + next wall advances.
- **Score**, **6-word progress** (the six walls / dots), and **sound** (SFX + optional music, mute toggle).
- Desktop (landscape) layout. Reuse the existing recognizer, `/ws/game`, and `useGame` reducer where possible.

### Non-Goals (v1)
- True 3D/WebGL engine or AR. The corridor is faked with CSS transforms + SVG (good enough, performant).
- Replicating TikTok's exact AR wooden-box geometry pixel-for-pixel.
- Changing the recognizer, the WS protocol, or backend scoring (easy mode already shipped).
- Lives / game-over, difficulty selector, leaderboard (still deferred).
- Mobile-portrait layout (desktop-first; should not break badly on narrow screens but not tuned).

## 3. Key Decisions (approved in brainstorming)

| Decision | Choice |
|---|---|
| Visual concept | Hole-in-the-wall corridor: striped rails + advancing wooden wall-boxes, word on the front wall |
| Orientation | Desktop landscape, full-bleed camera as the corridor backdrop |
| Round start | **Auto-start** when a word appears — no GO button |
| Scoring | Continuous live scoring over the open window (existing slot-constrained easy mode) |
| Countdown | Per-word ring/pill, default **6 s**; **loops/retries on timeout**, never a loss |
| Feedback | Live confidence meter + bubble word fills/glows; **confetti** on pass |
| Progress | Six wall-boxes down the corridor == the six words; mirrored as candy progress dots |
| Audio | SFX: pass "ding", timeout "whoosh", win fanfare; optional looping music; **mute toggle**, SFX on by default, music off by default |
| Theme | Candy: Fredoka bubble font, magenta word, periwinkle timer, lime success, cream wall, white outlines |

## 4. Architecture

Frontend-only change. The data flow with the backend is unchanged from the base spec: the browser streams JPEG frames over `/ws/game`; the server emits `progress`/`result`. What changes is **how a round is driven** (auto countdown loop instead of a GO click) and **how it's rendered** (corridor scene instead of plain panels).

```
/game page
├─ useGame (reducer)              ── sentence, roundIndex, score, streak, status (reused; minor additions)
├─ useWebcam                      ── stream + grabFrame (reused as-is)
├─ useGameSocket                  ── start_round / sendFrame / end_round, onMessage (reused as-is)
├─ useRoundLoop (NEW)             ── per-word auto countdown + capture loop + timeout-retry
├─ useSound (NEW)                 ── preload + play SFX/music, mute state (localStorage)
└─ components/game/ (restyled + new)
   ├─ CorridorScene (NEW)         ── striped rails + nested wall-frames + advance animation
   ├─ WallWord (NEW)              ── cream board + slot label + bubble word on the front wall
   ├─ TimerPill (NEW)             ── periwinkle pill countdown (Fredoka)
   ├─ ScoreBadge (NEW)            ── star + score, candy style
   ├─ ProgressDots (restyle Hud)  ── six candy dots, current/passed/upcoming
   ├─ ConfidenceMeter (restyle)   ── candy track docked bottom
   ├─ Confetti (NEW)              ── canvas-confetti burst on pass + win
   ├─ WinScreen (restyle ResultScreen) ── celebration + "New sentence"
   └─ MuteToggle (NEW)            ── SFX/music on-off
```

## 5. Components

### 5.1 Theme (`app/game/theme.css` or inline tokens)
- Load **Fredoka** (weights 500/600/700) via `next/font/google` (preferred) or a Google Fonts link.
- Candy tokens (CSS custom properties scoped to `.game-root`): `--candy-magenta:#FF3D9A`, `--candy-periwinkle:#7C8CF8`, `--candy-lime:#5DE0A6`, `--candy-sun:#FFE25A`, `--candy-wood:#C9A06A`, `--candy-cream:#F4E3C6`, `--rail-red:#E24B4A`. Bubble text = colored fill + white `-webkit-text-stroke`.
- Scope candy styles to the game route so the rest of the app keeps its current look.

### 5.2 `CorridorScene`
Renders the corridor with CSS transforms (no WebGL):
- **Camera backdrop:** the `<video>` element fills the scene (object-fit: cover), dimmed slightly so overlays read.
- **Striped rails:** an SVG overlay drawing two converging red/white dashed lines from the bottom corners to the vanishing point (as in the mockup).
- **Wall-boxes:** up to 3–4 wooden frame `<div>`s (thick wooden border, transparent hole), centered on the vanishing point at decreasing size/opacity = depth. The **front wall** is largest and holds `WallWord`.
- **Advance animation:** on pass, the front wall scales up + fades ("you pass through it"), each deeper wall transitions forward one slot (CSS `transform`/`opacity` transitions, ~450 ms). Driven by `roundIndex` so the scene is a pure function of game state. Honor `prefers-reduced-motion` (snap instead of animate).
- Props: `{ words: string[], slots: string[], roundIndex: number }`.

### 5.3 `useRoundLoop` (the new flow)
Replaces the GO-button logic in the page. Responsibilities:
- When a round becomes active (page mounted on a `playing` word), **auto** call `startRound(roundIndex, word)` and begin the frame capture loop (`grabFrame → sendFrame` at ~15 fps).
- Run a **countdown** from `ROUND_SECONDS` (default 6). On `result.pass` → stop, play pass SFX, fire confetti, dispatch `ROUND_PASS`. On countdown hitting 0 without a pass → play timeout SFX, **re-arm**: clear/reset the timer and re-`startRound` (fresh buffer), keep going. Never a loss.
- Cleanup on unmount / round change (clear interval + timeout) — same robustness as the current page (stop capture on early result; release on timeout).
- Exposes `{ secondsLeft, meter }` for the UI.

### 5.4 `useSound`
- Preload small audio files in `frontend/public/sounds/` (`pass.mp3`, `timeout.mp3`, `win.mp3`, optional `music.mp3`).
- `play(name)`; background music loops when enabled. `muted` + `musicOn` persisted to `localStorage` (`imconvo.game.muted`, `imconvo.game.music`). Respect browser autoplay rules (start music only after first user gesture).
- No-op gracefully if a file is missing.

### 5.5 Restyled / new chrome
- `TimerPill` — periwinkle pill, Fredoka, shows `0:0X`; pulses under ~2 s left.
- `ScoreBadge` — star + score, candy.
- `ProgressDots` — six dots: passed = lime + check, current = magenta ring, upcoming = translucent.
- `ConfidenceMeter` — candy track docked at the bottom of the scene; shows top guess + `NN%`; lime fill; "center your face" state when `face:false` (reuse existing logic).
- `Confetti` — wraps `canvas-confetti` (CDN: cdnjs). Burst on each pass; bigger burst on win.
- `WinScreen` — full-scene celebration (confetti, "You cleared the wall!", score/streak), **New sentence** button.
- `MuteToggle` — small speaker icon toggling SFX/music.
- Keep the existing **camera-error UI + "Enable camera" retry**, restyled to candy.

## 6. Round Flow (per word)

1. Word becomes current → `CorridorScene` shows it on the front wall; `useRoundLoop` auto-`startRound` + starts capture + 6 s countdown.
2. Frames stream continuously; `progress` updates `ConfidenceMeter` and (optionally) fills the word.
3. **Pass** (`result.pass`, easy mode = target in slot top-k): pass SFX + confetti + dot→lime; front wall opens, corridor advances; `ROUND_PASS`. If it was word 6 → `WinScreen` + win fanfare.
4. **Timeout** (0 s, no pass): timeout SFX; reset ring; re-`startRound`; retry the same wall. No loss.

## 7. State changes (`useGame`)
Mostly reused. Additions:
- `status` stays `"playing" | "won"` (no `"lost"`).
- Keep `score`, `streak`, `bestStreak`, `roundIndex`, `lastResult`. `attempts` optional.
- No reducer logic change required for the loop (the loop dispatches `ROUND_PASS`/`NEW_SENTENCE`); countdown/meter live in `useRoundLoop` local state, not the reducer.

## 8. Error Handling
- Camera denied / no device → restyled error card + "Enable camera" retry (reuse current `useWebcam` mapping). Countdown does not run until the feed is live.
- WS disconnect → pause loop, auto-reconnect (existing), resume current word.
- Missing audio files → silent no-op (game still fully playable).
- `prefers-reduced-motion` → disable corridor/confetti motion, keep instant state changes.
- Backend scoring error → existing `error` message path; UI shows a brief toast, keeps the round alive.

## 9. Testing
- **Pure/unit (vitest):**
  - `useGame` reducer unchanged behavior (existing tests stay green).
  - `useRoundLoop` (with mocked socket + timers via fake timers): auto-starts a round, advances on `result.pass`, re-arms on timeout without entering a lost state, cleans up timers on unmount.
  - `useSound`: respects `muted`/`musicOn` from localStorage; `play` no-ops when muted; tolerates missing files.
  - `ProgressDots` / `ConfidenceMeter` / `TimerPill` render states (passed/current/upcoming; face-false; low-time pulse).
- **Visual/manual:** the corridor scene, advance animation, confetti, and sound are verified by running `/game` (extends the existing manual-verification task). Webcam/audio mocked in unit tests.
- All existing backend tests remain unchanged and green.

## 10. Tunable Defaults
`ROUND_SECONDS = 6`, capture ~15 fps (reuse), confetti burst sizes (pass small, win large), SFX on / music off by default, corridor depth = up to 4 visible walls, advance animation ~450 ms.

## 11. Dependencies
- `canvas-confetti` (via cdnjs CDN script, consistent with the no-bundler-change preference) or the npm package if preferred.
- `Fredoka` Google font via `next/font`.
- Small royalty-free audio files committed under `frontend/public/sounds/` (or generated/silent placeholders if not provided).

## 12. Out of Scope / Future
Mobile-portrait layout, lives/game-over mode, difficulty selector, online leaderboard, real 3D/AR, custom wall textures per slot.
