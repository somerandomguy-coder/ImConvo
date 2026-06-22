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
  const { videoRef, error, grabFrame, retry } = useWebcam();
  const [meter, setMeter] = useState({ word: "", confidence: 0, face: true });
  const attemptingRef = useRef(false);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const windowTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopCapture = useCallback(() => {
    if (captureIntervalRef.current !== null) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (windowTimeoutRef.current !== null) {
      clearTimeout(windowTimeoutRef.current);
      windowTimeoutRef.current = null;
    }
  }, []);

  const onMessage = useCallback((msg: ServerMessage) => {
    if (msg.type === "progress") {
      setMeter({ word: msg.word, confidence: msg.confidence, face: msg.face });
    } else if (msg.type === "result") {
      attemptingRef.current = false;
      stopCapture();
      dispatch({ type: msg.pass ? "ROUND_PASS" : "ROUND_FAIL" });
      setMeter({ word: "", confidence: 0, face: true });
    }
  }, [dispatch, stopCapture]);

  const { connected, startRound, sendFrame, endRound } = useGameSocket(onMessage);

  const attempt = useCallback(() => {
    if (attemptingRef.current || state.status !== "playing") return;
    attemptingRef.current = true;
    setMeter({ word: "", confidence: 0, face: true });
    startRound(state.roundIndex, state.sentence.words[state.roundIndex]);
    stopCapture();
    captureIntervalRef.current = setInterval(() => {
      const f = grabFrame();
      if (f) sendFrame(f);
    }, FRAME_INTERVAL_MS);
    windowTimeoutRef.current = setTimeout(() => {
      stopCapture();
      if (attemptingRef.current) endRound();
      // Always release the guard so the button stays usable even if the
      // server sends no result (e.g. an empty capture window).
      attemptingRef.current = false;
    }, ROUND_WINDOW_MS);
  }, [state.status, state.roundIndex, state.sentence.words, startRound, sendFrame, endRound, grabFrame, stopCapture]);

  useEffect(() => {
    attemptingRef.current = false;
  }, [state.roundIndex, state.status]);

  useEffect(() => stopCapture, [stopCapture]);

  return (
    <main className="mx-auto max-w-xl space-y-6 p-6">
      <Hud score={state.score} streak={state.streak} bestStreak={state.bestStreak} attempts={state.attempts} />
      {error && (
        <div className="rounded border border-rose-300 bg-rose-50 p-3 text-center">
          <p className="text-rose-700">{error}</p>
          <button
            onClick={retry}
            className="mt-2 rounded bg-rose-600 px-4 py-2 text-sm text-white hover:bg-rose-700"
          >
            Enable camera
          </button>
        </div>
      )}
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
            disabled={!connected}
            className="w-full rounded bg-emerald-600 px-4 py-3 text-white disabled:opacity-50"
          >
            {connected ? "Ready… GO" : "Connecting…"}
          </button>
        </>
      )}
    </main>
  );
}
