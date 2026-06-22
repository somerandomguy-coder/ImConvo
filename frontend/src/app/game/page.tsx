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
