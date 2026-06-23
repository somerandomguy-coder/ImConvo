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
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 16 }}>
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
    </div>
  );
}
