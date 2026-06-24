"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useGame } from "../../hooks/useGame";
import { useWebcam } from "../../hooks/useWebcam";
import { useGameSocket, type ServerMessage } from "../../hooks/useGameSocket";
import { useRoundLoop } from "../../hooks/useRoundLoop";
import { useSound } from "../../hooks/useSound";
import { FaceStage } from "../../components/game/FaceStage";
import { WordPrompt } from "../../components/game/WordPrompt";
import { MouthBox } from "../../components/game/MouthBox";
import { TimerPill } from "../../components/game/TimerPill";
import { ScoreBadge } from "../../components/game/ScoreBadge";
import { ProgressDots } from "../../components/game/ProgressDots";
import { MuteToggle } from "../../components/game/MuteToggle";
import { Confetti } from "../../components/game/Confetti";
import { ConfidenceMeter } from "../../components/game/ConfidenceMeter";
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
    roundSeconds: 3,
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
    <div style={{ width: "100%", margin: 0, padding: 0 }}>
      <FaceStage videoRef={videoRef}>
        {state.status === "playing" && (
          <MouthBox bbox={loop.meter.bbox} videoRef={videoRef} label={loop.meter.face ? `MOUTH ${Math.round(loop.meter.confidence * 100)}%` : undefined} />
        )}
        <div style={{ position: "absolute", top: 18, left: 18, zIndex: 2 }}><MuteToggle muted={muted} onToggle={toggleMuted} /></div>
        <div style={{ position: "absolute", top: 18, left: "50%", transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, zIndex: 2 }}>
          {state.status === "playing" && <TimerPill secondsLeft={loop.secondsLeft} />}
          <ProgressDots total={total} roundIndex={Math.min(state.roundIndex, total - 1)} />
        </div>
        {state.status === "playing" && (
          <WordPrompt slot={state.sentence.slots[state.roundIndex]} word={state.sentence.words[state.roundIndex]} />
        )}
        <div style={{ position: "absolute", bottom: 18, left: 18, zIndex: 3 }}><ScoreBadge score={state.score} /></div>
        {state.status === "playing" && (
          <div style={{ position: "absolute", bottom: 18, right: 18, width: 260, zIndex: 3 }}>
            <ConfidenceMeter confidence={loop.meter.confidence} word={loop.meter.word} face={loop.meter.face} />
          </div>
        )}
        {state.status === "won" && <ResultScreen onNewSentence={() => dispatch({ type: "NEW_SENTENCE" })} />}
        {error && (
          <div style={{ position: "absolute", left: 18, right: 18, top: "50%", transform: "translateY(-50%)", textAlign: "center", color: "#fff", background: "rgba(0,0,0,0.6)", borderRadius: 12, padding: 16, zIndex: 4 }}>
            <p style={{ margin: "0 0 10px" }}>{error}</p>
            <button onClick={retry} style={{ background: "var(--candy-magenta)", color: "#fff", border: "3px solid #fff", borderRadius: 999, padding: "8px 20px", fontWeight: 600, cursor: "pointer" }}>Enable camera</button>
          </div>
        )}
      </FaceStage>
      <Confetti fireKey={confettiKey} big={state.status === "won"} />
    </div>
  );
}
