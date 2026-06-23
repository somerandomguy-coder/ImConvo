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
  const { active, roundIndex, target, roundSeconds = 6 } = params;
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
      const pp = ref.current;
      const f = pp.grabFrame();
      if (f) pp.sendFrame(f);
    }, p.frameIntervalMs ?? 66);
    tickRef.current = setInterval(() => {
      setSecondsLeft((s) => (s <= 0 ? 0 : s - 1));
    }, 1000);
  }, [stop]);

  useEffect(() => {
    if (!active) { stop(); return; }
    arm();
    return stop;
  }, [active, roundIndex, target, arm, stop]);

  useEffect(() => {
    if (!active || secondsLeft > 0 || passedRef.current) return;
    const p = ref.current;
    p.endRound();
    p.onTimeout?.();
    arm();
  }, [active, secondsLeft, arm]);

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
