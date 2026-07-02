import { useEffect, useRef } from "react";
import confetti from "canvas-confetti";

export function Confetti({ fireKey, big = false }: { fireKey: number; big?: boolean }) {
  const last = useRef(0);
  useEffect(() => {
    if (fireKey === last.current || fireKey <= 0) return;
    if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      last.current = fireKey;
      return;
    }
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
