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
