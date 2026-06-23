import { WoodPlank } from "./WoodPlank";

const PLANK_WIDTH = 280;
const PLANK_GAP = 80;
const STEP = PLANK_WIDTH + PLANK_GAP;

export function PlankCarousel({ words, slots, roundIndex }: { words: string[]; slots: string[]; roundIndex: number }) {
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 24,
        height: 400,
        overflow: "hidden",
        pointerEvents: "none",
      }}
    >
      {words.map((w, i) => {
        const offset = i - roundIndex;
        const isCurrent = offset === 0;
        const opacity = Math.abs(offset) > 1 ? 0 : isCurrent ? 1 : 0.85;
        const scale = isCurrent ? 1 : 0.92;
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: "50%",
              bottom: 0,
              transform: `translate3d(calc(-50% + ${offset * STEP}px), 0, 0) scale(${scale})`,
              opacity,
              transition:
                "transform 650ms cubic-bezier(0.55, 0.15, 0.35, 1), opacity 450ms ease",
              transformOrigin: "bottom center",
            }}
          >
            <WoodPlank word={w} slot={slots[i]} state={offset < 0 ? "passed" : offset === 0 ? "current" : "upcoming"} />
          </div>
        );
      })}
    </div>
  );
}
