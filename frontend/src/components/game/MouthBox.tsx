import type { LipBBox } from "../../hooks/useGameSocket";

export function MouthBox({ bbox, label }: { bbox: LipBBox | null; label?: string }) {
  if (!bbox) return null;
  const left = `${Math.max(0, Math.min(1, bbox.x)) * 100}%`;
  const top = `${Math.max(0, Math.min(1, bbox.y)) * 100}%`;
  const width = `${Math.max(0, Math.min(1, bbox.w)) * 100}%`;
  const height = `${Math.max(0, Math.min(1, bbox.h)) * 100}%`;
  return (
    <div
      data-testid="mouth-box"
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        border: "3px solid #5DE0A6",
        borderRadius: 4,
        boxShadow: "0 0 0 1px rgba(0,0,0,0.4), 0 0 12px rgba(93,224,166,0.45)",
        pointerEvents: "none",
        transition: "left 120ms linear, top 120ms linear, width 120ms linear, height 120ms linear",
        zIndex: 2,
      }}
    >
      <span
        style={{
          position: "absolute",
          left: -3,
          top: -3,
          width: 10,
          height: 10,
          borderTop: "3px solid #FFE25A",
          borderLeft: "3px solid #FFE25A",
        }}
      />
      <span
        style={{
          position: "absolute",
          right: -3,
          top: -3,
          width: 10,
          height: 10,
          borderTop: "3px solid #FFE25A",
          borderRight: "3px solid #FFE25A",
        }}
      />
      <span
        style={{
          position: "absolute",
          left: -3,
          bottom: -3,
          width: 10,
          height: 10,
          borderBottom: "3px solid #FFE25A",
          borderLeft: "3px solid #FFE25A",
        }}
      />
      <span
        style={{
          position: "absolute",
          right: -3,
          bottom: -3,
          width: 10,
          height: 10,
          borderBottom: "3px solid #FFE25A",
          borderRight: "3px solid #FFE25A",
        }}
      />
      {label && (
        <div
          style={{
            position: "absolute",
            top: -24,
            left: -3,
            background: "#5DE0A6",
            color: "#0f2a1d",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 700,
            whiteSpace: "nowrap",
            fontFamily: "monospace",
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
}
