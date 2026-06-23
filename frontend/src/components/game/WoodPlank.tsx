const DIGIT_TO_NUM: Record<string, string> = {
  zero: "0", one: "1", two: "2", three: "3", four: "4",
  five: "5", six: "6", seven: "7", eight: "8", nine: "9",
};

function bottomSymbol(slot: string, word: string): string | null {
  if (slot === "digit") return DIGIT_TO_NUM[word] ?? word;
  if (slot === "letter") return word.toUpperCase();
  return null;
}

export function WoodPlank({ word, slot, state: _state = "current" }: { word: string; slot: string; state?: "passed" | "current" | "upcoming" }) {
  const top = word.charAt(0).toUpperCase() + word.slice(1);
  const bottom = bottomSymbol(slot, word);
  const slotLabel = slot.toUpperCase();
  return (
    <div
      style={{
        width: 280,
        height: 360,
        background: "linear-gradient(180deg, #d4ad75 0%, #b78d54 100%)",
        borderTop: "5px solid #a87b3d",
        borderLeft: "5px solid #b89160",
        borderRight: "5px solid #8b6235",
        borderBottom: "5px solid #7a5524",
        borderRadius: 6,
        boxShadow: "inset 0 12px 0 rgba(255,255,255,0.12), inset 0 -12px 0 rgba(0,0,0,0.15), 0 12px 24px rgba(0,0,0,0.35)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 28,
        flexShrink: 0,
        position: "relative",
        overflow: "hidden",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          background:
            "repeating-linear-gradient(180deg, rgba(0,0,0,0) 0 14px, rgba(0,0,0,0.06) 14px 15px, rgba(0,0,0,0) 15px 38px, rgba(0,0,0,0.04) 38px 39px)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: 18,
          left: 0,
          right: 0,
          textAlign: "center",
          fontSize: 12,
          letterSpacing: 4,
          color: "#fffbe8",
          opacity: 0.85,
          fontWeight: 600,
        }}
      >
        {slotLabel}
      </div>
      <div
        className="bubble"
        style={{
          fontSize: bottom ? 52 : 84,
          fontWeight: 700,
          color: "#FF3D9A",
          WebkitTextStroke: bottom ? "3px #fff" : "4px #fff",
          textShadow: "0 3px 0 rgba(0,0,0,0.22)",
          position: "relative",
          lineHeight: 1,
          maxWidth: "92%",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {top}
      </div>
      {bottom !== null && (
        <div
          className="bubble"
          style={{
            fontSize: 120,
            fontWeight: 700,
            color: "#5DE0A6",
            WebkitTextStroke: "5px #fff",
            textShadow: "0 5px 0 rgba(0,0,0,0.28)",
            position: "relative",
            lineHeight: 1,
          }}
        >
          {bottom}
        </div>
      )}
    </div>
  );
}
