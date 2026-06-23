export function WordPrompt({ slot, word }: { slot: string; word: string }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 24,
        right: 24,
        background: "var(--candy-cream)",
        borderRadius: 20,
        border: "5px solid #fff",
        boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
        padding: "14px 36px 22px",
        textAlign: "center",
        zIndex: 3,
        minWidth: 260,
      }}
    >
      <div
        style={{
          fontSize: 13,
          letterSpacing: 5,
          color: "#b9803c",
          textTransform: "uppercase",
          fontWeight: 600,
          marginBottom: 4,
        }}
      >
        {slot}
      </div>
      <div
        className="bubble"
        style={{
          fontSize: 96,
          fontWeight: 700,
          color: "var(--candy-magenta)",
          WebkitTextStroke: "4px #fff",
          textShadow: "0 6px 0 rgba(0,0,0,0.22)",
          lineHeight: 1,
        }}
      >
        {word}
      </div>
    </div>
  );
}
