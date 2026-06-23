export function WordPrompt({ slot, word }: { slot: string; word: string }) {
  return (
    <div style={{ position: "absolute", top: 96, left: "50%", transform: "translateX(-50%)", background: "var(--candy-cream)", borderRadius: 16, border: "4px solid #fff", padding: "10px 30px", textAlign: "center" }}>
      <div style={{ fontSize: 12, letterSpacing: 4, color: "#b9803c", textTransform: "uppercase", fontWeight: 600 }}>{slot}</div>
      <div className="bubble" style={{ fontSize: 50, fontWeight: 700, color: "var(--candy-magenta)", lineHeight: 1.04 }}>{word}</div>
    </div>
  );
}
