export function ResultScreen({ onNewSentence }: { onNewSentence: () => void }) {
  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 18, background: "rgba(32,34,46,0.82)", textAlign: "center", padding: 24 }}>
      <p className="bubble" style={{ fontSize: 44, color: "var(--candy-sun)", margin: 0 }}>You cleared the wall!</p>
      <button
        onClick={onNewSentence}
        style={{ background: "var(--candy-magenta)", color: "#fff", border: "3px solid #fff", borderRadius: 999, padding: "12px 28px", fontSize: 20, fontWeight: 600, cursor: "pointer" }}
      >
        New sentence
      </button>
    </div>
  );
}
