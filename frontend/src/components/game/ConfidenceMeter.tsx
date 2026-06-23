export function ConfidenceMeter({ confidence, word, face }: { confidence: number; word: string; face: boolean }) {
  const pct = Math.max(0, Math.min(100, Math.round(confidence * 100)));
  if (!face) {
    return (
      <div style={{ background: "rgba(0,0,0,0.55)", color: "#FFE25A", padding: "8px 14px", borderRadius: 999, border: "2px solid #fff", fontWeight: 600, textAlign: "center" }}>
        Center your face in frame…
      </div>
    );
  }
  return (
    <div style={{ background: "rgba(0,0,0,0.55)", borderRadius: 14, border: "2px solid #fff", padding: "8px 12px", color: "#fff" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", fontSize: 13, marginBottom: 6 }}>
        <span>hearing: <span style={{ fontWeight: 700, color: "#FFE25A" }}>{word || "…"}</span></span>
        <span style={{ fontWeight: 700, color: "#5DE0A6" }}>{pct}%</span>
      </div>
      <div style={{ height: 12, borderRadius: 999, background: "rgba(255,255,255,0.22)", overflow: "hidden", border: "1.5px solid rgba(255,255,255,0.5)" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: "#5DE0A6", borderRadius: 999, transition: "width 180ms ease" }} />
      </div>
    </div>
  );
}
