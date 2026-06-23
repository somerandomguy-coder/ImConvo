export function ScoreBadge({ score }: { score: number }) {
  return (
    <div style={{ position: "absolute", bottom: 56, left: 20, display: "flex", alignItems: "center", gap: 7, color: "var(--candy-sun)", fontSize: 38 }}>
      <i className="ti ti-star" aria-hidden="true" style={{ fontSize: 28 }} />
      <span className="bubble">{score}</span>
    </div>
  );
}
