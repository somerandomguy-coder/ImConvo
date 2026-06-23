export function ProgressDots({ total, roundIndex }: { total: number; roundIndex: number }) {
  const dots = Array.from({ length: total }, (_, i) =>
    i < roundIndex ? "passed" : i === roundIndex ? "current" : "upcoming",
  );
  const base = { borderRadius: "50%", border: "2px solid #fff" } as const;
  const styleFor = (state: string) =>
    state === "passed"
      ? { ...base, width: 14, height: 14, background: "var(--candy-lime)" }
      : state === "current"
        ? { ...base, width: 17, height: 17, border: "3px solid #fff", background: "var(--candy-magenta)" }
        : { ...base, width: 14, height: 14, background: "rgba(255,255,255,0.35)" };
  return (
    <div style={{ position: "absolute", top: 74, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 8 }}>
      {dots.map((state, i) => (
        <span key={i} data-state={state} style={styleFor(state)} />
      ))}
    </div>
  );
}
