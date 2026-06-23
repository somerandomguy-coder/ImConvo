export function RailMinimap({ total, roundIndex }: { total: number; roundIndex: number }) {
  const w = 180, h = 140;
  const carX = 90;
  const carYStart = 110;
  const carYEnd = 30;
  const carY = carYStart + ((carYEnd - carYStart) * Math.min(roundIndex, total - 1)) / Math.max(total - 1, 1);
  return (
    <div style={{ position: "absolute", top: 16, right: 16, width: w, height: h, borderRadius: 14, border: "3px solid #fff", overflow: "hidden", background: "rgba(38,41,54,0.85)" }}>
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label="rail minimap">
        <polygon points={`0,${h} ${w},${h} ${w*0.66},${h*0.42} ${w*0.34},${h*0.42}`} fill="#383b4a" />
        <line x1={6} y1={h - 5} x2={w*0.34} y2={h*0.42 + 3} stroke="#fff" strokeWidth={8} />
        <line x1={w - 6} y1={h - 5} x2={w*0.66} y2={h*0.42 + 3} stroke="#fff" strokeWidth={8} />
        <line x1={6} y1={h - 5} x2={w*0.34} y2={h*0.42 + 3} stroke="#E24B4A" strokeWidth={8} strokeDasharray="10 10" />
        <line x1={w - 6} y1={h - 5} x2={w*0.66} y2={h*0.42 + 3} stroke="#E24B4A" strokeWidth={8} strokeDasharray="10 10" />
        {Array.from({ length: total }, (_, i) => {
          const state = i < roundIndex ? "passed" : i === roundIndex ? "current" : "upcoming";
          const depth = 1 - i / (total - 1 || 1);
          const ww = 80 * depth + 20;
          const hh = 60 * depth + 16;
          const x = (w - ww) / 2;
          const y = 70 - i * 8;
          const stroke = state === "passed" ? "#5DE0A6" : state === "current" ? "#C9A06A" : "#9c7842";
          const opacity = state === "upcoming" ? 0.45 : 0.95;
          return (
            <g key={i} data-wall data-state={state}>
              <rect x={x} y={y} width={ww} height={hh} rx={3} fill="none" stroke={stroke} strokeWidth={4} opacity={opacity}
                    style={{ transition: "all 450ms ease" }} />
            </g>
          );
        })}
        <g data-car transform={`translate(${carX} ${carY})`} style={{ transition: "transform 450ms ease" }}>
          <rect x={-14} y={-10} width={28} height={14} rx={3} fill="#FF3D9A" stroke="#fff" strokeWidth={2} />
          <circle cx={-8} cy={6} r={3} fill="#1d2030" stroke="#fff" strokeWidth={1.5} />
          <circle cx={8} cy={6} r={3} fill="#1d2030" stroke="#fff" strokeWidth={1.5} />
        </g>
      </svg>
      <div style={{ position: "absolute", top: 6, left: "50%", transform: "translateX(-50%)", fontSize: 10, color: "#fff", fontWeight: 600, background: "rgba(0,0,0,0.5)", padding: "2px 8px", borderRadius: 8 }}>
        wall {Math.min(roundIndex + 1, total)} / {total}
      </div>
    </div>
  );
}
