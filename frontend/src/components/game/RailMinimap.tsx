export function RailMinimap({ total, roundIndex }: { total: number; roundIndex: number }) {
  const w = 200, h = 160;
  const remaining = Math.max(total - roundIndex, 0);
  const visible = Math.min(remaining, 3);
  const walls = Array.from({ length: visible }, (_, i) => {
    const depth = 1 - i / 3;
    const ww = 110 * depth + 16;
    const hh = 78 * depth + 12;
    const x = (w - ww) / 2;
    const y = h - 30 - hh - i * 6;
    const opacity = depth;
    const stroke = i === 0 ? "#C9A06A" : "#9c7842";
    return { ww, hh, x, y, opacity, stroke, key: roundIndex + i };
  });
  return (
    <div style={{ position: "absolute", top: 16, right: 16, width: w, height: h, pointerEvents: "none" }}>
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label="rail minimap">
        <line x1={w * 0.08} y1={h - 6} x2={w * 0.42} y2={h * 0.38} stroke="#fff" strokeWidth={9} strokeLinecap="round" />
        <line x1={w * 0.92} y1={h - 6} x2={w * 0.58} y2={h * 0.38} stroke="#fff" strokeWidth={9} strokeLinecap="round" />
        <line x1={w * 0.08} y1={h - 6} x2={w * 0.42} y2={h * 0.38} stroke="#E24B4A" strokeWidth={9} strokeDasharray="12 12" strokeLinecap="round" />
        <line x1={w * 0.92} y1={h - 6} x2={w * 0.58} y2={h * 0.38} stroke="#E24B4A" strokeWidth={9} strokeDasharray="12 12" strokeLinecap="round" />
        {walls
          .slice()
          .reverse()
          .map((wall) => (
            <g key={wall.key} data-wall>
              <rect
                x={wall.x}
                y={wall.y}
                width={wall.ww}
                height={wall.hh}
                rx={4}
                fill="none"
                stroke={wall.stroke}
                strokeWidth={5}
                opacity={wall.opacity}
                style={{ transition: "all 500ms ease" }}
              />
            </g>
          ))}
        <g data-car transform={`translate(${w / 2} ${h - 18})`}>
          <rect x={-16} y={-12} width={32} height={16} rx={4} fill="#FF3D9A" stroke="#fff" strokeWidth={2} />
          <rect x={-8} y={-18} width={16} height={8} rx={2} fill="#FF6FB5" stroke="#fff" strokeWidth={1.5} />
          <circle cx={-10} cy={6} r={3.5} fill="#1d2030" stroke="#fff" strokeWidth={1.5} />
          <circle cx={10} cy={6} r={3.5} fill="#1d2030" stroke="#fff" strokeWidth={1.5} />
        </g>
      </svg>
      <div
        style={{
          position: "absolute",
          top: -2,
          left: "50%",
          transform: "translateX(-50%)",
          fontSize: 11,
          color: "#fff",
          fontWeight: 700,
          background: "#7C8CF8",
          padding: "3px 10px",
          borderRadius: 999,
          border: "2px solid #fff",
          letterSpacing: 0.5,
        }}
      >
        {Math.min(roundIndex + 1, total)} / {total}
      </div>
    </div>
  );
}
