export function DiagonalRails() {
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        width: "38%",
        height: "70%",
        pointerEvents: "none",
        opacity: 0.95,
      }}
      role="img"
      aria-label="diagonal rail decoration"
    >
      <line x1="100" y1="100" x2="55" y2="0" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
      <line x1="100" y1="100" x2="55" y2="0" stroke="#E24B4A" strokeWidth="3" strokeDasharray="5 5" strokeLinecap="round" />
    </svg>
  );
}
