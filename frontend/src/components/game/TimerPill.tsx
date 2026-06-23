export function TimerPill({ secondsLeft }: { secondsLeft: number }) {
  const s = Math.max(0, Math.ceil(secondsLeft));
  const label = `0:${s.toString().padStart(2, "0")}`;
  const low = s <= 2;
  return (
    <div
      className={low ? "timer-low" : undefined}
      style={{
        position: "absolute", top: 18, left: "50%", transform: "translateX(-50%)",
        background: "var(--candy-periwinkle)", color: "#fff", padding: "6px 26px",
        borderRadius: 999, fontSize: 28, border: "3px solid #fff",
      }}
    >
      <span className="bubble-thin">{label}</span>
    </div>
  );
}
