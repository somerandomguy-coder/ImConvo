export function MuteToggle({ muted, onToggle }: { muted: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      aria-label={muted ? "Unmute" : "Mute"}
      style={{ position: "absolute", top: 18, right: 18, width: 40, height: 40, borderRadius: "50%", border: "3px solid #fff", background: "var(--candy-periwinkle)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}
    >
      <i className={muted ? "ti ti-volume-off" : "ti ti-volume"} aria-hidden="true" style={{ fontSize: 20 }} />
    </button>
  );
}
