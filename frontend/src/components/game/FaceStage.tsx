import type { ReactNode, RefObject } from "react";
export function FaceStage({ videoRef, children }: { videoRef: RefObject<HTMLVideoElement | null>; children?: ReactNode }) {
  return (
    <div style={{ position: "relative", width: "100%", aspectRatio: "16/9", borderRadius: 24, overflow: "hidden", background: "#1d2030" }}>
      <video
        ref={videoRef}
        autoPlay playsInline muted
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", opacity: 1 }}
      />
      <div style={{ position: "absolute", inset: 0 }}>{children}</div>
    </div>
  );
}
