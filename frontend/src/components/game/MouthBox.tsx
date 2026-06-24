import { useEffect, useState } from "react";
import type { RefObject } from "react";
import type { LipBBox } from "../../hooks/useGameSocket";

type Rect = { left: number; top: number; width: number; height: number };

function coverRect(bbox: LipBBox, videoW: number, videoH: number, boxW: number, boxH: number): Rect {
  const natAspect = videoW / videoH;
  const containerAspect = boxW / boxH;
  let displayW: number, displayH: number, offsetX: number, offsetY: number;
  if (natAspect > containerAspect) {
    displayH = boxH;
    displayW = boxH * natAspect;
    offsetX = (displayW - boxW) / 2;
    offsetY = 0;
  } else {
    displayW = boxW;
    displayH = boxW / natAspect;
    offsetX = 0;
    offsetY = (displayH - boxH) / 2;
  }
  return {
    left: bbox.x * displayW - offsetX,
    top: bbox.y * displayH - offsetY,
    width: bbox.w * displayW,
    height: bbox.h * displayH,
  };
}

export function MouthBox({
  bbox,
  label,
  videoRef,
}: {
  bbox: LipBBox | null;
  label?: string;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const [rect, setRect] = useState<Rect | null>(null);

  useEffect(() => {
    if (!bbox) {
      setRect(null);
      return;
    }
    const v = videoRef.current;
    const container = v?.parentElement;
    if (!v || !container || !v.videoWidth || !v.videoHeight) {
      setRect(null);
      return;
    }
    setRect(coverRect(bbox, v.videoWidth, v.videoHeight, container.clientWidth, container.clientHeight));
  }, [bbox, videoRef]);

  useEffect(() => {
    const onResize = () => {
      const v = videoRef.current;
      const container = v?.parentElement;
      if (!bbox || !v || !container || !v.videoWidth || !v.videoHeight) return;
      setRect(coverRect(bbox, v.videoWidth, v.videoHeight, container.clientWidth, container.clientHeight));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [bbox, videoRef]);

  if (!rect) return null;
  return (
    <div
      data-testid="mouth-box"
      style={{
        position: "absolute",
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        border: "3px solid #5DE0A6",
        borderRadius: 4,
        boxShadow: "0 0 0 1px rgba(0,0,0,0.4), 0 0 12px rgba(93,224,166,0.45)",
        pointerEvents: "none",
        transition: "left 120ms linear, top 120ms linear, width 120ms linear, height 120ms linear",
        zIndex: 2,
      }}
    >
      <span style={{ position: "absolute", left: -3, top: -3, width: 10, height: 10, borderTop: "3px solid #FFE25A", borderLeft: "3px solid #FFE25A" }} />
      <span style={{ position: "absolute", right: -3, top: -3, width: 10, height: 10, borderTop: "3px solid #FFE25A", borderRight: "3px solid #FFE25A" }} />
      <span style={{ position: "absolute", left: -3, bottom: -3, width: 10, height: 10, borderBottom: "3px solid #FFE25A", borderLeft: "3px solid #FFE25A" }} />
      <span style={{ position: "absolute", right: -3, bottom: -3, width: 10, height: 10, borderBottom: "3px solid #FFE25A", borderRight: "3px solid #FFE25A" }} />
      {label && (
        <div
          style={{
            position: "absolute",
            top: -24,
            left: -3,
            background: "#5DE0A6",
            color: "#0f2a1d",
            padding: "2px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 700,
            whiteSpace: "nowrap",
            fontFamily: "monospace",
          }}
        >
          {label}
        </div>
      )}
    </div>
  );
}
