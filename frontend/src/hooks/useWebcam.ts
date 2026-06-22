import { useCallback, useEffect, useRef, useState } from "react";

const CAPTURE_W = 240;
const CAPTURE_H = 180;

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let unmounted = false;
    let stream: MediaStream | null = null;
    let pollId: ReturnType<typeof setInterval> | null = null;
    const markReady = () => {
      if (!unmounted) setReady(true);
    };
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((s) => {
        stream = s;
        if (unmounted) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          v.onloadedmetadata = markReady;
          v.oncanplay = markReady;
          v.play().catch(() => {});
          // Events above can be missed (StrictMode double-mount / handler
          // attached after the event fired). Poll the element's real state as
          // a bulletproof fallback: ready once it actually has pixel data.
          pollId = setInterval(() => {
            const el = videoRef.current;
            if (el && el.readyState >= 2 && el.videoWidth > 0) {
              markReady();
              if (pollId) clearInterval(pollId);
            }
          }, 200);
        }
      })
      .catch((e) => {
        if (!unmounted) setError(e?.message || "Camera permission denied");
      });
    return () => {
      unmounted = true;
      if (pollId) clearInterval(pollId);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const grabFrame = useCallback((): string | null => {
    const video = videoRef.current;
    // Gate on the element's real state rather than the React `ready` flag so
    // capture works as soon as the feed has pixels, regardless of event timing.
    if (!video || video.readyState < 2 || video.videoWidth === 0) return null;
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
      canvasRef.current.width = CAPTURE_W;
      canvasRef.current.height = CAPTURE_H;
    }
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    return canvasRef.current.toDataURL("image/jpeg", 0.7).split(",")[1] ?? null;
  }, []);

  return { videoRef, ready, error, grabFrame };
}
