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
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((s) => {
        stream = s;
        if (unmounted) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      })
      .catch((e) => {
        if (!unmounted) setError(e?.message || "Camera permission denied");
      });
    return () => {
      unmounted = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const grabFrame = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || !ready) return null;
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
      canvasRef.current.width = CAPTURE_W;
      canvasRef.current.height = CAPTURE_H;
    }
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H);
    return canvasRef.current.toDataURL("image/jpeg", 0.7).split(",")[1] ?? null;
  }, [ready]);

  return { videoRef, ready, error, grabFrame };
}
