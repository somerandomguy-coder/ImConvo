"use client";

import { useCamera } from "@/hooks/useCamera";
import { useCallback, useEffect, useRef, useState } from "react";

const MOCK_PHRASES = [
  "set blue by a five now",
  "place red at c two please",
  "bin green in f one again",
  "lay white with g nine soon",
];

export default function LiveCapture() {
  const { videoRef, isStreaming, error, startCamera, stopCamera } = useCamera();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [liveText, setLiveText] = useState("");
  const [frameCount, setFrameCount] = useState(0);

  const handleStart = useCallback(async () => {
    await startCamera();
  }, [startCamera]);

  const handleStop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    stopCamera();
    setLiveText("");
    setFrameCount(0);
  }, [stopCamera]);

  // Mock inference cycle when streaming begins
  useEffect(() => {
    if (isStreaming) {
      intervalRef.current = setInterval(() => {
        setFrameCount((c) => {
          const next = c + 1;
          setLiveText(MOCK_PHRASES[next % MOCK_PHRASES.length]);
          return next;
        });
      }, 4000);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isStreaming]);

  return (
    <div className="w-full space-y-6">
      {/* Camera feed */}
      <div className="overflow-hidden rounded-xl border border-border bg-card">
        {isStreaming ? (
          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="aspect-video w-full bg-black object-cover"
              style={{ transform: "scaleX(-1)" }}
            />
            {/* Live indicator */}
            <div className="absolute top-4 left-4 flex items-center gap-2 rounded-full bg-red-600/90 px-3 py-1">
              <span className="h-2 w-2 animate-pulse rounded-full bg-white" />
              <span className="text-xs font-medium text-white">LIVE</span>
            </div>
            {/* Frame counter */}
            <div className="absolute top-4 right-4 rounded-full bg-black/60 px-3 py-1">
              <span className="text-xs font-mono text-white">
                Cycle {frameCount}
              </span>
            </div>
          </div>
        ) : (
          <div className="flex aspect-video flex-col items-center justify-center bg-black/30">
            <svg
              className="mb-3 h-12 w-12 text-muted"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
              />
            </svg>
            <p className="text-sm text-muted">Camera is off</p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Live text result */}
      {liveText && (
        <div className="rounded-xl border border-border bg-card p-6">
          <p className="mb-3 text-xs font-medium uppercase tracking-widest text-muted">
            Live Prediction
          </p>
          <p className="font-mono text-xl text-foreground">{liveText}</p>
        </div>
      )}

      {/* Start / Stop button */}
      <button
        onClick={isStreaming ? handleStop : handleStart}
        className={`w-full rounded-lg py-3 text-sm font-medium transition-colors ${
          isStreaming
            ? "border border-red-500/50 text-red-400 hover:bg-red-500/10"
            : "bg-accent text-white hover:bg-accent-hover"
        }`}
      >
        {isStreaming ? "Stop Camera" : "Start Camera"}
      </button>
    </div>
  );
}
