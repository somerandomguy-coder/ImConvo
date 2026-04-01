"use client";

import { useState } from "react";
import VideoUploader from "@/components/VideoUploader";
import LiveCapture from "@/components/LiveCapture";
import ResultDisplay from "@/components/ResultDisplay";

type Mode = "upload" | "live";

const MOCK_RESULTS = [
  "set blue by a five now",
  "place red at c two please",
  "bin green in f one again",
];

export default function Home() {
  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resultText, setResultText] = useState("");
  const [uploadKey, setUploadKey] = useState(0);

  const handleAnalyze = () => {
    if (!file) return;
    setIsLoading(true);
    setResultText("");

    // Mock: simulate 3s inference delay
    setTimeout(() => {
      const mock = MOCK_RESULTS[Math.floor(Math.random() * MOCK_RESULTS.length)];
      setResultText(mock);
      setIsLoading(false);
    }, 3000);
  };

  const reset = () => {
    setFile(null);
    setResultText("");
    setIsLoading(false);
    setUploadKey((k) => k + 1);
  };

  const switchMode = (m: Mode) => {
    reset();
    setMode(m);
  };

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-16">
      <div className="w-full max-w-2xl space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            ImConvo{" "}
            <span className="ml-2 text-lg font-normal text-muted">
              Lip Reading AI
            </span>
          </h1>
          <p className="mt-2 text-sm text-muted">
            Upload a video or use your camera — our model predicts what&apos;s
            being said, no audio needed.
          </p>
        </div>

        {/* Mode tabs */}
        <div className="flex gap-1 rounded-lg bg-card p-1">
          <button
            onClick={() => switchMode("upload")}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              mode === "upload"
                ? "bg-accent text-white"
                : "text-muted hover:text-foreground"
            }`}
          >
            Upload
          </button>
          <button
            onClick={() => switchMode("live")}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
              mode === "live"
                ? "bg-accent text-white"
                : "text-muted hover:text-foreground"
            }`}
          >
            Live
          </button>
        </div>

        {/* Upload mode */}
        {mode === "upload" && (
          <>
            <VideoUploader
              key={uploadKey}
              onFileSelected={setFile}
              onRemoved={reset}
            />

            {file && !isLoading && !resultText && (
              <button
                onClick={handleAnalyze}
                className="w-full rounded-lg bg-accent py-3 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
              >
                Analyze Video
              </button>
            )}

            <ResultDisplay text={resultText} isLoading={isLoading} />

            {resultText && (
              <button
                onClick={reset}
                className="w-full rounded-lg border border-border py-3 text-sm font-medium text-muted transition-colors hover:border-foreground hover:text-foreground"
              >
                Analyze Another Video
              </button>
            )}
          </>
        )}

        {/* Live mode */}
        {mode === "live" && <LiveCapture />}
      </div>
    </div>
  );
}
