"use client";

import { useState } from "react";
import VideoUploader from "@/components/VideoUploader";
import ResultDisplay from "@/components/ResultDisplay";

const MOCK_RESULTS = [
  "set blue by a five now",
  "place red at c two please",
  "bin green in f one again",
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resultText, setResultText] = useState("");
  const [uploadKey, setUploadKey] = useState(0);

  const handleAnalyze = () => {
    if (!file) return;
    setIsLoading(true);
    setResultText("");
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

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-20">
      <div className="w-full max-w-lg space-y-5">
        <div className="space-y-1.5">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Lip Reading
          </h1>
          <p className="text-sm leading-relaxed text-muted">
            Drop a video clip — we&apos;ll read the lips and return the spoken text.
          </p>
        </div>

        <VideoUploader
          key={uploadKey}
          onFileSelected={setFile}
          onRemoved={reset}
        />

        {file && !isLoading && !resultText && (
          <button
            onClick={handleAnalyze}
            className="w-full rounded-lg bg-accent py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
          >
            Analyze
          </button>
        )}

        <ResultDisplay text={resultText} isLoading={isLoading} />

        {resultText && (
          <button
            onClick={reset}
            className="w-full rounded-lg border border-border py-2.5 text-sm text-muted transition-colors hover:border-foreground/20 hover:text-foreground"
          >
            Try another video
          </button>
        )}
      </div>
    </div>
  );
}
