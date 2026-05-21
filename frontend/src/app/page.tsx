"use client";

import { useState } from "react";
import VideoUploader from "@/components/VideoUploader";
import ResultDisplay from "@/components/ResultDisplay";
import {
  analyzeVideo,
  analyzeWordVideo,
  type WordAnalyzeResponse,
} from "@/utils/api";

const CTC_MODEL = "checkpoints/best_ctc_model_conformer_lite.keras";

type Mode = "character" | "word";

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "bg-green-500" : pct >= 50 ? "bg-yellow-500" : "bg-red-400";
  return (
    <div className="mt-2 flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-9 text-right text-xs text-muted">{pct}%</span>
    </div>
  );
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("character");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resultText, setResultText] = useState("");
  const [wordResult, setWordResult] = useState<WordAnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadKey, setUploadKey] = useState(0);

  const reset = () => {
    setFile(null);
    setResultText("");
    setWordResult(null);
    setError(null);
    setIsLoading(false);
    setUploadKey((k) => k + 1);
  };

  const handleModeChange = (m: Mode) => {
    setMode(m);
    reset();
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setResultText("");
    setWordResult(null);
    setError(null);
    try {
      if (mode === "character") {
        const res = await analyzeVideo({
          file,
          modelPath: CTC_MODEL,
          decoderMode: "greedy_ctc",
          llmPostprocess: true,
        });
        setResultText(res.llm?.corrected_text || res.predicted_text);
      } else {
        const res = await analyzeWordVideo(file);
        setWordResult(res);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const done = mode === "character" ? !!resultText : !!wordResult;

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-20">
      <div className="w-full max-w-lg space-y-5">

        {/* Header */}
        <div className="space-y-1.5">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Lip Reading</h1>
          <p className="text-sm leading-relaxed text-muted">
            Drop a video clip — we&apos;ll read the lips and return the spoken text.
          </p>
        </div>

        {/* Mode toggle */}
        <div className="flex rounded-lg border border-border bg-card p-1 w-fit">
          {(["character", "word"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => handleModeChange(m)}
              className={`rounded-md px-5 py-1.5 text-sm font-medium transition-colors ${
                mode === m ? "bg-accent text-white shadow-sm" : "text-muted hover:text-foreground"
              }`}
            >
              {m === "character" ? "Character level" : "Word level"}
            </button>
          ))}
        </div>

        {/* Upload */}
        <VideoUploader key={uploadKey} onFileSelected={setFile} onRemoved={reset} />

        {/* Analyze */}
        {file && !isLoading && !done && (
          <button
            onClick={handleAnalyze}
            className="w-full rounded-lg bg-accent py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
          >
            Analyze
          </button>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Character result — typing effect */}
        {mode === "character" && (
          <ResultDisplay text={resultText} isLoading={isLoading} />
        )}

        {/* Word result */}
        {mode === "word" && isLoading && (
          <div className="rounded-xl border border-border bg-card px-5 py-4 shadow-sm">
            <div className="flex items-center gap-3">
              <span className="flex gap-1">
                <span className="animate-bounce-dot h-1.5 w-1.5 rounded-full bg-accent" style={{ animationDelay: "0ms" }} />
                <span className="animate-bounce-dot h-1.5 w-1.5 rounded-full bg-accent" style={{ animationDelay: "160ms" }} />
                <span className="animate-bounce-dot h-1.5 w-1.5 rounded-full bg-accent" style={{ animationDelay: "320ms" }} />
              </span>
              <span className="text-sm text-muted">Analyzing lip movements…</span>
            </div>
          </div>
        )}

        {mode === "word" && wordResult && (
          <div key={wordResult.predicted_sentence} className="animate-fade-in space-y-4">
            {/* Predicted sentence */}
            <div className="rounded-xl border border-border bg-card px-5 py-4 shadow-sm">
              <p className="mb-2 text-xs text-muted">Predicted text</p>
              <p className="font-mono text-lg text-foreground">
                {wordResult.predicted_sentence || "(empty)"}
              </p>
            </div>

            {/* Slot predictions */}
            {wordResult.slot_predictions?.length > 0 && (
              <div>
                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">
                  Slot Predictions
                </p>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {wordResult.slot_predictions.map((slot) => (
                    <div
                      key={slot.slot}
                      className="rounded-lg border border-border bg-card p-3 shadow-sm"
                    >
                      <p className="text-xs text-muted capitalize">{slot.slot}</p>
                      <p className="mt-0.5 font-mono text-sm font-medium text-foreground">
                        {slot.word}
                      </p>
                      <ConfidenceBar value={slot.confidence} />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {done && (
          <button
            onClick={reset}
            className="w-full rounded-lg border border-border bg-card py-2.5 text-sm text-muted transition-colors hover:bg-border/40 hover:text-foreground"
          >
            Try another video
          </button>
        )}

      </div>
    </div>
  );
}
