"use client";

import { useEffect, useState } from "react";
import DemoResultPanel from "@/components/demo/DemoResultPanel";
import DemoVideoUploader from "@/components/demo/DemoVideoUploader";
import { analyzeDemoVideo, checkDemoHealth, type AnalyzeResponse, type HealthStatus } from "@/utils/demoApi";

const DEFAULT_MODEL_PATH = "checkpoints/best_ctc_model.keras";

export default function DemoInferencePage() {
  const [file, setFile] = useState<File | null>(null);
  const [modelPath, setModelPath] = useState(DEFAULT_MODEL_PATH);
  const [expectedText, setExpectedText] = useState("");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    checkDemoHealth()
      .then((data) => {
        if (!mounted) return;
        setHealth(data);
      })
      .catch((err: unknown) => {
        if (!mounted) return;
        const message =
          err instanceof Error ? err.message : "Failed to connect to inference API.";
        setError(message);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const runInference = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setIsLoading(true);

    try {
      const analyzed = await analyzeDemoVideo({
        file,
        modelPath,
        expectedText,
      });
      setResult(analyzed);
    } catch (err: unknown) {
      const message =
        (typeof err === "object" &&
          err &&
          "response" in err &&
          typeof (err as { response?: { data?: { detail?: string } } }).response
            ?.data?.detail === "string" &&
          (err as { response?: { data?: { detail?: string } } }).response?.data
            ?.detail) ||
        (err instanceof Error ? err.message : "Inference request failed.");
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setExpectedText("");
    setResult(null);
    setError(null);
  };

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-10">
      <div className="w-full max-w-4xl space-y-6">
        <section className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">
            Demo Inference (Isolated PoC)
          </h1>
          <p className="text-sm text-muted">
            Upload one video and run offline lip-reading inference with metrics,
            latency, and device specs.
          </p>
          <p className="text-xs text-muted">
            Tip: if your source is MPG, convert it to MP4 for browser preview.
            Example: <code>npm run demo:convert:s1</code>
          </p>
        </section>

        <section className="rounded-xl border border-border bg-card p-5">
          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-1">
              <span className="text-sm font-medium text-foreground">
                Model path
              </span>
              <input
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
                className="w-full rounded-md border border-border bg-black/20 px-3 py-2 text-sm text-foreground outline-none ring-accent/40 focus:ring-2"
                placeholder={DEFAULT_MODEL_PATH}
              />
            </label>

            <label className="space-y-1">
              <span className="text-sm font-medium text-foreground">
                Expected text (optional)
              </span>
              <input
                value={expectedText}
                onChange={(e) => setExpectedText(e.target.value)}
                className="w-full rounded-md border border-border bg-black/20 px-3 py-2 text-sm text-foreground outline-none ring-accent/40 focus:ring-2"
                placeholder="Type expected sentence for WER/CER"
              />
            </label>
          </div>
        </section>

        <DemoVideoUploader file={file} onChange={setFile} />

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            disabled={!file || isLoading}
            onClick={runInference}
            className="rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? "Analyzing..." : "Analyze Video"}
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-lg border border-border px-5 py-2.5 text-sm font-medium text-muted transition-colors hover:border-foreground hover:text-foreground"
          >
            Reset
          </button>
        </div>

        <DemoResultPanel
          health={health}
          result={result}
          isLoading={isLoading}
          error={error}
          file={file}
        />
      </div>
    </div>
  );
}
