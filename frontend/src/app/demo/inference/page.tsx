"use client";

import { useEffect, useState } from "react";
import DemoResultPanel from "@/components/demo/DemoResultPanel";
import DemoVideoUploader from "@/components/demo/DemoVideoUploader";
import {
  analyzeDemoExample,
  analyzeDemoVideo,
  checkDemoHealth,
  listDecoders,
  listDemoExamples,
  type AnalyzeResponse,
  type DecoderSpec,
  type HealthStatus,
} from "@/utils/demoApi";

const DEFAULT_MODEL_PATH = "checkpoints/best_ctc_model.keras";
const DEFAULT_DECODER_MODE = "greedy_ctc";
const DEFAULT_BEAM_WIDTH = 10;
const DEFAULT_DEBUG_TOP_K = 5;

export default function DemoInferencePage() {
  const [file, setFile] = useState<File | null>(null);
  const [modelPath, setModelPath] = useState(DEFAULT_MODEL_PATH);
  const [expectedText, setExpectedText] = useState("");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [selectedExample, setSelectedExample] = useState("");
  const [decoders, setDecoders] = useState<DecoderSpec[]>([]);
  const [decoderMode, setDecoderMode] = useState(DEFAULT_DECODER_MODE);
  const [beamWidth, setBeamWidth] = useState(DEFAULT_BEAM_WIDTH);

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
    listDecoders()
      .then((data) => {
        if (!mounted) return;
        setDecoders(data.decoders);
        setDecoderMode(data.default_mode || DEFAULT_DECODER_MODE);
      })
      .catch(() => {
        // Keep the page usable even if decoder discovery fails.
      });
    listDemoExamples(120)
      .then((data) => {
        if (!mounted) return;
        setExamples(data.examples);
        if (data.examples.length > 0) {
          setSelectedExample(data.examples[0]);
        }
      })
      .catch(() => {
        // Keep upload flow usable even if examples endpoint fails.
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
        decoderMode,
        beamWidth,
        debugTopK: DEFAULT_DEBUG_TOP_K,
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

  const runExampleInference = async () => {
    if (!selectedExample) return;
    setError(null);
    setResult(null);
    setIsLoading(true);
    setFile(null);

    try {
      const analyzed = await analyzeDemoExample({
        exampleName: selectedExample,
        modelPath,
        expectedText,
        decoderMode,
        beamWidth,
        debugTopK: DEFAULT_DEBUG_TOP_K,
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
        (err instanceof Error ? err.message : "Example inference request failed.");
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
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
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

            <label className="space-y-1">
              <span className="text-sm font-medium text-foreground">
                Decoder mode
              </span>
              <select
                value={decoderMode}
                onChange={(e) => setDecoderMode(e.target.value)}
                className="w-full rounded-md border border-border bg-black/20 px-3 py-2 text-sm text-foreground outline-none ring-accent/40 focus:ring-2"
              >
                {decoders.length === 0 && (
                  <option value={DEFAULT_DECODER_MODE}>Greedy CTC</option>
                )}
                {decoders.map((decoder) => (
                  <option key={decoder.mode} value={decoder.mode}>
                    {decoder.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-sm font-medium text-foreground">
                Server example (s3_processed)
              </span>
              <select
                value={selectedExample}
                onChange={(e) => setSelectedExample(e.target.value)}
                className="w-full rounded-md border border-border bg-black/20 px-3 py-2 text-sm text-foreground outline-none ring-accent/40 focus:ring-2"
              >
                {examples.length === 0 && (
                  <option value="">No examples available</option>
                )}
                {examples.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-sm font-medium text-foreground">
                Beam width
              </span>
              <input
                type="number"
                min={2}
                step={1}
                value={beamWidth}
                onChange={(e) => setBeamWidth(Number(e.target.value) || DEFAULT_BEAM_WIDTH)}
                className="w-full rounded-md border border-border bg-black/20 px-3 py-2 text-sm text-foreground outline-none ring-accent/40 focus:ring-2"
                disabled={decoderMode !== "beam_ctc"}
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
            disabled={!selectedExample || isLoading}
            onClick={runExampleInference}
            className="rounded-lg bg-card px-5 py-2.5 text-sm font-medium text-foreground ring-1 ring-border transition-colors hover:bg-card-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? "Analyzing..." : "Run Server Example"}
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
