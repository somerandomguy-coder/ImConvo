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

const DEFAULT_MODEL_PATH = "checkpoints/best_ctc_model_conformer_lite_gap_proj.keras";
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
  const [llmPostprocess, setLlmPostprocess] = useState(false);

  useEffect(() => {
    let mounted = true;
    checkDemoHealth()
      .then((data) => { if (mounted) setHealth(data); })
      .catch((err: unknown) => {
        if (!mounted) return;
        const message = err instanceof Error ? err.message : "Failed to connect to inference API.";
        setError(message);
      });
    listDecoders()
      .then((data) => {
        if (!mounted) return;
        setDecoders(data.decoders);
        setDecoderMode(data.default_mode || DEFAULT_DECODER_MODE);
      })
      .catch(() => {});
    listDemoExamples(120)
      .then((data) => {
        if (!mounted) return;
        setExamples(data.examples);
        if (data.examples.length > 0) setSelectedExample(data.examples[0]);
      })
      .catch(() => {});
    return () => { mounted = false; };
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
        llmPostprocess,
      });
      setResult(analyzed);
    } catch (err: unknown) {
      const message =
        (typeof err === "object" &&
          err &&
          "response" in err &&
          typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === "string" &&
          (err as { response?: { data?: { detail?: string } } }).response?.data?.detail) ||
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
        llmPostprocess,
      });
      setResult(analyzed);
    } catch (err: unknown) {
      const message =
        (typeof err === "object" &&
          err &&
          "response" in err &&
          typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === "string" &&
          (err as { response?: { data?: { detail?: string } } }).response?.data?.detail) ||
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

  const inputClass =
    "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-accent/60 focus:ring-1 focus:ring-accent/30 disabled:opacity-40";

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-12">
      <div className="w-full max-w-3xl space-y-6">
        {/* Page header */}
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Demo Inference
          </h1>
          <p className="text-sm text-muted">
            Run offline lip-reading with metrics, latency, and device info.
          </p>
        </div>

        {/* Settings panel */}
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-5">
          <p className="text-xs font-medium uppercase tracking-wider text-muted">Settings</p>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1.5">
              <span className="text-xs text-muted">Model path</span>
              <input
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
                className={inputClass}
                placeholder={DEFAULT_MODEL_PATH}
              />
            </label>

            <label className="space-y-1.5">
              <span className="text-xs text-muted">Expected text (optional)</span>
              <input
                value={expectedText}
                onChange={(e) => setExpectedText(e.target.value)}
                className={inputClass}
                placeholder="For WER/CER scoring"
              />
            </label>

            <label className="space-y-1.5">
              <span className="text-xs text-muted">Decoder</span>
              <select
                value={decoderMode}
                onChange={(e) => setDecoderMode(e.target.value)}
                className={inputClass}
              >
                {decoders.length === 0 && (
                  <option value={DEFAULT_DECODER_MODE}>Greedy CTC</option>
                )}
                {decoders.map((d) => (
                  <option key={d.mode} value={d.mode}>{d.label}</option>
                ))}
              </select>
            </label>

            <label className="space-y-1.5">
              <span className="text-xs text-muted">Beam width</span>
              <input
                type="number"
                min={2}
                step={1}
                value={beamWidth}
                onChange={(e) => setBeamWidth(Number(e.target.value) || DEFAULT_BEAM_WIDTH)}
                className={inputClass}
                disabled={decoderMode !== "beam_ctc"}
              />
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-6 border-t border-border pt-4">
            <label className="flex cursor-pointer items-center gap-2.5 select-none">
              <input
                type="checkbox"
                checked={llmPostprocess}
                onChange={(e) => setLlmPostprocess(e.target.checked)}
                className="h-4 w-4 rounded border-border accent-accent"
              />
              <span className="text-sm text-foreground">LLM post-processing (Gemini)</span>
            </label>
            {llmPostprocess && (
              <p className="text-xs text-muted">API key from server <code>.env</code></p>
            )}
          </div>
        </div>

        {/* Upload */}
        <DemoVideoUploader file={file} onChange={setFile} />

        {/* Server example */}
        {examples.length > 0 && (
          <div className="flex items-center gap-3">
            <select
              value={selectedExample}
              onChange={(e) => setSelectedExample(e.target.value)}
              className={`${inputClass} flex-1`}
            >
              {examples.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <button
              type="button"
              disabled={!selectedExample || isLoading}
              onClick={runExampleInference}
              className="rounded-lg border border-border px-4 py-2 text-sm text-muted transition-colors hover:border-foreground/30 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isLoading ? "Running…" : "Run server example"}
            </button>
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            disabled={!file || isLoading}
            onClick={runInference}
            className="rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isLoading ? "Analyzing…" : "Analyze Video"}
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-lg border border-border px-5 py-2.5 text-sm text-muted transition-colors hover:border-foreground/20 hover:text-foreground"
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
