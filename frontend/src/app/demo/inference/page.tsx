"use client";

import { useEffect, useState } from "react";
import DemoResultPanel from "@/components/demo/DemoResultPanel";
import WordResultPanel from "@/components/demo/WordResultPanel";
import DemoVideoUploader from "@/components/demo/DemoVideoUploader";
import {
  analyzeExample,
  analyzeVideo,
  analyzeWordVideo,
  analyzeWordExample,
  checkHealth,
  listCheckpoints,
  listDecoders,
  listExamples,
  type AnalyzeResponse,
  type CheckpointInfo,
  type WordAnalyzeResponse,
  type DecoderSpec,
  type HealthStatus,
} from "@/utils/api";

const DEFAULT_MODEL_PATH = "checkpoints/best_ctc_model_conformer_lite_gap_proj.keras";
const DEFAULT_DECODER_MODE = "greedy_ctc";
const DEFAULT_BEAM_WIDTH = 10;
const DEFAULT_DEBUG_TOP_K = 5;

const KNOWN_CHECKPOINTS: CheckpointInfo[] = [
  "best_ctc_model_bigru.keras",
  "best_ctc_model_bilstm.keras",
  "best_ctc_model_conformer_lite.keras",
  "best_ctc_model_conformer_lite_gap_proj.keras",
  "best_ctc_model_conformer_lite_resnet18.keras",
  "best_ctc_model_gru.keras",
  "best_ctc_model_tcn.keras",
  "best_ctc_model_transformer.keras",
  "best_ctc_model_transformer_medium.keras",
].map((name) => ({
  name,
  path: `checkpoints/${name}`,
  is_default: name === "best_ctc_model_conformer_lite_gap_proj.keras",
}));

type Mode = "character" | "word";

export default function DemoInferencePage() {
  const [mode, setMode] = useState<Mode>("character");
  const [file, setFile] = useState<File | null>(null);
  const [modelPath, setModelPath] = useState(DEFAULT_MODEL_PATH);
  const [expectedText, setExpectedText] = useState("");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [ctcResult, setCtcResult] = useState<AnalyzeResponse | null>(null);
  const [wordResult, setWordResult] = useState<WordAnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [selectedExample, setSelectedExample] = useState("");
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>(KNOWN_CHECKPOINTS);
  const [decoders, setDecoders] = useState<DecoderSpec[]>([]);
  const [decoderMode, setDecoderMode] = useState(DEFAULT_DECODER_MODE);
  const [beamWidth, setBeamWidth] = useState(DEFAULT_BEAM_WIDTH);
  const [llmPostprocess, setLlmPostprocess] = useState(false);

  useEffect(() => {
    let mounted = true;
    checkHealth()
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
    listCheckpoints()
      .then((data) => {
        if (!mounted) return;
        setCheckpoints(data.checkpoints);
        const def = data.checkpoints.find((c) => c.is_default);
        if (def) setModelPath(def.path);
      })
      .catch(() => {});
    listExamples(120)
      .then((data) => {
        if (!mounted) return;
        setExamples(data.examples);
        if (data.examples.length > 0) setSelectedExample(data.examples[0]);
      })
      .catch(() => {});
    return () => { mounted = false; };
  }, []);

  const extractError = (err: unknown) =>
    (typeof err === "object" && err && "response" in err &&
      typeof (err as { response?: { data?: { detail?: string } } }).response?.data?.detail === "string" &&
      (err as { response?: { data?: { detail?: string } } }).response?.data?.detail) ||
    (err instanceof Error ? err.message : "Inference request failed.");

  const runInference = async () => {
    if (!file) return;
    setError(null);
    setCtcResult(null);
    setWordResult(null);
    setIsLoading(true);
    try {
      if (mode === "word") {
        const res = await analyzeWordVideo(file, undefined);
        setWordResult(res);
      } else {
        const res = await analyzeVideo({ file, modelPath, expectedText, decoderMode, beamWidth, debugTopK: DEFAULT_DEBUG_TOP_K, llmPostprocess });
        setCtcResult(res);
      }
    } catch (err) {
      setError(extractError(err));
    } finally {
      setIsLoading(false);
    }
  };

  const runExampleInference = async () => {
    if (!selectedExample) return;
    setError(null);
    setCtcResult(null);
    setWordResult(null);
    setIsLoading(true);
    setFile(null);
    try {
      if (mode === "word") {
        const res = await analyzeWordExample(selectedExample);
        setWordResult(res);
      } else {
        const res = await analyzeExample({ exampleName: selectedExample, modelPath, expectedText, decoderMode, beamWidth, debugTopK: DEFAULT_DEBUG_TOP_K, llmPostprocess });
        setCtcResult(res);
      }
    } catch (err) {
      setError(extractError(err));
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setExpectedText("");
    setCtcResult(null);
    setWordResult(null);
    setError(null);
  };

  const inputClass =
    "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-accent/60 focus:ring-1 focus:ring-accent/30 disabled:opacity-40";

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-12">
      <div className="w-full max-w-3xl space-y-6">
        {/* Page header */}
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">Demo Inference</h1>
          <p className="text-sm text-muted">Run offline lip-reading with metrics, latency, and device info.</p>
        </div>

        {/* Mode toggle */}
        <div className="flex rounded-lg border border-border bg-card p-1 w-fit">
          {(["character", "word"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => { setMode(m); reset(); }}
              className={`rounded-md px-5 py-1.5 text-sm font-medium transition-colors ${
                mode === m
                  ? "bg-accent text-white shadow-sm"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {m === "character" ? "Character (CTC)" : "Word (GRID)"}
            </button>
          ))}
        </div>

        {/* Settings panel */}
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-5">
          <p className="text-xs font-medium uppercase tracking-wider text-muted">Settings</p>

          {mode === "character" ? (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="space-y-1.5">
                  <span className="text-xs text-muted">Model</span>
                  <select value={modelPath} onChange={(e) => setModelPath(e.target.value)} className={inputClass}>
                    {checkpoints.map((c) => (
                      <option key={c.path} value={c.path}>
                        {c.name}{c.is_default ? " (default)" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1.5">
                  <span className="text-xs text-muted">Expected text (optional)</span>
                  <input value={expectedText} onChange={(e) => setExpectedText(e.target.value)} className={inputClass} placeholder="For WER/CER scoring" />
                </label>
                <label className="space-y-1.5">
                  <span className="text-xs text-muted">Decoder</span>
                  <select value={decoderMode} onChange={(e) => setDecoderMode(e.target.value)} className={inputClass}>
                    {decoders.length === 0 && <option value={DEFAULT_DECODER_MODE}>Greedy CTC</option>}
                    {decoders.map((d) => <option key={d.mode} value={d.mode}>{d.label}</option>)}
                  </select>
                </label>
                <label className="space-y-1.5">
                  <span className="text-xs text-muted">Beam width</span>
                  <input type="number" min={2} step={1} value={beamWidth} onChange={(e) => setBeamWidth(Number(e.target.value) || DEFAULT_BEAM_WIDTH)} className={inputClass} disabled={decoderMode !== "beam_ctc"} />
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-6 border-t border-border pt-4">
                <label className="flex cursor-pointer items-center gap-2.5 select-none">
                  <input type="checkbox" checked={llmPostprocess} onChange={(e) => setLlmPostprocess(e.target.checked)} className="h-4 w-4 rounded border-border accent-accent" />
                  <span className="text-sm text-foreground">LLM post-processing (Gemini)</span>
                </label>
                {llmPostprocess && <p className="text-xs text-muted">API key from server <code>.env</code></p>}
              </div>
            </>
          ) : (
            <div className="text-sm text-muted space-y-1">
              <p>Uses the best word-level checkpoint from <code>checkpoints/word_level_grid/</code> automatically.</p>
              <p>Predicts 6 slots: <span className="text-foreground">command · color · preposition · letter · digit · adverb</span></p>
            </div>
          )}
        </div>

        {/* Upload */}
        <DemoVideoUploader file={file} onChange={setFile} />

        {/* Server example */}
        {examples.length > 0 && (
          <div className="flex items-center gap-3">
            <select value={selectedExample} onChange={(e) => setSelectedExample(e.target.value)} className={`${inputClass} flex-1`}>
              {examples.map((name) => <option key={name} value={name}>{name}</option>)}
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

        {mode === "character" ? (
          <DemoResultPanel health={health} result={ctcResult} isLoading={isLoading} error={error} />
        ) : (
          <WordResultPanel health={health} result={wordResult} isLoading={isLoading} error={error} />
        )}
      </div>
    </div>
  );
}
