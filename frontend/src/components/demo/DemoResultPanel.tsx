/* eslint-disable @next/next/no-img-element */
"use client";

import type { AnalyzeResponse, HealthStatus } from "@/utils/api";

interface DemoResultPanelProps {
  isLoading: boolean;
  error: string | null;
  health: HealthStatus | null;
  result: AnalyzeResponse | null;
}

function fmt(value: number | null, digits = 2) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function fmtPct(value: number | null) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

function fmtScore(value: number | null) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return value.toFixed(4);
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">{children}</p>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-lg border border-border bg-card p-3 shadow-sm ${className}`}>{children}</div>
  );
}

export default function DemoResultPanel({
  isLoading,
  error,
  health,
  result,
}: DemoResultPanelProps) {

  if (isLoading) {
    return (
      <Card>
        <p className="text-sm text-muted">Running inference…</p>
      </Card>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4">
        <p className="text-sm font-medium text-red-600">Inference failed</p>
        <p className="mt-1 text-sm text-red-500">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <Card>
        <p className="text-xs text-muted">
          API: {health?.status ?? "unknown"} &middot; model loaded: {String(health?.model_loaded ?? false)} &middot; device: {health?.device_used ?? "N/A"}
        </p>
      </Card>
    );
  }

  return (
    <div className="w-full space-y-4">
      {/* Server example preview — only shown when the API returns a preview_url */}
      {result.preview_url && (
        <section>
          <SectionLabel>Uploaded Video</SectionLabel>
          <video
            controls
            onLoadedMetadata={(e) => { e.currentTarget.muted = false; }}
            className="aspect-video w-full rounded-lg bg-black object-contain"
            src={`${process.env.NEXT_PUBLIC_DEMO_API_URL || "http://localhost:8001"}${result.preview_url}`}
          />
        </section>
      )}

      {/* Lip crop samples */}
      {result.crop_samples?.length > 0 && (
        <section>
          <SectionLabel>Cropped Lip Frames (MediaPipe · 6 samples)</SectionLabel>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {result.crop_samples.map((src, i) => (
              <div key={i} className="space-y-1">
                <img src={src} alt={`lip frame ${i}`} className="w-full rounded border border-border bg-black" />
                <p className="text-center text-xs text-muted">f{Math.round((i / 5) * 74)}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Prediction */}
      <section>
        <SectionLabel>Predicted Text</SectionLabel>
        <p className="rounded-lg border border-border bg-card px-4 py-3 font-mono text-lg text-foreground">
          {result.predicted_text || "(empty prediction)"}
        </p>
      </section>

      {/* LLM correction */}
      {result.llm?.corrected_text != null && (
        <section className="rounded-lg border border-green-200 bg-green-50 p-4">
          <p className="mb-1.5 text-xs font-medium uppercase tracking-wider text-green-700">
            LLM Corrected
            {result.llm.model && <span className="ml-1.5 normal-case text-green-500">({result.llm.model})</span>}
          </p>
          <p className="font-mono text-lg text-green-800">{result.llm.corrected_text || "(empty)"}</p>
          {(result.llm.wer != null || result.llm.cer != null) && (
            <div className="mt-2 flex gap-4 text-xs text-green-600">
              {result.llm.wer != null && <span>WER: {fmtPct(result.llm.wer)}</span>}
              {result.llm.cer != null && <span>CER: {fmtPct(result.llm.cer)}</span>}
            </div>
          )}
        </section>
      )}

      {/* Metrics */}
      <section className="grid gap-3 sm:grid-cols-3">
        <Card>
          <p className="text-xs text-muted">WER</p>
          <p className="mt-1 text-lg font-semibold text-foreground">{fmtPct(result.wer)}</p>
        </Card>
        <Card>
          <p className="text-xs text-muted">CER</p>
          <p className="mt-1 text-lg font-semibold text-foreground">{fmtPct(result.cer)}</p>
        </Card>
        <Card>
          <p className="text-xs text-muted">Reference source</p>
          <p className="mt-1 text-sm text-foreground">{result.reference_source}</p>
        </Card>
      </section>

      {/* Reference text */}
      <Card>
        <p className="text-xs text-muted">Reference text</p>
        <p className="mt-1 text-sm text-foreground">{result.reference_text || "N/A"}</p>
      </Card>

      {/* Latency */}
      <section className="grid gap-3 sm:grid-cols-3">
        <Card>
          <p className="text-xs text-muted">Preprocess (ms)</p>
          <p className="mt-1 text-sm text-foreground">{fmt(result.latency_ms.preprocess)}</p>
        </Card>
        <Card>
          <p className="text-xs text-muted">Inference (ms)</p>
          <p className="mt-1 text-sm text-foreground">{fmt(result.latency_ms.inference)}</p>
        </Card>
        <Card>
          <p className="text-xs text-muted">Total (ms)</p>
          <p className="mt-1 text-sm text-foreground">{fmt(result.latency_ms.total)}</p>
        </Card>
      </section>

      {/* Decoder */}
      <Card>
        <SectionLabel>Decoder</SectionLabel>
        <div className="space-y-0.5 text-sm text-foreground">
          <p>{result.decoder.label} &middot; beam width: {result.decoder.beam_width}</p>
          <p className="text-muted">Collapsed: {result.debug.collapsed_text || "N/A"}</p>
          {result.decoder.metadata.lm_type && (
            <p className="text-muted">LM: {result.decoder.metadata.lm_type} &middot; α={fmtScore(result.decoder.metadata.ngram_alpha ?? null)}</p>
          )}
        </div>
      </Card>

      {/* Decoder hypotheses */}
      <section>
        <SectionLabel>Top Hypotheses</SectionLabel>
        <div className="space-y-2">
          {result.decoder.hypotheses.slice(0, result.debug.decoder_top_k).map((h) => (
            <div key={`${h.rank}-${h.text}`} className="rounded-lg border border-border bg-card px-4 py-3 shadow-sm">
              <p className="text-xs text-muted">#{h.rank}</p>
              <p className="mt-0.5 font-mono text-sm text-foreground">{h.text || "(empty)"}</p>
              <p className="mt-1.5 text-xs text-muted">
                acoustic {fmtScore(h.acoustic_score)} &middot; lm {fmtScore(h.lm_score)} &middot; combined {fmtScore(h.combined_score)}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Raw CTC output */}
      <section>
        <SectionLabel>Raw CTC Output</SectionLabel>
        <pre className="max-h-40 overflow-auto rounded-lg border border-border bg-card p-3 text-xs leading-relaxed text-foreground">
          {result.debug.raw_timestep_text}
        </pre>
      </section>

      {/* Video stats */}
      <Card>
        <SectionLabel>Video Stats</SectionLabel>
        <div className="space-y-0.5 text-sm text-foreground">
          <p>{result.video_stats.filename}</p>
          <p className="text-muted">
            {result.video_stats.width ?? "N/A"}×{result.video_stats.height ?? "N/A"} &middot; {fmt(result.video_stats.fps)} fps &middot; {result.video_stats.frame_count ?? "N/A"} frames &middot; {fmt(result.video_stats.duration_sec)}s
          </p>
          <p className="text-muted">Shape: {result.video_stats.processed_shape.join(" × ")}</p>
        </div>
      </Card>

      {/* Device specs */}
      <Card>
        <SectionLabel>Device</SectionLabel>
        <div className="space-y-0.5 text-sm text-foreground">
          <p>{result.device_specs.device_used} &middot; TF {result.device_specs.tf_version}</p>
          <p className="text-muted">{result.device_specs.cpu_model ?? "N/A"} &middot; {result.device_specs.cpu_physical_cores ?? "N/A"}c/{result.device_specs.cpu_logical_cores ?? "N/A"}t</p>
          <p className="text-muted">RAM {fmt(result.device_specs.ram_total_gb)} GB &middot; GPU: {result.device_specs.gpu_names.length ? result.device_specs.gpu_names.join(", ") : "none"}</p>
        </div>
      </Card>

      <p className="text-xs text-muted/50">{result.model_path_used}</p>
    </div>
  );
}
