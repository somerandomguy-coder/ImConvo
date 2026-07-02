/* eslint-disable @next/next/no-img-element */
"use client";

import type { WordAnalyzeResponse, HealthStatus } from "@/utils/api";

interface WordResultPanelProps {
  isLoading: boolean;
  error: string | null;
  health: HealthStatus | null;
  result: WordAnalyzeResponse | null;
}

function fmt(value: number | null, digits = 2) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function fmtPct(value: number | null) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
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

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "bg-green-500" : pct >= 50 ? "bg-yellow-500" : "bg-red-400";
  return (
    <div className="mt-1.5 flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-9 text-right text-xs text-muted">{pct}%</span>
    </div>
  );
}

export default function WordResultPanel({ isLoading, error, health, result }: WordResultPanelProps) {
  if (isLoading) {
    return <Card><p className="text-sm text-muted">Running word-level inference…</p></Card>;
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
          API: {health?.status ?? "unknown"} · model loaded: {String(health?.model_loaded ?? false)} · device: {health?.device_used ?? "N/A"}
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
            src={`${process.env.NEXT_PUBLIC_DEMO_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"}${result.preview_url}`}
          />
        </section>
      )}

      {/* Lip crop samples */}
      {result.crop_samples?.length > 0 && (
        <section>
          <SectionLabel>Cropped Lip Frames (6 samples)</SectionLabel>
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

      {/* Predicted sentence */}
      <section>
        <SectionLabel>Predicted Sentence</SectionLabel>
        <p className="rounded-lg border border-border bg-card px-4 py-3 font-mono text-lg text-foreground">
          {result.predicted_sentence || "(empty prediction)"}
        </p>
      </section>

      {/* Slot breakdown */}
      <section>
        <SectionLabel>Slot Predictions</SectionLabel>
        <div className="grid gap-2 sm:grid-cols-3">
          {result.slot_predictions.map((slot) => (
            <Card key={slot.slot}>
              <p className="text-xs text-muted capitalize">{slot.slot}</p>
              <p className="mt-0.5 font-mono text-sm font-medium text-foreground">{slot.word}</p>
              <ConfidenceBar value={slot.confidence} />
            </Card>
          ))}
        </div>
      </section>

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

      {/* Video stats */}
      <Card>
        <SectionLabel>Video Stats</SectionLabel>
        <div className="space-y-0.5 text-sm text-foreground">
          <p>{result.video_stats.filename}</p>
          <p className="text-muted">
            {result.video_stats.width ?? "N/A"}×{result.video_stats.height ?? "N/A"} · {fmt(result.video_stats.fps)} fps · {result.video_stats.frame_count ?? "N/A"} frames · {fmt(result.video_stats.duration_sec)}s
          </p>
        </div>
      </Card>

      {/* Device */}
      <Card>
        <SectionLabel>Device</SectionLabel>
        <div className="space-y-0.5 text-sm text-foreground">
          <p>{result.device_specs.device_used} · TF {result.device_specs.tf_version}</p>
          <p className="text-muted">{result.device_specs.cpu_model ?? "N/A"}</p>
          <p className="text-muted">RAM {fmt(result.device_specs.ram_total_gb)} GB · GPU: {result.device_specs.gpu_names.length ? result.device_specs.gpu_names.join(", ") : "none"}</p>
        </div>
      </Card>

      <p className="text-xs text-muted/50">{result.model_path_used}</p>
    </div>
  );
}
