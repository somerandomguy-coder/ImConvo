"use client";

import { useEffect, useMemo } from "react";
import type { AnalyzeResponse, HealthStatus } from "@/utils/demoApi";

interface DemoResultPanelProps {
  isLoading: boolean;
  error: string | null;
  health: HealthStatus | null;
  result: AnalyzeResponse | null;
  file: File | null;
}

function displayNumber(value: number | null, digits: number = 2) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function displayPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

export default function DemoResultPanel({
  isLoading,
  error,
  health,
  result,
  file,
}: DemoResultPanelProps) {
  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const canPreview = !!file && file.type.startsWith("video/");

  if (isLoading) {
    return (
      <div className="w-full rounded-xl border border-border bg-card p-6">
        <p className="text-sm text-muted">Running offline inference...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full rounded-xl border border-red-600/60 bg-red-950/20 p-6">
        <p className="text-sm font-semibold text-red-300">Inference failed</p>
        <p className="mt-1 text-sm text-red-200">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="w-full rounded-xl border border-border bg-card p-6">
        <p className="text-sm text-muted">
          API status: {health?.status ?? "unknown"} | model loaded:{" "}
          {String(health?.model_loaded ?? false)} | device:{" "}
          {health?.device_used ?? "N/A"}
        </p>
      </div>
    );
  }

  return (
    <div className="w-full space-y-4 rounded-xl border border-border bg-card p-6">
      <section>
        <p className="mb-2 text-xs uppercase tracking-widest text-muted">
          Uploaded Video
        </p>
        {result.preview_url ? (
          <video
            controls
            className="aspect-video w-full rounded-md bg-black object-contain"
            src={`${process.env.NEXT_PUBLIC_DEMO_API_URL || "http://localhost:8001"}${result.preview_url}`}
          />
        ) : !file ? (
          <div className="rounded-md border border-border px-3 py-6 text-sm text-muted">
            No file selected.
          </div>
        ) : canPreview && previewUrl ? (
          <video
            controls
            className="aspect-video w-full rounded-md bg-black object-contain"
            src={previewUrl}
          />
        ) : (
          <div className="rounded-md border border-border px-3 py-6 text-sm text-muted">
            Preview not available for this format ({file.name}).
          </div>
        )}
      </section>

      <section>
        <p className="mb-2 text-xs uppercase tracking-widest text-muted">
          Predicted Text
        </p>
        <p className="rounded-md bg-black/30 px-3 py-2 font-mono text-lg text-foreground">
          {result.predicted_text || "(empty prediction)"}
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <div className="rounded-md border border-border p-3">
          <p className="text-xs uppercase tracking-wide text-muted">WER</p>
          <p className="mt-1 text-lg font-semibold text-foreground">
            {displayPercent(result.wer)}
          </p>
        </div>
        <div className="rounded-md border border-border p-3">
          <p className="text-xs uppercase tracking-wide text-muted">CER</p>
          <p className="mt-1 text-lg font-semibold text-foreground">
            {displayPercent(result.cer)}
          </p>
        </div>
        <div className="rounded-md border border-border p-3">
          <p className="text-xs uppercase tracking-wide text-muted">
            Reference Source
          </p>
          <p className="mt-1 text-sm text-foreground">{result.reference_source}</p>
        </div>
      </section>

      <section className="space-y-1 rounded-md border border-border p-3">
        <p className="text-xs uppercase tracking-wide text-muted">Reference Text</p>
        <p className="text-sm text-foreground">{result.reference_text || "N/A"}</p>
      </section>

      <section className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-md border border-border p-3">
          <p className="text-xs uppercase tracking-wide text-muted">
            Preprocess (ms)
          </p>
          <p className="mt-1 text-sm text-foreground">
            {displayNumber(result.latency_ms.preprocess)}
          </p>
        </div>
        <div className="rounded-md border border-border p-3">
          <p className="text-xs uppercase tracking-wide text-muted">
            Inference (ms)
          </p>
          <p className="mt-1 text-sm text-foreground">
            {displayNumber(result.latency_ms.inference)}
          </p>
        </div>
        <div className="rounded-md border border-border p-3">
          <p className="text-xs uppercase tracking-wide text-muted">Total (ms)</p>
          <p className="mt-1 text-sm text-foreground">
            {displayNumber(result.latency_ms.total)}
          </p>
        </div>
      </section>

      <section className="rounded-md border border-border p-3">
        <p className="mb-2 text-xs uppercase tracking-wide text-muted">Video Stats</p>
        <div className="grid gap-1 text-sm text-foreground">
          <p>Filename: {result.video_stats.filename}</p>
          <p>Size: {result.video_stats.size_bytes} bytes</p>
          <p>
            Resolution: {result.video_stats.width ?? "N/A"} x{" "}
            {result.video_stats.height ?? "N/A"}
          </p>
          <p>FPS: {displayNumber(result.video_stats.fps)}</p>
          <p>Frame Count: {result.video_stats.frame_count ?? "N/A"}</p>
          <p>Duration (s): {displayNumber(result.video_stats.duration_sec)}</p>
          <p>Processed Shape: {result.video_stats.processed_shape.join(" x ")}</p>
        </div>
      </section>

      <section className="rounded-md border border-border p-3">
        <p className="mb-2 text-xs uppercase tracking-wide text-muted">Device Specs</p>
        <div className="grid gap-1 text-sm text-foreground">
          <p>Runtime Device: {result.device_specs.device_used}</p>
          <p>TensorFlow: {result.device_specs.tf_version}</p>
          <p>CPU: {result.device_specs.cpu_model ?? "N/A"}</p>
          <p>
            CPU Cores: {result.device_specs.cpu_physical_cores ?? "N/A"} physical /{" "}
            {result.device_specs.cpu_logical_cores ?? "N/A"} logical
          </p>
          <p>
            RAM: {displayNumber(result.device_specs.ram_total_gb)} GB total /{" "}
            {displayNumber(result.device_specs.ram_available_gb)} GB available
          </p>
          <p>
            GPU(s):{" "}
            {result.device_specs.gpu_names.length
              ? result.device_specs.gpu_names.join(", ")
              : "None detected"}
          </p>
          <p>
            GPU Total Memory (MB):{" "}
            {result.device_specs.gpu_memory_total_mb ?? "N/A"}
          </p>
        </div>
      </section>

      <section className="rounded-md border border-border p-3 text-xs text-muted">
        <p>Model used: {result.model_path_used}</p>
      </section>
    </div>
  );
}
