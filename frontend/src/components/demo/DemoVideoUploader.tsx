"use client";

import { useEffect, useMemo, useRef } from "react";

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface DemoVideoUploaderProps {
  file: File | null;
  onChange: (file: File | null) => void;
}

export default function DemoVideoUploader({
  file,
  onChange,
}: DemoVideoUploaderProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
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

  return (
    <div className="w-full rounded-xl border border-border bg-card p-5">
      {!file && (
        <div className="flex flex-col items-start gap-3">
          <p className="text-sm text-muted">
            Upload one video for offline lip-reading inference.
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="video/*,.mpg,.mpeg,.avi,.mov,.webm,.mp4"
            onChange={(e) => onChange(e.target.files?.[0] || null)}
            className="block w-full text-sm text-foreground file:mr-4 file:rounded-md file:border-0 file:bg-accent file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-accent-hover"
          />
        </div>
      )}

      {file && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <p className="truncate text-sm text-muted">
              {file.name} ({formatFileSize(file.size)})
            </p>
            <button
              onClick={() => {
                if (inputRef.current) inputRef.current.value = "";
                onChange(null);
              }}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-muted transition-colors hover:border-foreground hover:text-foreground"
              type="button"
            >
              Remove
            </button>
          </div>

          {canPreview && previewUrl ? (
            <video
              controls
              className="aspect-video w-full rounded-lg bg-black object-contain"
              src={previewUrl}
            />
          ) : (
            <div className="rounded-lg border border-border px-4 py-10 text-center text-sm text-muted">
              Preview not available for this file format.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
