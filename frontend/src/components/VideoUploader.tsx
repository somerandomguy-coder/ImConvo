"use client";

import { useVideoUpload, formatFileSize } from "@/hooks/useVideoUpload";

interface VideoUploaderProps {
  onFileSelected: (file: File) => void;
  onRemoved?: () => void;
}

export default function VideoUploader({ onFileSelected, onRemoved }: VideoUploaderProps) {
  const { file, preview, canPlay, isDragActive, getRootProps, getInputProps, remove } =
    useVideoUpload(onFileSelected);

  return (
    <div className="w-full">
      {!file ? (
        <div
          {...getRootProps()}
          className={`group flex cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border px-8 py-14 text-center shadow-sm transition-all duration-200 ${
            isDragActive
              ? "border-accent/50 bg-accent/5"
              : "border-border bg-card hover:border-accent/30 hover:bg-card"
          }`}
        >
          <input {...getInputProps()} />
          <div
            className={`flex h-11 w-11 items-center justify-center rounded-full border transition-colors ${
              isDragActive
                ? "border-accent/40 bg-accent/10 text-accent"
                : "border-border bg-card text-muted group-hover:border-border/80"
            }`}
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">
              {isDragActive ? "Release to upload" : "Drop a video here"}
            </p>
            <p className="mt-1 text-xs text-muted">
              or click to browse &mdash; MP4, MOV, AVI, WebM
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-2.5">
          <div className="overflow-hidden rounded-xl border border-border shadow-sm">
            {canPlay ? (
              <video
                src={preview!}
                controls
                className="aspect-video w-full bg-black object-contain"
              />
            ) : (
              <div className="flex aspect-video flex-col items-center justify-center gap-2 bg-card">
                <svg className="h-9 w-9 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
                  />
                </svg>
                <p className="text-sm text-muted">{file.name}</p>
                <p className="text-xs text-muted/60">{formatFileSize(file.size)}</p>
              </div>
            )}
          </div>
          <div className="flex items-center justify-between text-xs text-muted">
            <span className="truncate">
              {file.name} &middot; {formatFileSize(file.size)}
            </span>
            <button
              onClick={() => {
                remove();
                onRemoved?.();
              }}
              className="ml-3 shrink-0 transition-colors hover:text-foreground"
            >
              Remove
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
