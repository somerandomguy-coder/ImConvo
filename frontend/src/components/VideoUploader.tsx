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
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-16 text-center transition-colors ${
            isDragActive
              ? "border-accent bg-accent/5"
              : "border-border hover:border-muted"
          }`}
        >
          <input {...getInputProps()} />
          <svg
            className="mb-4 h-10 w-10 text-muted"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
          <p className="text-sm font-medium text-foreground">
            {isDragActive ? "Drop your video here" : "Drag & drop a video file"}
          </p>
          <p className="mt-1 text-xs text-muted">
            or click to browse — MP4, MPG, AVI, MOV, WebM
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="overflow-hidden rounded-xl border border-border bg-card">
            {canPlay ? (
              <video
                src={preview!}
                controls
                className="aspect-video w-full bg-black object-contain"
              />
            ) : (
              <div className="flex aspect-video flex-col items-center justify-center bg-black/50">
                <svg
                  className="mb-3 h-12 w-12 text-muted"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
                  />
                </svg>
                <p className="text-sm font-medium text-foreground">{file.name}</p>
                <p className="mt-1 text-xs text-muted">
                  {formatFileSize(file.size)} — preview not available for this format
                </p>
              </div>
            )}
          </div>
          <div className="flex items-center justify-between">
            <p className="truncate text-sm text-muted">
              {file.name} ({formatFileSize(file.size)})
            </p>
            <button
              onClick={() => {
                remove();
                onRemoved?.();
              }}
              className="text-sm text-muted transition-colors hover:text-foreground"
            >
              Remove
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
