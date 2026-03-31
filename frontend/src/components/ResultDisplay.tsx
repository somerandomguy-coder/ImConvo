"use client";

import { useTypingEffect } from "@/hooks/useTypingEffect";

interface ResultDisplayProps {
  text: string;
  isLoading: boolean;
}

export default function ResultDisplay({ text, isLoading }: ResultDisplayProps) {
  const { displayed, isTyping } = useTypingEffect(text);

  if (!isLoading && !text) return null;

  return (
    <div className="w-full rounded-xl border border-border bg-card p-6">
      <p className="mb-3 text-xs font-medium uppercase tracking-widest text-muted">
        Predicted Text
      </p>
      {isLoading ? (
        <div className="flex items-center gap-3">
          <svg
            className="h-5 w-5 animate-spin text-accent"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span className="text-sm text-muted">Analyzing lip movements...</span>
        </div>
      ) : (
        <p className="font-mono text-xl leading-relaxed text-foreground">
          {displayed}
          {isTyping && <span className="animate-cursor">|</span>}
        </p>
      )}
    </div>
  );
}
