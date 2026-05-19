"use client";

interface ResultDisplayProps {
  text: string;
  isLoading: boolean;
}

export default function ResultDisplay({ text, isLoading }: ResultDisplayProps) {
  if (!isLoading && !text) return null;

  return (
    <div className="animate-fade-in-up w-full rounded-xl border border-border bg-card px-5 py-4 shadow-sm">
      <p className="mb-2.5 text-xs text-muted">Predicted text</p>
      {isLoading ? (
        <div className="flex items-center gap-3">
          <span className="flex gap-1">
            <span
              className="animate-bounce-dot h-1.5 w-1.5 rounded-full bg-accent"
              style={{ animationDelay: "0ms" }}
            />
            <span
              className="animate-bounce-dot h-1.5 w-1.5 rounded-full bg-accent"
              style={{ animationDelay: "160ms" }}
            />
            <span
              className="animate-bounce-dot h-1.5 w-1.5 rounded-full bg-accent"
              style={{ animationDelay: "320ms" }}
            />
          </span>
          <span className="text-sm text-muted">Analyzing lip movements…</span>
        </div>
      ) : (
        <p
          key={text}
          className="font-mono text-lg leading-relaxed text-foreground animate-fade-in"
        >
          {text}
        </p>
      )}
    </div>
  );
}
