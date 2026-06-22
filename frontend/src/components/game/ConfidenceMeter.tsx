export function ConfidenceMeter({ confidence, word, face }: { confidence: number; word: string; face: boolean }) {
  if (!face) {
    return <p className="text-amber-500">Center your face in frame…</p>;
  }
  const pct = Math.round(confidence * 100);
  return (
    <div className="w-full">
      <div className="flex justify-between text-sm">
        <span>{word || "…"}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-3 w-full rounded bg-gray-200">
        <div className="h-3 rounded bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
