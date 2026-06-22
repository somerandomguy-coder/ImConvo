export function ResultScreen({ onNewSentence }: { onNewSentence: () => void }) {
  return (
    <div className="text-center space-y-4">
      <p className="text-3xl font-bold">🎉 You read the whole sentence!</p>
      <button onClick={onNewSentence} className="rounded bg-emerald-600 px-4 py-2 text-white">
        New sentence
      </button>
    </div>
  );
}
