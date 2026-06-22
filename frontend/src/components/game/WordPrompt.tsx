export function WordPrompt({ word, slot }: { word: string; slot: string }) {
  return (
    <div className="text-center">
      <p className="text-xs uppercase tracking-widest text-gray-500">{slot}</p>
      <p className="text-5xl font-bold">{word}</p>
    </div>
  );
}
