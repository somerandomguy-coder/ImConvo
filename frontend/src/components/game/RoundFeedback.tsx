export function RoundFeedback({ result }: { result: "pass" | "fail" | null }) {
  if (!result) return null;
  return result === "pass" ? (
    <p className="text-emerald-600 font-semibold">✓ Correct!</p>
  ) : (
    <p className="text-rose-600 font-semibold">✗ Try again</p>
  );
}
