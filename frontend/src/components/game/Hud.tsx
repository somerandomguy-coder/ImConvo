export function Hud({ score, streak, bestStreak, attempts }: { score: number; streak: number; bestStreak: number; attempts: number }) {
  return (
    <div className="flex gap-6 text-sm">
      <span>Score: {score}</span>
      <span>Streak: {streak}</span>
      <span>Best: {bestStreak}</span>
      <span>Attempts: {attempts}</span>
    </div>
  );
}
