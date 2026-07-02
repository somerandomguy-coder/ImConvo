import { Fredoka } from "next/font/google";
import "./game.css";

const fredoka = Fredoka({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-fredoka" });

export default function GameLayout({ children }: { children: React.ReactNode }) {
  return <div className={`${fredoka.variable} game-root`}>{children}</div>;
}
