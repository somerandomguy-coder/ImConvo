import { useReducer } from "react";
import { generateSentence, type Sentence } from "../lib/gridGrammar";

export type GameStatus = "playing" | "won";

export type GameState = {
  sentence: Sentence;
  roundIndex: number;
  status: GameStatus;
  score: number;
  streak: number;
  bestStreak: number;
  attempts: number;
  lastResult: "pass" | "fail" | null;
};

export type GameAction =
  | { type: "ROUND_PASS" }
  | { type: "ROUND_FAIL" }
  | { type: "NEW_SENTENCE"; sentence?: Sentence }
  | { type: "SKIP_WORD" };

const BEST_KEY = "imconvo.game.bestStreak";
const SCORE_KEY = "imconvo.game.score";

function readNumber(key: string): number {
  if (typeof localStorage === "undefined") return 0;
  const raw = localStorage.getItem(key);
  return raw ? Number(raw) || 0 : 0;
}

function write(key: string, value: number): void {
  if (typeof localStorage !== "undefined") localStorage.setItem(key, String(value));
}

export function initGameState(sentence: Sentence = generateSentence()): GameState {
  return {
    sentence,
    roundIndex: 0,
    status: "playing",
    score: readNumber(SCORE_KEY),
    streak: 0,
    bestStreak: readNumber(BEST_KEY),
    attempts: 0,
    lastResult: null,
  };
}

export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "ROUND_PASS": {
      const score = state.score + 1;
      write(SCORE_KEY, score);
      const nextRound = state.roundIndex + 1;
      if (nextRound >= state.sentence.words.length) {
        const streak = state.streak + 1;
        const bestStreak = Math.max(state.bestStreak, streak);
        write(BEST_KEY, bestStreak);
        return { ...state, status: "won", score, streak, bestStreak, attempts: 0, lastResult: "pass" };
      }
      return { ...state, roundIndex: nextRound, score, attempts: 0, lastResult: "pass" };
    }
    case "ROUND_FAIL":
      return { ...state, attempts: state.attempts + 1, lastResult: "fail" };
    case "SKIP_WORD": {
      const nextRound = state.roundIndex + 1;
      if (nextRound >= state.sentence.words.length) {
        return { ...state, status: "won", attempts: 0, lastResult: null };
      }
      return { ...state, roundIndex: nextRound, attempts: 0, lastResult: null };
    }
    case "NEW_SENTENCE":
      return {
        ...state,
        sentence: action.sentence ?? generateSentence(),
        roundIndex: 0,
        status: "playing",
        streak: state.status === "won" ? state.streak : 0,
        attempts: 0,
        lastResult: null,
      };
    default:
      return state;
  }
}

export function useGame(initial?: Sentence) {
  return useReducer(gameReducer, initGameState(initial));
}
