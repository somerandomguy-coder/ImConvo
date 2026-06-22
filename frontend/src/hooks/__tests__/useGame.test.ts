import { beforeEach, describe, expect, it } from "vitest";
import { gameReducer, initGameState } from "../useGame";
import type { Sentence } from "../../lib/gridGrammar";

const SENTENCE: Sentence = {
  words: ["bin", "blue", "at", "a", "zero", "again"],
  slots: ["command", "color", "preposition", "letter", "digit", "adverb"],
};

describe("gameReducer", () => {
  beforeEach(() => localStorage.clear());

  it("advances round and scores on pass", () => {
    const s0 = initGameState(SENTENCE);
    const s1 = gameReducer(s0, { type: "ROUND_PASS" });
    expect(s1.roundIndex).toBe(1);
    expect(s1.score).toBe(1);
    expect(s1.status).toBe("playing");
    expect(s1.lastResult).toBe("pass");
  });

  it("stays on same round and counts attempt on fail (no lose state)", () => {
    const s0 = initGameState(SENTENCE);
    const s1 = gameReducer(s0, { type: "ROUND_FAIL" });
    expect(s1.roundIndex).toBe(0);
    expect(s1.score).toBe(0);
    expect(s1.attempts).toBe(1);
    expect(s1.status).toBe("playing");
  });

  it("wins after six passes and bumps streak", () => {
    let s = initGameState(SENTENCE);
    for (let i = 0; i < 6; i++) s = gameReducer(s, { type: "ROUND_PASS" });
    expect(s.status).toBe("won");
    expect(s.streak).toBe(1);
    expect(s.bestStreak).toBe(1);
    expect(localStorage.getItem("imconvo.game.bestStreak")).toBe("1");
  });

  it("starts a new sentence and resets round/attempts", () => {
    let s = initGameState(SENTENCE);
    s = gameReducer(s, { type: "ROUND_FAIL" });
    s = gameReducer(s, { type: "NEW_SENTENCE", sentence: SENTENCE });
    expect(s.roundIndex).toBe(0);
    expect(s.attempts).toBe(0);
    expect(s.status).toBe("playing");
  });
});
