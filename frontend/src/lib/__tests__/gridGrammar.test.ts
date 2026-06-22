import { describe, expect, it } from "vitest";
import { SLOT_NAMES, SLOT_VOCABS, generateSentence } from "../gridGrammar";

describe("gridGrammar", () => {
  it("has six slots in GRID order", () => {
    expect(SLOT_NAMES).toEqual([
      "command",
      "color",
      "preposition",
      "letter",
      "digit",
      "adverb",
    ]);
    expect(SLOT_VOCABS).toHaveLength(6);
  });

  it("generates one valid word per slot", () => {
    const s = generateSentence();
    expect(s.words).toHaveLength(6);
    expect(s.slots).toEqual([...SLOT_NAMES]);
    s.words.forEach((w, i) => {
      expect(SLOT_VOCABS[i]).toContain(w);
    });
  });

  it("is deterministic given an rng", () => {
    const rng = () => 0; // always picks the first word in each slot
    const s = generateSentence(rng);
    expect(s.words).toEqual(["bin", "blue", "at", "a", "zero", "again"]);
  });
});
