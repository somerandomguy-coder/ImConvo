export const SLOT_NAMES = [
  "command",
  "color",
  "preposition",
  "letter",
  "digit",
  "adverb",
] as const;

export const SLOT_VOCABS: readonly (readonly string[])[] = [
  ["bin", "lay", "place", "set"],
  ["blue", "green", "red", "white"],
  ["at", "by", "in", "with"],
  ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","x","y","z"],
  ["zero","one","two","three","four","five","six","seven","eight","nine"],
  ["again", "now", "please", "soon"],
];

export type Sentence = { words: string[]; slots: string[] };

export function generateSentence(rng: () => number = Math.random): Sentence {
  const words = SLOT_VOCABS.map((vocab) => vocab[Math.floor(rng() * vocab.length)]);
  return { words, slots: [...SLOT_NAMES] };
}
