"""LLM post-processor for GRID corpus lip-reading predictions.

Snaps noisy CTC output to the nearest valid GRID sentence using Gemini.
Falls back to local edit-distance search if the API is unavailable.
"""

from __future__ import annotations

import os
from functools import lru_cache

import requests

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

GRID_COMMANDS = ["bin", "lay", "place", "set"]
GRID_COLORS = ["blue", "green", "red", "white"]
GRID_PREPOSITIONS = ["at", "by", "in", "with"]
# GRID excludes 'w' due to syllable length ambiguity (25 letters)
GRID_LETTERS = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != "w"]
GRID_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
GRID_ADVERBS = ["again", "now", "please", "soon"]

_SYSTEM_PROMPT = f"""You are a corrector for a lip-reading model trained on the GRID corpus.

GRID sentences follow EXACTLY this 6-word structure:
  <command> <color> <preposition> <letter> <digit> <adverb>

Valid words per position:
- Commands:      {", ".join(GRID_COMMANDS)}
- Colors:        {", ".join(GRID_COLORS)}
- Prepositions:  {", ".join(GRID_PREPOSITIONS)}
- Letters:       {", ".join(GRID_LETTERS)}
- Digits:        {", ".join(GRID_DIGITS)}
- Adverbs:       {", ".join(GRID_ADVERBS)}

Rules:
1. Output EXACTLY 6 words separated by single spaces.
2. Each word MUST come from its position's valid set.
3. Pick the phonetically/visually closest word for each position.
4. Return ONLY the corrected sentence — no explanation, no punctuation, no quotes."""


class LLMPostprocessor:
    """Post-processes CTC predictions by snapping them to valid GRID sentences."""

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Pass api_key= or export GEMINI_API_KEY=..."
            )
        self._model = model
        self._cache: dict[str, str] = {}

    def correct(self, pred_text: str) -> str:
        """Snap noisy CTC text to the nearest valid GRID sentence.

        Returns:
            Corrected 5-word GRID sentence (lowercase).
        """
        pred_text = pred_text.strip().lower()
        if pred_text in self._cache:
            return self._cache[pred_text]

        try:
            url = f"{_GEMINI_API_BASE}/{self._model}:generateContent?key={self._api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f'Noisy prediction: "{pred_text}"\n\nCorrect:'}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 64,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"].get("parts", [])
            corrected = parts[0]["text"].strip().lower() if parts else _local_nearest(pred_text)
        except Exception as exc:
            print(f"[LLM] API error ({exc}); using edit-distance fallback.")
            corrected = _local_nearest(pred_text)

        self._cache[pred_text] = corrected
        return corrected

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Local fallback: minimum edit distance over all valid GRID sentences
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = (
                prev[j - 1]
                if a[i - 1] == b[j - 1]
                else 1 + min(prev[j], dp[j - 1], prev[j - 1])
            )
    return dp[n]


@lru_cache(maxsize=1)
def _all_grid_sentences() -> list[str]:
    sentences = []
    for cmd in GRID_COMMANDS:
        for color in GRID_COLORS:
            for prep in GRID_PREPOSITIONS:
                for letter in GRID_LETTERS:
                    for digit in GRID_DIGITS:
                        for adverb in GRID_ADVERBS:
                            sentences.append(f"{cmd} {color} {prep} {letter} {digit} {adverb}")
    return sentences


def _local_nearest(pred_text: str) -> str:
    return min(_all_grid_sentences(), key=lambda s: _edit_distance(pred_text, s))
