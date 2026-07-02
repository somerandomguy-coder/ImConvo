"""Build a cached GRID transcript trigram language model artifact.

This is used by the demo decoder mode `beam_ngram_grid`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.decoding import GRID_NGRAM_ARTIFACT, _build_grid_ngram_artifact


def main() -> None:
    output_path = Path(GRID_NGRAM_ARTIFACT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _build_grid_ngram_artifact(output_path)
    print(f"Built GRID n-gram LM artifact at: {output_path}")


if __name__ == "__main__":
    main()
