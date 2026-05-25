import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 1. Load character-level metrics from experiments.jsonl ]
EXPERIMENTS = Path("experiments.jsonl")
best_char = None
with open(EXPERIMENTS) as f:
    for line in f:
        d = json.loads(line)
        if (d.get("model_variant") == "conformer_lite"
                and d.get("frontend_model") == "flatten"
                and d.get("test_oos_wer") is not None):
            if best_char is None or d["test_oos_wer"] < best_char["test_oos_wer"]:
                best_char = d

char_wer = best_char["test_oos_wer"] * 100
char_cer = best_char["test_oos_cer"] * 100
print(f"[char]  WER={char_wer:.2f}%  CER={char_cer:.2f}%  run={best_char.get('run_id','')}")

# 2. Load word-level metrics from experiments_word_level.jsonl
WORD_EXPERIMENTS = Path("experiments_word_level.jsonl")
best_word = None
with open(WORD_EXPERIMENTS) as f:
    for line in f:
        d = json.loads(line)
        if d.get("test_oos_wer") is not None:
            if best_word is None or d["test_oos_wer"] < best_word["test_oos_wer"]:
                best_word = d

word_wer = best_word["test_oos_wer"] * 100
word_cer = best_word["test_oos_cer"] * 100
print(f"[word]  WER={word_wer:.2f}%  CER={word_cer:.2f}%  run={best_word.get('run_id','')}")

# 3. Plot 
BLUE  = "#4A90C4"
GREEN = "#7BB661"

groups  = ["WER (%)", "CER (%)"]
char_vals = [char_wer, char_cer]
word_vals = [word_wer, word_cer]

x      = np.arange(len(groups))
width  = 0.30
gap    = 0.04

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars_char = ax.bar(x - width/2 - gap/2, char_vals, width, color=BLUE,  label="Character-level", zorder=3)
bars_word = ax.bar(x + width/2 + gap/2, word_vals, width, color=GREEN, label="Word-level",       zorder=3)

# value labels on top of bars
for bar in list(bars_char) + list(bars_word):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.25,
            f"{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333")


# 5. Styling 
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel("Error Rate (%)", fontsize=10)
ax.set_ylim(0, max(char_wer, word_wer, char_cer, word_cer) + 5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#DDDDDD")
ax.set_axisbelow(True)

ax.legend(handles=[
    mpatches.Patch(color=BLUE,  label="Character-level"),
    mpatches.Patch(color=GREEN, label="Word-level"),
], fontsize=9, framealpha=0.9, edgecolor="#DDDDDD", loc="upper right")

fig.text(0.5, 0.97, "Character-level vs Word-level: Test OOS Performance",
         ha="center", va="top", fontsize=13, fontweight="bold")
fig.text(0.5, 0.91, "WER favours word-level; CER favours character-level",
         ha="center", va="top", fontsize=9.5, color="#666666")

plt.tight_layout(rect=[0, 0, 1, 0.90])

out = Path("reports/plots/char_vs_word_test_oos.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
