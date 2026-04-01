import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the project root (one level up)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Append the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import (BLANK_IDX, MAX_CHAR_LEN, char_indices_to_text,
                       parse_alignment_chars, text_to_char_indices)

# ── 1. Vocabulary Flow (String -> Indices -> String)
# format: (input_text, description)
vocab_tests = [
    ("hello", "Standard word"),
    ("blue at g 9", "GRID sentence"), # missing a 9 lol
    (" ", "Single space"),
    ("!!??", "Special chars (should be ignored)"),
]

print("=== 1. VOCABULARY DATA FLOW ===")
for text, label in vocab_tests:
    indices = text_to_char_indices(text)
    reconstructed = char_indices_to_text(indices)
    
    # We strip special chars in the reconstruction test because 
    # our CHAR_LIST only contains a-z and space.
    print(f"[{label:15}] '{text}' -> {indices} -> '{reconstructed}'")

# ── 2. Alignment & Padding Flow (File -> Padded Array)
# Let's mock a tiny alignment file content
mock_align_content = "0 100 bin\n100 200 blue\n200 300 sil\n"
mock_file = "test_sample.align"

with open(mock_file, "w") as f:
    f.write(mock_align_content)

print("\n=== 2. ALIGNMENT PADDING FLOW ===")
# Testing how a raw file becomes a fixed-size tensor
indices, length = parse_alignment_chars(mock_file)

print(f"[Raw Content]  : 'bin blue' (silence ignored)")
print(f"[True Length]  : {length} characters")
print(f"[Padded Array] : {indices[:15]}... (showing first 15)")
print(f"[Tail Check]   : Last 5 elements: {indices[-5:]} (should be {BLANK_IDX})")

if len(indices) == MAX_CHAR_LEN:
    print(f"✅ SUCCESS: Array size is exactly {MAX_CHAR_LEN}")

os.remove(mock_file)

# ── 3. Character mapping check
print("\n=== 3. SPECIFIC CHARACTER MAPPING ===")
char_checks = ['a', 'z', ' ', 'blank']
for c in char_checks:
    if c == 'blank':
        idx = BLANK_IDX
    else:
        idx = text_to_char_indices(c)[0]
    print(f"Char: '{c}' -> Index: {idx}")
