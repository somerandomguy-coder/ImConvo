import os
import sys

import cv2
import numpy as np
from src.utils import (BLANK_IDX, MAX_CHAR_LEN, SPACE_IDX, _face_to_mouth_bbox,
                       char_indices_to_text, parse_alignment_chars,
                       text_to_char_indices)


def test_vocabulary():
    print("--- Testing Vocabulary Logic ---")
    test_str = "hello world"
    indices = text_to_char_indices(test_str)
    
    # Check 1: Length and Space
    assert len(indices) == len(test_str), "Index list length mismatch"
    assert indices[5] == SPACE_IDX, "Space character index incorrect"
    
    # Check 2: Round-trip (Indices -> Text)
    reconstructed = char_indices_to_text(indices)
    assert reconstructed == test_str, f"Round-trip failed: {reconstructed} != {test_str}"
    
    # Check 3: CTC Blank handling
    indices_with_blank = indices + [BLANK_IDX]
    reconstructed_no_blank = char_indices_to_text(indices_with_blank)
    assert reconstructed_no_blank == test_str, "char_indices_to_text should skip BLANK_IDX"
    
    print("✓ Vocabulary logic passed!")

def test_alignment_padding():
    print("\n--- Testing Alignment Padding ---")
    # We'll mock a small alignment file for this
    dummy_align = "dummy.align"
    with open(dummy_align, "w") as f:
        f.write("0 100 bin\n100 200 blue\n200 300 sil\n") # 'sil' should be ignored
    
    indices, length = parse_alignment_chars(dummy_align)
    
    # Check 1: Fixed Output Length
    assert len(indices) == MAX_CHAR_LEN, f"Output must be exactly {MAX_CHAR_LEN}"
    
    # Check 2: Correct Truth Length (bin=3, space=1, blue=4 -> 8)
    # text_to_char_indices("bin blue") -> length 8
    assert length == 8, f"Expected length 8, got {length}"
    
    # Check 3: Hotfix Verification
    assert indices[length] == BLANK_IDX, "Padding should start with BLANK_IDX (27)"
    
    os.remove(dummy_align)
    print("✓ Alignment parsing and padding passed!")

def test_crop_math():
    print("\n--- Testing Mouth BBox Math ---")
    # Face: x=100, y=100, w=200, h=200 (Frame size 500x500)
    face_bbox = (100, 100, 200, 200)
    x1, y1, x2, y2 = _face_to_mouth_bbox(face_bbox, 500, 500)
    
    # Simple geometry check:
    # Mouth Top Ratio is 0.65. Face height is 200.
    # Mouth starts at y = 100 (face start) + (200 * 0.65) = 230.
    # With padding, it should be even lower.
    assert y1 > 200, "Mouth y_start seems too high (should be in lower face)"
    assert x2 > x1 and y2 > y1, "BBox coordinates invalid"
    assert x2 <= 500 and y2 <= 500, "BBox out of frame bounds"
    
    print("✓ Mouth BBox geometry passed!")

if __name__ == "__main__":
    try:
        test_vocabulary()
        test_alignment_padding()
        test_crop_math()
        print("\n[ALL UNIT TESTS PASSED]")
    except AssertionError as e:
        print(f"\n[TEST FAILED] {e}")
