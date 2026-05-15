import os
import sys
import tensorflow as tf

# Add project root to path
sys.path.append(os.getcwd())

from src.model import build_lipreading_ctc
from experiments.isolated_word_level.model import build_lipreading_isolated_word_classifier

def transplant():
    # 1. SETUP PATHS
    isolated_weights_path = "checkpoints/isolated_word_level/best_mv-conformer-lite_fe-flatten_be-conformer-lite_bs-48_t-25_20260514T072743Z_conformer-lite_flatten_conformer-lite_isolated_word_v1.weights.h5"
    output_path = "checkpoints/ctc_pretrained_backbone.weights.h5"

    print("--- Phase 1: Building Models ---")
    # Build Isolated Model (Source)
    isolated_model = build_lipreading_isolated_word_classifier(
        model_variant="conformer_lite",
        frontend_model="flatten",
        num_word_classes=51,
    )
    
    # Build CTC Model (Target)
    ctc_model = build_lipreading_ctc(
        model_variant="conformer_lite",
        frontend_model="flatten",
        num_chars=28 
    )

    # Initialize both with dummy data
    dummy_x = tf.zeros((1, 75, 80, 120, 1))
    _ = isolated_model(dummy_x, training=False)
    _ = ctc_model(dummy_x, training=False)
    
    # Load the 90% accuracy weights
    isolated_model.load_weights(isolated_weights_path, skip_mismatch=True)
    print("✓ Isolated weights loaded (with skip_mismatch for conformer_proj).")

    print("\n--- Phase 2: Surgical Layer Mapping ---")
    
    # List of attributes to copy from Isolated -> CTC
    # These names exist in BOTH your Isolated and CTC classes
    print("\n--- Phase 2: Surgical Layer Mapping ---")
    
    shared_layers = [
        'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
        'conformer_proj', 'conformer_pos_embed', 'conformer_blocks'
    ]

    for layer_name in shared_layers:
        if hasattr(isolated_model, layer_name) and hasattr(ctc_model, layer_name):
            src_attr = getattr(isolated_model, layer_name)
            tgt_attr = getattr(ctc_model, layer_name)
            
            # If it's a list (like conformer_blocks), it won't have set_weights
            # but its elements will.
            if isinstance(src_attr, list) or not hasattr(src_attr, 'set_weights'):
                try:
                    print(f"--- Processing Collection: {layer_name} ---")
                    for i, (src_block, tgt_block) in enumerate(zip(src_attr, tgt_attr)):
                        tgt_block.set_weights(src_block.get_weights())
                        print(f"  [✓] Copied weights for {layer_name} block {i}")
                except TypeError:
                    # This handles cases where it's neither a layer nor an iterable
                    print(f"[!] Skipping {layer_name}: Not a weight-bearing layer or list.")
            else:
                # Standard single layer (Conv3D, BatchNormalization, Dense, etc.)
                tgt_attr.set_weights(src_attr.get_weights())
                print(f"[✓] Copied weights for: {layer_name}")
    # 3. SAVE
    ctc_model.save_weights(output_path)
    print(f"\nSUCCESS: Transplant complete. Weights saved to {output_path}")

if __name__ == "__main__":
    transplant()