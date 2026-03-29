import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from src import (
    create_dataset_pipeline, 
    LipReadingCTC, 
    char_indices_to_text, 
    NUM_CHARS
)
# Re-use your compute functions from train.py
from train import compute_wer, compute_cer



def run_evaluation():

    TEST_SAMPLES = 200  # Only test 200 samples for a quick, accurate estimate
    BATCH_SIZE = 8
    num_steps = TEST_SAMPLES // BATCH_SIZE

    # 1. Setup paths
    checkpoint_path = "./checkpoints/best_ctc_model.keras"
    data_dir = "./data/"
    preprocessed_dir = "./data/preprocessed/"
    report_path = f"./reports/eval_result/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load Dataset (Specifically the validation/test portion)
    print("📂 Loading test dataset...")
    _, val_ds, val_paths, _, _ = create_dataset_pipeline(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        batch_size=BATCH_SIZE,
        val_split=0.2, 
        seed=42
    )

    # 3. Load Model
    print("🤖 Loading model weights...")
    model = LipReadingCTC(num_chars=NUM_CHARS)
    # Build model with dummy input
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32))
    model.load_weights(checkpoint_path)

    # 4. Evaluation Loop
    total_wer, total_cer, num_samples = 0.0, 0.0, 0
    
    print(f"📝 Evaluating {len(val_paths)} samples. Writing results to {report_path}...")
        
    with open(report_path, "w", buffering=1024*1024) as f:
        f.write(f"LipNet Evaluation Report - {datetime.now()}\n")
        f.write(f"{'='*60}\n\n")

        # Use .take() so it doesn't run through the entire 20% split
        for batch_idx, batch in enumerate(val_ds.take(num_steps)):
            x, y = batch
            
            # Inference & Decoding (Ensure this happens on GPU)
            logits = model(x, training=False)
            decoded_batch = model.decode_greedy(logits) 

            labels = y["labels"].numpy()
            lengths = y["label_length"].numpy()

            for i in range(len(labels)):
                gt_text = char_indices_to_text(labels[i][:lengths[i]].tolist())
                
                # Clean up prediction
                pred_indices = decoded_batch[i]
                pred_indices = pred_indices[pred_indices >= 0]
                pred_text = char_indices_to_text(pred_indices.tolist())

                # Metrics
                wer = compute_wer(gt_text, pred_text)
                cer = compute_cer(gt_text, pred_text)
                
                total_wer += wer
                total_cer += cer
                num_samples += 1

                # Disk write (Buffered)
                f.write(f"Sample {num_samples} | WER: {wer:.2f} | GT: '{gt_text}' | Pred: '{pred_text}'\n")

            # Progress indicator so you know it's alive
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {num_samples}/{TEST_SAMPLES} samples...")
        # Final Summary
        avg_wer = total_wer / max(num_samples, 1)
        avg_cer = total_cer / max(num_samples, 1)
        
        summary = (
            f"\n{'='*60}\n"
            f"FINAL SUMMARY\n"
            f"Total Samples: {num_samples}\n"
            f"Average WER:   {avg_wer:.4f} ({avg_wer*100:.1f}%)\n"
            f"Average CER:   {avg_cer:.4f} ({avg_cer*100:.1f}%)\n"
            f"{'='*60}\n"
        )
        f.write(summary)
        print(summary)
    from src.visualization import save_loss_plot
    save_loss_plot("./checkpoints/training_history.json")
if __name__ == "__main__":
    run_evaluation()