import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from clearml import Dataset, StorageManager, Task

from src import (NUM_CHARS, LipReadingCTC, char_indices_to_text,
                 create_dataset_pipeline)


# ---------------------------------------------------------------------------
# Metrics (Kept from original)
# ---------------------------------------------------------------------------
def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    r, h = len(ref_words), len(hyp_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): d[i][0] = i
    for j in range(h + 1): d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]: d[i][j] = d[i - 1][j - 1]
            else: d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+1)
    return d[r][h] / max(r, 1)

def compute_cer(reference: str, hypothesis: str) -> float:
    r, h = len(reference), len(hypothesis)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): d[i][0] = i
    for j in range(h + 1): d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if reference[i - 1] == hypothesis[j - 1]: d[i][j] = d[i - 1][j - 1]
            else: d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+1)
    return d[r][h] / max(r, 1)

# ---------------------------------------------------------------------------
# Main Evaluation Logic
# ---------------------------------------------------------------------------
def main():

    preprocessed_dir = "/home/nam/ImConvo/data/preprocessed"
    manifest_local_path = "/home/nam/ImConvo/data/preprocessed/manifest.txt"
    # 3. Load Dataset
    print(f"📂 Loading test samples from {preprocessed_dir}...")
    _, val_ds, _, _, _ = create_dataset_pipeline(
        preprocessed_dir=preprocessed_dir, 
        batch_size=48, 
        val_split=0.2, 
        seed=42,
        manifest_path=manifest_local_path 
    )

    # 4. Load Model
    print("🤖 Initializing LipReadingCTC...")
    model = LipReadingCTC(num_chars=NUM_CHARS)
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32)) # Dummy build
    model.load_weights("/home/nam/ImConvo/model_3.keras")

    # 5. Evaluation Loop
    total_wer, total_cer, num_samples = 0.0, 0.0, 0
    num_steps = 200 // 48
    

    for batch_idx, batch in enumerate(val_ds.take(num_steps)):
        x, y = batch
        logits = model(x, training=False)
        decoded_batch = model.decode_greedy(logits)

        labels = y["labels"].numpy()
        lengths = y["label_length"].numpy()

        for i in range(len(labels)):
            gt_text = char_indices_to_text(labels[i][: lengths[i]].tolist())
            pred_indices = decoded_batch[i]
            pred_indices = pred_indices[pred_indices >= 0]
            pred_text = char_indices_to_text(pred_indices.tolist())

            wer = compute_wer(gt_text, pred_text)
            cer = compute_cer(gt_text, pred_text)

            total_wer += wer
            total_cer += cer
            num_samples += 1

            # Log some examples to ClearML console
            if num_samples % 50 == 0:
                print(f"Sample {num_samples} | GT: {gt_text} | Pred: {pred_text}")

    # 6. Final Reporting
    avg_wer = total_wer / max(num_samples, 1)
    avg_cer = total_cer / max(num_samples, 1)


    print(f"\n✅ Evaluation Complete. Avg WER: {avg_wer:.4f}")
    report_dir = "./temp_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    report_filename = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(report_dir, report_filename)

    with open(report_path, "w") as f:
        f.write(f"--- LipReading AI Evaluation Report ---\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Total Samples Evaluated: {num_samples}\n")
        f.write(f"Average Word Error Rate: {avg_wer:.4f} ({avg_wer:.1%})\n")
        f.write(f"Average Char Error Rate: {avg_cer:.4f} ({avg_cer:.1%})\n")
        f.write(f"{'='*40}\n")
        f.write("\nNote: This report was generated automatically by the ImConvo Pipeline.")

    # --- PART C: The Bridge (Upload the report back to ClearML) ---
    # This makes the local file visible in the "Artifacts" tab of the UI!

    print(f"\n✓ Evaluation Complete.")
    print(f"📊 Live results: Check ClearML 'Scalars' tab.")
    print(f"📄 Local report: {report_path}")

    # 7. Visualization (The Loss Plot)
    print("📈 Generating training loss plots...")
    from src.visualization import save_loss_plot
    
    history_path = "/home/nam/ImConvo/model_3.json"
        
    # 2. Run your original visualization function
    # This usually saves a .png or .jpg
    plot_output_dir = "./temp_reports"
    plot_output_path = "./temp_reports/model_3.png"
    save_loss_plot(history_path, output_dir=plot_output_dir)


    print(f"✓ Loss plot uploaded to ClearML and saved to {plot_output_path}")
if __name__ == "__main__":
    main()
