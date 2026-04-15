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
    parser = argparse.ArgumentParser()
    # IDs passed from PipelineController
    parser.add_argument("--train_task_id", default="TOBE_OVERRIDDEN", help="Task ID of the trainer")
    parser.add_argument("--preprocess_task_id", default="TOBE_OVERRIDDEN", help="Task ID of preprocessor")
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--remote", action="store_true", help="Remote execution flag")

    args = parser.parse_args()

    # 1. Initialize ClearML
    task = Task.init(
        project_name="ImConvo",
        task_name="Model_Evaluation",
        task_type=Task.TaskTypes.testing
    )
    print('Arguments: {}'.format(args))
    remote = args.remote
    if remote:
        task.execute_remotely()

    # 2. Retrieve Model & Data Paths from Pipeline Parents
    print(f"🔗 Linking to Training Task")
    artifact_file_path = StorageManager.get_local_copy(remote_url=args.train_task_id)
    print(f"type of artifact_path: {type(artifact_file_path)}")
    if artifact_file_path == None:
        artifact_file_path = args.train_task_id
        print(f"{args.train_task_id} is not a valid url")

    if os.path.exists(str(artifact_file_path)) and os.path.isfile(str(artifact_file_path)):
        try:
            with open(artifact_file_path, 'r') as f:
                # If it's a JSON file, parse it and look for the 'id' key
                if str(artifact_file_path).endswith('.json'):
                    data = json.load(f)
                    # Look for 'id' at root or inside 'preview' based on your JSON structure
                    actual_parent_id = data.get('id') or data.get('preview', {}).get('id')
                    print(f"✓ Parsed ID from JSON file: {actual_parent_id}")
                else:
                    # If it's just a text file, read the raw string
                    actual_parent_id = f.read().strip()
                    print(f"✓ Read ID from text file: {actual_parent_id}")
        except Exception as e:
            print(f"  [WARN] Failed to parse file {artifact_file_path}: {e}")
    else:
        actual_parent_id = "b0e56b32aa0f424eb112098fabac8238"
        print(f"⚠️ Directory/File not found. Using hardcoded fallback ID: {actual_parent_id}")

    train_task = Task.get_task(task_id=actual_parent_id) # This node have only 1 parent 

    # Get the best model artifact. ClearML downloads it to a temp local path.
    # Note: Ensure your train.py used task.update_output_model() or task.upload_artifact('model')
    model_artifact = train_task.artifacts.get('best_ctc_model') 
    if not model_artifact:
        # Fallback: Check if it was registered as an Output Model
        models = train_task.get_models()
        checkpoint_path = models['output'][0].get_local_copy() if models.get('output') else None
    else:
        checkpoint_path = model_artifact.get_local_copy()

    # Get Preprocessed Data Path
    artifact_file_path = StorageManager.get_local_copy(remote_url=args.preprocess_task_id)
    print(f"type of artifact_path: {type(artifact_file_path)}")
    if artifact_file_path == None:
        artifact_file_path = args.preprocess_task_id
        print(f"{args.preprocess_task_id} is not a valid url")

    if os.path.exists(str(artifact_file_path)) and os.path.isfile(str(artifact_file_path)):
        try:
            with open(artifact_file_path, 'r') as f:
                # If it's a JSON file, parse it and look for the 'id' key
                if str(artifact_file_path).endswith('.json'):
                    data = json.load(f)
                    # Look for 'id' at root or inside 'preview' based on your JSON structure
                    actual_parent_id = data.get('id') or data.get('preview', {}).get('id')
                    print(f"✓ Parsed ID from JSON file: {actual_parent_id}")
                else:
                    # If it's just a text file, read the raw string
                    actual_parent_id = f.read().strip()
                    print(f"✓ Read ID from text file: {actual_parent_id}")
        except Exception as e:
            print(f"  [WARN] Failed to parse file {artifact_file_path}: {e}")
    else:
        actual_parent_id = "12ce84d486cf4ff494010dc2a7f48e7f"
        print(f"⚠️ Directory/File not found. Using hardcoded fallback ID: {actual_parent_id}")
    preprocess_task = Task.get_task(task_id=actual_parent_id) 

    # Re-using the manifest logic we built
    manifest_artifact = preprocess_task.artifacts.get('manifest')
    manifest_local_path = manifest_artifact.get_local_copy()
    print(f"📜 Manifest downloaded to: {manifest_local_path}")
    # Since data is local, we just need the directory string
    # We can fetch this from the parent's arguments
    preprocessed_dir = preprocess_task.get_parameters_as_dict().get('Args/output_dir', './data/preprocessed/')

    if not checkpoint_path:
        print("❌ Error: Could not retrieve model weights.")
        return

    # 3. Load Dataset
    print(f"📂 Loading test samples from {preprocessed_dir}...")
    _, val_ds, _, _, _ = create_dataset_pipeline(
        preprocessed_dir=preprocessed_dir, 
        batch_size=args.batch_size, 
        val_split=0.2, 
        seed=42,
        manifest_path=manifest_local_path 
    )

    # 4. Load Model
    print("🤖 Initializing LipReadingCTC...")
    model = LipReadingCTC(num_chars=NUM_CHARS)
    _ = model(np.random.randn(1, 75, 80, 120, 1).astype(np.float32)) # Dummy build
    model.load_weights(checkpoint_path)

    # 5. Evaluation Loop
    total_wer, total_cer, num_samples = 0.0, 0.0, 0
    num_steps = args.test_samples // args.batch_size
    
    logger = task.get_logger()

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

    # Report to ClearML Scalars (so you see them in the UI graphs)
    logger.report_scalar("Final Metrics", "WER", avg_wer, iteration=0)
    logger.report_scalar("Final Metrics", "CER", avg_cer, iteration=0)

    logger.report_text(f"Final Eval: WER {avg_wer:.2%}, CER {avg_cer:.2%}")
    print(f"\n✅ Evaluation Complete. Avg WER: {avg_wer:.4f}")
    report_dir = "./reports/eval_result"
    os.makedirs(report_dir, exist_ok=True)
    
    report_filename = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(report_dir, report_filename)

    with open(report_path, "w") as f:
        f.write(f"--- LipReading AI Evaluation Report ---\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model Task ID: {args.train_task_id}\n")
        f.write(f"Preprocess Task ID: {args.preprocess_task_id}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Total Samples Evaluated: {num_samples}\n")
        f.write(f"Average Word Error Rate: {avg_wer:.4f} ({avg_wer:.1%})\n")
        f.write(f"Average Char Error Rate: {avg_cer:.4f} ({avg_cer:.1%})\n")
        f.write(f"{'='*40}\n")
        f.write("\nNote: This report was generated automatically by the ImConvo Pipeline.")

    # --- PART C: The Bridge (Upload the report back to ClearML) ---
    # This makes the local file visible in the "Artifacts" tab of the UI!
    task.upload_artifact(name="final_eval_report", artifact_object=report_path)

    print(f"\n✓ Evaluation Complete.")
    print(f"📊 Live results: Check ClearML 'Scalars' tab.")
    print(f"📄 Local report: {report_path}")

    # 7. Visualization (The Loss Plot)
    print("📈 Generating training loss plots...")
    from src.visualization import save_loss_plot
    
    history_path = "./checkpoints/training_history.json"
        
    # 2. Run your original visualization function
    # This usually saves a .png or .jpg
    plot_output_dir = "./reports/plots"
    plot_output_path = "./reports/plots/training_summary.png"
    save_loss_plot(history_path, output_dir=plot_output_dir)

    # 3. Upload the Plot to ClearML "Plots" tab
    # This lets you see the graph directly in the browser!
    task.get_logger().report_image(
        title="Training Progress", 
        series="Loss and Accuracy", 
        iteration=0, 
        local_path=plot_output_path
    )
    print(f"✓ Loss plot uploaded to ClearML and saved to {plot_output_path}")
    task.close()
if __name__ == "__main__":
    main()
