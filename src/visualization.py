import matplotlib.pyplot as plt
import os
import json

def save_loss_plot(history_path, output_dir="reports/plots"):
    """Reads training_history.json and saves a Loss/Accuracy plot."""
    if not os.path.exists(history_path):
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # If you have WER/CER in history
    if 'avg_wer' in history:
        plt.subplot(1, 2, 2)
        plt.bar(['WER', 'CER'], [history['avg_wer'], history['avg_cer']], color=['blue', 'green'])
        plt.title('Final Error Rates')
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_summary.png"))
    plt.close()
    print(f"📊 Plot saved to {output_dir}/training_summary.png")