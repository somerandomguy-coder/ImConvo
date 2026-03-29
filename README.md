# ImConvo 👄
**Lip-reading AI developed by the Convobeo Team.**

ImConvo is a LipNet-based spatio-temporal model designed to interpret speech from visual lip movements. This repository includes a full pipeline for training, evaluating, and running real-time inference.

## 🚀 Quick Start (WSL2 / Linux)
The easiest way to set up the environment, dependencies, and data is to use the provided bash script:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

---

## 📸 Camera Setup Options

### Option A: Smartphone (Best for WSL2)
Since WSL2 cannot easily access internal webcams, use an **IP Webcam** app to stream via your local network.
1.  **Install App:** [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam) or iVCam (iOS).
2.  **Configure App (Crucial):**
    * **Resolution:** `352 x 288`
    * **Frame Rate:** `25 FPS` (The model is strictly temporal; wrong FPS = wrong results).
3.  **Run:** `python inference.py --ip http://192.168.0.69:8080`

### Option B: Direct USB/Integrated Camera
If you are running on Native Windows or Linux:
1.  Identify your camera index (usually `0` or `1`).
2.  **Run:** `python inference.py --ip 0`

---

## 📊 Project Structure & Reporting
The project is organized to keep your data and results separate:

* `data/`: Raw GRID dataset and preprocessed `.npy` files.
* `checkpoints/`: Holds `best_ctc_model.keras` and training history.
* `reports/`: 
    * `eval_results/`: Detailed text logs from `test.py` showing Word Error Rate (WER).
    * `plots/`: Visualizations of loss curves and performance summaries.
* `src/`: Core model architecture and utility functions.

---

## 🛠 Usage Modules

| Command | Description |
| :--- | :--- |
| `python scripts/preprocess.py` | Converts `.mpg` videos to normalized `.npy` sequences. |
| `python train.py` | Starts the CTC-based training loop. |
| `python test.py` | Runs evaluation on the test set and generates a report in `reports/`. |
| `python inference.py --ip <URL/ID>` | Launches the live monitor with the mouth bounding box. |

**Exit:** Press **'q'** while the video window is focused to stop any live process.

**Positioning:** Keep your lips within the yellow bounding box. A progress bar at the bottom will show you when the 75-frame buffer is full and ready for prediction.

