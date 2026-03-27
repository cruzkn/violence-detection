# Real-Time Violence Detection via Pose & Gesture Analysis in CCTV Surveillance Systems

> APU Master's Thesis Implementation | CT095-6-M-RMCE | March 2026  
> Student: **[Your Name]** | Supervisor: **[Supervisor Name]**

---

## 📦 Project Structure

```
violence-detection/
├── setup/
│   └── install_macos.sh          ← One-click macOS setup
├── data/
│   ├── download_datasets.py      ← Auto-download RWF-2000 + Hockey Fight
│   ├── raw/                      ← Raw video datasets (not committed to git)
│   └── processed/                ← Extracted skeleton .npy features
├── modules/
│   ├── 01_pose_estimation.py     ← YOLOv8-Pose skeleton extraction
│   ├── 02_lstm_classifier.py     ← Bi-LSTM violence classifier
│   ├── 03_transformer_classifier.py ← ST-Transformer (alternative)
│   └── 04_autoencoder.py         ← Anomaly detection + dual-stream scorer
├── train/
│   └── train_pipeline.py         ← Full end-to-end training pipeline
├── dashboard/
│   ├── app.py                    ← Flask + SocketIO real-time dashboard
│   └── templates/index.html      ← Officer alert web UI
├── demo/
│   └── run_demo.py               ← Live webcam/video demo
├── models/                       ← Saved checkpoints (not committed)
├── results/                      ← Evaluation outputs, plots
└── requirements.txt
```

---

## 🚀 Quick Start (macOS)

### 1. Clone and Setup
```bash
git clone https://github.com/cruzkn/violence-detection.git
cd violence-detection
bash setup/install_macos.sh
source venv/bin/activate
```

### 2. Download Datasets
```bash
# Option A: Hugging Face (auto-download RWF-2000)
python data/download_datasets.py --dataset rwf2000

# Option B: Hockey Fight via Kaggle (requires Kaggle API token)
python data/download_datasets.py --dataset hockey

# Verify what you have
python data/download_datasets.py --verify-only
```

> **RWF-2000 via Hugging Face:**  
> `git lfs install && git clone https://huggingface.co/datasets/DanJoshua/RWF-2000 data/raw/RWF-2000`

### 3. Run Full Training Pipeline
```bash
# Full pipeline: extract features → train LSTM → train AE → evaluate
python train/train_pipeline.py

# Skip feature extraction if already done
python train/train_pipeline.py --skip-extraction
```

### 4. Run Live Demo (Webcam)
```bash
python demo/run_demo.py                     # webcam
python demo/run_demo.py --source video.mp4  # video file
```

### 5. Start Alert Dashboard
```bash
python dashboard/app.py
# Open: http://localhost:5000
```

---

## 🧠 System Architecture

```
📷 CCTV Feed
    ↓
[M1] YOLOv8 Person Detection + Frame Segmentation (30-frame windows)
    ↓
[M2] YOLOv8-Pose Skeleton Extraction (17 keypoints × 3 = 51 features)
    ↓
[M3] Bi-LSTM / ST-Transformer Violence Classifier
    ↓  ↘
    ↓   [M4] Autoencoder Anomaly Score
    ↓  ↙
[Dual-Stream Fusion] α×classifier + (1-α)×AE_score
    ↓
[M5] Alert Engine → Officer Dashboard (Human-in-the-Loop)
```

---

## 📊 Datasets

| Dataset | Size | Download |
|---------|------|----------|
| RWF-2000 | 2,000 clips | [HuggingFace](https://huggingface.co/datasets/DanJoshua/RWF-2000) |
| Hockey Fight | 1,000 clips | [Kaggle](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes) |
| CCTV-Fights | 1,000 clips | [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) |

---

## 🎯 Target Performance

| Metric | Target | Status |
|--------|--------|--------|
| Accuracy | ≥ 90% | Training |
| F1-Score | ≥ 89% | Training |
| AUC-ROC | ≥ 0.92 | Training |
| Speed | ≥ 15 FPS | ✓ (CPU/MPS) |

---

## ⚙️ Module Commands Reference

```bash
# Extract pose features from a dataset split
python modules/01_pose_estimation.py --input data/raw/RWF-2000/train --output data/processed/train

# Live pose demo (webcam)
python modules/01_pose_estimation.py --demo

# Train Bi-LSTM
python modules/02_lstm_classifier.py --train --train-dir data/processed/train --val-dir data/processed/val

# Evaluate Bi-LSTM
python modules/02_lstm_classifier.py --eval --checkpoint models/lstm_best.pth

# Train Transformer (alternative)
python modules/03_transformer_classifier.py --train

# Train Autoencoder + calibrate threshold
python modules/04_autoencoder.py --train
python modules/04_autoencoder.py --threshold --val-dir data/processed/val

# Start dashboard
python dashboard/app.py
```

---

## 🔒 Ethics & Privacy

- **No facial recognition** — only anonymous skeleton coordinate vectors are processed
- **Data minimisation** — raw video is not stored beyond processing window
- **Human-in-the-loop** — all alerts require officer confirmation before action
- **PDPA Malaysia 2010** aligned — see policy framework document

---

## 📚 Key References

- Islam et al. (2023). Spatiotemporal Transformer for Violence Detection. *Pattern Recognition Letters*.
- Zhang et al. (2023). YOLOv8-Pose for edge surveillance. *Sensors*.
- Feng et al. (2022). Dual-stream violence detection. *Neurocomputing*.
- Cheng et al. (2021). RWF-2000 Dataset. *ICPR 2021*.

---

## 👨‍💻 Author

**[Your Full Name]** | [TP Number] | Asia Pacific University  
Supervisor: [Supervisor Name] | March 2026
