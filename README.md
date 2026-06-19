 Fitness Action Recognition HPC Pipeline

A robust, end-to-end Deep Learning pipeline for temporal action recognition and kinematic tracking of fitness exercises. Designed to run on High-Performance Computing (HPC) clusters (e.g., SLURM), this project combines spatial pose estimation, signal processing, and recurrent neural networks to accurately classify exercises and count repetitions in real-time.

## 🎯 Supported Exercises
* **Squat**
* **Push-up**
* **Barbell Biceps Curl**
* **Shoulder Press**

## 🧠 Architecture Overview

The system is built on a highly modular, decoupled architecture designed for fault tolerance and temporal coherence:

1.  **Spatial Pose Estimation (Keypoint R-CNN):** Extracts 17 COCO keypoints per frame from raw video data using a pre-trained ResNet50-FPN backbone.
2.  **Signal Processing (Kalman Filter):** A 1D Kalman Filter smooths the raw coordinates, mitigating high-frequency jittering and occlusions. It extracts 8 *view-invariant* relative joint angles, making the model agnostic to camera placement.
3.  **Temporal Classification (BiLSTM):** A Bidirectional LSTM network ingests sequences of 30 frames (angles only) to classify the ongoing exercise, leveraging contextual temporal information from both past and future states.
4.  **Finite State Machine & Tracking (FSM):** * **Temporal Smoother:** Applies confidence-weighted soft-voting to stabilize raw neural network predictions.
    * **Mutex Lock:** Prevents erratic class-switching during an active repetition using a frame-based watchdog timer.
    * **Kinematic Trackers:** Evaluates the Range of Motion (ROM) on specific joints (Flexion/Extension tracking) to increment the repetition counter only upon complete, physiologically valid movements.

## ✨ Key Features

* **HPC/SLURM Native:** CLI-driven execution (`extract`, `train`, `inference`, `diagnostics`) with automated staging, batch processing, and artifact synchronization.
* **Zero Data Leakage:** Implements a rigorous grouping split algorithm that strictly stratifies domains (organic vs. synthetic) and completely isolates video frames between Training and Validation sets.
* **Deterministic Execution:** Seed-locked environment for full reproducibility across distributed nodes.
* **Advanced Diagnostics Engine:** Automatically generates analytical artifacts during training, including:
    * Cross-Entropy Loss & Accuracy curves.
    * Precision-Recall (PR) Curves per class.
    * Kalman Filter signal attenuation plots.
    * Temporal Stability plots (FSM Mutex Override vs Raw Softmax).
* **Dynamic UI Renderer:** Resolution-independent OpenCV Heads-Up Display (HUD) overlaying the skeleton, bounding box, repetition counts, and concentric/eccentric states.

## 🚀 Installation & Requirements

### Dependencies
* Python 3.8+
* PyTorch & Torchvision (CUDA enabled recommended)
* OpenCV (`cv2`)
* Scikit-Learn
* Matplotlib
* NumPy

### Environment Setup
Create a virtual environment and install the required packages:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python scikit-learn matplotlib numpy
```

*Note: Ensure the Keypoint R-CNN weights (`keypointrcnn_resnet50_fpn_coco-fc266e95.pth`) are downloaded and placed in the `models/` directory for offline execution.*

## 🛠️ Usage (CLI Interface)

The pipeline is driven by a unified entry point (`main.py`) configurable via command-line arguments.

**1. Run Diagnostics (Unit Tests)**
Validates geometry matrices, FSM logic, Kalman tuning, and split integrity without requiring a GPU.
```bash
python main.py --task diagnostics
```

**2. Feature Extraction**
Parses raw videos (organic and synthetic), applies the R-CNN and Kalman Filter, and saves sequences as `.pt` tensors.
```bash
python main.py --task extract --feature-tag "size8_seq30_stride10" --batch-size 16
```

**3. Model Training**
Trains the BiLSTM on the extracted features. Handles early stopping, LR scheduling, and artifact generation.
```bash
python main.py --task train --feature-tag "size8_seq30_stride10" --model-tag "bilstm_v1" --lstm-batch-size 16
```

**4. Production Inference**
Runs the full pipeline (R-CNN $\rightarrow$ BiLSTM $\rightarrow$ FSM) on new videos, exporting rendered MP4s with the overlaid HUD and a CSV summary report.
```bash
python main.py --task inference --model-tag "bilstm_v1"
```

## 📊 Metrics & Performance

The model relies on a Hard-Voting aggregation logic at the video level during evaluation.
* **Metrics Tracked:** Macro F1-Score, Average Precision (AP), and Video-Level Accuracy.
* **Tolerance:** The FSM `ROM_THRESHOLD` is strictly set to 35.0 degrees to filter out micro-movements and false positives, ensuring that only full repetitions are logged.

## 🙏 Acknowledgements

Special thanks to [Riccardo Riccio][link] for providing the open-source real-time exercise recognition dataset on Kaggle, which served as the foundational data for this project.

[link]:https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset
---
*Developed for Computer Vision and Deep Learning academic research.*