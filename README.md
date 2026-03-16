# 👁 Black Eyes — Surveillance & Anomaly Detection

A real-time surveillance system combining **face recognition** and **anomaly detection** (fire, smoke, weapons) in a single unified dashboard. Built with Streamlit, OpenCV, and YOLOv8.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red?logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live Camera Feed** | Real-time face recognition + anomaly detection running in parallel |
| **Face Recognition** | OpenCV Haar cascade with cosine-similarity matching |
| **Anomaly Detection** | Custom YOLOv8 model detecting fire, smoke, guns, and knives |
| **Identity Database** | Add, update, and delete registered subjects via the UI |
| **Professional UI** | Dark cyberpunk-themed dashboard with real-time status panels |
| **Image Upload** | Analyse static images for both faces and anomalies |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Webcam (for live tracking)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ta-KC-trl/Black-eyes.git
cd Black-eyes

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download Model Weights

The custom-trained YOLOv8 model is not included in the repo (too large for Git). 
Place your trained `best.pt` inside the `models/` directory:

```
models/
  └── best.pt
```

> If you don't have a custom model, the app will fall back to the default `yolov8s.pt`.

### Run

```bash
streamlit run Tracking.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Black-eyes/
├── Tracking.py          # Main Streamlit application (3 pages)
├── utils.py             # Face detection, recognition, database ops
├── config.yaml          # Configuration (model paths, thresholds)
├── requirements.txt     # Python dependencies
├── models/              # YOLO model weights (gitignored)
│   └── best.pt
├── dataset/             # Face database
│   ├── data.yaml        # Dataset class definitions
│   └── database.pkl     # Encoded face database (gitignored)
├── Sample_images/       # Sample anomaly images for demo
├── Background/          # UI assets
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🖥 Pages

### 1. TRACKING
Live surveillance feed with dual detection:
- **Face Recognition**: Identifies known subjects, flags unknowns
- **Anomaly Detection**: Detects fire 🔥, smoke 💨, knives 🔪, and guns ⚠

### 2. DATABASE
Browse all registered subjects with their photos, IDs, and authorisation status.

### 3. UPDATING
Manage the identity database:
- **Add Subject** — Upload photo or use webcam capture
- **Delete Subject** — Remove by ID
- **Update Subject** — Change name, ID, or photo

---

## ⚙ Configuration

Edit `config.yaml` to customise paths and thresholds:

```yaml
PATH:
  DATASET_DIR: "dataset/"
  PKL_PATH: "dataset/database.pkl"

YOLO:
  MODEL_PATH: "models/best.pt"

DETECTION:
  ANOMALY_TOLERANCE: 0.6
```

---

## 🛡 Tech Stack

- **Frontend**: Streamlit with custom CSS
- **Face Detection**: OpenCV Haar Cascades
- **Face Matching**: Cosine similarity on 64×64 embeddings
- **Anomaly Detection**: YOLOv8 (custom-trained on fire/smoke/gun/knife)
- **Database**: Pickle-based local store

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

- **Ta-KC-trl** — [GitHub](https://github.com/Ta-KC-trl)

---

> *"Vigilance through intelligence."*
