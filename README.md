# 👁 Black Eyes — Surveillance & Anomaly Detection

A real-time surveillance system combining **face recognition** and **anomaly detection** (fire, smoke, knives, guns) in a unified cyberpunk-themed dashboard. Built with Streamlit, OpenCV, YOLOv11, and `face_recognition` (dlib 128-d embeddings).

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![YOLOv11](https://img.shields.io/badge/YOLOv11-ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live Camera Feed** | Real-time face recognition + anomaly detection in parallel |
| **Face Recognition** | 128-d dlib embeddings via `face_recognition`, with OpenCV Haar cascade fallback |
| **Multi-Face Support** | Detects and labels multiple faces simultaneously per frame |
| **Anomaly Detection** | Custom YOLOv11 model detecting fire 🔥, smoke 💨, knives 🔪, and guns |
| **Identity Registry** | Register, browse, and delete subjects entirely through the UI |
| **Smart DB Caching** | Database is cached in memory and only reloads when the file changes |
| **Professional UI** | Dark cyberpunk dashboard with animated scanline overlay and live status panels |
| **Image Upload Mode** | Analyse static images for both faces and anomalies |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Webcam (for live tracking)
- CMake + C++ build tools (required by `dlib` — see below)

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

> **Note on dlib / face_recognition (Windows):**  
> You need CMake and Visual C++ Build Tools installed.  
> Run: `pip install cmake` then `pip install dlib` before `pip install face_recognition`.  
> If dlib fails to build, the system automatically falls back to OpenCV Haar cascades.

### Model Weights

The custom-trained YOLOv11 model is not included in the repo. Place your trained `best.pt` inside `models/`:

```
models/
  └── best.pt
```

Train your own with:

```bash
python scripts/train_v11.py --data dataset/data.yaml --epochs 100 --model yolo11n.pt
```

> If no custom model is found, the app falls back to `yolov8n.pt` automatically.

### Run

```bash
streamlit run Tracking.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Black-eyes/
├── Tracking.py          # Main Streamlit app (Tracking / Database / Management pages)
├── utils.py             # Face detection, recognition, and database operations
├── train.py             # Simple YOLOv8 training launcher
├── config.yaml          # Central config (model paths, detection thresholds)
├── requirements.txt     # Python dependencies
├── scripts/
│   └── train_v11.py     # YOLOv11 training with argparse CLI
├── models/              # YOLO weights — gitignored, place best.pt here
├── dataset/
│   ├── data.yaml        # Dataset class definitions for training
│   └── database.pkl     # Face encoding store — gitignored (auto-created at runtime)
├── Sample_images/       # Demo images for the upload mode
├── Background/          # UI background assets
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🖥 Pages

### 1. TRACKING
Live surveillance with dual-mode detection:
- **Face Recognition** — Identifies known subjects by name, flags unknowns in red
- **Anomaly Detection** — Detects fire, smoke, knives, and guns via YOLOv11

Supports both live webcam and static image upload.

### 2. DATABASE
Browse all registered subjects — photo, name, and UUID displayed in a card layout.

### 3. MANAGEMENT
- **Add Personnel** — Upload a photo to register a new subject (face encoding stored automatically)
- **Remove Personnel** — Delete a subject by their ID

---

## ⚙ Configuration

Edit `config.yaml` to customise paths and thresholds:

```yaml
PATH:
  DATASET_DIR: "dataset/"
  PKL_PATH: "dataset/database.pkl"

YOLO:
  MODEL_PATH: "models/best.pt"
  BASE_MODEL: "yolo11n.pt"

DETECTION:
  FACE_TOLERANCE: 0.4      # Lower = stricter face match
  ANOMALY_TOLERANCE: 0.5   # YOLO confidence threshold
```

---

## 🛡 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit + custom CSS (cyberpunk theme) |
| Face Detection | `face_recognition` (dlib HOG/CNN) + OpenCV Haar cascade fallback |
| Face Matching | 128-d Euclidean distance embeddings |
| Anomaly Detection | YOLOv11 (Ultralytics) — custom-trained on fire/smoke/knife/gun |
| Object Tracking | Ultralytics YOLO inference pipeline |
| Database | Pickle-based local store with mtime-based memory cache |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

- **Ta-KC-trl** — [GitHub](https://github.com/Ta-KC-trl)

---

> *"Vigilance through intelligence."*
