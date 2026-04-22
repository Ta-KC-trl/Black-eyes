# 👁 Black Eyes — Surveillance & Anomaly Detection

Real-time surveillance system combining **face recognition** and **anomaly detection** (fire, smoke, knives, guns) in a cyberpunk-themed web dashboard. Powered by a Flask API backend with dlib 128-d embeddings and a custom YOLOv11 model.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask)
![YOLOv11](https://img.shields.io/badge/YOLOv11-ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live Camera Feed** | Face recognition + anomaly detection run in parallel on every frame |
| **Face Recognition** | dlib HOG detector + 128-d embeddings via `face_recognition` |
| **Anomaly Detection** | Custom YOLOv11 model detecting fire, smoke, knives, and guns |
| **Identity Registry** | Register, view, and delete subjects entirely through the browser UI |
| **Webcam Registration** | Capture a face directly from webcam to register a new subject |
| **Image Upload Mode** | Scan static images for both faces and anomalies |
| **Smart DB Caching** | Face database cached in memory, reloads only when file changes |
| **Floating Pill Nav** | Smooth animated bottom navigation bar (Tracking / Database / Management) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Webcam (for live tracking)
- CMake + C++ build tools (required by `dlib`)
- Git LFS (for model weights)

### 1 — Clone

```bash
git lfs install          # only needed once
git clone https://github.com/Ta-KC-trl/Black-eyes.git
cd Black-eyes
```

### 2 — Set up environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows / dlib note:**  
> You need CMake and Visual C++ Build Tools installed before `dlib` can build.  
> Run `pip install cmake` first if the build fails, or install `dlib-bin` as a pre-built wheel:
> ```bash
> pip install dlib-bin
> pip install face_recognition
> ```

### 4 — Run

```bash
python start.py
```

This single command:
- Starts the **Flask API** on `http://localhost:5000`
- Starts the **website** on `http://localhost:8080`
- Opens the browser automatically

---

## 📁 Project Structure

```
Black-eyes/
├── start.py                 # One-command launcher (API + website)
├── api.py                   # Flask backend — face recognition + YOLO detection
├── utils.py                 # Face database helpers (register, delete, cache)
├── config.yaml              # Central config (model paths, detection thresholds)
├── requirements.txt         # Python dependencies
├── docs/
│   └── index.html           # Web dashboard (3D hero, Tracking, Database, Management)
├── models/
│   └── best.pt              # Custom YOLOv11 weights (via Git LFS)
├── scripts/
│   └── prepare_and_train.py # Split dataset + fine-tune YOLO on your own data
├── dataset/
│   └── data.yaml            # Dataset class definitions for training
├── Sample_images/           # Demo images for upload mode
├── Tracking.py              # Legacy Streamlit app (kept for reference)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🖥 Pages

### TRACKING
Live surveillance via webcam or uploaded image.
- Green box = recognised face (name shown)
- Red box = unknown face
- Orange box = weapon / fire / smoke detected
- Status cards on the right update in real time

### DATABASE
Browse all registered subjects — photo, name, and ID.

### MANAGEMENT
- **Add Personnel** — upload a photo or capture from webcam → face encoding stored automatically
- **Remove Personnel** — delete a subject by ID

---

## ⚙ Configuration

Edit `config.yaml` to change paths and thresholds:

```yaml
YOLO:
  MODEL_PATH: "models/best.pt"
  BASE_MODEL: "yolo11n.pt"

DETECTION:
  FACE_TOLERANCE: 0.4      # Lower = stricter face match
  ANOMALY_TOLERANCE: 0.5   # YOLO confidence threshold
```

Sliders in the dashboard also let you adjust both values live without restarting.

---

## 🏋 Train Your Own Model

```bash
# Place labelled images in data/knife_openimages/ (or any YOLO-format dataset)
python scripts/prepare_and_train.py
```

Trains for up to 80 epochs with early stopping, freezes the first 10 layers to preserve existing class knowledge (fire/smoke/gun), and saves the best weights to `models/best.pt`.

---

## 🛡 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Flask 3 + Flask-CORS |
| Face Detection | `face_recognition` (dlib HOG, 50% downscale) |
| Face Matching | 128-d Euclidean distance embeddings |
| Anomaly Detection | YOLOv11 (Ultralytics) — custom-trained |
| Frontend | Vanilla HTML/CSS/JS + Three.js hero |
| Database | Pickle store with mtime-based memory cache |
| Parallelism | `concurrent.futures.ThreadPoolExecutor` |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

- **Ta-KC-trl** — [GitHub](https://github.com/Ta-KC-trl)

---

> *"Vigilance through intelligence."*
