"""
api.py — Black Eyes Flask backend
Runs face recognition (dlib) + YOLO knife/fire/gun detection on your GPU.
Start with: python api.py
Then open docs/index.html served from http://localhost:8080
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, os, cv2, numpy as np, yaml

app = Flask(__name__)
CORS(app)  # allow browser at any localhost port

# ── Config ────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE, "config.yaml")) as _f:
    _cfg = yaml.safe_load(_f)

FACE_TOLERANCE = _cfg["DETECTION"].get("FACE_TOLERANCE", 0.4)
ANOMALY_CONF   = _cfg["DETECTION"].get("ANOMALY_TOLERANCE", 0.5)

# ── YOLO ──────────────────────────────────────────────────────────────────────
from ultralytics import YOLO
_yolo_path = os.path.join(_BASE, _cfg["YOLO"]["MODEL_PATH"])
if not os.path.exists(_yolo_path):
    _yolo_path = _cfg["YOLO"]["BASE_MODEL"]
    print(f"[YOLO] custom model not found, falling back to {_yolo_path}")
yolo = YOLO(_yolo_path)
print(f"[YOLO] loaded — classes: {list(yolo.names.values())}")

# ── Face recognition ──────────────────────────────────────────────────────────
try:
    import face_recognition
    FACE_RECOG = True
    print("[Face] face_recognition (dlib) available")
except ImportError:
    FACE_RECOG = False
    print("[Face] face_recognition not found — face detection disabled")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def _to_b64(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "face_recognition": FACE_RECOG,
        "yolo_classes": list(yolo.names.values()),
    })

@app.route("/detect", methods=["POST"])
def detect():
    body = request.json
    frame_bgr = _decode(body["frame"])
    tolerance   = float(body.get("tolerance",   FACE_TOLERANCE))
    anomaly_conf = float(body.get("anomaly_conf", ANOMALY_CONF))
    h, w = frame_bgr.shape[:2]

    # ── Faces ─────────────────────────────────────────────────────────────────
    faces = []
    if FACE_RECOG:
        from utils import get_database_cached
        db = get_database_cached()

        known_enc, known_meta = [], []
        for p in db.values():
            enc = p.get("encoding")
            if enc is not None and len(enc) == 128:
                known_enc.append(np.array(enc))
                known_meta.append({"name": p["name"], "id": p["id"]})

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locs)

        for (top, right, bottom, left), enc in zip(locs, encs):
            name, conf = "Unknown", 0.0
            if known_enc:
                dists = face_recognition.face_distance(known_enc, enc)
                best  = int(np.argmin(dists))
                if dists[best] <= tolerance:
                    name = known_meta[best]["name"]
                    conf = round(float(1.0 - dists[best]), 2)
            faces.append({
                "box":   [left/w, top/h, right/w, bottom/h],
                "name":  name,
                "known": name != "Unknown",
                "conf":  conf,
            })

    # ── YOLO ──────────────────────────────────────────────────────────────────
    anomalies = []
    results = yolo(frame_bgr, conf=anomaly_conf, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        anomalies.append({
            "box":   [x1/w, y1/h, x2/w, y2/h],
            "label": yolo.names[int(box.cls[0])],
            "conf":  round(float(box.conf[0]), 2),
        })

    return jsonify({"faces": faces, "anomalies": anomalies})

@app.route("/db")
def get_db():
    from utils import get_database_cached
    db = get_database_cached()
    out = {}
    for k, p in db.items():
        if "image" in p and isinstance(p["image"], np.ndarray):
            photo = _to_b64(cv2.cvtColor(p["image"], cv2.COLOR_RGB2BGR))
        else:
            photo = p.get("photo", "")
        out[k] = {"name": p.get("name", ""), "id": p.get("id", k), "photo": photo}
    return jsonify(out)

@app.route("/register", methods=["POST"])
def register():
    body      = request.json
    name      = body.get("name", "").strip()
    person_id = body.get("id",   "").strip()
    photo_b64 = body.get("photo", "")
    if not name or not person_id or not photo_b64:
        return jsonify({"error": "Missing fields"}), 400

    frame_bgr = _decode(photo_b64)
    img_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    from utils import submitNew
    result = submitNew(name, person_id, img_rgb)
    if result == -1:
        return jsonify({"error": "No face detected — try a clearer photo"}), 400
    if result == 0:
        return jsonify({"error": "ID already exists"}), 400
    return jsonify({"success": True})

@app.route("/unregister/<person_id>", methods=["DELETE"])
def unregister(person_id):
    from utils import deleteOne
    if not deleteOne(person_id):
        return jsonify({"error": "ID not found"}), 404
    return jsonify({"success": True})

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🟢 Black Eyes API running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
