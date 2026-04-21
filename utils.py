"""
utils.py — Robust face detection, recognition, and database operations.
Falls back to OpenCV Haar Cascades if face_recognition (dlib) is not installed.
"""

import pickle as pkl
import os
import cv2
import numpy as np
import yaml

# ── Face Recognition Imports & Availability ────────────────────────────────────
try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except ImportError:
    FACE_RECOG_AVAILABLE = False

# ── Paths ───────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_BASE, "config.yaml")

if not os.path.exists(_CONFIG_PATH):
    _cfg = {
        "PATH": {"PKL_PATH": "dataset/database.pkl", "DATASET_DIR": "dataset/"},
        "YOLO": {"MODEL_PATH": "models/best.pt"},
        "DETECTION": {"ANOMALY_TOLERANCE": 0.6, "FACE_TOLERANCE": 0.4}
    }
else:
    with open(_CONFIG_PATH, "r") as _f:
        _cfg = yaml.safe_load(_f)

PKL_PATH    = os.path.join(_BASE, _cfg["PATH"].get("PKL_PATH", "dataset/database.pkl"))
DATASET_DIR = os.path.join(_BASE, _cfg["PATH"].get("DATASET_DIR", "dataset/"))

# ── Fallback Detector ────────────────────────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

def get_database():
    if not os.path.exists(PKL_PATH):
        return {}
    with open(PKL_PATH, "rb") as f:
        try:
            return pkl.load(f)
        except Exception:
            return {}

def _save_database(db):
    os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
    with open(PKL_PATH, "wb") as f:
        pkl.dump(db, f)

# ── Detection & Recognition Logic ───────────────────────────────────────────

def isFaceExists(image):
    if FACE_RECOG_AVAILABLE:
        return len(face_recognition.face_locations(image)) > 0
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5)
        return len(faces) > 0

# Module-level DB cache: (mtime, database_dict)
_db_cache: tuple = (None, {})

def get_database_cached():
    """Return the database, reloading from disk only when the file changes."""
    global _db_cache
    if not os.path.exists(PKL_PATH):
        return {}
    mtime = os.path.getmtime(PKL_PATH)
    if _db_cache[0] == mtime:
        return _db_cache[1]
    db = get_database()
    _db_cache = (mtime, db)
    return db

def recognize(image, tolerance=0.5):
    database = get_database_cached()
    # name/person_id reflect the PRIMARY (first) detected face for UI display
    name = person_id = "Unknown"

    if FACE_RECOG_AVAILABLE:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        known_encodings = []
        known_metadata  = []
        for p in database.values():
            enc = p.get("encoding")
            if enc is not None and len(enc) == 128:
                known_encodings.append(enc)
                known_metadata.append((p["name"], p["id"]))

        first_face = True
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_name = face_pid = "Unknown"
            best_sim = 0.0

            if known_encodings:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_idx = np.argmin(face_distances)
                best_sim = 1.0 - face_distances[best_idx]
                if face_distances[best_idx] <= tolerance:
                    face_name, face_pid = known_metadata[best_idx]

            if first_face:
                name, person_id = face_name, face_pid
                first_face = False

            color = (0, 255, 0) if face_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(image, face_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if face_name != "Unknown":
                cv2.putText(image, f"Match: {best_sim:.2f}", (left + 6, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5)
        sim_threshold = 1.0 - (tolerance * 0.5)
        first_face = True

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(image[y:y+h, x:x+w], (64, 64))
            embedding = face_roi.astype(np.float32).flatten() / 255.0

            face_name = face_pid = "Unknown"
            best_sim = -1
            for person in database.values():
                enc = person.get("encoding")
                if enc is None or len(enc) != len(embedding):
                    continue
                norm = np.linalg.norm(embedding) * np.linalg.norm(enc)
                sim = float(np.dot(embedding, enc) / norm) if norm != 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    if sim >= sim_threshold:
                        face_name, face_pid = person["name"], person["id"]

            if first_face:
                name, person_id = face_name, face_pid
                first_face = False

            color = (0, 255, 0) if face_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, face_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return image, name, person_id

def submitNew(name, person_id, image, old_idx=None):
    database = get_database()
    if not isinstance(image, np.ndarray):
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if FACE_RECOG_AVAILABLE:
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0: return -1
        encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])[0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0: return -1
        x, y, w, h = faces[0]
        face_roi = cv2.resize(image[y:y+h, x:x+w], (64, 64))
        encoding = face_roi.astype(np.float32).flatten() / 255.0

    existing_ids = [database[i]["id"] for i in database]
    idx = old_idx if old_idx is not None else len(database)
    while old_idx is None and idx in database: idx += 1
    if old_idx is None and person_id in existing_ids: return 0
            
    database[idx] = {"image": image, "id": person_id, "name": name, "encoding": encoding}
    _save_database(database)
    return True

def get_info_from_id(person_id):
    database = get_database()
    for idx, p in database.items():
        if str(p["id"]) == str(person_id):
            return p["name"], p["image"], idx
    return None, None, None

def deleteOne(person_id):
    database = get_database()
    for key, p in list(database.items()):
        if str(p["id"]) == str(person_id):
            del database[key]; _save_database(database)
            return True
    return False