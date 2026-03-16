"""
utils.py — Face detection, recognition, and database operations.
Uses OpenCV Haar cascades + cosine-similarity matching.
"""

import pickle as pkl
import os
import cv2
import numpy as np
import yaml

_BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_BASE, "config.yaml"), "r") as _f:
    _cfg = yaml.safe_load(_f)

PKL_PATH    = os.path.join(_BASE, _cfg["PATH"]["PKL_PATH"])
DATASET_DIR = os.path.join(_BASE, _cfg["PATH"]["DATASET_DIR"])

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

def _detect_faces(gray_img):
    faces = _face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [] if len(faces) == 0 else faces

def _get_embedding(rgb_img, x, y, w, h):
    face = cv2.resize(rgb_img[y:y+h, x:x+w], (64, 64))
    return face.astype(np.float32).flatten() / 255.0

def _cosine_sim(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm != 0 else 0.0

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

def isFaceExists(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return len(_detect_faces(gray)) > 0

def recognize(image, tolerance=0.5):
    database = get_database()
    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = _detect_faces(gray)
    name  = person_id = "Unknown"
    sim_threshold = 1.0 - (tolerance * 0.5)

    for (x, y, w, h) in faces:
        embedding = _get_embedding(image, x, y, w, h)
        name = person_id = "Unknown"
        best_sim = -1

        for idx, person in database.items():
            enc = person.get("encoding")
            if enc is None:
                continue
            sim = _cosine_sim(embedding, enc)
            if sim > best_sim:
                best_sim = sim
                if sim >= sim_threshold:
                    name      = person["name"]
                    person_id = person["id"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if name != "Unknown":
            cv2.putText(image, f"{best_sim:.2f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image, name, person_id

def submitNew(name, person_id, image, old_idx=None):
    database = get_database()
    if not isinstance(image, np.ndarray):
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _detect_faces(gray)
    if len(faces) == 0:
        return -1
    x, y, w, h  = faces[0]
    encoding     = _get_embedding(rgb, x, y, w, h)
    existing_ids = [database[i]["id"] for i in database]
    if old_idx is not None:
        idx = old_idx
    else:
        if person_id in existing_ids:
            return 0
        idx = len(database)
    database[idx] = {"image": rgb, "id": person_id, "name": name, "encoding": encoding}
    _save_database(database)
    return True

def get_info_from_id(person_id):
    database = get_database()
    for idx, person in database.items():
        if str(person["id"]) == str(person_id):
            return person["name"], person["image"], idx
    return None, None, None

def deleteOne(person_id):
    database = get_database()
    for key, person in list(database.items()):
        if str(person["id"]) == str(person_id):
            del database[key]
            _save_database(database)
            return True
    return False

def build_dataset():
    info = {}
    counter = 0
    for fname in os.listdir(DATASET_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(DATASET_DIR, fname)
        parts = fname.rsplit(".", 1)[0].split("_")
        pid   = parts[0]
        pname = " ".join(parts[1:])
        img   = cv2.imread(fpath)
        if img is None:
            continue
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = _detect_faces(gray)
        if len(faces) == 0:
            print(f"No face found in {fname}, skipping.")
            continue
        x, y, w, h = faces[0]
        info[counter] = {"image": rgb, "id": pid, "name": pname, "encoding": _get_embedding(rgb, x, y, w, h)}
        counter += 1
    _save_database(info)
    print(f"Saved {counter} entries.")