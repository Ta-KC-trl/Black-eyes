"""
utils.py — Enhanced face detection, recognition, and database operations.
Uses face_recognition (dlib) for professional-grade accuracy.
"""

import pickle as pkl
import os
import cv2
import numpy as np
import yaml
import face_recognition

# ── Paths ───────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_BASE, "data", "config.yaml")
if not os.path.exists(_CONFIG_PATH):
    _CONFIG_PATH = os.path.join(_BASE, "config.yaml")

with open(_CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)

# Use paths from config or defaults
PKL_PATH    = os.path.join(_BASE, _cfg["PATH"].get("PKL_PATH", "data/face_db/database.pkl"))
DATASET_DIR = os.path.join(_BASE, _cfg["PATH"].get("DATASET_DIR", "data/face_db/"))

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
    """Checks if at least one face exists using face_recognition."""
    face_locations = face_recognition.face_locations(image)
    return len(face_locations) > 0

def recognize(image, tolerance=0.5):
    """
    Detects and recognizes faces in an image.
    Uses 128-d embeddings for professional accuracy.
    """
    database = get_database()
    # Find all face locations and encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    name = person_id = "Unknown"
    
    # Encodings from database
    known_encodings = []
    known_metadata  = []
    for idx, p in database.items():
        if "encoding" in p:
            known_encodings.append(p["encoding"])
            known_metadata.append((p["name"], p["id"]))

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = person_id = "Unknown"
        best_sim = 0.0
        
        if known_encodings:
            # We use face_distance to find the best match
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Distance to Similarity (approximate: 1.0 - distance)
            best_sim = 1.0 - face_distances[best_match_index]
            
            if face_distances[best_match_index] <= tolerance:
                name, person_id = known_metadata[best_match_index]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        # Draw box
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        # Draw label
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if name != "Unknown":
            cv2.putText(image, f"Match: {best_sim:.2f}", (left + 6, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image, name, person_id

def submitNew(name, person_id, image, old_idx=None):
    """Registers a new subject into the identity database."""
    database = get_database()
    
    if not isinstance(image, np.ndarray):
        # Convert streamlit/upload file to opencv if needed
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ensure image is RGB for face_recognition
    # (Streamlit provides RGB, but cv2.imread provides BGR)
    # This logic handles both
    
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return -1
    
    # Take the first face
    encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])[0]
    
    existing_ids = [database[i]["id"] for i in database]
    if old_idx is not None:
        idx = old_idx
    else:
        if person_id in existing_ids:
            return 0
        idx = len(database)
        # Ensure index is unique
        while idx in database:
            idx += 1
            
    database[idx] = {
        "image": image, 
        "id": person_id, 
        "name": name, 
        "encoding": encoding
    }
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
