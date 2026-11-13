
import cv2
import json
import urllib.request
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# -------------------------------
# PROJECT PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "known_faces" / "face_db.json"

MODEL_DIR.mkdir(exist_ok=True)
DB_PATH.parent.mkdir(exist_ok=True)

YOLO_MODEL_PATH = MODEL_DIR / "yolov8n-face.pt"

YOLO_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/"
    "releases/download/1.0.0/yolov8n-face.pt"
)


# -------------------------------
# SETTINGS
# -------------------------------

SIMILARITY_THRESHOLD = 0.40


# -------------------------------
# DOWNLOAD YOLO MODEL
# -------------------------------

if not YOLO_MODEL_PATH.exists():

    print("Downloading YOLO face model...")

    urllib.request.urlretrieve(
        YOLO_MODEL_URL,
        YOLO_MODEL_PATH
    )

    print("YOLO model downloaded!")


# -------------------------------
# LOAD YOLO
# -------------------------------

print("Loading YOLO...")

yolo = YOLO(str(YOLO_MODEL_PATH))


# -------------------------------
# LOAD INSIGHTFACE
# -------------------------------

print("Loading face recognition model...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("Models loaded successfully!")


# -------------------------------
# FACE DETECTION
# -------------------------------

def detect_faces(image, image_size=640):

    results = yolo.predict(
        image,
        conf=0.25,
        imgsz=image_size,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None:
        return []

    return boxes.xyxy.cpu().numpy().astype(int)


# -------------------------------
# EXTRACT FACE EMBEDDING
# -------------------------------

def extract_embedding(image, box):

    x1, y1, x2, y2 = box

    height, width = image.shape[:2]

    # Add padding around the face
    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.25)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # InsightFace extracts the facial embedding
    faces = face_app.get(face_crop)

    if not faces:
        return None

    # Select the largest detected face in the crop
    face = max(
        faces,
        key=lambda f: (
            f.bbox[2] - f.bbox[0]
        ) * (
            f.bbox[3] - f.bbox[1]
        )
    )

    return face.normed_embedding


# -------------------------------
# LOAD DATABASE
# -------------------------------

def load_database():

    if not DB_PATH.exists():
        return {}

    with open(DB_PATH, "r") as file:
        return json.load(file)


# -------------------------------
# SAVE DATABASE
# -------------------------------

def save_database(database):

    with open(DB_PATH, "w") as file:

        json.dump(
            database,
            file,
            indent=4
        )


# -------------------------------
# MATCH FACE
# -------------------------------


def recognize_face(embedding, database=None):

    if database is None:
        database = load_database()

    best_name = "Unknown"
    best_score = -1

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    for name, stored_embeddings in database.items():

        for stored in stored_embeddings:

            stored = np.asarray(
                stored,
                dtype=np.float32
            )

            if stored.shape != embedding.shape:
                continue

            score = float(np.dot(embedding, stored))

            if score > best_score:
                best_score = score
                best_name = name

    if best_score < SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score
