from flask import Flask, request, jsonify
from ultralytics import YOLO
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

app = Flask(__name__)

# === Load YOLO model ===
print("[INFO] Loading YOLO model...")
yolo_model = YOLO("yolov8n-face.pt")  # gunakan model wajah YOLO
print("[INFO] YOLO loaded successfully.")

# === Direktori utama ===
DATASET_DIR = "known_faces"
NORMALIZED_DIR = "normalized"
LOG_DIR = "logs"

os.makedirs(os.path.join(NORMALIZED_DIR, "known_faces"), exist_ok=True)
os.makedirs(os.path.join(NORMALIZED_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(NORMALIZED_DIR, "known_faces_cropped"), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Load known faces (disesuaikan dengan pipeline YOLO) ===
known_encodings = []
known_names = []
print("[INFO] Loading known faces...")

MAX_HEIGHT = 1024  # Resize known_face jika terlalu besar

for name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    os.makedirs(os.path.join(NORMALIZED_DIR, "known_faces", name), exist_ok=True)
    os.makedirs(os.path.join(NORMALIZED_DIR, "known_faces_cropped", name), exist_ok=True)

    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(img_path)  # sudah RGB
        except Exception as e:
            print(f"[ERROR] Gagal load {img_path}: {e}")
            continue

        # --- Resize jika gambar terlalu besar ---
        h, w = image.shape[:2]
        if h > MAX_HEIGHT:
            scale = MAX_HEIGHT / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

        # --- Gunakan YOLO untuk mendeteksi wajah ---
        results = yolo_model(image)  # preprocessing otomatis
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if len(boxes) == 0:
            print(f"[WARN] No face detected in {img_path}, skipped.")
            continue

        # Ambil wajah pertama, tambahkan padding agar landmark face_recognition lebih stabil
        x1, y1, x2, y2 = boxes[0][:4].astype(int)
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        face_crop = np.ascontiguousarray(face_crop.astype(np.uint8))  # pastikan contiguous
        rgb_crop = face_crop  # sudah RGB dari load_image_file

        # Simpan crop untuk debug
        crop_path = os.path.join(NORMALIZED_DIR, "known_faces_cropped", name, filename)
        cv2.imwrite(crop_path, cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))

        # Encoding wajah
        encodings = face_recognition.face_encodings(rgb_crop)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"[WARN] Failed to encode {img_path}")

print(f"[INFO] Loaded {len(known_names)} known faces.")


# === Endpoint detect ===
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # --- YOLO otomatis handle preprocessing ---
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    print(f"[DEBUG] YOLO mendeteksi {len(boxes)} wajah.")

    if len(boxes) == 0:
        return jsonify({"action": "deny", "reason": "no_face_detected"})

    # === Bandingkan dengan known faces ===
    best_name = "unknown"
    best_confidence = 0.0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        # scale = 2.5
        scale = 1.3
        new_w = int(w * scale)
        new_h = int(h * scale)
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(frame.shape[1], cx + new_w // 2)
        y2 = min(frame.shape[0], cy + new_h // 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rgb_crop = np.ascontiguousarray(crop[:, :, ::-1].astype(np.uint8))  # RGB & contiguous

        # --- Debug save crop untuk dicek ---
        cv2.imwrite("debug_upload_crop.jpg", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))

        # --- Resize jika terlalu besar ---
        # MAX_CROP_HEIGHT = 1024
        MAX_CROP_HEIGHT = 384  # bisa dicoba 256 jika masih gagal encoding
        ch, cw = rgb_crop.shape[:2]
        if ch > MAX_CROP_HEIGHT:
            scale = MAX_CROP_HEIGHT / ch
            new_w = int(cw * scale)
            rgb_crop = cv2.resize(rgb_crop, (new_w, MAX_CROP_HEIGHT))
            cv2.imwrite("debug_upload_crop_shape.jpg", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))

        print(f"[DEBUG] Upload crop shape: {rgb_crop.shape}")

        encodings = face_recognition.face_encodings(rgb_crop)
        if len(encodings) == 0:
            print("[WARN] Failed to encode upload crop")
            continue
        
        print(f"[DEBUG] Found {len(encodings)} encodings in upload crop")

        encoding = encodings[0]
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            if matches[best_match_index] and confidence > best_confidence:
                best_confidence = float(confidence)
                best_name = known_names[best_match_index]

    if best_confidence > 0.6:
        result = {"action": "allow", "name": best_name, "confidence": best_confidence}
    else:
        result = {"action": "deny", "confidence": best_confidence}

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8123)
