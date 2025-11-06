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

# === Load known faces ===
known_encodings = []
known_names = []
print("[INFO] Loading known faces...")

for name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    os.makedirs(os.path.join(NORMALIZED_DIR, "known_faces", name), exist_ok=True)
    os.makedirs(os.path.join(NORMALIZED_DIR, "known_faces_cropped", name), exist_ok=True)

    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(img_path)
        except Exception as e:
            print(f"[ERROR] Gagal load {img_path}: {e}")
            continue

        # --- Normalisasi ukuran dan warna ---
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (400, 400))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Simpan hasil normalisasi
        norm_path = os.path.join(NORMALIZED_DIR, "known_faces", name, filename)
        cv2.imwrite(norm_path, image)

        # Deteksi wajah dulu
        rgb_image = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_image, model="cnn")

        if len(face_locations) == 0:
            print(f"[WARN] No face found in {img_path}, skipped.")
            continue

        # Ambil wajah pertama (asumsi satu wajah per foto)
        top, right, bottom, left = face_locations[0]
        face_crop = image[top:bottom, left:right]
        cropped_path = os.path.join(NORMALIZED_DIR, "known_faces_cropped", name, filename)
        if face_crop.size > 0:
            cv2.imwrite(cropped_path, face_crop)

        try:
            encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations, num_jitters=1)
        except TypeError as e:
            print(f"[ERROR] Failed to encode {img_path}: {e}")
            continue

        # Encoding wajah
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)

print(f"[INFO] Loaded {len(known_names)} known faces.")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # === Normalisasi ukuran ===
    h, w = frame.shape[:2]
    if w > 1600:
        scale = 1600 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # === Normalisasi pencahayaan ===
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    # Simpan hasil normalisasi (debug)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"{NORMALIZED_DIR}/uploads", exist_ok=True)
    normalized_path = f"{NORMALIZED_DIR}/uploads/{ts}_normalized.jpg"
    # cv2.imwrite(normalized_path, frame)
    # print(f"[DEBUG] Saved normalized upload image to {normalized_path}")

    # === Convert ke RGB dan resize ke 640 px (sesuai test_yolo.py) ===
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, int(640 * h / w)))
    cv2.imwrite(normalized_path, frame_resized)
    print(f"[DEBUG] Saved normalized upload image to {normalized_path}")

    # === Deteksi wajah dengan YOLO ===
    results = yolo_model.predict(frame_resized)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    print(f"[DEBUG] YOLO mendeteksi {len(boxes)} wajah.")

    # Simpan hasil deteksi YOLO (debug)
    os.makedirs("logs", exist_ok=True)
    debug_yolo_path = f"logs/{ts}_yolo_result.jpg"
    debug_frame = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(debug_yolo_path, debug_frame)
    print(f"[DEBUG] YOLO detection saved to {debug_yolo_path}")

    if len(boxes) == 0:
        print("[WARN] No faces detected by YOLO.")
        return jsonify({"action": "deny", "reason": "no_face_detected2"})

    # === Bandingkan dengan wajah yang dikenal ===
    best_name = "unknown"
    best_confidence = 0.0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4]

        # 🔹 Perluas bounding box agar lebih mencakup seluruh wajah
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2

        scale = 2.5  # perbesar kotak wajah 2.5× (bisa ubah ke 2.0–3.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(frame.shape[1], cx + new_w // 2)
        y2 = min(frame.shape[0], cy + new_h // 2)

        # 🔹 Crop wajah yang sudah diperluas
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Simpan hasil crop (debug)
        crop_path = f"logs/{ts}_crop_{i}.jpg"
        cv2.imwrite(crop_path, crop)
        print(f"[DEBUG] Saved expanded face crop to {crop_path}")

        # 🔹 Konversi ke RGB untuk face_recognition
        rgb_crop = crop[:, :, ::-1]

        encodings = face_recognition.face_encodings(rgb_crop)
        if len(encodings) == 0:
            print(f"[WARN] Tidak ditemukan landmark pada crop {i}.")
            continue

        encoding = encodings[0]
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]
            if matches[best_match_index] and confidence > best_confidence:
                best_confidence = float(confidence)
                best_name = known_names[best_match_index]

    # === Evaluasi hasil ===
    if best_confidence > 0.85:
        result = {"action": "allow", "name": best_name, "confidence": best_confidence}
    else:
        result = {"action": "deny", "confidence": best_confidence}

    log_path = f"logs/{ts}_result.jpg"
    cv2.imwrite(log_path, frame)
    print(f"[{ts}] Result: {result}")
    print(f"[DEBUG] Saved final detection result to {log_path}")

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8123)
