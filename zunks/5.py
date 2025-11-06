from flask import Flask, request, jsonify
from ultralytics import YOLO
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

app = Flask(__name__)

# === Load YOLO model (gunakan model yang sudah mendukung deteksi wajah) ===
print("[INFO] Loading YOLO model...")
yolo_model = YOLO("yolov8n-face.pt")  # pastikan model ini ada di folder yang sama
print("[INFO] YOLO loaded successfully.")

# === Load known faces ===
known_encodings = []
known_names = []
DATASET_DIR = "known_faces"

print("[INFO] Loading known faces...")
for name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
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

    # Resize jika gambar terlalu besar
    h, w = frame.shape[:2]
    if w > 1600:
        scale = 1600 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        print(f"[INFO] Resized frame to {frame.shape[:2]}")

    # === 1️⃣ Deteksi wajah dengan YOLO ===
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    faces_detected = 0
    best_name = "unknown"
    best_confidence = 0.0

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        faces_detected += 1

        # === 🧠 Tambahkan tahap normalisasi sebelum encoding ===
        # 1️⃣ Konversi ke RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # 2️⃣ Resize ke ukuran standar (160x160)
        normalized_crop = cv2.resize(rgb_crop, (160, 160))

        # 3️⃣ Normalisasi nilai piksel ke [0,1]
        normalized_crop = normalized_crop.astype("float32") / 255.0

        # === 2️⃣ Ekstraksi embedding dengan face_recognition ===
        # Catatan: library face_recognition menggunakan nilai RGB tanpa normalisasi internal,
        # jadi kita konversi balik ke uint8 (0–255) agar tetap kompatibel.
        normalized_crop_uint8 = (normalized_crop * 255).astype(np.uint8)

        encodings = face_recognition.face_encodings(normalized_crop_uint8)
        if len(encodings) == 0:
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

        # Simpan hasil deteksi (debug)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, best_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # === 3️⃣ Evaluasi hasil ===
    if faces_detected == 0:
        result = {"action": "deny", "reason": "no_face_detected"}
    elif best_confidence > 0.85:
        result = {"action": "allow", "name": best_name, "confidence": best_confidence}
    else:
        result = {"action": "deny", "confidence": best_confidence}

    # === 4️⃣ Logging hasil ===
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    cv2.imwrite(f"logs/{ts}_result.jpg", frame)
    print(f"[{ts}] Result: {result}")

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8123)
