from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

app = Flask(__name__)

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
    img_data = file.read()

    # Decode gambar dari buffer
    img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # --- Debug: pastikan frame valid ---
    if frame is None:
        print("[ERROR] Failed to decode image. Check upload format.")
        debug_path = f"debug_frames/invalid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
        os.makedirs("debug_frames", exist_ok=True)
        with open(debug_path, "wb") as f:
            f.write(img_data)
        print(f"[DEBUG] Saved invalid image data to {debug_path}")
        return jsonify({"error": "invalid_image_data"}), 400
    else:
        print(f"[INFO] Frame shape: {frame.shape}")

    # Resize untuk kecepatan (bisa ubah fx=1.0 jika wajah tidak terdeteksi)
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = frame  # Gunakan ukuran asli untuk debugging
    rgb_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    print(f"[DEBUG] Found {len(face_locations)} face locations")

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    print(f"[DEBUG] Encoded {len(face_encodings)} faces")

    face_locations2 = face_recognition.face_locations(rgb_frame, model="cnn")
    if len(face_locations) == 0:
        print("[WARN] No faces detected. Saving debug image...")
        cv2.imwrite(f"debug_frames/{datetime.now().strftime('%Y%m%d_%H%M%S')}_noface.jpg", frame)
    else:
        for (top, right, bottom, left) in face_locations:
          cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.imwrite(f"debug_frames/{datetime.now().strftime('%Y%m%d_%H%M%S')}_detected.jpg", frame)

    # --- Debug: tidak ada wajah terdeteksi ---
    if len(face_encodings) == 0:
        print(f"[WARN] No faces detected. Locations found: {len(face_locations)}")

        os.makedirs("debug_frames", exist_ok=True)
        debug_path = f"debug_frames/{datetime.now().strftime('%Y%m%d_%H%M%S')}_noface.jpg"
        cv2.imwrite(debug_path, frame)
        print(f"[DEBUG] Saved no-face frame to {debug_path}")

        return jsonify({"action": "deny", "reason": "no_face_detected"})

    # === Matching ===
    best_match_name = "unknown"
    best_confidence = 0.0

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        confidence = 1 - face_distances[best_match_index]
        if matches[best_match_index] and confidence > best_confidence:
            best_confidence = float(confidence)
            best_match_name = known_names[best_match_index]

    if best_confidence > 0.85:
        result = {"action": "allow", "name": best_match_name, "confidence": best_confidence}
    else:
        result = {"action": "deny", "confidence": best_confidence}

    # === Logging hasil ===
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{ts}.jpg"
    cv2.imwrite(log_path, frame)
    print(f"[{ts}] Result: {result} | Saved frame -> {log_path}")

    return jsonify(result)

if __name__ == "__main__":
    os.makedirs("debug_frames", exist_ok=True)
    app.run(host="0.0.0.0", port=8123)
