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
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Resize kecil untuk kecepatan
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(face_encodings) == 0:
        return jsonify({"action": "deny", "reason": "no_face_detected"})

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

    # Logging sederhana
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("logs", exist_ok=True)
    cv2.imwrite(f"logs/{ts}.jpg", frame)
    print(f"[{ts}] Result: {result}")

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
