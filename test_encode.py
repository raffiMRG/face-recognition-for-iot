from ultralytics import YOLO
import cv2
import numpy as np
import face_recognition
import os

# Direktori simpan
NORMALIZED_DIR = "normalized"
os.makedirs(NORMALIZED_DIR, exist_ok=True)

# Load YOLO-Face
yolo_model = YOLO("yolov8n-face.pt")

# Contoh gambar known face
img_path = "1.jpg"
image = cv2.imread(img_path)

if image is None:
    print(f"[ERROR] Tidak bisa membuka gambar: {img_path}")
    exit()

# --- Gunakan YOLO untuk deteksi wajah ---
results = yolo_model(image)  # YOLO otomatis preprocess

# Ambil bounding box wajah pertama (asumsi satu wajah per foto)
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
if len(boxes) == 0:
    print(f"[WARN] No face detected in {img_path}")
    exit()

x1, y1, x2, y2 = boxes[0][:4].astype(int)

# Crop wajah sesuai bounding box YOLO
face_crop = image[y1:y2, x1:x2]

# Resize jika terlalu besar agar face_recognition stabil
MAX_HEIGHT = 384
h, w = face_crop.shape[:2]
if h > MAX_HEIGHT:
    scale = MAX_HEIGHT / h
    new_w = int(w * scale)
    face_crop = cv2.resize(face_crop, (new_w, MAX_HEIGHT))

# Pastikan RGB dan contiguous
rgb_crop = np.ascontiguousarray(face_crop[:, :, ::-1].astype(np.uint8))

# Simpan hasil crop untuk debug
# crop_path = os.path.join(NORMALIZED_DIR, "crop_debug.jpg")
cv2.imwrite("crop_debug.jpg", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
print(f"[INFO] Crop wajah disimpan crop_debug.jpg")

# Deteksi wajah dulu
face_locations = face_recognition.face_locations(rgb_crop, model="cnn")

# Buat encoding wajah
print("[INFO] Lokasi wajah yang terdeteksi:\n")
print(face_locations)

encodings = face_recognition.face_encodings(rgb_crop, known_face_locations=face_locations, num_jitters=1)
if len(encodings) == 0:
    print("[WARN] Gagal encode wajah")
else:
    encoding = encodings[0]
    print("[INFO] Encoding wajah berhasil")
    print(encoding)
