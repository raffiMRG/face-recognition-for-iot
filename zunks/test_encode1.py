import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

# === Load YOLOv8-Face ===
model = YOLO("yolov8n-face.pt")

# === Gambar test ===
img_path = "1.jpg"  # ganti dengan path gambar kamu sendiri
img = cv2.imread(img_path)

if img is None:
    print(f"[ERROR] Tidak bisa membuka gambar: {img_path}")
    exit()

# === Jalankan deteksi wajah YOLO ===
results = model(img)
if len(results) == 0 or len(results[0].boxes) == 0:
    print("[WARN] Tidak ada wajah terdeteksi oleh YOLO")
    exit()

# Ambil bounding box wajah pertama
x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)

# Perbesar crop
w = x2 - x1
h = y2 - y1
cx = x1 + w//2
cy = y1 + h//2
scale = 1.3
new_w = int(w*scale)
new_h = int(h*scale)
x1 = max(0, cx - new_w//2)
y1 = max(0, cy - new_h//2)
x2 = min(img.shape[1], cx + new_w//2)
y2 = min(img.shape[0], cy + new_h//2)

# Crop wajah
face_crop = img[y1:y2, x1:x2]

# Resize agar tidak terlalu besar
MAX_HEIGHT = 384
ch, cw = face_crop.shape[:2]
if ch > MAX_HEIGHT:
    scale = MAX_HEIGHT / ch
    face_crop = cv2.resize(face_crop, (int(cw*scale), MAX_HEIGHT))

# Pastikan RGB & contiguous
rgb_crop = np.ascontiguousarray(face_crop[:, :, ::-1].astype(np.uint8))

# Encode wajah
encodings = face_recognition.face_encodings(rgb_crop)
if len(encodings) == 0:
    print("[WARN] Gagal encode wajah")
else:
    print(f"[INFO] Berhasil encode wajah, dimension: {encodings[0].shape}")

# Simpan crop untuk debug
cv2.imwrite("debug_crop.jpg", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
print("[INFO] Crop wajah disimpan ke debug_crop.jpg")

# Tampilkan crop wajah dengan OpenCV
cv2.imshow("Face Crop", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
