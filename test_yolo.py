from ultralytics import YOLO
import cv2

# Load model YOLOv8-Face
model = YOLO("yolov8n-face.pt")

# Gambar test
img_path = "1.jpg"  # ganti dengan path gambar kamu sendiri
img = cv2.imread(img_path)

if img is None:
    print(f"[ERROR] Tidak bisa membuka gambar: {img_path}")
    exit()

# Jalankan deteksi
results = model(img)

# Tampilkan hasil
for r in results:
    r.show()  # tampilkan window OpenCV dengan deteksi wajah

    # Simpan hasil deteksi ke file
    annotated = r.plot()
    cv2.imwrite("test_result.jpg", annotated)
    print("[INFO] Hasil disimpan ke test_result.jpg")
