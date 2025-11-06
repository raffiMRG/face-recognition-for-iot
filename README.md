## Face Detection & Recognition Using Yolo For IoT Dor Lock

First time i using yolo for face detection & recognition

---

## 🚀 Features

- Face detection menggunakan YOLO model.
- Face recognition menggunakan embedding vector.
- Dapat dijalankan di virtual environment (`venv`).

---

## 🛠️ Installation

### Clone Repository

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### Buat Virtual Environment

```bash
python -m venv venv
```

### Aktifkan Virtual Environment

windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Jalankan script utama

jalankan dalam venv

```bash
python3 app.py
```

---

### Test

endpoint

```bash
http://127.0.0.1:8123/detect
```

Request Body

`form-data`

`Key` = file

`Type` = `File`

`Value` = `IMG.jpg`

Response

```bash
{
    "action": "allow",
    "confidence": 0.7998242863355989,
    "name": "mrg"
}
```
