# Hệ Thống Điểm Danh Sinh Viên Thông Minh (AI Kiosk)

Dự án Kiosk điểm danh sử dụng công nghệ nhận diện khuôn mặt tiên tiến (**ArcFace ONNX + Mediapipe**), tích hợp hệ thống chống giả mạo đa tầng (**Multi-Scale Anti-Spoofing**), chạy trên nền tảng **FastAPI** hiệu năng cao.

## 🚀 Tính Năng Chính

- **ONNX Direct Inference:** ArcFace ONNX Runtime trực tiếp (~10-30ms/face), không qua wrapper nặng, hỗ trợ tự động phát hiện GPU CUDA.
- **Test-Time Augmentation (TTA):** Tạo 3 biến thể ảnh (Original + CLAHE + Bright) → trung bình embedding → nhận diện cực kỳ ổn định trong mọi điều kiện ánh sáng.
- **Chống Giả Mạo Đa Tầng (Multi-Layer Anti-Spoofing):**
  - **Tầng 1 — MiniFASNetV2 Multi-Scale:** Phân tích kết cấu da (cận cảnh) & bối cảnh (rộng 2.7x) để phát hiện ảnh/video giả mạo.
  - **Tầng 2 — Blink Detection:** Phát hiện nháy mắt qua Eye Aspect Ratio (EAR).
  - **Tầng 3 — FFT Moiré Detection:** Phát hiện vân pixel từ màn hình điện thoại/máy tính.
- **Smart Decision Logic:**
  - **Top-5 Voting Consensus:** Truy vấn 5 vectors gần nhất, chỉ xác nhận khi đa số phiếu bầu cho cùng 1 người.
  - **Ambiguity Gap Protection:** Từ chối nếu chênh lệch Top-1 và Top-2 quá nhỏ.
  - **SOTA Fast Pass:** Xác nhận tức thì khi Score > 0.75 và Votes ≥ 3/5.
- **5-Point Affine Alignment:** Chuẩn hóa khuôn mặt về template 112×112 chuẩn ArcFace từ 468 Mediapipe landmarks.
- **Qdrant Vector Database:** HNSW Index cho tốc độ tìm kiếm ổn định O(log N) ngay cả với hàng chục nghìn khuôn mặt.
- **Self-Learning:** Tự động lưu ảnh check-in và cập nhật database để cải thiện độ chính xác theo thời gian.

## 🛠 Yêu Cầu Hệ Thống

- **OS:** Windows 10/11, macOS, hoặc Linux.
- **Python:** 3.9 - 3.11 (Khuyên dùng 3.11).
- **Webcam:** Hỗ trợ HD 720p.
- **Docker:** Cần cho Qdrant Vector Database.
- **Thư viện chính:** mediapipe, onnxruntime, fastapi, uvicorn, qdrant-client, opencv-python.

## 📦 Cài Đặt

### 1. Clone dự án

```bash
git clone https://github.com/Thanhtra1702/student-face.git
cd student_face
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Tải model ArcFace ONNX

Tải file `w600k_r50.onnx` (~166MB) và đặt vào thư mục gốc:

```bash
curl -L -o w600k_r50.onnx https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx
```

### 5. Khởi động Qdrant (Docker)

```bash
docker-compose up -d
```

## 🗄 Khởi Tạo Dữ Liệu (Lần đầu chạy)

1. **Đặt ảnh khuôn mặt** vào thư mục `database/`. Tên file = MSSV (VD: `QE190099.jpg`).

2. **Khởi tạo Qdrant (Vector DB):**

   ```bash
   python init_qdrant.py
   ```

3. **Khởi tạo SQLite (Metadata DB):**

   ```bash
   python setup_database.py
   ```

## 👤 Thêm Người Dùng Mới

Đặt ảnh vào folder con trong `collected_faces/`, tên folder = ID người dùng:

```
collected_faces/
  └── QE190999/
        ├── anh1.jpg
        └── anh2.jpg
```

Chạy:

```bash
python process_collected_faces.py
```

Script sẽ tự động: detect → align → embedding → upsert Qdrant → cập nhật SQLite.

## 🖥 Chạy Ứng Dụng Kiosk

```bash
python app.py
```

- Truy cập: `http://localhost:5000`
- Nhấn **Ctrl+C**: Tắt ứng dụng.

## 📂 Cấu Trúc Dự Án

```text
📁 student_face/
├── 📄 app.py                      # Entry point (FastAPI Server & Camera Loop)
├── 📄 arcface_onnx.py             # ArcFace ONNX Engine (Singleton, GPU auto-detect)
├── 📄 anti_spoof.py               # Chống giả mạo Multi-Scale + Blink + FFT Moiré
├── 📄 recognition.py              # Nhận diện khuôn mặt (ONNX ArcFace + TTA + Voting)
├── 📄 core_state.py               # Quản lý trạng thái hệ thống (KioskState)
├── 📄 kiosk_db.py                 # Database Handler (Qdrant + SQLite)
├── 📄 init_qdrant.py              # Khởi tạo Vector Database từ ảnh
├── 📄 process_collected_faces.py  # Xử lý ảnh tự học & Augmentation
├── 📄 setup_database.py           # Khởi tạo SQLite metadata
├── 📄 w600k_r50.onnx              # Model ArcFace (ResNet50, 512d) — không upload git
├── 📄 MiniFASNetV2.onnx           # Model Anti-Spoofing
├── 📁 templates/                  # UI (Jinja2 Templates)
├── 📁 static/                     # Assets (CSS/JS/Images)
├── 📁 database/                   # Ảnh khuôn mặt gốc
├── 📁 collected_faces/            # Ảnh chờ xử lý / processed
└── 📁 qdrant_data/                # Vector Database
```

## ⚙️ Thông Số Kỹ Thuật

| Thông số | Giá trị |
|---|---|
| **Resolution** | 1280×720 (720p HD) |
| **Face Embedding** | ArcFace w600k_r50 (ONNX, 512d) |
| **Face Detection** | Mediapipe Face Mesh (468 landmarks) |
| **Alignment** | 5-Point Affine → 112×112 |
| **Vector DB** | Qdrant (HNSW, Cosine Similarity) |
| **Recognition Threshold** | 0.55 (cơ bản) / 0.75 (Fast Pass) |
| **FAS Threshold** | 0.85 (liveness) |
| **Blink Threshold** | 0.20 (EAR) |
| **Stability Hold** | 0.8s |

## ❓ Troubleshooting

- **Không nhận diện được ai:** Kiểm tra đã chạy `python init_qdrant.py` chưa. Nếu đổi model, phải re-index lại.
- **Lỗi "Không tìm thấy model ArcFace":** Tải `w600k_r50.onnx` và đặt vào thư mục gốc (xem bước 4 phần cài đặt).
- **Camera lag:** Đảm bảo PC hỗ trợ AVX2. Nếu có GPU NVIDIA, cài `onnxruntime-gpu` thay `onnxruntime` để tăng tốc.
- **Yêu cầu nháy mắt liên tục:** AI chưa đủ tin tưởng liveness (do ánh sáng). Đảm bảo khuôn mặt đủ sáng.
- **Qdrant connection error:** Kiểm tra Docker đang chạy: `docker-compose ps`.
