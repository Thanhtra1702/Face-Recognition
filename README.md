# Hệ Thống Điểm Danh Sinh Viên Thông Minh (AI Kiosk) - FastAPI & Anti-Spoofing

Dự án Kiosk điểm danh sử dụng công nghệ nhận diện khuôn mặt tiên tiến (ArcFace + Mediapipe), tích hợp hệ thống chống giả mạo đa tầng (Anti-Spoofing), chạy trên nền tảng FastAPI hiệu năng cao.

## 🚀 Tính Năng Chính

- **FastAPI Core:** Sử dụng FastAPI thay thế Flask để đạt tốc độ phản hồi cực nhanh, xử lý bất đồng bộ (Asynchronous) tối ưu.
- **Chống Giả Mạo Đa Tầng (Dual-Layer Anti-Spoofing):**
  - **Tầng 1 (Passive):** Sử dụng MiniFASNetV2 phân tích bề mặt da, phát hiện ảnh chụp hoặc video giả mạo với độ chính xác cao.
  - **Tầng 2 (Active Fallback):** Yêu cầu nháy mắt (Blink Detection) nếu AI nghi ngờ hoặc điều kiện ánh sáng không lý tưởng.
- **Nhận diện khuôn mặt HD:** Xử lý thời gian thực trên khung hình 1280x720, tracking mượt mà, độ trễ thấp.
- **Smart Snapshot:** Tự động "đóng băng" camera và vẽ khung xanh xác nhận khi nhận diện thành công.
- **Clean Snapshot & Self-Learning:** Lưu trữ song song ảnh gốc (Clean) để tự động training lại hệ thống, nâng cao độ chính xác theo thời gian.
- **Phân loại khoảng cách:** Chỉ kích hoạt nhận diện khi sinh viên đứng trong khoảng cách tối ưu (1.5m - 2m).

## 🛠 Yêu Cầu Hệ Thống

- **OS:** Windows 10/11, macOS, hoặc Linux.
- **Python:** 3.8 - 3.10 (Khuyên dùng 3.10).
- **Webcam:** Hỗ trợ HD 720p.
- **Thư viện chính:** mediapipe, deepface, fastapi, uvicorn, qdrant-client, onnxruntime, opencv-python.

## 📦 Cài Đặt

### 1. Clone dự án

```bash
git clone https://github.com/Thanhtra1702/Face-Recognition.git
cd student_face
```

### 2. Tạo môi trường ảo (Khuyên dùng)

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 🗄 Khởi Tạo Dữ Liệu (Lần đầu chạy)

1. **Khởi tạo Qdrant (Vector DB):**

   ```bash
   python init_qdrant.py
   ```

2. **Khởi tạo SQLite (Metadata DB):**

   ```bash
   python setup_database.py
   ```

3. **Xử lý ảnh tự học & Augmentation:**

   ```bash
   python process_collected_faces.py
   ```

## 🖥 Chạy Ứng Dụng Kiosk

```bash
python app.py
```

- Truy cập: `http://localhost:5000`
- Nhấn **Ctrl+C**: Tắt ứng dụng ngay lập tức (Xử lý dứt khoát).

## 📂 Cấu Trúc Dự Án (Modular Structure)

Dự án được phân tách thành các module chuyên biệt để dễ bảo trì và mở rộng:

```text
📁 student_face/
├── 📄 app.py                     # Entry point (FastAPI Server & Camera Loop)
├── 📄 anti_spoof.py              # Logic Chống giả mạo & Thị giác máy tính
├── 📄 recognition.py             # Logic Nhận diện khuôn mặt (DeepFace & Qdrant)
├── 📄 core_state.py              # Quản lý trạng thái hệ thống (KioskState)
├── 📄 kiosk_db.py                # Database Handler (Qdrant + SQLite)
├── 📄 process_collected_faces.py # Xử lý ảnh tự học & Augmentation
├── 📁 templates/                 # UI (FastAPI Jinja2 Templates)
├── 📁 static/                    # Assets (CSS/JS/Images)
├── 📁 collected_faces/           # Ảnh chờ xử lý / processed
├── 📄 MiniFASNetV2.onnx          # Model AI Chống giả mạo
└── 📁 qdrant_db/                 # Vector Database
```

## 🔒 Thông số tối ưu (Current Config)

- **Resolution:** 1280x720 (720p HD).
- **Threshold:** 0.45 (Cơ bản) / 0.65 (Xác nhận tức thì).
- **MiniFASNet Threshold:** 0.99 (Độ tin cậy liveness).
- **Blink Threshold:** 0.20 (Ngưỡng nháy mắt).
- **Image Enhance:** CLAHE 3.0 (Cân bằng sáng).

## ❓ Troubleshooting

- **Yêu cầu nháy mắt liên tục:** AI chưa đủ tin tưởng vào liveness (do ánh sáng hoặc chất lượng ảnh). Hãy đảm bảo khuôn mặt đủ sáng.
- **Tắt ứng dụng:** Nếu nhấn Ctrl+C không tắt, hãy check lại file `app.py` xem đã được cập nhật bản FastAPI mới nhất chưa.
- **Camera lag:** Đảm bảo PC của bạn hỗ trợ AVX2/FMA để thư viện TensorFlow/ONNX chạy nhanh hơn.
