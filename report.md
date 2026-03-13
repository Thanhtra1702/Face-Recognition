# Technical Report: High-Precision Attendance System with Multi-Scale Anti-Spoofing

**Project:** AI Smart Kiosk for Student Attendance
**Version:** 4.0 (ONNX Direct Inference + Multi-Scale Anti-Spoofing)
**Date:** March 13, 2026

---

## 1. Abstract

Báo cáo trình bày chi tiết về hệ thống điểm danh khuôn mặt thời gian thực, kết hợp các công nghệ SOTA (**ArcFace ONNX**, **Qdrant**, **Mediapipe**) và cơ chế bảo mật đa tầng. Phiên bản V4.0 đánh dấu bước nhảy quan trọng khi **loại bỏ hoàn toàn DeepFace wrapper**, chuyển sang **ONNX Runtime trực tiếp** — tăng tốc inference **5-10 lần** (từ 100-300ms xuống 10-30ms/face) trong khi giảm RAM sử dụng từ ~2GB xuống ~500MB. Hệ thống vẫn duy trì đầy đủ khả năng chống giả mạo thụ động mạnh mẽ (**Multi-Scale FAS**) và logic quyết định thông minh (**Top-5 Voting Consensus**).

---

## 2. Kiến trúc Hệ thống (System Architecture)

Hệ thống được xây dựng theo mô hình **Modular Architecture**, chia thành các khối chức năng độc lập:

- **Frontend (Web-based):** HTML/JS giao diện phẳng, hiển thị feedback màu sắc thời gian thực.
- **Application Logic (`app.py`):** Quản lý luồng Camera, xử lý Mediapipe Face Mesh và điều phối các Thread nhận diện.
- **ONNX Engine (`arcface_onnx.py`):** Singleton ArcFace ONNX Runtime với GPU auto-detect, session optimization.
- **Liveness Engine (`anti_spoof.py`):** Thực thi mô hình MiniFASNetV2 với chiến lược phân tích đa quy mô.
- **Recognition Engine (`recognition.py`):** TTA 3 variants, ONNX embedding extraction, Top-5 Voting Consensus.
- **Database Layer (`kiosk_db.py`):** Kết nối Qdrant (Vector) và SQLite (Metadata).

---

## 3. Các Giải pháp Công nghệ & Cải tiến

### 3.1. Pipeline Nhận diện SOTA (V4.0)

Mỗi khung hình đi qua một pipeline tinh gọn nhưng đạt độ chính xác cực cao:

1. **Detection (Mediapipe):** Tìm 468 điểm landmarks nhanh chóng trên CPU (~5ms).
2. **Standard 5-Point Alignment:** Sử dụng 5 điểm landmarks (mắt, mũi, miệng) để map chính xác vào template **112x112** tiêu chuẩn của ArcFace.
   - *Ưu điểm:* Đưa khuôn mặt về đúng "vùng hiểu biết" tốt nhất của AI, tối ưu hóa sai số do góc nghiêng.
3. **Test-Time Augmentation (TTA):** Đối với mỗi lần nhận diện, hệ thống tạo ra **3 biến thể** của khuôn mặt (Gốc, Cân bằng sáng CLAHE, Tăng cường độ tương phản).
   - *SOTA Technique:* Tính toán **trung bình cộng của 3 vector embedding** rồi re-normalize L2 để tạo "Robust Embedding" cuối cùng.
4. **ONNX Direct Inference:** Trích xuất embedding trực tiếp qua ONNX Runtime (~10-30ms/variant), không qua DeepFace wrapper.
   - *Model:* `w600k_r50.onnx` (ResNet50, trained on WebFace600K, 512-dimensional embedding)
   - *Preprocessing:* BGR→RGB, normalize `(pixel - 127.5) / 127.5`, sử dụng `cv2.dnn.blobFromImage`

### 3.2. Chống giả mạo Đa quy mô (Multi-Scale FAS)

Đây là cải tiến quan trọng nhất để chặn video replay mà không cần người dùng hành động:

- **Scale 1 (Tight Crop - 1.0x):** Tập trung vào kết cấu da (Skin Texture).
- **Scale 2 (Wide Crop - 2.7x):** Nhìn rộng ra bối cảnh xung quanh để phát hiện viền màn hình, ánh sáng phản xạ, mất chiều sâu.
- **Logic thắt chặt (The MIN Rule):**
  `Final_Score = min(Score_Tight, Score_Wide)`
  Chỉ cần 1 trong 2 quy mô bất thường → đánh dấu **Fake**.
- **Blink Detection:** EAR (Eye Aspect Ratio) từ 468 Mediapipe landmarks.
- **FFT Moiré Detection:** Phát hiện vân pixel đặc trưng của màn hình kỹ thuật số qua Fast Fourier Transform.

### 3.3. Logic Xác nhận Thông minh (Decision Making)

Hệ thống không đánh cược vào 1 frame duy nhất mà sử dụng cơ chế bình chọn:

- **Top-5 Voting:** Truy vấn 5 kết quả gần nhất trong Qdrant. Chỉ sinh viên đạt đa số phiếu bầu mới được ghi nhận.
- **Ambiguity Gap:** Nếu chênh lệch giữa Top-1 và Top-2 quá nhỏ (< 0.05 khi Score < 0.70), hệ thống từ chối.
- **SOTA Fast Pass:** Score > 0.75 + Votes ≥ 3/5 → xác nhận tức thì.
- **Consecutive Match:** Yêu cầu ít nhất 2 lần match liên tiếp trước khi xác nhận.

### 3.4. Kiểm soát độ ổn định (Stability Logic)

- **Face Stability:** Khuôn mặt giữ vị trí < 25px trong **0.8 giây**.
- **Head Pose Check:** Tỉ lệ khoảng cách mũi-mắt đảm bảo nhìn thẳng (0.4 < ratio < 2.5).
- **Blur Detection:** Laplacian variance > 60 mới cho phép nhận diện.

---

## 4. So sánh Hiệu năng V3.0 (DeepFace) vs V4.0 (ONNX)

| Metric | V3.0 (DeepFace) | V4.0 (ONNX Direct) | Cải thiện |
|---|---|---|---|
| **Embedding inference** | ~100-300ms/face | ~10-30ms/face | **5-10x nhanh hơn** |
| **TTA (3 variants)** | ~300-900ms | ~30-90ms | **10x nhanh hơn** |
| **Model startup** | ~5-10s (TF/Keras) | ~1-2s (ONNX) | **5x nhanh hơn** |
| **RAM usage** | ~1.5-2GB | ~500MB | **3-4x ít hơn** |
| **Dependencies** | ~2GB (deepface+tf+keras) | ~200MB (onnxruntime) | **10x nhỏ hơn** |

---

## 5. Thiết kế Trải nghiệm (UI/UX)

Hệ thống sử dụng phản hồi màu sắc tối giản (2px thickness):

- **TRẮNG:** Đang tìm kiếm hoặc đối tượng ở quá xa (Face width < 180px).
- **CAM:** Đã vào vị trí quét, đang âm thầm kiểm tra Liveness.
- **XANH LÁ:** Xác nhận danh tính thành công.

---

## 6. Hiệu năng Thực tế

- **Tốc độ:** Nhận diện hoàn tất trong < 1 giây kể từ khi sinh viên đứng yên.
- **Độ chính xác:** Score trung bình 0.70-0.77 với Votes 5/5 cho khuôn mặt đã đăng ký.
- **Khả năng mở rộng:** Qdrant HNSW Index duy trì tốc độ tìm kiếm O(log N) khi database mở rộng.

---

## 7. Kết luận

Hệ thống Smart Kiosk V4.0 đánh dấu bước tiến quan trọng khi loại bỏ hoàn toàn dependency nặng (TensorFlow, DeepFace) chuyển sang ONNX Runtime trực tiếp — giúp hệ thống nhẹ hơn, nhanh hơn, và dễ triển khai hơn. Kết hợp với pipeline anti-spoofing đa tầng và logic voting thông minh, hệ thống đạt được sự cân bằng tối ưu giữa **tốc độ**, **độ chính xác**, và **bảo mật**.

---
*Report Generated by Project Team — March 2026*
