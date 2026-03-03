# Technical Report: Real-Time Face Recognition Attendance System with Dual-Layer Anti-Spoofing

**Project:** Smart Student Attendance AI Kiosk
**Authors:** Thanhtra1702
**Date:** March 2026
**Version:** 2.0 (FastAPI + Modular Architecture)

---

## Abstract

Bài báo cáo kỹ thuật này trình bày chi tiết về việc thiết kế và triển khai một hệ thống điểm danh sinh viên tự động dựa trên nhận diện khuôn mặt thời gian thực. Hệ thống kết hợp mô hình **ArcFace** để trích xuất đặc trưng khuôn mặt, cơ sở dữ liệu vector **Qdrant** để tìm kiếm nhanh, và cơ chế chống giả mạo hai tầng (**MiniFASNetV2** + **Blink Detection**) nhằm ngăn chặn các hình thức gian lận. Toàn bộ hệ thống chạy trên nền tảng **FastAPI** với kiến trúc module hóa, đạt tốc độ xử lý dưới 2 giây cho mỗi lượt điểm danh.

---

## 1. Giới Thiệu

### 1.1. Bối cảnh & Vấn đề

Điểm danh truyền thống (gọi tên, ký tay, quẹt thẻ) gặp nhiều hạn chế: tốn thời gian, dễ gian lận (nhờ bạn điểm danh hộ), và khó quản lý khi số lượng sinh viên lớn. Các hệ thống nhận diện khuôn mặt cơ bản đã xuất hiện nhưng thường thiếu cơ chế chống giả mạo, dẫn đến việc sinh viên có thể dùng ảnh chụp hoặc video để qua mặt hệ thống.

### 1.2. Mục tiêu

- Xây dựng hệ thống điểm danh **không tiếp xúc** (contactless) với độ chính xác cao.
- Tích hợp cơ chế **chống giả mạo đa tầng** (Dual-Layer Anti-Spoofing) để ngăn chặn mọi hình thức gian lận phổ biến.
- Đảm bảo tốc độ phản hồi nhanh (< 2 giây) để phù hợp với môi trường thực tế.
- Hệ thống có khả năng **tự học** (Self-Learning) để cải thiện độ chính xác theo thời gian.

### 1.3. Phạm vi

Hệ thống được thiết kế cho môi trường giáo dục (trường đại học, cao đẳng) với quy mô từ vài trăm đến vài nghìn sinh viên. Kiosk sử dụng webcam HD 720p tiêu chuẩn, chạy trên máy tính phổ thông (không yêu cầu GPU chuyên dụng).

---

## 2. Kiến Trúc Hệ Thống (System Architecture)

### 2.1. Tổng quan kiến trúc

Hệ thống được thiết kế theo mô hình **Modular Architecture**, tách biệt rõ ràng các thành phần xử lý:

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                     │
│                   HTML/JS + Jinja2 Templates                │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI Server (app.py)                  │
│              API Routes + Camera Loop Thread                │
├─────────────┬─────────────────────┬─────────────────────────┤
│ anti_spoof  │   recognition.py    │     core_state.py       │
│   .py       │  (ArcFace + Qdrant) │   (KioskState Manager)  │
├─────────────┴─────────────────────┴─────────────────────────┤
│                      Data Layer                             │
│         Qdrant (Vector DB)    +    SQLite (Metadata)        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2. Mô tả các module

| Module | File | Chức năng |
| :--- | :--- | :--- |
| **Entry Point** | `app.py` | Khởi tạo FastAPI server, camera loop, xử lý API routes |
| **Anti-Spoofing** | `anti_spoof.py` | Chứa MiniFASNetV2 inference, hàm tính EAR (Eye Aspect Ratio), tiền xử lý ảnh CLAHE |
| **Recognition** | `recognition.py` | Trích xuất embedding bằng ArcFace, tìm kiếm trong Qdrant, logic xác nhận danh tính |
| **State Manager** | `core_state.py` | Quản lý trạng thái toàn cục (KioskState), thread-safe với Lock |
| **Database** | `kiosk_db.py` | Kết nối và truy vấn Qdrant + SQLite |
| **Self-Learning** | `process_collected_faces.py` | Xử lý ảnh thu thập được, augmentation, cập nhật Vector DB |

### 2.3. Tech Stack

| Thành phần | Công nghệ | Lý do chọn |
| :--- | :--- | :--- |
| **Web Framework** | FastAPI + Uvicorn | Hỗ trợ async, hiệu năng cao hơn Flask 3-5 lần |
| **Face Detection** | Mediapipe Face Mesh | 468 landmarks, chạy tốt trên CPU, tracking mượt |
| **Face Recognition** | DeepFace (ArcFace) | SOTA accuracy trên LFW benchmark (99.82%) |
| **Anti-Spoofing** | MiniFASNetV2 (ONNX) | Nhẹ, nhanh, chuyên biệt cho Face Anti-Spoofing |
| **Vector Database** | Qdrant | Tìm kiếm vector nhanh (HNSW), dễ triển khai |
| **Metadata DB** | SQLite | Nhẹ, không cần server riêng, đủ cho quy mô trường học |
| **Image Processing** | OpenCV + NumPy | Tiêu chuẩn ngành cho xử lý ảnh real-time |

---

## 3. Phương Pháp (Methodology)

### 3.1. Pipeline Xử Lý Tổng Quan

Mỗi frame từ camera đi qua pipeline sau:

```
Camera (720p) ──► Face Detection (Mediapipe) ──► Liveness Check ──► Recognition ──► Result
                         │                           │                    │
                    Tìm khuôn mặt              Người thật?          Ai đây?
                    Tính khoảng cách           (FAS + Blink)      (ArcFace + Qdrant)
```

**Giải thích đơn giản:** Hệ thống hoạt động giống như một chuỗi "cửa kiểm tra" tại sân bay. Bạn phải qua từng cửa một — nếu trượt ở bất kỳ cửa nào, bạn sẽ bị đưa về điểm xuất phát.

### 3.2. Face Detection & Tracking (Tìm và theo dõi khuôn mặt)

**Công nghệ:** Mediapipe Face Mesh

Mediapipe cung cấp 468 điểm landmark trên khuôn mặt. Hệ thống sử dụng các điểm này để:

1. **Xác định bounding box:** Tính tọa độ `(x_min, y_min, x_max, y_max)` bao quanh khuôn mặt.
2. **Chọn khuôn mặt chính (Focus Score):** Khi có nhiều người trong khung hình, hệ thống chọn khuôn mặt có **diện tích lớn nhất** và **gần tâm màn hình nhất** bằng công thức:

   ```
   focus_score = face_area / (distance_to_center + 1)
   ```

   Khuôn mặt có `focus_score` cao nhất sẽ được xử lý. Điều này đảm bảo hệ thống luôn ưu tiên người đang đứng ngay trước kiosk.

3. **Kiểm tra khoảng cách:** Khuôn mặt phải có chiều rộng > 180 pixels (tương đương khoảng cách 1.5-2m với camera 720p). Nếu quá nhỏ (đứng xa), hệ thống sẽ không kích hoạt nhận diện để tránh sai số.

**Cách hiểu đơn giản:** Mediapipe giống như một người bảo vệ đứng trước cửa — nó kiểm tra xem có ai đang đứng trước cửa không, và người đó có đứng đủ gần để được phục vụ hay không.

### 3.3. Dual-Layer Anti-Spoofing (Chống giả mạo hai tầng)

Đây là phần quan trọng nhất của hệ thống. Mục tiêu: **đảm bảo người đứng trước camera là người thật, không phải ảnh in, ảnh trên điện thoại, hay video phát lại.**

#### 3.3.1. Tầng 1 — Passive Anti-Spoofing (MiniFASNetV2)

**Nguyên lý:** Phân tích texture (kết cấu bề mặt) của vùng da trong ảnh.

- **Mô hình:** MiniFASNetV2, một biến thể nhẹ của kiến trúc MobileNetV2, được huấn luyện chuyên biệt cho tác vụ phân loại **Real vs Fake**.
- **Input:** Ảnh khuôn mặt được crop và resize về kích thước **80×80 pixels**, chuyển sang định dạng **NCHW** `(1, 3, 80, 80)`.
- **Output:** Hai giá trị (Fake probability, Real probability) qua hàm Softmax. Hệ thống lấy `prob[1]` (xác suất Real).
- **Ngưỡng quyết định:** `fas_score > 0.99` — nghĩa là mô hình phải **tin chắc 99%** rằng đây là người thật.
- **Inference Engine:** ONNX Runtime (CPU), đảm bảo tốc độ inference < 50ms.

**Cách hiểu đơn giản:** Hãy tưởng tượng bạn đưa tay sờ vào một bức tượng sáp và da người thật — dù nhìn giống nhau, nhưng kết cấu hoàn toàn khác. MiniFASNetV2 làm điều tương tự nhưng bằng "mắt AI": nó phân biệt da thật với bề mặt giấy/kính điện thoại thông qua các đặc trưng texture mà mắt thường không thấy được.

**Tại sao ngưỡng lại rất cao (0.99)?** Vì chúng ta muốn hệ thống chỉ xác nhận khi **thực sự chắc chắn**. Nếu hạ ngưỡng xuống, một bức ảnh in chất lượng cao có thể qua mặt hệ thống. Ngưỡng 0.99 đồng nghĩa với việc: "Tôi tin 99% đây là người thật". Nếu không đạt được mức tin tưởng này, hệ thống sẽ yêu cầu bước kiểm tra tầng 2.

#### 3.3.2. Tầng 2 — Active Anti-Spoofing (Blink Detection)

**Nguyên lý:** Yêu cầu hành động vật lý (nháy mắt) mà ảnh tĩnh hoặc video lặp không thể thực hiện.

- **Công cụ:** Mediapipe Face Mesh cung cấp các landmarks cụ thể quanh mắt.
- **Thuật toán:** Eye Aspect Ratio (EAR)

  ```
  EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
  ```

  Trong đó `p1...p6` là 6 điểm landmark quanh mỗi mắt:
  - `p1, p4`: hai góc mắt (ngang)
  - `p2, p3`: mí mắt trên
  - `p5, p6`: mí mắt dưới

- **Logic phát hiện nháy mắt:**
  1. Khi EAR giảm xuống dưới ngưỡng **0.20** → mắt đang nhắm (`_eye_closed = True`).
  2. Khi EAR tăng trở lại trên 0.20 và trước đó `_eye_closed = True` → ghi nhận 1 lần nháy mắt thành công.
  3. Nháy mắt phải xảy ra trong vòng **4 giây** gần nhất để được coi là hợp lệ.

- **Kết hợp hai tầng:** Hệ thống xác nhận "người thật" nếu **một trong hai điều kiện** sau thỏa mãn:

  ```
  is_live = (fas_score > 0.99) OR (thời gian từ lần nháy mắt cuối < 4 giây)
  ```

**Cách hiểu đơn giản:** Nếu bạn giơ ảnh lên camera, ảnh không thể nháy mắt. Nếu bạn phát video quay sẵn, video đó phải có đúng thời điểm nháy mắt khớp với lúc hệ thống kiểm tra — điều này gần như không thể. Đây chính là lý do tầng 2 tồn tại: nó là "bài test" mà chỉ người thật mới có thể vượt qua.

### 3.4. Face Recognition (Nhận diện khuôn mặt)

#### 3.4.1. Trích xuất đặc trưng (Feature Extraction) — ArcFace

**Mô hình:** ArcFace (Additive Angular Margin Loss), truy cập qua thư viện DeepFace.

- **Kiến trúc backbone:** ResNet-based encoder.
- **Output:** Vector embedding **512 chiều** (512-D float array).
- **Ý nghĩa:** Mỗi khuôn mặt được "mã hóa" thành một điểm trong không gian 512 chiều. Hai khuôn mặt cùng một người sẽ có các điểm nằm gần nhau, hai người khác nhau sẽ có các điểm nằm xa nhau.

**Tại sao chọn ArcFace thay vì FaceNet?**

| Tiêu chí | FaceNet | ArcFace |
| :--- | :--- | :--- |
| Loss Function | Triplet Loss | Additive Angular Margin Loss |
| Accuracy (LFW) | 99.63% | **99.82%** |
| Khả năng phân biệt người giống nhau | Tốt | **Rất tốt** (nhờ angular margin) |
| Ổn định qua góc chụp | Khá | **Tốt hơn** |

**Cách hiểu đơn giản:** Hãy tưởng tượng mỗi khuôn mặt là một ngôi sao trên bầu trời. ArcFace không chỉ xếp các ngôi sao vào đúng chòm sao (nhóm cùng người), mà còn đẩy các chòm sao ra xa nhau nhất có thể. Nhờ vậy, ngay cả khi hai người trông giống nhau (ví dụ anh chị em sinh đôi), "chòm sao" của họ vẫn tách biệt rõ ràng.

#### 3.4.2. Tìm kiếm trong cơ sở dữ liệu vector — Qdrant

Sau khi có vector embedding 512-D, hệ thống cần tìm vector nào trong database gần nhất với vector vừa thu được.

- **Database:** Qdrant (Vector Database), lưu trữ tất cả embedding của sinh viên đã đăng ký.
- **Thuật toán tìm kiếm:** HNSW (Hierarchical Navigable Small World)
- **Metric:** Cosine Similarity (đo góc giữa hai vector, không phụ thuộc vào độ lớn).

**HNSW hoạt động thế nào?**

Thay vì so sánh vector mới với **tất cả** vector trong database (Brute-force, O(N)), HNSW xây dựng một đồ thị đa lớp:

- **Lớp trên cùng:** Chỉ chứa vài "đại diện" tiêu biểu → tìm nhanh vùng lân cận.
- **Các lớp dưới:** Ngày càng chi tiết hơn → thu hẹp phạm vi tìm kiếm.
- **Kết quả:** Tìm được "hàng xóm gần nhất" chỉ trong O(log N) phép so sánh.

**Cách hiểu đơn giản:** Giống như tìm một cuốn sách trong thư viện. Thay vì rà từng kệ (mất hàng giờ), bạn hỏi thủ thư: "Sách thuộc thể loại gì?" → đi đến khu vực đó → "Tác giả bắt đầu bằng chữ gì?" → đến đúng kệ → lấy sách. Chỉ cần vài bước nhảy là tìm ra.

#### 3.4.3. Logic Xác Nhận Danh Tính (Identity Confirmation)

Hệ thống không chỉ lấy kết quả Top-1 mà áp dụng nhiều lớp kiểm tra:

1. **Ngưỡng tối thiểu (Threshold = 0.45):** Score dưới 0.45 → từ chối ngay, khuôn mặt không khớp với ai.
2. **Fast Pass (Score > 0.65):** Score rất cao → xác nhận ngay lập tức mà không cần chờ thêm.
3. **Consecutive Match (0.45 < Score < 0.65):** Score trung bình → yêu cầu nhận diện **2 lần liên tiếp** cùng một kết quả trước khi xác nhận. Điều này giảm thiểu rủi ro nhận nhầm do nhiễu ảnh.
4. **Ambiguity Check:** So sánh score của Top-1 với Top-2. Nếu hai người khác nhau nhưng score chênh lệch < 0.02, hệ thống từ chối xác nhận để tránh nhầm lẫn.

**Bảng tóm tắt logic:**

| Điều kiện | Hành động | Lý do |
| :--- | :--- | :--- |
| Score < 0.45 | ❌ Từ chối | Không đủ giống bất kỳ ai |
| Score > 0.65 | ✅ Xác nhận ngay | Độ tin cậy rất cao |
| 0.45 < Score < 0.65 | ⏳ Chờ khớp lần 2 | Cần xác minh thêm |
| Gap(Top1 - Top2) < 0.02 | ❌ Từ chối | Hai người quá giống nhau, rủi ro nhầm |

**Cách hiểu đơn giản:** AI không bao giờ "đoán bừa". Nếu nó không chắc chắn, nó thà nói "Tôi chưa nhận ra bạn, hãy thử lại" còn hơn là nói sai tên bạn. Đây là triết lý thiết kế cốt lõi: **an toàn hơn là tiện lợi**.

---

## 4. Tiền Xử Lý Ảnh (Image Preprocessing)

### 4.1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Trước khi đưa ảnh vào bất kỳ mô hình AI nào, hệ thống áp dụng bộ tiền xử lý 2 bước để chuẩn hóa chất lượng ảnh đầu vào.

**Quy trình:**

1. **Khử nhiễu (Denoise):** Áp dụng `GaussianBlur(3×3)` để làm mượt các hạt nhiễu từ cảm biến camera, đặc biệt trong điều kiện ánh sáng yếu.
2. Chuyển ảnh từ BGR sang không gian màu **LAB** (tách riêng kênh sáng L và kênh màu A, B).
3. Áp dụng CLAHE chỉ trên **kênh L** (Lightness) với `clipLimit=2.0` và `tileGridSize=(8,8)`. (Mức 2.0 được chọn để cân bằng giữa việc làm sáng mặt và giữ độ tự nhiên của da).
4. Ghép lại và chuyển về BGR.

> **Lưu ý quan trọng:** Bước tiền xử lý này được đồng bộ 100% giữa tất cả các module (`anti_spoof.py`, `recognition.py`, `init_qdrant.py`, `process_collected_faces.py`) để đảm bảo chất lượng embedding luôn đồng nhất.

**Tại sao dùng CLAHE thay vì Histogram Equalization thông thường?**

Histogram Equalization (HE) chuẩn hóa toàn bộ ảnh → dễ bị "cháy sáng" ở những vùng vốn đã sáng. CLAHE chia ảnh thành **ô nhỏ 8x8** và cân bằng **từng ô riêng biệt**, đồng thời giới hạn mức tương phản (`clipLimit`) để tránh khuếch đại nhiễu.

**Cách hiểu đơn giản:** Giống như khi bạn chỉnh ảnh selfie — thay vì tăng sáng toàn bộ ảnh (làm trắng bệch mặt nhưng nền vẫn tối), CLAHE chỉ "bật đèn" ở những vùng tối trên mặt mà không ảnh hưởng đến các vùng khác.

---

## 5. Cơ Chế Tự Học (Self-Learning Mechanism)

### 5.1. Thu thập ảnh tự động

Khi một sinh viên được nhận diện thành công, hệ thống tự động lưu một **"Clean Snapshot"** — ảnh gốc không có khung viền hay watermark — vào thư mục `collected_faces/{student_id}/`.

### 5.2. Quy trình xử lý (`process_collected_faces.py`)

Script này chạy offline (không phải real-time) để:

1. Duyệt qua tất cả ảnh mới thu thập được.
2. Áp dụng GaussianBlur + CLAHE (đồng bộ với real-time).
3. Dùng `DeepFace.extract_faces(mediapipe, align=True)` để detect và align khuôn mặt — **đồng bộ hoàn toàn với pipeline real-time** trong `recognition.py`.
4. **Lưu avatar** từ ảnh gốc (không CLAHE) vào `database/` để giữ màu tự nhiên.
5. Áp dụng **Data Augmentation** (xoay ±5°, thay đổi sáng/tối/tương phản, lật ngang, blur nhẹ) trên aligned face → tạo **8 biến thể**.
6. Trích xuất embedding bằng `DeepFace.represent(skip)` và cập nhật vào Qdrant.

---

## 6. Khởi tạo và Quản lý Database (`init_qdrant.py`)

Hệ thống được thiết kế để việc quản lý dữ liệu trở nên cực kỳ đơn giản:

1. **Auto-Standardization**: Khi người dùng đưa ảnh gốc vào thư mục `database/` (dù là ảnh toàn thân, ảnh thẻ hay ảnh selfie), script sẽ tự động detect khuôn mặt và **ghi đè** bằng bản Portrait đã crop chuẩn (30% padding).
2. **Đa định dạng**: Hỗ trợ linh hoạt các định dạng `.jpg`, `.jpeg`, `.png`, và đặc biệt là `.webp`.
3. **Robust Fallback**: Hệ thống sử dụng pipeline 2 lớp:
    - **Mediapipe**: Ưu tiên tốc độ và độ chuẩn chân dung.
    - **RetinaFace**: Sử dụng làm fallback cho các ca khó (ánh sáng yếu, mặt nghiêng). Loại bỏ hoàn toàn OpenCV Haar Cascades để tránh nhận nhầm vật thể (như áo khoác) là khuôn mặt.

### 5.3. Lợi ích

**Cách hiểu đơn giản:** Lần đầu đăng ký, hệ thống chỉ có 3-5 ảnh của bạn. Sau mỗi lần điểm danh thành công, nó "nhớ" thêm một góc mặt mới. Sau 1 tháng, nó đã có hàng chục mẫu — kể cả khi bạn đổi kiểu tóc, đeo kính, hay khuôn mặt thay đổi theo thời gian, AI vẫn nhận ra bạn vì nó đã "thấy" bạn ở rất nhiều trạng thái khác nhau.

---

## 6. Cấu Hình & Thông Số Tối Ưu

| Thông số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| Camera Resolution | 1280 × 720 (720p HD) | Đủ nét để nhận diện, không quá nặng cho CPU |
| FAS Threshold | 0.99 | Chỉ chấp nhận khi tin chắc 99% là người thật |
| Blink EAR Threshold | 0.20 | Ngưỡng phát hiện mắt nhắm |
| Blink Validity Window | 4 giây | Nháy mắt phải xảy ra trong 4 giây gần nhất |
| Recognition Threshold | 0.45 (min) / 0.65 (fast pass) | Cân bằng giữa tốc độ và độ chính xác |
| Ambiguity Gap | 0.02 | Khoảng cách tối thiểu giữa Top-1 và Top-2 |
| Consecutive Match Required | 2 lần | Số lần khớp liên tiếp cần thiết (khi score trung bình) |
| CLAHE clipLimit | 2.0 | Cân bằng sáng tối ưu cho chân dung |
| CLAHE tileGridSize | 8 × 8 | Kích thước ô lưới cho CLAHE |
| Face Min Width | 180 pixels | Kích thước tối thiểu khuôn mặt để kích hoạt nhận diện |
| FAS Check Frequency | Mỗi 3 frames | Giảm tải CPU bằng cách không check mọi frame |
| Image Formats | .jpg, .png, .webp | Các định dạng ảnh được hỗ trợ trong hệ thống |

---

## 7. Luồng Xử Lý Real-Time Chi Tiết (Detailed Runtime Flow)

```
[Camera Thread - Chạy liên tục]
│
├── 1. Capture frame (1280x720, flip ngang)
├── 2. Mediapipe Face Mesh → Tìm khuôn mặt
│       ├── Không tìm thấy → Quay lại bước 1
│       └── Tìm thấy → Chọn best face (focus_score)
│
├── 3. Kiểm tra khoảng cách (face_width > 180px?)
│       ├── Không đủ gần → Hiển thị khung trắng, chờ
│       └── Đủ gần → Hiển thị khung xanh dương
│
├── 4. Liveness Check
│       ├── 4a. MiniFASNetV2 predict (mỗi 3 frames)
│       │       └── fas_score > 0.99? → is_pass_fas
│       ├── 4b. Blink Detection (mọi frame)
│       │       └── Nháy mắt trong 4s gần nhất? → is_pass_blink
│       └── is_live = is_pass_fas OR is_pass_blink
│
├── 5. Nếu is_live AND status == "SCANNING":
│       ├── Crop khuôn mặt từ RAW FRAME (padding 40%, ảnh sạch)
│       ├── Áp dụng GaussianBlur + CLAHE preprocessing
│       └── Gửi sang Recognition Thread (async)
│
└── 6. [Recognition Thread - Bất đồng bộ]
        ├── DeepFace.extract_faces (Mediapipe, align=True)
        ├── DeepFace.represent (ArcFace, 512-D embedding)
        ├── Qdrant query (Top-3 kết quả)
        ├── Ambiguity Check (Gap >= 0.02?)
        ├── Threshold Check (Score > 0.45?)
        ├── Consecutive Match Check (2 lần liên tiếp?)
        │
        ├── ✅ CONFIRM → Hiển thị khung XANH LÁ + thông tin SV
        │       └── Lưu Clean Snapshot → collected_faces/
        └── ❌ REJECT → Quay lại SCANNING
```

---

## 8. Hạn Chế & Hướng Phát Triển

### 8.1. Hạn chế hiện tại

- **Ánh sáng cực đoan:** Trong điều kiện gần như tối hoàn toàn hoặc ngược sáng mạnh, CLAHE có thể không đủ bù đắp.
- **Blink Detection bypass:** Video deepfake chất lượng cao có thể mô phỏng nháy mắt tự nhiên. Cần bổ sung thêm các challenge khác (quay đầu, cười).
- **Đơn camera:** Hệ thống hiện chỉ dùng 1 camera 2D, không có thông tin độ sâu (3D).

### 8.2. Hướng phát triển

- Tích hợp **3D Face Anti-Spoofing** sử dụng camera depth (Intel RealSense).
- Thêm **Head Pose Estimation** để yêu cầu xoay đầu như một challenge bổ sung.
- Triển khai **GPU inference** (CUDA/TensorRT) để xử lý nhiều kiosk đồng thời.
- Xây dựng **Dashboard quản lý** cho giảng viên theo dõi lịch sử điểm danh.

---

## 9. Kết Luận

Hệ thống AI Kiosk đã đạt được sự cân bằng giữa **bảo mật** (Dual-Layer Anti-Spoofing), **tốc độ** (< 2 giây/lượt, nhờ FastAPI + HNSW), và **độ chính xác** (ArcFace 99.82% trên LFW). Kiến trúc module hóa cho phép dễ dàng nâng cấp từng thành phần mà không ảnh hưởng đến toàn hệ thống. Cơ chế Self-Learning đảm bảo hệ thống ngày càng thông minh hơn theo thời gian sử dụng.

---

## 10. Tham Khảo (References)

1. Deng, J. et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
2. Yu, Z. et al. (2020). *Searching Central Difference Convolutional Networks for Face Anti-Spoofing.* CVPR 2020.
3. Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* Google Research.
4. Malkov, Y. & Yashunin, D. (2020). *Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs.* IEEE TPAMI.
5. Sokolova, M. & Lapalme, G. (2009). *A Systematic Analysis of Performance Measures for Classification Tasks.* Information Processing & Management.

---

*Technical Report — Generated: March 3, 2026*
