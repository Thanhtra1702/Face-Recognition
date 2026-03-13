import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import uuid

from arcface_onnx import ArcFaceONNX

# Cấu hình
COLLECTION_NAME = "student_faces"
IMAGE_DIR = "./database"
MODEL_PATH = "w600k_r50.onnx"

# ── Template tọa độ chuẩn ArcFace 112x112 (đồng bộ với recognition.py) ──
ARCFACE_DST = np.array([
    [30.2946, 51.6963],  # L-Eye
    [65.5318, 51.5014],  # R-Eye
    [48.0252, 71.7366],  # Nose
    [33.5493, 92.3655],  # L-Mouth
    [62.7299, 92.2041],  # R-Mouth
], dtype=np.float32)

# --- AI ENHANCEMENT HELPERS ---
def preprocess_frame(frame):
    """Cân bằng sáng và khử nhiễu để AI dễ đọc hơn"""
    try:
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    except:
        return frame


def align_face_mediapipe(frame, face_landmarks, target_size=(112, 112)):
    """
    Align khuôn mặt bằng Mediapipe landmarks → template ArcFace 112x112.
    Đồng bộ hoàn toàn với recognition.py/get_standard_aligned_face.
    """
    h, w = frame.shape[:2]
    landmarks = face_landmarks.landmark

    l_eye = np.mean([
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[133].x * w, landmarks[133].y * h)
    ], axis=0)
    r_eye = np.mean([
        (landmarks[362].x * w, landmarks[362].y * h),
        (landmarks[263].x * w, landmarks[263].y * h)
    ], axis=0)
    nose = (landmarks[1].x * w, landmarks[1].y * h)
    l_mouth = (landmarks[61].x * w, landmarks[61].y * h)
    r_mouth = (landmarks[291].x * w, landmarks[291].y * h)

    src = np.array([l_eye, r_eye, nose, l_mouth, r_mouth], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST)
    if M is None:
        return cv2.resize(frame, target_size)

    aligned = cv2.warpAffine(frame, M, target_size, borderMode=cv2.BORDER_CONSTANT)
    return aligned


def get_face_bbox_from_landmarks(face_landmarks, h, w):
    """Tính bounding box từ Mediapipe landmarks."""
    x_coords = [lm.x * w for lm in face_landmarks.landmark]
    y_coords = [lm.y * h for lm in face_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return x_min, y_min, x_max, y_max


def init_qdrant():
    # ── Load ArcFace ONNX ──
    print(f"🔧 Loading ArcFace ONNX model: {MODEL_PATH}")
    arcface = ArcFaceONNX(MODEL_PATH)

    # ── Init Mediapipe Face Mesh (cho static images) ──
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,       # Tối ưu cho ảnh tĩnh (không tracking)
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # ── Khởi tạo Qdrant ──
    client = QdrantClient(host="localhost", port=6333)

    print(f"Đang tạo lại collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=arcface.embedding_size, distance=Distance.COSINE),
    )

    # ── Lấy danh sách ảnh ──
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    if not image_files:
        print("Không tìm thấy ảnh nào trong thư mục database!")
        return

    print(f"Bắt đầu xử lý {len(image_files)} ảnh (Mediapipe Align + ONNX ArcFace + Augmentation)...")

    points = []

    for filename in tqdm(image_files):
        student_id = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGE_DIR, filename)

        try:
            # 1. Đọc và tiền xử lý ảnh
            img_raw = cv2.imread(img_path)
            if img_raw is None:
                continue
            h_orig, w_orig = img_raw.shape[:2]
            img_processed = preprocess_frame(img_raw)

            # 2. Detect face bằng Mediapipe
            rgb_img = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)

            if not results.multi_face_landmarks:
                print(f"⚠️ Không phát hiện khuôn mặt trong {filename}")
                continue

            face_landmarks = results.multi_face_landmarks[0]

            # 3. Smart Portrait Crop (crop từ ảnh gốc, không CLAHE)
            x_min, y_min, x_max, y_max = get_face_bbox_from_landmarks(face_landmarks, h_orig, w_orig)
            fw, fh = x_max - x_min, y_max - y_min
            p = 0.3  # 30% padding cho avatar thoáng
            cx1 = max(0, int(x_min - fw * p))
            cy1 = max(0, int(y_min - fh * p))
            cx2 = min(w_orig, int(x_max + fw * p))
            cy2 = min(h_orig, int(y_max + fh * p))

            portrait_img = img_raw[cy1:cy2, cx1:cx2]
            if portrait_img.size > 0:
                cv2.imwrite(img_path, portrait_img)

            # 4. Align face → 112x112 (đồng bộ với recognition.py)
            # Dùng ảnh processed (CLAHE) cho alignment giống live
            aligned_face = align_face_mediapipe(img_processed, face_landmarks)

            # 5. Tạo biến thể (Augmentation x3)
            variants = [
                ("orig", aligned_face),
                ("bright", cv2.convertScaleAbs(aligned_face, alpha=1.15, beta=20)),
                ("dark", cv2.convertScaleAbs(aligned_face, alpha=0.85, beta=-15)),
            ]

            # 6. ONNX ArcFace → embedding
            for var_name, var_img in variants:
                try:
                    embedding = arcface.get_embedding(var_img, normalize=True)
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "student_id": student_id,
                            "filename": filename,
                            "variant": var_name,
                        }
                    ))
                except Exception:
                    pass

        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {e}")

    # ── Push dữ liệu lên Qdrant ──
    if points:
        batch_size = 100
        print(f"Đang lưu {len(points)} vector vào DB...")

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
            )
        print(f"✅ Đã lưu {len(points)} vector vào Qdrant thành công!")
    else:
        print("Không có vector nào được tạo.")

    face_mesh.close()

if __name__ == "__main__":
    init_qdrant()
