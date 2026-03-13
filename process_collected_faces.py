import os
import cv2
import shutil
import numpy as np
import mediapipe as mp
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import sys
import datetime
import sqlite3
import random
import uuid

from arcface_onnx import get_arcface_model

sys.stdout.reconfigure(encoding='utf-8')

COLLECTED_DIR = "collected_faces"
PROCESSED_DIR = "collected_faces/processed"
DATABASE_DIR = "database"
COLLECTION_NAME = "student_faces"

# ── Template tọa độ chuẩn ArcFace 112x112 (đồng bộ với recognition.py) ──
ARCFACE_DST = np.array([
    [30.2946, 51.6963],  # L-Eye
    [65.5318, 51.5014],  # R-Eye
    [48.0252, 71.7366],  # Nose
    [33.5493, 92.3655],  # L-Mouth
    [62.7299, 92.2041],  # R-Mouth
], dtype=np.float32)

# --- AI ENHANCEMENT HELPERS (Đồng bộ với app.py và init_qdrant.py) ---
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
    """Align khuôn mặt bằng Mediapipe landmarks → template ArcFace 112x112."""
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
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))


def process_collected_images():
    if not os.path.exists(COLLECTED_DIR):
        print(f"Không tìm thấy thư mục {COLLECTED_DIR}")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # ── Load models ──
    arcface = get_arcface_model()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # Lấy danh sách ảnh (bao gồm cả trong subfolders)
    image_files = []
    for root, dirs, files in os.walk(COLLECTED_DIR):
        if 'processed' in root.replace('\\', '/').split('/'):
            continue
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("Không có ảnh nào cần xử lý.")
        return

    print(f"🔍 Tìm thấy {len(image_files)} ảnh cần xử lý...")
    
    # Init Qdrant Client
    client = QdrantClient(host="localhost", port=6333)

    count_success = 0
    
    for file_path in image_files:
        filename = os.path.basename(file_path)
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        
        if parent_dir_name == "collected_faces" or parent_dir_name == "processed":
            print(f"⚠️ Bỏ qua ảnh không nằm trong thư mục MSSV: {filename}")
            continue
            
        mssv = parent_dir_name
        
        print(f"\n📸 Đang xử lý: {filename} (MSSV: {mssv})")
        
        # 1. Đọc ảnh gốc
        img_raw = cv2.imread(file_path)
        if img_raw is None:
            print("❌ Lỗi đọc ảnh.")
            continue
        
        img_height, img_width = img_raw.shape[:2]
        img_for_ai = preprocess_frame(img_raw)

        try:
            # --- DETECT FACE bằng Mediapipe ---
            rgb_img = cv2.cvtColor(img_for_ai, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)

            if not results.multi_face_landmarks:
                print("❌ Không phát hiện khuôn mặt.")
                continue

            # Chọn khuôn mặt tốt nhất (lớn nhất + gần trung tâm nhất)
            best_face = None
            best_score = -1
            img_center_x = img_width / 2
            img_center_y = img_height / 2

            for face_landmarks in results.multi_face_landmarks:
                x_min, y_min, x_max, y_max = get_face_bbox_from_landmarks(
                    face_landmarks, img_height, img_width
                )
                area = (x_max - x_min) * (y_max - y_min)
                face_cx = (x_min + x_max) / 2
                face_cy = (y_min + y_max) / 2
                distance = ((face_cx - img_center_x)**2 + (face_cy - img_center_y)**2)**0.5
                max_distance = (img_width**2 + img_height**2)**0.5
                distance_score = 1 - (distance / max_distance)
                area_score = area / (img_width * img_height)
                total_score = 0.7 * area_score + 0.3 * distance_score

                if total_score > best_score:
                    best_score = total_score
                    best_face = face_landmarks

            # Align face → 112x112
            aligned_face = align_face_mediapipe(img_for_ai, best_face)
            
            # Crop & Save Avatar
            target_path = os.path.join(DATABASE_DIR, f"{mssv}.jpg")
            
            x_min, y_min, x_max, y_max = get_face_bbox_from_landmarks(
                best_face, img_height, img_width
            )
            fw, fh = x_max - x_min, y_max - y_min
            padding = 0.3
            x1 = max(0, int(x_min - fw * padding))
            y1 = max(0, int(y_min - fh * padding))
            x2 = min(img_width, int(x_max + fw * padding))
            y2 = min(img_height, int(y_max + fh * padding))
            
            face_crop = img_raw[y1:y2, x1:x2]
            
            # --- SMART AVATAR SELECTION ---
            should_save_image = True
            if os.path.exists(target_path):
                try:
                    old_img = cv2.imread(target_path)
                    if old_img is not None:
                        old_h, old_w = old_img.shape[:2]
                        new_h, new_w = face_crop.shape[:2]
                        if new_w * new_h <= old_w * old_h:
                            should_save_image = False
                            print(f"ℹ️ Giữ nguyên Avatar cũ (Mới: {new_w}x{new_h} <= Cũ: {old_w}x{old_h})")
                        else:
                            print(f"🆙 Cập nhật Avatar chất lượng cao hơn ({old_w}x{old_h} -> {new_w}x{new_h})")
                except:
                    pass

            if should_save_image:
                cv2.imwrite(target_path, face_crop)
                print(f"✅ Đã lưu Avatar mới: {target_path}")

            # 2. Update Qdrant — ONNX Direct (Augmentation x4)
            variants = [
                ("orig", aligned_face),
                ("flip", cv2.flip(aligned_face, 1)),
                ("bright", cv2.convertScaleAbs(aligned_face, alpha=1.15, beta=20)),
                ("dark", cv2.convertScaleAbs(aligned_face, alpha=0.85, beta=-15))
            ]
            
            for var_name, var_img in variants:
                try:
                    embedding = arcface.get_embedding(var_img, normalize=True)
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding.tolist(),
                                payload={"student_id": mssv, "variant": var_name}
                            )
                        ]
                    )
                except:
                    pass
            
            print(f"✅ Đã thêm 4 variants vào Qdrant cho {mssv}.")

            # 3. Di chuyển ảnh vào processed
            try:
                dest_dir = os.path.join(PROCESSED_DIR, mssv)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = os.path.join(dest_dir, f"{timestamp}_{filename}")
                
                shutil.move(file_path, dest_path)
                print(f"📦 Đã lưu trữ ảnh gốc vào: {dest_path}")
            except Exception as e:
                print(f"⚠️ Không thể lưu trữ file {filename}: {e}")
            
            # Xóa thư mục rỗng
            parent_dir = os.path.dirname(file_path)
            if not os.listdir(parent_dir) and parent_dir != COLLECTED_DIR:
                try:
                    os.rmdir(parent_dir)
                except: pass
                
            # 4. Cập nhật SQLite
            try:
                conn = sqlite3.connect('student_info.db')
                cursor = conn.cursor()
                
                cursor.execute("SELECT id FROM students WHERE id = ?", (mssv,))
                if cursor.fetchone() is None:
                    schedule_templates = [
                        ('TMG301 (12:30-14:45)', '314'),
                        ('SEG301 (12:30-14:45)', '318'),
                        ('DPL302m (12:30-14:45)', '305'),
                        ('IOT102 (07:30-09:45)', '205'),
                        ('MAD101 (10:00-12:15)', '401')
                    ]
                    sched, room = random.choice(schedule_templates)
                    name = f"Sinh viên {mssv}"
                    
                    cursor.execute("INSERT INTO students (id, name, schedule, room) VALUES (?, ?, ?, ?)", 
                                (mssv, name, sched, room))
                    conn.commit()
                    print(f"📝 Đã thêm {mssv} vào SQLite database.")
                
                conn.close()
            except Exception as db_err:
                print(f"⚠️ Lỗi cập nhật SQLite: {db_err}")

            count_success += 1

        except Exception as e:
            print(f"❌ Lỗi xử lý {filename}: {e}")

    face_mesh.close()

    print("\n" + "="*50)
    print(f"🎉 Hoàn tất! Đã xử lý thành công {count_success}/{len(image_files)} ảnh.")

if __name__ == "__main__":
    process_collected_images()