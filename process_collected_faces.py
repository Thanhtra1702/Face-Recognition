import os
import cv2
import shutil
import numpy as np
from deepface import DeepFace
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import sys
import datetime
import sqlite3
import random

sys.stdout.reconfigure(encoding='utf-8')

COLLECTED_DIR = "collected_faces"
PROCESSED_DIR = "collected_faces/processed"
DATABASE_DIR = "database"
COLLECTION_NAME = "student_faces"

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

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))
# -----------------------------------------------------------------------

def process_collected_images():
    if not os.path.exists(COLLECTED_DIR):
        print(f"Không tìm thấy thư mục {COLLECTED_DIR}")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # Lấy danh sách ảnh (bao gồm cả trong subfolders)
    image_files = []
    for root, dirs, files in os.walk(COLLECTED_DIR):
        # Bỏ qua thư mục 'processed'
        if 'processed' in root.replace('\\', '/').split('/'):
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("Không có ảnh nào cần xử lý.")
        return

    print(f"🔍 Tìm thấy {len(image_files)} ảnh cần xử lý...")
    
    # Init Qdrant Client một lần
    client = QdrantClient(host="localhost", port=6333)

    count_success = 0
    
    for file_path in image_files:
        filename = os.path.basename(file_path)
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        
        # LOGIC CHUẨN: Chỉ nhận ảnh trong Folder con (collected_faces/MSSV/...)
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
        
        # Tạo bản CLAHE chỉ cho AI đọc (không ghi đè ảnh gốc)
        img_for_ai = preprocess_frame(img_raw)

        try:
            # --- DETECT & EXTRACT FACE (Đồng bộ pipeline với recognition.py) ---
            # Dùng extract_faces để lấy aligned face + facial_area
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=img_for_ai,
                    detector_backend="mediapipe",
                    enforce_detection=True,
                    align=True
                )
            except:
                # Nếu Mediapipe xịt, dùng RetinaFace (Cực kỳ chính xác cho ảnh tĩnh)
                face_objs = DeepFace.extract_faces(
                    img_path=img_for_ai,
                    detector_backend="retinaface",
                    enforce_detection=True,
                    align=True
                )

            if not face_objs:
                print("❌ Không phát hiện khuôn mặt.")
                continue

            # Logic chọn khuôn mặt tốt nhất (Diện tích + Vị trí trung tâm)
            img_height, img_width = img_raw.shape[:2]
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            
            best_face_obj = None
            best_score = -1
            
            for face_obj in face_objs:
                fa = face_obj['facial_area']
                area = fa['w'] * fa['h']
                face_center_x = fa['x'] + fa['w'] / 2
                face_center_y = fa['y'] + fa['h'] / 2
                distance = ((face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2)**0.5
                max_distance = (img_width**2 + img_height**2)**0.5
                distance_score = 1 - (distance / max_distance)
                max_area = img_width * img_height
                area_score = area / max_area
                total_score = 0.7 * area_score + 0.3 * distance_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_face_obj = face_obj
            
            # Lấy aligned face cho AI (đã detect + align, đồng bộ recognition.py dòng 28-30)
            aligned_face = best_face_obj['face']
            if aligned_face.max() <= 1.0:
                aligned_face = (aligned_face * 255).astype(np.uint8)
            
            # Crop & Save Avatar
            target_path = os.path.join(DATABASE_DIR, f"{mssv}.jpg")
            
            # Tính toán vùng crop có padding (30% padding cho avatar thoáng và đẹp)
            facial_area = best_face_obj['facial_area']
            padding = 0.3
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            x_pad, y_pad = int(w * padding), int(h * padding)
            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(img_width, x + w + x_pad)
            y2 = min(img_height, y + h + y_pad)
            
            # Crop từ ảnh GỐC (không CLAHE) để lưu avatar đẹp tự nhiên
            face_crop = img_raw[y1:y2, x1:x2]
            
            # --- LOGIC CHỌN ẢNH TỐT NHẤT (SMART AVATAR SELECTION) ---
            should_save_image = True
            if os.path.exists(target_path):
                # Nếu ảnh đã tồn tại, so sánh chất lượng (Dựa trên độ phân giải)
                try:
                    old_img = cv2.imread(target_path)
                    if old_img is not None:
                        old_h, old_w = old_img.shape[:2]
                        new_h, new_w = face_crop.shape[:2]
                        
                        old_area = old_w * old_h
                        new_area = new_w * new_h
                        
                        # Chỉ thay thế nếu ảnh mới LỚN HƠN ảnh cũ (Rõ nét hơn)
                        if new_area <= old_area:
                            should_save_image = False
                            print(f"ℹ️ Giữ nguyên Avatar cũ (Mới: {new_w}x{new_h} <= Cũ: {old_w}x{old_h})")
                        else:
                            print(f"🆙 Cập nhật Avatar chất lượng cao hơn ({old_w}x{old_h} -> {new_w}x{new_h})")
                except:
                    pass # Lỗi đọc ảnh cũ -> Cứ ghi đè cho chắc

            if should_save_image:
                cv2.imwrite(target_path, face_crop)
                print(f"✅ Đã lưu Avatar mới: {target_path}")
            # --------------------------------------------------------

            # 2. Update Qdrant (Augmentation x8 trên aligned face)
            # Dùng aligned_face (đã detect + align + CLAHE) cho augmentation
            # Sau đó represent(skip) → đồng bộ hoàn toàn với recognition.py
            variants = [
                ("orig", aligned_face),
                ("flip", cv2.flip(aligned_face, 1)),
                ("rot_p5", rotate_image(aligned_face, 5)),
                ("rot_m5", rotate_image(aligned_face, -5)),
                ("bright", cv2.convertScaleAbs(aligned_face, alpha=1.2, beta=30)),
                ("dark", cv2.convertScaleAbs(aligned_face, alpha=0.8, beta=-20)),
                ("contrast", cv2.convertScaleAbs(aligned_face, alpha=1.5, beta=0)),
                ("blur", cv2.GaussianBlur(aligned_face, (3, 3), 0))
            ]
            
            import uuid
            for var_name, var_img in variants:
                try:
                    # represent(skip): Không detect lại, đồng bộ với recognition.py
                    results_var = DeepFace.represent(
                        img_path=var_img,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="skip",
                        align=False
                    )
                    if results_var:
                        embedding = results_var[0]['embedding']
                        point_id = str(uuid.uuid4())
                        client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[
                                PointStruct(
                                    id=point_id,
                                    vector=embedding,
                                    payload={"student_id": mssv, "variant": var_name}
                                )
                            ]
                        )
                except:
                    pass
            
            print(f"✅ Đã thêm 8 variants vào Qdrant cho {mssv}.")

            # 3. Lưu trữ: Di chuyển ảnh vào thư mục processed thay vì xóa (để đối soát)
            try:
                # Tạo cấu trúc thư mục MSSV bên trong processed
                dest_dir = os.path.join(PROCESSED_DIR, mssv)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                # Di chuyển file (thêm timestamp để tránh trùng tên nếu 1 MSSV có nhiều ảnh)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = os.path.join(dest_dir, f"{timestamp}_{filename}")
                
                shutil.move(file_path, dest_path)
                print(f"📦 Đã lưu trữ ảnh gốc vào: {dest_path}")
            except Exception as e:
                print(f"⚠️ Không thể lưu trữ file {filename}: {e}")
            
            # Xóa thư mục rỗng trong collected_faces nếu cần
            parent_dir = os.path.dirname(file_path)
            if not os.listdir(parent_dir) and parent_dir != COLLECTED_DIR:
                try:
                    os.rmdir(parent_dir)
                except: pass
                
            # 4. Tự động cập nhật thông tin vào SQLite (Cực kỳ quan trọng)
            try:
                conn = sqlite3.connect('student_info.db')
                cursor = conn.cursor()
                
                # Kiểm tra xem sinh viên đã có trong hệ thống chưa
                cursor.execute("SELECT id FROM students WHERE id = ?", (mssv,))
                if cursor.fetchone() is None:
                    # Nếu chưa có, tạo dữ liệu mẫu giống setup_database.py
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

    print("\n" + "="*50)
    print(f"🎉 Hoàn tất! Đã xử lý thành công {count_success}/{len(image_files)} ảnh.")

if __name__ == "__main__":
    process_collected_images()