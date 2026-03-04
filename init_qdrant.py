import os
import sys
import cv2  # Added
import numpy as np # Added
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from deepface import DeepFace
from tqdm import tqdm

# Cấu hình
COLLECTION_NAME = "student_faces"
IMAGE_DIR = "./database"
MODEL_NAME = "ArcFace"

# --- AI ENHANCEMENT HELPERS (Copy từ app.py để đồng bộ) ---
def preprocess_frame(frame):
    """Cân bằng sáng và khử nhiễu để AI dễ đọc hơn"""
    try:
        # 1. Khử nhiễu nhẹ
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 2. Chuyển sang LAB để cân bằng sáng (CLAHE)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    except:
        return frame
# Helper xoay ảnh
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def init_qdrant():
    # Khởi tạo client Qdrant lưu trữ local
    client = QdrantClient(host="localhost", port=6333)

    # Tạo lại collection
    print(f"Đang tạo lại collection '{COLLECTION_NAME}' với model {MODEL_NAME}...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    if not image_files:
        print("Không tìm thấy ảnh nào trong thư mục database!")
        return

    print(f"Bắt đầu xử lý {len(image_files)} ảnh (Auto-Crop + Augmentation)...")
    
    points = []
    import uuid

    for filename in tqdm(image_files):
        student_id = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            # 1. Đọc và tiền xử lý ảnh
            img_raw = cv2.imread(img_path)
            if img_raw is None: continue
            h_orig, w_orig = img_raw.shape[:2]
            img_processed = preprocess_frame(img_raw)
            
            # 2. Extract aligned face (Dùng Mediapipe, fallback RetinaFace)
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=img_processed,
                    detector_backend="mediapipe",
                    enforce_detection=True,
                    align=True
                )
            except:
                face_objs = DeepFace.extract_faces(
                    img_path=img_processed,
                    detector_backend="retinaface",
                    enforce_detection=True,
                    align=True
                )
            
            if not face_objs:
                print(f"⚠️ Không phát hiện khuôn mặt trong {filename}")
                continue
            
            # 3. TỰ ĐỘNG CẬP NHẬT ẢNH DATABASE (Smart Portrait Crop)
            # Chúng ta crop từ img_raw để giữ chất lượng gốc, không lấy bản đã CLAHE
            fa = face_objs[0]['facial_area']
            p = 0.3 # 30% padding cho avatar thoáng
            x1 = max(0, int(fa['x'] - fa['w'] * p))
            y1 = max(0, int(fa['y'] - fa['h'] * p))
            x2 = min(w_orig, int(fa['x'] + fa['w'] + fa['w'] * p))
            y2 = min(h_orig, int(fa['y'] + fa['h'] + fa['h'] * p))
            
            portrait_img = img_raw[y1:y2, x1:x2]
            if portrait_img.size > 0:
                cv2.imwrite(img_path, portrait_img)
            
            # 4. Lấy aligned face (đã detect + align) để tạo vector
            aligned_face = face_objs[0]['face']
            if aligned_face.max() <= 1.0:
                aligned_face = (aligned_face * 255).astype(np.uint8)
            
            # 5. Tạo các biến thể (Augmentation x3 — KHÔNG dùng flip vì gây false positive)
            # Flip variant bị loại: mặt lật của người A dễ match nhầm mặt gốc người B
            variants = [
                ("orig", aligned_face),
                ("bright", cv2.convertScaleAbs(aligned_face, alpha=1.15, beta=20)),
                ("dark", cv2.convertScaleAbs(aligned_face, alpha=0.85, beta=-15)),
            ]
            
            # 6. Tạo vector và lưu vào Qdrant (detect_backend='skip')
            for var_name, var_img in variants:
                try:
                    results = DeepFace.represent(
                        img_path=var_img, 
                        model_name=MODEL_NAME, 
                        detector_backend="skip",
                        align=False,
                        enforce_detection=False
                    )
                    
                    if results:
                        embedding = results[0]["embedding"]
                        points.append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "student_id": student_id, 
                                "filename": filename,
                                "variant": var_name
                            }
                        ))
                except:
                    pass
                    
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {e}")

    # Push dữ liệu lên Qdrant
    if points:
        # Chia nhỏ batches để tránh lỗi memory nếu quá nhiều
        batch_size = 100
        print(f"Đang lưu {len(points)} vector vào DB...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        print(f"✅ Đã lưu {len(points)} vector vào Qdrant thành công!")
    else:
        print("Không có vector nào được tạo.")

if __name__ == "__main__":
    init_qdrant()
