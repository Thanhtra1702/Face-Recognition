import cv2
import numpy as np
import datetime
import time
from arcface_onnx import get_arcface_model

COLLECTION_NAME = "student_faces"

def is_blurry(image, threshold=60):
    """Kiểm tra ảnh có bị nhòe (motion blur) hay không"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def get_standard_aligned_face(frame, landmarks, target_size=(112, 112)):
    """
    Standard Face Alignment (Affine Transform) mapping 5 landmarks to 112x112 template.
    This is the standard for ArcFace/InsightFace to achieve SOTA accuracy.
    """
    h, w = frame.shape[:2]
    
    # 1. Trích xuất 5 điểm landmarks chính
    # Mắt trái, Mắt phải, Mũi, Khóe miệng trái, Khóe miệng phải
    l_eye = np.mean([ (landmarks[33].x * w, landmarks[33].y * h), (landmarks[133].x * w, landmarks[133].y * h) ], axis=0)
    r_eye = np.mean([ (landmarks[362].x * w, landmarks[362].y * h), (landmarks[263].x * w, landmarks[263].y * h) ], axis=0)
    nose = (landmarks[1].x * w, landmarks[1].y * h)
    l_mouth = (landmarks[61].x * w, landmarks[61].y * h)
    r_mouth = (landmarks[291].x * w, landmarks[291].y * h)
    
    src = np.array([l_eye, r_eye, nose, l_mouth, r_mouth], dtype=np.float32)
    
    # 2. Template tọa độ chuẩn cho ArcFace (112x112)
    dst = np.array([
        [30.2946, 51.6963], # L-Eye
        [65.5318, 51.5014], # R-Eye
        [48.0252, 71.7366], # Nose
        [33.5493, 92.3655], # L-Mouth
        [62.7299, 92.2041]  # R-Mouth
    ], dtype=np.float32)
    
    # 3. Tính toán Ma trận Affine
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:
        return cv2.resize(frame, target_size)
    
    aligned = cv2.warpAffine(frame, M, target_size, borderMode=cv2.BORDER_CONSTANT)
    return aligned

def run_recognition_async(face_crop, full_frame, state, x_min, y_min, x_max, y_max):
    """Xử lý nhận diện khuôn mặt — ONNX Direct Inference (5-10x faster)"""
    try:
        # --- 0. SHARPNESS CHECK ---
        blurry, val = is_blurry(face_crop)
        if blurry:
            with state.lock: state.status = "SCANNING"
            return
        
        # --- 1. GET ARCFACE MODEL (Singleton — chỉ load 1 lần) ---
        arcface = get_arcface_model()
        
        # --- 2. TEST-TIME AUGMENTATION (TTA) — ONNX Direct ---
        # Tạo 3 phiên bản ảnh để lấy embedding trung bình -> Cực kỳ ổn định
        
        # Variant 1: Original (BGR, đã aligned)
        # face_crop đã là ảnh aligned 112x112 từ app.py
        
        # Variant 2: CLAHE (Handle shadows)
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        img_clahe = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
        
        # Variant 3: Bright (Handle low light)
        img_bright = cv2.convertScaleAbs(face_crop, alpha=1.2, beta=10)
        
        # --- ONNX DIRECT INFERENCE (thay DeepFace.represent) ---
        # Mỗi lần gọi chỉ ~10-30ms thay vì 100-300ms qua DeepFace
        variants = [face_crop, img_clahe, img_bright]
        embeddings = []
        
        for var_img in variants:
            try:
                vec = arcface.get_embedding(var_img, normalize=True)
                embeddings.append(vec)
            except Exception:
                pass
        
        if not embeddings:
            with state.lock: state.status = "SCANNING"
            return
            
        # 3. AVG EMBEDDING (Robust Vector)
        final_embedding = np.mean(embeddings, axis=0)
        # Re-normalize trung bình
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm
        final_embedding = final_embedding.tolist()
        
        # --- 4. SEARCH DATABASE (Limit 5 for Voting) ---
        search_res = state.db.client.query_points(
            collection_name=COLLECTION_NAME,
            query=final_embedding,
            limit=5
        ).points
        
        found = False
        if search_res:
            best_match = search_res[0]
            score = best_match.score
            current_sid = best_match.payload['student_id']
            
            # --- ADVANCED CONSENSUS LOGIC ---
            votes = {}
            for res in search_res:
                sid = res.payload['student_id']
                votes[sid] = votes.get(sid, 0) + 1
            
            top_voter = max(votes, key=votes.get)
            vote_count = votes[top_voter]
            
            # Gap Check (với người khác - competitor)
            competitor_score = 0
            for res in search_res:
                if res.payload['student_id'] != current_sid:
                    competitor_score = res.score
                    break
            gap = score - competitor_score if competitor_score > 0 else 1.0

            accepted_sid = None
            # SOTA Threshold: Nâng ngưỡng lên 0.55 cho ổn định
            if score > 0.55 and (current_sid == top_voter) and (vote_count >= 2):
                # Gap Protection: Nếu điểm thấp (<0.70) thì Gap phải rõ rệt (>0.05)
                if score < 0.70 and gap < 0.05:
                    print(f"⚠️ Nhập nhằng (Gap: {gap:.4f}) - Đang tìm người tốt nhất...")
                else:
                    accepted_sid = current_sid
            
            if accepted_sid:
                with state.lock:
                    if state.last_recognized_sid == accepted_sid:
                        state.consecutive_match_count += 1
                    else:
                        state.last_recognized_sid = accepted_sid
                        state.consecutive_match_count = 1
                    
                    # SOTA FAST PASS: Score cao (0.75) + Đa số áp đảo (3/5)
                    is_very_sure = (score > 0.75) and (vote_count >= 3)
                    required_matches = 2 
                    
                    if state.consecutive_match_count >= required_matches or is_very_sure:
                        print(f"✅ [{accepted_sid}] Score: {score:.3f} | Gap: {gap:.3f} | Votes: {vote_count}/5 {'(SOTA FAST)' if is_very_sure else ''}")
                        
                        name, sch, room = state.db.get_student_info(accepted_sid)
                        state.student_data = {
                            "name": name, "student_id": accepted_sid,
                            "schedule": sch, "room": room,
                            "checkin_time": datetime.datetime.now().strftime("%H:%M %d/%m")
                        }
                        state.clean_snapshot = full_frame.copy()
                        state.status = "CONFIRM"
                        state.progress = 100
                        state.consecutive_match_count = 0 
                        found = True
                    else:
                        state.status = "PROCESSING"; state.progress = 95; found = True
            else:
                if score > 0.45:
                    print(f"❌ Rejected {current_sid}: Score={score:.3f}, Votes={vote_count}/5, Gap={gap:.3f}")

        if not found:
            with state.lock: state.status = "SCANNING"; state.progress = 0

    except Exception as e:
        print(f"🔥 ONNX AI Exception: {e}")
        with state.lock: state.status = "SCANNING"
    
    if state.status == "PROCESSING":
        with state.lock: state.process_start_time = time.time() - 0.4
