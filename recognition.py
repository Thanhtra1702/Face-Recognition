import cv2
import numpy as np
import datetime
import time
from deepface import DeepFace

COLLECTION_NAME = "student_faces"

def is_blurry(image, threshold=60): # Tăng nhẹ ngưỡng để lọc gắt hơn
    """Kiểm tra ảnh có bị nhòe (motion blur) hay không"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def get_aligned_face(frame, landmarks, target_size=(112, 112)):
    """
    Sử dụng 468 landmarks của Mediapipe để align khuôn mặt cực nhanh bằng Affine Transform.
    Giúp bỏ qua bước DeepFace.extract_faces (tiết kiệm 100-200ms).
    """
    h, w = frame.shape[:2]
    
    # Lấy tọa độ mắt trái và mắt phải (trung tâm)
    #landmark idx: mắt trái (33, 133), mắt phải (362, 263)
    left_eye = np.mean([ (landmarks[33].x * w, landmarks[33].y * h), (landmarks[133].x * w, landmarks[133].y * h) ], axis=0)
    right_eye = np.mean([ (landmarks[362].x * w, landmarks[362].y * h), (landmarks[263].x * w, landmarks[263].y * h) ], axis=0)
    
    # Tính góc xoay
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Tính tâm giữa 2 mắt
    eye_center = ( (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2 )
    
    # Ma trận xoay
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    
    # Xoay toàn bộ frame để mặt thẳng đứng
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # Crop lại vùng mặt từ frame đã xoay
    # Sử dụng bounding box đã có nhưng lấy rộng ra một chút
    # (Để đơn giản, trong bước này recognition.py sẽ nhận ảnh đã được align từ app.py)
    return rotated

def run_recognition_async(face_crop, full_frame, state, x_min, y_min, x_max, y_max):
    """Xử lý nhận diện khuôn mặt SIÊU TỐC"""
    try:
        # --- 0. SHARPNESS CHECK ---
        blurry, val = is_blurry(face_crop)
        if blurry:
            with state.lock: state.status = "SCANNING"
            return
        
        # --- 1. PREPARE IMAGE ---
        # face_crop lúc này đã được app.py align sơ bộ hoặc là crop chất lượng cao
        input_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize về đúng chuẩn ArcFace (112x112)
        input_img = cv2.resize(input_img, (112, 112))

        # --- 2. GET EMBEDDING (detector='skip' is the key for speed) ---
        results = DeepFace.represent(
            img_path=input_img,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False # Đã tự align bên ngoài hoặc skip để nhanh
        )
        
        if not results: 
            with state.lock: state.status = "SCANNING"
            return
            
        embedding = results[0]["embedding"]
        
        # --- 3. SEARCH DATABASE (Top-3 for speed) ---
        search_res = state.db.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=3
        ).points
        
        found = False
        if search_res:
            best_match = search_res[0]
            score = best_match.score
            current_sid = best_match.payload['student_id']
            
            # --- CONSENSUS VOTING ---
            votes = {}
            for res in search_res:
                sid = res.payload['student_id']
                votes[sid] = votes.get(sid, 0) + 1
            
            top_voter = max(votes, key=votes.get)
            vote_count = votes[top_voter]
            
            # Tính Gap với người khác
            competitor_score = 0
            for res in search_res[1:]:
                if res.payload['student_id'] != current_sid:
                    competitor_score = res.score
                    break
            gap = score - competitor_score if competitor_score > 0 else 1.0

            accepted_sid = None
            # Ngưỡng tối ưu: 0.50
            if score > 0.50 and (current_sid == top_voter) and (vote_count >= 2):
                # Kiểm tra Ambiguity Gap gắt hơn cho các trường hợp điểm thấp
                if score < 0.65 and gap < 0.03:
                    print(f"⚠️ Nhập nhằng (Gap: {gap:.4f}) - Bỏ qua")
                else:
                    accepted_sid = current_sid
            
            if accepted_sid:
                with state.lock:
                    if state.last_recognized_sid == accepted_sid:
                        state.consecutive_match_count += 1
                    else:
                        state.last_recognized_sid = accepted_sid
                        state.consecutive_match_count = 1
                    
                    # CỰC NHANH: Fast pass > 0.72 + Consensus 3/3
                    is_very_sure = (score > 0.72) and (vote_count >= 3)
                    required_matches = 2 
                    
                    if state.consecutive_match_count >= required_matches or is_very_sure:
                        print(f"✅ [{accepted_sid}] Score: {score:.3f} | Gap: {gap:.3f} | Votes: {vote_count} {'(FAST)' if is_very_sure else ''}")
                        
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
                # Log lý do từ chối để debug nhanh
                if score > 0.50:
                    print(f"❌ Rejected {current_sid}: Score={score:.3f}, Votes={vote_count}, Gap={gap:.3f}")

        if not found:
            with state.lock: state.status = "SCANNING"; state.progress = 0

    except Exception as e:
        print(f"🔥 AI Exception: {e}")
        with state.lock: state.status = "SCANNING"
    
    if state.status == "PROCESSING":
        with state.lock: state.process_start_time = time.time() - 0.4 
