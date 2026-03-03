import cv2
import numpy as np
import datetime
import time
from deepface import DeepFace

COLLECTION_NAME = "student_faces"

def preprocess_frame(frame):
    """Tiền xử lý ảnh (Khử nhiễu + Cân bằng sáng)"""
    try:
        # 1. Khử nhiễu nhẹ
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        # 2. Cân bằng sáng (CLAHE)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Giảm xuống 2.0 cho tự nhiên
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return frame

def run_recognition_async(face_crop, full_frame, state, x_min, y_min, x_max, y_max):
    """Xử lý nhận diện khuôn mặt bất đồng bộ"""
    try:
        # --- 0. PREPROCESS ---
        processed_crop = preprocess_frame(face_crop)
        input_frame = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
        
        # --- 1. DETECT & EXTRACT FACE ---
        face_objs = DeepFace.extract_faces(
            img_path=input_frame,
            detector_backend="mediapipe",
            enforce_detection=False, 
            align=True
        )
        
        if not face_objs:
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0
            return

        current_face = face_objs[0]["face"]
        if current_face.max() <= 1.0:
            current_face = (current_face * 255).astype(np.uint8)

        # --- 2. GET EMBEDDING ---
        results = DeepFace.represent(
            img_path=current_face,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True
        )
        
        if not results: 
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0
            return
            
        embedding = results[0]["embedding"]
        
        # --- 3. SEARCH DATABASE ---
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
            print(f"🎯 Top 1: {current_sid} - Score: {score:.4f}")
            
            accepted_sid = None
            is_passing_score = score > 0.45
            is_ambiguous = False
            competitor_score = 0
            for res in search_res[1:]:
                if res.payload['student_id'] != current_sid:
                    competitor_score = res.score
                    break
            
            if competitor_score > 0:
                gap = score - competitor_score
                if gap < 0.02 and score < 0.65: 
                    is_ambiguous = True
                    print(f"⚠️ Nhập nhằng giữa {current_sid} và người khác (Gap: {gap:.4f})")
            
            if is_passing_score and not is_ambiguous:
                accepted_sid = current_sid
            
            if accepted_sid:
                with state.lock:
                    if state.last_recognized_sid == accepted_sid:
                        state.consecutive_match_count += 1
                    else:
                        state.last_recognized_sid = accepted_sid
                        state.consecutive_match_count = 1
                    
                    print(f"🔄 Khớp lần {state.consecutive_match_count}/2 cho ID: {accepted_sid}")
                    is_very_sure = score > 0.65
                    
                    if state.consecutive_match_count >= 2 or is_very_sure:
                        print(f"✅ XÁC NHẬN CHÍNH XÁC{' (FAST)' if is_very_sure else ''}: {accepted_sid}")
                        name, sch, room = state.db.get_student_info(accepted_sid)
                        state.student_data = {
                            "name": name,
                            "student_id": accepted_sid,
                            "schedule": sch,
                            "room": room,
                            "checkin_time": datetime.datetime.now().strftime("%H:%M %d/%m")
                        }
                        state.clean_snapshot = full_frame.copy()
                        
                        display_frame = full_frame.copy()
                        t, l = 3, 40
                        cv2.line(display_frame, (x_min, y_min), (x_min + l, y_min), (73, 132, 30), t)
                        cv2.line(display_frame, (x_min, y_min), (x_min, y_min + l), (73, 132, 30), t)
                        cv2.line(display_frame, (x_max, y_min), (x_max - l, y_min), (73, 132, 30), t)
                        cv2.line(display_frame, (x_max, y_min), (x_max, y_min + l), (73, 132, 30), t)
                        cv2.line(display_frame, (x_min, y_max), (x_min + l, y_max), (73, 132, 30), t)
                        cv2.line(display_frame, (x_min, y_max), (x_min, y_max - l), (73, 132, 30), t)
                        cv2.line(display_frame, (x_max, y_max), (x_max - l, y_max), (73, 132, 30), t)
                        cv2.line(display_frame, (x_max, y_max), (x_max, y_max - l), (73, 132, 30), t)
                        
                        state.frame = display_frame
                        state.status = "CONFIRM"
                        state.progress = 100
                        state.consecutive_match_count = 0 
                        found = True
                    else:
                        state.status = "PROCESSING"
                        state.progress = 95
                        found = True
            else:
                print(f"❌ Low Score (< 0.45) hoặc Ambiguous")
        else:
            print("❌ DB Empty")
        
        if not found:
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0

    except Exception as e:
        print(f"🔥 AI Exception: {e}")
        with state.lock:
            state.status = "SCANNING"
            state.progress = 0
    
    if state.status == "PROCESSING":
        with state.lock:
            state.process_start_time = time.time() - 0.4 
