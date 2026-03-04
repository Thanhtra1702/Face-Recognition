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
        # --- 0. CONVERT (preprocess already done in app.py) ---
        input_frame = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Downsize for faster SSD detection (max 300px)
        h_crop, w_crop = input_frame.shape[:2]
        max_dim = max(h_crop, w_crop)
        if max_dim > 300:
            scale = 300 / max_dim
            input_frame = cv2.resize(input_frame, (int(w_crop * scale), int(h_crop * scale)))
        
        # --- 1. DETECT & ALIGN FACE (ssd: fast + accurate) ---
        face_objs = DeepFace.extract_faces(
            img_path=input_frame,
            detector_backend="ssd",
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
        # Tăng limit lên 5 để consensus voting (bình chọn đa số)
        search_limit = 5
        search_res = state.db.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=search_limit
        ).points
        
        found = False
        if search_res:
            best_match = search_res[0]
            score = best_match.score
            current_sid = best_match.payload['student_id']
            print(f"🎯 Top 1: {current_sid} - Score: {score:.4f}")
            
            # --- CONSENSUS VOTING ---
            # Đếm số lần mỗi ID xuất hiện trong Top Top-K
            votes = {}
            for res in search_res:
                sid = res.payload['student_id']
                votes[sid] = votes.get(sid, 0) + 1
            
            top_voter = max(votes, key=votes.get)
            vote_count = votes[top_voter]
            
            accepted_sid = None
            # Ngưỡng mới: 0.55 thay vì 0.45
            is_passing_score = score > 0.55
            is_ambiguous = False
            
            # Gap Check: So sánh với người khác (competitor)
            competitor_score = 0
            for res in search_res[1:]:
                if res.payload['student_id'] != current_sid:
                    competitor_score = res.score
                    break
            
            if competitor_score > 0:
                gap = score - competitor_score
                # Ngưỡng gap mới: 0.05 và score < 0.70
                if gap < 0.05 and score < 0.70: 
                    is_ambiguous = True
                    print(f"⚠️ Nhập nhằng giữa {current_sid} và người khác (Gap: {gap:.4f})")
            
            # Consensus Check: ID đạt score cao nhất phải là ID có đa số vote
            is_consensus_met = (current_sid == top_voter) and (vote_count >= 2)
            if vote_count == 1 and score < 0.70:
                is_consensus_met = False # Match quá yếu, chỉ có 1 variant khớp
            
            if is_passing_score and not is_ambiguous and is_consensus_met:
                accepted_sid = current_sid
            
            if accepted_sid:
                with state.lock:
                    if state.last_recognized_sid == accepted_sid:
                        state.consecutive_match_count += 1
                    else:
                        state.last_recognized_sid = accepted_sid
                        state.consecutive_match_count = 1
                    
                    # Ngưỡng fast-pass mới: 0.75
                    is_very_sure = score > 0.75
                    required_matches = 3 # Cần 3 frame cho chắc chắn thay vì 2
                    
                    print(f"🔄 Khớp lần {state.consecutive_match_count}/{required_matches} cho ID: {accepted_sid}")
                    
                    if state.consecutive_match_count >= required_matches or is_very_sure:
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
                if not is_passing_score:
                    print(f"❌ Low Score (< 0.55)")
                elif is_ambiguous:
                    print(f"❌ Ambiguous (Gap < 0.05)")
                elif not is_consensus_met:
                    print(f"❌ No Consensus (Voted: {top_voter} x{vote_count})")
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
