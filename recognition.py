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
        search_res = state.db.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=5
        ).points
        
        found = False
        if search_res:
            best_match = search_res[0]
            score = best_match.score
            current_sid = best_match.payload['student_id']
            
            # Log top-K results for debugging
            for i, res in enumerate(search_res):
                print(f"  #{i+1}: {res.payload['student_id']} (var: {res.payload.get('variant','?')}) - Score: {res.score:.4f}")
            
            accepted_sid = None
            SCORE_THRESHOLD = 0.45
            
            # --- STEP A: Count votes (consensus) ---
            sid_votes = {}
            for res in search_res:
                sid = res.payload['student_id']
                sid_votes[sid] = sid_votes.get(sid, 0) + 1
            
            my_votes = sid_votes.get(current_sid, 0)
            total_results = len(search_res)
            
            # Find strongest competitor
            competitor_score = 0
            competitor_sid = None
            for res in search_res[1:]:
                if res.payload['student_id'] != current_sid:
                    competitor_score = res.score
                    competitor_sid = res.payload['student_id']
                    break
            gap = score - competitor_score if competitor_score > 0 else 1.0
            
            print(f"  📊 Votes: {current_sid}={my_votes}/{total_results} | Gap: {gap:.4f} | Competitor: {competitor_sid}")
            
            # --- STEP B: Decision logic ---
            reject_reason = None
            
            # B1: Score too low → always reject
            if score < SCORE_THRESHOLD:
                reject_reason = f"low_score({score:.4f}<{SCORE_THRESHOLD})"
            
            # B2: Consensus disagrees with top score → reject
            elif total_results >= 3:
                top_vote_sid = max(sid_votes, key=sid_votes.get)
                if top_vote_sid != current_sid and sid_votes[top_vote_sid] >= 3:
                    reject_reason = f"consensus_disagree(top={current_sid}, majority={top_vote_sid} {sid_votes[top_vote_sid]}/{total_results})"
            
            # B3: Strong consensus (>=3 votes) → SKIP ambiguity, accept
            #     This is the key fix: 4/5 votes for same person = confident match
            if reject_reason is None and my_votes >= 3:
                accepted_sid = current_sid
                print(f"🎯 ACCEPTED (strong consensus {my_votes}/{total_results}): {accepted_sid} (Score: {score:.4f})")
            
            # B4: Weak consensus (1-2 votes) → apply ambiguity check
            elif reject_reason is None:
                if gap < 0.03 and score < 0.60:
                    reject_reason = f"ambiguous(votes={my_votes}/{total_results}, gap={gap:.4f})"
                else:
                    accepted_sid = current_sid
                    print(f"🎯 ACCEPTED (gap ok): {accepted_sid} (Score: {score:.4f}, Gap: {gap:.4f})")
            
            if reject_reason:
                print(f"❌ REJECTED {current_sid}: {reject_reason}")
            
            if accepted_sid:
                with state.lock:
                    if state.last_recognized_sid == accepted_sid:
                        state.consecutive_match_count += 1
                    else:
                        state.last_recognized_sid = accepted_sid
                        state.consecutive_match_count = 1
                    
                    print(f"🔄 Khớp lần {state.consecutive_match_count}/2 cho ID: {accepted_sid}")
                    FAST_PASS_THRESHOLD = 0.68
                    is_very_sure = score > FAST_PASS_THRESHOLD
                    
                    if state.consecutive_match_count >= 2 or is_very_sure:
                        print(f"✅ XÁC NHẬN CHÍNH XÁC{' (FAST)' if is_very_sure else ''}: {accepted_sid} (Score: {score:.4f})")
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
            # Rejection was already logged above
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
