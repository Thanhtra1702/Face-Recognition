from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time
import datetime
import os
import numpy as np
from deepface import DeepFace
from qdrant_client import QdrantClient
from pydantic import BaseModel

# Import DB handler từ module cũ
from kiosk_db import DatabaseHandler

import mediapipe as mp
import onnxruntime as ort

import signal
import sys

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# --- PRELOAD MODEL (Để khởi động nhanh hơn) ---
print("🚀 Đang tải model AI (FastAPI)...")
try:
    # Preload ArcFace model bằng cách tạo embedding giả
    dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
    DeepFace.represent(img_path=dummy_img, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
    print("✅ Model AI đã sẵn sàng!")
except Exception as e:
    print(f"❌ Error preloading model: {e}")

# --- QDRANT CLIENT ---
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "student_faces"

# --- ANTI-SPOOF MODULE ---
class AntiSpoof:
    def __init__(self, model_path="MiniFASNetV2.onnx"):
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name

    def predict(self, face_img):
        """Dự đoán liveness: Trả về score (càng cao càng giống người thật)"""
        try:
            # Resize về 80x80 theo chuẩn MiniFASNetV2
            resized = cv2.resize(face_img, (80, 80))
            # Chuẩn hóa (MiniFASNet dùng NCHW và chuẩn hóa cơ bản)
            img = resized.astype(np.float32)
            img = np.expand_dims(img, axis=0) # (1, 80, 80, 3)
            img = np.transpose(img, (0, 3, 1, 2)) # (1, 3, 80, 80)
            
            ort_inputs = {self.input_name: img}
            ort_outs = self.ort_session.run(None, ort_inputs)
            # Output thường là Softmax [Fake, Real]
            result = ort_outs[0][0]
            exp_res = np.exp(result - np.max(result))
            prob = exp_res / exp_res.sum()
            return prob[1] # Xác suất là Real
        except Exception as e:
            print(f"AntiSpoof Error: {e}")
            return 0.0

# --- GLOBAL STATE ---
class KioskState:
    def __init__(self):
        self.frame = None
        self.clean_snapshot = None # Bản ảnh cực sạch để lưu DB
        self.lock = threading.Lock()
        self.status = "SCANNING"  # SCANNING, PROCESSING, CONFIRM, SUCCESS
        self.progress = 0
        self.student_data = None 
        self.last_scan_time = 0
        self.process_start_time = 0
        self.db = DatabaseHandler()
        self.anti_spoof = AntiSpoof()
        self.running = True # Cờ kiểm soát vòng lặp
        self.pending_crop = None
        # Liveness State
        self.is_live = False
        self.fas_score = 0.0
        self.blink_count = 0
        self.last_blink_time = 0
        self.blink_threshold = 0.20
        # Verification State
        self.consecutive_match_count = 0
        self.last_recognized_sid = None
        self.is_near = False # Trạng thái khoảng cách mới
        
state = KioskState()

# Handle Ctrl+C
def signal_handler(sig, frame):
    print('\n👋 Đang tắt hệ thống (FastAPI) NGAY LẬP TỨC...')
    state.running = False
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- BLINK DETECTION HELPERS ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, w, h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((lm.x * w, lm.y * h))
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h_dist = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

def preprocess_frame(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return frame

def run_recognition_async(face_crop, full_frame, state, x_min, y_min, x_max, y_max):
    try:
        input_frame = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
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

# --- CAMERA THREAD ---
def camera_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while state.running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        current_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            screen_center_x, screen_center_y = w // 2, h // 2
            best_face_data = None
            max_focus_score = -1

            for face_landmarks in results.multi_face_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, cx), min(y_min, cy)
                    x_max, y_max = max(x_max, cx), max(y_max, cy)
                
                area = (x_max - x_min) * (y_max - y_min)
                face_center_x = (x_min + x_max) / 2
                face_center_y = (y_min + y_max) / 2
                dist_to_center = ((face_center_x - screen_center_x)**2 + (face_center_y - screen_center_y)**2)**0.5
                focus_score = area / (dist_to_center + 1) 
                
                if focus_score > max_focus_score:
                    max_focus_score = focus_score
                    best_face_data = (x_min, y_min, x_max, y_max)

            if best_face_data:
                x_min, y_min, x_max, y_max = best_face_data
                
                try:
                    fas_pad = int((x_max - x_min) * 0.1)
                    fx1, fy1 = max(0, x_min - fas_pad), max(0, y_min - fas_pad)
                    fx2, fy2 = min(w, x_max + fas_pad), min(h, y_max + fas_pad)
                    fas_face = raw_frame[fy1:fy2, fx1:fx2]
                    
                    if fas_face.size > 0:
                        if not hasattr(state, '_fas_frame_counter'): state._fas_frame_counter = 0
                        state._fas_frame_counter += 1
                        if state._fas_frame_counter % 3 == 0 or not state.is_live:
                            score = state.anti_spoof.predict(fas_face)
                            with state.lock:
                                state.fas_score = score
                    
                    face_landmarks = results.multi_face_landmarks[0]
                    ear_left = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
                    ear_right = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
                    ear = (ear_left + ear_right) / 2.0
                    
                    with state.lock:
                        if ear < state.blink_threshold:
                            if not hasattr(state, '_eye_closed'): state._eye_closed = True
                        else:
                            if hasattr(state, '_eye_closed') and state._eye_closed:
                                state.blink_count += 1
                                state._eye_closed = False
                                state.last_blink_time = time.time()
                        
                        is_pass_fas = state.fas_score > 0.99
                        is_pass_blink = (time.time() - state.last_blink_time < 4.0)
                        state.is_live = is_pass_fas or is_pass_blink
                except Exception as e:
                    print(f"Liveness Logic Error: {e}")
                
                face_width = x_max - x_min
                is_near_enough = face_width > 180 
                
                with state.lock:
                    state.is_near = is_near_enough
                
                color = (255, 255, 255)
                if is_near_enough: color = (33, 111, 242)
                if state.status == "CONFIRM": color = (73, 132, 30)
                
                t, l = 3, 40
                cv2.line(frame, (x_min, y_min), (x_min + l, y_min), color, t)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + l), color, t)
                cv2.line(frame, (x_max, y_min), (x_max - l, y_min), color, t)
                cv2.line(frame, (x_max, y_min), (x_max, y_min + l), color, t)
                cv2.line(frame, (x_min, y_max), (x_min + l, y_max), color, t)
                cv2.line(frame, (x_min, y_max), (x_min, y_max - l), color, t)
                cv2.line(frame, (x_max, y_max), (x_max - l, y_max), color, t)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - l), color, t)
                
                if is_near_enough and state.is_live and state.status == "SCANNING" and (current_time - state.last_scan_time > 1.0):
                    pad_w = int((x_max - x_min) * 0.4)
                    pad_h = int((y_max - y_min) * 0.4)
                    x1, y1 = max(0, x_min - pad_w), max(0, y_min - pad_h)
                    x2, y2 = min(w, x_max + pad_w), min(h, y_max + pad_h)
                    face_crop = frame[y1:y2, x1:x2].copy()
                    
                    if face_crop.size > 0:
                        with state.lock:
                            state.status = "PROCESSING"
                            state.process_start_time = current_time
                            state.progress = 0
                            state.pending_crop = face_crop

        if state.status == "PROCESSING":
            elapsed = current_time - state.process_start_time
            if elapsed < 0:
                with state.lock: state.progress = 90 + int((current_time * 10) % 9)
            else:
                prog = int((elapsed / 0.2) * 90)
                with state.lock: state.progress = min(90, max(0, prog))
                if elapsed > 0.1:
                    with state.lock:
                        ai_input = state.pending_crop
                    
                    if ai_input is not None:
                        processed_ai_frame = preprocess_frame(ai_input.copy())
                        threading.Thread(target=run_recognition_async, 
                                       args=(processed_ai_frame, raw_frame.copy(), state, x_min, y_min, x_max, y_max), 
                                       daemon=True).start()
                        with state.lock: 
                            state.process_start_time = current_time + 1000 
                            state.pending_crop = None
                    else:
                        with state.lock: state.status = "SCANNING"

        if state.status != "CONFIRM":
            with state.lock:
                state.frame = frame.copy()
        
        time.sleep(0.005) 

# Start Thread
t = threading.Thread(target=camera_worker, daemon=True)
t.start()

# --- WEB ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_frames():
    while True:
        with state.lock:
            if state.frame is None: 
                time.sleep(0.01)
                continue
            _, buffer = cv2.imencode('.jpg', state.frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status")
async def get_status():
    with state.lock:
        return {
            "status": state.status,
            "progress": int(state.progress),
            "data": state.student_data,
            "is_near": bool(state.is_near),
            "is_live": bool(state.is_live),
            "fas_score": float(state.fas_score),
            "blink_count": int(state.blink_count)
        }

class ActionRequest(BaseModel):
    action: str

@app.post("/api/action")
async def handle_action(req: ActionRequest):
    action = req.action
    
    if action == 'confirm':
        if state.student_data:
            sid = state.student_data['student_id']
            try:
                student_collect_dir = os.path.join("collected_faces", sid)
                if not os.path.exists(student_collect_dir):
                    os.makedirs(student_collect_dir)
                filename = f"{int(time.time())}.jpg"
                save_path = os.path.join(student_collect_dir, filename)
                with state.lock:
                    target_image = state.clean_snapshot if state.clean_snapshot is not None else state.frame
                    if target_image is not None:
                        cv2.imwrite(save_path, target_image)
                        print(f"📸 Đã lưu ảnh SẠCH vào folder tự học: {save_path}")
            except Exception as e:
                print(f"⚠️ Lỗi lưu ảnh tự học: {e}")
            print(f"CONFIRMED: {sid}")
    
    with state.lock:
        state.status = "SCANNING"
        state.student_data = None
        state.last_scan_time = time.time() + 0.5 
        
    return {"success": True}

def main():
    import uvicorn
    try:
        # Giảm log_level xuống 'error' để tránh làm phiền và giúp tắt nhanh hơn
        uvicorn.run(app, host="0.0.0.0", port=5000, log_level="error")
    except KeyboardInterrupt:
        state.running = False
        os._exit(0)

if __name__ == "__main__":
    main()
