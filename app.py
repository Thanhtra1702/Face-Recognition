from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time
import os
import numpy as np
from deepface import DeepFace
from pydantic import BaseModel

# Import modules tự tạo
from core_state import KioskState, setup_signals
from anti_spoof import AntiSpoof, calculate_ear, preprocess_frame, LEFT_EYE, RIGHT_EYE, mp_face_mesh
from recognition import run_recognition_async

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# --- GLOBAL STATE ---
state = KioskState()
setup_signals(state)

# --- PRELOAD MODEL ---
print("🚀 Đang tải model AI (FastAPI - Modular)...")
try:
    dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
    DeepFace.represent(img_path=dummy_img, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
    print("✅ Model AI đã sẵn sàng!")
except Exception as e:
    print(f"❌ Error preloading model: {e}")

# --- AI HELPERS: ALIGNMENT ON CPU ---
def get_affine_aligned_face(frame, landmarks, target_size=(112, 112)):
    """Align khuôn mặt bằng toán học (Affine) dựa trên landmarks - CỰC NHANH"""
    h, w = frame.shape[:2]
    # Landmark idx Mediapipe cho tâm mắt
    # Trái: 33, 133 -> trung tâm 468 landmarks
    l_eye = np.mean([ (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h), (landmarks.landmark[133].x * w, landmarks.landmark[133].y * h) ], axis=0)
    r_eye = np.mean([ (landmarks.landmark[362].x * w, landmarks.landmark[362].y * h), (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h) ], axis=0)
    
    dX = r_eye[0] - l_eye[0]
    dY = r_eye[1] - l_eye[1]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Giữ tỉ lệ mắt ở vị trí 35% chiều ngang
    dist = np.sqrt(dX**2 + dY**2)
    desired_dist = (0.7 - 0.3) * target_size[0]
    scale = desired_dist / dist
    
    eye_center = ( (l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2 )
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    
    # Điều chỉnh dịch chuyển để mắt nằm ở hàng 35% chiều dọc
    tX = target_size[0] * 0.5
    tY = target_size[1] * 0.35
    M[0, 2] += (tX - eye_center[0])
    M[1, 2] += (tY - eye_center[1])
    
    aligned_face = cv2.warpAffine(frame, M, (target_size[0], target_size[1]), flags=cv2.INTER_CUBIC)
    return aligned_face

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
    
    frame_count = 0
    face_stable_start_time = 0 
    last_face_center = None # Để kiểm tra đứng yên
    cached_face_data = None  
    cached_landmarks = None

    while state.running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        current_time = time.time()
        frame_count += 1
        
        run_detection = (frame_count % 2 == 0) or cached_face_data is None
        
        if run_detection:
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
                    cached_face_data = best_face_data
                    cached_landmarks = results.multi_face_landmarks[0]
                else:
                    cached_face_data = None
                    face_stable_start_time = 0
            else:
                cached_face_data = None
                face_stable_start_time = 0
        
        if cached_face_data:
                x_min, y_min, x_max, y_max = cached_face_data
                h, w = frame.shape[:2]
                
                # --- ANTISPOOF & LANDMARKS (Multi-Scale) ---
                try:
                    # 1. Dự đoán bằng Multi-Scale: Crop hẹp (Texture) và Crop rộng (Context)
                    # FAS Net chuẩn yêu cầu crop rộng khoảng 2.7x so với khuôn mặt
                    w_f, h_f = x_max - x_min, y_max - y_min
                    
                    # Crop 1: Texture (1.0x)
                    fas_pad1 = int(w_f * 0.15)
                    fx1, fy1 = max(0, x_min - fas_pad1), max(0, y_min - fas_pad1)
                    fx2, fy2 = min(w, x_max + fas_pad1), min(h, y_max + fas_pad1)
                    face_tight = raw_frame[fy1:fy2, fx1:fx2]
                    
                    # Crop 2: Context (2.7x) - Để bắt viền màn hình điện thoại
                    fas_pad2 = int(w_f * 0.8) # Mở rộng ra xung quanh
                    fw1, fh1 = max(0, x_min - fas_pad2), max(0, y_min - fas_pad2)
                    fw2, fh2 = min(w, x_max + fas_pad2), min(h, y_max + fas_pad2)
                    face_wide = raw_frame[fh1:fh2, fw1:fw2]
                    
                    if face_tight.size > 0:
                        if not hasattr(state, '_fas_frame_counter'): state._fas_frame_counter = 0
                        state._fas_frame_counter += 1
                        
                        # Quét thưa để mượt camera
                        fas_interval = 10 if state.is_live else 5
                        if state._fas_frame_counter % fas_interval == 0:
                            score = state.anti_spoof.predict(face_tight, face_wide)
                            with state.lock:
                                state.fas_score = (state.fas_score * 0.5) + (score * 0.5)
                    
                    ear_left = calculate_ear(cached_landmarks.landmark, LEFT_EYE, w, h)
                    ear_right = calculate_ear(cached_landmarks.landmark, RIGHT_EYE, w, h)
                    ear = (ear_left + ear_right) / 2.0
                    
                    with state.lock:
                        if ear < state.blink_threshold: state._eye_closed = True
                        elif hasattr(state, '_eye_closed') and state._eye_closed:
                            state.blink_count += 1
                            state._eye_closed = False
                            state.last_blink_time = time.time()
                        
                        # Ngưỡng nới lỏng 0.85 cho người đeo kính + Nháy mắt
                        is_pass_fas = state.fas_score > 0.85
                        is_pass_blink = (time.time() - state.last_blink_time < 5.0)
                        state.is_live = is_pass_fas and is_pass_blink
                except: pass
                
                # --- HEAD POSE CHECK ---
                nose_tip = cached_landmarks.landmark[1]
                l_eye_corner = cached_landmarks.landmark[33]
                r_eye_corner = cached_landmarks.landmark[263]
                dist_l = abs(nose_tip.x - l_eye_corner.x)
                dist_r = abs(nose_tip.x - r_eye_corner.x)
                turn_ratio = dist_l / (dist_r + 1e-6)
                is_looking_straight = (0.4 < turn_ratio < 2.5) # Nới rất lỏng
                
                # Debug print mỗi 30 frames
                if frame_count % 30 == 0:
                   print(f"Status: FAS={state.fas_score:.2f}({'OK' if is_pass_fas else 'FAIL'}), "
                         f"Blink={'OK' if is_pass_blink else 'WAITING'}, "
                         f"Straight={'OK' if is_looking_straight else 'FAIL'}")
                
                face_width = x_max - x_min
                is_near_enough = face_width > 180 
                face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                # --- STABILITY CHECK ---
                is_moving = False
                if last_face_center is not None:
                    dist = np.sqrt((face_center[0] - last_face_center[0])**2 + (face_center[1] - last_face_center[1])**2)
                    if dist > 25: is_moving = True # Nới lỏng độ nhạy rung lắc lên 25px
                
                last_face_center = face_center

                if is_near_enough and state.is_live and is_looking_straight and not is_moving:
                    if face_stable_start_time == 0: face_stable_start_time = current_time
                else: 
                    face_stable_start_time = 0 
                
                stable_duration = current_time - face_stable_start_time if face_stable_start_time > 0 else 0
                
                with state.lock: state.is_near = is_near_enough
                
                # --- SIMPLE UI COLORS ---
                color = (255, 255, 255)
                thickness = 2
                if is_near_enough:
                    color = (33, 165, 255) # Cam
                if state.status == "CONFIRM":
                    color = (0, 255, 0) # Xanh lá
                
                t, l = thickness, 40
                cv2.line(frame, (x_min, y_min), (x_min + l, y_min), color, t)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + l), color, t)
                cv2.line(frame, (x_max, y_min), (x_max - l, y_min), color, t)
                cv2.line(frame, (x_max, y_min), (x_max, y_min + l), color, t)
                cv2.line(frame, (x_min, y_max), (x_min + l, y_max), color, t)
                cv2.line(frame, (x_min, y_max), (x_min, y_max - l), color, t)
                cv2.line(frame, (x_max, y_max), (x_max - l, y_max), color, t)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - l), color, t)
                
                # --- TRIGGER RECOGNITION (0.8s STABILITY) ---
                if is_near_enough and state.is_live and is_looking_straight and \
                   state.status == "SCANNING" and (stable_duration > 0.8) and \
                   (current_time - state.last_scan_time > 0.5):
                    
                    # ALIGN TRỰC TIẾP BẰNG CPU TẠI ĐÂY
                    aligned_crop = get_affine_aligned_face(raw_frame, cached_landmarks)
                    
                    if aligned_crop.size > 0:
                        with state.lock:
                            state.status = "PROCESSING"
                            state.process_start_time = current_time
                            state.progress = 0
                            state.pending_crop = aligned_crop # Gửi ảnh ĐÃ ALIGN
                            
        if state.status == "PROCESSING":
            elapsed = current_time - state.process_start_time
            if elapsed > 0.01:
                with state.lock: ai_input = state.pending_crop
                if ai_input is not None:
                    threading.Thread(target=run_recognition_async, 
                                   args=(ai_input.copy(), raw_frame.copy(), state, x_min, y_min, x_max, y_max), 
                                   daemon=True).start()
                    with state.lock: 
                        state.process_start_time = current_time + 1000 
                        state.pending_crop = None
                else:
                    with state.lock: state.status = "SCANNING"

        if state.status != "CONFIRM":
            with state.lock: state.frame = frame.copy()
        time.sleep(0.001) 

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
                time.sleep(0.01); continue
            _, buffer = cv2.imencode('.jpg', state.frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/status")
async def get_status():
    with state.lock:
        return {
            "status": state.status, "progress": int(state.progress),
            "data": state.student_data, "is_near": bool(state.is_near),
            "is_live": bool(state.is_live), "fas_score": float(state.fas_score),
            "blink_count": int(state.blink_count)
        }

class ActionRequest(BaseModel):
    action: str

@app.post("/api/action")
async def handle_action(req: ActionRequest):
    if req.action == 'confirm' and state.student_data:
        sid = state.student_data['student_id']
        try:
            student_collect_dir = os.path.join("collected_faces", sid)
            if not os.path.exists(student_collect_dir): os.makedirs(student_collect_dir)
            save_path = os.path.join(student_collect_dir, f"{int(time.time())}.jpg")
            with state.lock:
                target_image = state.clean_snapshot if state.clean_snapshot is not None else state.frame
                if target_image is not None: cv2.imwrite(save_path, target_image)
        except Exception as e: print(f"⚠️ Error: {e}")
    
    with state.lock:
        state.status = "SCANNING"; state.student_data = None
        state.last_scan_time = time.time() + 0.5 
    return {"success": True}

if __name__ == "__main__":
    import uvicorn
    try: uvicorn.run(app, host="0.0.0.0", port=5000, log_level="error")
    except KeyboardInterrupt:
        state.running = False
        os._exit(0)
