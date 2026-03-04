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
    
    # Frame skipping for smoother camera
    frame_count = 0
    cached_face_data = None  # (x_min, y_min, x_max, y_max)
    cached_landmarks = None

    while state.running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        raw_frame = frame.copy()
        current_time = time.time()
        frame_count += 1
        
        # Only run face_mesh every 2nd frame for smoother display
        run_detection = (frame_count % 2 == 0) or cached_face_data is None
        
        if run_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
        
        if run_detection and results.multi_face_landmarks:
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
        
        # Use cached or fresh face data
        h, w = frame.shape[:2]
        if cached_face_data:
                x_min, y_min, x_max, y_max = cached_face_data
                
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
                    
                    face_landmarks = cached_landmarks
                    if face_landmarks:
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
                
                if is_near_enough and state.is_live and state.status == "SCANNING" and (current_time - state.last_scan_time > 0.3):
                    pad_w = int((x_max - x_min) * 0.25)
                    pad_h = int((y_max - y_min) * 0.25)
                    x1, y1 = max(0, x_min - pad_w), max(0, y_min - pad_h)
                    x2, y2 = min(w, x_max + pad_w), min(h, y_max + pad_h)
                    face_crop = raw_frame[y1:y2, x1:x2].copy()
                    
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
                if elapsed > 0.01:
                    with state.lock:
                        ai_input = state.pending_crop
                    
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
            with state.lock:
                state.frame = frame.copy()
        
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
        # FastAPI server running on uvicorn
        # Chỉ hiện log Error để tắt máy nhanh hơn (1 lần Ctrl+C)
        uvicorn.run(app, host="0.0.0.0", port=5000, log_level="error")
    except KeyboardInterrupt:
        state.running = False
        os._exit(0)

if __name__ == "__main__":
    main()
