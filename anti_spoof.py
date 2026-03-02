import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

# --- BLINK DETECTION HELPERS ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, w, h):
    """Tính Eye Aspect Ratio (Tỷ lệ mở mắt)"""
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((lm.x * w, lm.y * h))
    
    # Tính khoảng cách dọc (Vertical)
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    
    # Tính khoảng cách ngang (Horizontal)
    h_dist = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

def preprocess_frame(frame):
    """
    Sử dụng CLAHE để cân bằng độ tương phản, giúp AI nhận diện tốt hơn 
    """
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return frame

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
