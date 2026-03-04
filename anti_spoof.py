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
    Khử nhiễu + CLAHE để cân bằng độ tương phản, giúp AI nhận diện tốt hơn.
    Đồng bộ với init_qdrant.py và process_collected_faces.py.
    """
    try:
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except:
        return frame

def detect_screen_moire(face_img):
    """
    Sử dụng Fast Fourier Transform (FFT) để phát hiện vân pixel của màn hình điện thoại/máy tính.
    Màn hình kỹ thuật số sẽ để lại các thành phần tần số cao đặc trưng mà da người không có.
    """
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Chuyển sang miền tần số
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
        
        # Tập trung vào các vùng tần số cao (vùng rìa ngoài của spectrum)
        # Loại bỏ tâm (tần số thấp/độ sáng trung bình)
        crow, ccol = h // 2, w // 2
        magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10] = 0
        
        high_freq_mean = np.mean(magnitude_spectrum)
        # Nới lỏng ngưỡng để không bị bắt nhầm người thật đeo kính
        return high_freq_mean > 110 
    except:
        return False

# --- ANTI-SPOOF MODULE ---
class AntiSpoof:
    def __init__(self, model_path="MiniFASNetV2.onnx"):
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name

    def predict(self, face_img, wide_face_img=None):
        """
        Dự đoán liveness bằng phương pháp Multi-Scale (Đa quy mô).
        Dựa trên kết cấu da (cận cảnh) và bối cảnh (rộng).
        """
        try:
            # Quy mô 1: Texture khuôn mặt (Cận cảnh)
            resized1 = cv2.resize(face_img, (80, 80))
            img1 = resized1.astype(np.float32)
            img1 = np.transpose(np.expand_dims(img1, axis=0), (0, 3, 1, 2))
            score1 = self._run_model(img1)
            
            if wide_face_img is None:
                return score1
            
            # Quy mô 2: Context (Rộng)
            resized2 = cv2.resize(wide_face_img, (80, 80))
            img2 = resized2.astype(np.float32)
            img2 = np.transpose(np.expand_dims(img2, axis=0), (0, 3, 1, 2))
            score2 = self._run_model(img2)
            
            # CHẾ ĐỘ THẮT CHẶT: Lấy điểm thấp nhất trong 2 quy mô (Minimum)
            # Nếu một trong hai nghi ngờ là giả -> Từ chối ngay.
            final_score = min(score1, score2)
            return final_score
            
        except Exception as e:
            print(f"AntiSpoof Error: {e}")
            return 0.0

    def _run_model(self, img_input):
        ort_inputs = {self.input_name: img_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        result = ort_outs[0][0]
        exp_res = np.exp(result - np.max(result))
        prob = exp_res / exp_res.sum()
        return prob[1]
