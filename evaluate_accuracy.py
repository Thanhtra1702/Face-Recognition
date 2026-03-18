import os
import cv2
import numpy as np
import mediapipe as mp
from qdrant_client import QdrantClient
from arcface_onnx import get_arcface_model
from tqdm import tqdm

# --- CONFIG ---
COLLECTION_NAME = "student_faces"
TEST_DIR = "test_faces"
THRESHOLD = 0.55  # SOTA threshold as in recognition.py

def get_standard_aligned_face(frame, face_landmarks, target_size=(112, 112)):
    """Same alignment logic as recognition.py"""
    h, w = frame.shape[:2]
    landmarks = face_landmarks.landmark
    # Standard 5 landmarks for ArcFace
    l_eye = np.mean([ (landmarks[33].x * w, landmarks[33].y * h), (landmarks[133].x * w, landmarks[133].y * h) ], axis=0)
    r_eye = np.mean([ (landmarks[362].x * w, landmarks[362].y * h), (landmarks[263].x * w, landmarks[263].y * h) ], axis=0)
    nose = (landmarks[1].x * w, landmarks[1].y * h)
    l_mouth = (landmarks[61].x * w, landmarks[61].y * h)
    r_mouth = (landmarks[291].x * w, landmarks[291].y * h)
    
    src = np.array([l_eye, r_eye, nose, l_mouth, r_mouth], dtype=np.float32)
    dst = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]], dtype=np.float32)
    
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None: return cv2.resize(frame, target_size)
    return cv2.warpAffine(frame, M, target_size, borderMode=cv2.BORDER_CONSTANT)

def evaluate():
    # ── Models ──
    arcface = get_arcface_model()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # ── Qdrant ──
    client = QdrantClient(host="localhost", port=6333)

    # ── Collect Test Files ──
    test_files = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                mssv = os.path.basename(root)
                test_files.append((os.path.join(root, file), mssv))

    if not test_files:
        print("No test images found in test_faces/.")
        return

    print(f"🔬 Starting evaluation on {len(test_files)} images...")
    
    correct = 0
    incorrect = 0
    rejected = 0
    total = len(test_files)

    for file_path, true_sid in tqdm(test_files):
        img = cv2.imread(file_path)
        if img is None: continue
        
        # 1. Align face
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            rejected += 1
            continue
        
        landmarks = results.multi_face_landmarks[0]
        face_aligned = get_standard_aligned_face(img, landmarks)
        
        # 2. Avg Embedding (TTA) as in recognition.py
        # Original
        vec_orig = arcface.get_embedding(face_aligned, normalize=True)
        # CLAHE
        lab = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        img_clahe = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
        vec_clahe = arcface.get_embedding(img_clahe, normalize=True)
        # Bright
        img_bright = cv2.convertScaleAbs(face_aligned, alpha=1.2, beta=10)
        vec_bright = arcface.get_embedding(img_bright, normalize=True)
        
        # Mean
        final_vec = np.mean([vec_orig, vec_clahe, vec_bright], axis=0)
        norm = np.linalg.norm(final_vec)
        final_vec = (final_vec / norm).tolist() if norm > 0 else final_vec.tolist()

        # 3. Search Database
        search_res = client.query_points(collection_name=COLLECTION_NAME, query=final_vec, limit=5).points
        
        if not search_res:
            rejected += 1
            continue
            
        # 4. Consensus Logic
        best_match = search_res[0]
        score = best_match.score
        pred_sid = best_match.payload['student_id']
        
        votes = {}
        for res in search_res:
            sid = res.payload['student_id']
            votes[sid] = votes.get(sid, 0) + 1
        
        top_voter = max(votes, key=votes.get)
        vote_count = votes[top_voter]
        
        # Matching Logic from recognition.py: score > 0.55 and (current_sid == top_voter) and (vote_count >= 2)
        if score > THRESHOLD and (pred_sid == top_voter) and (vote_count >= 2):
            if pred_sid == true_sid:
                correct += 1
            else:
                incorrect += 1
                # print(f"❌ Mismatch: True={true_sid}, Pred={pred_sid}, Score={score:.3f}")
        else:
            rejected += 1 # Low confidence or inconsistent

    face_mesh.close()

    # --- SUMMARY ---
    accuracy = (correct / total) * 100 if total > 0 else 0
    print("\n" + "="*30)
    print(f"📊 EVALUATION RESULTS")
    print(f"Total:      {total}")
    print(f"Correct:    {correct}")
    print(f"Incorrect:  {incorrect}")
    print(f"Rejected:   {rejected} (Threshold: {THRESHOLD})")
    print(f"Accuracy:   {accuracy:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()
