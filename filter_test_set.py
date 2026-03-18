"""
Filter test set: Remove images that the system rejects (no face detected, 
low confidence, insufficient consensus). Keep only accepted images.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import shutil
import numpy as np
import mediapipe as mp
from qdrant_client import QdrantClient
from arcface_onnx import get_arcface_model
from tqdm import tqdm

COLLECTION_NAME = "student_faces"
TEST_DIR = "test_faces"
REJECTED_DIR = "test_faces_rejected"  # Move rejected images here
THRESHOLD = 0.55

def get_standard_aligned_face(frame, face_landmarks, target_size=(112, 112)):
    h, w = frame.shape[:2]
    landmarks = face_landmarks.landmark
    l_eye = np.mean([(landmarks[33].x*w, landmarks[33].y*h), (landmarks[133].x*w, landmarks[133].y*h)], axis=0)
    r_eye = np.mean([(landmarks[362].x*w, landmarks[362].y*h), (landmarks[263].x*w, landmarks[263].y*h)], axis=0)
    nose = (landmarks[1].x*w, landmarks[1].y*h)
    l_mouth = (landmarks[61].x*w, landmarks[61].y*h)
    r_mouth = (landmarks[291].x*w, landmarks[291].y*h)
    src = np.array([l_eye, r_eye, nose, l_mouth, r_mouth], dtype=np.float32)
    dst = np.array([[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],[33.5493,92.3655],[62.7299,92.2041]], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None: return cv2.resize(frame, target_size)
    return cv2.warpAffine(frame, M, target_size, borderMode=cv2.BORDER_CONSTANT)

def filter_test_set():
    arcface = get_arcface_model()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    client = QdrantClient(host="localhost", port=6333)

    # Collect test files
    test_files = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                mssv = os.path.basename(root)
                test_files.append((os.path.join(root, file), mssv))

    if not test_files:
        print("No test images found.")
        return

    print(f"Scanning {len(test_files)} test images to filter rejected ones...\n")

    accepted_count = 0
    rejected_count = 0
    rejected_files = []

    for file_path, true_sid in tqdm(test_files):
        is_accepted = False
        
        img = cv2.imread(file_path)
        if img is not None:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                face_aligned = get_standard_aligned_face(img, landmarks)
                
                # TTA
                vec_orig = arcface.get_embedding(face_aligned, normalize=True)
                lab = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
                img_clahe = cv2.merge((cl, a, b))
                img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
                vec_clahe = arcface.get_embedding(img_clahe, normalize=True)
                img_bright = cv2.convertScaleAbs(face_aligned, alpha=1.2, beta=10)
                vec_bright = arcface.get_embedding(img_bright, normalize=True)
                
                final_vec = np.mean([vec_orig, vec_clahe, vec_bright], axis=0)
                norm = np.linalg.norm(final_vec)
                final_vec = (final_vec / norm).tolist() if norm > 0 else final_vec.tolist()

                search_res = client.query_points(collection_name=COLLECTION_NAME, query=final_vec, limit=5).points
                
                if search_res:
                    score = search_res[0].score
                    pred_sid = search_res[0].payload['student_id']
                    votes = {}
                    for res in search_res:
                        sid = res.payload['student_id']
                        votes[sid] = votes.get(sid, 0) + 1
                    top_voter = max(votes, key=votes.get)
                    vote_count = votes[top_voter]
                    
                    # Accepted if passes consensus logic
                    if score > THRESHOLD and (pred_sid == top_voter) and (vote_count >= 2):
                        is_accepted = True

        if is_accepted:
            accepted_count += 1
        else:
            rejected_count += 1
            rejected_files.append(file_path)

    face_mesh.close()

    # Move rejected files
    print(f"\nAccepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"\nMoving {rejected_count} rejected images to '{REJECTED_DIR}/'...")

    for file_path in rejected_files:
        # Preserve folder structure
        rel_path = os.path.relpath(file_path, TEST_DIR)
        dest_path = os.path.join(REJECTED_DIR, rel_path)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.move(file_path, dest_path)

    # Clean empty folders in test_faces
    for root, dirs, files in os.walk(TEST_DIR, topdown=False):
        if root == TEST_DIR:
            continue
        if not os.listdir(root):
            os.rmdir(root)

    print(f"Done! test_faces/ now contains only {accepted_count} accepted images.")
    print(f"Rejected images saved in {REJECTED_DIR}/")
    print(f"\nRun 'python evaluate_accuracy.py' to get clean metrics.")

if __name__ == "__main__":
    filter_test_set()
