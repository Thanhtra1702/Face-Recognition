"""
Detailed Evaluation Script — Reports both Open-Set and Closed-Set (sufficient enrollment) results.
Produces metrics suitable for international paper submission.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import mediapipe as mp
from qdrant_client import QdrantClient
from arcface_onnx import get_arcface_model
from tqdm import tqdm
from collections import defaultdict

# --- CONFIG ---
COLLECTION_NAME = "student_faces"
TEST_DIR = "test_faces"
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

def evaluate():
    arcface = get_arcface_model()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    client = QdrantClient(host="localhost", port=6333)

    # ── Count how many vectors each student_id has in Qdrant ──
    print("📊 Counting enrolled vectors per identity in Qdrant...")
    # Scroll through all points to count per-identity enrollment
    enrollment_count = defaultdict(int)
    offset = None
    while True:
        result = client.scroll(collection_name=COLLECTION_NAME, limit=1000, offset=offset, with_payload=True, with_vectors=False)
        points, next_offset = result
        for p in points:
            sid = p.payload.get("student_id", "")
            enrollment_count[sid] += 1
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"   Total identities in DB: {len(enrollment_count)}")
    
    # Distribution of enrollment counts
    dist = defaultdict(int)
    for sid, cnt in enrollment_count.items():
        dist[cnt] += 1
    print("   Enrollment distribution (vectors per identity):")
    for k in sorted(dist.keys()):
        print(f"     {k} vectors: {dist[k]} identities")

    # ── Collect Test Files ──
    test_files = []
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                mssv = os.path.basename(root)
                test_files.append((os.path.join(root, file), mssv))

    if not test_files:
        print("No test images found.")
        return

    print(f"\n🔬 Starting evaluation on {len(test_files)} test images...\n")

    # Per-image results
    results = []  # list of (true_sid, pred_sid_or_None, score, vote_count, enrollment_vectors, status)
    
    score_list_correct = []
    score_list_incorrect = []

    for file_path, true_sid in tqdm(test_files):
        img = cv2.imread(file_path)
        if img is None:
            results.append((true_sid, None, 0, 0, enrollment_count.get(true_sid, 0), "no_image"))
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb_img)

        if not res.multi_face_landmarks:
            results.append((true_sid, None, 0, 0, enrollment_count.get(true_sid, 0), "no_face"))
            continue

        face_lm = res.multi_face_landmarks[0]
        face_aligned = get_standard_aligned_face(img, face_lm)

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

        if not search_res:
            results.append((true_sid, None, 0, 0, enrollment_count.get(true_sid, 0), "no_result"))
            continue

        best = search_res[0]
        score = best.score
        pred_sid = best.payload['student_id']

        votes = {}
        for r in search_res:
            sid = r.payload['student_id']
            votes[sid] = votes.get(sid, 0) + 1
        top_voter = max(votes, key=votes.get)
        vote_count = votes[top_voter]

        n_enrolled = enrollment_count.get(true_sid, 0)

        if score > THRESHOLD and (pred_sid == top_voter) and (vote_count >= 2):
            if pred_sid == true_sid:
                results.append((true_sid, pred_sid, score, vote_count, n_enrolled, "correct"))
                score_list_correct.append(score)
            else:
                results.append((true_sid, pred_sid, score, vote_count, n_enrolled, "incorrect"))
                score_list_incorrect.append(score)
        else:
            results.append((true_sid, pred_sid, score, vote_count, n_enrolled, "rejected"))

    face_mesh.close()

    # ═══════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════

    total = len(results)
    correct = sum(1 for r in results if r[5] == "correct")
    incorrect = sum(1 for r in results if r[5] == "incorrect")
    rejected = sum(1 for r in results if r[5] in ("rejected", "no_face", "no_result", "no_image"))
    no_face = sum(1 for r in results if r[5] == "no_face")

    print("\n" + "="*60)
    print("📊 EVALUATION REPORT — OPEN SET (ALL IDENTITIES)")
    print("="*60)
    print(f"  Total Test Images:     {total}")
    print(f"  Correct (TP):          {correct}")
    print(f"  Incorrect (FP):        {incorrect}")
    print(f"  Rejected:              {rejected}  (incl. {no_face} face detection failures)")
    print(f"  ─────────────────────────────────")
    print(f"  Accuracy (TP/Total):   {correct/total*100:.2f}%")
    print(f"  FAR (FP/Total):        {incorrect/total*100:.2f}%")
    print(f"  Rejection Rate:        {rejected/total*100:.2f}%")
    accepted = correct + incorrect
    precision = correct / accepted * 100 if accepted > 0 else 0
    print(f"  Precision (TP/(TP+FP)):{precision:.2f}%")
    if score_list_correct:
        print(f"  Avg Score (Correct):   {np.mean(score_list_correct):.4f}")
    if score_list_incorrect:
        print(f"  Avg Score (Incorrect): {np.mean(score_list_incorrect):.4f}")

    # ── CLOSED-SET: Only identities with >= N enrollment vectors ──
    for min_vectors in [4, 8, 12]:
        subset = [r for r in results if r[4] >= min_vectors]
        if not subset:
            continue
        sub_total = len(subset)
        sub_correct = sum(1 for r in subset if r[5] == "correct")
        sub_incorrect = sum(1 for r in subset if r[5] == "incorrect")
        sub_rejected = sum(1 for r in subset if r[5] not in ("correct", "incorrect"))
        sub_accepted = sub_correct + sub_incorrect
        sub_acc = sub_correct / sub_total * 100 if sub_total > 0 else 0
        sub_far = sub_incorrect / sub_total * 100 if sub_total > 0 else 0
        sub_prec = sub_correct / sub_accepted * 100 if sub_accepted > 0 else 0
        sub_rej = sub_rejected / sub_total * 100 if sub_total > 0 else 0

        print(f"\n{'='*60}")
        print(f"📊 CLOSED-SET (Enrollment ≥ {min_vectors} vectors)")
        print(f"{'='*60}")
        print(f"  Test Images:           {sub_total}")
        print(f"  Correct:               {sub_correct}")
        print(f"  Incorrect:             {sub_incorrect}")
        print(f"  Rejected:              {sub_rejected}")
        print(f"  ─────────────────────────────────")
        print(f"  Accuracy:              {sub_acc:.2f}%")
        print(f"  FAR:                   {sub_far:.2f}%")
        print(f"  Rejection Rate:        {sub_rej:.2f}%")
        print(f"  Precision:             {sub_prec:.2f}%")

    # ── Rejection Analysis by enrollment size ──
    print(f"\n{'='*60}")
    print(f"📊 REJECTION ANALYSIS BY ENROLLMENT SIZE")
    print(f"{'='*60}")
    print(f"  {'Enrolled Vectors':<20} {'Total':<8} {'Correct':<10} {'Rejected':<10} {'Rej Rate':<10}")
    print(f"  {'─'*58}")
    
    buckets = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 999, "13+")]
    for lo, hi, label in buckets:
        bucket = [r for r in results if lo <= r[4] <= hi]
        if not bucket:
            continue
        b_total = len(bucket)
        b_correct = sum(1 for r in bucket if r[5] == "correct")
        b_rejected = sum(1 for r in bucket if r[5] not in ("correct", "incorrect"))
        b_rej_rate = b_rejected / b_total * 100 if b_total > 0 else 0
        print(f"  {label:<20} {b_total:<8} {b_correct:<10} {b_rejected:<10} {b_rej_rate:.1f}%")

    print(f"\n{'='*60}")
    print("✅ Evaluation complete.")
    print(f"{'='*60}")

if __name__ == "__main__":
    evaluate()
