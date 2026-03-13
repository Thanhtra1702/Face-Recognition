"""
ArcFace ONNX — Direct ONNX Runtime inference for face embedding extraction.
Thay thế DeepFace wrapper, nhanh hơn 5-10x.

Model: w600k_r50.onnx (ResNet50, trained on WebFace600K, 512-d embedding)
Input: 112x112 BGR aligned face
Output: 512-d float32 embedding vector
"""

import cv2
import numpy as np
import onnxruntime as ort
import os

__all__ = ["ArcFaceONNX"]

# ── Singleton: Load model 1 lần duy nhất cho toàn bộ ứng dụng ──────────────
_model_instance = None

def get_arcface_model(model_path="w600k_r50.onnx"):
    """Lấy singleton ArcFaceONNX. Chỉ load model 1 lần duy nhất."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ArcFaceONNX(model_path)
    return _model_instance


class ArcFaceONNX:
    """
    ArcFace ONNX Runtime Inference Engine.
    
    - Input: Aligned face 112x112 BGR
    - Preprocessing: BGR→RGB, normalize (pixel - 127.5) / 127.5
    - Output: L2-normalized 512-d embedding vector
    """

    def __init__(self, model_path: str = "w600k_r50.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy model ArcFace: {model_path}\n"
                f"Tải về: https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx"
            )

        self.input_size = (112, 112)

        # ── Chọn provider tối ưu ──
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }))
        providers.append("CPUExecutionProvider")

        # ── Tối ưu hóa session ──
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4  # Tối ưu cho CPU multi-core

        try:
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        except Exception:
            # Fallback CPU nếu CUDA lỗi
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.embedding_size = self.session.get_outputs()[0].shape[1]

        active = self.session.get_providers()
        gpu_tag = "🎮 GPU" if "CUDAExecutionProvider" in active else "🔧 CPU"
        print(f"✅ ArcFace ONNX loaded ({gpu_tag}) | Embedding: {self.embedding_size}d | Providers: {active}")

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh khuôn mặt cho ArcFace inference.
        
        Args:
            face_bgr: Ảnh BGR (bất kỳ kích thước)
        Returns:
            Blob NCHW float32 đã normalize, shape (1, 3, 112, 112)
        """
        # Resize về 112x112
        resized = cv2.resize(face_bgr, self.input_size)
        
        # cv2.dnn.blobFromImage: BGR→RGB + Normalize (pixel - 127.5) / 127.5
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 127.5,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob

    def get_embedding(self, face_bgr: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Trích xuất embedding vector từ ảnh khuôn mặt đã aligned.
        
        Args:
            face_bgr: Ảnh BGR, đã aligned (tốt nhất 112x112, nếu khác sẽ tự resize)
            normalize: L2 normalize output embedding (mặc định True cho cosine similarity)
        Returns:
            Embedding vector float32 shape (512,)
        """
        blob = self.preprocess(face_bgr)
        embedding = self.session.run(self.output_names, {self.input_name: blob})[0].flatten()

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def get_embedding_from_rgb(self, face_rgb: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Trích xuất embedding từ ảnh RGB (đã aligned).
        Dùng cho TTA variants đã convert sang RGB trước.
        
        Args:
            face_rgb: Ảnh RGB, đã aligned
            normalize: L2 normalize
        Returns:
            Embedding vector float32 shape (512,)
        """
        # Convert RGB→BGR rồi dùng pipeline chuẩn
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        return self.get_embedding(face_bgr, normalize)

    def batch_embeddings(self, faces_bgr: list, normalize: bool = True) -> list:
        """
        Trích xuất embeddings cho nhiều ảnh cùng lúc (sequential, không batch ONNX).
        
        Args:
            faces_bgr: List các ảnh BGR đã aligned
            normalize: L2 normalize
        Returns:
            List các embedding vectors
        """
        return [self.get_embedding(face, normalize) for face in faces_bgr]
