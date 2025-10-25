import base64
import cv2
import numpy as np

# Cache DeepFace import to avoid repeated lazy imports
_deepface = None

def _get_deepface():
    """Get cached DeepFace instance."""
    global _deepface
    if _deepface is None:
        from deepface import DeepFace #type: ignore
        _deepface = DeepFace
    return _deepface


def decode_image(image):
    try:
        data = base64.b64decode(image)
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def gen_embeddings(image):
    try:
        DeepFace = _get_deepface()
        # Optimized settings for faster processing
        result = DeepFace.represent(
            img_path=image, 
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv",  # Faster than default
            align=True
        )
        embeddings = np.array(result[0]['embedding'], dtype=np.float32)
        nrm = np.linalg.norm(embeddings)
        return embeddings / nrm if nrm > 0 else embeddings
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return None


def compare_embeddings(emb_1, emb_2, threshold):
    cos_sim = np.dot(emb_1, emb_2) / (np.linalg.norm(emb_1) * np.linalg.norm(emb_2))
    return cos_sim > (1 - threshold)


