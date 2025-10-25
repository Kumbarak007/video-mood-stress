from deepface import DeepFace
from PIL import Image
import numpy as np
import tempfile
import os

def get_detector(name="mtcnn"):
    """
    Returns the name of the detector backend.
    Supported: 'mtcnn', 'retinaface', 'opencv', 'ssd', 'dlib', 'mediapipe'
    """
    return name

def detect_faces(detector_name, img):
    """
    Detect faces from a NumPy array (OpenCV image) or file path.
    """
    try:
        if isinstance(img, np.ndarray):
            # Convert NumPy array to PIL Image
            pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_img.save(tmp.name)
                temp_path = tmp.name
            # Detect faces
            faces = DeepFace.detect_faces(
                img_path=temp_path,
                detector_backend=detector_name
            )
            # Clean up
            os.unlink(temp_path)
            return faces
        else:
            # Assume it's a file path
            return DeepFace.detect_faces(
                img_path=img,
                detector_backend=detector_name
            )
    except Exception as e:
        print(f"⚠️ Face detection failed: {e}")
        return []