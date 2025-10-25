def estimate_stress(box, keypoints=None):
    """
    Estimate stress score based on facial geometry.
    Simple heuristic: higher eyebrow position + tighter mouth = higher stress.
    """
    if not keypoints:
        return 0.0
    
    # Extract key points (if available)
    left_eye = keypoints.get('left_eye', [0, 0])
    right_eye = keypoints.get('right_eye', [0, 0])
    nose = keypoints.get('nose', [0, 0])
    mouth_left = keypoints.get('mouth_left', [0, 0])
    mouth_right = keypoints.get('mouth_right', [0, 0])
    
    # Calculate simple stress indicators
    eye_height = abs(left_eye[1] - right_eye[1])  # Vertical eye separation
    mouth_width = abs(mouth_right[0] - mouth_left[0])  # Mouth width
    face_width = box[2]  # Width of bounding box
    
    # Normalize
    eye_ratio = eye_height / face_width if face_width > 0 else 0
    mouth_ratio = mouth_width / face_width if face_width > 0 else 0
    
    # Stress score: higher eye movement + smaller mouth = higher stress
    stress = (eye_ratio * 0.7) + ((1 - mouth_ratio) * 0.3)
    
    return min(max(stress * 100, 0), 100)  # Scale to 0-100