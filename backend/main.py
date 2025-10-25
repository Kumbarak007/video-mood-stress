"""
Command-Line Interface for Emotion & Stress Detection
Usage:
    python main.py --image path/to/image.jpg
    python main.py --video 0               # Webcam
    python main.py --video path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
import os
from backend.ml.emotion_model import load_emotion_model
from backend.ml.face_detector import get_detector, detect_faces
from backend.ml.stress_score import estimate_stress

# Constants
MODEL_PATH = "frontend/models/emotion_model.h5"  # Point to frontend/models/
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def process_frame(frame, model, detector):
    """Process a single frame and return annotated frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detect_faces(detector, rgb)
    
    for face_info in detections:
        region = face_info.get('region')
        if region is None:
            continue
            
        x, y, w, h = region
        face_img = rgb[y:y + h, x:x + w]
        if face_img.size == 0:
            continue
            
        face_img = cv2.resize(face_img, (96, 96)) / 255.0
        pred = model.predict(np.expand_dims(face_img, 0), verbose=0)[0]
        emotion = EMOTIONS[np.argmax(pred)]
        stress = estimate_stress(region, face_info.get('keypoints', {}))
        
        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{emotion}, Stress: {stress:.1f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return frame

def run_on_image(image_path, model, detector):
    """Run inference on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    annotated = process_frame(frame, model, detector)
    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, annotated)
    print(f"✅ Result saved to: {output_path}")

def run_on_video(video_source, model, detector):
    """Run inference on video stream (webcam or file)."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Could not open video source")
        return
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated = process_frame(frame, model, detector)
        cv2.imshow('Emotion & Stress Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Emotion & Stress Detection CLI")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please place 'emotion_model.h5' in the 'frontend/models/' folder.")
        return
    
    print("Loading model...")
    model = load_emotion_model(MODEL_PATH)
    detector = get_detector("mtcnn")
    print("✅ Model loaded successfully")
    
    if args.image:
        run_on_image(args.image, model, detector)
    else:
        video_source = int(args.video) if args.video.isdigit() else args.video
        run_on_video(video_source, model, detector)

if __name__ == "__main__":
    main()