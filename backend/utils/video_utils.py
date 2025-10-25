# backend/utils/video_utils.py
import cv2

def get_webcam_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None

def draw_label(frame, text, coords, color=(0, 255, 0)):
    x, y = coords
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
