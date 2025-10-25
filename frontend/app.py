# frontend_app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import time

# --- Configuration ---
MODEL_PATH = "models/emotion_model.h5"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Emotion icons for better UX
EMOTION_ICONS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

# --- Page Configuration ---
st.set_page_config(
    page_title="MindCare - Mood & Stress Detection",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern, Calming Design ---
st.markdown("""
<style>
    /* Main color scheme - wellness oriented */
    :root {
        --primary-blue: #4A90E2;
        --soft-green: #7BC96F;
        --calm-teal: #5DADE2;
        --neutral-grey: #F5F7FA;
        --text-dark: #2C3E50;
        --warning-amber: #F39C12;
        --danger-red: #E74C3C;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background: linear-gradient(to bottom right, #E8F5E9, #E3F2FD);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 3rem 2rem 2rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: var(--shadow-soft);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.1) rotate(180deg); }
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        padding: 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.75rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Privacy notice styling */
    .privacy-notice {
        background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 100%);
        border-left: 5px solid #7BC96F;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        box-shadow: var(--shadow-soft);
        animation: slideIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(123, 201, 111, 0.2);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .privacy-notice p {
        margin: 0;
        color: #2C3E50;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Result card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: var(--shadow-soft);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.8);
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 100%;
        animation: gradientSlide 3s ease infinite;
    }
    
    @keyframes gradientSlide {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-hover);
    }
    
    .result-card h3 {
        color: #2C3E50;
        margin-top: 0;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    .emotion-display {
        font-size: 4.5rem;
        text-align: center;
        margin: 1.5rem 0;
        animation: emotionPulse 2s ease-in-out infinite;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.1));
    }
    
    @keyframes emotionPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .emotion-label {
        font-size: 1.8rem;
        text-align: center;
        font-weight: 700;
        color: #2C3E50;
        text-transform: capitalize;
        letter-spacing: -0.5px;
    }
    
    .stress-indicator {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .stress-low {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #1B5E20;
        border: 3px solid #7BC96F;
        box-shadow: 0 4px 12px rgba(123, 201, 111, 0.3);
    }
    
    .stress-medium {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        color: #E65100;
        border: 3px solid #F39C12;
        box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);
    }
    
    .stress-high {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        color: #B71C1C;
        border: 3px solid #E74C3C;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
    }
    
    /* Confidence meter */
    .confidence-bar {
        background: linear-gradient(135deg, #E0E0E0 0%, #F5F5F5 100%);
        border-radius: 15px;
        height: 28px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .confidence-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 100%;
        height: 100%;
        transition: width 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        animation: gradientMove 3s ease infinite;
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.15rem;
        font-weight: 700;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.3px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 14px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow-hover);
    }
    
    /* Loading animation */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    .spinner {
        border: 4px solid #E0E0E0;
        border-top: 4px solid #4A90E2;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🧘 MindCare</h1>
    <p>Real-Time Mood & Stress Detection </p>
</div>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'show_privacy' not in st.session_state:
    st.session_state.show_privacy = True
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False

# --- Privacy Notice with Dismiss ---
if st.session_state.show_privacy:
    privacy_col1, privacy_col2 = st.columns([20, 1])
    with privacy_col1:
        st.markdown("""
        <div class="privacy-notice">
            <p>
                <span style="font-size: 1.2rem;">🔒</span>
                <strong>Your Privacy Matters:</strong> All video processing happens locally on your device. 
                Your data is protected with end-to-end encryption and is never stored or transmitted.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with privacy_col2:
        if st.button("✕", key="dismiss_privacy"):
            st.session_state.show_privacy = False
            st.rerun()

# --- Load Model (Cached) ---
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

# --- Load Face Detector (Cached) ---
@st.cache_resource
def load_face_detector():
    return MTCNN()

# --- Stress Estimation Function (from app.py) ---
def estimate_stress_from_emotion(emotion, confidence_scores):
    """
    Estimate stress based on emotion probabilities
    """
    stress_emotions = [0, 2, 5]  # angry, fear, sad
    stress_score = sum(confidence_scores[i] for i in stress_emotions)
    
    if stress_score > 0.5:
        return "High", "stress-high"
    elif stress_score > 0.25:
        return "Medium", "stress-medium"
    else:
        return "Low", "stress-low"

# --- Model Loading with Status ---
with st.spinner("🔄 Initializing AI models..."):
    model, error = load_emotion_model()
    if model is None:
        st.error(f"❌ **Error loading emotion detection model:**\n\n`{error}`")
        st.info("💡 Please ensure your trained model is placed in the `models/` folder with the correct path.")
        st.stop()
    
    detector = load_face_detector()
    st.success("✅ AI models loaded successfully!")

# --- Main Layout ---
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 📹 Live Video Feed")
    frame_placeholder = st.empty()
    
    # Control buttons
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        start_button = st.button("▶ Start Detection", type="primary", use_container_width=True)
    with button_col2:
        stop_button = st.button("⏹ Stop Detection", use_container_width=True)

with col2:
    st.markdown("### 📊 Detection Results")
    
    # Placeholders for results
    emotion_placeholder = st.empty()
    stress_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    # Initial state
    with emotion_placeholder.container():
        st.markdown("""
        <div class="result-card">
            <div class="emotion-display">🤖</div>
            <div class="emotion-label">Waiting to detect...</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stress_placeholder.container():
        st.markdown("""
        <div class="result-card">
            <h3>Stress Level</h3>
            <div class="stress-indicator" style="background: #F5F5F5; color: #757575; border: 2px solid #E0E0E0;">
                Not Detected
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with confidence_placeholder.container():
        st.markdown("""
        <div class="result-card">
            <h3>Confidence Level</h3>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: 0%;"></div>
            </div>
            <p style="text-align: center; color: #757575;">0%</p>
        </div>
        """, unsafe_allow_html=True)

# --- Detection Loop ---
if start_button:
    st.session_state.detection_active = True

if stop_button:
    st.session_state.detection_active = False

if st.session_state.detection_active:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ **Could not access webcam.** Please check your camera permissions and ensure no other application is using it.")
        st.session_state.detection_active = False
        st.stop()
    
    st.info("🎥 **Camera active.** Detection in progress... Press 'Stop Detection' to end.")
    
    frame_count = 0
    
    while st.session_state.detection_active:
        ret, frame = cap.read()
        
        if not ret:
            st.warning("⚠️ Failed to grab frame from camera.")
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces
            detections = detector.detect_faces(rgb_frame)
            
            detected_emotion = "None"
            emotion_icon = "🤖"
            stress_level = "Not Detected"
            stress_class = "stress-low"
            max_confidence = 0.0
            
            for detection in detections:
                x, y, w, h = detection['box']
                confidence = detection['confidence']
                
                if confidence < 0.9:
                    continue
                
                x, y = max(0, x), max(0, y)
                w, h = max(1, w), max(1, h)
                
                face = rgb_frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    continue
                
                # Preprocess face
                face_resized = cv2.resize(face, (96, 96))
                face_normalized = face_resized.astype('float32') / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)
                
                # Predict emotion
                predictions = model.predict(face_input, verbose=0)[0]
                emotion_idx = np.argmax(predictions)
                detected_emotion = EMOTIONS[emotion_idx]
                emotion_icon = EMOTION_ICONS.get(detected_emotion, "😐")
                max_confidence = predictions[emotion_idx]
                
                # Estimate stress
                stress_level, stress_class = estimate_stress_from_emotion(
                    detected_emotion, predictions
                )
                
                # Draw on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (123, 222, 111), 3)
                
                label = f"{detected_emotion.capitalize()} ({max_confidence*100:.1f}%)"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (123, 222, 111), 2)
                
                stress_label = f"Stress: {stress_level}"
                cv2.putText(frame, stress_label, (x, y+h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (74, 144, 226), 2)
            
            # Display frame
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            # Update results
            with emotion_placeholder.container():
                st.markdown(f"""
                <div class="result-card">
                    <div class="emotion-display">{emotion_icon}</div>
                    <div class="emotion-label">{detected_emotion.capitalize()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stress_placeholder.container():
                st.markdown(f"""
                <div class="result-card">
                    <h3>Stress Level</h3>
                    <div class="stress-indicator {stress_class}">
                        {stress_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with confidence_placeholder.container():
                confidence_pct = int(max_confidence * 100)
                st.markdown(f"""
                <div class="result-card">
                    <h3>Confidence Level</h3>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_pct}%;"></div>
                    </div>
                    <p style="text-align: center; color: #2C3E50; font-weight: 600;">{confidence_pct}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"⚠️ Detection error: {e}")
            continue
        
        frame_count += 1
        time.sleep(0.03)  # ~30 FPS
        
        # Check stop condition
        if not st.session_state.detection_active:
            break
    
    cap.release()
    st.success("✅ Detection stopped successfully")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
   <div class="info-box">
    <ol style="padding-left: 1.2rem; margin: 0; color: black;">
        <li>Click <strong>Start Detection</strong> to begin</li>
        <li>Position your face clearly in the camera frame</li>
        <li>The AI will analyze your facial expressions in real-time</li>
        <li>View your emotion and stress level on the right panel</li>
        <li>Click <strong>Stop Detection</strong> when finished</li>
    </ol>
</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 😊 Emotion Categories")
    for emotion, icon in EMOTION_ICONS.items():
        st.markdown(f"{icon} **{emotion.capitalize()}**")
    
    st.markdown("---")
    
    st.markdown("## 🎯 Stress Indicators")
    st.markdown("""
   <div style="margin-top: 1rem;">
    <div style="background: #E8F5E9; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 3px solid #7BC96F; color: black;">
        <strong>🟢 Low:</strong> Calm, positive state
    </div>
    <div style="background: #FFF3E0; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 3px solid #F39C12; color: black;">
        <strong>🟡 Medium:</strong> Mild stress detected
    </div>
    <div style="background: #FFEBEE; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 3px solid #E74C3C; color: black;">
        <strong>🔴 High:</strong> Elevated stress level
    </div>
</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## ⚙️ System Info")
    st.info(f"**Model:** `{MODEL_PATH}`")
    st.success("**Status:** Ready")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #757575; font-size: 0.85rem;">
        <p style="margin: 0;">Built with Tech Tycoons for mental wellness</p>
        <p style="margin: 0.5rem 0 0 0;">© 2025 MindCare AI</p>
    </div>
    """, unsafe_allow_html=True)