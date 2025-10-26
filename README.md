VIDEO-MOOD-STRESS
A beginner-friendly real-time facial emotion and stress detection system powered by deep learning. This project uses your webcam to recognize emotions and estimate stress, showing live predictions in a simple interactive dashboard.

Features
Real-time emotion detection from video using deep learning

Stress level estimation and visualization

Interactive dashboard built with Streamlit

Modular, well-documented Python code

Folder Structure
VIDEO-MOOD-STRESS/
├── backend/
│   ├── ml/
│   │   ├── emotion_model.py
│   │   ├── face_detector.py
│   │   ├── stress_score.py
│   │   └── __init__.py
│   └── utils/
│       └── video_utils.py
│   └── main.py
├── frontend/
│   ├── components/
│   │   ├── emotion_chart.py
│   │   ├── live_feed.py
│   │   └── stress_meter.py
│   └── app.py
├── models/
│   └── emotion_model.h5
├── requirements.txt
├── README.md



Requirements
Python 3.11 or 3.12 (Not 3.13)
Webcam

Installation
Clone the repository

git clone https://github.com/Kumbarak007/video-mood-stress

cd VIDEO-MOOD-STRESS

Create a virtual environment (optional but recommended)

python -m venv myenv
source myenv/bin/activate 

Install dependencies

pip install -r requirements.txt

How to Run
Make sure models/emotion_model.h5 exists. Place a pre-trained emotion model here if it's missing.

Start the frontend dashboard:

streamlit run frontend/app.py
The web browser will open (default: http://localhost:8501). Grant camera access if prompted.

See your real-time video, emotion predictions, and stress chart.

How It Works
Face Detection: Uses OpenCV to locate faces in webcam frames.

Emotion Recognition: Loads a deep learning model (emotion_model.h5) with DeepFace and TensorFlow to predict your emotions.

Stress Estimation: Applies logic (in stress_score.py) to estimate stress from detected emotions.

Visualization: The Streamlit dashboard displays live video, graphs, and stress meters.

Customization
Improve accuracy by training or replacing models/emotion_model.h5 with your own model.

Edit components (frontend/components/) to add new features or change the web dashboard layout.

Troubleshooting
Webcam issues: Ensure no other application is using your camera.

Dependency errors: Double-check your Python version and reinstall with pip install -r requirements.txt.

Performance: A machine with a dedicated GPU may provide smoother real-time results.

Credits
Uses open-source libraries: OpenCV, Streamlit, DeepFace, TensorFlow.

Based on open datasets/models for facial emotion recognition.

License
This project is for educational and research use.
