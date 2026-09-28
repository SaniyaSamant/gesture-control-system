from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
import pickle
import pyautogui
import screen_brightness_control as sbc
import mediapipe as mp
from tensorflow.keras.models import load_model
import pythoncom
import comtypes.client
import threading
import time

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and label encoder with error handling
model_path = os.path.abspath("gesture_control_ppt\gesture_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path)

label_encoder_path = os.path.abspath("gesture_control_ppt\label_encoder.pkl")
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to extract landmarks
def extract_landmarks(hand_landmarks):
    return np.array([coord for landmark in hand_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)])

# Function to open PowerPoint in fullscreen mode
def open_ppt(filepath):
    pythoncom.CoInitialize()
    try:
        ppt_app = comtypes.client.CreateObject("PowerPoint.Application")
        ppt_app.Visible = 1
        presentation = ppt_app.Presentations.Open(os.path.abspath(filepath))
        presentation.SlideShowSettings.StartingSlide = 1
        presentation.SlideShowSettings.EndingSlide = presentation.Slides.Count
        presentation.SlideShowSettings.AdvanceMode = 1  # Manual slide control
        presentation.SlideShowWindow.View.GotoSlide(1)
        return ppt_app, presentation
    except Exception as e:
        print(f"Error opening PowerPoint: {e}")
    finally:
        pythoncom.CoUninitialize()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/instructions')
def instructions():
    return render_template("instructions.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['ppt']
        if file and file.filename.endswith(".pptx"):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('presentation', filename=file.filename))
    return render_template("upload.html")

@app.route('/presentation/<filename>')
def presentation(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Open PPT & Start Gesture Control in separate threads
    threading.Thread(target=open_ppt, args=(filepath,), daemon=True).start()
    threading.Thread(target=gesture_control, daemon=True).start()
    
    return render_template("presentation.html")

# Gesture control function
def gesture_control():
    cap = cv2.VideoCapture(0)
    last_action_time = time.time()
    action_delay = 0.5  # Prevents rapid consecutive actions
    confidence_threshold = 0.85  # Adjust confidence threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = extract_landmarks(hand_landmarks).reshape(1, -1)

                # Proper Normalization
                if np.max(landmarks) != 0:
                   landmarks = landmarks / np.max(landmarks)

                prediction = model.predict(landmarks)
                gesture_index = np.argmax(prediction)
                gesture_name = label_encoder.inverse_transform([gesture_index])[0]

                if time.time() - last_action_time > action_delay:
                    if gesture_name == "Prev":
                        print("Gesture detected: Next Slide")
                        pyautogui.press("right")  # Next Slide
                    elif gesture_name == "Next":
                        print("Gesture detected: Previous Slide")
                        pyautogui.press("left")  # Previous Slide
                    elif gesture_name == "ThumbUp":
                        print("Gesture detected: Increase Volume")
                        pyautogui.press("volumeup")  # Increase Volume
                    elif gesture_name == "ThumbDown":
                        print("Gesture detected: Decrease Volume")
                        pyautogui.press("volumedown")  # Decrease Volume
                    elif gesture_name == "Index":
                        print("Gesture detected: Increase Brightness")
                        current_brightness = sbc.get_brightness()[0]
                        sbc.set_brightness(min(current_brightness + 10, 100))
                    elif gesture_name == "Two":
                        print("Gesture detected: Decrease Brightness")
                        current_brightness = sbc.get_brightness()[0]
                        sbc.set_brightness(max(current_brightness - 10, 0))
                    elif gesture_name == "Palm":
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        screenshot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"screenshot_{timestamp}.png")
                        pyautogui.screenshot(screenshot_path)
                        print(f"Screenshot saved: {screenshot_path}")

                    last_action_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Serving static files (CSS, images, etc.)
@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True)
