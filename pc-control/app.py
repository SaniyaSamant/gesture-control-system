from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import time
import datetime
import random
import webbrowser
import os

app = Flask(__name__)

# Create a 'screenshot' folder if it doesn't exist
if not os.path.exists('screenshot'):
    os.makedirs('screenshot')

# Load trained Random Forest model
with open('random_forest_gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture to action mapping
gesture_actions = {
    'Index': lambda: pyautogui.press('playpause'),
    'Prev': lambda: pyautogui.hotkey('ctrl', 'tab'),
    'Palm': lambda: pyautogui.hotkey('win', 'd'),
    'Next': lambda: pyautogui.hotkey('ctrl', 'shift', 'tab'),
    'ThumbDown': lambda: webbrowser.open('https://www.google.com'),
    'ThumbUp': lambda: pyautogui.screenshot(f'screenshot/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'),
    'Two': lambda: pyautogui.alert(random.choice([ 
        "Believe in yourself!", 
        "Keep pushing!", 
        "You are capable of amazing things.", 
        "Make it happen!" 
    ]))
}

last_gesture = None
last_action_time = 0
delay_seconds = 4

def gen_frames():
    global last_gesture, last_action_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if len(coords) == 63:
                prediction = model.predict([coords])[0]

                current_time = time.time()
                if prediction != last_gesture or (current_time - last_action_time > delay_seconds):
                    action = gesture_actions.get(prediction)
                    if action:
                        action()
                        last_action_time = current_time
                        last_gesture = prediction

                cv2.putText(image, f'Gesture: {prediction}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
