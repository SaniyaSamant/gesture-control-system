import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hand module

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# File to save dataset
CSV_FILE = "gesture_dataset.csv"

# Column names for CSV (21 landmarks, each with x, y, z)
columns = ["gesture"] + [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]

# Check if the file exists, if not, create it with headers
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.to_csv(CSV_FILE, index=False)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

gesture_label = input("Enter gesture label: ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save data to CSV
            row = [gesture_label] + landmarks
            df.loc[len(df)] = row
            df.to_csv(CSV_FILE, index=False)

    # Display the frame
    cv2.putText(frame, f"Collecting data for: {gesture_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Collection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()