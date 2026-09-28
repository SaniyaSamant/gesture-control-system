# Hand Gesture Control System

A real-time hand gesture recognition system with two modes — **PC Control** and **PowerPoint Presentation Control** — using MediaPipe, OpenCV, and machine learning models.

## Modes

### 1. PC Control Mode (`pc-control/`)
Control your computer using hand gestures via your webcam. A trained **Random Forest** model recognizes gestures and triggers system actions in real time through a Flask web interface.

| Gesture | Action |
|---------|--------|
| Index | Play / Pause media |
| Next | Next tab (Ctrl+Shift+Tab) |
| Prev | Previous tab (Ctrl+Tab) |
| Palm | Show desktop (Win+D) |
| ThumbUp | Take screenshot |
| ThumbDown | Open browser (Google) |
| Two | Show motivational alert |

### 2. PowerPoint Presentation Mode (`ppt-control/`)
Control PowerPoint slideshows hands-free using a **deep learning model (Keras/LSTM)**. Upload a `.pptx` file through the web interface and navigate slides with gestures.

| Gesture | Action |
|---------|--------|
| Next | Next slide (→) |
| Prev | Previous slide (←) |
| ThumbUp | Increase volume |
| ThumbDown | Decrease volume |
| Index | Increase brightness |
| Two | Decrease brightness |
| Palm | Take screenshot |

---

## Tech Stack

- **Computer Vision**: MediaPipe Hands, OpenCV
- **ML Models**: Random Forest (pc-control), Keras Deep Learning (ppt-control)
- **Backend**: Flask
- **System Automation**: PyAutoGUI, screen-brightness-control
- **PowerPoint Automation**: comtypes, python-pptx

---

## Project Structure

```
gesture-control-system/
├── pc-control/
│   ├── app.py              # Flask app — webcam stream + gesture inference
│   ├── train.py            # Train Random Forest model
│   └── templates/
│       └── index.html
│
├── ppt-control/
│   ├── app.py              # Flask app — upload PPT + gesture control
│   ├── train.py            # Train Keras model
│   ├── dataset.py          # Dataset collection script
│   └── templates/
│       ├── index.html
│       ├── upload.html
│       ├── presentation.html
│       └── instructions.html
│
└── models/
    ├── random_forest_gesture_model.pkl   # PC control model
    ├── gesture_model.h5                  # PPT control model
    └── label_encoder.pkl                 # Label encoder for PPT model
```

---

## Setup

### Prerequisites
- Python 3.8+
- Webcam
- Windows (for PowerPoint automation in ppt-control)

### Install Dependencies

**PC Control:**
```bash
cd pc-control
pip install flask opencv-python mediapipe numpy scikit-learn pyautogui
```

**PPT Control:**
```bash
cd ppt-control
pip install flask opencv-python mediapipe numpy tensorflow pyautogui screen-brightness-control comtypes
```

---

## Running the Apps

### PC Control
```bash
cd pc-control
python app.py
```
Open `http://localhost:5000` and allow webcam access.

### PPT Control
```bash
cd ppt-control
python app.py
```
Open `http://localhost:5000`, upload a `.pptx` file, and control your slides with gestures.

---

## Training the Models

### PC Control (Random Forest)
```bash
cd pc-control
python train.py
# Outputs: random_forest_gesture_model.pkl (move to models/)
```

### PPT Control (Keras)
```bash
cd ppt-control
python dataset.py   # Collect gesture data
python train.py     # Train the model
# Outputs: gesture_model.h5, label_encoder.pkl (move to models/)
```

---

## How It Works

1. **MediaPipe** detects the hand and extracts 21 landmark coordinates (x, y, z) per frame — a 63-dimensional feature vector.
2. The feature vector is fed into the trained model for gesture classification.
3. The predicted gesture triggers a mapped system action via PyAutoGUI or comtypes.
4. A 4-second debounce prevents repeated accidental triggers.

---

## Screenshots

> Add screenshots or a demo GIF here.
