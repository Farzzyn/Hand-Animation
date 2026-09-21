# Hand Animation & Tracking

Real-time hand tracking and visual effect warping using Google MediaPipe and OpenCV.

## Requirements

- Python 3.10+ (Recommended Python 3.11 or 3.12)
- Webcam

## Installation

1. Clone or open this repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

*(Or manually: `pip install opencv-python mediapipe numpy`)*

## Running the Application

### Option 1: Run Python Script (Recommended)
```bash
python main.py
```

### Option 2: Run via Jupyter Notebook
Open [prototype_01.ipynb](prototype_01.ipynb) in VS Code or Jupyter Lab / Notebook and run the cells sequentially.

## Controls
- Ensure your webcam is connected.
- Hold two hands in front of the camera (thumb, index, and middle fingertips form warped effect regions).
- Press **`q`** or **`ESC`** to exit the video window.
