# Detailed Setup Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10+ |
| RAM | 4 GB | 8 GB |
| Storage | 500 MB | 1 GB |
| Camera | Any webcam | HD webcam (720p+) |
| OS | Windows 10 / macOS 11 / Ubuntu 20.04 | Latest version |
| GPU | Not required | NVIDIA GPU (optional, for speed) |

---

## Step-by-Step Installation

### 1. Install Python

**macOS:**
```bash
brew install python@3.10
```

**Windows:**
Download from https://python.org/downloads and check "Add to PATH".

**Ubuntu:**
```bash
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip
```

### 2. Create the Project Folder

```bash
mkdir -p ~/Desktop/Blind!
cd ~/Desktop/Blind!
```

### 3. Set Up Virtual Environment

```bash
python3 -m venv venv
```

Activate it:
```bash
# macOS/Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `opencv-python` — Computer vision library
- `ultralytics` — YOLOv8 framework
- `pyttsx3` — Offline text-to-speech
- `numpy` — Numerical computing

### 5. Verify Installation

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import pyttsx3; print('pyttsx3 OK')"
```

### 6. Download YOLOv8 Model

The model (`yolov8n.pt`) is downloaded automatically when you first run the app.
It's only ~6 MB (Nano variant).

To download manually:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("Model downloaded!")
```

### 7. Test Your Webcam

```python
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("Webcam works! Frame size:", frame.shape)
else:
    print("ERROR: Cannot access webcam")
cap.release()
```

### 8. Run the Application

```bash
python main.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot open webcam" | Check camera permissions in System Preferences / Settings |
| `ModuleNotFoundError` | Make sure virtual environment is activated |
| pyttsx3 crashes on macOS | Run: `pip install pyobjc` |
| Slow FPS | Close other applications, reduce `FRAME_WIDTH` in config.py |
| Model download fails | Check internet connection, or download from ultralytics.com |
| Black window | Try changing `CAMERA_INDEX` to 1 in config.py |

---

## macOS-Specific Notes

If pyttsx3 gives errors on macOS, install the Objective-C bridge:
```bash
pip install pyobjc
```

Grant camera permissions:
- Go to **System Preferences → Privacy & Security → Camera**
- Allow Terminal / your IDE

---

## Windows-Specific Notes

If you get `espeak` errors with pyttsx3:
```bash
pip install pyttsx3==2.90
```

The SAPI5 engine is used by default on Windows, which should work out of the box.
