# Step-by-Step Implementation Guide

This guide walks through how to build this project from scratch.
Use this for understanding the development process and for viva explanation.

---

## Phase 1: Project Setup

### Step 1.1 — Create the project folder
```bash
mkdir -p ~/Desktop/Blind!
cd ~/Desktop/Blind!
```

### Step 1.2 — Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
```

### Step 1.3 — Install required libraries
```bash
pip install opencv-python ultralytics pyttsx3 numpy
```

### Step 1.4 — Create the file structure
```
Blind!/
├── main.py
├── config.py
├── object_detector.py
├── stair_detector.py
├── distance_estimator.py
├── voice_assistant.py
├── utils.py
└── requirements.txt
```

**Why modular?**
Each module handles ONE responsibility. This makes the code:
- Easy to test individually
- Easy to explain in viva
- Easy to maintain and extend

---

## Phase 2: Configuration (config.py)

### Step 2.1 — Define camera settings
```python
CAMERA_INDEX = 0        # Webcam ID
FRAME_WIDTH = 640       # Standard processing width
FRAME_HEIGHT = 480      # Standard processing height
```

### Step 2.2 — Define YOLOv8 settings
```python
YOLO_MODEL_PATH = "yolov8n.pt"  # Nano model for speed
YOLO_CONFIDENCE = 0.45          # 45% minimum confidence
```

### Step 2.3 — Define distance estimation constants
```python
FOCAL_LENGTH = 600              # Pixels (from pinhole model)
KNOWN_WIDTHS = {                # Real-world object widths in cm
    "person": 45,
    "chair": 50,
    ...
}
```

**Teacher note:** The focal length is derived from the pinhole camera
model. For a typical laptop webcam, it ranges from 500-700 pixels.

---

## Phase 3: Utilities (utils.py)

### Step 3.1 — Image preprocessing functions
Build helper functions that demonstrate fundamental CV concepts:

1. `resize_frame()` — Resizes camera frames to standard 640×480
2. `convert_to_grayscale()` — BGR to single-channel intensity image
3. `apply_gaussian_blur()` — Noise reduction using Gaussian kernel
4. `enhance_edges()` — Sharpening via Laplacian kernel convolution
5. `apply_canny_edges()` — Canny edge detection (4-stage algorithm)

### Step 3.2 — Feature detection (for syllabus)
6. `detect_orb_features()` — ORB keypoints (FAST + BRIEF)
7. `draw_orb_features()` — Visualize keypoints on frame

### Step 3.3 — Navigation helpers
8. `get_object_zone()` — Classify object position as LEFT/CENTER/RIGHT
9. `get_direction_instruction()` — Convert zone to spoken instruction

### Step 3.4 — UI drawing functions
10. `draw_zone_lines()` — Draw vertical zone dividers
11. `draw_status_bar()` — Translucent bar with navigation text
12. `draw_title_bar()` — Translucent header with app title

---

## Phase 4: Object Detection (object_detector.py)

### Step 4.1 — Load the YOLOv8 model
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Loads once, reused for every frame
```

### Step 4.2 — Write the detection function
```python
def detect_objects(frame):
    results = model(frame, conf=0.45, verbose=False)
    # Extract bounding boxes, labels, confidence scores
    # Return list of detection dictionaries
```

### Step 4.3 — Write the drawing function
```python
def draw_detections(frame, detections, distances):
    # Color-code by distance: Red/Orange/Green
    # Draw bounding box + label + distance text
```

**Key concept:** YOLO divides the image into a grid, predicts bounding
boxes per cell, and uses NMS to remove duplicates — all in one pass.

---

## Phase 5: Distance Estimation (distance_estimator.py)

### Step 5.1 — Implement the pinhole formula
```python
def estimate_distance(label, bbox_width):
    known_width = KNOWN_WIDTHS.get(label, 50)
    distance = (known_width * FOCAL_LENGTH) / bbox_width
    return distance  # in cm
```

### Step 5.2 — Add helper functions
- `estimate_all_distances()` — Process all detections at once
- `get_closest_object()` — Find the nearest obstacle
- `classify_distance()` — Categorize as DANGER/WARNING/SAFE

**Teacher note:** This directly demonstrates the pinhole camera model
from the Camera Geometry syllabus topic.

---

## Phase 6: Stair Detection (stair_detector.py)

### Step 6.1 — Design the algorithm
1. Crop the lower 60% of frame (ROI — stairs are on the ground)
2. Convert to grayscale
3. Apply Gaussian blur
4. Run Canny edge detection
5. Apply Hough Line Transform
6. Filter for horizontal lines (±15° tolerance)
7. Count lines — if ≥ 4, stairs detected

### Step 6.2 — Implement the detection
```python
def detect_stairs(frame):
    # ... (algorithm above)
    return stairs_detected, stair_region, line_count
```

### Step 6.3 — Add visual warning
```python
def draw_stair_warning(frame, stair_region):
    # Semi-transparent magenta overlay on stair region
    # Bold "STAIRS DETECTED" warning text
```

**Key concepts:** Canny edge detection + Hough Transform — both are
core syllabus topics demonstrated in a practical context.

---

## Phase 7: Voice Assistant (voice_assistant.py)

### Step 7.1 — Initialize pyttsx3
```python
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 160)    # Words per minute
engine.setProperty('volume', 1.0)  # Max volume
```

### Step 7.2 — Implement cooldown mechanism
```python
_last_spoken = {}  # Track when each message was last spoken
COOLDOWN = 3.0     # Seconds between repeated messages
```

### Step 7.3 — Implement threaded speech
```python
def speak(message):
    # Check cooldown → skip if too recent
    # Run TTS in background thread → don't block video
```

### Step 7.4 — Priority-based message generation
```python
def generate_navigation_message(detections, distances, stairs):
    # Priority: Stairs > Danger > Warning > Clear
    # Returns (message, priority_level)
```

---

## Phase 8: Main Loop (main.py)

### Step 8.1 — Open webcam
```python
cap = cv2.VideoCapture(0)
```

### Step 8.2 — Process each frame
```python
while True:
    ret, frame = cap.read()
    frame = resize_frame(frame)

    detections = detect_objects(frame)
    distances = estimate_all_distances(detections)
    stairs, region, _ = detect_stairs(frame)
    message, priority = generate_navigation_message(...)

    speak(message)
    draw_detections(frame, detections, distances)
    cv2.imshow("AI Navigation Assistant", frame)
```

### Step 8.3 — Handle keyboard input
- `q` → Quit
- `f` → Toggle ORB feature visualization
- `s` → Save screenshot

---

## Phase 9: Testing & Demo

### Test each module independently:
```python
# Test object detector
python -c "from object_detector import detect_objects; print('OK')"

# Test distance estimation
python -c "from distance_estimator import estimate_distance; print(estimate_distance('person', 150))"

# Test voice
python -c "from voice_assistant import speak; speak('Hello')"
```

### Run the full system:
```bash
python main.py
```

### Demo checklist:
- [ ] Objects detected with bounding boxes
- [ ] Distance shown on labels
- [ ] Color coding: red/orange/green
- [ ] Zone lines visible
- [ ] Voice alerts triggered
- [ ] Stair detection works (test by pointing camera at stairs)
- [ ] ORB features toggle with 'f'
- [ ] Screenshot saves with 's'

---

## Phase 10: Documentation & Submission

1. Write README.md (included in project)
2. Prepare PPT slides (see docs/ppt_content.md)
3. Study viva questions (see docs/viva_questions.md)
4. Push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit: AI Navigation Assistant"
git remote add origin https://github.com/YOUR_USERNAME/blind-navigation-assistant.git
git push -u origin main
```
