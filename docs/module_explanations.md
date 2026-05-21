# Module Explanations — Line-by-Line Guide

This document explains the key code sections in each module for viva preparation.

---

## 1. config.py — Configuration

**Purpose:** Single source of truth for all parameters.

**Key concepts:**
- `FOCAL_LENGTH = 600` — This represents the camera's focal length in pixels. It comes from the pinhole camera model. A typical laptop webcam has a focal length around 500-700 pixels.
- `KNOWN_WIDTHS` — A dictionary mapping object names to their real-world widths in centimeters. For example, a person's shoulder width is ~45 cm.

**Why a separate config file?**
- Allows tuning without changing logic
- Makes the codebase maintainable
- Professional software engineering practice

---

## 2. object_detector.py — YOLOv8 Detection

### Model Loading
```python
model = YOLO(YOLO_MODEL_PATH)
```
- Loads YOLOv8 Nano (~3.2M parameters, ~6 MB)
- Loaded once at module import (singleton pattern)
- Nano variant chosen for real-time CPU performance

### detect_objects() — Step by Step
```python
results = model(frame, conf=YOLO_CONFIDENCE, iou=YOLO_IOU_THRESHOLD, verbose=False)
```
1. **Input:** BGR frame from OpenCV
2. **Internal processing by YOLO:**
   - Converts BGR → RGB
   - Resizes to 640×640
   - Normalizes pixels to [0,1]
   - Forward pass through CNN
   - Non-Maximum Suppression (NMS)
3. **Output:** List of bounding boxes with class IDs and confidence scores

```python
x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
```
- `box.xyxy` → bounding box in (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
- `.cpu().numpy()` → converts from PyTorch tensor to NumPy array
- `.astype(int)` → rounds to integer pixel coordinates

```python
label = model.names[class_id]
```
- YOLO has 80 COCO classes (person, car, chair, etc.)
- `model.names` is a dictionary: `{0: "person", 1: "bicycle", ...}`

---

## 3. distance_estimator.py — Pinhole Model

### The Core Formula
```python
distance = (known_width * FOCAL_LENGTH) / bbox_width
```

**Derivation from pinhole camera model:**
```
Real World          Camera Sensor
                    (Image Plane)
    W ─────────┐
    │           │        w (pixels)
    │     D     │──────▶ │
    │           │        │
    W ─────────┘        f (focal length)

By similar triangles:
    w / f = W / D

Rearranging:
    D = (W × f) / w
```

Where:
- `W` = real object width (cm) → `known_width`
- `f` = focal length (pixels) → `FOCAL_LENGTH`
- `w` = bounding box width (pixels) → `bbox_width`
- `D` = distance (cm) → result

**Example:** A person (W=45cm) with bounding box width of 150 pixels:
```
D = (45 × 600) / 150 = 180 cm = 1.8 meters
```

---

## 4. stair_detector.py — Edge Detection + Hough Lines

### Algorithm Step by Step:

**Step 1: Region of Interest (ROI)**
```python
y_start = int(h * 0.4)
roi = frame[y_start:h, :]
```
- Only look at the lower 60% of the frame
- Stairs are always on the ground, not in the sky

**Step 2: Grayscale + Blur**
```python
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```
- Grayscale: reduces 3-channel image to 1-channel
- Gaussian blur: smooths noise that would create false edges

**Step 3: Canny Edge Detection**
```python
edges = cv2.Canny(blurred, 50, 150)
```
- Low threshold (50): edges below this are discarded
- High threshold (150): edges above this are strong edges
- Edges between thresholds are kept only if connected to strong edges

**Step 4: Hough Line Transform**
```python
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, ...)
```
- Transforms edge pixels to Hough space (rho, theta)
- Accumulator votes for line parameters
- Lines with enough votes are detected

**Step 5: Filter for Horizontal Lines**
```python
angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
if angle < 15 or angle > 165:  # nearly horizontal
```
- Stairs have horizontal edges (the step edges)
- We filter out vertical/diagonal lines
- If ≥4 horizontal lines → stairs detected

---

## 5. voice_assistant.py — TTS with Cooldown

### Cooldown Mechanism
```python
if message in _last_spoken:
    elapsed = current_time - _last_spoken[message]
    if elapsed < VOICE_COOLDOWN:
        return  # Too soon — skip
```
- Without cooldown: "Obstacle ahead" would repeat 30 times/second
- With 3-second cooldown: says it once, waits 3 seconds before repeating

### Threading
```python
thread = threading.Thread(target=_speak_threaded, args=(message,), daemon=True)
thread.start()
```
- TTS runs in a background thread so the video doesn't freeze
- `daemon=True` means the thread dies when the main program exits
- A `threading.Lock` prevents two messages from playing simultaneously

### Priority System
```
1. STAIR   → "Caution, stairs detected ahead"
2. DANGER  → "Warning! [object] [direction], very close"
3. WARNING → "Caution, [object] [direction]"
4. CLEAR   → "Path is clear"
```

---

## 6. utils.py — Image Processing Toolkit

### Gaussian Blur
```python
cv2.GaussianBlur(frame, (5, 5), 0)
```
- `(5, 5)` = kernel size (5×5 pixel neighborhood)
- `0` = sigma (auto-calculated from kernel size)
- Reduces noise while preserving edges

### Edge Enhancement (Sharpening)
```python
kernel = [[ 0, -1,  0],
          [-1,  5, -1],
          [ 0, -1,  0]]
```
- Center weight (5) keeps original pixel intensity
- Negative neighbors subtract surrounding blur
- Result: edges become sharper and more defined

### ORB Feature Detection
```python
orb = cv2.ORB_create(nfeatures=200)
keypoints, descriptors = orb.detectAndCompute(gray, None)
```
- ORB = Oriented FAST + Rotated BRIEF
- FAST: detects corners quickly
- BRIEF: creates a binary descriptor for each keypoint
- Useful for feature matching, not directly used in navigation

### Zone Classification
```python
if relative_x < 0.33: return "LEFT"
elif relative_x > 0.66: return "RIGHT"
else: return "CENTER"
```
- Frame divided into 3 equal vertical strips
- Object position determines navigation instruction

---

## 7. main.py — The Main Loop

### Pipeline Order (each frame):
1. `cap.read()` — Capture frame from webcam
2. `resize_frame()` — Standardize resolution
3. `detect_objects()` — YOLOv8 inference
4. `estimate_all_distances()` — Pinhole distance calculation
5. `detect_stairs()` — Edge-based stair detection (every 3rd frame)
6. `generate_navigation_message()` — Determine what to say
7. `speak()` — Trigger voice alert (async)
8. `draw_*()` — Render UI overlays
9. `cv2.imshow()` — Display annotated frame

### FPS Calculation
```python
fps = 1.0 / (current_time - prev_time + 1e-6)
```
- Time difference between consecutive frames
- `+ 1e-6` prevents division by zero
- Displayed on the UI for performance monitoring
