# Syllabus Alignment — Computer Vision Concepts Mapping

This document maps every CV syllabus topic to the exact location in our project code.

---

## 1. Fundamentals of Image Processing

### Grayscale Conversion
- **File:** `utils.py` → `convert_to_grayscale()`
- **File:** `stair_detector.py` → line `gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)`
- **Concept:** Uses luminosity formula: `gray = 0.299R + 0.587G + 0.114B`
- **Why:** Reduces computational cost from 3 channels to 1 channel

### Image Preprocessing (Resize)
- **File:** `utils.py` → `resize_frame()`
- **Concept:** Bilinear interpolation to resize frames to 640×480
- **Why:** Standardizes input for all processing modules

### Gaussian Blur
- **File:** `utils.py` → `apply_gaussian_blur()`
- **File:** `stair_detector.py` → blur before Canny
- **Concept:** 2D Gaussian kernel convolution for noise reduction
- **Why:** Removes high-frequency noise that causes false edge detections

### Filtering
- **File:** `utils.py` → `enhance_edges()` — sharpening filter
- **Concept:** 2D convolution with a custom 3×3 kernel
- **Why:** Demonstrates kernel-based image filtering

### Edge Enhancement
- **File:** `utils.py` → `enhance_edges()`
- **Concept:** Laplacian-style sharpening kernel boosts edge pixels
- **Why:** Makes edges more prominent for downstream detection

---

## 2. Camera Geometry

### Webcam as Pinhole Camera
- **File:** `distance_estimator.py` → entire module
- **File:** `config.py` → `FOCAL_LENGTH`
- **Concept:** The webcam is modeled as a pinhole camera with a single focal point
- **Formula:** `distance = (real_width × focal_length) / pixel_width`

### Object Size Variation with Distance
- **File:** `distance_estimator.py` → `estimate_distance()`
- **Concept:** Objects appear smaller as they move farther away (inverse relationship)
- **Why:** Bounding box width is inversely proportional to distance

### Perspective Concepts
- **File:** `config.py` → `KNOWN_WIDTHS` dictionary
- **Concept:** Known real-world dimensions combined with apparent image size → depth

---

## 3. Motion and Scene Understanding

### Relative Object Movement
- **File:** `main.py` → main loop processes each frame independently
- **Concept:** By processing consecutive frames, we observe how objects move relative to the camera
- **Why:** Real-time processing enables understanding of dynamic scenes

### Environmental Perception
- **File:** `voice_assistant.py` → `generate_navigation_message()`
- **Concept:** Combining multiple detection outputs into a single coherent scene description
- **Why:** The system "understands" its environment and provides actionable guidance

---

## 4. Feature Detection

### Edge Detection (Canny)
- **File:** `stair_detector.py` → `cv2.Canny(blurred, 50, 150)`
- **File:** `utils.py` → `apply_canny_edges()`
- **Concept:** Multi-stage edge detector (gradient, NMS, hysteresis thresholding)
- **Why:** Foundation of stair detection pipeline

### Hough Line Transform
- **File:** `stair_detector.py` → `cv2.HoughLinesP()`
- **Concept:** Transforms edge pixels to parameter space to find straight lines
- **Why:** Detects the horizontal edges of stair steps

### ORB / FAST Feature Detection
- **File:** `utils.py` → `detect_orb_features()`
- **Concept:** FAST corner detection + BRIEF binary descriptors + orientation
- **Why:** Demonstrates keypoint detection (toggle with 'f' key in the app)

---

## 5. Color Processing

### RGB Frames
- **File:** All modules work with BGR frames (OpenCV's default color order)
- **Concept:** Each pixel has 3 channels (Blue, Green, Red) with values 0-255
- **File:** `config.py` → Color constants defined in BGR format

### Grayscale Conversion
- **File:** `utils.py` → `convert_to_grayscale()`
- **Concept:** Weighted sum of RGB channels produces single-channel intensity image
- **Why:** Required for edge detection, ORB features, and reducing computation

---

## 6. Image Segmentation and Advanced CV

### Object Localization
- **File:** `object_detector.py` → YOLOv8 bounding boxes
- **Concept:** Localizing objects in the image using (x1, y1, x2, y2) coordinates
- **Why:** Knowing WHERE objects are is essential for navigation guidance

### Scene Understanding
- **File:** `voice_assistant.py` → combines zone position + distance + stair status
- **File:** `main.py` → orchestrates all modules into coherent scene analysis
- **Concept:** Higher-level understanding of the spatial layout
- **Why:** The system interprets the scene to provide useful guidance

### Modern Deep Learning Detection
- **File:** `object_detector.py` → YOLOv8 (convolutional neural network)
- **Concept:** End-to-end trainable detector using CSPDarknet backbone + PANet neck + detection head
- **Why:** State-of-the-art accuracy and speed for real-time detection

---

## Summary Table

| # | Syllabus Topic | Module | Function/Line |
|---|---------------|--------|---------------|
| 1 | Grayscale | utils.py | `convert_to_grayscale()` |
| 2 | Gaussian Blur | utils.py | `apply_gaussian_blur()` |
| 3 | Edge Enhancement | utils.py | `enhance_edges()` |
| 4 | Canny Edges | stair_detector.py | `cv2.Canny()` |
| 5 | Hough Lines | stair_detector.py | `cv2.HoughLinesP()` |
| 6 | Pinhole Model | distance_estimator.py | `estimate_distance()` |
| 7 | Object Detection | object_detector.py | `detect_objects()` |
| 8 | ORB Features | utils.py | `detect_orb_features()` |
| 9 | Color Processing | utils.py, config.py | BGR/grayscale |
| 10 | Scene Understanding | voice_assistant.py | `generate_navigation_message()` |
| 11 | Localization | object_detector.py | Bounding box extraction |
| 12 | Deep Learning | object_detector.py | YOLOv8 CNN |
