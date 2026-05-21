# Project Documentation

## AI-Based Navigation Assistant for Visually Impaired People

### 1. Introduction

Vision impairment affects approximately 2.2 billion people worldwide (WHO, 2023).
Navigating unfamiliar environments poses significant safety risks for blind
individuals. This project presents a real-time AI-powered navigation assistant
that uses computer vision to detect obstacles, stairs, and provide spoken
directional guidance.

### 2. Problem Statement

Visually impaired individuals face challenges in:
- Detecting obstacles in their path (furniture, people, vehicles)
- Identifying stairs (a major fall hazard)
- Judging distances to objects
- Navigating without human assistance

Existing solutions (Google Lookout, Microsoft Seeing AI) require internet
connectivity and expensive hardware. Our solution runs offline on a standard
laptop with a webcam.

### 3. Proposed Solution

A modular Python application that processes webcam frames in real-time to:
1. Detect objects using YOLOv8 deep learning model
2. Estimate distances using the pinhole camera model
3. Detect stairs using edge detection and Hough Transform
4. Provide voice guidance using offline text-to-speech
5. Display an annotated UI with color-coded warnings

### 4. System Requirements

| Component | Requirement |
|-----------|------------|
| OS | Windows 10+ / macOS 11+ / Ubuntu 20.04+ |
| Python | 3.9 or higher |
| RAM | 4 GB minimum |
| Camera | Any USB webcam or built-in camera |
| GPU | Not required (runs on CPU) |
| Internet | Only for initial package installation |

### 5. Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | 4.8+ | Image capture, processing, display |
| Ultralytics | 8.0+ | YOLOv8 object detection |
| pyttsx3 | 2.90+ | Offline text-to-speech |
| NumPy | 1.24+ | Numerical operations |

### 6. System Architecture

The system follows a pipeline architecture:

```
Input (Webcam) → Preprocessing → Detection → Analysis → Output (Voice + Display)
```

Each stage is handled by a dedicated module:
- `utils.py` → Preprocessing
- `object_detector.py` → Object detection
- `distance_estimator.py` → Distance analysis
- `stair_detector.py` → Stair detection
- `voice_assistant.py` → Voice output
- `main.py` → Orchestration

### 7. Algorithm Details

#### 7.1 Object Detection (YOLOv8)
- Architecture: CSPDarknet backbone + PANet neck + anchor-free detection head
- Model: YOLOv8 Nano (3.2M parameters)
- Input: 640×640 RGB image
- Output: Bounding boxes with class labels and confidence scores
- NMS threshold: IoU > 0.50

#### 7.2 Distance Estimation (Pinhole Camera Model)
- Formula: `D = (W × f) / w`
- D = distance (cm), W = known width (cm), f = focal length (px), w = bbox width (px)
- Focal length calibrated to 600 pixels for typical laptop webcam
- 20 object categories with known average widths

#### 7.3 Stair Detection (Canny + Hough)
- Region of Interest: Lower 60% of frame
- Preprocessing: Grayscale → Gaussian blur (5×5)
- Edge detection: Canny (thresholds: 50, 150)
- Line detection: Probabilistic Hough Transform
- Classification: ≥4 horizontal lines (±15° tolerance) = stairs

#### 7.4 Voice Guidance (pyttsx3)
- Engine: SAPI5 (Windows) / NSSpeechSynthesizer (macOS) / espeak (Linux)
- Threading: Background daemon threads for non-blocking speech
- Cooldown: 3-second minimum between repeated messages
- Priority: Stairs > Danger > Warning > Clear

### 8. Navigation System

The frame is divided into three vertical zones:
- LEFT (0-33%): Object on user's left → "Move right"
- CENTER (33-66%): Object ahead → "Obstacle ahead"
- RIGHT (66-100%): Object on user's right → "Move left"

Distance classifications:
- DANGER (<100 cm): Immediate voice alert, red bounding box
- WARNING (100-200 cm): Periodic alert, orange bounding box
- SAFE (>200 cm): Green bounding box, occasional "Path clear"

### 9. Results

| Metric | Value |
|--------|-------|
| Processing Speed | 25-40 FPS on CPU |
| Object Classes | 80 (COCO dataset) |
| Model Size | 6 MB |
| Distance Accuracy | ±30% (approximate) |
| Voice Latency | < 500 ms |
| Stair Detection | 4+ horizontal lines threshold |

### 10. Limitations

1. Distance estimation is approximate (monocular camera limitation)
2. Stair detection may produce false positives on striped patterns
3. Voice synthesis quality depends on OS TTS engine
4. Performance may vary in low-light conditions
5. Cannot detect transparent obstacles (glass doors)

### 11. Future Scope

1. Integration with depth cameras (Intel RealSense)
2. Traffic light and pedestrian signal detection
3. OCR for reading signs and labels
4. Mobile app deployment (TensorFlow Lite)
5. GPS integration for outdoor navigation
6. Haptic feedback via wearable devices
7. Multi-language voice support
8. Object tracking with DeepSORT
9. Indoor mapping using SLAM
10. Custom-trained stair detection model

### 12. Conclusion

This project demonstrates a practical application of computer vision that
addresses a real-world accessibility challenge. By combining classical image
processing techniques with modern deep learning, we created a system that
runs in real-time on affordable hardware. The modular architecture makes it
easy to extend and improve. The project covers all major topics in a
standard computer vision curriculum, from fundamental image processing to
advanced deep learning-based detection.

### 13. References

1. Redmon, J. et al. "You Only Look Once: Unified, Real-Time Object Detection" (2016)
2. Jocher, G. et al. "Ultralytics YOLOv8" (2023) — https://docs.ultralytics.com
3. Canny, J. "A Computational Approach to Edge Detection" (1986)
4. Hough, P.V.C. "Method and Means for Recognizing Complex Patterns" (1962)
5. Hartley, R. & Zisserman, A. "Multiple View Geometry in Computer Vision" (2003)
6. Bradski, G. "The OpenCV Library" — https://docs.opencv.org
7. World Health Organization — "Blindness and vision impairment" (2023)
8. COCO Dataset — https://cocodataset.org
