# AI-Based Navigation Assistant for Visually Impaired People

> **Using Computer Vision and Deep Learning to Help the Blind Navigate Safely**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [Module Descriptions](#module-descriptions)
8. [Syllabus Alignment](#syllabus-alignment)
9. [Screenshots](#screenshots)
10. [Future Scope](#future-scope)
11. [Contributors](#contributors)

---

## Overview

This project is a **real-time AI-powered navigation assistant** designed to help visually impaired individuals move safely through indoor and outdoor environments. Using nothing more than a standard webcam or chest-mounted camera, the system:

- **Detects obstacles** (people, chairs, tables, vehicles, etc.) using YOLOv8
- **Estimates approximate distances** using the pinhole camera model
- **Detects stairs** using classical edge detection and Hough Line Transform
- **Provides spoken navigation guidance** via offline text-to-speech
- **Displays a professional annotated UI** with bounding boxes, warnings, and directions

The project demonstrates a practical application of computer vision that is **socially impactful**, **technically sound**, and **aligned with academic CV syllabi**.

---

## Features

| Feature | Technology | Description |
|---------|-----------|-------------|
| Object Detection | YOLOv8 Nano | Real-time detection of 80+ object classes |
| Distance Estimation | Pinhole Camera Model | Approximate distance from bounding box width |
| Stair Detection | Canny + Hough Lines | Finds horizontal edges indicating stairs |
| Voice Guidance | pyttsx3 (offline) | Spoken alerts with cooldown to avoid repetition |
| Zone Navigation | Frame Division | LEFT/CENTER/RIGHT zones for directional guidance |
| ORB Features | OpenCV ORB | Optional keypoint visualization (press 'f') |
| Screenshot | OpenCV imwrite | Press 's' to save annotated frames |

---

## Architecture

```
┌──────────────┐
│   WEBCAM     │
│  (Input)     │
└──────┬───────┘
       │ frame
       ▼
┌──────────────┐    ┌───────────────────┐
│  Resize &    │───▶│  Object Detector  │──── detections
│  Preprocess  │    │  (YOLOv8 Nano)    │
│  (utils.py)  │    └───────────────────┘
└──────┬───────┘              │
       │                      ▼
       │             ┌──────────────────┐
       │             │   Distance       │──── distances (cm)
       │             │   Estimator      │
       │             │  (pinhole model) │
       │             └──────────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐    ┌───────────────────┐
│    Stair     │    │  Voice Assistant  │──── spoken alerts
│   Detector   │    │  (pyttsx3)        │
│ (Canny+Hough)│    └───────────────────┘
└──────┬───────┘              │
       │                      │
       ▼                      ▼
┌─────────────────────────────────────────┐
│          ANNOTATED DISPLAY              │
│  (bounding boxes, zones, status bar)    │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
Blind!/
├── main.py                 # Entry point — main loop
├── config.py               # All tunable parameters
├── object_detector.py      # YOLOv8-based object detection
├── stair_detector.py       # Stair detection (Canny + Hough)
├── distance_estimator.py   # Pinhole model distance estimation
├── voice_assistant.py      # Text-to-speech guidance
├── utils.py                # Image processing utilities
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── docs/
    ├── setup_guide.md      # Detailed setup instructions
    ├── architecture.md     # System architecture deep dive
    ├── module_explanations.md  # Line-by-line code explanations
    ├── syllabus_alignment.md   # CV syllabus mapping
    ├── viva_questions.md   # Viva Q&A preparation
    └── ppt_content.md      # Presentation slide content
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- A working webcam
- macOS / Windows / Linux

### Step 1: Clone or Navigate to the Project

```bash
cd /path/to/Blind!
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download YOLOv8 Model

The model downloads automatically on first run. Alternatively:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Downloads ~6 MB nano model
```

### Step 5: Run the Application

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `f` | Toggle ORB feature visualization |
| `s` | Save a screenshot |

---

## Module Descriptions

### `config.py`
Central configuration. All thresholds, colors, and model paths are defined here. Change values without touching any logic.

### `object_detector.py`
Loads YOLOv8 Nano and runs inference. Returns a list of detections with bounding boxes, class labels, and confidence scores.

### `distance_estimator.py`
Uses the pinhole camera formula `distance = (known_width × focal_length) / pixel_width` to estimate how far each object is.

### `stair_detector.py`
Applies Canny edge detection and Hough Line Transform to the lower 60% of the frame. If multiple horizontal lines are found, stairs are flagged.

### `voice_assistant.py`
Offline TTS using pyttsx3. Runs in a background thread with a cooldown timer to prevent repeating the same alert too frequently.

### `utils.py`
Image processing helpers: grayscale conversion, Gaussian blur, edge enhancement, ORB feature detection, zone classification, and UI drawing functions.

### `main.py`
Orchestrates the entire pipeline. Captures frames, runs all modules, generates navigation messages, and displays the annotated output.

---

## Syllabus Alignment

| Syllabus Topic | Where It's Used |
|---------------|-----------------|
| Grayscale Conversion | `utils.py` → `convert_to_grayscale()` |
| Gaussian Blur | `utils.py` → `apply_gaussian_blur()`, `stair_detector.py` |
| Edge Enhancement | `utils.py` → `enhance_edges()` (sharpening kernel) |
| Canny Edge Detection | `stair_detector.py` → stair detection pipeline |
| Hough Transform | `stair_detector.py` → finding horizontal lines |
| Pinhole Camera Model | `distance_estimator.py` → distance formula |
| Object Localization | `object_detector.py` → YOLOv8 bounding boxes |
| Deep Learning Detection | `object_detector.py` → YOLOv8 CNN |
| ORB/FAST Features | `utils.py` → `detect_orb_features()` |
| Color Processing | `utils.py` → BGR/grayscale conversion |
| Scene Understanding | `voice_assistant.py` → navigation message generation |
| Image Segmentation | `object_detector.py` → object localization |

---

## Future Scope

1. **Depth Camera Integration** — Use Intel RealSense for true depth estimation
2. **Traffic Light Detection** — Recognize pedestrian signals for road crossing
3. **Text/Sign Reading** — OCR with Tesseract for reading signs and labels
4. **GPS Integration** — Combine with GPS for outdoor route planning
5. **Mobile App** — Port to Android/iOS using TensorFlow Lite
6. **Haptic Feedback** — Vibration wristband for silent alerts
7. **Indoor Mapping** — SLAM-based room mapping for familiar environments
8. **Multi-language Support** — Voice alerts in Hindi, Spanish, etc.
9. **Cloud Processing** — Offload heavy models to edge servers for speed
10. **Wearable Design** — Custom 3D-printed camera mount for glasses

---

## Contributors

| Name | Role |
|------|------|
| [Your Name] | Developer & Researcher |
| [Guide Name] | Project Guide |

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

> *Built with ❤️ for accessibility and social impact.*
