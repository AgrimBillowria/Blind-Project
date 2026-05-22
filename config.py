"""
config.py — Central Configuration for the Navigation Assistant
==============================================================
All tunable parameters are stored here so you can adjust the system
without editing logic in other modules.

Syllabus concepts covered:
  • Camera Geometry  → FOCAL_LENGTH, KNOWN_WIDTHS (pinhole model)
  • Image Processing → FRAME_WIDTH, FRAME_HEIGHT (resolution settings)
"""

# ─────────────────────────────────────────────
# 1. Camera / Frame Settings
# ─────────────────────────────────────────────
CAMERA_INDEX = 0                # Webcam index (0 = built-in laptop camera on Windows)
FRAME_WIDTH = 640               # Processing width in pixels
FRAME_HEIGHT = 480              # Processing height in pixels

# ─────────────────────────────────────────────
# 2. YOLOv8 Model Settings
# ─────────────────────────────────────────────
YOLO_MODEL_PATH = "yolov8n.pt"  # YOLOv8 Nano — fast, lightweight
YOLO_CONFIDENCE = 0.45          # Minimum confidence to accept a detection
YOLO_IOU_THRESHOLD = 0.50      # Non-max suppression IoU threshold

# ─────────────────────────────────────────────
# 3. Distance Estimation Parameters
# ─────────────────────────────────────────────
# Approximate focal length (pixels) — calibrated for a typical laptop webcam.
# This comes from the pinhole camera model:  f = (pixel_width × real_distance) / real_width
FOCAL_LENGTH = 600

# Known average widths of common objects in centimeters.
# Used with the pinhole formula: distance = (KNOWN_WIDTH × FOCAL_LENGTH) / pixel_width
KNOWN_WIDTHS = {
    "person":     45,   # average shoulder width
    "chair":      50,
    "table":      90,
    "car":       180,
    "truck":     250,
    "bicycle":    60,
    "motorcycle": 80,
    "dog":        40,
    "cat":        25,
    "bottle":      8,
    "cup":        8,
    "laptop":     35,
    "cell phone": 7,
    "backpack":   35,
    "handbag":    30,
    "suitcase":   50,
    "umbrella":   100,
    "bus":       250,
    "bench":     120,
    "tv":         80,
    "book":       20,
}

# Default width (cm) for objects not in the dictionary above.
DEFAULT_OBJECT_WIDTH = 50

# ─────────────────────────────────────────────
# 4. Navigation / Zone Settings
# ─────────────────────────────────────────────
# The frame is divided into 3 vertical zones: LEFT | CENTER | RIGHT
# These fractions define the boundaries.
LEFT_ZONE = 0.33               # x < 33% of frame → object is on the left
RIGHT_ZONE = 0.66              # x > 66% of frame → object is on the right

# Distance thresholds (in cm)
DANGER_DISTANCE = 100           # < 1 m  → immediate warning
WARNING_DISTANCE = 200          # < 2 m  → caution
SAFE_DISTANCE = 300             # > 3 m  → generally safe

# ─────────────────────────────────────────────
# 5. Stair Detection Parameters
# ─────────────────────────────────────────────
STAIR_CANNY_LOW = 50            # Canny edge detector — low threshold
STAIR_CANNY_HIGH = 150          # Canny edge detector — high threshold
STAIR_MIN_LINES = 4             # Minimum horizontal lines to classify as stairs
STAIR_HOUGH_THRESHOLD = 80      # Hough line accumulator threshold
STAIR_ANGLE_TOLERANCE = 15      # Degrees from horizontal to still count as "horizontal"

# ─────────────────────────────────────────────
# 6. Voice Assistant Settings
# ─────────────────────────────────────────────
VOICE_COOLDOWN = 3.0            # Seconds between repeated voice alerts
VOICE_RATE = 160                # Words per minute for TTS engine
VOICE_VOLUME = 1.0              # Volume (0.0 to 1.0)

# ─────────────────────────────────────────────
# 7. UI / Display Settings
# ─────────────────────────────────────────────
# Colors in BGR (OpenCV format)
COLOR_DANGER = (0, 0, 255)      # Red
COLOR_WARNING = (0, 165, 255)   # Orange
COLOR_SAFE = (0, 255, 0)        # Green
COLOR_STAIR = (255, 0, 255)     # Magenta
COLOR_INFO = (255, 255, 0)      # Cyan
COLOR_ZONE_LINE = (100, 100, 100)  # Gray for zone dividers

FONT_SCALE = 0.55
FONT_THICKNESS = 2

# Window title
WINDOW_TITLE = "AI Navigation Assistant — Blind!"
