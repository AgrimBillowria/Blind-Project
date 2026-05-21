"""
utils.py — Utility Functions for Image Processing
===================================================
This module contains helper functions used across the project.

Syllabus concepts covered:
  • Image Processing Fundamentals → grayscale conversion, Gaussian blur, edge enhancement
  • Color Processing              → RGB ↔ BGR, grayscale conversion
  • Feature Detection             → Canny edge detection, ORB keypoint detection
"""

import cv2
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT


# ─────────────────────────────────────────────────────────────
# 1. Frame Preprocessing
# ─────────────────────────────────────────────────────────────

def resize_frame(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """
    Resize a frame to a standard resolution.

    Why we do this:
      • Ensures consistent processing speed regardless of camera resolution.
      • All downstream modules expect the same frame dimensions.

    Syllabus link: Image Processing — resizing / interpolation.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def convert_to_grayscale(frame):
    """
    Convert a BGR frame to grayscale.

    How it works:
      OpenCV uses the luminosity formula:
        gray = 0.299×R + 0.587×G + 0.114×B
      This weights green most heavily because the human eye is most
      sensitive to green light.

    Syllabus link: Color Processing — grayscale conversion.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(frame, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to reduce noise.

    How it works:
      A 2D Gaussian kernel is convolved with the image. Each pixel
      becomes a weighted average of its neighbours, where weights
      follow a bell curve. Larger kernel → more smoothing.

    Syllabus link: Image Processing — Gaussian blur / filtering.
    """
    return cv2.GaussianBlur(frame, kernel_size, 0)


def enhance_edges(frame):
    """
    Enhance edges using a sharpening kernel.

    How it works:
      We convolve the image with a 3×3 Laplacian-style kernel:
        [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]]
      The center weight (5) keeps the original pixel strong while
      the negative neighbours subtract the blur, boosting edges.

    Syllabus link: Image Processing — edge enhancement / filtering.
    """
    # Define a 3×3 sharpening kernel
    sharpen_kernel = np.array([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]
    ], dtype=np.float32)

    # Apply the kernel via 2D convolution
    sharpened = cv2.filter2D(frame, -1, sharpen_kernel)
    return sharpened


def apply_canny_edges(gray_frame, low_threshold=50, high_threshold=150):
    """
    Detect edges using the Canny edge detector.

    How it works (4 stages):
      1. Gaussian smoothing (already applied if you pre-blur)
      2. Gradient computation via Sobel in X and Y
      3. Non-maximum suppression — thin edges to 1-pixel width
      4. Hysteresis thresholding — keep strong edges, extend to weak
         edges that are connected to strong ones.

    Parameters:
      low_threshold  — edges below this gradient magnitude are discarded
      high_threshold — edges above this are kept unconditionally

    Syllabus link: Feature Detection — edge detection (Canny).
    """
    return cv2.Canny(gray_frame, low_threshold, high_threshold)


# ─────────────────────────────────────────────────────────────
# 2. Feature Detection (Optional — for demonstration)
# ─────────────────────────────────────────────────────────────

def detect_orb_features(gray_frame, max_features=200):
    """
    Detect ORB (Oriented FAST and Rotated BRIEF) keypoints.

    How it works:
      • FAST detector finds corner-like points rapidly.
      • BRIEF descriptor computes a binary string for each keypoint.
      • ORB adds rotation invariance via intensity centroid orientation.

    This function is included to demonstrate feature detection for
    the syllabus. It is NOT used in the main navigation pipeline
    but can be enabled for demo purposes.

    Syllabus link: Feature Detection — ORB / FAST keypoints.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)
    return keypoints, descriptors


def draw_orb_features(frame, keypoints):
    """
    Draw detected ORB keypoints on the frame for visualization.
    """
    return cv2.drawKeypoints(
        frame, keypoints, None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


# ─────────────────────────────────────────────────────────────
# 3. Zone Classification
# ─────────────────────────────────────────────────────────────

def get_object_zone(x_center, frame_width):
    """
    Determine which zone an object falls in: LEFT, CENTER, or RIGHT.

    The frame is split into three equal vertical strips:
      ┌────────┬────────┬────────┐
      │  LEFT  │ CENTER │ RIGHT  │
      │ 0–33%  │ 33–66% │ 66–100%│
      └────────┴────────┴────────┘

    This is used to give directional guidance:
      • Object in LEFT zone   → "Move right"
      • Object in CENTER zone → "Obstacle ahead"
      • Object in RIGHT zone  → "Move left"

    Syllabus link: Scene Understanding — spatial reasoning.
    """
    relative_x = x_center / frame_width

    if relative_x < 0.33:
        return "LEFT"
    elif relative_x > 0.66:
        return "RIGHT"
    else:
        return "CENTER"


def get_direction_instruction(zone):
    """
    Convert a zone label into a spoken navigation instruction.
    """
    if zone == "LEFT":
        return "Move right"
    elif zone == "RIGHT":
        return "Move left"
    else:
        return "Obstacle ahead"


# ─────────────────────────────────────────────────────────────
# 4. Drawing Helpers
# ─────────────────────────────────────────────────────────────

def draw_zone_lines(frame):
    """
    Draw vertical zone dividers on the frame for visualization.
    """
    h, w = frame.shape[:2]
    left_x = int(w * 0.33)
    right_x = int(w * 0.66)

    cv2.line(frame, (left_x, 0), (left_x, h), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (right_x, 0), (right_x, h), (100, 100, 100), 1, cv2.LINE_AA)

    # Zone labels at the top
    cv2.putText(frame, "LEFT", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, "CENTER", (left_x + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, "RIGHT", (right_x + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


def draw_status_bar(frame, status_text, color=(0, 255, 0)):
    """
    Draw a translucent status bar at the bottom of the frame.

    Uses addWeighted to blend the overlay — demonstrates alpha blending.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw a filled rectangle at the bottom
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)

    # Blend the overlay with the original frame (50% opacity)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Put the status text
    cv2.putText(frame, status_text, (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return frame


def draw_title_bar(frame, title="AI Navigation Assistant"):
    """
    Draw a translucent title bar at the top of the frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, title, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame
