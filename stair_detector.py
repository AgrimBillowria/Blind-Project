"""
stair_detector.py — Stair Detection Using Edge Detection & Hough Lines
========================================================================
Detects stairs by finding multiple horizontal lines in a region of interest.

Syllabus: Edge Detection (Canny), Feature Detection (Hough Transform),
          Image Processing (blur, grayscale), Contour Analysis
"""

import cv2
import numpy as np
from config import (
    STAIR_CANNY_LOW, STAIR_CANNY_HIGH,
    STAIR_MIN_LINES, STAIR_HOUGH_THRESHOLD,
    STAIR_ANGLE_TOLERANCE, COLOR_STAIR,
)


def detect_stairs(frame):
    """
    Detect stairs in a frame using edge detection + Hough Line Transform.

    Algorithm:
      1. Crop the lower 60% of the frame (stairs are usually on the ground).
      2. Convert to grayscale.
      3. Apply Gaussian blur to reduce noise.
      4. Run Canny edge detection to find edges.
      5. Use Hough Line Transform to find straight lines.
      6. Filter for nearly-horizontal lines (within ±15° of horizontal).
      7. If we find >= STAIR_MIN_LINES horizontal lines → stairs detected.

    Returns:
        stairs_detected (bool): True if stairs are found
        stair_region    (tuple): (y_start, y_end) of the ROI, or None
        line_count      (int): Number of horizontal lines found
    """
    h, w = frame.shape[:2]

    # Step 1: Region of Interest — lower 60% of frame
    y_start = int(h * 0.4)
    roi = frame[y_start:h, :]

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian blur to smooth noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Canny edge detection
    edges = cv2.Canny(blurred, STAIR_CANNY_LOW, STAIR_CANNY_HIGH)

    # Step 5: Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,                          # Distance resolution (pixels)
        theta=np.pi / 180,             # Angle resolution (radians)
        threshold=STAIR_HOUGH_THRESHOLD,
        minLineLength=w * 0.3,         # Lines must be at least 30% of frame width
        maxLineGap=20                   # Allow small gaps in lines
    )

    if lines is None:
        return False, None, 0

    # Step 6: Filter for horizontal lines
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle of the line
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Keep lines within tolerance of horizontal (0° or 180°)
        if angle < STAIR_ANGLE_TOLERANCE or angle > (180 - STAIR_ANGLE_TOLERANCE):
            horizontal_lines.append(line[0])

    # Step 7: Check if enough horizontal lines → stairs
    line_count = len(horizontal_lines)
    stairs_detected = line_count >= STAIR_MIN_LINES

    if stairs_detected:
        return True, (y_start, h), line_count
    return False, None, line_count


def draw_stair_warning(frame, stair_region):
    """Draw a stair warning overlay on the frame."""
    if stair_region is None:
        return frame

    h, w = frame.shape[:2]
    y_start, y_end = stair_region

    # Draw a semi-transparent magenta overlay on the stair region
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_end), COLOR_STAIR, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Draw border
    cv2.rectangle(frame, (0, y_start), (w - 1, y_end - 1), COLOR_STAIR, 2)

    # Warning text
    cv2.putText(
        frame, "!! STAIRS DETECTED !!",
        (w // 2 - 130, y_start + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_STAIR, 2, cv2.LINE_AA
    )

    return frame
