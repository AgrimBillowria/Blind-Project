"""
object_detector.py — Real-Time Object Detection using YOLOv8
=============================================================
This module loads a YOLOv8 Nano model and runs inference on each frame.

Syllabus concepts covered:
  • Image Segmentation & Advanced CV → deep-learning-based object detection
  • Object Localization              → bounding boxes, confidence scores
  • Scene Understanding              → identifying objects in the environment

How YOLOv8 works (simplified):
  1. The input image is resized and passed through a CNN backbone.
  2. The backbone extracts feature maps at multiple scales.
  3. A detection head predicts bounding boxes, class probabilities,
     and objectness scores for each grid cell.
  4. Non-Maximum Suppression (NMS) removes duplicate detections.
  5. We get a list of (x1, y1, x2, y2, confidence, class_id) results.
"""

from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE,
    YOLO_IOU_THRESHOLD,
    COLOR_DANGER,
    COLOR_WARNING,
    COLOR_SAFE,
    FONT_SCALE,
    FONT_THICKNESS,
    DANGER_DISTANCE,
    WARNING_DISTANCE,
)
import cv2


# ─────────────────────────────────────────────────────────────
# Load the YOLO model once at module level (singleton pattern)
# ─────────────────────────────────────────────────────────────
# YOLOv8n is the "nano" variant — only ~3.2 M parameters.
# It runs in real-time even on CPU-only laptops.

print("[INFO] Loading YOLOv8 model...")
model = YOLO(YOLO_MODEL_PATH)
print("[INFO] YOLOv8 model loaded successfully.")


def detect_objects(frame):
    """
    Run YOLOv8 inference on a single frame.

    Parameters:
        frame (numpy.ndarray): A BGR image from OpenCV.

    Returns:
        detections (list of dict): Each dict contains:
            - "label"      : str   — class name (e.g., "person")
            - "confidence" : float — detection confidence (0-1)
            - "bbox"       : tuple — (x1, y1, x2, y2) pixel coordinates
            - "x_center"   : int   — horizontal center of the bounding box
            - "y_center"   : int   — vertical center of the bounding box
            - "width"      : int   — bounding box width in pixels
            - "height"     : int   — bounding box height in pixels

    How this function works step-by-step:
        1. We call model(frame) which internally:
           a. Converts the frame to RGB (YOLO expects RGB, OpenCV gives BGR).
           b. Resizes to 640×640.
           c. Normalizes pixel values to [0, 1].
           d. Runs the neural network forward pass.
           e. Applies NMS to filter overlapping boxes.
        2. We iterate through the results and extract bounding boxes.
        3. We filter by confidence threshold.
        4. We return a clean list of dictionaries.
    """
    # Run inference with the configured thresholds
    # verbose=False suppresses YOLO's own print statements
    results = model(
        frame,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU_THRESHOLD,
        verbose=False
    )

    detections = []

    # results is a list (one per image); we only pass one image
    for result in results:
        # result.boxes contains all detected bounding boxes
        boxes = result.boxes

        for box in boxes:
            # ── Extract bounding box coordinates ──
            # box.xyxy gives [[x1, y1, x2, y2]] in pixel coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # ── Extract confidence score ──
            confidence = float(box.conf[0].cpu().numpy())

            # ── Extract class ID and map to class name ──
            class_id = int(box.cls[0].cpu().numpy())
            label = model.names[class_id]  # e.g., "person", "chair"

            # ── Compute bounding box properties ──
            width = x2 - x1
            height = y2 - y1
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })

    return detections


def draw_detections(frame, detections, distances=None):
    """
    Draw bounding boxes and labels on the frame.

    Parameters:
        frame       : BGR image to draw on (modified in-place)
        detections  : list of detection dicts from detect_objects()
        distances   : optional dict mapping index → estimated distance (cm)

    Color coding:
        • Red    → danger  (distance < 100 cm)
        • Orange → warning (distance < 200 cm)
        • Green  → safe    (distance >= 200 cm or unknown)

    Syllabus link: Object Localization — visualizing bounding boxes.
    """
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        # ── Choose color based on estimated distance ──
        color = COLOR_SAFE  # default green
        if distances and i in distances:
            dist = distances[i]
            if dist < DANGER_DISTANCE:
                color = COLOR_DANGER    # Red — too close
            elif dist < WARNING_DISTANCE:
                color = COLOR_WARNING   # Orange — caution
            else:
                color = COLOR_SAFE      # Green — safe

        # ── Draw the bounding box ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Build the label text ──
        text = f"{label} {conf:.0%}"
        if distances and i in distances:
            text += f" | {distances[i]:.0f}cm"

        # ── Draw a filled background for the label ──
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)

        # ── Draw the label text in white ──
        cv2.putText(
            frame, text, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
            (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
        )

    return frame
