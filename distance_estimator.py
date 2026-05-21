"""
distance_estimator.py — Approximate Distance Estimation
=========================================================
Estimates the distance to detected objects using the pinhole camera model.

Syllabus concepts covered:
  • Camera Geometry     → pinhole camera model, focal length
  • Perspective         → objects appear smaller as they move further away
  • Motion & Scene      → understanding spatial layout of the scene

──────────────────────────────────────────────────────────────────────
THE PINHOLE CAMERA MODEL (explained simply)
──────────────────────────────────────────────────────────────────────

A webcam can be modeled as a pinhole camera. The key relationship is:

    pixel_width     focal_length
    ──────────── = ──────────────
    real_width      real_distance

Rearranging:

    real_distance = (real_width × focal_length) / pixel_width

Where:
  • real_width   = known average width of the object (in cm)
  • focal_length = camera focal length (in pixels) — a constant
  • pixel_width  = width of the bounding box in the image (in pixels)
  • real_distance = estimated distance from camera to object (in cm)

This is an APPROXIMATION. It assumes:
  1. The object is roughly facing the camera.
  2. The object's width matches our assumed average.
  3. Lens distortion is minimal.

For a student project, this level of accuracy is perfectly acceptable
and clearly demonstrates understanding of camera geometry.
──────────────────────────────────────────────────────────────────────
"""

from config import FOCAL_LENGTH, KNOWN_WIDTHS, DEFAULT_OBJECT_WIDTH


def estimate_distance(label, bbox_width):
    """
    Estimate the distance from the camera to an object.

    Parameters:
        label      (str) : The object class name (e.g., "person")
        bbox_width (int) : Width of the bounding box in pixels

    Returns:
        distance   (float) : Estimated distance in centimeters

    Step-by-step:
        1. Look up the known real-world width for this object class.
        2. Apply the pinhole formula:
               distance = (known_width × focal_length) / bbox_width
        3. Return the estimated distance in cm.

    Example:
        A "person" has known width = 45 cm.
        If their bounding box is 150 pixels wide:
            distance = (45 × 600) / 150 = 180 cm ≈ 1.8 meters
    """
    # Avoid division by zero (edge case: very tiny or glitched bounding box)
    if bbox_width <= 0:
        return float('inf')

    # Step 1: Look up known width (or use default)
    known_width = KNOWN_WIDTHS.get(label, DEFAULT_OBJECT_WIDTH)

    # Step 2: Apply the pinhole camera formula
    distance = (known_width * FOCAL_LENGTH) / bbox_width

    return distance


def estimate_all_distances(detections):
    """
    Estimate distances for all detections in a frame.

    Parameters:
        detections (list of dict): Output from object_detector.detect_objects()

    Returns:
        distances (dict): Maps detection index → estimated distance in cm

    This function iterates through every detection and calls
    estimate_distance() for each one.
    """
    distances = {}

    for i, det in enumerate(detections):
        label = det["label"]
        bbox_width = det["width"]
        dist = estimate_distance(label, bbox_width)
        distances[i] = dist

    return distances


def get_closest_object(detections, distances):
    """
    Find the closest detected object.

    Parameters:
        detections (list): List of detection dicts
        distances  (dict): Maps index → distance in cm

    Returns:
        (detection_dict, distance, index) or (None, float('inf'), -1)

    This is used to determine the most urgent obstacle for voice alerts.
    """
    if not detections or not distances:
        return None, float('inf'), -1

    # Find the index with the minimum distance
    closest_idx = min(distances, key=distances.get)
    closest_dist = distances[closest_idx]
    closest_det = detections[closest_idx]

    return closest_det, closest_dist, closest_idx


def classify_distance(distance_cm):
    """
    Classify a distance into a danger level.

    Returns:
        "DANGER"  — less than 100 cm (1 meter)
        "WARNING" — less than 200 cm (2 meters)
        "SAFE"    — 200 cm or more

    These thresholds are also defined in config.py.
    """
    if distance_cm < 100:
        return "DANGER"
    elif distance_cm < 200:
        return "WARNING"
    else:
        return "SAFE"
