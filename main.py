"""
main.py — AI-Based Navigation Assistant for Visually Impaired People
=====================================================================
Main entry point. Captures webcam feed, runs all detection modules,
displays the annotated UI, and triggers voice guidance.

Usage:
  python main.py                  # Live webcam (default)
  python main.py --video FILE     # Process a video file
  python main.py --image FILE     # Process a single image
  python main.py --camera 1       # Use camera index 1 instead of 0

Press 'q' to quit  |  Press 'f' to toggle ORB features  |  Press 's' to take screenshot
"""

import cv2
import time
import sys
import argparse
import os
import numpy as np

# ── Import project modules ──
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    WINDOW_TITLE, COLOR_DANGER, COLOR_WARNING,
    COLOR_SAFE, COLOR_STAIR, COLOR_INFO,
)
from object_detector import detect_objects, draw_detections
from stair_detector import detect_stairs, draw_stair_warning
from distance_estimator import (
    estimate_all_distances, estimate_all_distances_lidar,
    get_closest_object, classify_distance
)
from voice_assistant import speak, generate_navigation_message
from utils import (
    resize_frame, draw_zone_lines, draw_status_bar, draw_title_bar,
    detect_orb_features, draw_orb_features, convert_to_grayscale,
)


def parse_arguments():
    """Parse command-line arguments for input source selection."""
    parser = argparse.ArgumentParser(
        description="AI Navigation Assistant for Visually Impaired People"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file to process instead of webcam")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file to process (single frame mode)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help=f"Camera index to use (default: {CAMERA_INDEX})")
    parser.add_argument("--iphone", action="store_true",
                        help="Use wireless iPhone 13 Pro LiDAR sensor (Arvos)")
    return parser.parse_args()


def open_camera(camera_index):
    """
    Try to open the webcam, with fallback to other camera indices.
    Returns an opened VideoCapture or None.
    """
    # Try the requested camera index first
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        cap.release()

    # Try alternative camera indices (0, 1, 2)
    for idx in [0, 1, 2]:
        if idx == camera_index:
            continue
        print(f"[INFO] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Camera found at index {idx}")
                return cap
            cap.release()

    return None


def process_frame(frame, frame_count, show_features, depth_frame=None):
    """
    Run the full detection pipeline on a single frame.

    Returns:
        frame        : annotated frame
        nav_message  : navigation instruction string
        priority     : priority level string
    """
    # ── Step 1: Resize for consistent processing ──
    frame = resize_frame(frame)

    # ── Step 2: Object detection (YOLOv8) ──
    detections = detect_objects(frame)

    # ── Step 3: Distance estimation (LiDAR vs Pinhole) ──
    if depth_frame is not None:
        distances = estimate_all_distances_lidar(detections, depth_frame, frame.shape)
    else:
        distances = estimate_all_distances(detections)

    # ── Step 4: Stair detection (every 3rd frame to save CPU) ──
    stairs_detected = False
    stair_region = None
    if frame_count % 3 == 0:
        stairs_detected, stair_region, _ = detect_stairs(frame)

    # ── Step 5: Generate navigation message ──
    nav_message, priority = generate_navigation_message(
        detections, distances, stairs_detected
    )

    # ── Step 6: Draw UI overlays ──
    draw_zone_lines(frame)
    draw_detections(frame, detections, distances)

    if stairs_detected:
        draw_stair_warning(frame, stair_region)

    # ORB features (optional toggle)
    if show_features:
        gray = convert_to_grayscale(frame)
        keypoints, _ = detect_orb_features(gray)
        frame = draw_orb_features(frame, keypoints)

    # Title bar
    title_text = "AI Navigation Assistant (LiDAR)" if depth_frame is not None else "AI Navigation Assistant"
    draw_title_bar(frame, title_text)

    # Status bar with navigation message
    status_color = {
        "DANGER": COLOR_DANGER,
        "WARNING": COLOR_WARNING,
        "STAIR": COLOR_STAIR,
        "CLEAR": COLOR_SAFE,
    }.get(priority, COLOR_INFO)
    draw_status_bar(frame, f"NAV: {nav_message}", status_color)

    # Object count
    cv2.putText(
        frame, f"Objects: {len(detections)}", (FRAME_WIDTH - 130, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
    )

    return frame, nav_message, priority


def run_image_mode(image_path):
    """Process a single image — useful for demos and testing."""
    print(f"[INFO] Processing image: {image_path}")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    speak("Processing image")
    frame, nav_message, priority = process_frame(frame, 1, False)
    speak(nav_message)

    print(f"[RESULT] {nav_message} (Priority: {priority})")
    cv2.imshow(WINDOW_TITLE, frame)
    cv2.imwrite("result_output.jpg", frame)
    print("[INFO] Result saved to result_output.jpg")
    print("[INFO] Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_mode(source, is_webcam=True, camera_index=0):
    """
    Main processing loop for webcam or video file.

    Pipeline per frame:
      1. Capture frame
      2. Resize to standard resolution
      3. Run YOLOv8 object detection
      4. Estimate distances using pinhole model
      5. Detect stairs using edge detection
      6. Generate navigation instructions
      7. Trigger voice alerts
      8. Draw UI overlays
      9. Display the annotated frame
    """
    # ── Open the video source ──
    if is_webcam:
        cap = open_camera(camera_index)
        if cap is None:
            print()
            print("=" * 60)
            print("  ⚠️  CAMERA ACCESS DENIED / NOT FOUND")
            print("=" * 60)
            print()
            print("  macOS requires camera permission for Terminal.")
            print()
            print("  To fix this:")
            print("  1. Open System Settings (or System Preferences)")
            print("  2. Go to Privacy & Security → Camera")
            print("  3. Enable access for 'Terminal' (or your IDE)")
            print("  4. Restart Terminal and run again")
            print()
            print("  Alternatively, run with a video or image:")
            print("    python main.py --video path/to/video.mp4")
            print("    python main.py --image path/to/photo.jpg")
            print()
            print("=" * 60)
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        source_name = "Webcam"
    else:
        if not os.path.exists(source):
            print(f"[ERROR] Video file not found: {source}")
            sys.exit(1)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {source}")
            sys.exit(1)
        source_name = os.path.basename(source)

    # State variables
    show_features = False
    frame_count = 0
    fps = 0
    prev_time = time.time()
    screenshot_count = 0

    print(f"[INFO] System started. Source: {source_name}")
    speak("Navigation assistant started")

    # ══════════════════════════════════════════════
    #  MAIN LOOP — processes every frame
    # ══════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            if not is_webcam:
                print("[INFO] Video ended. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[ERROR] Failed to read frame.")
            break

        frame_count += 1

        # Run the full pipeline
        frame, nav_message, priority = process_frame(frame, frame_count, show_features)

        # ── Trigger voice alert ──
        if priority in ("DANGER", "STAIR"):
            speak(nav_message)
        elif priority == "WARNING" and frame_count % 10 == 0:
            speak(nav_message)
        elif priority == "CLEAR" and frame_count % 30 == 0:
            speak(nav_message)

        # ── FPS counter ──
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 110, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        # ── Display ──
        cv2.imshow(WINDOW_TITLE, frame)

        # ── Handle keyboard input ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quitting...")
            speak("Navigation assistant stopped")
            break
        elif key == ord('f'):
            show_features = not show_features
            state = "ON" if show_features else "OFF"
            print(f"[INFO] ORB feature visualization: {state}")
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System shut down cleanly.")


def run_iphone_mode():
    """
    Main processing loop for iPhone 13 Pro LiDAR mode.
    """
    from iphone_lidar import IPhoneLiDARStream

    stream = IPhoneLiDARStream()
    try:
        stream.start()
    except RuntimeError as e:
        print()
        print("=" * 60)
        print("  ⚠️  IPHONE CONNECTION ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)
        sys.exit(1)

    show_features = False
    frame_count = 0
    fps = 0
    prev_time = time.time()
    screenshot_count = 0

    print(f"[INFO] System started. Source: {stream.device_name}")
    speak("iPhone LiDAR navigation assistant started")

    while True:
        success, frame, depth_frame = stream.read()
        if not success:
            print("[ERROR] Failed to read frame from iPhone LiDAR stream.")
            break

        frame_count += 1

        # Run the full pipeline (using LiDAR distances)
        frame, nav_message, priority = process_frame(
            frame, frame_count, show_features, depth_frame=depth_frame
        )

        # ── Trigger voice alert ──
        if priority in ("DANGER", "STAIR"):
            speak(nav_message)
        elif priority == "WARNING" and frame_count % 10 == 0:
            speak(nav_message)
        elif priority == "CLEAR" and frame_count % 30 == 0:
            speak(nav_message)

        # ── FPS counter ──
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 110, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )

        # ── Colormap and overlay the depth map for visualization ──
        depth_vis = depth_frame.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0.0
        max_depth = 5.0
        depth_vis = np.clip(depth_vis, 0.0, max_depth)
        depth_norm = (depth_vis / max_depth * 255.0).astype(np.uint8)
        depth_norm_inv = 255 - depth_norm
        depth_colored = cv2.applyColorMap(depth_norm_inv, cv2.COLORMAP_JET)

        # Resize depth map visualization to match frame height for side-by-side display
        depth_h_target = frame.shape[0]
        depth_w_target = int(depth_colored.shape[1] * depth_h_target / depth_colored.shape[0])
        depth_colored_resized = cv2.resize(depth_colored, (depth_w_target, depth_h_target))

        # Concatenate RGB feed and Depth map side-by-side
        combined_display = np.hstack((frame, depth_colored_resized))

        # ── Display ──
        cv2.imshow(WINDOW_TITLE, combined_display)

        # ── Handle keyboard input ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            speak("Navigation assistant stopped")
            break
        elif key == ord('f'):
            show_features = not show_features
            state = "ON" if show_features else "OFF"
            print(f"[INFO] ORB feature visualization: {state}")
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_rgb_{screenshot_count}.jpg"
            filename_depth = f"screenshot_depth_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            cv2.imwrite(filename_depth, depth_colored)
            print(f"[INFO] Screenshots saved: {filename} and {filename_depth}")

    # ── Cleanup ──
    stream.release()
    cv2.destroyAllWindows()
    print("[INFO] System shut down cleanly.")


def main():
    """Entry point — parse args and launch the appropriate mode."""
    args = parse_arguments()

    print("=" * 60)
    print("  AI Navigation Assistant for Visually Impaired People")
    print("  ─────────────────────────────────────────────────────")
    print("  Press 'q' to quit")
    print("  Press 'f' to toggle ORB feature visualization")
    print("  Press 's' to save a screenshot")
    print("=" * 60)

    if args.image:
        run_image_mode(args.image)
    elif args.video:
        run_video_mode(args.video, is_webcam=False)
    elif args.iphone:
        run_iphone_mode()
    else:
        run_video_mode(None, is_webcam=True, camera_index=args.camera)


if __name__ == "__main__":
    main()
