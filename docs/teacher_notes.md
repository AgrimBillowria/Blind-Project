# Teacher / Viva Explanation Notes

These notes are designed to help you explain the project to your teacher
during the presentation and viva. Read through these before your demo.

---

## How to Introduce the Project (30 seconds)

> "Our project is an AI-based navigation assistant for visually impaired people.
> It uses a webcam to detect obstacles, stairs, and estimate distances in real-time,
> then provides spoken voice guidance. We combine classical computer vision
> techniques like edge detection and the Hough Transform with modern deep learning
> using YOLOv8 for object detection. The system runs entirely offline on a
> standard laptop without any GPU."

---

## Key Technical Points to Highlight

### 1. "We use TWO approaches — classical CV + deep learning"
- **Classical CV:** Canny edge detection, Hough Line Transform, Gaussian blur,
  image filtering → used for stair detection
- **Deep Learning:** YOLOv8 CNN → used for object detection
- This shows understanding of BOTH traditional and modern approaches

### 2. "Distance estimation uses the pinhole camera model"
- Draw the pinhole diagram on the whiteboard:
  ```
  Object (W cm) ←——— D cm ———→ Lens ←— f pixels —→ Image (w pixels)
  
  D = (W × f) / w
  ```
- This is from the Camera Geometry syllabus
- Example: Person (45cm wide), bbox 150px → distance = 180cm

### 3. "Voice guidance uses threading to avoid blocking"
- Without threading: video freezes while speech plays
- With threading: speech runs in background, video stays smooth
- Cooldown timer prevents repeating the same message

### 4. "The system processes frames in a pipeline"
- Capture → Resize → Detect → Estimate → Speak → Display
- Each step takes a few milliseconds
- Total: 25-40 FPS on CPU

---

## Common Teacher Questions & Quick Answers

### "Why YOLOv8 and not YOLOv5 or SSD?"
YOLOv8 Nano is the latest and fastest in the YOLO family. It uses an
anchor-free detection head which is simpler and more accurate. The Nano
variant has only 3.2M parameters, making it perfect for real-time CPU use.

### "Is the distance estimation accurate?"
It's approximate — within ±30%. For navigation, we don't need centimeter
accuracy. We need to know: "is this object dangerously close or safely far?"
The classification into DANGER/WARNING/SAFE zones is what matters.

### "Why not use a depth camera?"
Depth cameras (like Intel RealSense) cost $200+ and require specific
hardware. Our approach uses a standard $10 webcam, making it accessible
to anyone. This is listed as a future improvement.

### "How does stair detection work?"
Stairs have a distinctive visual pattern: multiple horizontal edges
(the step edges) stacked vertically. We detect these using:
1. Canny edge detection → finds all edges
2. Hough Line Transform → finds straight lines among the edges
3. Angle filtering → keeps only horizontal lines (±15°)
4. Counting → if ≥4 horizontal lines, it's stairs

### "What happens if the camera is blocked or broken?"
The code checks `if not ret: break` after each frame capture. If the
camera fails, the program exits cleanly with an error message.

### "Can this work on a phone?"
Not directly, but it can be ported using:
- TensorFlow Lite or ONNX for mobile model inference
- OpenCV's Android/iOS bindings
- This is listed in our Future Scope section

---

## How to Structure Your Viva Answers

Use this 3-part formula for every answer:

1. **WHAT** — State what the concept/technique is
2. **HOW** — Explain briefly how it works
3. **WHERE** — Point to where it's used in your code

**Example:**
> "Gaussian blur is a noise reduction technique [WHAT] that convolves the
> image with a 2D bell-curve kernel, making each pixel a weighted average
> of its neighbors [HOW]. We use it in utils.py and in the stair detector
> before running Canny edge detection [WHERE]."

---

## Demo Script (5 minutes)

1. **Start the app:** `python main.py` — show the webcam window
2. **Show zone lines:** Point out LEFT/CENTER/RIGHT zones
3. **Detect a person:** Walk in front of camera → show bounding box + distance
4. **Show color coding:** Move close (red) → move back (green)
5. **Listen to voice:** "Warning, person directly ahead, very close"
6. **Show stairs:** Point camera at stairs (or a picture of stairs)
7. **Toggle ORB features:** Press 'f' → show keypoints
8. **Take screenshot:** Press 's' → show saved file
9. **Show FPS counter:** Point out real-time performance
10. **Quit cleanly:** Press 'q' → "Navigation assistant stopped"

---

## If Something Goes Wrong During Demo

| Problem | Quick Fix |
|---------|-----------|
| No detection | Move closer to objects, check YOLO_CONFIDENCE in config.py |
| Low FPS | Close other apps, reduce FRAME_WIDTH to 480 |
| No voice | Check speaker volume, try `pyttsx3.init()` in Python shell |
| Webcam error | Try CAMERA_INDEX = 1 in config.py |
| False stair detection | Increase STAIR_MIN_LINES to 6 in config.py |

---

## Closing Statement for Presentation

> "Our project demonstrates how computer vision and AI can create socially
> impactful technology. It combines fundamental image processing concepts like
> edge detection and Gaussian filtering with modern deep learning for object
> detection. The modular design makes it easy to extend — future work could
> include depth cameras, mobile deployment, and multi-language support.
> Thank you."
