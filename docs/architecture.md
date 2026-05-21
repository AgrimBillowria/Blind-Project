# System Architecture — Deep Dive

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN LOOP (main.py)                      │
│                                                                 │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌─────────┐ │
│  │ CAPTURE  │──▶│ PREPROCESS │──▶│  DETECT    │──▶│ DISPLAY │ │
│  │ (webcam) │   │ (resize)   │   │ (all modules)│  │ (UI)    │ │
│  └──────────┘   └────────────┘   └────────────┘   └─────────┘ │
│                                         │                       │
│                                         ▼                       │
│                                  ┌────────────┐                 │
│                                  │   VOICE    │                 │
│                                  │ (pyttsx3)  │                 │
│                                  └────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Frame (BGR, 640×480)
    │
    ├──▶ object_detector.detect_objects(frame)
    │       └──▶ Returns: List[{label, confidence, bbox, x_center, width, ...}]
    │
    ├──▶ distance_estimator.estimate_all_distances(detections)
    │       └──▶ Returns: Dict[index → distance_cm]
    │
    ├──▶ stair_detector.detect_stairs(frame)
    │       └──▶ Returns: (bool, region_tuple, line_count)
    │
    └──▶ voice_assistant.generate_navigation_message(detections, distances, stairs)
            └──▶ Returns: (message_string, priority_level)
```

## Module Dependency Graph

```
config.py (no dependencies — pure constants)
    │
    ├── utils.py (depends on: config)
    ├── object_detector.py (depends on: config, ultralytics)
    ├── distance_estimator.py (depends on: config)
    ├── stair_detector.py (depends on: config)
    ├── voice_assistant.py (depends on: config, pyttsx3)
    │
    └── main.py (depends on: ALL modules above)
```

## Threading Model

```
Main Thread:
  └── Video capture + processing + display (30 FPS loop)

Background Thread(s):
  └── voice_assistant._speak_threaded()
      - Created per voice alert
      - Daemon thread (dies when main thread exits)
      - Protected by threading.Lock to prevent concurrent speech
```

## Frame Processing Pipeline (per frame)

| Step | Module | Time Cost | Description |
|------|--------|-----------|-------------|
| 1 | OpenCV | ~1 ms | Capture frame from webcam |
| 2 | utils | <1 ms | Resize to 640×480 |
| 3 | object_detector | ~15-30 ms | YOLOv8 inference |
| 4 | distance_estimator | <1 ms | Pinhole formula per detection |
| 5 | stair_detector | ~5 ms | Canny + Hough (every 3rd frame) |
| 6 | voice_assistant | <1 ms | Message generation (speech is async) |
| 7 | utils | ~2 ms | Draw all UI overlays |
| 8 | OpenCV | ~1 ms | Display frame |
| **Total** | | **~25-40 ms** | **~25-40 FPS on CPU** |

## Zone Navigation System

```
Frame Width: 640 pixels
┌──────────────┬──────────────┬──────────────┐
│     LEFT     │    CENTER    │    RIGHT     │
│   0-212px    │  213-422px   │  423-640px   │
│              │              │              │
│  "Move right"│"Obstacle     │  "Move left" │
│              │   ahead"     │              │
└──────────────┴──────────────┴──────────────┘

Object's x_center determines the zone.
Navigation instruction is the OPPOSITE direction:
  - Object on LEFT → tell user to move RIGHT
  - Object on RIGHT → tell user to move LEFT
  - Object in CENTER → warn "obstacle ahead"
```

## Distance Classification

```
Distance (cm)    Level      Color     Voice Behavior
─────────────    ─────      ─────     ──────────────
0 - 99           DANGER     Red       Immediate alert
100 - 199        WARNING    Orange    Alert every 10 frames
200+             SAFE       Green     "Path clear" every 30 frames
```
