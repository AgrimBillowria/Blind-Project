"""
server.py — FastAPI backend for the AI Navigation Assistant
============================================================
Streams annotated video frames over WebSocket to the React frontend.

Run with:
    uvicorn server:app --reload --port 8000

Endpoints:
    GET  /cameras          — list available camera devices
    POST /upload-video     — upload a video file, returns video_id
    WS   /ws               — bidirectional stream (send config, receive frames)
"""

from __future__ import annotations

import asyncio
import base64
import os
import platform
import queue as q_module
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import (
    COLOR_DANGER, COLOR_INFO, COLOR_SAFE, COLOR_STAIR, COLOR_WARNING,
    FRAME_WIDTH, FRAME_HEIGHT,
)
from distance_estimator import estimate_all_distances
from object_detector import detect_objects, draw_detections
from stair_detector import detect_stairs, draw_stair_warning
from utils import (
    draw_status_bar, draw_title_bar, draw_zone_lines, resize_frame,
)
from voice_assistant import generate_navigation_message

# ── Uploaded video files are stored here temporarily ──
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Navigation Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Camera utilities
# ─────────────────────────────────────────────

def _try_open_camera(idx: int) -> cv2.VideoCapture | None:
    """
    Try to open camera at `idx`. Uses DirectShow on Windows for reliability.
    Drains up to 5 warm-up frames before accepting (some drivers return
    black frames on first reads).
    """
    backends = [cv2.CAP_DSHOW, 0] if platform.system() == "Windows" else [0]
    for backend in backends:
        cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0 and frame.any():
                return cap
        cap.release()
    return None


def _open_camera(camera_index: int) -> cv2.VideoCapture | None:
    cap = _try_open_camera(camera_index)
    if cap is not None:
        return cap
    for idx in [0, 1, 2]:
        if idx == camera_index:
            continue
        cap = _try_open_camera(idx)
        if cap is not None:
            return cap
    return None


# ─────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────

@app.get("/cameras")
def list_cameras():
    """
    Detect and return available camera devices.
    Uses the same warm-up logic as _try_open_camera() so cameras that
    return black/corrupt frames on the first read are still detected.
    """
    cameras = []
    for idx in range(5):
        cap = _try_open_camera(idx)
        if cap is not None:
            label = f"Camera {idx}" + ("  (Built-in)" if idx == 0 else "")
            cameras.append({"index": idx, "label": label})
            cap.release()
    return {"cameras": cameras}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Save an uploaded video to the uploads/ directory.
    Returns a video_id that the WebSocket client uses to reference the file.
    """
    ext = Path(file.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    if ext not in allowed:
        return {"error": f"Unsupported file type '{ext}'. Use: {', '.join(allowed)}"}

    video_id = str(uuid.uuid4())
    filepath = UPLOAD_DIR / f"{video_id}{ext}"
    filepath.write_bytes(await file.read())

    return {"video_id": video_id, "original_name": file.filename}


# ─────────────────────────────────────────────
# Detection pipeline — runs in a background thread
# ─────────────────────────────────────────────

def _detection_loop(
    cap: cv2.VideoCapture,
    mode: str,
    result_q: q_module.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Capture frames, run the full detection pipeline, encode to JPEG,
    and push result dicts onto result_q.

    Always puts None as a sentinel on the queue before returning so the
    async WebSocket handler knows the stream has ended.
    """
    frame_count = 0
    last_stairs = False
    last_stair_region = None
    prev_time = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if mode == "video":
                    # Loop the video instead of stopping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break  # webcam disconnected

            frame_count += 1

            # ── Detection pipeline ──
            frame = resize_frame(frame)
            detections = detect_objects(frame)
            distances = estimate_all_distances(detections)

            if frame_count % 3 == 0:
                last_stairs, last_stair_region, _ = detect_stairs(frame)

            nav_message, priority = generate_navigation_message(
                detections, distances, last_stairs
            )

            # ── Draw overlays ──
            draw_zone_lines(frame)
            draw_detections(frame, detections, distances)
            if last_stairs and last_stair_region:
                draw_stair_warning(frame, last_stair_region)

            draw_title_bar(frame, "AI Navigation Assistant")

            status_color = {
                "DANGER": COLOR_DANGER,
                "WARNING": COLOR_WARNING,
                "STAIR": COLOR_STAIR,
                "CLEAR": COLOR_SAFE,
            }.get(priority, COLOR_INFO)
            draw_status_bar(frame, f"NAV: {nav_message}", status_color)

            # FPS counter (top right)
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (FRAME_WIDTH - 110, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"Objects: {len(detections)}",
                (FRAME_WIDTH - 130, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

            # ── Encode frame as JPEG → base64 ──
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            result = {
                "type": "frame",
                "frame": frame_b64,
                "nav_message": nav_message,
                "priority": priority,
                "object_count": len(detections),
                "fps": round(fps, 1),
            }

            # Drop frames if the client can't keep up (queue full)
            try:
                result_q.put_nowait(result)
            except q_module.Full:
                pass

    except Exception as exc:
        result_q.put({"type": "error", "message": str(exc)})
    finally:
        cap.release()
        result_q.put(None)  # sentinel — signals the async handler to stop


# ─────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket lifecycle:
      1. Client connects and sends a JSON config:
            Camera: {"mode": "camera", "camera_index": 0}
            Video:  {"mode": "video",  "video_id": "<uuid>"}
      2. Server opens the capture source and starts the detection thread.
      3. Server streams frame messages:
            {"type": "frame", "frame": "<base64>", "nav_message": "...",
             "priority": "DANGER|WARNING|STAIR|CLEAR", "object_count": N, "fps": F}
      4. Client closes the connection to stop streaming.
         Server catches WebSocketDisconnect and sets the stop flag.
    """
    await websocket.accept()

    mode = "camera"
    video_id = None
    stop_event = threading.Event()
    thread: threading.Thread | None = None

    try:
        # Step 1 — receive start config
        config = await websocket.receive_json()
        mode = config.get("mode", "camera")
        camera_index = int(config.get("camera_index", 0))
        video_id = config.get("video_id")

        # Step 2 — open capture source
        if mode == "video" and video_id:
            matches = list(UPLOAD_DIR.glob(f"{video_id}.*"))
            if not matches:
                await websocket.send_json(
                    {"type": "error", "message": "Uploaded video not found. Please re-upload."}
                )
                return
            cap = cv2.VideoCapture(str(matches[0]))
        else:
            cap = _open_camera(camera_index)

        if cap is None or not cap.isOpened():
            await websocket.send_json(
                {"type": "error", "message": f"Could not open camera {camera_index}."}
            )
            return

        # Step 3 — start detection thread
        result_q: q_module.Queue = q_module.Queue(maxsize=3)
        thread = threading.Thread(
            target=_detection_loop,
            args=(cap, mode, result_q, stop_event),
            daemon=True,
        )
        thread.start()

        # Step 4 — relay results to client
        loop = asyncio.get_event_loop()
        while True:
            # result_q.get() blocks until a result is ready — run in executor
            # so we don't block the asyncio event loop
            result = await loop.run_in_executor(None, result_q.get)
            if result is None:
                break  # detection loop ended (video finished or camera lost)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass  # client navigated away or clicked Stop — normal exit
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        stop_event.set()
        if thread is not None:
            thread.join(timeout=3.0)
        # Delete the uploaded video file after the session ends
        if video_id:
            for f in UPLOAD_DIR.glob(f"{video_id}.*"):
                try:
                    f.unlink()
                except OSError:
                    pass
