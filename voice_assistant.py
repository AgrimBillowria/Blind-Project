"""
voice_assistant.py — Text-to-Speech Voice Guidance System
==========================================================
Provides spoken navigation instructions using pyttsx3 (offline TTS).
Uses a cooldown timer to avoid overwhelming the user with repeated alerts.

Syllabus: Scene Understanding (translating visual data to guidance)
"""

import pyttsx3
import threading
import time
from config import VOICE_COOLDOWN, VOICE_RATE, VOICE_VOLUME, FRAME_WIDTH, DANGER_DISTANCE, WARNING_DISTANCE


# Track the last time each message was spoken to implement cooldown
_last_spoken = {}
_speak_lock = threading.Lock()


def speak(message):
    """
    Speak a message using text-to-speech in a background thread.

    Cooldown logic:
      If the same message was spoken less than VOICE_COOLDOWN seconds ago,
      it is skipped. This prevents the system from repeating
      "Obstacle ahead" every single frame (30+ times per second).

    Threading:
      TTS runs in a separate thread so it doesn't block the main
      video loop. The lock ensures only one message plays at a time.
    """
    current_time = time.time()

    # Check cooldown — skip if this message was said recently
    if message in _last_spoken:
        elapsed = current_time - _last_spoken[message]
        if elapsed < VOICE_COOLDOWN:
            return  # Too soon — skip

    # Update the timestamp
    _last_spoken[message] = current_time

    # Run TTS in a background thread to avoid blocking the video feed
    thread = threading.Thread(target=_speak_threaded, args=(message,), daemon=True)
    thread.start()


def _speak_threaded(message):
    """Internal: run TTS engine in a thread-safe manner.

    A new engine instance is created per call because pyttsx3 uses COM
    (SAPI5) on Windows, which has thread affinity. Reusing an engine
    created on the main thread from a worker thread causes deadlocks.
    """
    with _speak_lock:
        try:
            _engine = pyttsx3.init()
            _engine.setProperty('rate', VOICE_RATE)
            _engine.setProperty('volume', VOICE_VOLUME)
            _engine.say(message)
            _engine.runAndWait()
            _engine.stop()
        except Exception:
            pass


def generate_navigation_message(detections, distances, stairs_detected):
    """
    Generate a spoken navigation instruction based on current scene.

    Priority order:
      1. Stairs detected → "Caution, stairs ahead"
      2. Closest obstacle in DANGER zone → directional warning
      3. Closest obstacle in WARNING zone → caution message
      4. No obstacles → "Path is clear"

    Returns:
        message (str): The message to speak
        priority (str): "DANGER", "WARNING", "STAIR", or "CLEAR"
    """
    # Priority 1: Stairs
    if stairs_detected:
        return "Caution, stairs detected ahead", "STAIR"

    # Priority 2–3: Check obstacles
    if detections and distances:
        # Find the closest object
        closest_idx = min(distances, key=distances.get)
        closest_dist = distances[closest_idx]
        closest_det = detections[closest_idx]

        x_center = closest_det["x_center"]
        label = closest_det["label"]

        # Determine zone (LEFT / CENTER / RIGHT)
        relative_x = x_center / FRAME_WIDTH

        if relative_x < 0.33:
            direction = "on your left, move right"
        elif relative_x > 0.66:
            direction = "on your right, move left"
        else:
            direction = "directly ahead"

        if closest_dist < DANGER_DISTANCE:
            msg = f"Warning! {label} {direction}, very close"
            return msg, "DANGER"
        elif closest_dist < WARNING_DISTANCE:
            msg = f"Caution, {label} {direction}"
            return msg, "WARNING"

    # Priority 4: All clear
    return "Path is clear", "CLEAR"
