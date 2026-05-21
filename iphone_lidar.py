"""
iphone_lidar.py — iPhone 13 Pro LiDAR Stream Interface
======================================================
This module interfaces with the Record3D library to stream real-time RGB and
dense LiDAR depth maps from a USB-connected iPhone/iPad.

Usage:
    from iphone_lidar import IPhoneLiDARStream
    stream = IPhoneLiDARStream()
    stream.start()
    while True:
        success, rgb_bgr, depth = stream.read()
"""

import cv2
import numpy as np
import threading
import time

try:
    from record3d import Record3DStream
    RECORD3D_AVAILABLE = True
except ImportError:
    RECORD3D_AVAILABLE = False


class IPhoneLiDARStream:
    def __init__(self):
        self.session = None
        self.event = threading.Event()
        self.connected = False
        self.device_name = "iPhone 13 Pro (LiDAR)"

    def on_new_frame(self):
        """Callback invoked by the Record3D library on a background thread."""
        self.event.set()

    def start(self):
        """
        Discovers connected iOS devices and connects to the first available one.
        Raises RuntimeError if no devices are found or record3d is not installed.
        """
        if not RECORD3D_AVAILABLE:
            raise RuntimeError(
                "The 'record3d' library is not installed or failed to load. "
                "Ensure CMake is installed and run: pip install record3d"
            )

        print("[INFO] Scanning for connected iOS devices via USB...")
        devices = Record3DStream.get_connected_devices()
        if not devices:
            raise RuntimeError(
                "No iOS devices detected via USB. Make sure:\n"
                "  1. Your iPhone is connected to your Mac with a USB cable.\n"
                "  2. The 'Record3D' app is open on your iPhone.\n"
                "  3. In Settings > Live RGBD Video Streaming, 'USB' is selected.\n"
                "  4. In the Record tab, the streaming is started (red toggle button ON).\n"
                "  5. Your Mac is 'Trusted' on the iPhone."
            )

        dev = devices[0]
        self.device_name = getattr(dev, "product", "iOS LiDAR Device")
        print(f"[INFO] Connecting to {self.device_name}...")

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        
        # Connect to the selected device
        self.session.connect(dev)
        self.connected = True
        print(f"[INFO] Connected to {self.device_name} successfully!")

    def read(self):
        """
        Reads the next RGB and LiDAR depth frame.
        
        Returns:
            ret (bool): True if frames read successfully, False otherwise.
            rgb_bgr (np.ndarray): BGR image frame (640x480 or similar).
            depth (np.ndarray): LiDAR depth map in meters (float32, e.g., 256x192).
        """
        if not self.connected or self.session is None:
            return False, None, None

        # Wait for the callback to signal that a new frame has arrived.
        # Use a timeout of 2.0 seconds to prevent blocking indefinitely if disconnected.
        frame_ready = self.event.wait(timeout=2.0)
        if not frame_ready:
            print("[WARNING] Timeout waiting for next frame from Record3D.")
            return False, None, None

        # Get depth and RGB arrays
        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        
        # Reset the event for the next frame
        self.event.clear()

        if rgb is None or depth is None:
            return False, None, None

        # Convert RGB (from Record3D) to BGR (expected by OpenCV and our pipeline)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Flip frames if required (TrueDepth needs mirroring, LiDAR usually doesn't, 
        # but we alignment check to match the physical camera orientation).
        # On standard iOS rear LiDAR setups, we want to align correctly.
        # If needed, the orientation can be rotated/flipped depending on the device:
        # e.g., rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_90_CLOCKWISE)
        
        return True, rgb_bgr, depth

    def release(self):
        """Closes the streaming session and releases resources."""
        self.connected = False
        if self.session is not None:
            print(f"[INFO] Disconnecting from {self.device_name}...")
            # Attempt to call disconnect if supported by the Record3DStream
            if hasattr(self.session, "disconnect"):
                try:
                    self.session.disconnect()
                except Exception as e:
                    print(f"[DEBUG] Error calling disconnect: {e}")
            self.session = None
