"""
iphone_lidar.py — iPhone 13 Pro LiDAR Stream Interface
======================================================
This module interfaces with the Arvos library to stream real-time RGB and
dense LiDAR depth maps wirelessly from an iOS device.

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
import asyncio

try:
    from arvos import ArvosServer
    ARVOS_AVAILABLE = True
except ImportError:
    ARVOS_AVAILABLE = False


class IPhoneLiDARStream:
    def __init__(self):
        self.server = None
        self.event = threading.Event()
        self.connected = False
        self.device_name = "iPhone 13 Pro (LiDAR via Arvos)"
        
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_intrinsics = None
        self.running = False
        
        self.thread = None
        self.loop = None
        self.server_task = None

    async def _on_camera(self, camera_frame):
        """Callback invoked by the ArvosServer when a new camera frame is received."""
        try:
            nparr = np.frombuffer(camera_frame.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                self.latest_rgb = img
                self.latest_intrinsics = camera_frame.intrinsics
                self.event.set()
        except Exception as e:
            print(f"[DEBUG] Error decoding camera frame: {e}")

    async def _on_depth(self, depth_frame):
        """Callback invoked by the ArvosServer when a new depth frame is received."""
        try:
            if depth_frame.format == "point_cloud":
                pts = depth_frame.to_point_cloud()
                if pts is not None and self.latest_rgb is not None and self.latest_intrinsics is not None:
                    h, w = self.latest_rgb.shape[:2]
                    self.latest_depth = self.project_point_cloud(pts, self.latest_intrinsics, w, h)
            elif depth_frame.format == "raw_depth":
                # Reshape raw float32 buffer
                num_elements = len(depth_frame.data) // 4
                # Standard ARKit depth map resolutions: 256x192 or similar
                if num_elements == 256 * 192:
                    self.latest_depth = np.frombuffer(depth_frame.data, dtype=np.float32).reshape((192, 256))
                elif num_elements == 192 * 256:
                    self.latest_depth = np.frombuffer(depth_frame.data, dtype=np.float32).reshape((256, 192))
                else:
                    self.latest_depth = np.frombuffer(depth_frame.data, dtype=np.float32)
        except Exception as e:
            print(f"[DEBUG] Error decoding depth frame: {e}")

    async def _on_connect(self, client_id):
        """Callback when the iPhone connects to our server."""
        print(f"\n[INFO] iPhone connected: {client_id}")
        self.connected = True

    async def _on_disconnect(self, client_id):
        """Callback when the iPhone disconnects from our server."""
        print(f"\n[INFO] iPhone disconnected: {client_id}")
        self.connected = False

    def project_point_cloud(self, points, intrinsics, width, height):
        """
        Projects 3D point cloud [x, y, z, r, g, b] onto the 2D image plane 
        using camera intrinsics to produce an aligned depth map.
        """
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        if points is None or len(points) == 0:
            return depth_map
            
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]
        
        # Absolute depth value
        depths = np.abs(zs)
        
        # Filter points that are too close or too far
        valid = (depths > 0.1) & (depths < 10.0)
        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]
        depths = depths[valid]
        
        if len(depths) == 0:
            return depth_map
            
        # Project using perspective projection:
        # u = fx * (x / z) + cx
        # v = cy - fy * (y / z) (ARKit Y points up, image Row increases down)
        u = (intrinsics.fx * xs / depths) + intrinsics.cx
        v = intrinsics.cy - (intrinsics.fy * ys / depths)
        
        u_idx = np.round(u).astype(np.int32)
        v_idx = np.round(v).astype(np.int32)
        
        # Keep inside image bounds
        in_bounds = (u_idx >= 0) & (u_idx < width) & (v_idx >= 0) & (v_idx < height)
        u_idx = u_idx[in_bounds]
        v_idx = v_idx[in_bounds]
        depths = depths[in_bounds]
        
        if len(depths) == 0:
            return depth_map
            
        # Sort so closer points override farther points
        sort_idx = np.argsort(-depths)
        u_idx = u_idx[sort_idx]
        v_idx = v_idx[sort_idx]
        depths = depths[sort_idx]
        
        # Fill the depth map
        depth_map[v_idx, u_idx] = depths
        return depth_map

    def start(self):
        """
        Starts the Arvos server on port 9090 in a background thread
        and prints the QR code for scanning.
        """
        if not ARVOS_AVAILABLE:
            raise RuntimeError(
                "The 'arvos' library is not installed. "
                "Run: pip install arvos-sdk"
            )

        print("[INFO] Starting Arvos Server for wireless LiDAR streaming...")
        self.server = ArvosServer(host="0.0.0.0", port=9090)
        
        # Set up async callbacks
        self.server.on_camera = self._on_camera
        self.server.on_depth = self._on_depth
        self.server.on_connect = self._on_connect
        self.server.on_disconnect = self._on_disconnect
        
        self.running = True
        
        # Run loop in background thread
        self.thread = threading.Thread(target=self._run_server_thread, daemon=True)
        self.thread.start()
        
        # Wait to ensure server initialized
        time.sleep(1.0)

    def _run_server_thread(self):
        """Background thread worker to run the asyncio loop for the Arvos Server."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.server_task = self.loop.create_task(self.server.start())
        try:
            self.loop.run_until_complete(self.server_task)
        except asyncio.CancelledError:
            print("[INFO] Arvos Server task cancelled successfully.")
        except Exception as e:
            if self.running:
                print(f"[ERROR] Arvos Server encountered error: {e}")
        finally:
            try:
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            
            try:
                self.loop.close()
            except Exception:
                pass

    def read(self):
        """
        Reads the next RGB frame and projects the corresponding LiDAR depth map.
        
        Returns:
            ret (bool): True if frames read successfully, False otherwise.
            rgb_bgr (np.ndarray): BGR image frame (640x480 or similar).
            depth (np.ndarray): LiDAR depth map in meters (float32).
        """
        if not self.running or self.server is None:
            return False, None, None

        # Wait for camera frame with a timeout of 60.0 seconds for initial connection
        # once connected, event will trigger rapidly at 30 fps
        timeout_val = 60.0 if not self.connected else 2.0
        frame_ready = self.event.wait(timeout=timeout_val)
        if not frame_ready or self.latest_rgb is None:
            return False, None, None

        self.event.clear()
        
        # Fallback if depth map is missing
        depth_frame = self.latest_depth
        if depth_frame is None:
            h, w = self.latest_rgb.shape[:2]
            depth_frame = np.zeros((h, w), dtype=np.float32)
            
        return True, self.latest_rgb.copy(), depth_frame.copy()

    def release(self):
        """Closes the streaming session and releases resources."""
        self.running = False
        self.connected = False
        print("[INFO] Shutting down Arvos wireless server...")
        
        # Stop loop if running by cancelling the server task
        if self.loop and self.loop.is_running() and self.server_task:
            self.loop.call_soon_threadsafe(self.server_task.cancel)
            
        if self.thread:
            self.thread.join(timeout=3.0)
            
        self.server = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_intrinsics = None
