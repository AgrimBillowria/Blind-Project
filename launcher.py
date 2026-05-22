"""
launcher.py — Graphical Mode Launcher for the AI Navigation Assistant
======================================================================
Shows a small tkinter window on startup so the user can pick:
  • Camera Mode — live webcam feed with a camera selector
  • Video Mode  — process a video file chosen via file-browse dialog

Returns a result dict to main.py; returns None if the window is closed.
"""

import platform
import tkinter as tk
from tkinter import ttk, filedialog


def detect_cameras(max_index=5):
    """
    Scan camera indices 0 to max_index-1 and return a list of
    (index, label) tuples for each camera that successfully opens
    and delivers at least one frame.

    Uses DirectShow on Windows for faster, more reliable detection.
    """
    import cv2

    available = []
    backends = [cv2.CAP_DSHOW] if platform.system() == "Windows" else [0]

    for idx in range(max_index):
        found = False
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    label = f"Camera {idx}"
                    if idx == 0:
                        label += "  (Built-in)"
                    available.append((idx, label))
                    found = True
            cap.release()
            if found:
                break

    return available


def show_launcher():
    """
    Display the launcher window and block until the user clicks Start or
    closes the window.

    Returns:
        dict  — {"mode": "camera"|"video",
                 "camera_index": int,
                 "video_path": str | None}
        None  — user closed the window without starting
    """
    result = {"mode": None, "camera_index": 0, "video_path": None}

    # ── Detect available cameras before building the UI ──
    print("[INFO] Detecting cameras...")
    cameras = detect_cameras()
    if not cameras:
        cameras = []  # handled in UI

    # ─────────────────────────────────────────────
    # Build the tkinter window
    # ─────────────────────────────────────────────
    root = tk.Tk()
    root.title("AI Navigation Assistant")
    root.geometry("420x320")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    # Centre the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 420) // 2
    y = (root.winfo_screenheight() - 320) // 2
    root.geometry(f"420x320+{x}+{y}")

    # ── Colour palette ──
    BG       = "#1e1e2e"
    PANEL_BG = "#2a2a3e"
    ACCENT   = "#7c3aed"       # purple
    TEXT     = "#e2e8f0"
    SUBTEXT  = "#94a3b8"
    BTN_FG   = "#ffffff"
    BTN_BG   = "#7c3aed"
    BTN_HOVER= "#6d28d9"
    START_BG = "#059669"       # green
    START_HOV= "#047857"
    DISABLED = "#374151"

    FONT_TITLE = ("Segoe UI", 13, "bold")
    FONT_LABEL = ("Segoe UI", 10)
    FONT_SMALL = ("Segoe UI", 9)
    FONT_BTN   = ("Segoe UI", 10, "bold")

    # ── Header ──
    header = tk.Frame(root, bg=ACCENT, height=50)
    header.pack(fill="x")
    header.pack_propagate(False)
    tk.Label(
        header,
        text="AI Navigation Assistant",
        font=FONT_TITLE,
        bg=ACCENT, fg=BTN_FG,
        padx=16,
    ).pack(side="left", pady=12)

    # ── Body frame ──
    body = tk.Frame(root, bg=BG, padx=20, pady=14)
    body.pack(fill="both", expand=True)

    # ── Mode selection label + radio buttons ──
    tk.Label(body, text="Select Mode:", font=FONT_LABEL,
             bg=BG, fg=TEXT).pack(anchor="w")

    mode_var = tk.StringVar(value="camera")

    radio_row = tk.Frame(body, bg=BG)
    radio_row.pack(fill="x", pady=(4, 12))

    def make_radio(parent, text, value):
        rb = tk.Radiobutton(
            parent, text=text, variable=mode_var, value=value,
            font=FONT_LABEL, bg=BG, fg=TEXT,
            selectcolor=PANEL_BG, activebackground=BG,
            activeforeground=TEXT,
            indicatoron=0,                    # button-style radio
            relief="flat", bd=0,
            padx=14, pady=6,
            cursor="hand2",
            command=on_mode_change,
        )
        return rb

    # We create them after on_mode_change is defined — see below

    # ── Camera panel ──
    cam_panel = tk.Frame(body, bg=PANEL_BG, padx=12, pady=10)

    tk.Label(cam_panel, text="Camera:", font=FONT_LABEL,
             bg=PANEL_BG, fg=TEXT).grid(row=0, column=0, sticky="w", padx=(0, 10))

    cam_labels = [lbl for _, lbl in cameras]
    cam_indices = [idx for idx, _ in cameras]

    cam_combo = ttk.Combobox(
        cam_panel,
        values=cam_labels if cam_labels else ["No cameras found"],
        state="readonly" if cam_labels else "disabled",
        width=28,
        font=FONT_SMALL,
    )
    cam_combo.grid(row=0, column=1, sticky="ew")
    if cam_labels:
        cam_combo.current(0)

    cam_panel.columnconfigure(1, weight=1)

    # ── Video panel ──
    vid_panel = tk.Frame(body, bg=PANEL_BG, padx=12, pady=10)

    video_path_var = tk.StringVar(value="")

    def browse_video():
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            video_path_var.set(path)
            # Show only the filename, not the full path, in the label
            short = path if len(path) <= 32 else "…" + path[-29:]
            vid_file_label.config(text=short, fg=TEXT)
            _update_start_button()

    browse_btn = tk.Button(
        vid_panel, text="Browse…",
        font=FONT_BTN, bg=BTN_BG, fg=BTN_FG,
        activebackground=BTN_HOVER, activeforeground=BTN_FG,
        relief="flat", padx=10, pady=4,
        cursor="hand2",
        command=browse_video,
    )
    browse_btn.grid(row=0, column=0, padx=(0, 10))

    vid_file_label = tk.Label(
        vid_panel, text="No file selected",
        font=FONT_SMALL, bg=PANEL_BG, fg=SUBTEXT,
        anchor="w", width=26,
    )
    vid_file_label.grid(row=0, column=1, sticky="ew")
    vid_panel.columnconfigure(1, weight=1)

    # ── Start button ──
    start_btn = tk.Button(
        body, text="Start",
        font=("Segoe UI", 11, "bold"),
        bg=START_BG, fg=BTN_FG,
        activebackground=START_HOV, activeforeground=BTN_FG,
        disabledforeground="#6b7280",
        relief="flat", padx=28, pady=8,
        cursor="hand2",
    )

    # ── Logic helpers ──
    def _update_start_button():
        mode = mode_var.get()
        if mode == "camera":
            state = "normal" if cam_labels else "disabled"
            bg = START_BG if cam_labels else DISABLED
        else:
            state = "normal" if video_path_var.get() else "disabled"
            bg = START_BG if video_path_var.get() else DISABLED
        start_btn.config(state=state, bg=bg)

    def on_mode_change():
        mode = mode_var.get()
        if mode == "camera":
            vid_panel.pack_forget()
            cam_panel.pack(fill="x", pady=(0, 12))
        else:
            cam_panel.pack_forget()
            vid_panel.pack(fill="x", pady=(0, 12))
        _update_start_button()

    # Now create the radio buttons (on_mode_change is defined)
    rb_cam = make_radio(radio_row, "  Camera Mode", "camera")
    rb_vid = make_radio(radio_row, "  Video Mode", "video")
    rb_cam.pack(side="left", padx=(0, 10))
    rb_vid.pack(side="left")

    # Initial panel state
    cam_panel.pack(fill="x", pady=(0, 12))

    def on_start():
        mode = mode_var.get()
        result["mode"] = mode
        if mode == "camera":
            sel = cam_combo.current()
            result["camera_index"] = cam_indices[sel] if cam_indices else 0
        else:
            result["video_path"] = video_path_var.get()
        root.destroy()

    start_btn.config(command=on_start)
    start_btn.pack(pady=(4, 0))

    _update_start_button()

    # ── Style the combobox ──
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "TCombobox",
        fieldbackground=BG,
        background=PANEL_BG,
        foreground=TEXT,
        arrowcolor=TEXT,
        selectbackground=ACCENT,
        selectforeground=BTN_FG,
    )

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    if result["mode"] is None:
        return None
    return result
