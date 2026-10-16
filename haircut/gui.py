# haircut/gui.py
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from haircut.config import CONFIG
from haircut.enforcer import HaircutEnforcer
from haircut.preprocessing import binarize_hair

# ── Design tokens ─────────────────────────────────────────────
COLOR_HEADER_BG       = "#013b14"
COLOR_HEADER_ACCENT   = "#fbb034"
COLOR_HEADER_TEXT     = "#ffffff"
COLOR_APP_BG          = "#f8f9fa"
COLOR_PANEL_BG        = "#ffffff"
COLOR_PANEL_BORDER    = "#cccccc"
COLOR_TEXT_PRIMARY    = "#222222"
COLOR_TEXT_MUTED      = "#6b7280"
COLOR_STATUS_OK       = "#00a651"
COLOR_STATUS_BAD      = "#ed1c24"
COLOR_STATUS_WARN     = "#fbb034"
COLOR_STATUS_NEUTRAL  = "#6b7280"
COLOR_BUTTON_BG       = "#00a2e8"
COLOR_BUTTON_BG_HOVER = "#0090d0"
COLOR_BUTTON_BG_ACTIVE= "#ed1c24"
COLOR_BUTTON_TEXT     = "#ffffff"

FONT_FAMILY      = "Segoe UI"
FONT_PANEL_TITLE = (FONT_FAMILY, 12, "bold")
FONT_STATUS      = (FONT_FAMILY, 32, "bold")
FONT_LABEL       = (FONT_FAMILY, 10)
FONT_VALUE       = (FONT_FAMILY, 11, "bold")
FONT_BUTTON      = (FONT_FAMILY, 11, "bold")

STATUS_PRESENTATION = {
    "ok":           ("COMPLIANT",       COLOR_STATUS_OK),
    "female_pass":  ("PASS",            COLOR_STATUS_OK),
    "violation":    ("VIOLATION",       COLOR_STATUS_BAD),
    "no_face":      ("STEP INTO FRAME", COLOR_STATUS_NEUTRAL),
    "side_profile": ("SIDE PROFILE",    COLOR_STATUS_WARN),
}

STATUS_SUBTEXT = {
    "ok":           "Subject meets the haircut policy.",
    "female_pass":  "Female detected — automatically compliant.",
    "violation":    "Hair length exceeds the allowed limit.",
    "no_face":      "No face detected. Please face the camera.",
    "side_profile": "Side profile detected — checking ear & neck regions.",
}


class HaircutEnforcerApp(tk.Tk):
    def __init__(self, camera_index: int = 0, model_path: str | None = None):
        super().__init__()
        self.title("USJ-R SIS — Haircut Policy Enforcement")
        self.configure(bg=COLOR_APP_BG)
        self.geometry("1280x760")
        self.minsize(1100, 680)

        self.enforcer = HaircutEnforcer(model_path=model_path)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["FRAME_WIDTH"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
        if not self.cap.isOpened():
            messagebox.showerror("Camera error",
                                 f"Could not open camera index {camera_index}.")
            self.destroy(); return

        self.show_mask = False
        self._photo   = None

        self._build_header()
        self._build_body()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(15, self._update_frame)

    # ── Header ────────────────────────────────────────────────
    def _build_header(self) -> None:
        header = tk.Frame(self, bg=COLOR_HEADER_BG, height=72)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        tk.Canvas(header, width=56, height=56,
                  bg=COLOR_HEADER_BG, highlightthickness=0).pack(
                      side="left", padx=(20, 12), pady=8)
        tk.Label(header, text="Haircut Policy Gate Monitor",
                 font=(FONT_FAMILY, 11, "italic"),
                 fg=COLOR_HEADER_ACCENT, bg=COLOR_HEADER_BG).pack(side="right", padx=24)
        tk.Frame(self, bg=COLOR_HEADER_ACCENT, height=5).pack(side="top", fill="x")

    # ── Body ──────────────────────────────────────────────────
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=COLOR_APP_BG)
        body.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=340)
        body.rowconfigure(0, weight=1)

        feed_panel = self._make_panel(body, "LIVE CAMERA FEED")
        feed_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.video_label = tk.Label(feed_panel, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        sidebar = tk.Frame(body, bg=COLOR_APP_BG)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        # Status card
        status_panel = self._make_panel(sidebar, "ENFORCEMENT STATUS")
        status_panel.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.status_label = tk.Label(
            status_panel, text="STEP INTO FRAME",
            font=FONT_STATUS, fg=COLOR_STATUS_NEUTRAL, bg=COLOR_PANEL_BG,
            wraplength=300, justify="center")
        self.status_label.pack(fill="x", padx=14, pady=(8, 6))
        self.status_sub = tk.Label(
            status_panel, text="Awaiting subject…",
            font=FONT_LABEL, fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG)
        self.status_sub.pack(fill="x", padx=14, pady=(0, 14))

        # Stats card
        stats_panel = self._make_panel(sidebar, "DEBUG STATS")
        stats_panel.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.stat_vars: dict[str, tk.StringVar] = {}
        for key, label in [
            ("classifier",   "Classifier"),
            ("gender",       "Gender (SVM)"),
            ("gender_conf",  "Gender Confidence"),
        ]:
            self.stat_vars[key] = self._add_stat_row(stats_panel, label)

        # Controls card
        controls_panel = self._make_panel(sidebar, "CONTROLS")
        controls_panel.grid(row=2, column=0, sticky="ew")
        self.mask_button = tk.Button(
            controls_panel, text="Toggle Debug Mask",
            font=FONT_BUTTON, fg=COLOR_BUTTON_TEXT, bg=COLOR_BUTTON_BG,
            activebackground=COLOR_BUTTON_BG_HOVER, activeforeground=COLOR_BUTTON_TEXT,
            relief="flat", borderwidth=0, cursor="hand2",
            padx=16, pady=10, command=self._toggle_mask)
        self.mask_button.pack(fill="x", padx=14, pady=(10, 8))
        self.mask_button.bind("<Enter>", lambda _e: self.mask_button.configure(
            bg=COLOR_BUTTON_BG_ACTIVE if self.show_mask else COLOR_BUTTON_BG_HOVER))
        self.mask_button.bind("<Leave>", lambda _e: self.mask_button.configure(
            bg=COLOR_BUTTON_BG_ACTIVE if self.show_mask else COLOR_BUTTON_BG))
        self.mask_state_label = tk.Label(
            controls_panel, text="Mask overlay: OFF",
            font=FONT_LABEL, fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG)
        self.mask_state_label.pack(fill="x", padx=14, pady=(0, 14))

    # ── Panel helpers ─────────────────────────────────────────
    def _make_panel(self, parent: tk.Misc, title: str) -> tk.Frame:
        panel = tk.Frame(parent, bg=COLOR_PANEL_BG,
                         highlightbackground=COLOR_PANEL_BORDER,
                         highlightthickness=1, bd=0)
        title_bar = tk.Frame(panel, bg=COLOR_PANEL_BG)
        title_bar.pack(fill="x", padx=14, pady=(12, 4))
        tk.Label(title_bar, text=title, font=FONT_PANEL_TITLE,
                 fg=COLOR_HEADER_BG, bg=COLOR_PANEL_BG).pack(side="left")
        tk.Frame(panel, bg=COLOR_HEADER_ACCENT, height=2).pack(fill="x", padx=14)
        return panel

    def _add_stat_row(self, panel: tk.Frame, label: str) -> tk.StringVar:
        row = tk.Frame(panel, bg=COLOR_PANEL_BG)
        row.pack(fill="x", padx=14, pady=4)
        tk.Label(row, text=label, font=FONT_LABEL,
                 fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side="left")
        var = tk.StringVar(value="—")
        tk.Label(row, textvariable=var, font=FONT_VALUE,
                 fg=COLOR_TEXT_PRIMARY, bg=COLOR_PANEL_BG).pack(side="right")
        return var

    # ── Frame loop ────────────────────────────────────────────
    def _update_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.after(50, self._update_frame); return
        frame = cv2.flip(frame,1)
        annotated, det = self.enforcer.analyze_frame(frame)

        if self.show_mask:
            annotated = self._apply_debug_mask(frame, annotated, det)

        self._render_video(annotated)
        self._update_status(det)
        self._update_stats(det)
        self.after(15, self._update_frame)

    def _apply_debug_mask(self, frame, annotated, det) -> np.ndarray:
        out = annotated.copy()

        # ── Forehead (bang) ROI — existing red tint ───────────
        if det.forehead_roi_box is not None:
            rx, ry, rw, rh = det.forehead_roi_box
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi  = gray[ry:ry + rh, rx:rx + rw]
                if roi.size > 0:
                    mask = binarize_hair(roi)
                    tint = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    tint[mask > 0] = [0, 0, 180]
                    out[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(
                        out[ry:ry + rh, rx:rx + rw], 0.5, tint, 0.5, 0)
            except Exception:
                pass

        # ── Upper hair ROI — purple tint ──────────────────────
        if det.upper_hair_roi_box is not None and det.upper_hair_mask is not None:
            rx, ry, rw, rh = det.upper_hair_roi_box
            try:
                if rw > 0 and rh > 0:
                    mask_r = cv2.resize(
                        det.upper_hair_mask, (rw, rh),
                        interpolation=cv2.INTER_NEAREST)
                    tint = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
                    tint[mask_r > 0] = [0, 0, 180]
                    out[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(
                        out[ry:ry + rh, rx:rx + rw], 0.5, tint, 0.5, 0)
            except Exception:
                pass

        # ── Profile ROIs (ear / neck) — existing red tint ─────
        for box, mask in [(det.ear_roi_box, det.ear_mask),
                          (det.neck_roi_box, det.neck_mask)]:
            if box is None or mask is None:
                continue
            rx, ry, rw, rh = box
            try:
                if rw <= 0 or rh <= 0:
                    continue
                mask_r = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
                tint   = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
                tint[mask_r > 0] = [0, 0, 180]
                out[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(
                    out[ry:ry + rh, rx:rx + rw], 0.5, tint, 0.5, 0)
            except Exception:
                pass

        return out

    def _render_video(self, frame_bgr: np.ndarray) -> None:
        target_w = max(self.video_label.winfo_width(), 320)
        target_h = max(self.video_label.winfo_height(), 240)
        h, w     = frame_bgr.shape[:2]
        scale    = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized  = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)

    def _update_status(self, det) -> None:
        text, color = STATUS_PRESENTATION.get(det.status, ("UNKNOWN", COLOR_STATUS_NEUTRAL))
        self.status_label.configure(text=text, fg=color)
        self.status_sub.configure(text=STATUS_SUBTEXT.get(det.status, ""))

    def _update_stats(self, det) -> None:
        self.stat_vars["classifier"].set(det.classifier_used or "—")
        self.stat_vars["gender"].set(det.gender.upper() if det.gender != "unknown" else "—")
        self.stat_vars["gender_conf"].set(
            f"{det.gender_confidence * 100:5.1f} %" if det.gender_confidence else "—")

    # ── Controls ──────────────────────────────────────────────
    def _toggle_mask(self) -> None:
        self.show_mask = not self.show_mask
        self.mask_state_label.configure(
            text=f"Mask overlay: {'ON' if self.show_mask else 'OFF'}",
            fg=COLOR_STATUS_BAD if self.show_mask else COLOR_TEXT_MUTED)
        self.mask_button.configure(
            bg=COLOR_BUTTON_BG_ACTIVE if self.show_mask else COLOR_BUTTON_BG)

    def _on_close(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.destroy()