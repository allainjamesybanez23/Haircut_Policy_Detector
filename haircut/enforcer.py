# haircut/enforcer.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from haircut.config import CONFIG
from haircut.models import Detection
from haircut.preprocessing import (
    binarize_hair,
    binarize_side_hair,
    build_hog_descriptor,
    extract_hog_features,
    preprocess_frame,
)


def load_svm_model(model_path: str):
    try:
        import joblib
    except ImportError:
        print("[WARN] joblib not installed."); return None, None
    if not Path(model_path).exists():
        print(f"[INFO] No model at {model_path} — gender gate disabled (treating all as male).")
        return None, None
    try:
        bundle = joblib.load(model_path)
        print(f"[OK] Loaded gender SVM from {model_path}")
        return bundle["clf"], bundle["scaler"]
    except Exception as e:
        print(f"[WARN] Could not load model: {e}"); return None, None


class HaircutEnforcer:
    def __init__(self, model_path: Optional[str] = None):
        self.face_cascade    = self._load_cascade("haarcascade_frontalface_default.xml")
        self.eye_cascade     = self._load_cascade("haarcascade_eye.xml")
        self.profile_cascade = self._load_cascade("haarcascade_profileface.xml")
        self.clahe           = cv2.createCLAHE(
            clipLimit=CONFIG["CLAHE_CLIP"], tileGridSize=CONFIG["CLAHE_GRID"])
        self.hog             = build_hog_descriptor()
        self.svm_clf, self.svm_scaler = (
            load_svm_model(model_path) if model_path else (None, None))
        self._eyebrow_history:     List[int] = []
        self._last_violation_time: float     = 0.0
        self.frame_count     = 0
        self.violation_count = 0
        if CONFIG["LOG_VIOLATIONS"]:
            Path(CONFIG["LOG_DIR"]).mkdir(exist_ok=True)

    # ── Cascade loader ────────────────────────────────────────
    @staticmethod
    def _load_cascade(filename: str) -> cv2.CascadeClassifier:
        path = os.path.join(cv2.data.haarcascades, filename)  # type: ignore
        if not os.path.exists(path):
            path = filename
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            raise FileNotFoundError(f"Cascade not found: {filename}")
        print(f"[OK] Loaded cascade: {filename}")
        return clf

    # ── Eyebrow smoothing ─────────────────────────────────────
    def _smooth_eyebrow_y(self, raw_y: int) -> int:
        self._eyebrow_history.append(raw_y)
        if len(self._eyebrow_history) > CONFIG["EYEBROW_SMOOTH_N"]:
            self._eyebrow_history.pop(0)
        return int(np.mean(self._eyebrow_history))

    # ── Gender classification ─────────────────────────────────
    def _classify_gender(self, face_gray_raw: np.ndarray) -> Tuple[str, float]:
        if self.svm_clf is None:
            return "male", 0.0
        try:
            feat_n  = self.svm_scaler.transform(  # type: ignore
                extract_hog_features(face_gray_raw, self.hog).reshape(1, -1))
            proba   = self.svm_clf.predict_proba(feat_n)[0]
            classes = list(self.svm_clf.classes_)
            female_idx  = classes.index(1) if 1 in classes else 1
            male_idx    = 1 - female_idx
            female_conf = float(proba[female_idx])
            male_conf   = float(proba[male_idx])
            thresh = CONFIG["GENDER_CONF_THRESHOLD"]
            if female_conf >= thresh:
                return "female", female_conf
            elif male_conf >= thresh:
                return "male", male_conf
            else:
                return "male", max(female_conf, male_conf)
        except Exception as e:
            print(f"[WARN] Gender SVM predict failed: {e}")
            return "male", 0.0

    # ── Frontal hair rule ─────────────────────────────────────
    def _threshold_classify(self, roi_gray: np.ndarray, roi_y1: int,
                            violation_line_y: int) -> Tuple[bool, Optional[int], int, float]:
        hair_mask        = binarize_hair(roi_gray)
        hair_pixel_count = int(np.count_nonzero(hair_mask))
        hair_density_pct = (1.0 - roi_gray.mean() / 255.0) * 100.0
        lowest_hair_y    = None
        is_violation     = False
        hair_rows = np.where(hair_mask.any(axis=1))[0]
        if len(hair_rows) and hair_pixel_count >= CONFIG["MIN_HAIR_AREA"]:
            lowest_hair_y = roi_y1 + int(np.percentile(hair_rows, CONFIG["HAIR_ROW_PERCENTILE"]))
            if lowest_hair_y >= violation_line_y:
                is_violation = True
        return is_violation, lowest_hair_y, hair_pixel_count, hair_density_pct

    # ── Vertical (tall/spiky) hair check ─────────────────────
    def _check_vertical_hair(
        self,
        gray_eq: np.ndarray,
        frame_shape: Tuple[int, int],
        fx: int, fy: int, fw: int, fh: int,
        annotated: np.ndarray,
        det: Detection,
    ) -> bool:
        """
        Extends the ROI search region far above the Haar cascade face box to
        detect hair that is styled vertically upwards (tall/spiky).

        Returns True if a vertical-height violation is detected.
        """
        H, W = frame_shape

        # ── Upper ROI bounds ──────────────────────────────────
        # Extend sideways 10 % of face width in each direction for coverage.
        upper_x1 = max(0, fx - int(fw * 0.10))
        upper_x2 = min(W, fx + fw + int(fw * 0.10))
        # Extend upwards by UPPER_HAIR_SEARCH_FRAC × fh above the face box top.
        upper_y1 = max(0, fy - int(fh * CONFIG["UPPER_HAIR_SEARCH_FRAC"]))
        upper_y2 = fy  # stop at the face-box top; forehead ROI handles below

        roi_w = upper_x2 - upper_x1
        roi_h = upper_y2 - upper_y1
        det.upper_hair_roi_box = (upper_x1, upper_y1, roi_w, roi_h)

        # ── Vertical limit line (max allowed hair top) ────────
        # Any hair pixel above this y-coordinate is a violation.
        det.vertical_limit_y = max(0, fy - int(fh * CONFIG["MAX_HAIR_HEIGHT_FRAC"]))

        # Draw the upper ROI bounding box
        cv2.rectangle(annotated, (upper_x1, upper_y1), (upper_x2, upper_y2),
                      CONFIG["UPPER_ROI_COLOR"], 1)
        cv2.putText(annotated, "Upper Hair ROI",
                    (upper_x1 + 4, upper_y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    CONFIG["UPPER_ROI_COLOR"], 1, cv2.LINE_AA)

        # Draw the vertical height limit as a dashed horizontal line
        self._dashed_hline(annotated, det.vertical_limit_y,
                           upper_x1, upper_x2, CONFIG["VERT_LIMIT_COLOR"])
        cv2.putText(
            annotated,
            f"Max vert. height ({CONFIG['MAX_HAIR_HEIGHT_FRAC']})",
            (upper_x1 + 4, det.vertical_limit_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33,
            CONFIG["VERT_LIMIT_COLOR"], 1, cv2.LINE_AA,
        )

        # Guard: ROI must have meaningful size
        if roi_h < 5 or roi_w < 5:
            return False

        upper_roi = gray_eq[upper_y1:upper_y2, upper_x1:upper_x2]
        if upper_roi.size == 0:
            return False

        mask = binarize_hair(upper_roi)
        det.upper_hair_mask = mask

        hair_rows  = np.where(mask.any(axis=1))[0]
        hair_count = int(np.count_nonzero(mask))

        if len(hair_rows) == 0 or hair_count < CONFIG["UPPER_MIN_HAIR_AREA"]:
            # No significant hair in the upper zone — compliant
            return False

        # Topmost (minimum row index) hair pixel in absolute frame coordinates
        det.highest_hair_y = upper_y1 + int(hair_rows.min())

        # Draw the detected hair-top line
        cv2.line(
            annotated,
            (upper_x1, det.highest_hair_y),
            (upper_x2, det.highest_hair_y),
            CONFIG["UPPER_ROI_COLOR"], 2,
        )
        cv2.putText(annotated, "Hair top",
                    (upper_x1 + 4, det.highest_hair_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    CONFIG["UPPER_ROI_COLOR"], 1, cv2.LINE_AA)

        # Violation: hair top is strictly ABOVE (smaller y) the limit line
        is_violation = det.highest_hair_y < det.vertical_limit_y
        det.vertical_violation = is_violation
        return is_violation

    # ── Profile ROI analysis ──────────────────────────────────
    def _analyze_profile_roi(self, gray_eq: np.ndarray,
                             box: Tuple[int, int, int, int],
                             ) -> Tuple[float, bool, Optional[np.ndarray]]:
        x, y, bw, bh = box
        if bw < 8 or bh < 8:
            return 0.0, False, None
        crop = gray_eq[y:y + bh, x:x + bw]
        if crop.size == 0:
            return 0.0, False, None
        crop_eq = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4)).apply(crop)
        mask    = binarize_side_hair(crop_eq)
        density = float(np.count_nonzero(mask)) / mask.size
        hair_pixels = crop_eq[mask > 0]
        if len(hair_pixels) == 0:
            return density, False, mask
        mean_darkness = float(hair_pixels.mean())
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_blob = max(
            (stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, n_labels)), default=0)
        is_violation = (
            density > CONFIG["PROFILE_DENSITY_MAX"]
            and mean_darkness < CONFIG["PROFILE_DARK_THRESH"]
            and max_blob >= CONFIG["PROFILE_MIN_CLUSTER"]
        )
        return density, is_violation, mask

    # ── Drawing helpers ───────────────────────────────────────
    @staticmethod
    def _dashed_hline(img, y, x1, x2, color, dash=8):
        for xd in range(x1, x2, dash * 2):
            cv2.line(img, (xd, y), (min(xd + dash, x2), y), color, 1)

    def _draw_profile_rois(self, img: np.ndarray, det: Detection) -> None:
        pairs = [
            (det.ear_roi_box,  det.ear_violation,  det.ear_density,  "Ear/Sideburn"),
            (det.neck_roi_box, det.neck_violation, det.neck_density, "Neck/Back"),
        ]
        for box, is_viol, density, label in pairs:
            if not box:
                continue
            x, y, bw, bh = box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), CONFIG["ROI_COLOR"], 1)
            color = CONFIG["VIOLATION_COLOR"] if is_viol else CONFIG["OK_COLOR"]
            cv2.putText(img, label, (x + 4, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    def _draw_gate_ui(self, img: np.ndarray, det: Detection,
                      extra_lines: Optional[List[str]] = None) -> None:
        h, w     = img.shape[:2]
        header_h = CONFIG["UI_HEADER_H"]
        alpha    = CONFIG["UI_PANEL_ALPHA"]
        status_map = {
            "ok":           ("COMPLIANT",                    CONFIG["OK_COLOR"]),
            "female_pass":  ("PASS",                         CONFIG["OK_COLOR"]),
            "violation":    ("POLICY VIOLATION - SEE GUARD", CONFIG["VIOLATION_COLOR"]),
            "no_face":      ("STEP INTO FRAME",              (180, 180, 180)),
            "no_eyes":      ("FACE CAMERA DIRECTLY",         CONFIG["WARN_COLOR"]),
            "side_profile": ("SIDE PROFILE — checking…",     CONFIG["WARN_COLOR"]),
        }
        if det.violation:
            status_text, status_color = status_map["violation"]
        else:
            status_text, status_color = status_map.get(det.status, ("...", (150, 150, 150)))

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        cv2.putText(img, CONFIG["UI_TITLE"], (10, header_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_SMALL"], (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(img, status_text, (10, header_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_LARGE"], status_color, 2, cv2.LINE_AA)

        if det.gender != "unknown":
            badge = f"Gender: {det.gender.upper()}  ({det.gender_confidence * 100:.0f}%)"
            cv2.putText(img, badge, (w - 280, header_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_SMALL"],
                        CONFIG["OK_COLOR"] if det.gender == "female" else CONFIG["WARN_COLOR"],
                        1, cv2.LINE_AA)

        if det.violation:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), CONFIG["VIOLATION_COLOR"], 4)

    def _log(self, frame: np.ndarray, det: Detection) -> None:
        cv2.imwrite(
            str(Path(CONFIG["LOG_DIR"]) /
                f"violation_{time.strftime('%Y%m%d_%H%M%S')}_{self.violation_count:04d}.jpg"),
            frame,
        )

    # ── Side-profile branch ───────────────────────────────────
    def _check_side_profile(self, frame: np.ndarray, gray_eq: np.ndarray,
                            annotated: np.ndarray, det: Detection,
                            ) -> Tuple[np.ndarray, Detection]:
        H, W = gray_eq.shape
        profiles = self.profile_cascade.detectMultiScale(
            gray_eq, scaleFactor=CONFIG["PROFILE_SCALE"],
            minNeighbors=CONFIG["PROFILE_NEIGHBORS"],
            minSize=CONFIG["PROFILE_MIN_SIZE"],
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        facing_right = False
        if not len(profiles):
            gray_flipped = cv2.flip(gray_eq, 1)
            profiles = self.profile_cascade.detectMultiScale(
                gray_flipped, scaleFactor=CONFIG["PROFILE_SCALE"],
                minNeighbors=CONFIG["PROFILE_NEIGHBORS"],
                minSize=CONFIG["PROFILE_MIN_SIZE"],
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if not len(profiles):
                det.status = "no_face"
                self._draw_gate_ui(annotated, det)
                return annotated, det
            facing_right = True
            profiles = np.array([[W - fx - fw, fy, fw, fh] for fx, fy, fw, fh in profiles])

        fx, fy, fw, fh = (int(v) for v in max(profiles, key=lambda r: r[2] * r[3]))
        det.profile_face_box = (fx, fy, fw, fh)
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (210, 210, 210), 1)
        cv2.putText(annotated, "Right-facing" if facing_right else "Left-facing",
                    (fx + 4, max(fy - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    CONFIG["UI_FONT_SMALL"], (210, 210, 210), 1, cv2.LINE_AA)

        gray_raw  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_crop = gray_raw[fy:fy + fh, fx:fx + fw]
        if face_crop.size > 0:
            det.gender, det.gender_confidence = self._classify_gender(face_crop)
        if det.gender == "female":
            det.status = "female_pass"
            self._draw_gate_ui(annotated, det)
            return annotated, det

        def _roi_box(xfrac, yfrac, mirror):
            if mirror:
                xfrac = (1.0 - xfrac[1], 1.0 - xfrac[0])
            return (
                max(0, fx + int(fw * xfrac[0])),
                max(0, fy + int(fh * yfrac[0])),
                max(0, min(W, fx + int(fw * xfrac[1])) - max(0, fx + int(fw * xfrac[0]))),
                max(0, min(H, fy + int(fh * yfrac[1])) - max(0, fy + int(fh * yfrac[0]))),
            )

        det.ear_roi_box  = _roi_box(CONFIG["PROFILE_EAR_X"],  CONFIG["PROFILE_EAR_Y"],  facing_right)
        det.neck_roi_box = _roi_box(CONFIG["PROFILE_NECK_X"], CONFIG["PROFILE_NECK_Y"], facing_right)
        det.ear_density,  det.ear_violation,  det.ear_mask  = self._analyze_profile_roi(gray_eq, det.ear_roi_box)
        det.neck_density, det.neck_violation, det.neck_mask = self._analyze_profile_roi(gray_eq, det.neck_roi_box)
        det.status = "side_profile"
        self._draw_profile_rois(annotated, det)

        now = time.monotonic()
        if det.ear_violation or det.neck_violation:
            det.violation = True
            det.status    = "violation"
            self._last_violation_time = now
            self.violation_count += 1
            if CONFIG["LOG_VIOLATIONS"]:
                self._log(frame, det)
        else:
            det.status = "ok"
        if (not det.violation and det.profile_face_box
                and now - self._last_violation_time < CONFIG["VIOLATION_HOLD_S"]):
            det.violation = True
            det.status    = "violation"
        self._draw_gate_ui(annotated, det)
        return annotated, det

    # ── Main analysis entry point ─────────────────────────────
    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Detection]:
        self.frame_count += 1
        det       = Detection()
        annotated = frame.copy()
        gray_raw  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq   = preprocess_frame(frame, self.clahe)

        faces = self.face_cascade.detectMultiScale(
            gray_eq, scaleFactor=CONFIG["FACE_SCALE"],
            minNeighbors=CONFIG["FACE_NEIGHBORS"],
            minSize=CONFIG["FACE_MIN_SIZE"],
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if not len(faces):
            return self._check_side_profile(frame, gray_eq, annotated, det)

        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        det.face_box = (fx, fy, fw, fh)
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (210, 210, 210), 1)

        eyes = self.eye_cascade.detectMultiScale(
            gray_eq[fy:fy + int(fh * 0.55), fx:fx + fw],
            scaleFactor=CONFIG["EYE_SCALE"],
            minNeighbors=CONFIG["EYE_NEIGHBORS"],
            minSize=(22, 22),
        )
        if not len(eyes):
            det.status = "no_eyes"
            self._draw_gate_ui(annotated, det)
            return annotated, det

        abs_eyes = [(fx + ex, fy + ey, ew, eh) for ex, ey, ew, eh in eyes]
        det.eye_boxes = abs_eyes
        for (ex, ey, ew, eh) in abs_eyes:
            cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (190, 190, 60), 1)
        det.eyebrow_y = self._smooth_eyebrow_y(min(abs_eyes, key=lambda e: e[1])[1])
        self._dashed_hline(annotated, det.eyebrow_y, fx, fx + fw, CONFIG["BROW_LINE_COLOR"])
        cv2.putText(annotated, "Eyebrow level", (fx + 4, det.eyebrow_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, CONFIG["BROW_LINE_COLOR"], 1, cv2.LINE_AA)

        face_crop_raw = gray_raw[fy:fy + fh, fx:fx + fw]
        det.face_crop_raw = face_crop_raw.copy() if face_crop_raw.size > 0 else None
        if face_crop_raw.size > 0:
            det.gender, det.gender_confidence = self._classify_gender(face_crop_raw)
        det.classifier_used = f"gender_svm({det.gender})" if self.svm_clf else "no_model"

        if det.gender == "female":
            det.status = "female_pass"
            self._draw_gate_ui(annotated, det)
            return annotated, det

        # ── Male — forehead (bang) check ──────────────────────
        roi_y1        = max(0, fy - int(fh * CONFIG["FOREHEAD_TOP_FRAC"]))
        eyebrow_limit = (det.eyebrow_y - 8) if det.eyebrow_y else (fy + int(fh * CONFIG["FOREHEAD_BOT_FRAC"]))
        roi_y2        = max(
            min(frame.shape[0], fy + int(fh * CONFIG["FOREHEAD_BOT_FRAC"]), eyebrow_limit),
            roi_y1 + 15,
        )
        roi_x1 = fx + int(fw * CONFIG["FOREHEAD_LEFT_FRAC"])
        roi_x2 = fx + fw - int(fw * CONFIG["FOREHEAD_RIGHT_FRAC"])
        det.forehead_roi_box = (roi_x1, roi_y1, max(1, roi_x2 - roi_x1), roi_y2 - roi_y1)
        cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), CONFIG["ROI_COLOR"], 1)
        cv2.putText(annotated, "Hair ROI", (roi_x1 + 4, roi_y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, CONFIG["ROI_COLOR"], 1, cv2.LINE_AA)

        det.violation_line_y = roi_y1 + int((roi_y2 - roi_y1) * CONFIG["HAIR_VIOLATION_FRAC"])
        self._dashed_hline(annotated, det.violation_line_y, roi_x1, roi_x2, CONFIG["THRESH_LINE_COLOR"])
        cv2.putText(
            annotated,
            f"Violation line ({CONFIG['HAIR_VIOLATION_FRAC']})",
            (roi_x1 + 4, det.violation_line_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, CONFIG["THRESH_LINE_COLOR"], 1, cv2.LINE_AA)

        roi_gray     = gray_eq[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_gray_raw = gray_raw[roi_y1:roi_y2, roi_x1:roi_x2]
        det.roi_gray     = roi_gray.copy()     if roi_gray.size     else None
        det.roi_gray_raw = roi_gray_raw.copy() if roi_gray_raw.size else None

        bang_violation = False
        if roi_gray.size > 0 and roi_gray.shape[0] >= 5 and roi_gray.shape[1] >= 5:
            bang_violation, det.lowest_hair_y, det.hair_pixel_count, det.hair_density_pct = \
                self._threshold_classify(roi_gray, roi_y1, det.violation_line_y)

        if det.lowest_hair_y is not None:
            cv2.line(annotated, (roi_x1, det.lowest_hair_y), (roi_x2, det.lowest_hair_y),
                     CONFIG["HAIR_LINE_COLOR"], 2)
            cv2.putText(annotated, "Hair edge", (roi_x1 + 4, det.lowest_hair_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, CONFIG["HAIR_LINE_COLOR"], 1, cv2.LINE_AA)

        # ── Male — vertical (tall/spiky) hair check ───────────
        vertical_violation = self._check_vertical_hair(
            gray_eq, gray_eq.shape[:2], fx, fy, fw, fh, annotated, det)

        # ── Combine violations and finalise status ────────────
        is_viol = bang_violation or vertical_violation
        now = time.monotonic()
        if is_viol:
            det.violation = True
            det.status    = "violation"
            self._last_violation_time = now
            self.violation_count += 1
            if CONFIG["LOG_VIOLATIONS"]:
                self._log(frame, det)
        else:
            det.status = "ok"
        if (not det.violation and det.face_box
                and now - self._last_violation_time < CONFIG["VIOLATION_HOLD_S"]):
            det.violation = True
            det.status    = "violation"

        self._draw_gate_ui(annotated, det)
        return annotated, det