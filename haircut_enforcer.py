"""
Haircut Policy Enforcement System
===================================
School gate real-time hair length checker using OpenCV Haar Cascades
+ HOG/SVM learned classifier for improved accuracy.

Camera    : Eye-level frontal (direct face-on view)
Lighting  : Indoor fluorescent (stable intensity, minimal shadows)
Mode      : Live webcam / IP camera — real-time streaming

Policy    : Hair must NOT extend below the violation line inside the forehead ROI.
            Violation line position is controlled by HAIR_VIOLATION_FRAC.

──────────────────────────────────────────────────────────────────────────────
TRAINING WORKFLOW
──────────────────────────────────────────────────────────────────────────────

Step 1 — Collect labelled samples (live webcam):
    python haircut_enforcer.py --mode collect --dataset my_dataset
    Keys:  P = save positive (hair in forehead ROI = VIOLATION)
           N = save negative (no hair / compliant)
           Q = quit collection

    Samples are saved as 64×64 crops of the forehead ROI, already aligned
    to the face detector.  Aim for 200+ samples per class.

Step 2 — Train the SVM classifier:
    python haircut_enforcer.py --mode train_svm --dataset my_dataset --model models/hair_svm.joblib

    This will:
      • Load positive / negative images from my_dataset/
      • Apply 8 augmentations per image (flip, brightness, noise, rotate, shadow…)
      • Extract HOG feature vectors (64×64, 8×8 cells, 9 orientation bins)
      • Train a linear SVM with 5-fold cross-validation
      • Report precision / recall / F1 on the held-out validation split
      • Save the trained model to the specified path

Step 3 — Evaluate on validation set:
    python haircut_enforcer.py --mode eval --dataset my_dataset --model models/hair_svm.joblib

Step 4 — Run live enforcement with the learned model:
    python haircut_enforcer.py --mode live --model models/hair_svm.joblib

    If --model is omitted the system falls back to the threshold-based detector.

──────────────────────────────────────────────────────────────────────────────
Tuning notes for this deployment:
  - CLAHE is lighter (clip=1.8) since fluorescent light is stable
  - FACE_NEIGHBORS=6 is stricter to avoid false positives at a busy gate
  - Eye-level camera means near-zero head tilt — frontal cascade is optimal
  - No tilt correction needed (side-angle or overhead would need it)
  - HAIR_VIOLATION_FRAC controls fallback strictness: lower = stricter (0.3 is strict)
  - HAIR_ROW_PERCENTILE controls hair edge position: lower = higher edge

Author: Computer Vision Pipeline — OpenCV + NumPy + scikit-learn
"""

import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  — Tuned for eye-level frontal, fluorescent, live
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Camera
    "CAMERA_INDEX": 0,           # Integer for webcam; RTSP URL string for IP cam
                                 # e.g. "rtsp://admin:pass@192.168.1.100/stream"
    "FRAME_WIDTH":  640,
    "FRAME_HEIGHT": 480,
    "TARGET_FPS":   30,

    # Preprocessing
    "GAUSS_KERNEL":  (5, 5),     # Odd integers only
    "CLAHE_CLIP":    1.8,
    "CLAHE_GRID":    (8, 8),
    "MORPH_KERNEL":  (3, 3),

    # Hair binarization (fallback when no SVM model loaded)
    "HAIR_THRESH":         80,   # Pixels darker than this = hair candidate (frontal ROI)
    "SIDE_HAIR_ABS_THRESH": 90,  # Gate 2 absolute threshold for side-profile ROIs.
                                 # Hair is typically <80; skin (even in shadow) is >100.
                                 # Lower = stricter (fewer pixels pass). Range: 60–120.
    "MIN_HAIR_AREA":      300,   # Ignore tiny dark specks (noise floor)

    # Hair edge position: percentile of hair rows used as the "hair edge" line.
    # Lower = edge moves higher (stricter). Range: 50–100.
    "HAIR_ROW_PERCENTILE": 85,

    # Violation threshold (fallback): fraction of ROI height below which hair = violation.
    "HAIR_VIOLATION_FRAC": 0.9,

    # Haar cascade parameters
    "FACE_SCALE":     1.15,
    "FACE_NEIGHBORS": 6,
    "EYE_SCALE":      1.08,
    "EYE_NEIGHBORS":  9,
    "FACE_MIN_SIZE":  (90, 90),

    # Forehead ROI definition (fractions of face height)
    "FOREHEAD_TOP_FRAC": 0.08,
    "FOREHEAD_BOT_FRAC": 0.18,

    # Smoothing: rolling window for stable eyebrow Y estimate
    "EYEBROW_SMOOTH_N": 5,

    # HOG parameters (must match what was used during training)
    "HOG_WIN_SIZE":   (64, 64),
    "HOG_BLOCK":      (16, 16),
    "HOG_BLOCK_STEP": (8, 8),
    "HOG_CELL":       (8, 8),
    "HOG_BINS":       9,

    # Visual output colors (BGR)
    "VIOLATION_COLOR":  (0,   0, 220),
    "OK_COLOR":         (0, 200,  60),
    "WARN_COLOR":       (0, 165, 255),
    "ROI_COLOR":        (255, 220, 80),
    "HAIR_LINE_COLOR":  (80,  80, 255),
    "BROW_LINE_COLOR":  (255, 180,  0),
    "THRESH_LINE_COLOR":(0,  220,  0),

    # Logging
    "LOG_VIOLATIONS": False,
    "LOG_DIR":        "violation_logs",

    # Gate display: hold VIOLATION result this many seconds after detection
    "VIOLATION_HOLD_S": 1.0,

    # Profile (side-face) fallback
    "PROFILE_SCALE":     1.15,
    "PROFILE_NEIGHBORS": 5,
    "PROFILE_MIN_SIZE":  (90, 90),

    # ── Side-profile ROI fractions (relative to profile bbox) ─────────────
    # haarcascade_profileface.xml detects LEFT-facing heads.  In that bbox:
    #   - Nose/chin → LEFT edge of bbox
    #   - Ear/sideburn → RIGHT ~60-97 % of bbox width
    #   - Neck/back-of-head hair → lower-right quadrant
    #
    # ORIGINAL BUG: (0.55, 0.95) / (0.30, 0.65) placed ear box too far left,
    # cutting across the cheek.  Neck box (0.50, 1.00) was too wide.
    # FIXED fractions below are tighter and correctly target the hair zones.
    "PROFILE_EAR_X":  (0.62, 0.97),   # tighter to RIGHT edge where ear/sideburn is
    "PROFILE_EAR_Y":  (0.22, 0.58),   # starts higher (top of ear) down to sideburn
    "PROFILE_NECK_X": (0.78, 1.22),   # excludes far-right neck skin and left face
    "PROFILE_NECK_Y": (0.60, 0.90),   # lower 30 % of bbox (chin/neck junction)

    "PROFILE_EAR_COLOR":  (60, 200, 255),   # orange-ish dashed border
    "PROFILE_NECK_COLOR": (255, 120, 200),  # magenta-ish dashed border

    # ── Unified Gate-Monitor UI layout ─────────────────────────────────────
    # Pixel offsets are used consistently across every camera function so
    # all live windows look identical.
    "UI_HEADER_H":    48,     # px - top status bar height
    "UI_FOOTER_H":    56,     # px - bottom debug/button panel height
    "UI_PANEL_ALPHA": 0.72,   # semi-transparency for both panels
    "UI_FONT_LARGE":  0.70,   # font scale - primary status text
    "UI_FONT_MED":    0.48,   # font scale - secondary info (collect stats, etc.)
    "UI_FONT_SMALL":  0.36,   # font scale - debug telemetry lines
    "UI_TITLE":       "HAIRCUT POLICY GATE MONITOR",
}


# ─────────────────────────────────────────────────────────────
# HOG FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def build_hog_descriptor() -> cv2.HOGDescriptor:
    """Create a HOGDescriptor using the CONFIG parameters."""
    return cv2.HOGDescriptor(
        CONFIG["HOG_WIN_SIZE"],
        CONFIG["HOG_BLOCK"],
        CONFIG["HOG_BLOCK_STEP"],
        CONFIG["HOG_CELL"],
        CONFIG["HOG_BINS"],
    )


def extract_hog_features(roi_gray: np.ndarray,
                          hog: cv2.HOGDescriptor) -> np.ndarray:
    """
    Resize roi_gray to HOG_WIN_SIZE, apply CLAHE, then compute HOG vector.
    Returns a 1-D float32 array.
    """
    win_w, win_h = CONFIG["HOG_WIN_SIZE"]
    resized = cv2.resize(roi_gray, (win_w, win_h), interpolation=cv2.INTER_AREA)
    clahe   = cv2.createCLAHE(clipLimit=CONFIG["CLAHE_CLIP"],
                               tileGridSize=CONFIG["CLAHE_GRID"])
    eq      = clahe.apply(resized)
    eq      = cv2.GaussianBlur(eq, CONFIG["GAUSS_KERNEL"], 0)
    feat    = hog.compute(eq)
    return feat.flatten().astype(np.float32)


# ─────────────────────────────────────────────────────────────
# INTEGRAL IMAGE  — O(1) rectangular pixel sum
# ─────────────────────────────────────────────────────────────

def build_integral_image(gray: np.ndarray) -> np.ndarray:
    return cv2.integral(gray)


def rect_sum(integral: np.ndarray,
             r1: int, c1: int, r2: int, c2: int) -> int:
    return (int(integral[r2, c2])
            - int(integral[r1, c2])
            - int(integral[r2, c1])
            + int(integral[r1, c1]))


# ─────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_frame(frame: np.ndarray,
                     clahe: cv2.CLAHE
                     ) -> Tuple[np.ndarray, np.ndarray]:
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq  = clahe.apply(gray)
    gray_eq  = cv2.GaussianBlur(gray_eq, CONFIG["GAUSS_KERNEL"], 0)
    normalized = gray_eq.astype(np.float32) / 255.0
    return gray_eq, normalized


def binarize_hair(roi_gray: np.ndarray) -> np.ndarray:
    """
    Frontal forehead ROI binarizer.

    BUG FIX: The original code used ``cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU``
    which causes OpenCV to *ignore* the manual ``HAIR_THRESH`` value entirely —
    Otsu overwrites it with its own computed threshold.  That makes the CONFIG
    knob ``+/-`` useless and causes unstable results across lighting changes.

    Fix: use the manual threshold directly (``THRESH_BINARY_INV`` only).
    Otsu is kept as an *optional* fallback only when the ROI histogram is
    bimodal enough to benefit from it (detected by checking the Otsu threshold
    diverges < 30 points from the manual one).
    """
    manual_thresh = CONFIG["HAIR_THRESH"]

    # Compute Otsu threshold without applying it
    otsu_val, _ = cv2.threshold(
        roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Only use Otsu if it agrees closely with the manual threshold;
    # otherwise trust the operator-tuned value.
    use_thresh = int(otsu_val) if abs(int(otsu_val) - manual_thresh) < 30 else manual_thresh

    _, binary = cv2.threshold(roi_gray, use_thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG["MORPH_KERNEL"])
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def binarize_side_hair(roi_gray: np.ndarray) -> np.ndarray:
    """
    Side-profile ROI binarizer using a HYBRID dual-gate approach.

    ROOT CAUSE OF THE OVER-DETECTION BUG (ear=1398px, neck=1836px):
    ────────────────────────────────────────────────────────────────
    The previous version used adaptiveThreshold with C=8 alone.
    Adaptive thresholding flags any pixel that is > C intensity units
    BELOW its local neighbourhood mean.  With C=8 on a textured skin
    region, many pixels naturally differ from their neighbours by more
    than 8 units (pores, shadows, lighting gradients across the cheek
    and neck).  Result: almost the entire ROI is marked as "hair".

    The fix uses TWO independent gates.  A pixel is only counted as hair
    if it passes BOTH:

      Gate 1 — Adaptive (handles non-uniform lighting)
               Pixel must be darker than its 21×21 neighbourhood mean by
               at least C=22 units.  C=22 is strict enough to ignore
               ordinary skin texture but still catches a dark hair strand
               against a mid-tone background.

      Gate 2 — Absolute (ground truth: hair is genuinely dark)
               Pixel's raw intensity must be < SIDE_HAIR_ABS_THRESH (90).
               Skin (even in shadow) typically reads 100–200 on a CLAHE-
               enhanced image.  Actual hair reads 20–80.  This gate stops
               skin folds and shadow edges from passing Gate 1 alone.

    Combined: binary = adaptive_mask AND absolute_mask

    Additionally, connected-component filtering removes blobs smaller than
    MIN_BLOB_PX (40 px²).  A hair strand at gate distance covers at least
    40 px² in a 64–150 px wide ROI.  Isolated specks (noise, pores) are
    smaller and are discarded.
    """
    MIN_BLOB_PX = 40   # minimum connected component area to keep

    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    # ── Gate 1: adaptive threshold (lighting-invariant) ───────────────────
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,   # odd; smaller block → more local comparison
        C=22,           # raised from 8 → only flag pixels significantly
    )                   # darker than their neighbourhood (not just textured)

    # ── Gate 2: absolute intensity (hair is genuinely dark) ───────────────
    abs_thresh = CONFIG.get("SIDE_HAIR_ABS_THRESH", 90)
    _, absolute_mask = cv2.threshold(blurred, abs_thresh, 255, cv2.THRESH_BINARY_INV)

    # Both gates must agree
    binary = cv2.bitwise_and(adaptive_mask, absolute_mask)

    # Morphological cleanup
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  open_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # ── Connected-component filter: discard tiny blobs ────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for lbl in range(1, n_labels):   # 0 is background
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_BLOB_PX:
            filtered[labels == lbl] = 255

    return filtered


# ─────────────────────────────────────────────────────────────
# DETECTION RESULT CONTAINER
# ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    face_box:          Optional[Tuple[int,int,int,int]] = None
    eye_boxes:         List[Tuple]                      = field(default_factory=list)
    eyebrow_y:         Optional[int]                    = None
    forehead_roi_box:  Optional[Tuple[int,int,int,int]] = None
    roi_gray:          Optional[np.ndarray]             = None  # copy for collect/save after frame returns
    lowest_hair_y:     Optional[int]                    = None
    violation_line_y:  Optional[int]                    = None
    hair_pixel_count:  int                              = 0
    hair_density_pct:  float                            = 0.0
    violation:         bool                             = False
    status:            str                              = "no_face"
    classifier_used:   str                              = "threshold"   # "svm" or "threshold"
    svm_confidence:    float                            = 0.0
    profile_face_box:  Optional[Tuple[int,int,int,int]] = None
    ear_roi_box:       Optional[Tuple[int,int,int,int]] = None
    neck_roi_box:      Optional[Tuple[int,int,int,int]] = None
    ear_pixel_count:   int                              = 0
    neck_pixel_count:  int                              = 0


# ─────────────────────────────────────────────────────────────
# INTERACTIVE BUTTON CONTROL
# ─────────────────────────────────────────────────────────────

class Button:
    """Clickable button for controlling parameters in live mode."""
    
    def __init__(self, x: int, y: int, w: int, h: int, label: str, 
                 action_fn, button_id: str):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.action_fn = action_fn
        self.button_id = button_id
        self.hovered = False
    
    def contains_point(self, px: int, py: int) -> bool:
        """Check if point (px, py) is inside this button."""
        return (self.x <= px < self.x + self.w and 
                self.y <= py < self.y + self.h)
    
    def draw(self, img: np.ndarray):
        """Draw button on image."""
        color = (100, 200, 100) if self.hovered else (80, 150, 80)
        text_color = (255, 255, 255)
        
        # Draw button rectangle
        cv2.rectangle(img, (self.x, self.y), 
                      (self.x + self.w, self.y + self.h),
                      color, -1)
        
        # Draw border
        cv2.rectangle(img, (self.x, self.y), 
                      (self.x + self.w, self.y + self.h),
                      (255, 255, 255), 2)
        
        # Draw text (centered)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        
        cv2.putText(img, self.label, (text_x, text_y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    def on_click(self):
        """Trigger the action."""
        if self.action_fn:
            self.action_fn()


def _param_button_action(kind: str, delta: float) -> None:
    """Shared click handler for violation-line vs threshold buttons."""
    if kind == "viol":
        v = max(0.05, min(1.0, round(CONFIG["HAIR_VIOLATION_FRAC"] + delta, 2)))
        CONFIG["HAIR_VIOLATION_FRAC"] = v
        suffix = " (stricter)" if delta < 0 else " (more lenient)"
        print(f"[INFO] HAIR_VIOLATION_FRAC → {CONFIG['HAIR_VIOLATION_FRAC']}{suffix}")
    else:
        CONFIG["HAIR_THRESH"] = max(10, min(240, CONFIG["HAIR_THRESH"] + int(delta)))
        print(f"[INFO] HAIR_THRESH → {CONFIG['HAIR_THRESH']}")


def _create_param_buttons(frame_w: int, frame_h: int) -> List[Button]:
    """
    Create the four parameter buttons.

    LAYOUT FIX: Buttons are now positioned inside the dedicated footer panel
    (bottom UI_FOOTER_H pixels) so they don't overlap the camera feed or the
    debug telemetry lines that are drawn at the TOP of the footer panel.

    Footer panel height = UI_FOOTER_H (56 px by default).
    Debug text occupies the upper ~28 px (2 lines × 14 px).
    Buttons occupy the lower 28 px of the footer.
    """
    footer_h  = CONFIG["UI_FOOTER_H"]
    button_h  = 22
    button_w  = 50
    # Position buttons in the bottom portion of the footer panel
    button_y  = frame_h - (button_h + 6)   # 6 px margin from bottom edge

    specs = [
        ("[<<]", 10,  "viol",   -0.05, "decrease_violation"),
        ("[>>]", 68,  "viol",   +0.05, "increase_violation"),
        ("[-]",  126, "thresh", -5,    "decrease_threshold"),
        ("[+]",  184, "thresh", +5,    "increase_threshold"),
    ]
    out: List[Button] = []
    for label, x, kind, delta, bid in specs:
        out.append(Button(
            x=x, y=button_y, w=button_w, h=button_h,
            label=label,
            action_fn=(lambda k=kind, d=delta: _param_button_action(k, d)),
            button_id=bid,
        ))
    return out


def _handle_param_keys(key: int) -> bool:
    """Handle + = - [ ] , . ; return True if key was consumed."""
    if key in (ord('+'), ord('=')):
        CONFIG["HAIR_THRESH"] = min(CONFIG["HAIR_THRESH"] + 5, 240)
        print(f"[INFO] HAIR_THRESH → {CONFIG['HAIR_THRESH']}")
        return True
    if key == ord('-'):
        CONFIG["HAIR_THRESH"] = max(CONFIG["HAIR_THRESH"] - 5, 10)
        print(f"[INFO] HAIR_THRESH → {CONFIG['HAIR_THRESH']}")
        return True
    if key == ord('['):
        CONFIG["HAIR_VIOLATION_FRAC"] = max(
            round(CONFIG["HAIR_VIOLATION_FRAC"] - 0.05, 2), 0.05)
        print(f"[INFO] HAIR_VIOLATION_FRAC → {CONFIG['HAIR_VIOLATION_FRAC']} (stricter)")
        return True
    if key == ord(']'):
        CONFIG["HAIR_VIOLATION_FRAC"] = min(
            round(CONFIG["HAIR_VIOLATION_FRAC"] + 0.05, 2), 1.0)
        print(f"[INFO] HAIR_VIOLATION_FRAC → {CONFIG['HAIR_VIOLATION_FRAC']} (more lenient)")
        return True
    if key == ord(','):
        CONFIG["HAIR_ROW_PERCENTILE"] = max(CONFIG["HAIR_ROW_PERCENTILE"] - 5, 10)
        print(f"[INFO] HAIR_ROW_PERCENTILE → {CONFIG['HAIR_ROW_PERCENTILE']} (edge higher)")
        return True
    if key == ord('.'):
        CONFIG["HAIR_ROW_PERCENTILE"] = min(CONFIG["HAIR_ROW_PERCENTILE"] + 5, 100)
        print(f"[INFO] HAIR_ROW_PERCENTILE → {CONFIG['HAIR_ROW_PERCENTILE']} (edge lower)")
        return True
    if key == ord(';'):
        CONFIG["SIDE_HAIR_ABS_THRESH"] = max(CONFIG["SIDE_HAIR_ABS_THRESH"] - 5, 30)
        print(f"[INFO] SIDE_HAIR_ABS_THRESH -> {CONFIG['SIDE_HAIR_ABS_THRESH']} (stricter side gate)")
        return True
    if key == ord("'"):
        CONFIG["SIDE_HAIR_ABS_THRESH"] = min(CONFIG["SIDE_HAIR_ABS_THRESH"] + 5, 180)
        print(f"[INFO] SIDE_HAIR_ABS_THRESH -> {CONFIG['SIDE_HAIR_ABS_THRESH']} (more lenient side gate)")
        return True
    return False


# ─────────────────────────────────────────────────────────────
# AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────

class Augmentor:
    """
    Applies a suite of augmentations to a single grayscale ROI crop.
    Returns a list of augmented images (including the original).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def augment(self, img: np.ndarray) -> List[np.ndarray]:
        """Return original + up to 8 augmented variants."""
        results = [img.copy()]

        # 1. Horizontal flip
        results.append(cv2.flip(img, 1))

        # 2. Brightness up (+40)
        bright = np.clip(img.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        results.append(bright)

        # 3. Brightness down (−40)
        dark = np.clip(img.astype(np.int16) - 40, 0, 255).astype(np.uint8)
        results.append(dark)

        # 4. Gaussian noise
        noise  = self.rng.normal(0, 12, img.shape).astype(np.int16)
        noisy  = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        results.append(noisy)

        # 5. Slight rotation (±8°)
        h, w    = img.shape[:2]
        angle   = self.rng.uniform(-8, 8)
        M       = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT_101)
        results.append(rotated)

        # 6. Simulated shadow (horizontal gradient overlay)
        shadow_dir = self.rng.integers(0, 2)  # 0=left, 1=right
        shadow = np.linspace(0.55, 1.0, w if shadow_dir == 0 else h)
        if shadow_dir == 1:
            shadow = shadow[:, np.newaxis]
        else:
            shadow = shadow[np.newaxis, :]
        shadowed = np.clip(img.astype(np.float32) * shadow, 0, 255).astype(np.uint8)
        results.append(shadowed)

        # 7. Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
        results.append(blurred)

        # 8. Flip + noise combined
        fn = np.clip(cv2.flip(img, 1).astype(np.int16) + noise, 0, 255).astype(np.uint8)
        results.append(fn)

        return results


# ─────────────────────────────────────────────────────────────
# DATASET MANAGER
# ─────────────────────────────────────────────────────────────

class DatasetManager:
    """
    Manages the on-disk dataset layout:

        <root>/
            positive/   ← forehead ROI crops where hair is in violation
            negative/   ← forehead ROI crops that are compliant

    Images are saved as 64×64 grayscale PNG files.
    """

    def __init__(self, root: str):
        self.root     = Path(root)
        self.pos_dir  = self.root / "positive"
        self.neg_dir  = self.root / "negative"
        self.pos_dir.mkdir(parents=True, exist_ok=True)
        self.neg_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Dataset] Root: {self.root.resolve()}")

    def save_sample(self, roi_gray: np.ndarray, label: str) -> Path:
        """Save a single ROI crop.  label must be 'positive' or 'negative'."""
        assert label in ("positive", "negative"), "label must be positive or negative"
        target_dir = self.pos_dir if label == "positive" else self.neg_dir
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = target_dir / f"{label}_{ts}.png"
        resized = cv2.resize(roi_gray,
                             CONFIG["HOG_WIN_SIZE"],
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), resized)
        return out

    def load_dataset(self,
                     val_split: float = 0.2,
                     augment: bool    = True
                     ) -> Tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
        """
        Load all images, optionally augment the training split, and return
        (X_train, y_train, X_val, y_val) as HOG feature arrays.

        BUG FIX — chronological split replaced with shuffled split:
        The original code used ``sorted(src_dir.glob("*.png"))`` then split
        by index, so the *most-recently collected* samples always ended up in
        the validation set.  If the operator collected all positives in one
        session and negatives in another, one class could be entirely absent
        from validation (silent class-imbalance in eval).

        Fix: shuffle path lists with a fixed random seed before splitting so
        each class is proportionally represented in both splits and the split
        is reproducible across runs.
        """
        import random as _random   # local import to avoid polluting module namespace
        rng = _random.Random(42)   # fixed seed → reproducible train/val split

        hog = build_hog_descriptor()
        aug = Augmentor() if augment else None
        X_tr, y_tr, X_va, y_va = [], [], [], []

        for label_str, label_int in [("positive", 1), ("negative", 0)]:
            src_dir = self.pos_dir if label_str == "positive" else self.neg_dir
            paths   = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
            if not paths:
                print(f"[WARN] No images in {src_dir}")
                continue

            # ── SHUFFLED split (reproducible) ─────────────────────────────
            paths = list(paths)
            rng.shuffle(paths)                           # in-place shuffle

            split_idx   = max(1, int(len(paths) * (1.0 - val_split)))
            train_paths = paths[:split_idx]
            val_paths   = paths[split_idx:]

            print(f"[{label_str}] train={len(train_paths)}  val={len(val_paths)}")

            for p in train_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                variants = aug.augment(img) if aug else [img]
                for v in variants:
                    feat = extract_hog_features(v, hog)
                    X_tr.append(feat)
                    y_tr.append(label_int)

            for p in val_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                feat = extract_hog_features(img, hog)
                X_va.append(feat)
                y_va.append(label_int)

        X_tr = np.array(X_tr, dtype=np.float32)
        y_tr = np.array(y_tr, dtype=np.int32)
        # Guard against all-empty val set (e.g. if only one class collected)
        feat_dim = X_tr.shape[1] if len(X_tr) > 0 else 0
        X_va = (np.array(X_va, dtype=np.float32)
                if X_va else np.empty((0, feat_dim), dtype=np.float32))
        y_va = np.array(y_va, dtype=np.int32)

        print(f"[Dataset] Train: {len(X_tr)}  Val: {len(X_va)}")
        return X_tr, y_tr, X_va, y_va


# ─────────────────────────────────────────────────────────────
# SVM TRAINER
# ─────────────────────────────────────────────────────────────

def train_svm(dataset_root: str,
              model_path:   str = "models/hair_svm.joblib",
              val_split:    float = 0.2) -> None:
    """
    Full training pipeline:
      1. Load & augment dataset
      2. Normalise features (zero mean, unit variance)
      3. Cross-validate C parameter on training set
      4. Fit final SVM on all training data
      5. Evaluate on validation split
      6. Save model + scaler to model_path
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import (classification_report,
                                     confusion_matrix, f1_score)
        import joblib
    except ImportError:
        print("[ERROR] scikit-learn and joblib are required for training.")
        print("        pip install scikit-learn joblib")
        sys.exit(1)

    dm = DatasetManager(dataset_root)
    X_tr, y_tr, X_va, y_va = dm.load_dataset(val_split=val_split, augment=True)

    if len(X_tr) == 0:
        print("[ERROR] Training set is empty.  Run --mode collect first.")
        sys.exit(1)

    # ── Normalise ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_va_n = scaler.transform(X_va) if len(X_va) > 0 else X_va

    # ── Cross-validate C ──────────────────────────────────────
    print("\n[Train] Cross-validating C parameter …")
    best_c, best_f1 = 0.1, 0.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for c in [0.01, 0.1, 1.0, 10.0]:
        clf_cv = SVC(kernel="linear", C=c, probability=True, random_state=42)
        scores = cross_val_score(clf_cv, X_tr_n, y_tr, cv=cv,
                                 scoring="f1", n_jobs=-1)
        mean_f1 = scores.mean()
        print(f"  C={c:<6}  CV F1={mean_f1:.4f}  ±{scores.std():.4f}")
        if mean_f1 > best_f1:
            best_f1, best_c = mean_f1, c

    print(f"\n[Train] Best C={best_c}  CV F1={best_f1:.4f}")

    # ── Final fit ─────────────────────────────────────────────
    clf = SVC(kernel="linear", C=best_c, probability=True, random_state=42)
    clf.fit(X_tr_n, y_tr)
    print("[Train] SVM fitted on full training set.")

    # ── Validation report ─────────────────────────────────────
    if len(X_va_n) > 0:
        y_pred = clf.predict(X_va_n)
        print("\n── Validation set ────────────────────────────────────")
        print(classification_report(y_va, y_pred,
                                    target_names=["compliant", "violation"]))
        cm = confusion_matrix(y_va, y_pred)
        print("Confusion matrix (rows=actual, cols=predicted):")
        print(f"  {'':12s}  compliant  violation")
        print(f"  {'compliant':12s}  {cm[0,0]:9d}  {cm[0,1]:9d}")
        print(f"  {'violation':12s}  {cm[1,0]:9d}  {cm[1,1]:9d}")
        val_f1 = f1_score(y_va, y_pred)
        print(f"\nVal F1: {val_f1:.4f}")
    else:
        print("[WARN] Validation set is empty — increase dataset size.")

    # ── Save ──────────────────────────────────────────────────
    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "scaler": scaler}, str(out))
    print(f"\n[Saved] Model → {out.resolve()}")


def evaluate_model(dataset_root: str, model_path: str) -> None:
    """Run evaluation on the full validation split of the dataset."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import joblib
    except ImportError:
        print("[ERROR] pip install scikit-learn joblib")
        sys.exit(1)

    bundle = joblib.load(model_path)
    clf, scaler = bundle["clf"], bundle["scaler"]

    dm = DatasetManager(dataset_root)
    _, _, X_va, y_va = dm.load_dataset(val_split=0.2, augment=False)

    if len(X_va) == 0:
        print("[WARN] Validation set is empty.")
        return

    X_va_n = scaler.transform(X_va)
    y_pred  = clf.predict(X_va_n)

    print("\n── Evaluation ────────────────────────────────────────")
    print(classification_report(y_va, y_pred,
                                target_names=["compliant", "violation"]))
    cm = confusion_matrix(y_va, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':12s}  compliant  violation")
    print(f"  {'compliant':12s}  {cm[0,0]:9d}  {cm[0,1]:9d}")
    print(f"  {'violation':12s}  {cm[1,0]:9d}  {cm[1,1]:9d}")


# ─────────────────────────────────────────────────────────────
# SVM MODEL LOADER
# ─────────────────────────────────────────────────────────────

def load_svm_model(model_path: str):
    """
    Attempt to load a saved joblib bundle.
    Returns (clf, scaler) or (None, None) if loading fails.
    """
    try:
        import joblib
    except ImportError:
        print("[WARN] joblib not installed — SVM disabled.  pip install joblib")
        return None, None

    if not Path(model_path).exists():
        print(f"[INFO] No model file at {model_path} — using threshold detector.")
        return None, None

    try:
        bundle = joblib.load(model_path)
        clf, scaler = bundle["clf"], bundle["scaler"]
        print(f"[OK] Loaded SVM model from {model_path}")
        return clf, scaler
    except Exception as e:
        print(f"[WARN] Could not load model: {e} — using threshold detector.")
        return None, None


# ─────────────────────────────────────────────────────────────
# MAIN ENFORCER CLASS
# ─────────────────────────────────────────────────────────────

class HaircutEnforcer:

    def __init__(self, model_path: Optional[str] = None):
        self.face_cascade = self._load_cascade(
            "haarcascade_frontalface_default.xml")
        self.eye_cascade  = self._load_cascade(
            "haarcascade_eye.xml")
        self.profile_cascade = self._load_cascade(
            "haarcascade_profileface.xml")

        self.clahe = cv2.createCLAHE(
            clipLimit=CONFIG["CLAHE_CLIP"],
            tileGridSize=CONFIG["CLAHE_GRID"]
        )

        self.hog = build_hog_descriptor()

        # ── Optional SVM classifier ───────────────────────────
        self.svm_clf    = None
        self.svm_scaler = None
        if model_path:
            self.svm_clf, self.svm_scaler = load_svm_model(model_path)

        self._eyebrow_history: List[int] = []
        self._last_violation_time: float = 0.0
        self.frame_count    = 0
        self.violation_count = 0

        if CONFIG["LOG_VIOLATIONS"]:
            Path(CONFIG["LOG_DIR"]).mkdir(exist_ok=True)

    # ── cascade loader ────────────────────────────────────────

    @staticmethod
    def _load_cascade(filename: str) -> cv2.CascadeClassifier:
        path = os.path.join(cv2.data.haarcascades, filename)
        if not os.path.exists(path):
            path = filename
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            raise FileNotFoundError(
                f"Cascade not found: {filename}\n"
                f"Tried: {path}\n"
                "Run:  pip install opencv-python"
            )
        print(f"[OK] Loaded cascade: {filename}")
        return clf

    # ── smoothed eyebrow Y ────────────────────────────────────

    def _smooth_eyebrow_y(self, raw_y: int) -> int:
        self._eyebrow_history.append(raw_y)
        if len(self._eyebrow_history) > CONFIG["EYEBROW_SMOOTH_N"]:
            self._eyebrow_history.pop(0)
        return int(np.mean(self._eyebrow_history))

    # ── SVM-based hair check ──────────────────────────────────

    def _svm_classify(self, roi_gray: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (is_violation, confidence_0_to_1).
        Falls back to (False, 0.0) on any error.
        """
        if self.svm_clf is None or self.svm_scaler is None:
            return False, 0.0
        try:
            feat = extract_hog_features(roi_gray, self.hog).reshape(1, -1)
            feat_n = self.svm_scaler.transform(feat)
            proba  = self.svm_clf.predict_proba(feat_n)[0]
            # class order from SVM: [0=compliant, 1=violation]
            classes = list(self.svm_clf.classes_)
            viol_idx = classes.index(1) if 1 in classes else 1
            confidence = float(proba[viol_idx])
            is_viol    = bool(self.svm_clf.predict(feat_n)[0] == 1)
            return is_viol, confidence
        except Exception as e:
            print(f"[WARN] SVM predict failed: {e}")
            return False, 0.0

    # ── main per-frame pipeline ───────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Detection]:
        self.frame_count += 1
        det       = Detection()
        annotated = frame.copy()

        # ── 1. Preprocessing ──────────────────────────────────
        gray_eq, normalized = preprocess_frame(frame, self.clahe)

        # ── 2. Face detection ─────────────────────────────────
        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor  = CONFIG["FACE_SCALE"],
            minNeighbors = CONFIG["FACE_NEIGHBORS"],
            minSize      = CONFIG["FACE_MIN_SIZE"],
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        if not len(faces):
            return self._check_side_profile(frame, gray_eq, annotated, det)

        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        det.face_box = (fx, fy, fw, fh)
        cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), (210, 210, 210), 1)

        # ── 3. Eye detection + eyebrow Y estimation ───────────
        eye_search_h = int(fh * 0.55)
        face_roi = gray_eq[fy : fy + eye_search_h, fx : fx + fw]

        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor  = CONFIG["EYE_SCALE"],
            minNeighbors = CONFIG["EYE_NEIGHBORS"],
            minSize      = (22, 22)
        )

        if not len(eyes):
            det.status = "no_eyes"
            self._draw_status(annotated, det)
            return annotated, det

        abs_eyes = [(fx + ex, fy + ey, ew, eh) for ex, ey, ew, eh in eyes]
        det.eye_boxes = abs_eyes

        for (ex, ey, ew, eh) in abs_eyes:
            cv2.rectangle(annotated, (ex, ey), (ex+ew, ey+eh), (190, 190, 60), 1)

        raw_brow_y    = min(abs_eyes, key=lambda e: e[1])[1]
        det.eyebrow_y = self._smooth_eyebrow_y(raw_brow_y)

        self._dashed_hline(annotated, det.eyebrow_y,
                           fx, fx + fw, CONFIG["BROW_LINE_COLOR"])
        cv2.putText(annotated, "Eyebrow level",
                    (fx + 4, det.eyebrow_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    CONFIG["BROW_LINE_COLOR"], 1, cv2.LINE_AA)

        # ── 4. Forehead / hair ROI ────────────────────────────
        roi_y1 = max(0, fy - int(fh * CONFIG["FOREHEAD_TOP_FRAC"]))

        brow_buffer = 8
        roi_y2 = min(
            frame.shape[0],
            fy + int(fh * CONFIG["FOREHEAD_BOT_FRAC"]),
            det.eyebrow_y - brow_buffer if det.eyebrow_y else frame.shape[0]
        )
        roi_y2 = max(roi_y2, roi_y1 + 10)

        roi_x1, roi_x2 = fx, fx + fw
        det.forehead_roi_box = (roi_x1, roi_y1, fw, roi_y2 - roi_y1)

        cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2),
                      CONFIG["ROI_COLOR"], 1)
        cv2.putText(annotated, "Hair ROI",
                    (roi_x1 + 4, roi_y1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    CONFIG["ROI_COLOR"], 1, cv2.LINE_AA)

        # ── 5. Violation threshold line ───────────────────────
        roi_h_px = roi_y2 - roi_y1
        violation_line_y = roi_y1 + int(roi_h_px * CONFIG["HAIR_VIOLATION_FRAC"])
        det.violation_line_y = violation_line_y

        self._dashed_hline(annotated, violation_line_y,
                           roi_x1, roi_x2, CONFIG["THRESH_LINE_COLOR"])
        cv2.putText(annotated,
                    f"Violation line ({CONFIG['HAIR_VIOLATION_FRAC']})",
                    (roi_x1 + 4, violation_line_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                    CONFIG["THRESH_LINE_COLOR"], 1, cv2.LINE_AA)

        # ── 6. Extract ROI ────────────────────────────────────
        roi_gray = gray_eq[roi_y1:roi_y2, roi_x1:roi_x2]
        det.roi_gray = roi_gray.copy() if roi_gray.size else None

        if roi_gray.size == 0 or roi_gray.shape[0] < 5 or roi_gray.shape[1] < 5:
            det.status = "ok"
            self._draw_status(annotated, det)
            return annotated, det

        # ── 7. POLICY CHECK ───────────────────────────────────

        now = time.monotonic()

        if self.svm_clf is not None:
            # ── SVM path ──────────────────────────────────────
            is_viol, confidence = self._svm_classify(roi_gray)
            det.classifier_used = "svm"
            det.svm_confidence  = confidence

            if is_viol:
                det.violation = True
                det.status    = "violation"
                self._last_violation_time = now
                self.violation_count += 1
                if CONFIG["LOG_VIOLATIONS"]:
                    self._log(frame, det)
            else:
                det.status = "ok"

        else:
            # ── Threshold path (original) ─────────────────────
            hair_mask = binarize_hair(roi_gray)

            roi_h, roi_w = roi_gray.shape
            det.hair_pixel_count = int(hair_mask.sum() // 255)

            integral   = build_integral_image(roi_gray)
            total_luma = rect_sum(integral, 0, 0, roi_h, roi_w)
            if total_luma > 0:
                det.hair_density_pct = (
                    1.0 - (total_luma / (roi_h * roi_w * 255))
                ) * 100.0

            hair_rows = np.where(hair_mask.any(axis=1))[0]
            if len(hair_rows) and det.hair_pixel_count >= CONFIG["MIN_HAIR_AREA"]:
                edge_row = int(np.percentile(hair_rows,
                                             CONFIG["HAIR_ROW_PERCENTILE"]))
                det.lowest_hair_y = roi_y1 + edge_row

                cv2.line(annotated,
                         (roi_x1, det.lowest_hair_y),
                         (roi_x2, det.lowest_hair_y),
                         CONFIG["HAIR_LINE_COLOR"], 2)
                cv2.putText(annotated, "Hair edge",
                            (roi_x1 + 4, det.lowest_hair_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                            CONFIG["HAIR_LINE_COLOR"], 1, cv2.LINE_AA)

            if det.lowest_hair_y is not None:
                if det.lowest_hair_y >= violation_line_y:
                    det.violation = True
                    det.status    = "violation"
                    self._last_violation_time = now
                    self.violation_count += 1
                    if CONFIG["LOG_VIOLATIONS"]:
                        self._log(frame, det)
                else:
                    det.status = "ok"
            else:
                det.status = "ok"

        # Violation hold
        if (not det.violation and
                now - self._last_violation_time < CONFIG["VIOLATION_HOLD_S"]):
            det.violation = True
            det.status    = "violation"

        # ── 8. Annotate ───────────────────────────────────────
        self._draw_gate_ui(annotated, det)
        return annotated, det

    # ── side-profile fallback ─────────────────────────────────

    def _check_side_profile(self, frame: np.ndarray, gray_eq: np.ndarray,
                             annotated: np.ndarray, det: Detection
                             ) -> Tuple[np.ndarray, Detection]:
        """
        Fallback when the frontal cascade finds no face.

        COORDINATE MATH FIX — explained in full:
        ─────────────────────────────────────────
        The Haar profile cascade (haarcascade_profileface.xml) detects
        LEFT-facing heads.  The detected bbox (fx, fy, fw, fh) is in image
        coordinates and covers the full silhouette of the side of the head.

        Pass 1  — run on gray_eq directly → detects left-facing subjects.
                  No coordinate transform needed.

        Pass 2  — flip gray_eq horizontally (cv2.flip(., 1)) → now a
                  right-facing subject looks LEFT-facing to the cascade.
                  The cascade returns (fx_f, fy_f, fw_f, fh_f) in FLIPPED
                  image coordinates.  To convert back to original coordinates:

                      fx_orig = frame_width  - fx_f  - fw_f
                      fy_orig = fy_f                          (y is unaffected)
                      fw_orig = fw_f
                      fh_orig = fh_f

                  This maps the right edge of the flipped bbox to the left
                  edge of the same physical region in the original image.

        Sub-ROI fractions:
                  The bbox is then divided into ear/sideburn and neck/back
                  regions using fractions of fw and fh.  For Pass 2 (right-
                  facing, mirror=True) the x-fractions are flipped:
                      x1_new = 1 - x2_old,  x2_new = 1 - x1_old
                  so that the ear/neck ROIs still land on the correct physical
                  side of the head.

        Preprocessing fix:
                  The original code ran binarize_hair() (global Otsu) on the
                  side ROI, which fails because skin and background have similar
                  intensity to each other — Otsu picks a threshold that cuts
                  through the middle of the skin band rather than separating
                  dark hair from bright skin.  We now use binarize_side_hair()
                  (adaptive Gaussian threshold) on a CLAHE-enhanced crop.
        """
        H, W = gray_eq.shape

        # ── Pass 1: left-facing ───────────────────────────────
        profiles = self.profile_cascade.detectMultiScale(
            gray_eq,
            scaleFactor  = CONFIG["PROFILE_SCALE"],
            minNeighbors = CONFIG["PROFILE_NEIGHBORS"],
            minSize      = CONFIG["PROFILE_MIN_SIZE"],
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )

        facing_right = False

        if not len(profiles):
            # ── Pass 2: right-facing (run cascade on flipped image) ───────
            gray_flipped = cv2.flip(gray_eq, 1)
            profiles = self.profile_cascade.detectMultiScale(
                gray_flipped,
                scaleFactor  = CONFIG["PROFILE_SCALE"],
                minNeighbors = CONFIG["PROFILE_NEIGHBORS"],
                minSize      = CONFIG["PROFILE_MIN_SIZE"],
                flags        = cv2.CASCADE_SCALE_IMAGE,
            )
            if not len(profiles):
                det.status = "no_face"
                self._draw_gate_ui(annotated, det)
                return annotated, det

            facing_right = True
            # Mirror detected bboxes from flipped-image coords → original coords.
            # Formula: fx_orig = W - fx_flipped - fw  (symmetric reflection)
            profiles = np.array([
                [W - fx_f - fw_f, fy_f, fw_f, fh_f]
                for fx_f, fy_f, fw_f, fh_f in profiles
            ])

        # Pick the largest detected profile bbox (most likely the subject)
        fx, fy, fw, fh = max(profiles, key=lambda r: r[2] * r[3])
        fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
        det.profile_face_box = (fx, fy, fw, fh)

        # Draw the profile face bbox in a neutral grey
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (210, 210, 210), 1)
        side_label = "Right-facing" if facing_right else "Left-facing"
        cv2.putText(annotated, side_label, (fx + 4, max(fy - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_SMALL"],
                    (210, 210, 210), 1, cv2.LINE_AA)

        # Temporarily show "checking" status while ROIs are analysed
        det.status = "side_profile"

        def _roi_box(xfrac: Tuple[float, float],
                     yfrac: Tuple[float, float],
                     mirror: bool) -> Tuple[int, int, int, int]:
            """
            Convert bbox-relative fractions to absolute pixel coordinates.

            When mirror=True (right-facing subject) the x-fractions are
            reflected so the ear/neck land on the correct physical side.

              mirror=False (left-facing):  ear is at bbox-RIGHT (high x-frac)
              mirror=True  (right-facing): ear is at bbox-LEFT  (low x-frac)
                → xfrac becomes (1-x2, 1-x1)
            """
            if mirror:
                xfrac = (1.0 - xfrac[1], 1.0 - xfrac[0])

            # Clamp to frame boundaries
            x1 = max(0, fx + int(fw * xfrac[0]))
            x2 = min(W, fx + int(fw * xfrac[1]))
            y1 = max(0, fy + int(fh * yfrac[0]))
            y2 = min(H, fy + int(fh * yfrac[1]))

            roi_w = max(0, x2 - x1)
            roi_h = max(0, y2 - y1)
            return x1, y1, roi_w, roi_h

        ear_box  = _roi_box(CONFIG["PROFILE_EAR_X"],  CONFIG["PROFILE_EAR_Y"],  facing_right)
        neck_box = _roi_box(CONFIG["PROFILE_NECK_X"], CONFIG["PROFILE_NECK_Y"], facing_right)
        det.ear_roi_box, det.neck_roi_box = ear_box, neck_box

        def _dark_pixel_count(box: Tuple[int, int, int, int]) -> int:
            """
            Count dark (hair) pixels in *box* using the side-profile
            adaptive binarizer.

            Preprocessing chain applied to the ROI crop:
              1. Extract raw crop from gray_eq (already CLAHE-enhanced globally)
              2. Apply a local CLAHE pass to boost intra-ROI contrast
                 (ear ROIs can be dim compared to the face centre)
              3. Run adaptive threshold (binarize_side_hair)
              4. Use integral image for an O(1) pixel count

            BUG FIX (original): The old code called
              ``build_integral_image(mask // 255)``
            which produces an integral of 0/1 values, and then
              ``rect_sum(integral, 0, 0, h, w)``
            to get the total.  That is mathematically correct, BUT the
            variables h and w inside _dark_count were taken from the
            *box tuple* (not the crop), so if the crop was clipped at the
            frame boundary the shape could differ → index-out-of-bounds risk.
            Fixed: use the crop's actual .shape.
            """
            x, y, bw, bh = box
            if bw < 8 or bh < 8:
                return 0

            crop = gray_eq[y:y + bh, x:x + bw]
            if crop.size == 0:
                return 0

            # Local contrast boost — keep clipLimit LOW (1.5 not 2.5).
            # A high clipLimit over-amplifies skin texture, which then
            # triggers Gate 1 of binarize_side_hair even when no hair is present.
            local_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            crop_eq = local_clahe.apply(crop)

            mask = binarize_side_hair(crop_eq)

            # Use actual crop shape (may differ from bw/bh at frame edge)
            ch, cw = mask.shape
            integral = build_integral_image(mask // 255)
            return int(rect_sum(integral, 0, 0, ch, cw))

        det.ear_pixel_count  = _dark_pixel_count(ear_box)
        det.neck_pixel_count = _dark_pixel_count(neck_box)

        # Draw the sub-ROI boxes using the unified helper
        self._draw_profile_rois(annotated, det)

        # ── Violation decision ────────────────────────────────
        min_area = CONFIG["MIN_HAIR_AREA"]
        is_viol  = (det.ear_pixel_count >= min_area or
                    det.neck_pixel_count >= min_area)

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

        # Violation hold (keep VIOLATION banner for VIOLATION_HOLD_S seconds)
        if (not det.violation and
                now - self._last_violation_time < CONFIG["VIOLATION_HOLD_S"]):
            det.violation = True
            det.status    = "violation"

        self._draw_gate_ui(annotated, det)
        return annotated, det

    def _draw_profile_rois(self, img: np.ndarray, det: Detection):
        """Draw dashed rectangles around the ear/sideburn and neck/back ROIs."""
        for box, color, label in (
            (det.ear_roi_box,  CONFIG["PROFILE_EAR_COLOR"],  "Ear/Sideburn"),
            (det.neck_roi_box, CONFIG["PROFILE_NECK_COLOR"], "Neck/Back"),
        ):
            if not box:
                continue
            x, y, w, h = box
            self._dashed_hline(img, y,     x, x + w, color)
            self._dashed_hline(img, y + h, x, x + w, color)
            cv2.line(img, (x,     y), (x,     y + h), color, 1)
            cv2.line(img, (x + w, y), (x + w, y + h), color, 1)
            cv2.putText(img, label, (x + 4, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    # ── drawing helpers ───────────────────────────────────────

    @staticmethod
    def _dashed_hline(img: np.ndarray, y: int,
                      x1: int, x2: int, color: Tuple, dash: int = 8):
        for xd in range(x1, x2, dash * 2):
            cv2.line(img, (xd, y), (min(xd + dash, x2), y), color, 1)

    # ── UNIFIED GATE-MONITOR UI ───────────────────────────────
    # Both _draw_status and _draw_debug have been merged into a single
    # _draw_gate_ui() method.  This guarantees every camera mode
    # (live, collect, batch preview) renders the exact same layout:
    #
    #   ┌─────────────────────────────────────────────────────┐
    #   │  HAIRCUT POLICY GATE MONITOR   │  status text       │  ← header H px
    #   ├─────────────────────────────────────────────────────┤
    #   │                                                     │
    #   │              live camera feed                       │
    #   │          (ROI boxes / lines drawn here)             │
    #   │                                                     │
    #   ├─────────────────────────────────────────────────────┤
    #   │  debug telemetry line 1  │  debug telemetry line 2  │  ← footer H px
    #   │  [<<] [>>] [-] [+]  buttons live here               │
    #   └─────────────────────────────────────────────────────┘
    #
    # If a violation is active, a coloured border is drawn around the full frame.

    def _draw_gate_ui(self, img: np.ndarray, det: Detection,
                      extra_lines: Optional[List[str]] = None) -> None:
        """
        Draw the unified Gate-Monitor UI on *img* in-place.

        Parameters
        ----------
        img         : annotated BGR frame (modified in-place)
        det         : Detection result for the current frame
        extra_lines : optional list of extra text lines to append in the footer
                      (used by collect mode to show sample counts, etc.)
        """
        h, w = img.shape[:2]
        header_h = CONFIG["UI_HEADER_H"]
        footer_h = CONFIG["UI_FOOTER_H"]
        alpha    = CONFIG["UI_PANEL_ALPHA"]

        status_map = {
            "ok":           ("COMPLIANT",                     CONFIG["OK_COLOR"]),
            "violation":    ("POLICY VIOLATION - SEE GUARD",  CONFIG["VIOLATION_COLOR"]),
            "no_face":      ("STEP INTO FRAME",               (180, 180, 180)),
            "no_eyes":      ("FACE CAMERA DIRECTLY",          CONFIG["WARN_COLOR"]),
            "side_profile": ("SIDE PROFILE - checking...",    CONFIG["WARN_COLOR"]),
        }
        status_text, status_color = status_map.get(
            det.status, ("...", (150, 150, 150))
        )

        # ── Header panel ─────────────────────────────────────
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        # Left: system title
        cv2.putText(img, CONFIG["UI_TITLE"],
                    (10, header_h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_SMALL"],
                    (160, 160, 160), 1, cv2.LINE_AA)

        # Right of title: status
        cv2.putText(img, status_text,
                    (10, header_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_LARGE"],
                    status_color, 2, cv2.LINE_AA)

        # ── Footer panel ─────────────────────────────────────
        footer_y = h - footer_h
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, footer_y), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay2, alpha, img, 1.0 - alpha, 0, img)

        # Build debug telemetry lines
        clf_label = (
            f"SVM conf={det.svm_confidence:.2f}"
            if det.classifier_used == "svm" else "Threshold"
        )
        debug_lines: List[str] = [
            f"Frame {self.frame_count}  |  Violations: {self.violation_count}"
            f"  |  {clf_label}"
        ]
        if det.eyebrow_y:
            debug_lines.append(
                f"BrowY={det.eyebrow_y}px"
                + (f"  HairEdgeY={det.lowest_hair_y}px"
                   f"  ViolLineY={det.violation_line_y}px"
                   if det.lowest_hair_y else "")
            )
        if det.hair_density_pct:
            debug_lines.append(
                f"Density={det.hair_density_pct:.1f}%"
                f"  Thresh={CONFIG['HAIR_THRESH']}"
                f"  ViolFrac={CONFIG['HAIR_VIOLATION_FRAC']}"
                f"  Pct={CONFIG['HAIR_ROW_PERCENTILE']}"
            )
        if det.profile_face_box:
            debug_lines.append(
                f"Profile  ear={det.ear_pixel_count}px"
                f"  neck={det.neck_pixel_count}px"
                f"  min={CONFIG['MIN_HAIR_AREA']}"
            )
        if extra_lines:
            debug_lines.extend(extra_lines)

        # Render each line inside footer — lines stack upward from the bottom
        line_h = 14   # px per debug line
        # Reserve first 40 px from footer_y for clickable buttons
        text_start_y = footer_y + 12
        for i, ln in enumerate(debug_lines):
            y_pos = text_start_y + i * line_h
            if y_pos > h - 4:
                break
            cv2.putText(img, ln, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_SMALL"],
                        (190, 190, 190), 1, cv2.LINE_AA)

        # ── Violation frame border ────────────────────────────
        if det.violation:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1),
                          CONFIG["VIOLATION_COLOR"], 4)

    # ── Legacy wrappers kept so nothing breaks if called directly ─
    def _draw_status(self, img: np.ndarray, det: Detection):
        """Deprecated — calls unified _draw_gate_ui internally."""
        self._draw_gate_ui(img, det)

    def _draw_debug(self, img: np.ndarray, det: Detection):
        """Deprecated — merged into _draw_gate_ui."""
        pass  # already drawn by _draw_gate_ui via _draw_status

    def _log(self, frame: np.ndarray, det: Detection):
        ts    = time.strftime("%Y%m%d_%H%M%S")
        fname = Path(CONFIG["LOG_DIR"]) / \
                f"violation_{ts}_{self.violation_count:04d}.jpg"
        cv2.imwrite(str(fname), frame)


# ─────────────────────────────────────────────────────────────
# SHARED CAMERA LOOP
# ─────────────────────────────────────────────────────────────

def _open_camera() -> cv2.VideoCapture:
    """Open and configure the camera from CONFIG. Exits on failure."""
    src = CONFIG["CAMERA_INDEX"]
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    cap.set(cv2.CAP_PROP_FPS,          CONFIG["TARGET_FPS"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {src}")
        sys.exit(1)
    return cap


def _camera_loop(enforcer:     "HaircutEnforcer",
                 window_title:  str,
                 overlay_fn,           # (annotated, det) -> None  — draw extra overlays
                 key_fn,               # (key, det, annotated) -> bool  — return True to quit
                 show_mask_ref: list,  # [bool] — mutable flag so key_fn can toggle it
                 buttons: Optional[List[Button]] = None,  # list of clickable buttons
                 mouse_callback = None  # optional mouse event handler
                 ) -> None:
    """
    Single shared camera loop used by both live enforcement and collect mode.

    Reads frames, runs enforcer.analyze_frame, applies the optional binary
    mask overlay, calls overlay_fn for mode-specific drawing, shows the
    window, and dispatches keys to key_fn(key, det, annotated).  Stops when key_fn returns True.
    
    If buttons are provided, they will be drawn and their click events handled.
    """
    cap = _open_camera()
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    
    # Setup mouse callback if provided
    if mouse_callback or buttons:
        def _mouse_event(event, x, y, flags, userdata):
            if buttons:
                # Check for button clicks
                if event == cv2.EVENT_LBUTTONDOWN:
                    for btn in buttons:
                        if btn.contains_point(x, y):
                            btn.on_click()
                # Update hover state
                for btn in buttons:
                    btn.hovered = btn.contains_point(x, y)
            
            # Call custom mouse callback if provided
            if mouse_callback:
                mouse_callback(event, x, y, flags, userdata)
        
        cv2.setMouseCallback(window_title, _mouse_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame drop — retrying…")
            time.sleep(0.05)
            continue

        annotated, det = enforcer.analyze_frame(frame)

        # Optional binary hair-mask overlay (toggled via T key in live mode)
        if show_mask_ref[0] and det.forehead_roi_box:
            rx, ry, rw, rh = det.forehead_roi_box
            gray_eq, _ = preprocess_frame(frame, enforcer.clahe)
            roi_g  = gray_eq[ry:ry+rh, rx:rx+rw]
            mask   = binarize_hair(roi_g)
            mask3  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask3[mask > 0] = [0, 0, 180]
            blend  = cv2.addWeighted(
                annotated[ry:ry+rh, rx:rx+rw], 0.5, mask3, 0.5, 0)
            annotated[ry:ry+rh, rx:rx+rw] = blend

        # Mode-specific overlays (collect stats, extra highlights, etc.)
        overlay_fn(annotated, det)
        
        # Draw buttons
        if buttons:
            for btn in buttons:
                btn.draw(annotated)

        cv2.imshow(window_title, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key_fn(key, det, annotated):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# TRAINING DATA PREPROCESSOR (24×24 samples for Haar training)
# ─────────────────────────────────────────────────────────────

class TrainingPipeline:
    """Legacy Haar training data preprocessor (kept for compatibility)."""
    TRAIN_SIZE = (24, 24)
    _clahe     = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized   = cv2.resize(gray, self.TRAIN_SIZE, interpolation=cv2.INTER_AREA)
        equalized = self._clahe.apply(resized)
        smoothed  = cv2.GaussianBlur(equalized, CONFIG["GAUSS_KERNEL"], 0)
        return (smoothed.astype(np.float32) / 255.0)

    def process_dataset(self, dataset_root: str, out_dir: str):
        src, dst = Path(dataset_root), Path(out_dir)
        for split in ("positive", "negative"):
            out_split = dst / split
            out_split.mkdir(parents=True, exist_ok=True)
            paths = []
            for img_path in sorted((src / split).glob("*.jpg")) + \
                            sorted((src / split).glob("*.png")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                sample = self._preprocess(img)
                out_p  = out_split / f"{split}_{len(paths):05d}.png"
                cv2.imwrite(str(out_p), (sample * 255).astype(np.uint8))
                paths.append(str(out_p))
            (dst / f"{split}.txt").write_text("\n".join(paths))
            print(f"[{split}] {len(paths)} samples written.")


# ─────────────────────────────────────────────────────────────
# BATCH / OFFLINE MODE
# ─────────────────────────────────────────────────────────────

def run_batch(image_paths: List[str],
              save_dir: str = "batch_results",
              model_path: Optional[str] = None):
    enforcer = HaircutEnforcer(model_path=model_path)
    out = Path(save_dir)
    out.mkdir(exist_ok=True)
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read {path}")
            continue
        annotated, det = enforcer.analyze_frame(img)
        fname = Path(path).stem + f"_{det.status}.jpg"
        cv2.imwrite(str(out / fname), annotated)
        print(f"[{'VIOLATION' if det.violation else 'ok':9s}] {path}")
    print(f"\nSaved to: {save_dir}/")


# ─────────────────────────────────────────────────────────────
# LIVE ENFORCEMENT MODE
# ─────────────────────────────────────────────────────────────

def run_live(model_path: Optional[str] = None):
    """
    Real-time gate enforcement.  Uses the shared _camera_loop.

    Controls:
        Q        — quit
        S        — save snapshot
        T        — toggle hair binary mask overlay
        + / =    — raise HAIR_THRESH by 5
        -        — lower HAIR_THRESH by 5
        [ / ]    — decrease / increase HAIR_VIOLATION_FRAC by 0.05
        , / .    — decrease / increase HAIR_ROW_PERCENTILE by 5
        
    Buttons (clickable on screen):
        [<<] / [>>]  — adjust violation line (stricter / more lenient)
        [-]  / [+]   — adjust hair threshold
    """
    enforcer   = HaircutEnforcer(model_path=model_path)
    show_mask  = [False]   # mutable so the key callback can toggle it

    model_label = f"SVM: {model_path}" if model_path else "Threshold (no model)"
    print("=" * 60)
    print("  Haircut Policy Enforcement System  |  LIVE MODE")
    print(f"  Classifier: {model_label}")
    print("  Q=quit  S=snapshot  T=mask  +/-=threshold")
    print("  [/]=violation fraction  ,/.=percentile")
    print("  Click buttons: [<<] [>>] [-] [+]")
    print("=" * 60)

    # Create buttons for violation line and threshold adjustment
    frame_w = CONFIG["FRAME_WIDTH"]
    frame_h = CONFIG["FRAME_HEIGHT"]
    
    buttons = _create_param_buttons(frame_w, frame_h)

    def overlay_fn(annotated, det):
        pass

    def key_fn(key, det, annotated):
        if key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"snapshot_{ts}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[INFO] Snapshot: {fname}")
        elif key == ord('t'):
            show_mask[0] = not show_mask[0]
            print(f"[INFO] Mask overlay: {'ON' if show_mask[0] else 'OFF'}")
        _handle_param_keys(key)
        return False

    _camera_loop(enforcer,
                 "Haircut Policy — Gate Monitor",
                 overlay_fn,
                 key_fn,
                 show_mask,
                 buttons=buttons)

    print(f"\nSession done. Total violations logged: {enforcer.violation_count}")


# ─────────────────────────────────────────────────────────────
# COLLECT MODE — live labelling of training samples
# ─────────────────────────────────────────────────────────────

def run_collect(dataset_root: str = "dataset"):
    """
    Live data collection for training.  Uses the exact same camera loop,
    enforcer, and preprocessing as live enforcement — no duplicated code.

    Press P to save a POSITIVE sample (hair in ROI = violation),
    N for NEGATIVE (compliant), Q to quit.
    
    Keyboard controls:
        P      — save POSITIVE sample (hair in ROI)
        N      — save NEGATIVE sample (compliant / no hair)
        Q      — quit collection
        +/-    — hair threshold  [/] — violation line  ,/. — hair row percentile

    Buttons (clickable on screen):
        [<<] / [>>]  — adjust violation line
        [-]  / [+]   — adjust hair threshold

    Same tuning keys as live: +/- threshold, [/] violation line, ,/. percentile.
    """
    enforcer  = HaircutEnforcer()   # no SVM during collection
    dm        = DatasetManager(dataset_root)
    show_mask = [False]             # collect mode doesn't need mask, but loop expects it

    counts = {
        "positive": len(list(dm.pos_dir.glob("*.png"))),
        "negative": len(list(dm.neg_dir.glob("*.png"))),
    }

    print("=" * 60)
    print("  COLLECT MODE — Training Data Collection")
    print(f"  Dataset: {dm.root.resolve()}")
    print(f"  Existing — positives: {counts['positive']}  negatives: {counts['negative']}")
    print("  P = save POSITIVE (hair in ROI = violation)")
    print("  N = save NEGATIVE (compliant / no hair in ROI)")
    print("  +/- = threshold  [/] = violation line  ,/. = percentile")
    print("  Q = quit")
    print("=" * 60)

    # Create buttons for parameter adjustment
    frame_w = CONFIG["FRAME_WIDTH"]
    frame_h = CONFIG["FRAME_HEIGHT"]
    
    buttons = _create_param_buttons(frame_w, frame_h)

    def overlay_fn(annotated, det):
        """Draw collection-specific overlays below the unified header panel."""
        # Render sample counts in the footer area — uses UI_FONT_MED so it
        # matches the system style and does NOT overlap the header (y=48+).
        header_h = CONFIG["UI_HEADER_H"]
        stats = (f"Saved  pos:{counts['positive']}  neg:{counts['negative']}"
                 f"   target: 200+ each")
        cv2.putText(annotated, stats,
                    (8, header_h + 18),          # just below the header bar
                    cv2.FONT_HERSHEY_SIMPLEX, CONFIG["UI_FONT_MED"],
                    (200, 200, 50), 1, cv2.LINE_AA)
        # Brighter ROI highlight so the labeller can see exactly what is saved
        if det.forehead_roi_box:
            rx, ry, rw, rh = det.forehead_roi_box
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh),
                          (0, 255, 255), 2)

    def key_fn(key, det, annotated):
        """Handle P / N labelling keys and parameter adjustment.  Returns False (never quits on its own)."""
        label = None
        if key in (ord('p'), ord('P')):
            label = "positive"
        elif key in (ord('n'), ord('N')):
            label = "negative"
        _handle_param_keys(key)

        if label is not None:
            roi = det.roi_gray
            if roi is not None and roi.size > 0:
                saved = dm.save_sample(roi, label)
                counts[label] += 1
                symbol = "+" if label == "positive" else "-"
                print(f"[{symbol}] {label.upper()} saved "
                      f"({counts[label]}): {saved.name}")
            else:
                print("[WARN] No face detected — cannot save sample.")

        return False

    _camera_loop(enforcer,
                 "Collect Training Data  —  P=positive  N=negative  Q=quit",
                 overlay_fn,
                 key_fn,
                 show_mask,
                 buttons=buttons)

    print(f"\nCollection done.  "
          f"Positives: {counts['positive']}  Negatives: {counts['negative']}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Haircut Policy Enforcement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  live        Real-time enforcement via webcam (default)
  collect     Interactively label & save training samples from webcam
  train_svm   Train HOG+SVM classifier on collected dataset
  eval        Evaluate a saved model on the validation split
  batch       Process a list of image files offline
  train       Legacy: generate Haar training preprocessed samples

Examples:
  # Step 1 — collect 200+ samples per class
  python haircut_enforcer.py --mode collect --dataset my_dataset

  # Step 2 — train
  python haircut_enforcer.py --mode train_svm \\
      --dataset my_dataset --model models/hair_svm.joblib

  # Step 3 — evaluate
  python haircut_enforcer.py --mode eval \\
      --dataset my_dataset --model models/hair_svm.joblib

  # Step 4 — live enforcement with learned model
  python haircut_enforcer.py --mode live --model models/hair_svm.joblib

  # Batch mode
  python haircut_enforcer.py --mode batch --model models/hair_svm.joblib \\
      --images photo1.jpg photo2.jpg
        """
    )
    p.add_argument("--mode",
                   choices=["live", "collect", "train_svm", "eval",
                             "batch", "train"],
                   default="live")
    p.add_argument("--images",   nargs="*",
                   help="Image paths (batch mode)")
    p.add_argument("--dataset",  default="dataset",
                   help="Dataset root directory  (default: dataset/)")
    p.add_argument("--model",    default=None,
                   help="Path to saved SVM model (.joblib)")
    p.add_argument("--out",      default="processed_dataset",
                   help="Output dir for legacy Haar training preprocessor")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation (default: 0.2)")
    args = p.parse_args()

    if args.mode == "live":
        run_live(model_path=args.model)

    elif args.mode == "collect":
        run_collect(dataset_root=args.dataset)

    elif args.mode == "train_svm":
        model_out = args.model or "models/hair_svm.joblib"
        train_svm(dataset_root=args.dataset,
                  model_path=model_out,
                  val_split=args.val_split)

    elif args.mode == "eval":
        if not args.model:
            print("[ERROR] --model is required for eval mode.")
            sys.exit(1)
        evaluate_model(dataset_root=args.dataset, model_path=args.model)

    elif args.mode == "batch":
        if not args.images:
            print("[ERROR] Provide --images for batch mode")
            sys.exit(1)
        run_batch(args.images, model_path=args.model)

    elif args.mode == "train":
        TrainingPipeline().process_dataset(args.dataset, args.out)