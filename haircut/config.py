# haircut/config.py
# Single source of truth for all tuneable values.
# Every other module imports CONFIG from here.
# Runtime mutations (e.g. _handle_param_keys) propagate automatically
# because all imports share the same dict object.

CONFIG: dict = {
    # ── Camera ────────────────────────────────────────────────
    "CAMERA_INDEX":  0,
    "FRAME_WIDTH":   1280,
    "FRAME_HEIGHT":  720,
    "TARGET_FPS":    30,

    # ── Preprocessing ─────────────────────────────────────────
    "GAUSS_KERNEL":  (5, 5),
    "CLAHE_CLIP":    5.8,
    "CLAHE_GRID":    (8, 8),
    "MORPH_KERNEL":  (3, 3),

    # ── Frontal hair thresholds ───────────────────────────────
    "HAIR_THRESH":          80,
    "SIDE_HAIR_ABS_THRESH": 90,
    "MIN_HAIR_AREA":        300,
    "HAIR_ROW_PERCENTILE":  90,
    "HAIR_VIOLATION_FRAC":  0.75,

    # ── Frontal face cascade ──────────────────────────────────
    "FACE_SCALE":      1.15,
    "FACE_NEIGHBORS":  6,
    "FACE_MIN_SIZE":   (90, 90),

    # ── Eye cascade ───────────────────────────────────────────
    "EYE_SCALE":      1.08,
    "EYE_NEIGHBORS":  9,

    # ── Forehead ROI fractions ────────────────────────────────
    "FOREHEAD_TOP_FRAC":   0.05,
    "FOREHEAD_BOT_FRAC":   0.25,
    "FOREHEAD_LEFT_FRAC":  0.3,
    "FOREHEAD_RIGHT_FRAC": 0.3,

    # ── Vertical (tall/spiky) hair detection ──────────────────
    # ROI extends this many face-heights above the face-box top (fy).
    # e.g. 1.5 → searches up to 1.5 × fh pixels above fy.
    "UPPER_HAIR_SEARCH_FRAC": 0.8,
    # Hair whose topmost pixel is higher than (fy - MAX_HAIR_HEIGHT_FRAC × fh)
    # triggers a vertical violation.
    # e.g. 0.6 → hair taller than 60 % of face height above the cascade box is illegal.
    "MAX_HAIR_HEIGHT_FRAC":   0.60,
    # Minimum segmented hair pixels in the upper ROI to treat as real hair.
    "UPPER_MIN_HAIR_AREA":    150,

    # ── Eyebrow smoothing ─────────────────────────────────────
    "EYEBROW_SMOOTH_N": 8,

    # ── HOG / Gender SVM ──────────────────────────────────────
    "HOG_WIN_SIZE":   (64, 64),
    "HOG_BLOCK":      (16, 16),
    "HOG_BLOCK_STEP": (8, 8),
    "HOG_CELL":       (8, 8),
    "HOG_BINS":       9,
    "GENDER_CONF_THRESHOLD": 0.70,

    # ── Side profile cascade ──────────────────────────────────
    "PROFILE_SCALE":     1.15,
    "PROFILE_NEIGHBORS": 5,
    "PROFILE_MIN_SIZE":  (90, 90),

    # ── Profile ROI fractions ─────────────────────────────────
    "PROFILE_EAR_X":  (0.62, 0.97),
    "PROFILE_EAR_Y":  (0.22, 0.58),
    "PROFILE_NECK_X": (0.78, 1.22),
    "PROFILE_NECK_Y": (0.60, 0.90),

    # ── Profile violation thresholds ──────────────────────────
    "PROFILE_DARK_THRESH":  40,
    "PROFILE_DENSITY_MAX":  0.25,
    "PROFILE_MIN_CLUSTER":  80,

    # ── Anti-flicker ──────────────────────────────────────────
    "VIOLATION_HOLD_S": 1.0,

    # ── Violation logging ─────────────────────────────────────
    "LOG_VIOLATIONS": False,
    "LOG_DIR":        "violation_logs",

    # ── Annotation colours (BGR) ──────────────────────────────
    "VIOLATION_COLOR":    (0,   0, 220),
    "OK_COLOR":           (0, 200,  60),
    "WARN_COLOR":         (0, 165, 255),
    "ROI_COLOR":          (255, 220,  80),
    "HAIR_LINE_COLOR":    (80,   80, 255),
    "BROW_LINE_COLOR":    (255, 180,   0),
    "THRESH_LINE_COLOR":  (0,  220,   0),
    "PROFILE_EAR_COLOR":  (60, 200, 255),
    "PROFILE_NECK_COLOR": (255, 120, 200),
    # Upper / vertical hair annotations
    "UPPER_ROI_COLOR":    (200,  80, 255),   # purple box around upper search zone
    "VERT_LIMIT_COLOR":   (0,   60, 255),    # red dashed line = max allowed height

    # ── OpenCV UI (headless mode) ─────────────────────────────
    "UI_HEADER_H":    48,
    "UI_PANEL_ALPHA": 0.72,
    "UI_FONT_LARGE":  0.70,
    "UI_FONT_MED":    0.48,
    "UI_FONT_SMALL":  0.36,
    "UI_TITLE":       "HAIRCUT POLICY GATE MONITOR",
}