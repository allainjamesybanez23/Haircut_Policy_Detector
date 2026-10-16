# haircut/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    face_box:          Optional[Tuple[int, int, int, int]] = None
    face_crop_raw:     Optional[np.ndarray]                = None
    eye_boxes:         List[Tuple]                         = field(default_factory=list)
    eyebrow_y:         Optional[int]                       = None
    forehead_roi_box:  Optional[Tuple[int, int, int, int]] = None
    roi_gray:          Optional[np.ndarray]                = None
    roi_gray_raw:      Optional[np.ndarray]                = None
    lowest_hair_y:     Optional[int]                       = None
    violation_line_y:  Optional[int]                       = None
    hair_pixel_count:  int                                 = 0
    hair_density_pct:  float                               = 0.0
    # ── Vertical (tall/spiky) hair fields ─────────────────────
    upper_hair_roi_box: Optional[Tuple[int, int, int, int]] = None
    upper_hair_mask:    Optional[np.ndarray]                = None
    highest_hair_y:     Optional[int]                       = None
    vertical_limit_y:   Optional[int]                       = None
    vertical_violation: bool                                = False
    # ─────────────────────────────────────────────────────────
    violation:         bool                                = False
    status:            str                                 = "no_face"
    classifier_used:   str                                 = "threshold"
    gender:            str                                 = "unknown"
    gender_confidence: float                               = 0.0
    profile_face_box:  Optional[Tuple[int, int, int, int]] = None
    ear_roi_box:       Optional[Tuple[int, int, int, int]] = None
    neck_roi_box:      Optional[Tuple[int, int, int, int]] = None
    ear_density:       float                               = 0.0
    neck_density:      float                               = 0.0
    ear_violation:     bool                                = False
    neck_violation:    bool                                = False
    ear_mask:          Optional[np.ndarray]                = None
    neck_mask:         Optional[np.ndarray]                = None