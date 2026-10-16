# haircut/preprocessing.py
from __future__ import annotations

import cv2
import numpy as np

from haircut.config import CONFIG


def build_hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        CONFIG["HOG_WIN_SIZE"], CONFIG["HOG_BLOCK"],
        CONFIG["HOG_BLOCK_STEP"], CONFIG["HOG_CELL"], CONFIG["HOG_BINS"],
    )


def extract_hog_features(roi_gray: np.ndarray, hog: cv2.HOGDescriptor) -> np.ndarray:
    win_w, win_h = CONFIG["HOG_WIN_SIZE"]
    resized = cv2.resize(roi_gray, (win_w, win_h), interpolation=cv2.INTER_AREA)
    clahe   = cv2.createCLAHE(clipLimit=CONFIG["CLAHE_CLIP"], tileGridSize=CONFIG["CLAHE_GRID"])
    eq      = cv2.GaussianBlur(clahe.apply(resized), CONFIG["GAUSS_KERNEL"], 0)
    return hog.compute(eq).flatten().astype(np.float32)  # type: ignore

def preprocess_frame(frame: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    return cv2.GaussianBlur(
        clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
        CONFIG["GAUSS_KERNEL"], 0,
    )


def binarize_hair(roi_gray: np.ndarray) -> np.ndarray:
    manual = CONFIG["HAIR_THRESH"]
    otsu_val, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = int(otsu_val) if abs(int(otsu_val) - manual) < 30 else manual
    _, binary = cv2.threshold(roi_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG["MORPH_KERNEL"])
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def binarize_side_hair(roi_gray: np.ndarray) -> np.ndarray:
    blurred  = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 22,
    )
    _, absolute = cv2.threshold(
        blurred, CONFIG["SIDE_HAIR_ABS_THRESH"], 255, cv2.THRESH_BINARY_INV,
    )
    binary = cv2.morphologyEx(
        cv2.bitwise_and(adaptive, absolute),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 40:
            filtered[labels == lbl] = 255
    return filtered