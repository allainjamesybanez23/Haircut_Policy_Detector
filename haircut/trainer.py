# haircut/trainer.py
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from haircut.config import CONFIG
from haircut.preprocessing import build_hog_descriptor, extract_hog_features


class DatasetManager:
    def __init__(self, root: str):
        self.root       = Path(root)
        self.male_dir   = self.root / "male"
        self.female_dir = self.root / "female"
        self.male_dir.mkdir(parents=True, exist_ok=True)
        self.female_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Dataset] Root: {self.root.resolve()}")

    def save_sample(self, roi_gray: np.ndarray, label: str) -> Path:
        assert label in ("male", "female")
        target_dir = self.male_dir if label == "male" else self.female_dir
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = target_dir / f"{label}_{ts}.png"
        cv2.imwrite(str(out),
                    cv2.resize(roi_gray, CONFIG["HOG_WIN_SIZE"], interpolation=cv2.INTER_AREA))
        return out

    def load_dataset(self, val_split: float = 0.2,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import random as _random
        rng = _random.Random(42)
        hog = build_hog_descriptor()
        X_tr, y_tr, X_va, y_va = [], [], [], []
        for label_str, label_int in [("male", 0), ("female", 1)]:
            src_dir = self.male_dir if label_str == "male" else self.female_dir
            paths   = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
            if not paths:
                print(f"[WARN] No images in {src_dir}"); continue
            paths = list(paths)
            rng.shuffle(paths)
            split_idx   = max(1, int(len(paths) * (1.0 - val_split)))
            train_paths = paths[:split_idx]
            val_paths   = paths[split_idx:]
            print(f"[{label_str}] train={len(train_paths)}  val={len(val_paths)}")
            noise_rng = np.random.default_rng(42)
            for p in train_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                noise = noise_rng.normal(0, 12, img.shape).astype(np.int16)
                for v in [
                    img,
                    cv2.flip(img, 1),
                    np.clip(img.astype(np.int16) + 40, 0, 255).astype(np.uint8),
                    np.clip(img.astype(np.int16) - 40, 0, 255).astype(np.uint8),
                    np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8),
                ]:
                    X_tr.append(extract_hog_features(v, hog))
                    y_tr.append(label_int)
            for p in val_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                X_va.append(extract_hog_features(img, hog))
                y_va.append(label_int)
        X_tr = np.array(X_tr, dtype=np.float32)
        y_tr = np.array(y_tr, dtype=np.int32)
        feat_dim = X_tr.shape[1] if len(X_tr) > 0 else 0
        X_va = (np.array(X_va, dtype=np.float32) if X_va
                else np.empty((0, feat_dim), dtype=np.float32))
        y_va = np.array(y_va, dtype=np.int32)
        print(f"[Dataset] Train: {len(X_tr)}  Val: {len(X_va)}")
        return X_tr, y_tr, X_va, y_va


def train_svm(dataset_root: str, model_path: str = "models/hair_svm.joblib",
              val_split: float = 0.2) -> None:
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        import joblib
    except ImportError:
        print("[ERROR] pip install scikit-learn joblib"); sys.exit(1)

    dm = DatasetManager(dataset_root)
    X_tr, y_tr, X_va, y_va = dm.load_dataset(val_split=val_split)
    if len(X_tr) == 0:
        print("[ERROR] Training set is empty. Run --mode collect first."); sys.exit(1)

    scaler  = StandardScaler()
    X_tr_n  = scaler.fit_transform(X_tr)
    X_va_n  = scaler.transform(X_va) if len(X_va) > 0 else X_va

    print("\n[Train] Cross-validating C parameter …")
    best_c, best_f1 = 0.1, 0.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for c in [0.01, 0.1, 1.0, 10.0]:
        scores = cross_val_score(
            SVC(kernel="linear", C=c, probability=True, random_state=42),
            X_tr_n, y_tr, cv=cv, scoring="f1", n_jobs=-1)
        print(f"  C={c:<6}  CV F1={scores.mean():.4f}  ±{scores.std():.4f}")
        if scores.mean() > best_f1:
            best_f1, best_c = scores.mean(), c
    print(f"\n[Train] Best C={best_c}  CV F1={best_f1:.4f}")

    clf = SVC(kernel="linear", C=best_c, probability=True, random_state=42)
    clf.fit(X_tr_n, y_tr)

    if len(X_va_n) > 0:
        y_pred = clf.predict(X_va_n)
        print("\n── Validation ─────────────────────────────────────────")
        print(classification_report(y_va, y_pred, target_names=["male", "female"]))
        cm = confusion_matrix(y_va, y_pred)
        print(f"Confusion matrix:\n"
              f"  {'':10s}  male  female\n"
              f"  {'male':10s}  {cm[0,0]:4d}  {cm[0,1]:6d}\n"
              f"  {'female':10s}  {cm[1,0]:4d}  {cm[1,1]:6d}\n"
              f"  Val F1: {f1_score(y_va, y_pred):.4f}")
    else:
        print("[WARN] Validation set is empty.")

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "scaler": scaler}, str(out))
    print(f"\n[Saved] Gender SVM → {out.resolve()}")