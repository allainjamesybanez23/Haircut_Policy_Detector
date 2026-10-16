# haircut/cli.py
from __future__ import annotations

import time

import cv2

from haircut.config import CONFIG
from haircut.enforcer import HaircutEnforcer
from haircut.preprocessing import binarize_hair, preprocess_frame
from haircut.trainer import DatasetManager


def _handle_param_keys(key: int) -> bool:
    if key in (ord('+'), ord('=')):
        CONFIG["HAIR_THRESH"] = min(CONFIG["HAIR_THRESH"] + 5, 240)
        print(f"[INFO] HAIR_THRESH → {CONFIG['HAIR_THRESH']}")
    elif key == ord('-'):
        CONFIG["HAIR_THRESH"] = max(CONFIG["HAIR_THRESH"] - 5, 10)
        print(f"[INFO] HAIR_THRESH → {CONFIG['HAIR_THRESH']}")
    elif key == ord('['):
        CONFIG["HAIR_VIOLATION_FRAC"] = max(round(CONFIG["HAIR_VIOLATION_FRAC"] - 0.05, 2), 0.05)
        print(f"[INFO] HAIR_VIOLATION_FRAC → {CONFIG['HAIR_VIOLATION_FRAC']}")
    elif key == ord(']'):
        CONFIG["HAIR_VIOLATION_FRAC"] = min(round(CONFIG["HAIR_VIOLATION_FRAC"] + 0.05, 2), 1.0)
        print(f"[INFO] HAIR_VIOLATION_FRAC → {CONFIG['HAIR_VIOLATION_FRAC']}")
    elif key == ord(','):
        CONFIG["HAIR_ROW_PERCENTILE"] = max(CONFIG["HAIR_ROW_PERCENTILE"] - 5, 10)
        print(f"[INFO] HAIR_ROW_PERCENTILE → {CONFIG['HAIR_ROW_PERCENTILE']}")
    elif key == ord('.'):
        CONFIG["HAIR_ROW_PERCENTILE"] = min(CONFIG["HAIR_ROW_PERCENTILE"] + 5, 100)
        print(f"[INFO] HAIR_ROW_PERCENTILE → {CONFIG['HAIR_ROW_PERCENTILE']}")
    elif key == ord(';'):
        CONFIG["SIDE_HAIR_ABS_THRESH"] = max(CONFIG["SIDE_HAIR_ABS_THRESH"] - 5, 30)
        print(f"[INFO] SIDE_HAIR_ABS_THRESH → {CONFIG['SIDE_HAIR_ABS_THRESH']}")
    elif key == ord("'"):
        CONFIG["SIDE_HAIR_ABS_THRESH"] = min(CONFIG["SIDE_HAIR_ABS_THRESH"] + 5, 180)
        print(f"[INFO] SIDE_HAIR_ABS_THRESH → {CONFIG['SIDE_HAIR_ABS_THRESH']}")
    else:
        return False
    return True


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CONFIG["CAMERA_INDEX"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["FRAME_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["FRAME_HEIGHT"])
    cap.set(cv2.CAP_PROP_FPS,          CONFIG["TARGET_FPS"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {CONFIG['CAMERA_INDEX']}")
        raise RuntimeError("Camera unavailable.")
    return cap


def _camera_loop(enforcer, window_title, overlay_fn, key_fn, show_mask_ref) -> None:
    cap = _open_camera()
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame drop — retrying…"); time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)
        annotated, det = enforcer.analyze_frame(frame)
        if show_mask_ref[0]:
            if det.status != "side_profile" and det.forehead_roi_box:
                rx, ry, rw, rh = det.forehead_roi_box
                gray_eq = preprocess_frame(frame, enforcer.clahe)
                mask    = binarize_hair(gray_eq[ry:ry + rh, rx:rx + rw])
                mask3   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask3[mask > 0] = [0, 0, 180]
                annotated[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(
                    annotated[ry:ry + rh, rx:rx + rw], 0.5, mask3, 0.5, 0)
            elif det.status == "side_profile":
                for box, mask in [(det.ear_roi_box, det.ear_mask),
                                  (det.neck_roi_box, det.neck_mask)]:
                    if box and mask is not None:
                        rx, ry, rw, rh = box
                        if rw > 0 and rh > 0:
                            try:
                                mask_r  = cv2.resize(mask, (rw, rh),
                                                     interpolation=cv2.INTER_NEAREST)
                                overlay = annotated[ry:ry + rh, rx:rx + rw].copy()
                                tint    = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
                                tint[mask_r > 0] = [0, 0, 180]
                                annotated[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(
                                    overlay, 0.6, tint, 0.4, 0)
                            except Exception:
                                pass
        overlay_fn(annotated, det)
        cv2.imshow(window_title, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key_fn(key, det, annotated):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_live(model_path=None) -> None:
    enforcer  = HaircutEnforcer(model_path=model_path)
    show_mask = [False]
    print("=" * 60)
    print("  Haircut Policy Enforcement  |  LIVE MODE")
    print(f"  Gender SVM: {'loaded: ' + model_path if model_path else 'not loaded (treat all as male)'}")
    print("  Q=quit  S=snapshot  T=mask  +/-=threshold")
    print("=" * 60)

    def key_fn(key, det, annotated):
        if key == ord('s'):
            fname = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[INFO] Snapshot: {fname}")
        elif key == ord('t'):
            show_mask[0] = not show_mask[0]
            print(f"[INFO] Mask overlay: {'ON' if show_mask[0] else 'OFF'}")
        _handle_param_keys(key)
        return False

    _camera_loop(enforcer, "Haircut Policy — Gate Monitor", lambda a, d: None, key_fn, show_mask)
    print(f"\nSession done. Total violations: {enforcer.violation_count}")


def run_collect(dataset_root: str = "my_dataset") -> None:
    enforcer = HaircutEnforcer()
    dm       = DatasetManager(dataset_root)
    counts   = {
        "male":   len(list(dm.male_dir.glob("*.png"))),
        "female": len(list(dm.female_dir.glob("*.png"))),
    }
    print("=" * 60)
    print(f"  COLLECT MODE  |  Dataset: {dm.root.resolve()}")
    print(f"  Existing — male: {counts['male']}  female: {counts['female']}")
    print("  M=male  F=female  Q=quit")
    print("=" * 60)

    def overlay_fn(annotated, det):
        cv2.putText(
            annotated,
            f"Saved  male:{counts['male']}  female:{counts['female']}  target: 200+ each",
            (8, CONFIG["UI_HEADER_H"] + 18), cv2.FONT_HERSHEY_SIMPLEX,
            CONFIG["UI_FONT_MED"], (200, 200, 50), 1, cv2.LINE_AA)
        if det.face_box:
            fx, fy, fw, fh = det.face_box
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)

    def key_fn(key, det, annotated):
        label = ("male"   if key in (ord('m'), ord('M'))
                 else "female" if key in (ord('f'), ord('F')) else None)
        _handle_param_keys(key)
        if label is not None:
            if det.face_crop_raw is not None and det.face_crop_raw.size > 0:
                saved = dm.save_sample(det.face_crop_raw, label)
                counts[label] += 1
                print(f"[{'M' if label == 'male' else 'F'}] {label.upper()} "
                      f"({counts[label]}): {saved.name}")
            else:
                print("[WARN] No valid face crop — cannot save sample.")
        return False

    _camera_loop(enforcer, "Collect — M=male  F=female  Q=quit",
                 overlay_fn, key_fn, [False])
    print(f"\nDone. male:{counts['male']}  female:{counts['female']}")