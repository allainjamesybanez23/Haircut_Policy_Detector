# main.py  — single entry point for all modes
from __future__ import annotations

import argparse

from haircut.config import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="USJ-R Haircut Policy Enforcement System")
    parser.add_argument("--mode", choices=["gui", "live", "collect", "train_svm"],
                        default="gui")
    parser.add_argument("--camera",    type=int,   default=CONFIG["CAMERA_INDEX"])
    parser.add_argument("--model",     type=str,   default=None,
                        help="Path to trained gender SVM .joblib (optional).")
    parser.add_argument("--dataset",   type=str,   default="my_dataset")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    if args.mode == "gui":
        from haircut.gui import HaircutEnforcerApp
        HaircutEnforcerApp(camera_index=args.camera, model_path=args.model).mainloop()

    elif args.mode == "live":
        from haircut.cli import run_live
        run_live(model_path=args.model)

    elif args.mode == "collect":
        from haircut.cli import run_collect
        run_collect(dataset_root=args.dataset)

    elif args.mode == "train_svm":
        from haircut.trainer import train_svm
        train_svm(
            dataset_root=args.dataset,
            model_path=args.model or "models/gender_svm.joblib",
            val_split=args.val_split,
        )


if __name__ == "__main__":
    main()