# Haircut Policy Detector

A real-time camera app that checks students' haircuts against the USJ-R grooming policy, built with Python and OpenCV.

## What it does

- Detects faces and eyes with OpenCV Haar cascades.
- Classifies gender with an SVM trained on HOG features, so the right policy rules are applied.
- Segments the hair region (CLAHE + thresholding) to flag bangs that fall too low on the forehead and hair that is too tall, with a separate side-profile check.
- Shows a Tkinter dashboard with the live feed, a Compliant / Violation status, gender confidence and a debug mask overlay.

## Run it

```
pip install -r requirements.txt
python -m haircut.main --model models/gender_svm.joblib
```

Other modes: `--mode live` (OpenCV window), `--mode collect` (save face samples to `my_dataset/`), `--mode train_svm` (retrain the gender model).

The training images and violation screenshots are not included in this repository.
