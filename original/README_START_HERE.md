# PCB preprocessing — teammate 1 (interim revision)

This package is a **partial improvement**, not a universal PCB detector. It preserves the original B01/B02 geometry and now compares any number of board IDs in the Stage 3a report. Stage 3b requires QC calibration and records rejected images. **New board geometry still needs a validated alignment/crop calibration**; do not claim B03/B04 support has been tested.

## Install

```bash
python -m pip install numpy opencv-python
```

Extract the original dataset so its root contains `metadata.csv` and `raw/`. From the directory containing these scripts:

```bash
python stage1_dataset.py --root PATH_TO_DATASET --out PATH_TO_OUTPUT --check-images
python stage3a_probe.py --root PATH_TO_DATASET --out PATH_TO_OUTPUT
python stage3b_export.py --root PATH_TO_DATASET --out PATH_TO_OUTPUT
```

Stage 3b requires `PATH_TO_OUTPUT/_stage3/calibration_good_only.json`, which Stage 3a writes **only if calibration succeeds**. Do not run Stage 3b until Stage 3a reports have been inspected. Failed QC samples are listed in `dataset_aligned/manifest.csv` but are not exported. Keep raw images unchanged.

## Known limitations / next work

- The homography anchors and fixed crop still assume the original PCB mounting-hole layout.
- GOOD-only calibration is currently shared across boards; new physical board designs may require independent per-design templates and calibration.
- New PCB types require representative GOOD images to determine whether the geometry is compatible.
- No real-image end-to-end success claim is made for this revision.
