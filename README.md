# AquaVision: Monocular Underwater Perception for Fish Analytics and AUV Navigation

Fish detection, tracking, species classification and length estimation from a single (monocular) camera on AUVs.

This repository currently contains the **Stage A fish segmenter** (YOLO11s-seg) and the data preparation / augmentation code used to train it.

## Repository contents

| Path | What it is |
|---|---|
| `data_aug/` | Data preparation and augmentation scripts (dataset merge, 70/20/10 split, cached underwater copies `__aug` / `__ovl` / `__comp`, clean-vs-turbid split) |
| `training/AquaVision_Segmenter_Training_3h.ipynb` | Colab notebook that trains and tests both arms (baseline and AquaVision recipe) |
| `MODEL_PERFORMANCE.md` | Current results vs baseline |

> Datasets, `work.zip` and the `runs/` folder are **not** stored here (too large). Model weights are shared separately.

## Datasets

- **Brackish** and **DeepFish**, merged into one single-class detector dataset (`fish`).
- Both have boxes only, so most training masks are GrabCut pseudo-masks.
- Split 70/20/10 (train/val/test) per dataset. Validation and test images are never augmented.
- Original images: 14,653 train (10,093 Brackish, 4,560 DeepFish), 5,115 val, 2,799 test.

## Segmenter

- **Model:** YOLO11s-seg, COCO-pretrained, 640x640, `nc=1` (fish)
- **Framework:** Ultralytics 8.4.x (trained on 8.4.208), Colab T4 GPU

### Two arms, same model, same time budget, same validation and test splits

| Arm | Train data | Augmentation |
|---|---|---|
| `default` (baseline) | original train images only | stock Ultralytics defaults |
| `aug_single` (AquaVision recipe) | originals + cached underwater copies | flip 0.5, rotation 10 deg, translate 0.1, scale 0.5, HSV (0.015 / 0.5 / 0.3), mosaic 1.0, MixUp 0.1 |

Shared settings: `time=1.1` h per arm (about 10 epochs), cosine LR, patience 25, `close_mosaic=2` (mosaic and MixUp off for the last 2 epochs), seed 0, batch size found by AutoBatch on arm 1 (28) and reused for arm 2.

The 100-epoch recipe does not fit in a 3-hour Colab session, so `time=` compresses the same schedule into the budget.

## Results (aug_single, held-out test split)

| Subset | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| ALL (n=2,799) | 0.295 | 0.169 | 0.276 | 0.151 |
| Brackish (n=2,141) | 0.051 | 0.024 | 0.036 | 0.013 |
| DeepFish (n=658) | 0.546 | 0.314 | 0.534 | 0.296 |

Baseline comparison and the full table (precision, recall, length proxies, clean/turbid halves) are in [`MODEL_PERFORMANCE.md`](MODEL_PERFORMANCE.md).

## Limitations

- Early checkpoint: about 10 epochs, not a converged result.
- Brackish test performance is very low compared with DeepFish and with validation. This needs investigation.
- Mask metrics are measured against pseudo-masks, so treat them as relative (baseline vs augmented), not absolute.
- Many `__ovl` / `__comp` training copies were reported as corrupt or missing labels and ignored by Ultralytics.
- Mean box IoU and mask boundary error are length proxies, not measured fish length.

## Quick start

```bash
pip install -U ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("best.pt")                      # weights are shared separately
r = model.predict("frame.jpg", imgsz=640, conf=0.25)[0]
# r.boxes    -> detections
# r.masks.xy -> fish outlines
```

## Reproducing the training

1. Run the data preparation notebook and save `work.zip` to Google Drive under `MyDrive/aquavision/`.
2. Open `training/AquaVision_Segmenter_Training_3h.ipynb` in Colab (Runtime > Change runtime type > T4 GPU).
3. Run all cells. Finished arms are skipped and interrupted arms are resumed if Colab disconnects.

## Team

- (add names and roles)
