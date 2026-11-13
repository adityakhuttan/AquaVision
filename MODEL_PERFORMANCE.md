# AquaVision Segmenter: Current Performance (aug_single)

**Model:** YOLO11s-seg (COCO-pretrained), 640x640, one class (`fish`)
**Weights:** `runs/segmenter_aug_single_pdf3h/weights/best.pt`
**Recipe:** AquaVision recipe (Sec. 6.1): cached underwater copies + online geometric aug, mosaic/MixUp off for the last 2 epochs
**Training:** about 10 epochs (time-boxed to ~1.1 h on a Colab T4, batch 28, seed 0, imgsz 640). This is an early checkpoint; the 100-epoch recipe does not fit the budget.
**Evaluation:** held-out test split (2,799 images), conf = 0.001 for mAP; length proxies use conf = 0.25 on up to 600 images.

## Validation (best.pt, 5,115 images)

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.473 | 0.454 |
| Recall | 0.367 | 0.348 |
| mAP@0.5 | 0.329 | 0.323 |
| mAP@0.5:0.95 | 0.201 | not recorded |

## Test results: aug_single vs baseline

Fill the **Baseline** column from `runs/segmenter_default_pdf3h/test_metrics.csv` (same test split, same schedule), then compute **Gain** = aug_single - baseline.

### ALL (test, n = 2,799)

| Metric | Baseline (default) | aug_single | Gain |
|---|---|---|---|
| Box P | TBD | 0.4378 | |
| Box R | TBD | 0.3646 | |
| Box mAP@0.5 | TBD | 0.2949 | |
| Box mAP@0.5:0.95 | TBD | 0.1687 | |
| Mask P | TBD | 0.3788 | |
| Mask R | TBD | 0.3641 | |
| Mask mAP@0.5 | TBD | 0.2764 | |
| Mask mAP@0.5:0.95 | TBD | 0.1507 | |
| Mean box IoU (length proxy) | TBD | 0.3377 | |
| Mask boundary error (lower is better) | TBD | 0.0048 | |

### Per subset (aug_single only; add baseline the same way)

| Subset | n | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | Mean box IoU | Boundary err |
|---|---|---|---|---|---|---|---|
| Brackish | 2,141 | 0.0505 | 0.0240 | 0.0364 | 0.0131 | 0.1908 | 0.0066 |
| DeepFish | 658 | 0.5459 | 0.3137 | 0.5336 | 0.2961 | 0.4280 | 0.0042 |
| Clean half | 738 | 0.2787 | 0.1808 | 0.2670 | 0.1739 | 0.3054 | 0.0040 |
| Turbid half | 1,049 | 0.633 | 0.339 | 0.587 | n/a | n/a | n/a |

Turbid-half precision/recall (box): 0.669 / 0.580; (mask): 0.634 / 0.515. Mask mAP@0.5:0.95, IoU and boundary error for the turbid half were not captured in the run log; read them from `test_metrics.csv`.

## Known limitations

- **Short training.** About 10 epochs instead of the 100-epoch recipe, so all numbers are a first checkpoint, not a converged result.
- **Brackish gap.** Brackish test mAP50 (0.05) is far below its validation level (overall val mAP50 0.329) and far below DeepFish (0.55). Brackish is the weak point and needs investigation before relying on the model for that footage.
- **Pseudo-masks.** Brackish and DeepFish have boxes only, so most training and test masks are GrabCut pseudo-masks. Mask mAP and boundary error are useful for relative comparison (baseline vs augmented), not as absolute segmentation quality.
- **Dropped training copies.** Many `__ovl` / `__comp` copies were reported as corrupt (missing images/labels) and ignored by Ultralytics, so the aug arm effectively trained on originals plus `__aug` copies.
- **Length proxies** (mean box IoU, boundary error) are proxies, not measured fish length.

## Usage

```python
from ultralytics import YOLO
model = YOLO("best.pt")
r = model.predict("frame.jpg", imgsz=640, conf=0.25)[0]
# r.boxes -> detections, r.masks.xy -> fish outlines
```

Trained with `ultralytics==8.4.208`.
