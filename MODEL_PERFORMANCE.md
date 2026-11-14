# AquaVision: Model Performance

All numbers are on **held-out test splits**, split **70/20/10 by video clip** so that no clip is shared
between splits. They were re-verified locally on a CPU-only Windows laptop. Raw outputs are in
`evaluation/`; the full discussion is in `report/AquaVision_Final_Report.tex`.

## 1. Species classifier (YOLO11s-cls, 224 px, Fish4Knowledge, 23 species)

Same test split for every run: 2,737 crops, identical per species, checked in code.

| Metric | Baseline (no aug) | Augmented run 1 | **Ensemble + flip (final)** |
|---|---|---|---|
| Top-1 accuracy | 0.9868 | 0.9872 | **0.9887** |
| Top-5 accuracy | 0.998 | 0.999 | n/a |
| Macro-F1 (23 species) | 0.9208 | 0.9399 | **0.9428** |
| Macro-F1 95% bootstrap CI | 0.859–0.956 | 0.856–0.964 | n/a |
| Macro-F1, 16 clip-split species | 0.9404 | 0.9553 | **0.9615** |
| Macro-F1 head (≥1000 train) / mid / tail (<100) | 0.992 / 0.928 / 0.879 | 0.993 / 0.946 / 0.909 | 0.992 / **0.956** / 0.908 |
| Test errors | 36 | 35 | **31** |
| Rare → fish_01 / fish_01 → rare | 19 / 6 | 7 / 18 | 11 / 11 |
| Validation loss | rises after epoch 28 (overfit) | keeps falling to about epoch 50 | n/a |

**Augmented run 1 settings**
- Species-safe underwater augmentation (`f4k_prepare.py`): no colour cast, plus turbidity, haze,
  CLAHE, motion blur and erasing.
- Rare species oversampled towards 300 crops; head classes softly capped at 2,500 (`--target 300
  --max_per_class 2500 --ceil_mult 10 --max_aug_ratio 12 --extra_aug 0.2`).
- Mild online augmentation: flip 0.5, scale 0.25, S/V 0.2, erasing 0.1, no RandAugment.
- Training: 60 epochs, batch 64, ImageNet-pretrained.

**Ensemble**
- 0.25 × baseline + 0.75 × augmented, both run on the crop and its mirror image.
- Weight and flip were chosen on the **validation** split, then the test split was scored once.
- Prior (logit) correction was also tried: no gain.

**Significance.** Against augmented run 1, the ensemble fixed 10 errors and broke 6 (McNemar
p = 0.45). The macro-F1 difference has a 95% CI of [−0.023, +0.028]. The direction is consistent,
but the gain is not statistically significant on this test set.

### Unknown-species cut-off
The confidence threshold keeps 98% of the correct validation predictions:

| Model | Threshold | F4K test kept | Accuracy on kept | Brackish fish rejected (n = 1,523) |
|---|---|---|---|---|
| Ensemble + flip | 0.99827 | 96.5 % | 99.7 % | **98.2 %** |
| Augmented alone (fast mode) | 0.99790 | 95.9 % | 99.6 % | 96.5 % |

DeepFish species were not tested.

## 2. Detector / segmenter (YOLO11s-seg, 640 px, Brackish + DeepFish)

| Run | Split | Test box mAP50 / 50-95 | Test mask mAP50 / 50-95 | P / R | Status |
|---|---|---|---|---|---|
| v1 | frame-level (**leaky**) | 0.884 / 0.566 | 0.787 / 0.344 | 0.90 / 0.80 | inflated: do not report |
| **v2 (final)** | clip-level | **0.670 / 0.394** | **0.676 / 0.351** | 0.87 / 0.60 | current weights |
| v2, Brackish only (578 imgs) | clip-level | 0.664 / 0.391 | 0.670 / 0.349 | 0.88 / 0.58 | |
| v2, DeepFish only (62 imgs, 30 fish) | clip-level | 0.824 / 0.485 | 0.824 / 0.441 | 0.86 / 0.82 | |

**v2 setup.** Trained 100 epochs, but the best validation epoch was 20. Batch 16, no augmentation,
COCO-pretrained.

**Diagnosis**
- **Overfitting:** training mask loss falls to 0.008 while validation mask loss rises eightfold.
- **Size shift between splits:** train fish have a median size of 44 px (81 % small schooling fish);
  test fish 99 px (large single fish). Recall never exceeds 0.77.
- **Rectangle masks:** Brackish has boxes only, so its masks are rectangles.

**Prepared, not yet run:** baseline v3 (`--size_strata 2`, which gives test median 54 px, plus
`patience=25`), then augmented runs A–D in `training/train_seg_aug.ipynb`. All are scored with
`evaluation/eval_seg.py`: per dataset, clean/turbid, recall by size, precision.

### Superseded: earlier Colab segmenter runs (kept for the record; do not report)

These runs replaced the earlier version of this file. They used a **separate data build** (`work.zip`:
Brackish frames extracted from video, GrabCut pseudo-masks; split 14,653 train (10,093 Brackish, 4,560
DeepFish) / 5,115 val / 2,799 test), so they **cannot be compared with the clip-level results above**.

**Setup.** Both are YOLO11s-seg (COCO-pretrained), 640 px, one class, trained on a Colab T4 under a
time budget: the 100-epoch schedule was compressed with `time=`, so they are early checkpoints, not
converged models.
- `default` arm: original images with stock Ultralytics augmentation.
- `aug_single` arm: cached underwater copies plus online geometric augmentation (flip 0.5, rotation
  10°, translate 0.1, scale 0.5, HSV 0.015/0.5/0.3, mosaic 1.0, MixUp 0.1).

**Run A: `pdf3h`** (about 1.1 h per arm, about 10 epochs, batch 28, mosaic and MixUp off for the last
2 epochs). Only `aug_single` was recorded:

| aug_single, test | n | Box P / R | Box mAP50 / 50-95 | Mask mAP50 / 50-95 | Mean box IoU | Mask boundary error |
|---|---|---|---|---|---|---|
| All | 2,799 | 0.438 / 0.365 | 0.295 / 0.169 | 0.276 / 0.151 | 0.338 | 0.0048 |
| Brackish | 2,141 | n/a | **0.051** / 0.024 | 0.036 / 0.013 | 0.191 | 0.0066 |
| DeepFish | 658 | n/a | 0.546 / 0.314 | 0.534 / 0.296 | 0.428 | 0.0042 |
| Clean half | 738 | n/a | 0.279 / 0.181 | 0.267 / 0.174 | 0.305 | 0.0040 |
| Turbid half | 1,049 | 0.669 / 0.580 | 0.633 / 0.339 | 0.587 / n/a | n/a | n/a |

Validation (5,115 images): box P 0.473, R 0.367, mAP50 0.329, mAP50-95 0.201.

**Run B: `t14`** (1.4 h per arm, stopped after 12–13 of 100 epochs, batch 32, mosaic and MixUp off
for the last 3 epochs). Both arms on the same split:

| Test subset | Box mAP50, default | Box mAP50, aug_single | Gain | Mask mAP50, default | Mask mAP50, aug_single |
|---|---|---|---|---|---|
| All (2,799) | **0.319** | 0.291 | −0.028 | **0.288** | 0.276 |
| Brackish (2,141) | **0.093** | 0.069 | −0.024 | **0.063** | 0.049 |
| DeepFish (658) | **0.572** | 0.520 | −0.052 | **0.545** | 0.516 |
| Clean half | **0.337** | 0.240 | −0.097 | **0.314** | 0.232 |
| Turbid half | **0.725** | 0.657 | −0.068 | **0.650** | 0.616 |

**Why these runs are superseded**
- **Not converged.** Every curve was still improving when the time budget stopped training, so at
  about 12 epochs augmentation had not yet paid off: `default` beats `aug_single` on every subset.
- **Brackish labels look broken in this build.** Brackish scored 0.05–0.09 against 0.52–0.57 for
  DeepFish in the same model, and against 0.664 for our v2 on the Roboflow Brackish frames. That gap
  points to boxes that do not line up with the fish in the video-extracted frames (frame offset or
  ×2 scale), not to a model problem.
- **Many augmented copies were dropped.** Numerous `__ovl` / `__comp` copies were reported as corrupt
  (missing images or labels) and ignored by Ultralytics. The `aug_single` arm therefore effectively
  trained on originals plus `__aug` copies.
- **Masks were measured against GrabCut pseudo-masks.** Mask mAP and boundary error are only useful
  as relative comparisons within that build. Mean box IoU and boundary error are length *proxies*,
  not measured fish length.

## 3. INT8 quantization (laptop CPU, Intel i5-1135G7, batch 1, ONNX Runtime static INT8)

| Model | Format | Size (MB) | ms / image | Accuracy |
|---|---|---|---|---|
| Classifier | PyTorch | 11.1 | 54.6 | macro-F1 0.940, top-1 0.987 |
| Classifier | ONNX FP32 | 21.9 | 36.4 | macro-F1 0.940, top-1 0.987 |
| Classifier | **ONNX INT8** | **5.8** | **16.8** | macro-F1 **0.938**, top-1 **0.986** |
| Detector (Brackish test) | PyTorch | 20.5 | 321 | box 0.664, mask 0.670 mAP50 |
| Detector | ONNX FP32 | 40.6 | 367 | box 0.725, mask 0.715 mAP50 |
| Detector | ONNX INT8, all layers | 10.9 | 160 | **box 0.000: the model breaks** |
| Detector | **ONNX INT8, head FP32** | **15.2** | **239** | box **0.720**, mask **0.712** mAP50 |

ONNX FP32 scores higher than PyTorch on the detector, most likely because the fixed-size ONNX model
sees a square 640×640 input while PyTorch validation uses a rectangular letterbox. INT8 is therefore
compared with ONNX FP32. The script is `evaluation/quant_local_test/quant_bench.py`.

## 4. End-to-end (desktop app, laptop CPU)
- Held-out Brackish test clip (168 frames): 1 fish tracked for 64 frames, size Large, species
  Unknown (correct: not a reef species). Throughput 2.4 frames/s.
- Species path, ensemble: about 171 ms per fish crop. Fast single-model mode: about 20 ms.

## Weights and usage

| File | Model | Run |
|---|---|---|
| `detector_seg.pt` | YOLO11s-seg detector / segmenter | seg v2, clip-level split (Kaggle, Ultralytics 8.4.190) |
| `species_cls.pt` | YOLO11s-cls, augmented | cls augmented run 1 |
| `species_cls_base.pt` | YOLO11s-cls, baseline | cls baseline (ensemble partner) |

Weights are shared separately (GitHub Release / Drive), not stored in git.

```python
from ultralytics import YOLO

det = YOLO("detector_seg.pt")
r = det.predict("frame.jpg", imgsz=640, conf=0.25)[0]
# r.boxes -> detections, r.masks.xy -> fish outlines

# species: ensemble + Unknown cut-off, as used by the desktop app
import cv2, pipeline   # dashboard/desktop_app/pipeline.py
species, conf = pipeline.classify_crop(cv2.imread("fish_crop.jpg"))
```
