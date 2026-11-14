# AquaVision: Monocular Underwater Perception for Fish Analytics on AUVs

Fish **detection, segmentation, tracking-based counting, relative length and species classification**
from a single (monocular) camera on an autonomous underwater vehicle (AUV). Two trained models, a
Gradio web dashboard and an offline Windows desktop app.

> **Project:** AI Lab, School of Engineering and Applied Science, Ahmedabad University (application track)..

---

## Results at a glance (held-out test splits)

| Component | Model | Result |
|---|---|---|
| Fish detector / segmenter | YOLO11s-seg, 640 px | box mAP50 **0.670**, mAP50-95 0.394; mask mAP50 **0.676**, mAP50-95 0.351 |
| Species classifier (final) | Ensemble of 2 × YOLO11s-cls, 224 px | top-1 **98.9 %**, macro-F1 **0.943** (baseline 0.921) |
| Unknown-species safeguard | Confidence cut-off 0.998 | rejects **98.2 %** of unseen Brackish fish; keeps 96.5 % of known fish at 99.7 % accuracy |
| INT8 classifier (CPU) | ONNX Runtime, static INT8 | **5.8 MB**, **17 ms** per fish (3.3× faster), macro-F1 0.938 |
| INT8 detector (CPU) | INT8, output head kept FP32 | 1.5× faster, 2.7× smaller, −0.005 box mAP50 |

All splits are **70/20/10 by video clip**, and no clip appears in more than one split. Every number was
re-checked on a CPU-only Windows laptop.

## How it works

```
frame ──► YOLO11s-seg (boxes + masks, class "fish")
              ├─► ByteTrack ──────────► unique fish count, track IDs, movement
              ├─► PCA on mask ────────► relative length → Small / Medium / Large
              └─► crop ─► YOLO11s-cls ensemble ─► 1 of 23 species, or "Unknown"
```

- **Two stages, because of the data.** The detection datasets have only a single "fish" class, and
  the species labels exist only as pre-cropped images. No image has both.
- **Relative size only.** A single camera has no depth, so length is reported relative to the frame
  diagonal. Centimetres need a pixels-per-cm calibration at the fish's distance.
- **Transfer learning.** The detector starts from COCO weights and the classifier from ImageNet weights.

## Repository layout

| Path | What it is |
|---|---|
| `data_aug/` | Data preparation and augmentation: `prepare_data.py` (detector set; clip-level split, `--size_strata`, GrabCut `--pseudo_masks`), `f4k_prepare.py` (species set; per-class clip split, class balancing), `augment_offline.py` (cached `__aug` / `__ovl` / `__comp` copies, train split only), `underwater_aug.py` (9 underwater transforms), `turbid_split.py` (clean vs turbid), `ablation_prep.py` / `ablation_run.py` |
| `training/` | **Kaggle** notebooks: `train_segmenter_kaggle.ipynb` (detector baseline), `train_seg_aug.ipynb` (augmented detector runs A–D), `train_classifier_kaggle.ipynb` (classifier baseline), `train_cls_aug.ipynb` (augmented classifier). `train_aug_config.py` holds the online augmentation and two-phase training. |
| `evaluation/` | `eval_seg.py` (shared detector evaluation: per dataset, clean/turbid, recall by fish size, precision), `cls_local_test/` (ensemble, flip and unknown-species evaluation: `make_brackish_crops.py` → `cls_predict.py` → `cls_analyse.py`), `quant_local_test/` (INT8 benchmark: `quant_bench.py`, results JSON) |
| `dashboard/` | `Web_Dashboard.ipynb` (Gradio), `desktop_app/` (offline Windows Tkinter app: `pipeline.py`, `desktop_app.py`, `setup_windows.bat`, `run.bat`, tests, `samples/`; `weights/` holds the model files, which are not in git) |
| `MODEL_PERFORMANCE.md` | Detailed results and run history |

Datasets, prepared splits and run folders are too large for the repo; they are shared on Google
Drive. **Model weights are shared separately** (see [Weights](#weights)).

## Datasets

| Dataset | Used for | As used |
|---|---|---|
| **Brackish** (Pedersen et al., 2019), Roboflow 960×540 export | detector | 5,846 frames with fish, 14,120 boxes, 55 clips; boxes only (`fish` + `small_fish` → one class) |
| **DeepFish** (Saleh et al., 2020), Segmentation subset | detector | 620 images, 374 real fish masks; background frames capped 1:1 |
| **Fish4Knowledge** (Boom et al., 2014) | species | 27,370 crops, 23 reef species, 770:1 class imbalance |

OZFish was not used: it is box-only and added no masks.

**Leakage fix.** Roboflow file names end in a random hash, which hid the source clip. As a result,
near-identical neighbouring frames landed in both train and test, and the first detector scored
0.884 mAP50. Frames are now renamed `<clip>_frame_<NNNN>`, all splits are clip-level, and leakage is
asserted in code. The honest score is **0.670**.

## Models

### Detector: YOLO11s-seg
- Trained on Kaggle T4: 640 px, batch 16, up to 100 epochs; best validation epoch kept.
- **Test:** all 0.670 / 0.394 box, 0.676 / 0.351 mask (mAP50 / mAP50-95); precision 0.87, recall 0.60.
- **Per dataset:** Brackish box mAP50 0.664; DeepFish 0.824.

**Known issues**
- Overfits after about epoch 20.
- The clip split put small schooling fish in train and large single fish in test, which caps recall
  at about 0.77.
- Brackish masks are rectangles, because Brackish has boxes only.

**Fixes prepared but not yet run**
- A size-balanced split (`--size_strata 2`) and early stopping (`patience=25`).
- Augmented runs A–D: geometric → + photometric copies → + overlays and DeepFish→Brackish fish
  pasting → + GrabCut pseudo-masks.

### Species classifier: YOLO11s-cls ensemble

| Test split (2,737 crops) | Baseline | Augmented | **Ensemble (final)** |
|---|---|---|---|
| Top-1 | 0.987 | 0.987 | **0.989** |
| Macro-F1 (23 species) | 0.921 | 0.940 | **0.943** |
| Macro-F1, 16 clip-split species | 0.940 | 0.955 | **0.961** |
| Head / mid / tail macro-F1 | 0.992 / 0.928 / 0.879 | 0.993 / 0.946 / 0.909 | 0.992 / **0.956** / 0.908 |
| Test errors | 36 | 35 | **31** |

**Augmented model**
- Species-safe underwater augmentation: no colour cast, because body colour identifies a species.
- Rare species oversampled to about 300 crops; dominant classes softly capped at 2,500.
- Fixed the baseline's overfitting, but flipped its bias from "everything is fish_01" to
  over-calling rare species.

**Ensemble**
- 0.25 × baseline + 0.75 × augmented, each averaged with a mirror-image crop.
- The two models err in opposite directions, so averaging cancels much of both biases.

**Statistical caveat.** Some species have only 2–3 test crops, so macro-F1 has a 95% CI of about
±0.05. Against the augmented model alone, the ensemble fixed 10 errors and broke 6 (McNemar
p = 0.45): consistent, but not yet significant.

**Unknown species.** Fish4Knowledge covers 23 Taiwanese reef species. Fish whose ensemble confidence
is below **0.99827** are labelled `Unknown`. In fast single-model mode the cut-off is **0.99790**.

### Weights

| File | Model |
|---|---|
| `detector_seg.pt` | detector (seg v2) |
| `species_cls.pt` | augmented classifier |
| `species_cls_base.pt` | baseline classifier (ensemble partner; optional) |

Place them in `dashboard/desktop_app/weights/`, or select them in the dashboards.

## Dashboards

**Gradio web dashboard** (`dashboard/Web_Dashboard.ipynb`; Colab or Hugging Face Spaces)
- Tabs: **Image**, **Video**, **Datasets**, **About**.
- Output: annotated video (IDs, movement trails, species, size class), a per-fish table (species,
  confidence, size, frames seen, path length, speed, heading) and CSV downloads.
- Sliders: confidence, IoU, size boundaries, trail length, re-classification interval, pixels-per-cm.

**Offline Windows desktop app** (`dashboard/desktop_app/`)
- No internet needed after setup.
- Live annotated playback with pause and stop, partial results when stopped, CSV export.
- Ensemble + `Unknown` cut-off; fast single-model mode is 8× faster per fish.
- Tested on Windows 11: regression tests and GUI smoke tests pass; about **2.4 frames/s** on a laptop CPU.

```bat
:: Windows, once (needs internet), then every launch
setup_windows.bat
run.bat
```

**Use sample** loads `samples/brackish_test_clip.mp4`, a held-out test clip the detector never
trained on. Expected result: 1 fish tracked, size "Large", species "Unknown".

## Quick start (Python)

```python
from ultralytics import YOLO

det = YOLO("detector_seg.pt")
r = det.track("clip.mp4", tracker="bytetrack.yaml", imgsz=640, conf=0.25, persist=True)
# per frame: r[i].boxes.xyxy, r[i].boxes.id (track IDs), r[i].masks.xy (fish outlines)

# species with the ensemble + Unknown cut-off (from the desktop app's pipeline.py)
import cv2, pipeline
species, conf = pipeline.classify_crop(cv2.imread("fish_crop.jpg"))   # e.g. ("fish_05", 0.9995) or ("Unknown", 0.71)
```

## Reproducing the training (Kaggle)

1. **Create two Kaggle datasets:**
   - `aquavision-scripts`: the `.py` files from `data_aug/`, `training/` and `evaluation/eval_seg.py`
   - `aquavision-brackish`: the Roboflow Brackish 960×540 YOLOv11 zip

   DeepFish is downloaded from Hugging Face (`Alzayats/DeepFish`) inside the notebook, and
   Fish4Knowledge is cloned from GitHub.
2. **Set up the notebook:** Accelerator **GPU T4 x2**, Internet **On**, both datasets attached.
3. **Run the training:** use **Save Version → Save & Run All (Commit)**. It runs headless to the end
   and survives disconnects.
4. **Download the results** from the version's **Output** tab: the run folder (`best.pt`,
   `test_metrics.json`, `results.csv`, …).

Order:

```
train_segmenter_kaggle.ipynb → train_seg_aug.ipynb (RUN = A, B, C, D)
train_classifier_kaggle.ipynb → train_cls_aug.ipynb
```

The test split is identical across baseline and augmented runs. The notebooks check this and stop
if it differs.

## Earlier Colab segmenter runs (superseded)

A previous 3-hour Colab pipeline trained two arms (`default`, `aug_single`) on a different data build
(`work.zip`, Brackish frames extracted from video). Both runs stopped after about 10–13 epochs,
and Brackish scored only 0.05–0.09 box mAP50 against 0.52–0.57 for DeepFish. That gap points to
misaligned Brackish labels in that build, not to the model. These runs are kept in
`MODEL_PERFORMANCE.md` for the record, but **they are not comparable with the results above and
should not be reported.**

## Limitations

- **Counts are lower bounds:** detector recall is 0.60.
- **Speed:** CPU throughput (about 2.4 fps) suits post-mission analysis, not real-time steering.
- **Species:** only meaningful for the 23 Fish4Knowledge reef species; others are `Unknown`
  (tested on Brackish; DeepFish species untested).
- **Length:** relative (monocular); Brackish masks are rectangles, so precise outlines come from
  DeepFish only.
- **Tracking:** MOTA, IDF1 and count error are not yet measured; they need tracking ground truth
  (BrackishMOT).

## Future work

- **Detector:** run the prepared experiments (size-balanced baseline, augmented runs A–D); target
  box mAP50 ≥ 0.72 and recall ≥ 0.70.
- **On-board deployment:** TensorRT FP16/INT8 on a Jetson-class module, or OpenVINO / ONNX Runtime on
  a companion computer, keeping the detector head in FP32. Quantization-aware training if
  post-training INT8 loses accuracy.
- **Evaluation:** tracking metrics (MOTA, IDF1, count error) on BrackishMOT.
- **Species:** hierarchical family-then-species classification.
- **Model:** a one-line upgrade to YOLO26.

## Team

| Name | Role |
|---|---|
| Mihiran Chakraborty | Model Training / Quantization Research |
| Aditya Khuttan | Data Preparation and Augmentation |
| Mudaliar Krishnan | Gradio Dashboard Design and Deployment |
| Aditya Sahu | Dashboard Prototyping and Testing |

## References

Ultralytics YOLO11 · ByteTrack (Zhang et al., ECCV 2022) · Brackish (Pedersen et al., CVPR-W 2019) ·
DeepFish (Saleh et al., Sci. Rep. 2020) · Fish4Knowledge (Boom et al., Ecol. Inform. 2014) ·
Integer quantization (Jacob et al., CVPR 2018) · BrackishMOT (Pedersen et al., SCIA 2023).
