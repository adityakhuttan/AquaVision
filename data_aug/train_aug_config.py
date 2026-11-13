"""
train_aug_config.py -- hand-off for the training teammates (report Sec. 6.1 / 6.3)
==================================================================================
NOT executed during data prep (needs `pip install ultralytics` + GPU). Kept here so the online
half of the augmentation scheme is documented next to the offline half.

Tier (i) GEOMETRIC is done online by Ultralytics:
    flip, small rotation, scale/zoom, bounding-box-safe crop (translate+scale, boxes/masks are
    transformed and clipped together), mosaic, mixup.
Tier (ii)/(iii) photometric + occlusion + domain are already cached offline (augment_offline.py).

"Mosaic and MixUp are disabled in the final epochs"  -> close_mosaic (Ultralytics switches
    mosaic / mixup / copy_paste off for the last N epochs).
"Synthetic overlays disabled in the final epochs"    -> phase 2 trains on data_late.yaml, whose
    image list excludes the __ovl (overlay) and __comp (composite) copies.
"""
from pathlib import Path

GEOMETRIC = dict(
    fliplr=0.5, flipud=0.0,
    degrees=10.0,            # small rotation
    translate=0.1, scale=0.5,  # zoom / crop (box-safe)
    shear=0.0, perspective=0.0,
    mosaic=1.0, mixup=0.1, copy_paste=0.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,   # Ultralytics' own mild HSV jitter on top of the cached UW copies
)


def train_detector(det_dir="/content/work/det", weights="yolo11s-seg.pt", epochs=100,
                   late_epochs=15, imgsz=640, batch=16, patience=25,
                   project="/content/drive/MyDrive/aquavision/runs", name="det"):
    """Two-phase detector training (v7 fixes, from the segmenter baseline: best epoch was 20/100).

    * patience=25 in phase 1 -> stops once val fitness stops improving (Ultralytics default of 100
      would never trigger inside an 85-epoch phase).
    * phase 2 is a NEW train() call, so LR warm-up would restart and wreck the converged model.
      Fix: warmup_epochs=0, low lr0, and an EXPLICIT optimizer (with optimizer='auto' Ultralytics
      ignores lr0 and picks its own LR).
    * batch=16 / imgsz=640 / yolo11s-seg.pt to match the baseline run (fair comparison).
    Alternative (single run, no overlay switch-off): epochs=100, close_mosaic=15, patience=25 on data_full.yaml.
    """
    from ultralytics import YOLO
    det = Path(det_dir)
    m = YOLO(weights)
    # phase 1: everything (offline copies incl. overlays/composites), mosaic+mixup ON
    m.train(data=str(det / "data_full.yaml"), epochs=epochs - late_epochs, imgsz=imgsz, batch=batch,
            patience=patience, close_mosaic=0, project=project, name=f"{name}_p1", **GEOMETRIC)
    # phase 2: final epochs, mosaic/mixup OFF and no overlays/composites in the image list
    m2 = YOLO(str(m.trainer.best))
    late = dict(GEOMETRIC, mosaic=0.0, mixup=0.0, copy_paste=0.0)
    m2.train(data=str(det / "data_late.yaml"), epochs=late_epochs, imgsz=imgsz, batch=batch,
             patience=late_epochs,                      # no early stop inside the short phase
             optimizer="SGD", lr0=0.001, lrf=0.1, warmup_epochs=0,
             project=project, name=f"{name}_p2", **late)
    return m2


# Online (label-safe) augmentation for the classifier - brief P3: the baseline memorised the train set
# (train loss 0.000 at epoch 55, val loss lowest at epoch 28). These change every epoch.
# Compare with the baseline's args.yaml and change ONLY the data + these flags. Argument names follow
# Ultralytics' classification trainer - check `yolo cfg` if your version complains.
CLS_ONLINE = dict(fliplr=0.5, flipud=0.0,          # fish look the same mirrored; never flip vertically
                  scale=0.5,                       # mild scale / crop
                  hsv_h=0.01, hsv_s=0.3, hsv_v=0.2,   # a little HSV jitter (hue is a species cue -> keep it small)
                  degrees=0.0)


def train_classifier(cls_dir="/content/work/cls", weights="yolo11s-cls.pt", epochs=60, imgsz=224, batch=64,
                     project="/content/drive/MyDrive/aquavision/runs", name="cls_aug", device=0):
    # train/ is already class-balanced offline (f4k_prepare.py); val/ and test/ are untouched.
    # Same model + schedule as the baseline (yolo11s-cls, 224, batch 64, 60 epochs) -> fair comparison.
    from ultralytics import YOLO
    m = YOLO(weights)
    m.train(data=cls_dir, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
            project=project, name=name, **CLS_ONLINE)
    return m
