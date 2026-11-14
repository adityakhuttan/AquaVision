#!/usr/bin/env python
"""
eval_seg.py -- one evaluation for every detector/segmenter run (baseline and augmented)
=======================================================================================
Same numbers, same test frames, same code for every run, so runs can be compared directly.

  python eval_seg.py --weights runs/x/weights/best.pt --det work/det --out runs/x

Writes <out>/test_metrics.json with
  all / brackish / deepfish       box + mask mAP50, mAP50-95, P, R  (per-dataset test subsets)
  clean / turbid                  same, on the turbid_split.py halves of the test set (originals only)
  recall_by_size                  recall at conf 0.25, IoU 0.5, by fish size at the 640 px model input:
                                  small < 24 px, medium 24-64 px, large >= 64 px (sqrt of box area;
                                  a Brackish 960x540 fish of 36 / 96 px in the frame = 24 / 64 px here)
  precision_at_conf               TP / FP / precision at conf 0.25
and copies the test-split plots (PR curves, confusion matrix, example predictions) to <out>/test_plots/.
Only ORIGINAL test images are used (augmented copies only ever exist in train/).
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SIZE_BINS = (("small", 0, 24), ("medium", 24, 64), ("large", 64, 1e9))   # px at the 640 model input
PLOTS = ("confusion_matrix.png", "BoxPR_curve.png", "MaskPR_curve.png", "BoxF1_curve.png", "MaskF1_curve.png",
         "val_batch0_pred.jpg", "val_batch0_labels.jpg", "val_batch1_pred.jpg", "val_batch1_labels.jpg")


def score(m, n_img=None, n_inst=None):
    r = {"box_mAP50": m.box.map50, "box_mAP50_95": m.box.map, "mask_mAP50": m.seg.map50,
         "mask_mAP50_95": m.seg.map, "box_P": m.box.mp, "box_R": m.box.mr}
    r = {k: round(float(v), 4) for k, v in r.items()}
    if n_img is not None:
        r.update(images=n_img, instances=n_inst)
    return r


def subset_yaml(det: Path, name: str, images):
    """data yaml whose test split is exactly `images` (absolute paths)."""
    lst = det / "lists" / f"eval_{name}.txt"
    lst.parent.mkdir(exist_ok=True)
    lst.write_text("\n".join(str(p) for p in images) + "\n")
    y = det / f"data_eval_{name}.yaml"
    y.write_text(f"path: {det}\ntrain: images/train\nval: images/val\ntest: {lst}\nnc: 1\nnames:\n  0: fish\n")
    return str(y)


def gt_boxes(label: Path):
    """normalised (x1, y1, x2, y2) of every polygon in a YOLO-seg label file."""
    out = []
    for line in label.read_text().splitlines():
        v = line.split()
        if len(v) >= 7:
            p = np.array(list(map(float, v[1:])), np.float32).reshape(-1, 2)
            out.append([*p.min(0), *p.max(0)])
    return np.array(out, np.float32).reshape(-1, 4)


def iou(a, b):
    """IoU matrix between boxes a (N,4) and b (M,4), same units."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt, pr, thr=0.5):
    """greedy one-to-one matching by IoU (predictions already sorted by confidence) -> matched gt mask."""
    hit = np.zeros(len(gt), bool)
    if len(gt) and len(pr):
        m = iou(pr, gt)
        for i in range(len(pr)):
            j = np.argmax(np.where(hit, -1, m[i]))
            if not hit[j] and m[i, j] >= thr:
                hit[j] = True
    return hit


def size_recall(model, det: Path, images, conf=0.25, imgsz=640):
    tot = {n: [0, 0] for n, _, _ in SIZE_BINS}
    tp = fp = 0
    for i in range(0, len(images), 32):
        chunk = images[i:i + 32]
        for p, r in zip(chunk, model.predict([str(x) for x in chunk], conf=conf, imgsz=imgsz, verbose=False)):
            H, W = r.orig_shape
            gt = gt_boxes(det / "labels" / "test" / f"{p.stem}.txt") * np.array([W, H, W, H], np.float32)
            order = np.argsort(-r.boxes.conf.cpu().numpy())
            pr = r.boxes.xyxy.cpu().numpy()[order]
            hit = match(gt, pr)
            tp += int(hit.sum()); fp += int(len(pr) - hit.sum())
            px = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) * imgsz / max(H, W)
            for name, lo, hi in SIZE_BINS:
                sel = (px >= lo) & (px < hi)
                tot[name][0] += int(hit[sel].sum()); tot[name][1] += int(sel.sum())
    rec = {n: {"recall": round(h / max(1, t), 4), "fish": t} for n, (h, t) in tot.items()}
    return rec, {"conf": conf, "TP": tp, "FP": fp, "precision": round(tp / max(1, tp + fp), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--det", required=True, help="prepared dataset folder (manifest.csv, data.yaml)")
    ap.add_argument("--out", required=True, help="run folder: test_metrics.json + test_plots/ go here")
    ap.add_argument("--project", default=None, help="where Ultralytics writes its val folders (default: <out>/..)")
    a = ap.parse_args()
    from ultralytics import YOLO

    det, out = Path(a.det).resolve(), Path(a.out)
    project = a.project or str(out.parent)
    model = YOLO(a.weights)
    rows = [r for r in csv.DictReader(open(det / "manifest.csv")) if r["split"] == "test"]
    img = lambda rs: [det / "images" / "test" / r["file"] for r in rs]
    n_inst = lambda rs: sum(int(r["n_instances"]) for r in rs)
    res = {}

    m = model.val(data=subset_yaml(det, "all", img(rows)), split="test", project=project,
                  name=f"{out.name}_test", exist_ok=True)
    res["all"] = score(m, len(rows), n_inst(rows))
    (out / "test_plots").mkdir(parents=True, exist_ok=True)
    for t in PLOTS:
        src = Path(project) / f"{out.name}_test" / t
        if src.exists():
            shutil.copy(src, out / "test_plots" / t)

    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        m = model.val(data=subset_yaml(det, ds, img(sub)), split="test", project=project,
                      name=f"{out.name}_test_{ds}", exist_ok=True, plots=False)
        res[ds] = score(m, len(sub), n_inst(sub))

    # clean vs turbid halves (turbid_split.py scores ORIGINAL images only, per-dataset median of train)
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "turbid_split.py"), "--det", str(det)], check=True)
    lab = {r["file"]: r["label"] for r in csv.DictReader(open(det / "turbidity.csv")) if r["split"] == "test"}
    for t in ("clean", "turbid"):
        sub = [r for r in rows if lab.get(r["file"]) == t]
        m = model.val(data=subset_yaml(det, t, img(sub)), split="test", project=project,
                      name=f"{out.name}_test_{t}", exist_ok=True, plots=False)
        res[t] = score(m, len(sub), n_inst(sub))

    res["recall_by_size"], res["precision_at_conf"] = size_recall(model, det, img(rows))
    (out / "test_metrics.json").write_text(json.dumps(res, indent=1))

    print(f"\n{'subset':9s} {'imgs':>5s} {'fish':>5s}  box50  box50-95  mask50  mask50-95  recall")
    for k in ("all", "brackish", "deepfish", "clean", "turbid"):
        if k in res:
            v = res[k]
            print(f"{k:9s} {v['images']:5d} {v['instances']:5d}  {v['box_mAP50']:.3f}  {v['box_mAP50_95']:.3f}"
                  f"     {v['mask_mAP50']:.3f}   {v['mask_mAP50_95']:.3f}     {v['box_R']:.3f}")
    print("recall by fish size (conf 0.25, IoU 0.5):",
          {k: f"{v['recall']:.3f} (n={v['fish']})" for k, v in res["recall_by_size"].items()})
    print("precision at conf 0.25:", res["precision_at_conf"])
    print("->", out / "test_metrics.json")


if __name__ == "__main__":
    main()
