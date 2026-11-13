#!/usr/bin/env python
"""
turbid_split.py -- held-out clean-vs-turbid split (report Sec. 6.1, last sentence)
==================================================================================
"Value is verified by per-category ablation and a held-out clean-vs-turbid split."
Brackish/DeepFish carry no visibility label in this copy, so turbidity is MEASURED per image:
    dark-channel mean (haze veil)  UP,   RMS contrast DOWN,   mean saturation DOWN,   sharpness DOWN
Each cue is z-scored against that dataset's TRAIN originals, averaged -> turbidity score.
Threshold = quantile `--q` (default median) of the dataset's train scores, so each dataset
is split into its own clearer / murkier half (no dataset-identity shortcut). `--global_thr`
standardises over all datasets together instead.

Only ORIGINAL images are scored (augmented __aug/__ovl/__comp copies are ignored).
Writes into <det>/lists:
    test_clean.txt  test_turbid.txt  val_clean.txt  val_turbid.txt  train_clean.txt
    data_test_clean.yaml / data_test_turbid.yaml   (val+test point to that subset -> model.val())
    data_trainclean.yaml  (train on CLEAN only, val=val_clean, test=test_turbid -> held-out robustness)
and <det>/turbidity.csv (dataset, split, file, score, label).
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TAGS = ("__aug", "__ovl", "__comp", "__abl")


def cues(path):
    im = cv2.imread(str(path))
    if im is None:
        return None
    f = 320.0 / max(im.shape[:2])
    if f < 1:
        im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    dark = cv2.erode(im.min(axis=2), np.ones((9, 9), np.uint8)).mean()
    return np.array([dark, gray.std(), hsv[..., 1].mean(), cv2.Laplacian(gray, cv2.CV_64F).var()], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--q", type=float, default=0.5, help="train-score quantile separating clean/turbid")
    ap.add_argument("--global_thr", action="store_true", help="one threshold over all datasets")
    a = ap.parse_args()
    det = Path(a.det)
    SIGN = np.array([+1, -1, -1, -1], np.float64)       # +dark, -contrast, -saturation, -sharpness

    items = defaultdict(list)                            # dataset -> [(split, path, cue)]
    for split in ("train", "val", "test"):
        for p in sorted((det / "images" / split).glob("*.jpg")):
            if any(t in p.stem for t in TAGS):
                continue
            c = cues(p)
            if c is not None:
                items[p.stem.split("__")[0]].append((split, p, c))
    assert items, "no images found - run prepare_data.py first"

    groups = {"ALL": [x for v in items.values() for x in v]} if a.global_thr else items
    rows, thr_info = [], {}
    for g, lst in groups.items():
        tr = np.stack([c for s, _, c in lst if s == "train"])
        mu, sd = tr.mean(0), tr.std(0) + 1e-6
        score = lambda c: float((((c - mu) / sd) * SIGN).mean())
        thr = float(np.quantile([score(c) for s, _, c in lst if s == "train"], a.q))
        thr_info[g] = round(thr, 3)
        for s, p, c in lst:
            sc = score(c)
            ds = p.stem.split("__")[0]
            rows.append([ds, s, p.name, round(sc, 4), "turbid" if sc > thr else "clean"])
    print("thresholds:", thr_info)

    with open(det / "turbidity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "file", "score", "label"])
        w.writerows(rows)
    lst_dir = det / "lists"
    lst_dir.mkdir(exist_ok=True)
    out = defaultdict(list)
    for ds, s, fn, sc, lab in rows:
        out[(s, lab)].append(str((det / "images" / s / fn).resolve()))
    for (s, lab), v in out.items():
        (lst_dir / f"{s}_{lab}.txt").write_text("\n".join(v) + "\n")
    y = (det / "data.yaml").read_text()

    def with_(**kw):
        t = y
        for k, v in kw.items():
            t = re.sub(rf"(?m)^{k}:.*$", f"{k}: {(lst_dir / v).resolve()}", t)
        return t

    (det / "data_test_clean.yaml").write_text(with_(val="test_clean.txt", test="test_clean.txt"))
    (det / "data_test_turbid.yaml").write_text(with_(val="test_turbid.txt", test="test_turbid.txt"))
    (det / "data_trainclean.yaml").write_text(with_(train="train_clean.txt", val="val_clean.txt", test="test_turbid.txt"))

    import collections
    cnt = collections.Counter((r[0], r[1], r[4]) for r in rows)
    print(f"{'dataset':10s} {'split':6s} {'clean':>7s} {'turbid':>7s}")
    for ds in sorted({r[0] for r in rows}):
        for s in ("train", "val", "test"):
            print(f"{ds:10s} {s:6s} {cnt[(ds, s, 'clean')]:7d} {cnt[(ds, s, 'turbid')]:7d}")
    print("wrote lists/*, data_test_clean.yaml, data_test_turbid.yaml, data_trainclean.yaml, turbidity.csv")


if __name__ == "__main__":
    main()
