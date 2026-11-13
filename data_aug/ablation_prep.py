#!/usr/bin/env python
"""
ablation_prep.py -- per-category augmentation ablation sets (report Sec. 6.1, last sentence)
==========================================================================================
For ONE fixed random subset of labelled TRAIN originals (so every arm sees the same sources),
make one cached copy per image with exactly ONE augmentation category switched on (p=1):

    colorcast  turbidity  clahe  erasing  haze  motionblur  vignette   (tier ii / iii-erasing)
    overlay                                                              (tier iii synthetic overlay)
    domain                                                               (tier iii cross-dataset colour transfer)
    composite                                                            (tier iii cross-dataset fish paste)

Arms (each = a train image list + a yaml; val/test are the normal untouched ones):
    abl_baseline    originals only
    abl_control_dup originals + the SAME subset duplicated (control: gain is not just 'more iterations')
    abl_<category>  originals + that category's copies
    abl_all         originals + all categories (the combined effect)

Copies live in <out>/<category>/{images,labels}/train/<stem>__abl.jpg (labels copied; composite appends a polygon).
Train each arm with the same short schedule (ablation_run.py) and compare mAP / IoU on
data_test_clean.yaml vs data_test_turbid.yaml (from turbid_split.py).

    python ablation_prep.py --det /content/work/det --out /content/work/ablation --frac 0.15 --max_per_cat 1500
"""
import argparse
import csv
import random
import re
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import underwater_aug as uw
from underwater_aug import (Sequential, UWColorCast, UWDomainTransfer, UWHaze, UWMotionBlur,
                            UWRandomErasing, UWStochasticCLAHE, UWSyntheticOverlay, UWTurbidity,
                            UWVignetteIllumination, compute_lab_stats)
import augment_offline as ao

SINGLE = {
    "colorcast": lambda: UWColorCast(p=1.0),
    "turbidity": lambda: UWTurbidity(p=1.0),
    "clahe": lambda: UWStochasticCLAHE(p=1.0),
    "erasing": lambda: UWRandomErasing(p=1.0),
    "haze": lambda: UWHaze(p=1.0),
    "motionblur": lambda: UWMotionBlur(p=1.0),
    "vignette": lambda: UWVignetteIllumination(p=1.0),
    "overlay": lambda: UWSyntheticOverlay(p=1.0),
}


def read(p):
    im = cv2.imread(str(p))
    return im


def shrink(im, max_side):
    if max_side and max(im.shape[:2]) > max_side:
        f = max_side / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * f), int(im.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return im


def seed_for(a, cat, stem):
    s = zlib.crc32(f"{a.seed}:{cat}:{stem}".encode()) & 0x7FFFFFFF
    random.seed(s)
    np.random.seed(s)
    uw.set_seed(s)
    return np.random.default_rng(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--out", default="/content/work/ablation")
    ap.add_argument("--frac", type=float, default=0.15, help="fraction of labelled train originals used as sources")
    ap.add_argument("--max_per_cat", type=int, default=1500)
    ap.add_argument("--max_side", type=int, default=960)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    det, out = Path(a.det), Path(a.out)
    originals = [p for p in sorted((det / "images" / "train").glob("*.jpg")) if not ao.is_copy(p)]
    labelled = [p for p in originals if (det / "labels" / "train" / f"{p.stem}.txt").read_text().strip()]
    n = max(1, min(int(a.frac * len(labelled)), a.max_per_cat))
    subset = random.Random(a.seed).sample(labelled, min(n, len(labelled)))
    print(f"{len(originals)} originals, {len(labelled)} labelled -> {len(subset)} source images per category")
    by_ds = defaultdict(list)
    for p in originals:
        by_ds[p.stem.split("__")[0]].append(p)
    print("sources per dataset:", {k: sum(1 for p in subset if p.stem.startswith(k + "__")) for k in by_ds})

    stats = {k: compute_lab_stats(v) for k, v in by_ds.items()} if len(by_ds) >= 2 else {}
    donors = ao.build_donors(det, originals, seed=a.seed) if len(by_ds) >= 2 else {}
    can_comp = len([k for k, v in donors.items() if v]) >= 2
    cats = dict(SINGLE)
    if stats:
        cats["domain"] = None
    else:
        print("[skip] domain: needs >= 2 datasets")
    if can_comp:
        cats["composite"] = None
    else:
        print("[skip] composite: fewer than 2 datasets with mask polygons (build with --pseudo_masks)")

    orig_paths = [str(p.resolve()) for p in originals]
    arms, y = {}, (det / "data.yaml").read_text()
    made = {}
    for cat in cats:
        idir, ldir = out / cat / "images" / "train", out / cat / "labels" / "train"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        files = []
        for p in subset:
            ds = p.stem.split("__")[0]
            im = read(p)
            if im is None:
                continue
            im = shrink(im, a.max_side)
            rng = seed_for(a, cat, p.stem)
            lab = (det / "labels" / "train" / f"{p.stem}.txt").read_text()
            new_lab = lab
            if cat in SINGLE:
                t = SINGLE[cat]()
                res = t(force_apply=True, image=im.copy())["image"]
            elif cat == "domain":
                ref = [s for k, v in stats.items() if k != ds for s in v]
                res = UWDomainTransfer(ref_stats=ref, alpha=(0.7, 1.0), p=1.0)(force_apply=True, image=im.copy())["image"]
            else:  # composite
                other = [d for d in donors if d != ds and donors[d]]
                if not other:
                    continue
                dp, poly = random.choice(donors[random.choice(other)])
                dimg = read(dp)
                r = ao.paste_fish(im, dimg, poly, rng) if dimg is not None else None
                if r is None:
                    continue
                res, newp = r
                new_lab = lab.rstrip("\n") + "\n" + "0 " + " ".join(f"{v:.6f}" for v in newp.reshape(-1)) + "\n"
            name = f"{p.stem}__abl"
            cv2.imwrite(str(idir / f"{name}.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (ldir / f"{name}.txt").write_text(new_lab)
            files.append(str((idir / f"{name}.jpg").resolve()))
        made[cat] = files
        arms[f"abl_{cat}"] = orig_paths + files
        print(f"[{cat:10s}] {len(files)} copies")

    arms["abl_baseline"] = orig_paths
    arms["abl_control_dup"] = orig_paths + [str(p.resolve()) for p in subset]
    arms["abl_all"] = orig_paths + [f for v in made.values() for f in v]

    (out / "lists").mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, lst in arms.items():
        lp = out / "lists" / f"{arm}.txt"
        lp.write_text("\n".join(lst) + "\n")
        yp = out / f"data_{arm}.yaml"
        yp.write_text(re.sub(r"(?m)^train:.*$", f"train: {lp.resolve()}", y))
        rows.append([arm, len(lst), str(yp)])
    with open(out / "ablation_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "train_images", "yaml"])
        w.writerows(rows)
    print(f"\n{len(arms)} arms -> {out}/ablation_index.csv")
    for r in rows:
        print(f"  {r[0]:18s} {r[1]:7d} imgs")


if __name__ == "__main__":
    main()
