#!/usr/bin/env python
"""
f4k_prepare.py -- AquaVision Stage-B data prep (Fish4Knowledge, 23 species)
===========================================================================
Report refs: Sec. 2 (23-class classifier), Sec. 5.2 (macro-F1 because classes are
imbalanced), Sec. 6.1 (augmentation), Sec. 6.2 (70/20/10 split), Sec. 6.3 (class-balanced).

What this does differently from the old write_classifier():
  1. Finds the class folders by itself (works with 'fish4knowedge' typo / nested folders,
     '01_Dascyllus_reticulatus' or 'fish_01' naming) and keeps the species names.
  2. Group-aware 70/20/10 split PER CLASS (same video/clip never in 2 splits) with a
     guaranteed minimum of val/test images for rare classes, so macro-F1 is computable.
  3. Balances ONLY the train split:
        - dominant classes are capped (ceil_mult x target / max_per_class)
        - rare classes are oversampled with classifier-safe underwater augmentation,
          but at most `max_aug_ratio` synthetic copies per real image (no 1 -> 500 blow-up)
        - classes that stay below target are flagged in the report instead of hidden
  4. Never augments val/test. Verifies there is no leakage.
  6. Same three tiers as the detector data (report Sec. 6.1): geometric + underwater photometric +
     occlusion/domain (small synthetic overlay, random erasing, cross-dataset LAB colour transfer toward
     Brackish/DeepFish frames via --ref_det). Cross-dataset fish COMPOSITING is not applied to crops
     (F4K crops have no clean fish cut-out mask) - state this in the report.
  5. Writes report artefacts: class_report.csv, class_balance.png, rare_class_preview.jpg,
     split_counts.json, class_weights.json, data_card.json.

Usage (Colab):
    %cd /content/aquavision
    !python f4k_prepare.py --f4k /content/data/fish4knowledge --out /content/work/cls
    !python f4k_prepare.py --f4k /content/data/fish4knowledge --dry_run     # counts only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
CLS_RE = re.compile(r"^(?:fish_?)?(\d{1,2})(?:[_\-](.*))?$", re.I)   # 01_Name | fish_01 | fish01
SPLITS = ("train", "val", "test")


def _sid(c):
    """species id for labels: 'fish_17' / '17_name' -> '17' (c[:2] would give 'fi' for fish_NN folders)"""
    return f"{int(re.search(r'\d+', c).group()):02d}"


def _rng():
    """Same seeded generator as the underwater_aug transforms -> `--seed` really makes the synthetic crops reproducible."""
    import underwater_aug as uw
    return uw._rng()


# ------------------------------------------------------------------ discovery
def find_root(root: Path) -> Path:
    """Descend (max 3 levels) to the folder that directly contains the class folders."""
    root = Path(root)
    for dp, dn, _ in os.walk(root):
        if len(Path(dp).relative_to(root).parts) > 3:
            dn[:] = []
            continue
        if sum(bool(CLS_RE.match(d)) for d in dn) >= 10:
            return Path(dp)
    return root


def norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def scan(root: Path):
    root = find_root(root)
    files, names = defaultdict(list), {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() not in IMG_EXT:
                continue
            rel = p.relative_to(root)
            if any("mask" in x.lower() for x in rel.parts[:-1]):
                continue
            for part in reversed(rel.parent.parts):
                m = CLS_RE.match(part)
                if m:
                    n = int(m.group(1))
                    files[n].append(p)
                    if m.group(2):
                        names.setdefault(n, norm(m.group(2)))
                    break
    classes = {}
    for n in sorted(files):
        label = f"{n:02d}_{names[n]}" if n in names else f"fish_{n:02d}"
        classes[label] = sorted(files[n])
    print(f"[f4k] root={root}")
    print(f"[f4k] {len(classes)} classes, {sum(len(v) for v in classes.values())} crops")
    if len(classes) != 23:
        print(f"  [warn] expected 23 species folders, found {len(classes)} -> check the path / unzip")
    return classes


def group_of(p: Path) -> str:
    """fish_<clip-id>_<n>.png -> clip-id (same clip must stay in one split)."""
    m = re.match(r"fish_(\d+)", p.stem, re.I)
    return m.group(1) if m else p.stem


# ------------------------------------------------------------------ split
def split_class(files, seed=42, ratios=(0.7, 0.2, 0.1), min_eval=2):
    """Per-class split. Returns ({split: [paths]}, mode). Guarantees val/test >= min_eval
    (when the class is big enough) and keeps clip groups intact whenever possible."""
    n = len(files)
    out = {s: [] for s in SPLITS}
    if n < 3:                                           # cannot evaluate; keep for training
        out["train"] = list(files)
        return out, "train-only"
    n_val = max(min_eval, round(ratios[1] * n))
    n_test = max(min_eval, round(ratios[2] * n))
    while n - n_val - n_test < max(1, n // 2) and (n_val > 1 or n_test > 1):
        if n_val >= n_test:
            n_val -= 1
        else:
            n_test -= 1
    tgt = {"train": n - n_val - n_test, "val": n_val, "test": n_test}

    groups = defaultdict(list)
    for f in files:
        groups[group_of(f)].append(f)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))            # big groups first (stable -> random ties)

    mode = "group"
    if len(keys) >= 10:
        cur = {s: 0 for s in SPLITS}
        for k in keys:
            s = max(SPLITS, key=lambda x: tgt[x] - cur[x])      # largest absolute deficit
            out[s].extend(groups[k])
            cur[s] += len(groups[k])
        if not out["val"] or not out["test"]:                   # one giant clip ate everything
            mode = "image"
    else:
        mode = "image"
    if mode == "image":                                         # few clips -> per-image split
        out = {s: [] for s in SPLITS}
        items = sorted(files)
        rng.shuffle(items)
        out["test"] = items[:tgt["test"]]
        out["val"] = items[tgt["test"]:tgt["test"] + tgt["val"]]
        out["train"] = items[tgt["test"] + tgt["val"]:]
    return out, mode


# ------------------------------------------------------------------ augmentation
class ScaleCropJitter:
    """Random crop of 75-100 % area, resized back (tiny crops -> zoom/scale tier of Sec. 6.1)."""
    def __init__(self, p=0.7, area=(0.75, 1.0)):
        self.p, self.area = p, area

    def __call__(self, image=None, **kw):
        rng = _rng()
        if rng.random() >= self.p:
            return {"image": image}
        h, w = image.shape[:2]
        f = np.sqrt(rng.uniform(*self.area))
        ch, cw = max(4, int(h * f)), max(4, int(w * f))
        y, x = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        return {"image": cv2.resize(image[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)}


class BrightnessContrast:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image=None, **kw):
        rng = _rng()
        if rng.random() >= self.p:
            return {"image": image}
        a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        return {"image": np.clip(image.astype(np.float32) * a + b, 0, 255).astype(np.uint8)}

def build_cls_aug(strength=1.0, ref_stats=None):
    """Classifier-SAFE underwater pipeline. Differences vs the detector pipeline:
    colour cast is removed/near-zero (body colour is a species cue), domain transfer is weak and rare,
    erasing is small, and the synthetic rock/coral overlay is tiny (would hide the whole fish on a 40 px crop)."""
    from underwater_aug import (FlipRotate, Sequential, UWColorCast, UWHaze, UWMotionBlur,
                                UWRandomErasing, UWStochasticCLAHE, UWTurbidity,
                                UWVignetteIllumination, UWDomainTransfer, UWSyntheticOverlay)
    s = lambda p: float(min(1.0, p * strength))
    return Sequential([
        FlipRotate(max_deg=15),
        ScaleCropJitter(),
        BrightnessContrast(),
        # Almost no cast: body colour is the species cue. Even the earlier "mild" setting (0.8-1.0 / 0-0.04)
        # moved yellow fish by ~20 deg of hue in a synthetic test. Detector pipeline keeps its strong cast.
        UWColorCast(red_atten=(0.92, 1.0), hue_pull=(0.0, 0.0), p=s(0.4)),
        UWTurbidity(sat_scale=(0.6, 0.95), p=s(0.4)),
        UWStochasticCLAHE(p=s(0.4)),
        UWHaze(density=(0.1, 0.2), p=s(0.12)),
        UWMotionBlur(kernel=(3, 5), p=s(0.1)),
        UWVignetteIllumination(p=s(0.12)),
        UWRandomErasing(n_holes=(1, 2), size=(0.05, 0.15), p=s(0.3)),
        # tier (iii): synthetic occlusion (small blob, rare) + cross-dataset colour transfer (weak, rarer)
        UWSyntheticOverlay(n_blobs=(1, 1), size=(0.03, 0.08), p=s(0.08)),
    ] + ([UWDomainTransfer(ref_stats=ref_stats, alpha=(0.1, 0.3), p=s(0.15))] if ref_stats else []))


def load_ref_stats(det_dir, n=300, seed=0):
    """LAB colour statistics of detector frames (Brackish/DeepFish originals) for cross-dataset domain transfer."""
    if not det_dir:
        return None
    imgs = [p for p in sorted((Path(det_dir) / "images" / "train").glob("*.jpg"))
            if not any(t in p.stem for t in ("__aug", "__ovl", "__comp"))]
    if not imgs:
        print("[f4k] no detector images found -> domain transfer skipped")
        return None
    from underwater_aug import compute_lab_stats
    st = compute_lab_stats(imgs, n=n, seed=seed)
    print(f"[f4k] domain-transfer reference stats from {len(st)} detector frames")
    return st


def make_aug(aug, src):
    for _ in range(5):                                  # make sure the copy really differs
        out = aug(src.copy())
        # really different AND not destroyed (all-black / all-white) -> species cues survive
        if out.shape == src.shape and not np.array_equal(out, src) and 8 < float(out.mean()) < 247:
            return out
    return out


# ------------------------------------------------------------------ main build
def build(a):
    classes = scan(Path(a.f4k))
    assert classes, "No class folders found - check --f4k path"
    rng = random.Random(a.seed)

    # ---- 1. split
    splits, modes = {}, {}
    for cls, files in classes.items():
        splits[cls], modes[cls] = split_class(files, a.seed, min_eval=a.min_eval)
    raw = {c: len(v) for c, v in classes.items()}
    tr0 = {c: len(splits[c]["train"]) for c in classes}

    # ---- 2. targets
    target = int(np.median(list(tr0.values()))) if a.target == "median" else int(a.target)
    target = max(1, min(target, a.max_per_class))
    ceiling = min(a.max_per_class, int(a.ceil_mult * target))
    print(f"[balance] target={target} per class | ceiling={ceiling} | max_aug_ratio={a.max_aug_ratio}")
    print("          (no cap at all: --ceil_mult 1000 --max_per_class 100000 ; --max_per_class alone does NOT remove the "
          "ceil_mult x target cap)")

    plan = {}
    for c in classes:
        n = tr0[c]
        keep = min(n, ceiling)
        synth = 0
        if keep < target and keep > 0:
            synth = min(target - keep, int(a.max_aug_ratio * keep))
        synth = max(synth, int(a.extra_aug * keep))   # every class also gets some augmented copies (PDF Sec 6.1)
        plan[c] = (keep, synth)

    dropped = sum(max(0, tr0[c] - plan[c][0]) for c in classes)
    print(f"[balance] real TRAIN crops thrown away by the cap: {dropped} of {sum(tr0.values())} "
          f"({100 * dropped / max(1, sum(tr0.values())):.0f} %)  <- keep this small for the head classes")
    img_mode = [c for c in classes if modes[c] == "image"]
    if img_mode:
        print(f"[note] {len(img_mode)} classes are split PER IMAGE (too few clips): {[_sid(c) for c in img_mode]} "
              f"-> clips may be shared between train and test; also report macro-F1 over the clip-split classes only")

    if a.dry_run:
        print_table(classes, raw, splits, plan, modes, target)
        return

    # ---- 3. write
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    import underwater_aug as uw
    uw.set_seed(a.seed)                                   # reproducible synthetic crops (same seed -> same files)
    aug = build_cls_aug(a.strength, load_ref_stats(a.ref_det))
    synth_log = {}
    for c in classes:
        for s in SPLITS:
            (out / s / c).mkdir(parents=True, exist_ok=True)
        for s in ("val", "test"):
            for p in splits[c][s]:
                shutil.copy2(p, out / s / c / p.name)
        tr = list(splits[c]["train"])
        keep, n_syn = plan[c]
        if len(tr) > keep:
            tr = rng.sample(tr, keep)                   # random undersample of dominant classes
        for p in tr:
            shutil.copy2(p, out / "train" / c / p.name)
        # synthetic copies: round-robin over real images so every source is used equally often
        order, made = [], []
        for k in range(n_syn):
            if not order:
                order = tr[:]
                rng.shuffle(order)
            src_p = order.pop()
            src = cv2.imread(str(src_p))
            if src is None:
                continue
            name = f"aug{k:05d}__{src_p.stem}.png"
            cv2.imwrite(str(out / "train" / c / name), make_aug(aug, src))
            made.append(name)
        synth_log[c] = made
        print(f"  [{c:34s}] raw={raw[c]:6d} train {tr0[c]:5d} -> {len(tr):5d} real + {len(made):5d} synth")

    # ---- 4. leakage check
    leaks = 0
    for c in classes:
        g = {s: {group_of(p) for p in splits[c][s]} for s in SPLITS}
        f = {s: {p.name for p in splits[c][s]} for s in SPLITS}
        for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
            leaks += len(f[x] & f[y])
            if modes[c] == "group":
                leaks += len(g[x] & g[y])
    print(f"[check] leakage between splits: {'NONE' if leaks == 0 else str(leaks) + ' !!'}")

    # ---- 5. artefacts
    rows = print_table(classes, raw, splits, plan, modes, target, synth_log)
    with open(out / "class_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "raw", "train_real", "train_synth", "train_final", "val", "test",
                    "synth_per_real", "split_mode", "flags"])
        w.writerows(rows)
    (out / "split_counts.json").write_text(json.dumps(
        {c: {s: len(splits[c][s]) for s in SPLITS} for c in classes}, indent=1))
    # effective-number class weights (for a custom loss / sampler; computed on REAL train counts)
    beta = 0.999
    wts = {c: (1 - beta) / (1 - beta ** max(1, plan[c][0])) for c in classes}
    z = sum(wts.values())
    (out / "class_weights.json").write_text(json.dumps({c: round(len(wts) * v / z, 4) for c, v in wts.items()}, indent=1))
    (out / "data_card.json").write_text(json.dumps(vars(a) | {"target": target, "ceiling": ceiling,
                                                              "n_classes": len(classes)}, indent=1))
    plot_balance(out, classes, tr0, plan)
    rare_preview(out, classes, raw, synth_log, extra_ids=[int(x) for x in a.preview_ids.split(",") if x.strip()])
    print(f"[classifier] done -> {out}")


def print_table(classes, raw, splits, plan, modes, target, synth_log=None):
    print(f"\n{'class':34s} {'raw':>6s} {'trReal':>6s} {'trSyn':>6s} {'trAll':>6s} {'val':>5s} {'test':>5s}  flags")
    rows = []
    for c in classes:
        keep, syn = plan[c]
        syn = len(synth_log[c]) if synth_log else syn
        fl = []
        if raw[c] < 30:
            fl.append("VERY-LOW(<30 raw)")
        elif raw[c] < 150:
            fl.append("LOW")
        if keep > 0 and syn / keep >= 5:
            fl.append(f"heavy-oversample x{syn / keep:.0f}")
        if keep + syn < 0.9 * target:
            fl.append("below-target")
        if len(splits[c]["test"]) < 5:
            fl.append("test<5:F1-noisy")
        if modes[c] != "group":
            fl.append(f"split={modes[c]}")
        if len(splits[c]["train"]) > keep:
            fl.append("capped")
        r = [c, raw[c], keep, syn, keep + syn, len(splits[c]["val"]), len(splits[c]["test"]),
             round(syn / keep, 1) if keep else 0, modes[c], " ".join(fl)]
        rows.append(r)
        print(f"{c:34s} {raw[c]:6d} {keep:6d} {syn:6d} {keep + syn:6d} {r[5]:5d} {r[6]:5d}  {' '.join(fl)}")
    print()
    return rows


def plot_balance(out, classes, tr0, plan):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(classes)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, [tr0[c] for c in names], 0.4, label="train before (real)")
    ax.bar(x + 0.2, [plan[c][0] + plan[c][1] for c in names], 0.4, label="train after (real+aug)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_sid(c) for c in names])
    ax.set_xlabel("species id")
    ax.set_ylabel("train crops (log)")
    ax.set_title("Fish4Knowledge: class balance before / after")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)


def rare_preview(out, classes, raw, synth_log, n_cls=4, n_aug=5, sz=128, extra_ids=()):
    """Preview grid: the rarest classes + any extra species ids (weak mid-size classes) -> LOOK at it:
    the fish must still be recognisable (strong haze/erasing can delete the species cues)."""
    rarest = sorted(classes, key=lambda c: raw[c])[:n_cls]
    for c in classes:
        cid = int(re.search(r"\d+", c).group())
        if cid in extra_ids and c not in rarest:
            rarest.append(c)
    rows = []
    for c in rarest:
        real = sorted((out / "train" / c).glob("*.png")) + sorted((out / "train" / c).glob("*.jpg"))
        real = [p for p in real if not p.name.startswith("aug")]
        if not real:
            continue
        tiles = [cv2.imread(str(real[0]))] + [cv2.imread(str(out / "train" / c / n)) for n in synth_log[c][:n_aug]]
        row = []
        for t in tiles:
            if t is None:
                t = np.zeros((sz, sz, 3), np.uint8)
            h, w = t.shape[:2]
            f = sz / max(h, w)
            t = cv2.resize(t, (max(1, int(w * f)), max(1, int(h * f))))
            canvas = np.zeros((sz, sz, 3), np.uint8)
            canvas[:t.shape[0], :t.shape[1]] = t
            row.append(canvas)
        row += [np.zeros((sz, sz, 3), np.uint8)] * (n_aug + 1 - len(row))
        img = np.hstack(row)
        cv2.putText(img, c, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(img)
    if rows:
        cv2.imwrite(str(out / "rare_class_preview.jpg"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f4k", required=True, help="folder containing the 23 species folders (nested ok)")
    ap.add_argument("--out", default="/content/work/cls")
    ap.add_argument("--target", default="median", help="'median' or an int: train crops per class after balancing")
    ap.add_argument("--max_per_class", type=int, default=1500, help="hard cap on train crops per class")
    ap.add_argument("--ceil_mult", type=float, default=3.0, help="dominant classes cut to ceil_mult x target "
                    "(ceiling = min(max_per_class, ceil_mult x target)); default 3 x median is VERY aggressive")
    ap.add_argument("--max_aug_ratio", type=float, default=12, help="max synthetic copies per real train image")
    ap.add_argument("--min_eval", type=int, default=2, help="min val and test images per class")
    ap.add_argument("--strength", type=float, default=1.0, help="scales augmentation probabilities")
    ap.add_argument("--ref_det", default="/content/work/det", help="detector folder whose frames give the domain-transfer colour stats ('' = off)")
    ap.add_argument("--extra_aug", type=float, default=0.0, help="extra synthetic copies per REAL train image for ALL classes (0.2 = +20%%)")
    ap.add_argument("--preview_ids", default="", help="extra species ids for rare_class_preview.jpg, e.g. 8,13,14")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true", help="only print the plan, write nothing")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
