#!/usr/bin/env python
"""
prepare_data.py -- AquaVision detector data prep (report Sec. 2 / 6, phase 1)
=============================================================================
Brackish + DeepFish (+ OZFish if present) -> ONE single-class ("fish") YOLO-seg dataset,
70/20/10 split PER DATASET, split by video/sequence group (no frame leakage).

  python prepare_data.py detector --brackish B --deepfish D [--ozfish O] --out /content/work/det
  python prepare_data.py viz --det /content/work/det --n 12
  python prepare_data.py classifier --f4k F --out /content/work/cls      (forwards to f4k_prepare.py)

Handles what the real data looks like:
  * Brackish label ids are 1-based (1 fish, 2 small_fish, 3 crab, 4 shrimp, 5 jellyfish,
    6 starfish) and boxes are in PIXELS -> read class names from the COCO json, keep fish +
    small_fish, normalise by the real image size.
  * DeepFish copy with YOLO box txt files (no pixel masks) -> rectangle polygons, or GrabCut
    pseudo-masks with --pseudo_masks. If a masks/ folder exists, real masks are used.
  * Output folder is wiped first, so no stale (wrongly labelled) augmented copies survive.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import shutil

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FISH_NAMES = {"fish", "small_fish", "smallfish"}
SPLITS = (("train", 0.70), ("val", 0.20), ("test", 0.10))
SKIP_TXT = {"classes.txt", "labels.txt", "readme.txt", "obj.names", "train.txt", "valid.txt",
            "val.txt", "test.txt"}


@dataclass
class Instance:
    box: tuple | None = None          # (cx, cy, w, h)  normalised OR pixels
    poly: np.ndarray | None = None    # (N, 2)           normalised OR pixels


@dataclass
class Sample:
    dataset: str
    img: Path
    instances: list = field(default_factory=list)
    group: str = ""


# ------------------------------------------------------------------ helpers
def list_images(root: Path):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


def norm_name(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def group_key(stem: str) -> str:
    """'vid12_frame_000345' -> 'vid12' (strip the trailing frame counter)."""
    g = re.sub(r"[_\-\s]?(frame|img|image|f)?[_\-\s]?\d+$", "", stem, flags=re.I)
    return g or stem


def group_split(items, key, ratios=SPLITS, seed=42, quiet=False):
    """Group-aware split that fills the split with the biggest remaining deficit
    (old version could leave test=0 when a few large groups overshoot the 90% mark)."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    keys = sorted(groups)
    if len(keys) < 10:
        if not quiet:
            print(f"  [warn] only {len(keys)} groups found; falling back to per-image split")
        groups = {i: [it] for i, it in enumerate(items)}
        keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))          # big groups first, random tie-break
    total = sum(len(v) for v in groups.values())
    tgt = {n: r * total for n, r in ratios}
    cur = {n: 0 for n, _ in ratios}
    out = {n: [] for n, _ in ratios}
    for k in keys:
        n = max(tgt, key=lambda x: tgt[x] - cur[x])
        out[n].extend(groups[k])
        cur[n] += len(groups[k])
    return out


def poly_from_box(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def grabcut_poly(img, box, iters=3):
    """Pseudo-mask for a normalised (cx,cy,w,h) box via GrabCut -> normalised polygon or None."""
    H, W = img.shape[:2]
    cx, cy, w, h = box
    x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
    x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    mx, my = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
    X1, Y1, X2, Y2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
    crop = img[Y1:Y2, X1:X2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (x1 - X1, y1 - Y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = ((mask == 1) | (mask == 3)).astype(np.uint8)
    inbox = np.zeros_like(fg)
    inbox[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 1
    fg *= inbox
    frac = fg.sum() / max(1, rect[2] * rect[3])
    if frac < 0.2 or frac > 0.95:
        return None
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    c = c.astype(np.float32) + np.array([X1, Y1], np.float32)
    return c / np.array([W, H], np.float32)


def parse_yolo_txt(path: Path, keep=None):
    """Lines 'cls cx cy w h' (box) or 'cls x1 y1 x2 y2 ...' (polygon). keep=None -> all classes."""
    inst = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            v = list(map(float, parts[1:]))
        except ValueError:
            continue
        if keep is not None and c not in keep:
            continue
        if len(v) == 4:
            inst.append(Instance(box=tuple(v)))
        elif len(v) >= 6 and len(v) % 2 == 0:
            inst.append(Instance(poly=np.array(v, np.float32).reshape(-1, 2)))
    return inst


# ------------------------------------------------------------------ Brackish
def brackish_fish_ids(root: Path):
    """Class ids that mean 'fish' (fish + small_fish) from the COCO json (1-based), else fallbacks."""
    for jf in sorted(Path(root).rglob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("categories"):
            cats = {int(c["id"]): norm_name(c["name"]) for c in d["categories"]}
            ids = {i for i, n in cats.items() if n in FISH_NAMES}
            if ids:
                print(f"[brackish] class ids from {jf.name}: {cats} -> fish ids {sorted(ids)}")
                return ids
    for f in Path(root).rglob("classes.txt"):
        names = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        ids = {i for i, n in enumerate(names) if norm_name(n) in FISH_NAMES}
        if ids:
            print(f"[brackish] class ids from classes.txt: {names} -> fish ids {sorted(ids)}")
            return ids
    print("[brackish] no class-name file found -> assuming Brackish default 1-based ids {1,2}")
    return {1, 2}


def load_brackish(root: Path):
    root = Path(root)
    fish_ids = brackish_fish_ids(root)
    imgs = [p for p in list_images(root) if "brackishmot" not in str(p).lower()]
    all_txt = [t for t in root.rglob("*.txt") if t.name.lower() not in SKIP_TXT
               and "brackishmot" not in str(t).lower()]
    yolo_txt = [t for t in all_txt if "annotations_yolo" in str(t).lower()]
    txts = {t.stem: t for t in (yolo_txt or all_txt)}
    samples, nolabel = [], 0
    for p in imgs:
        t = txts.get(p.stem)
        if t is None:
            nolabel += 1
            continue
        samples.append(Sample("brackish", p, parse_yolo_txt(t, fish_ids), group_key(p.stem)))
    bg = sum(1 for s in samples if not s.instances)
    print(f"[brackish] {len(samples)} images, {sum(len(s.instances) for s in samples)} fish instances, "
          f"{bg} background frames, {nolabel} images without label file (skipped)")
    return samples


# ------------------------------------------------------------------ OZFish (optional)
def load_ozfish(root: Path):
    root = Path(root)
    imgs = list_images(root)
    txts = {t.stem: t for t in root.rglob("*.txt") if t.name.lower() not in SKIP_TXT}
    samples = []
    for p in imgs:
        t = txts.get(p.stem)
        if t is not None:
            samples.append(Sample("ozfish", p, parse_yolo_txt(t), group_key(p.stem)))
    print(f"[ozfish] {len(samples)} images, {sum(len(s.instances) for s in samples)} boxes")
    return samples


# ------------------------------------------------------------------ DeepFish
def mask_to_polys(mask: np.ndarray, min_area_frac=1e-4, eps=0.002):
    H, W = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area_frac * H * W:
            continue
        c = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(c) >= 3:
            out.append(c.astype(np.float32) / np.array([W, H], np.float32))
    return out


def load_deepfish(root: Path):
    root = Path(root)
    all_imgs = list_images(root)
    is_mask = lambda p: any(x.lower() in ("masks", "mask") for x in p.parts)
    masks = {re.sub(r"_?mask$", "", p.stem, flags=re.I): p for p in all_imgs if is_mask(p)}
    samples = []
    if masks:                                                  # real pixel masks
        for p in all_imgs:
            if is_mask(p) or p.stem not in masks:
                continue
            m = cv2.imread(str(masks[p.stem]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            samples.append(Sample("deepfish", p, [Instance(poly=q) for q in mask_to_polys(m)], group_key(p.stem)))
        print(f"[deepfish] mode=masks: {len(samples)} images, {sum(len(s.instances) for s in samples)} polygons")
        return samples
    txts = {t.stem: t for t in root.rglob("*.txt") if t.name.lower() not in SKIP_TXT}
    for p in all_imgs:
        t = txts.get(p.stem)
        if t is None:
            continue
        samples.append(Sample("deepfish", p, parse_yolo_txt(t), group_key(p.stem)))
    bg = sum(1 for s in samples if not s.instances)
    print(f"[deepfish] mode=yolo-box: {len(samples)} images, {sum(len(s.instances) for s in samples)} instances "
          f"({bg} background frames)")
    print("  [note] no pixel masks in this copy -> rectangle polygons (use --pseudo_masks for GrabCut masks)")
    return samples


# ------------------------------------------------------------------ writer
def to_norm_poly(ins, W, H, img, pseudo):
    """-> (normalised polygon (N,2), kind) or None."""
    if ins.poly is not None:
        p = np.array(ins.poly, np.float32)
        if p.max() > 1.5:
            p = p / np.array([W, H], np.float32)
        return np.clip(p, 0, 1), "mask"
    cx, cy, w, h = ins.box
    if max(ins.box) > 1.5:                                     # pixel units
        cx, w, cy, h = cx / W, w / W, cy / H, h / H
    x1, y1, x2, y2 = (max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2))
    if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
        return None
    if pseudo:
        poly = grabcut_poly(img, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        if poly is not None:
            return np.clip(poly, 0, 1), "pseudo"
    return poly_from_box(x1, y1, x2, y2), "box"


def fmt_poly(p):
    return "0 " + " ".join(f"{v:.6f}" for v in np.clip(p, 0, 1).reshape(-1))


def write_detector(samples, out: Path, max_side=1280, pseudo=False, seed=42, bg_ratio=1.0):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)                                     # never keep stale copies
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s.dataset].append(s)
    rows, stats = [], defaultdict(int)
    raw_stats = {}                                             # true counts BEFORE the empty-frame cap
    for ds, items in by_ds.items():
        pos = [s for s in items if s.instances]
        neg = [s for s in items if not s.instances]
        raw_stats[ds] = {"raw_images": len(items), "raw_labelled": len(pos), "raw_empty": len(neg),
                         "raw_instances": sum(len(s.instances) for s in items)}
        if bg_ratio is not None and len(neg) > bg_ratio * len(pos):
            neg = random.Random(seed).sample(neg, int(bg_ratio * len(pos)))
            print(f"[{ds}] background frames capped to {len(neg)} (bg_ratio={bg_ratio})")
        items = pos + neg
        sp = group_split(items, key=lambda s: s.group, seed=seed)
        print(f"[split:{ds}] " + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))
        for split, lst in sp.items():
            for s in lst:
                img = cv2.imread(str(s.img))
                if img is None:
                    stats["unreadable"] += 1
                    continue
                H, W = img.shape[:2]
                lines, kinds = [], set()
                for ins in s.instances:
                    r = to_norm_poly(ins, W, H, img, pseudo)
                    if r is None or len(r[0]) < 3:
                        continue
                    lines.append(fmt_poly(r[0]))
                    kinds.add(r[1])
                if max_side and max(H, W) > max_side:
                    f = max_side / max(H, W)
                    img = cv2.resize(img, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
                name = f"{ds}__{s.img.stem}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                rows.append([ds, split, f"{name}.jpg", len(lines), "+".join(sorted(kinds)) or "background"])
                stats[f"{split}_images"] += 1
                stats[f"{split}_instances"] += len(lines)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "file", "n_instances", "label_kind"])
        w.writerows(rows)
    (out / "raw_counts.json").write_text(json.dumps(raw_stats, indent=1))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: fish\n")
    print("[detector] done:", dict(stats))
    print(f"[detector] data.yaml -> {out / 'data.yaml'}")


# ------------------------------------------------------------------ viz
def viz(det: Path, n=12, seed=0):
    det = Path(det)
    imgs = [p for p in sorted((det / "images" / "train").glob("*.jpg")) if "__aug" not in p.stem]
    with_lab = [p for p in imgs if (det / "labels" / "train" / f"{p.stem}.txt").read_text().strip()]
    pick = random.Random(seed).sample(with_lab, min(n, len(with_lab)))
    Path("viz").mkdir(exist_ok=True)
    for i, p in enumerate(pick):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        for line in (det / "labels" / "train" / f"{p.stem}.txt").read_text().splitlines():
            v = list(map(float, line.split()[1:]))
            pts = (np.array(v, np.float32).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.polylines(im, [pts], True, (0, 255, 0), 2)
        cv2.imwrite(f"viz/{i:02d}_{p.stem[:40]}.jpg", im)
    print(f"saved {len(pick)} overlays to viz/")


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detector")
    d.add_argument("--brackish")
    d.add_argument("--ozfish")
    d.add_argument("--deepfish")
    d.add_argument("--out", default="work/det")
    d.add_argument("--max_side", type=int, default=1280)
    d.add_argument("--pseudo_masks", action="store_true")
    d.add_argument("--bg_ratio", type=float, default=1.0, help="max background frames per dataset = ratio x labelled frames")
    d.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("classifier")
    c.add_argument("--f4k", required=True)
    c.add_argument("--out", default="work/cls")
    v = sub.add_parser("viz")
    v.add_argument("--det", required=True)
    v.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "detector":
        samples = []
        if a.brackish:
            samples += load_brackish(a.brackish)
        if a.ozfish:
            samples += load_ozfish(a.ozfish)
        if a.deepfish:
            samples += load_deepfish(a.deepfish)
        assert samples, "no data loaded - check the dataset paths"
        write_detector(samples, Path(a.out), a.max_side, a.pseudo_masks, a.seed, a.bg_ratio)
    elif a.cmd == "viz":
        viz(Path(a.det), a.n)
    else:
        here = Path(__file__).resolve().parent / "f4k_prepare.py"
        sys.exit(subprocess.run([sys.executable, str(here), "--f4k", a.f4k, "--out", a.out]).returncode)


if __name__ == "__main__":
    main()
