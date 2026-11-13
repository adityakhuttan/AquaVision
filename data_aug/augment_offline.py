#!/usr/bin/env python
"""
augment_offline.py -- cache underwater-augmented copies of the TRAIN split (report Sec. 6.1)
===========================================================================================
Three kinds of cached copies (val/test are NEVER touched):

  __aug{k}   tier (ii) photometric + tier (iii) random erasing + cross-dataset LAB colour
             transfer ("domain mixing").  Labels are copied unchanged.
  __ovl0     same as above PLUS a synthetic rock/coral overlay (occlusion).  Labels unchanged.
  __comp0    cross-dataset COMPOSITING: a fish cut out (polygon mask) of an image from ANOTHER
             dataset is pasted into this image.  New polygon is appended to the labels.
             Needs real/pseudo masks (polygons with >4 vertices) -> run prepare_data with
             --pseudo_masks, otherwise this step is skipped with a message.

Tier (i) geometric aug + mosaic/mixup stay ONLINE in Ultralytics (see train_aug_config.py).

Report: "overlays ... disabled in the final epochs" -> two image lists are written:
    lists/train_full.txt  (everything)          -> data_full.yaml   (phase 1)
    lists/train_late.txt  (no __ovl / __comp)   -> data_late.yaml   (last ~15 epochs)

RESUMABLE + PARALLEL
    * every copy has a fixed name (<stem>__aug0 ...) and is written atomically (image first,
      label last), so a copy counts as done only when both files exist.  If Colab stops or you
      press Stop, just run the SAME command again: finished copies are skipped.
    * work is spread over --workers CPU processes (0 = all cores).
    * each source image gets its own seed (from --seed + file name), so results do not depend
      on the number of workers or on how many times the run was resumed.
    * --max_side N downsizes the SOURCE before augmenting (labels are normalised, so they stay
      valid) -> much faster and smaller cache.  Training is at 640 px, so 960 loses nothing.

    python augment_offline.py --det /content/work/det --copies 2 --domain_mix \
           --overlay_frac 0.1 --composite_frac 0.1 --max_side 960 --workers 0
"""
import argparse
import multiprocessing as mp
import os
import random
import re
import shutil
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import underwater_aug as uw
from underwater_aug import Sequential, UWSyntheticOverlay, build_underwater_transforms, compute_lab_stats

TAGS = ("__aug", "__ovl", "__comp")
G = {}          # config shared with forked worker processes


def is_copy(p: Path) -> bool:
    return any(t in p.stem for t in TAGS)


def clean(det: Path):
    n = 0
    for p in list((det / "images" / "train").glob("*.jpg")):
        if is_copy(p):
            p.unlink()
            (det / "labels" / "train" / f"{p.stem}.txt").unlink(missing_ok=True)
            n += 1
    shutil.rmtree(det / "lists", ignore_errors=True)
    print(f"cleaned {n} old cached copies")


def changed(pipe, im, tries=4):
    out = pipe(im.copy())
    for _ in range(tries):
        if not np.array_equal(out, im):
            break
        out = pipe(im.copy())
    return out


# ------------------------------------------------------------------ compositing
def read_polys(lab: Path):
    polys = []
    for line in lab.read_text().splitlines():
        v = line.split()
        if len(v) >= 7:
            polys.append(np.array(list(map(float, v[1:])), np.float32).reshape(-1, 2))
    return polys


def build_donors(det: Path, imgs, per_ds=1500, seed=0):
    """Donor fish = polygons with >4 vertices (real or GrabCut masks, not plain rectangles)."""
    donors = defaultdict(list)
    for p in imgs:
        ds = p.stem.split("__")[0]
        lab = det / "labels" / "train" / f"{p.stem}.txt"
        for poly in read_polys(lab):
            if len(poly) <= 4:
                continue
            (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
            if 0.002 <= (x1 - x0) * (y1 - y0) <= 0.4:
                donors[ds].append((p, poly))
    rs = random.Random(seed)
    for ds in donors:
        rs.shuffle(donors[ds])
        donors[ds] = donors[ds][:per_ds]
    return donors


def paste_fish(target, donor_img, poly, rng):
    """Cut `poly` out of donor_img, scale/flip/colour-match, blend into target.
    Returns (image, new normalised polygon) or None."""
    H, W = target.shape[:2]
    h, w = donor_img.shape[:2]
    pts = poly * np.array([w, h], np.float32)
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    cw, ch = x1 - x0, y1 - y0
    if cw < 8 or ch < 8:
        return None
    crop = donor_img[y0:y1, x0:x1].copy()
    pc = pts - np.array([x0, y0], np.float32)
    mask = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(mask, [pc.astype(np.int32)], 255)
    if rng.random() < 0.5:                                      # horizontal flip
        crop, mask = crop[:, ::-1].copy(), mask[:, ::-1].copy()
        pc[:, 0] = cw - pc[:, 0]
    s = rng.uniform(0.6, 1.4) * W / w                           # relative to image width
    s = min(s, 0.6 * W / cw, 0.6 * H / ch)
    nw, nh = max(8, int(cw * s)), max(8, int(ch * s))
    if nw >= W or nh >= H:
        return None
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # colour-match the cut-out to the target scene (Reinhard in LAB, partial)
    lt = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    lc = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = mask > 127
    if inside.sum() > 10:
        mc, sc = lc[inside].mean(0), lc[inside].std(0) + 1e-6
        mt, st = lt.reshape(-1, 3).mean(0), lt.reshape(-1, 3).std(0) + 1e-6
        a = rng.uniform(0.4, 0.8)
        lc = a * ((lc - mc) / sc * st + mt) + (1 - a) * lc
        crop = cv2.cvtColor(np.clip(lc, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    ox, oy = int(rng.integers(0, W - nw + 1)), int(rng.integers(0, H - nh + 1))
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    out = target.copy()
    reg = out[oy:oy + nh, ox:ox + nw].astype(np.float32)
    out[oy:oy + nh, ox:ox + nw] = np.clip(crop * alpha + reg * (1 - alpha), 0, 255).astype(np.uint8)
    newp = pc * np.array([nw / cw, nh / ch], np.float32) + np.array([ox, oy], np.float32)
    return out, np.clip(newp / np.array([W, H], np.float32), 0, 1)


# ------------------------------------------------------------------ one source image
def _done(img_dir: Path, lab_dir: Path, name: str) -> bool:
    return (lab_dir / f"{name}.txt").exists() and (img_dir / f"{name}.jpg").exists()


def _save(img_dir: Path, lab_dir: Path, name: str, img, text: str):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False
    tmp = img_dir / f"{name}.part"                     # not *.jpg -> never picked up half-written
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, img_dir / f"{name}.jpg")
    (lab_dir / f"{name}.txt").write_text(text)         # label LAST = "this copy is complete"
    return True


def process(task):
    i, ps = task
    a, det = G["a"], G["det"]
    cv2.setNumThreads(1)
    p = Path(ps)
    ds = p.stem.split("__")[0]
    img_dir, lab_dir = det / "images" / "train", det / "labels" / "train"

    want = [("aug", k) for k in range(a.copies)]
    if i in G["ovl_set"]:
        want.append(("ovl", 0))
    if i in G["comp_set"]:
        want.append(("comp", 0))
    todo = [(t, k) for t, k in want if not _done(img_dir, lab_dir, f"{p.stem}__{t}{k}")]
    res = {"aug": 0, "ovl": 0, "comp": 0}
    if not todo:
        return res

    seed = zlib.crc32(f"{a.seed}:{p.stem}".encode()) & 0x7FFFFFFF      # per-image, worker-independent
    random.seed(seed)
    np.random.seed(seed)
    uw.set_seed(seed)
    rng = np.random.default_rng(seed)

    im = cv2.imread(str(p))
    if im is None:
        return res
    if a.max_side and max(im.shape[:2]) > a.max_side:
        f = a.max_side / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * f), int(im.shape[0] * f)), interpolation=cv2.INTER_AREA)
    lab_txt = (lab_dir / f"{p.stem}.txt").read_text()

    stats = G["stats_by_ds"]
    ref = ([s for k, v in stats.items() if k != ds for s in v] or None) if stats else None
    pipe = Sequential(build_underwater_transforms(ref_stats=ref, strength=a.strength, overlay=False))

    for tag, k in todo:
        name = f"{p.stem}__{tag}{k}"
        if tag == "aug":
            if _save(img_dir, lab_dir, name, changed(pipe, im), lab_txt):
                res["aug"] += 1
        elif tag == "ovl":
            opipe = Sequential([UWSyntheticOverlay(p=1.0)] + build_underwater_transforms(
                ref_stats=ref, strength=a.strength, overlay=False))
            if _save(img_dir, lab_dir, name, changed(opipe, im), lab_txt):
                res["ovl"] += 1
        else:                                                   # comp
            donors = G["donors"]
            other = [d for d in donors if d != ds and donors[d]]
            if not other:
                continue
            dp, poly = random.choice(donors[random.choice(other)])
            dimg = cv2.imread(str(dp))
            r = paste_fish(im, dimg, poly, rng) if dimg is not None else None
            if r is None:
                continue
            out, newp = r
            line = "0 " + " ".join(f"{v:.6f}" for v in newp.reshape(-1))
            text = lab_txt.rstrip("\n") + ("\n" if lab_txt.strip() else "") + line + "\n"
            if _save(img_dir, lab_dir, name, out, text):
                res["comp"] += 1
    return res


# ------------------------------------------------------------------ main
def write_lists(det: Path, n_new: dict, n_src: int):
    allp = sorted((det / "images" / "train").glob("*.jpg"))
    late = [p for p in allp if "__ovl" not in p.stem and "__comp" not in p.stem]
    (det / "lists").mkdir(exist_ok=True)
    (det / "lists" / "train_full.txt").write_text("\n".join(str(p.resolve()) for p in allp) + "\n")
    (det / "lists" / "train_late.txt").write_text("\n".join(str(p.resolve()) for p in late) + "\n")
    y = (det / "data.yaml").read_text()
    for nm in ("full", "late"):
        (det / f"data_{nm}.yaml").write_text(
            re.sub(r"(?m)^train:.*$", f"train: {(det / 'lists' / f'train_{nm}.txt').resolve()}", y))
    tot = {t: sum(1 for p in allp if f"__{t}" in p.stem) for t in ("aug", "ovl", "comp")}
    print(f"this run created {n_new['aug']} __aug + {n_new['ovl']} __ovl + {n_new['comp']} __comp copies")
    print(f"in train now: {tot['aug']} __aug + {tot['ovl']} __ovl + {tot['comp']} __comp "
          f"(originals: {n_src}) -> train total {len(allp)}, late-phase list {len(late)}")
    print("yaml: data_full.yaml (phase 1) / data_late.yaml (final epochs, no overlays/composites)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--copies", type=int, default=2, help="__aug copies per train image")
    ap.add_argument("--strength", type=float, default=1.5, help="scales all probabilities")
    ap.add_argument("--domain_mix", action="store_true", help="cross-dataset LAB colour transfer")
    ap.add_argument("--overlay_frac", type=float, default=0.1, help="fraction of train images that get an __ovl copy")
    ap.add_argument("--composite_frac", type=float, default=0.0, help="fraction that get a cross-dataset __comp copy")
    ap.add_argument("--only_labelled", action="store_true", help="skip background frames as sources")
    ap.add_argument("--clean", action="store_true", help="delete existing cached copies first (NOT needed to resume)")
    ap.add_argument("--max_side", type=int, default=0, help="downsize sources to this long side before augmenting (0 = keep)")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (0 = all CPU cores)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cv2.setNumThreads(1)          # OpenCV's own thread pool + fork() can hang; we parallelise per image instead

    det = Path(a.det)
    if a.clean:
        clean(det)
    for t in (det / "images" / "train").glob("*.part"):          # leftovers of an interrupted run
        t.unlink()
    imgs = [p for p in sorted((det / "images" / "train").glob("*.jpg")) if not is_copy(p)]
    if a.only_labelled:
        imgs = [p for p in imgs if (det / "labels" / "train" / f"{p.stem}.txt").read_text().strip()]

    # ---- domain statistics (needs >= 2 datasets to be meaningful)
    by_ds = defaultdict(list)
    for p in imgs:
        by_ds[p.stem.split("__")[0]].append(p)
    stats_by_ds = {}
    if a.domain_mix:
        if len(by_ds) < 2:
            print("[warn] --domain_mix needs >= 2 datasets in train; colour transfer skipped")
        else:
            stats_by_ds = {k: compute_lab_stats(v) for k, v in by_ds.items()}
            print("domain pools:", {k: len(v) for k, v in stats_by_ds.items()})

    rs = random.Random(a.seed)                                   # same picks on every (re)run
    ovl_set = set(rs.sample(range(len(imgs)), int(a.overlay_frac * len(imgs)))) if a.overlay_frac > 0 else set()
    comp_set = set(rs.sample(range(len(imgs)), int(a.composite_frac * len(imgs)))) if a.composite_frac > 0 else set()

    donors = {}
    if comp_set:
        donors = build_donors(det, imgs, seed=a.seed)
        print("composite donor pools (fish with real/pseudo masks):", {k: len(v) for k, v in donors.items()})
        if len([k for k, v in donors.items() if v]) < 2:
            print("[warn] fewer than 2 datasets have mask polygons -> compositing skipped "
                  "(re-run prepare_data.py with --pseudo_masks)")
            comp_set = set()

    G.update(a=a, det=det, stats_by_ds=stats_by_ds, donors=donors, ovl_set=ovl_set, comp_set=comp_set)

    workers = a.workers or os.cpu_count() or 1
    tasks = list(enumerate(map(str, imgs)))
    n = {"aug": 0, "ovl": 0, "comp": 0}
    t0 = time.time()
    print(f"{len(tasks)} source images, {workers} worker(s), copies={a.copies}, max_side={a.max_side or 'orig'}")

    def tally(done, r):
        for k in n:
            n[k] += r[k]
        if done % 250 == 0 or done == len(tasks):
            print(f"  [{done}/{len(tasks)}] {time.time() - t0:.0f}s  new copies so far: "
                  f"{n['aug']} aug / {n['ovl']} ovl / {n['comp']} comp", flush=True)

    try:
        if workers == 1:
            for done, t in enumerate(tasks, 1):
                tally(done, process(t))
        else:
            with mp.get_context("fork").Pool(workers) as pool:
                for done, r in enumerate(pool.imap_unordered(process, tasks, chunksize=8), 1):
                    tally(done, r)
    except KeyboardInterrupt:
        print("\nInterrupted - nothing is lost. Run the same command again to continue "
              "(lists/yaml are written only when the run finishes).")
        raise SystemExit(1)

    write_lists(det, n, len(imgs))


if __name__ == "__main__":
    main()
