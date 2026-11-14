import random
import re
from collections import Counter
from pathlib import Path
import pipeline
import cv2
import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}
NAMES = ["DeepFish", "OZFish", "Brackish", "Fish4Knowledge"]
SPECIES_SET = "Fish4Knowledge"
DEFAULT_PATHS = {
    "DeepFish": str(pipeline.BASE_DIR / "data/deepfish"),
    "OZFish": str(pipeline.BASE_DIR / "data/ozfish"),
    "Brackish": str(pipeline.BASE_DIR / "data/brackish"),
    "Fish4Knowledge": str(pipeline.BASE_DIR / "data/fish4knowledge"),
}
MAX_GALLERY = 24
_cache = {}

def scan(path, refresh=False):
    if not str(path).strip():
        raise ValueError("Choose a dataset folder first.")
    p = Path(str(path).strip())
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Folder not found: {path}")
    key = str(p.resolve())
    if refresh or key not in _cache:
        imgs, vids = [], []
        for f in sorted(p.rglob("*")):
            ext = f.suffix.lower()
            if not f.is_file():
                continue
            if ext in IMG_EXT:
                imgs.append(f)
            elif ext in VID_EXT:
                vids.append(f)
        _cache[key] = (p, imgs, vids)
    return _cache[key]

def scan_summary(path, refresh=False):
    root, imgs, vids = scan(path, refresh)
    sub = Counter()
    for f in imgs:
        rel = f.relative_to(root)
        sub[rel.parts[0] if len(rel.parts) > 1 else "(top level)"] += 1
    return {
        "n_images": len(imgs), "n_videos": len(vids),
        "videos": [str(v.relative_to(root)) for v in vids[:500]],
        "subfolders": dict(sub.most_common(30)),
        "n_subfolders": len(sub),
    }

def resolve_video(path, rel):
    root, _, _ = scan(path)
    f = (root / rel).resolve()
    if not f.is_relative_to(root.resolve()):
        raise ValueError("Video must be inside the dataset folder.")
    if not f.is_file():
        raise ValueError(f"Video not found: {f}")
    return str(f)

def label_count(img_path):
    parts = list(Path(img_path).parts)
    lp = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            lp = Path(*parts).with_suffix(".txt")
            break
    if lp is None:
        lp = Path(img_path).with_suffix(".txt")
    if not lp.exists():
        return None
    return sum(1 for ln in lp.read_text().splitlines() if ln.strip())

def _thumb(rgb, side=640):
    h, w = rgb.shape[:2]
    s = side / max(h, w)
    return cv2.resize(rgb, (int(w * s), int(h * s))) if s < 1 else rgb

def _norm(s):
    return re.sub(r"[\W_]+", "", str(s).lower())

def check_species(imgs, per_class, seed):
    clf = pipeline.get_classifier()
    if clf is None:
        raise ValueError("Species classifier weights not found - set them in 'Models & settings'.")
    by = {}
    for f in imgs:
        by.setdefault(f.parent.name, []).append(f)
    if len(by) < 2:
        raise ValueError("Expected one sub-folder per species (found fewer than 2 folders).")
    rng = random.Random(seed)
    model_names = {_norm(v) for v in clf.names.values()}
    rows, gallery = [], []
    for cls, files in sorted(by.items()):
        for f in rng.sample(files, min(int(per_class), len(files))):
            im = cv2.imread(str(f))
            if im is None:
                continue
            sp, sc = pipeline.classify_crop(im)
            rows.append({"file": f.name, "true species": cls, "predicted": sp, "confidence": round(sc, 3)})
            if len(gallery) < MAX_GALLERY:
                gallery.append((_thumb(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 256), f"{cls} -> {sp} ({sc:.0%})"))
    df = pd.DataFrame(rows)
    comparable = bool(len(df)) and any(_norm(t) in model_names for t in df["true species"].unique())
    if comparable:
        df["correct"] = [_norm(a) == _norm(b) for a, b in zip(df["true species"], df["predicted"])]
    return df, comparable, gallery

def species_summary(df, comparable):
    if not len(df):
        return "No images could be read.", None
    if not comparable:
        return "Predictions made, but folder names do not match classes.", None
    acc = float(df["correct"].mean())
    per = (df.groupby("true species")["correct"].agg(["count", "sum"]).rename(columns={"count": "crops", "sum": "correct"}))
    per["accuracy (%)"] = (per["correct"] / per["crops"] * 100).round(1)
    return f"### Species top-1 accuracy: **{acc * 100:.1f}%**", per.reset_index()

def check_images(imgs, n, st, seed):
    rng = random.Random(seed)
    sample = rng.sample(imgs, min(int(n), len(imgs)))
    gallery, rows = [], []
    for f in sample:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        out, dets, _ = pipeline.process_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), st)
        gt = label_count(f)
        sp = Counter(d["species"] for d in dets)
        rows.append({
            "file": f.name, "fish detected": len(dets), "GT fish": gt,
            "count match": None if gt is None else len(dets) == gt,
            "species": ", ".join(f"{k} x{v}" for k, v in sp.most_common()) or "-",
            "sizes": ", ".join(sorted({d["size"] for d in dets})) or "-",
            "mean det. conf.": round(float(np.mean([d["conf"] for d in dets])), 3) if dets else None,
        })
        if len(gallery) < MAX_GALLERY:
            gallery.append((_thumb(out), f"{f.name}: {len(dets)} fish"))
    return gallery, pd.DataFrame(rows)

def detect_summary(df, n_images):
    if not len(df):
        return "No images could be read."
    md = f"### Tested {len(df)} of {n_images} images\\nAverage fish detected per image: **{df['fish detected'].mean():.2f}**"
    return md

def sample_dataset(name, path, n, per_class, seed, st):
    root, imgs, vids = scan(path)
    if not imgs:
        raise ValueError(f"No images found under {root}")
    if name == SPECIES_SET:
        df, comparable, gallery = check_species(imgs, per_class, seed)
        md, per = species_summary(df, comparable)
        return {"gallery": gallery, "table": df, "summary": md, "per_class": per}
    gallery, df = check_images(imgs, n, st, seed)
    return {"gallery": gallery, "table": df, "summary": detect_summary(df, len(imgs)), "per_class": None}

def check_all(paths, n, per_class, seed, st, progress=None):
    rows = []
    for i, name in enumerate(NAMES):
        row = {"Dataset": name, "Folder found": False, "Images": None, "Videos": None, "Tested": None, "Avg fish / image": None, "GT count match (%)": None, "Species top-1 (%)": None, "Notes": ""}
        path = paths.get(name, "")
        try:
            root, imgs, vids = scan(path)
        except ValueError as e:
            row["Notes"] = str(e)
            rows.append(row)
            continue
        row.update({"Folder found": True, "Images": len(imgs), "Videos": len(vids)})
        try:
            if imgs:
                result = sample_dataset(name, path, n, per_class, seed, st)
                table = result["table"]
                row["Tested"] = len(table)
                if name == SPECIES_SET and "correct" in table:
                    row["Species top-1 (%)"] = round(table["correct"].mean() * 100, 1)
                elif "fish detected" in table:
                    row["Avg fish / image"] = round(table["fish detected"].mean(), 2)
                    matches = table["count match"].dropna()
                    row["GT count match (%)"] = round(matches.mean() * 100, 1) if len(matches) else None
                row["Notes"] = result["summary"]
            else:
                row["Notes"] = "No images; select a video for tracking."
        except Exception as exc:
            row["Notes"] = str(exc)
        rows.append(row)
        if progress:
            progress((i + 1) / len(NAMES))
    return pd.DataFrame(rows)