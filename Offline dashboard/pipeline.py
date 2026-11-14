import os
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))
from ultralytics import YOLO
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


CFG = {
    "det": os.getenv("DET_WEIGHTS", str(BASE_DIR / "weights/detector_seg.pt")),
    "cls": os.getenv("CLS_WEIGHTS", str(BASE_DIR / "weights/species_cls.pt")),
}
DET_IMGSZ = 640
CLS_IMGSZ = 224
MAX_VOTES_PER_TRACK = 12
SIZE_LABELS = ["Small", "Medium", "Large"]

_models = {"det": None, "cls": None}

@dataclass
class Settings:
    conf: float = 0.25
    iou: float = 0.5
    t_small: float = 0.08
    t_large: float = 0.20
    do_species: bool = True
    classify_every: int = 5
    max_frames: int = 0
    px_per_cm: float = 0.0
    trail_len: int = 40
    min_track_frames: int = 3

def set_weights(det=None, cls=None):
    if det:
        CFG["det"] = det
    if cls:
        CFG["cls"] = cls
    reload_models()

def reload_models():
    _models["det"] = None
    _models["cls"] = None

def _load(path, fallback=None):
    candidate = Path(path).expanduser() if path else None
    if candidate and candidate.is_file():
        model = YOLO(str(candidate.resolve()))
        return model
    raise FileNotFoundError(f"Local model not found: {path}. Select your team's trained weights in Models. No downloads are performed.")


def get_detector(fresh=False, weights=None):
    if weights:
        return _load(weights)
    if fresh:
        return _load(CFG["det"])
    if _models["det"] is None:
        _models["det"] = _load(CFG["det"])
        if _models["det"].task not in {"detect", "segment"}:
            _models["det"] = None
            raise ValueError("Detector weights must be a detection or segmentation model.")
    return _models["det"]

def get_classifier():
    if not Path(CFG["cls"]).is_file():
        return None
    if _models["cls"] is None:
        _models["cls"] = _load(CFG["cls"])
        if _models["cls"].task != "classify":
            _models["cls"] = None
            raise ValueError("Species weights must be a classification model.")
    return _models["cls"]

def model_status():
    d = "found" if os.path.exists(CFG["det"]) else "MISSING (select custom detector)"
    c = "found" if os.path.exists(CFG["cls"]) else "MISSING (species unavailable)"
    return f"Detector `{CFG['det']}`: {d}  \\nClassifier `{CFG['cls']}`: {c}"

def classify_crop(crop_bgr):
    clf = get_classifier()
    if clf is None or crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", 0.0
    res = clf.predict(crop_bgr, imgsz=CLS_IMGSZ, verbose=False, device="cpu")[0]
    idx = int(res.probs.top1)
    return clf.names[idx], float(res.probs.top1conf)

def mask_pixels(poly, shape):
    h, w = shape[:2]
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(canvas, [(poly - np.array([x0, y0])).astype(np.int32)], 1)
    ys, xs = np.nonzero(canvas)
    if len(xs) < 5:
        return None
    return np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)

def pca_length(points):
    pts = points - points.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt[0]
    return float(proj.max() - proj.min() + 1.0)

def size_class(rel_len, t_small, t_large):
    if rel_len < t_small:
        return SIZE_LABELS[0]
    if rel_len < t_large:
        return SIZE_LABELS[1]
    return SIZE_LABELS[2]

def colour_for(key):
    rng = np.random.RandomState(abs(hash(str(key))) % (2**31))
    return tuple(int(c) for c in rng.randint(60, 255, 3))

def draw_detection(img, det, alpha=0.35):
    colour = colour_for(det["id"] if det["id"] is not None else det["species"])
    poly = det.get("poly")
    if poly is not None and len(poly) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly.astype(np.int32)], colour)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [poly.astype(np.int32)], True, colour, 2)
    x1, y1, x2, y2 = [int(v) for v in det["box"]]
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    tag = f"ID {det['id']} | " if det["id"] is not None else ""
    species_label = det['species'] if det['species'] == 'Unknown' else f"{det['species']} {det['sconf']:.0%}"
    label = f"{tag}{species_label} | {det['size']}"
    if det.get("length_cm") is not None:
        label += f" | {det['length_cm']:.1f} cm"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1, th + 6)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 4, ty + bl - 4), colour, -1)
    cv2.putText(img, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def extract_detections(result, frame, st, votes=None, frame_idx=0, scale=1.0):
    dets = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    h, w = frame.shape[:2]
    diag = float(np.hypot(h, w))
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    ids = (result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else [None] * len(boxes))
    polys = result.masks.xy if result.masks is not None else [None] * len(boxes)
    for box, conf, tid, poly in zip(boxes, confs, ids, polys):
        x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
        x2, y2 = min(int(box[2]), w), min(int(box[3]), h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        pts = None
        if poly is not None and len(poly) >= 3:
            pts = mask_pixels(np.asarray(poly), frame.shape)
        if pts is None:
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        length_px = pca_length(pts)
        rel = length_px / diag
        length_cm = None
        if st.px_per_cm and st.px_per_cm > 0:
            length_cm = (length_px / scale) / st.px_per_cm
        species, sconf = "Unknown", 0.0
        if st.do_species:
            crop = frame[y1:y2, x1:x2]
            if tid is None or votes is None:
                species, sconf = classify_crop(crop)
            else:
                v = votes[tid]
                n = len(v["confs"])
                if n == 0 or (n < MAX_VOTES_PER_TRACK and frame_idx % max(st.classify_every, 1) == 0):
                    sp, sc = classify_crop(crop)
                    v["votes"][sp] += sc
                    v["confs"].append(sc)
                species = v["votes"].most_common(1)[0][0]
                sconf = float(np.mean(v["confs"]))
        dets.append(dict(
            id=tid, box=(x1, y1, x2, y2), poly=None if poly is None else np.asarray(poly),
            conf=float(conf), species=species, sconf=sconf, length_px=length_px,
            rel=rel, length_cm=length_cm, size=size_class(rel, st.t_small, st.t_large),
            centre=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        ))
    return dets

def export_csv(df, name):
    path = os.path.join(tempfile.mkdtemp(), name)
    df.to_csv(path, index=False)
    return path

IMG_COLS = ["Fish", "Species", "Species conf.", "Size class", "Rel. length (% diag)", "Length (cm)", "Det. conf."]

def process_image(image_rgb, st, det_weights=None):
    det_model = get_detector(weights=det_weights)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    res = det_model.predict(bgr, imgsz=DET_IMGSZ, conf=st.conf, iou=st.iou, verbose=False, device="cpu")[0]
    dets = extract_detections(res, bgr, st)
    out = bgr.copy()
    for d in dets:
        draw_detection(out, d)
    rows = [{
        "Fish": i + 1, "Species": d["species"], "Species conf.": round(d["sconf"], 3),
        "Size class": d["size"], "Rel. length (% diag)": round(d["rel"] * 100, 1),
        "Length (cm)": None if d["length_cm"] is None else round(d["length_cm"], 1),
        "Det. conf.": round(d["conf"], 3),
    } for i, d in enumerate(dets)]
    df = pd.DataFrame(rows, columns=IMG_COLS)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), dets, df

def _reencode_h264(src):
    if not shutil.which("ffmpeg"):
        return src
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
        return dst
    except Exception:
        return src

def _movement(path, fps, first, last, diag):
    p = np.asarray(path, np.float32)
    if len(p) < 2:
        return 0.0, 0.0, 0.0, 0.0
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    path_len = float(steps.sum())
    net = p[-1] - p[0]
    disp = float(np.linalg.norm(net))
    dur = max((last - first) / fps, 1e-6)
    heading = float(np.degrees(np.arctan2(-net[1], net[0])))
    return path_len / diag * 100, disp / diag * 100, path_len / diag * 100 / dur, heading

def iter_video(video_path, st, write_video=False, max_width=1280, progress=None, det_weights=None, stop_event=None):
    det_model = get_detector(fresh=True, weights=det_weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")
    scale = min(1.0, max_width / W) if max_width else 1.0
    outW, outH = max(int(W * scale) // 2 * 2, 2), max(int(H * scale) // 2 * 2, 2)
    diag = float(np.hypot(outW, outH))
    writer, raw_path = None, None
    if write_video:
        raw_path = os.path.join(tempfile.mkdtemp(), "annotated.mp4")
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (outW, outH))
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise ValueError("Video writer could not open. Check available codecs.")
    votes = defaultdict(lambda: {"votes": Counter(), "confs": []})
    info = defaultdict(lambda: {"rel": [], "px": [], "det": [], "first": None, "last": None, "n": 0, "path": []})
    records, frame_rows = [], []
    limit = int(st.max_frames) if st.max_frames else 10**12
    n_total = min(total, limit) if total else 0
    idx = 0
    try:
        while idx < limit:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != outW or frame.shape[0] != outH:
                frame = cv2.resize(frame, (outW, outH))
            res = det_model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=DET_IMGSZ, conf=st.conf, iou=st.iou, verbose=False, device="cpu")[0]
            dets = extract_detections(res, frame, st, votes=votes, frame_idx=idx, scale=scale)
            for d in dets:
                tid = d["id"]
                if tid is None:
                    continue
                ti = info[tid]
                ti["rel"].append(d["rel"])
                ti["px"].append(d["length_px"] / scale)
                ti["det"].append(d["conf"])
                ti["first" ] = idx if ti["first"] is None else ti["first"]
                ti["last"] = idx
                ti["n"] += 1
                ti["path"].append(d["centre"])
                d["size"] = size_class(float(np.median(ti["rel"])), st.t_small, st.t_large)
                x1, y1, x2, y2 = d["box"]
                records.append((idx, tid, x1 / scale, y1 / scale, x2 / scale, y2 / scale))
                frame_rows.append({
                    "frame": idx + 1, "time_s": round(idx / fps, 3), "track_id": tid,
                    "species": d["species"], "species_conf": round(d["sconf"], 3),
                    "size_class": d["size"], "rel_length_pct_diag": round(d["rel"] * 100, 2),
                    "length_cm": None if d["length_cm"] is None else round(d["length_cm"], 2),
                    "x1": round(x1 / scale, 1), "y1": round(y1 / scale, 1),
                    "x2": round(x2 / scale, 1), "y2": round(y2 / scale, 1),
                })
            for d in dets:
                if d["id"] is None:
                    continue
                path = info[d["id"]]["path"][-max(int(st.trail_len), 2):]
                if len(path) >= 2:
                    cv2.polylines(frame, [np.array(path, np.int32)], False, colour_for(d["id"]), 2, cv2.LINE_AA)
            for d in dets:
                draw_detection(frame, d)
            seen = sum(1 for t in info.values() if t["n"] >= st.min_track_frames)
            cv2.rectangle(frame, (0, 0), (270, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"Unique fish so far: {seen}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if writer is not None:
                writer.write(frame)
            idx += 1
            yield {"image": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "detections": dets,
                   "frame": idx, "total": n_total, "fps": fps, "count": seen}
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    out_video = _reencode_h264(raw_path) if write_video else None
    rows = []
    for tid, ti in sorted(info.items()):
        if ti["n"] < st.min_track_frames:
            continue
        v = votes[tid]
        species = v["votes"].most_common(1)[0][0] if v["votes"] else "Unknown"
        sconf = float(np.mean(v["confs"])) if v["confs"] else 0.0
        rel = float(np.median(ti["rel"]))
        path_len, disp, speed, heading = _movement(ti["path"], fps, ti["first"], ti["last"], diag)
        row = {
            "Track ID": tid, "Species": species, "Species conf.": round(sconf, 3),
            "Size class": size_class(rel, st.t_small, st.t_large),
            "Rel. length (% diag)": round(rel * 100, 1),
            "Length (cm)": (round(float(np.median(ti["px"])) / st.px_per_cm, 1) if st.px_per_cm and st.px_per_cm > 0 else None),
            "Frames seen": ti["n"],
            "First (s)": round(ti["first"] / fps, 1), "Last (s)": round(ti["last"] / fps, 1),
            "Path (% diag)": round(path_len, 1), "Net disp. (% diag)": round(disp, 1),
            "Speed (% diag/s)": round(speed, 1), "Heading (deg)": round(heading, 0),
        }
        rows.append(row)
    return dict(out_video=out_video, tracks=pd.DataFrame(rows), frames=pd.DataFrame(frame_rows), frame_records=records, fps=fps, n_frames=idx, count=len(rows))

def summarize_tracks(tracks, n_frames, fps, calibrated):
    s = f"### Unique fish (de-duplicated by ByteTrack): **{len(tracks)}**\\n"
    s += f"_Processed {n_frames} frames at {fps:.1f} FPS._\\n\\n"
    if len(tracks):
        sp = Counter(tracks["Species"])
        s += "\\n".join(f"- {k}: {v}" for k, v in sp.most_common())
        sz = Counter(tracks["Size class"])
        s += "\\n\\n**Size classes:** " + ", ".join(f"{k}: {v}" for k, v in sz.items())
        s += f"\\n\\n**Mean speed:** {tracks['Speed (% diag/s)'].mean():.1f} % of frame diagonal / s"
    return s


def process_video(video_path, st, write_video=True, max_width=1280, progress=None, det_weights=None):
    stream = iter_video(video_path, st, write_video, max_width, progress, det_weights)
    while True:
        try:
            packet = next(stream)
            if progress:
                progress(packet["frame"] / max(packet["total"], 1))
        except StopIteration as done:
            return done.value


def video_preview(path):
    """Validate and decode the first frame off the GUI thread."""
    path = str(Path(path).resolve())
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError('Could not open video. Try the included samples/fish_test.mp4.')
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Could not decode video. Try an MP4 (H.264) or AVI (MJPEG) file.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        return dict(path=path, image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    fps=fps, total=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    finally:
        cap.release()


def iter_original_video(path, stop_event=None):
    """Play local video without needing any model weights."""
    metadata = video_preview(path)
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            yield dict(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame=idx,
                       fps=metadata['fps'], total=metadata['total'], count=0)
    finally:
        cap.release()
    return None
