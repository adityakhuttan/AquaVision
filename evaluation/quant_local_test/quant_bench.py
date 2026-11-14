"""INT8 quantization benchmark (CPU, onnxruntime): PyTorch FP32 vs ONNX FP32 vs ONNX INT8 (static, QDQ).
Classifier: F4K test split (2,737 crops). Detector: Brackish part of the seg-v2 test split (578 frames)."""
import csv, json, random, re, shutil, sys, time, zipfile
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from sklearn.metrics import f1_score

SP, PROJ = Path(sys.argv[1]), Path(sys.argv[2])
W = SP / "quant"; W.mkdir(exist_ok=True)
RES = PROJ / "results"
out = {}

# ------------------------------------------------------------------ preprocessing (matches Ultralytics)
def cls_input(bgr, s=224):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]; f = s / min(h, w)
    rgb = cv2.resize(rgb, (max(s, round(w * f)), max(s, round(h * f))), interpolation=cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]; y, x = (h - s) // 2, (w - s) // 2
    return (rgb[y:y + s, x:x + s].astype(np.float32) / 255).transpose(2, 0, 1)[None]

def det_input(bgr, s=640):
    h, w = bgr.shape[:2]; r = min(s / h, s / w); nh, nw = round(h * r), round(w * r)
    canvas = np.full((s, s, 3), 114, np.uint8); top, left = (s - nh) // 2, (s - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None]

class Reader(CalibrationDataReader):
    def __init__(self, name, arrays): self.name, self.it = name, iter(arrays)
    def get_next(self):
        a = next(self.it, None); return None if a is None else {self.name: a}

def export(pt, imgsz, tag):
    src = W / f"{tag}.pt"; shutil.copy(pt, src)
    onnx = Path(YOLO(str(src)).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False, verbose=False))
    return onnx

def quantize(fp32, arrays, tag, exclude=None):
    import onnx
    name = onnx.load(str(fp32)).graph.input[0].name
    int8 = W / f"{tag}_int8.onnx"
    quantize_static(str(fp32), str(int8), Reader(name, arrays), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_exclude=exclude or [])
    return int8

def latency(model_path, img, imgsz, n=40):
    m = YOLO(str(model_path), task="classify" if imgsz == 224 else "segment")
    for _ in range(5): m.predict(img, imgsz=imgsz, verbose=False, device="cpu")
    t = [m.predict(img, imgsz=imgsz, verbose=False, device="cpu")[0].speed["inference"] for _ in range(n)]
    return float(np.median(t))

mb = lambda p: round(Path(p).stat().st_size / 1e6, 2)

# ------------------------------------------------------------------ CLASSIFIER
data = SP / "data" / "cls_split"
test = sorted(p for p in (data / "test").rglob("*.jpg")) + sorted(p for p in (data / "test").rglob("*.png"))
calib = random.Random(0).sample(sorted((data / "val").rglob("*.*")), 300)
c_fp32 = export(RES / "cls_aug1" / "best.pt", 224, "cls_aug1")
c_int8 = quantize(c_fp32, [cls_input(cv2.imread(str(p))) for p in calib], "cls_aug1")

def cls_eval(path):
    m = YOLO(str(path), task="classify"); names = None; yt, yp = [], []
    for i in range(0, len(test), 64):
        chunk = test[i:i + 64]
        for p, r in zip(chunk, m.predict([str(x) for x in chunk], imgsz=224, verbose=False, device="cpu")):
            names = r.names; yt.append(p.parent.name); yp.append(names[int(r.probs.top1)])
    return {"top1": round(float(np.mean(np.array(yt) == np.array(yp))), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)}

probe = cv2.imread(str(test[0]))
out["classifier"] = {}
for tag, path in (("pytorch_fp32", RES / "cls_aug1" / "best.pt"), ("onnx_fp32", c_fp32), ("onnx_int8", c_int8)):
    r = {"size_mb": mb(path), "latency_ms": round(latency(path, probe, 224), 2), **cls_eval(path)}
    out["classifier"][tag] = r; print("cls", tag, r, flush=True)

# ------------------------------------------------------------------ DETECTOR: Brackish test frames from the seg-v2 split
z = zipfile.ZipFile(PROJ / "Brackish Underwater.v2-960x540.yolov11.zip"); names = z.namelist()
RF = re.compile(r"^(?P<clip>.+)-(?P<frame>\d+)_jpg\.rf\.[0-9a-f]+$")
lookup = {}
for n in names:
    if n.endswith(".jpg"):
        m = RF.match(n.rsplit("/", 1)[1][:-4])
        if m: lookup[f"{m['clip']}_frame_{int(m['frame']):04d}"] = n
split = {}
for r in csv.DictReader(open(RES / "seg_baseline" / "manifest.csv")):
    if r["dataset"] == "brackish":
        split[r["file"][len("brackish__"):-4]] = r["split"]
D = W / "det_eval"
for s in ("images/test", "labels/test", "images/train", "labels/train"):
    (D / s).mkdir(parents=True, exist_ok=True)
for stem, sp in split.items():
    if sp != "test" or stem in ("",):
        continue
    n = lookup[stem]; (D / "images/test" / f"{stem}.jpg").write_bytes(z.read(n))
    lab = n.replace("/images/", "/labels/")[:-4] + ".txt"
    lines = []
    for l in z.read(lab).decode().splitlines():
        v = l.split()
        if len(v) >= 5 and int(v[0]) in (1, 4):
            cx, cy, w, h = map(float, v[1:5]); x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            lines.append(f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
    (D / "labels/test" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
(D / "data.yaml").write_text(f"path: {D}\ntrain: images/test\nval: images/test\ntest: images/test\nnc: 1\nnames:\n  0: fish\n")
print("detector eval frames:", len(list((D / "images/test").glob("*.jpg"))), flush=True)
train_stems = [s for s, sp in split.items() if sp == "train"]
calib_d = [det_input(cv2.imdecode(np.frombuffer(z.read(lookup[s]), np.uint8), 1))
           for s in random.Random(0).sample(train_stems, 200)]
d_fp32 = export(RES / "seg_baseline" / "best.pt", 640, "det_v2")
d_int8 = quantize(d_fp32, calib_d, "det_v2")
d_int8_body = quantize(d_fp32, calib_d, "det_v2_body",
                       exclude=[n.name for n in __import__("onnx").load(str(d_fp32)).graph.node if "model.23" in n.name])
probe_d = cv2.imread(str(sorted((D / "images/test").glob("*.jpg"))[0]))
out["detector"] = {}
for tag, path in (("pytorch_fp32", RES / "seg_baseline" / "best.pt"), ("onnx_fp32", d_fp32),
                  ("onnx_int8_full", d_int8), ("onnx_int8_head_fp32", d_int8_body)):
    m = YOLO(str(path), task="segment")
    v = m.val(data=str(D / "data.yaml"), split="test", imgsz=640, batch=1, device="cpu", verbose=False, plots=False,
              project=str(W / "val"), name=tag, exist_ok=True)
    r = {"size_mb": mb(path), "latency_ms": round(latency(path, probe_d, 640), 1),
         "box_mAP50": round(float(v.box.map50), 4), "box_mAP50_95": round(float(v.box.map), 4),
         "mask_mAP50": round(float(v.seg.map50), 4)}
    out["detector"][tag] = r; print("det", tag, r, flush=True)

(W / "quant_results.json").write_text(json.dumps(out, indent=1)); print("->", W / "quant_results.json")
