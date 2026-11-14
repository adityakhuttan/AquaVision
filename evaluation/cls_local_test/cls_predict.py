"""Run both classifiers (plain + horizontally flipped input) on F4K val/test and Brackish crops; save probabilities."""
import sys, json
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

DATA, BREK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MODELS = {"base": sys.argv[4], "aug": sys.argv[5]}
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
sets = {s: sorted(p for p in (DATA / s).rglob("*") if p.suffix.lower() in EXT) for s in ("val", "test")}
sets["brackish"] = sorted(BREK.glob("*.jpg"))
out = {}
for mname, w in MODELS.items():
    m = YOLO(w)
    names = [m.names[i] for i in range(len(m.names))]
    out[f"names_{mname}"] = np.array(names)
    for s, files in sets.items():
        for flip in (0, 1):
            P = []
            for i in range(0, len(files), 64):
                ims = [cv2.imread(str(p)) for p in files[i:i + 64]]
                if flip:
                    ims = [cv2.flip(im, 1) for im in ims]
                P += [r.probs.data.cpu().numpy() for r in m.predict(ims, imgsz=224, verbose=False)]
            out[f"{mname}_{s}_{'flip' if flip else 'plain'}"] = np.stack(P).astype(np.float32)
            print(mname, s, "flip" if flip else "plain", len(P), flush=True)
for s, files in sets.items():
    out[f"files_{s}"] = np.array([str(p) for p in files])
    if s != "brackish":
        out[f"y_{s}"] = np.array([p.parent.name for p in files])
np.savez_compressed(OUT, **out)
print("saved", OUT)
