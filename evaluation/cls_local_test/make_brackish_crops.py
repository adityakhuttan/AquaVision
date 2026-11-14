"""Fish crops from the Roboflow Brackish zip (species NOT in Fish4Knowledge) -> open-set test for the classifier."""
import re, sys, zipfile, random
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

ZIP, OUT, PER_CLIP = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUT.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(ZIP); names = set(z.namelist())
RF = re.compile(r"^(?P<clip>.+)-(?P<frame>\d+)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+$")
by_clip = defaultdict(list)
for n in sorted(names):
    if not n.endswith(".jpg"):
        continue
    d, f = n.rsplit("/", 1); stem = f[:-4]; m = RF.match(stem)
    lab = d.rsplit("/", 1)[0] + "/labels/" + stem + ".txt"
    if m and lab in names:
        bx = [list(map(float, l.split()[1:5])) for l in z.read(lab).decode().splitlines()
              if l.split() and int(l.split()[0]) in (1, 4)]       # fish, small_fish
        if bx:
            by_clip[m["clip"]].append((n, bx))
rng = random.Random(0); n_out = 0
for clip, frames in sorted(by_clip.items()):
    cand = [(n, b) for n, bx in frames for b in bx]
    for k, (n, (cx, cy, w, h)) in enumerate(rng.sample(cand, min(PER_CLIP, len(cand)))):
        im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        H, W = im.shape[:2]
        pw, ph = w * W * 1.15, h * H * 1.15                        # 15 % context, like a detector crop
        if min(pw, ph) < 16:
            continue
        x1, y1 = int(max(0, cx * W - pw / 2)), int(max(0, cy * H - ph / 2))
        x2, y2 = int(min(W, cx * W + pw / 2)), int(min(H, cy * H + ph / 2))
        cv2.imwrite(str(OUT / f"{clip}__{k:03d}.jpg"), im[y1:y2, x1:x2]); n_out += 1
print(f"{n_out} Brackish fish crops from {len(by_clip)} clips -> {OUT}")
