"""
underwater_aug.py  --  AquaVision underwater augmentation toolkit
==================================================================
Implements the three-tier augmentation scheme from the report (Sec. 6.1):

  (i)   geometric  -> handled by Ultralytics itself (flip, rotation, scale, mosaic, mixup)
                      + a tiny label-free flip/rotate helper for classifier crops here
  (ii)  underwater photometric (all label-preserving, written for OpenCV **BGR** images):
          UWColorCast, UWTurbidity, UWVignetteIllumination, UWStochasticCLAHE,
          UWHaze, UWMotionBlur
  (iii) occlusion / domain: UWRandomErasing, UWSyntheticOverlay, UWDomainTransfer

IMPORTANT: Ultralytics and cv2.imread both give BGR arrays, so every transform
here assumes BGR uint8 (H, W, 3).

Every transform is a real `albumentations.ImageOnlyTransform` when Albumentations
is installed (so it can be passed to `model.train(augmentations=[...])`), and a
tiny drop-in shim otherwise (so the offline scripts run without it).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:  # ------------------------------------------------------------------ backend
    import albumentations as A
    _Base = A.ImageOnlyTransform
    HAS_ALBU = True
except ImportError:  # minimal shim with the same calling convention
    A = None
    HAS_ALBU = False

    class _Base:  # type: ignore
        def __init__(self, p: float = 0.5, **_):
            self.p = p

        def __call__(self, force_apply: bool = False, **data):
            if force_apply or np.random.random() < self.p:
                data["image"] = self.apply(data["image"])
            return data


_SEEDED = None


def set_seed(seed=None):
    """Make augmentation reproducible (offline scripts). None -> back to fresh OS entropy."""
    global _SEEDED
    _SEEDED = None if seed is None else np.random.default_rng(seed)
    if seed is not None:
        np.random.seed(seed)


def _rng() -> np.random.Generator:
    # default: fresh OS-seeded generator => safe with DataLoader workers
    return _SEEDED if _SEEDED is not None else np.random.default_rng()


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _lowfreq_noise(h: int, w: int, rng, scales=(4, 8, 16)) -> np.ndarray:
    """Smooth 0..1 noise field (sum of upsampled random grids) -> fog / blobs."""
    acc = np.zeros((h, w), np.float32)
    for i, s in enumerate(scales):
        g = rng.random((s, max(2, int(s * w / max(h, 1))))).astype(np.float32)
        acc += cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) / (i + 1)
    acc -= acc.min()
    return acc / (acc.max() + 1e-6)


# =============================================================== (ii) PHOTOMETRIC
class UWColorCast(_Base):
    """Blue-green cast: attenuate red (fast in water) + pull hue toward cyan.
    Directional, NOT symmetric hue jitter."""

    def __init__(self, red_atten=(0.45, 0.9), hue_pull=(0.05, 0.30), p=0.5):
        super().__init__(p=p)
        self.red_atten, self.hue_pull = red_atten, hue_pull

    def apply(self, img, **params):
        rng = _rng()
        out = img.astype(np.float32)
        out[..., 2] *= rng.uniform(*self.red_atten)          # R (BGR idx 2)
        out[..., 1] *= rng.uniform(1.0, 1.08)                # G
        out[..., 0] *= rng.uniform(1.0, 1.08)                # B
        hsv = cv2.cvtColor(_u8(out), cv2.COLOR_BGR2HSV).astype(np.float32)
        t = rng.uniform(*self.hue_pull)
        h = hsv[..., 0]                                      # OpenCV hue 0..179, cyan ~ 90
        diff = ((90.0 - h + 90.0) % 180.0) - 90.0            # shortest signed distance to cyan
        hsv[..., 0] = (h + t * diff) % 180.0
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def get_transform_init_args_names(self):
        return ("red_atten", "hue_pull")


class UWTurbidity(_Base):
    """Saturation down-scaling (turbidity) + mild speckle noise (particulate scatter)."""

    def __init__(self, sat_scale=(0.4, 0.9), speckle_sigma=(0.02, 0.08), p=0.4):
        super().__init__(p=p)
        self.sat_scale, self.speckle_sigma = sat_scale, speckle_sigma

    def apply(self, img, **params):
        rng = _rng()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(*self.sat_scale)
        out = cv2.cvtColor(_u8(hsv), cv2.COLOR_HSV2BGR).astype(np.float32)
        sigma = rng.uniform(*self.speckle_sigma)
        out *= 1.0 + rng.normal(0.0, sigma, out.shape).astype(np.float32)   # multiplicative = speckle
        return _u8(out)

    def get_transform_init_args_names(self):
        return ("sat_scale", "speckle_sigma")


class UWVignetteIllumination(_Base):
    """Radial vignette + optional localized spotlight hot-spot (ROV/AUV lamps)."""

    def __init__(self, vignette=(0.2, 0.7), hotspot=(0.0, 0.5), gain=(0.7, 1.1), p=0.25):
        super().__init__(p=p)
        self.vignette, self.hotspot, self.gain = vignette, hotspot, gain

    def apply(self, img, **params):
        rng = _rng()
        h, w = img.shape[:2]
        gh, gw = 64, 64                                      # compute on a small grid, upsample
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        yy, xx = yy / (gh - 1), xx / (gw - 1)
        cx, cy = 0.5 + rng.uniform(-0.15, 0.15), 0.5 + rng.uniform(-0.15, 0.15)
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / 0.5
        field = 1.0 - rng.uniform(*self.vignette) * np.clip(r2, 0, 1)
        if rng.random() < 0.6:                               # spotlight
            sx, sy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            sig = rng.uniform(0.05, 0.2)
            field += rng.uniform(*self.hotspot) * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sig ** 2))
        field *= rng.uniform(*self.gain)
        field = cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return _u8(img.astype(np.float32) * field[..., None])

    def get_transform_init_args_names(self):
        return ("vignette", "hotspot", "gain")


class UWStochasticCLAHE(_Base):
    """CLAHE on the L channel, applied stochastically so the model sees both
    enhanced and raw-turbid frames."""

    def __init__(self, clip=(1.5, 4.0), tiles=(8, 8), p=0.4):
        super().__init__(p=p)
        self.clip, self.tiles = clip, tiles

    def apply(self, img, **params):
        rng = _rng()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(rng.uniform(*self.clip)), tileGridSize=tuple(self.tiles))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def get_transform_init_args_names(self):
        return ("clip", "tiles")


class UWHaze(_Base):
    """Synthetic haze: I' = I*t + A*(1-t), with a spatially-varying low-frequency
    transmission map t and a blue-green veiling light A."""

    def __init__(self, density=(0.15, 0.6), p=0.25):
        super().__init__(p=p)
        self.density = density

    def apply(self, img, **params):
        rng = _rng()
        h, w = img.shape[:2]
        n = _lowfreq_noise(h, w, rng)
        d = rng.uniform(*self.density)
        t = (1.0 - d * (0.4 + 0.6 * n))[..., None]
        A_ = np.array([rng.uniform(170, 220), rng.uniform(150, 200), rng.uniform(90, 150)], np.float32)  # B,G,R
        return _u8(img.astype(np.float32) * t + A_ * (1.0 - t))

    def get_transform_init_args_names(self):
        return ("density",)


class UWMotionBlur(_Base):
    """Directional blur (random angle, small kernel): fish swimming + vehicle drift."""

    def __init__(self, kernel=(3, 9), p=0.2):
        super().__init__(p=p)
        self.kernel = kernel

    def apply(self, img, **params):
        rng = _rng()
        k = int(rng.integers(self.kernel[0], self.kernel[1] + 1)) | 1       # force odd
        ker = np.zeros((k, k), np.float32)
        ang = rng.uniform(0, math.pi)
        c = k // 2
        dx, dy = math.cos(ang) * c, math.sin(ang) * c
        cv2.line(ker, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        ker /= ker.sum()
        return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT)

    def get_transform_init_args_names(self):
        return ("kernel",)


# ============================================================ (iii) OCCLUSION / DOMAIN
class UWRandomErasing(_Base):
    """Cutout patches (kept small so a fish is never fully erased while still labelled)."""

    def __init__(self, n_holes=(1, 4), size=(0.05, 0.2), p=0.4):
        super().__init__(p=p)
        self.n_holes, self.size = n_holes, size

    def apply(self, img, **params):
        rng = _rng()
        out = img.copy()
        h, w = img.shape[:2]
        for _ in range(int(rng.integers(self.n_holes[0], self.n_holes[1] + 1))):
            ph, pw = int(h * rng.uniform(*self.size)), int(w * rng.uniform(*self.size))
            y, x = int(rng.integers(0, max(1, h - ph))), int(rng.integers(0, max(1, w - pw)))
            mode = rng.integers(0, 3)
            if mode == 0:
                out[y:y + ph, x:x + pw] = img.mean(axis=(0, 1))                       # mean fill
            elif mode == 1:
                out[y:y + ph, x:x + pw] = rng.integers(0, 256, (ph, pw, 3), dtype=np.uint8)  # noise
            else:
                out[y:y + ph, x:x + pw] = int(rng.integers(0, 256))                   # gray
        return out

    def get_transform_init_args_names(self):
        return ("n_holes", "size")


class UWSyntheticOverlay(_Base):
    """Rock / coral / plant-like blobs (dark earthy or green, soft edges, biased to the
    lower frame = substrate). Crude stand-in for real cut-out silhouettes.
    Flagged `rare` -> switched off in the final epochs (see set_late_phase)."""

    rare = True
    _PALETTE = [(40, 55, 70), (60, 80, 60), (35, 60, 35), (90, 90, 90), (30, 40, 55), (50, 100, 70)]  # BGR

    def __init__(self, n_blobs=(1, 3), size=(0.05, 0.2), p=0.1):
        super().__init__(p=p)
        self.n_blobs, self.size = n_blobs, size

    def apply(self, img, **params):
        rng = _rng()
        h, w = img.shape[:2]
        layer = np.zeros_like(img)
        mask = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(self.n_blobs[0], self.n_blobs[1] + 1))):
            r = rng.uniform(*self.size) * min(h, w)
            cx = rng.uniform(0, w)
            cy = h * rng.uniform(0.6, 1.05) if rng.random() < 0.6 else rng.uniform(0, h)
            m = int(rng.integers(7, 12))
            ang = np.sort(rng.uniform(0, 2 * math.pi, m))
            rad = r * rng.uniform(0.5, 1.2, m)
            pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
            col = np.array(self._PALETTE[int(rng.integers(len(self._PALETTE)))], np.float32)
            col = col * rng.uniform(0.7, 1.3)
            cv2.fillPoly(layer, [pts], tuple(float(c) for c in np.clip(col, 0, 255)))
            cv2.fillPoly(mask, [pts], 255)
        k = int(max(3, (min(h, w) * 0.01))) | 1
        alpha = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32)[..., None] / 255.0 * rng.uniform(0.75, 0.95)
        return _u8(img * (1 - alpha) + layer * alpha)

    def get_transform_init_args_names(self):
        return ("n_blobs", "size")


def compute_lab_stats(paths: Sequence[Path], n: int = 200, seed: int = 0):
    """(mean, std) per LAB channel for a random subset of reference images."""
    rs = np.random.RandomState(seed)
    paths = list(paths)
    rs.shuffle(paths)
    stats = []
    for p in paths[:n]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (128, 128))
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
        stats.append((lab.mean(0), lab.std(0) + 1e-6))
    return stats


class UWDomainTransfer(_Base):
    """Cross-dataset domain mixing: Reinhard colour transfer in LAB toward a random
    image from *another* dataset (partial blend). Label-preserving, so it reduces
    overfitting to one site's water colour/clutter profile."""

    def __init__(self, ref_stats=None, alpha=(0.5, 1.0), p=0.3):
        super().__init__(p=p)
        self.ref_stats, self.alpha = ref_stats or [], alpha

    def apply(self, img, **params):
        if not self.ref_stats:
            return img
        rng = _rng()
        m_r, s_r = self.ref_stats[int(rng.integers(len(self.ref_stats)))]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        flat = lab.reshape(-1, 3)
        m_s, s_s = flat.mean(0), flat.std(0) + 1e-6
        mixed = (lab - m_s) / s_s * s_r + m_r
        a = rng.uniform(*self.alpha)
        out = _u8(a * mixed + (1 - a) * lab)
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    def get_transform_init_args_names(self):
        return ("alpha",)


# ====================================================== crop helper (classifier only)
class FlipRotate:
    """Mild geometric aug for fish crops (no boxes to update)."""

    def __init__(self, max_deg=15, p_flip=0.5, p_rot=0.5):
        self.max_deg, self.p_flip, self.p_rot = max_deg, p_flip, p_rot

    def __call__(self, image=None, **kw):
        rng = _rng()
        img = image
        if rng.random() < self.p_flip:
            img = cv2.flip(img, 1)
        if rng.random() < self.p_rot:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-self.max_deg, self.max_deg), 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return {"image": img}


# =============================================================== PIPELINE FACTORY
def build_underwater_transforms(ref_stats=None, strength: float = 1.0, overlay: bool = True):
    """Probabilities follow Sec. 5 of the augmentation strategy:
         always-on  (~0.4-0.5): colour cast, turbidity
         frequent   (0.3-0.5) : CLAHE, random erasing
         occasional (0.15-0.3): haze, motion blur, vignette/illumination
         rare       (0.05-0.1): synthetic overlays   (disabled in final epochs)
       `strength` scales every probability (use >1 offline, <1 to dial down)."""
    s = lambda p: float(min(1.0, p * strength))
    tf = [
        UWColorCast(p=s(0.5)),
        UWTurbidity(p=s(0.4)),
        UWStochasticCLAHE(p=s(0.4)),
        UWRandomErasing(p=s(0.4)),
        UWHaze(p=s(0.25)),
        UWMotionBlur(p=s(0.2)),
        UWVignetteIllumination(p=s(0.25)),
    ]
    if overlay:
        tf.append(UWSyntheticOverlay(p=s(0.1)))
    if ref_stats:
        tf.append(UWDomainTransfer(ref_stats=ref_stats, p=s(0.3)))
    return tf


def set_late_phase(transforms, late: bool = True):
    """Switch off the 'rare' transforms (synthetic occlusion overlays) for the final
    fine-tuning epochs so mask/box geometry statistics stay realistic."""
    for t in transforms:
        if getattr(t, "rare", False):
            if not hasattr(t, "_p0"):
                t._p0 = t.p
            t.p = 0.0 if late else t._p0


class Sequential:
    """Dependency-free Compose for the offline scripts."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image):
        for t in self.transforms:
            image = t(image=image)["image"]
        return image


# ----------------------------------------------------------------- visual preview
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Preview each underwater transform on one image")
    ap.add_argument("image")
    ap.add_argument("--out", default="aug_preview.jpg")
    a = ap.parse_args()
    im = cv2.imread(a.image)
    assert im is not None, "cannot read image"
    im = cv2.resize(im, (320, 320))
    tiles = [("original", im)]
    for t in build_underwater_transforms():
        tiles.append((type(t).__name__, t(force_apply=True, image=im.copy())["image"]))
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * 320, cols * 320, 3), np.uint8)
    for i, (name, t) in enumerate(tiles):
        r, c = divmod(i, cols)
        t = t.copy()
        cv2.putText(t, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        canvas[r * 320:(r + 1) * 320, c * 320:(c + 1) * 320] = t
    cv2.imwrite(a.out, canvas)
    print("saved", a.out)
