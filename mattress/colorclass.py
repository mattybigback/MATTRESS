"""Classify a print as colour, mono (neutral B&W) or toned-mono (sepia / yellowed B&W)."""
from __future__ import annotations

import cv2
import numpy as np

from .io import to_float


def lab_float(img: np.ndarray) -> np.ndarray:
    x = to_float(img)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, 2)
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB)  # L 0..100, a/b ~ -128..127


def classify(img: np.ndarray, cfg) -> dict:
    bc = cfg.bw
    x = to_float(img)
    if x.ndim == 2:
        return {"class": "mono", "colour_frac": 0.0, "hue_spread": 0.0, "cast_ab": [0.0, 0.0]}
    h, w = x.shape[:2]
    s = min(1.0, 800 / max(h, w))
    small = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x
    # ignore 4% border (paper borders, lid slivers)
    bh, bw = int(small.shape[0] * 0.04), int(small.shape[1] * 0.04)
    small = small[bh: small.shape[0] - bh, bw: small.shape[1] - bw]
    lab = lab_float(small)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    # ignore near-black / near-white pixels where chroma is noise
    valid = (L > 8) & (L < 97)
    if valid.sum() < 100:
        valid = np.ones_like(L, bool)
    raw_chroma = np.hypot(a[valid], b[valid])
    cast = np.array([np.median(a[valid]), np.median(b[valid])])
    ca, cb = a[valid] - cast[0], b[valid] - cast[1]
    chroma = np.hypot(ca, cb)
    coloured = chroma > bc.chroma_thresh
    colour_frac = float(coloured.mean())

    # hue spread of the raw (cast-included) colour: toned prints have one tight hue
    hue = np.arctan2(b[valid], a[valid])
    strong = raw_chroma > bc.chroma_thresh
    if strong.sum() > 50:
        c, s_ = np.cos(hue[strong]).mean(), np.sin(hue[strong]).mean()
        R = np.hypot(c, s_)
        hue_spread = float(np.degrees(np.sqrt(max(0.0, -2 * np.log(max(R, 1e-6))))))  # circular std
    else:
        hue_spread = 0.0

    if colour_frac >= bc.colour_frac:
        cls = "colour"
    elif float(np.median(raw_chroma)) > bc.chroma_thresh and hue_spread < bc.hue_spread_deg:
        cls = "toned-mono"
    else:
        cls = "mono"
    return {
        "class": cls,
        "colour_frac": round(colour_frac, 4),
        "hue_spread": round(hue_spread, 2),
        "median_chroma": round(float(np.median(raw_chroma)), 2),
        "cast_ab": [round(float(cast[0]), 2), round(float(cast[1]), 2)],
    }


def to_neutral_gray(img: np.ndarray) -> np.ndarray:
    """float RGB -> float gray using LAB lightness (drops sepia/yellow cast)."""
    x = to_float(img)
    if x.ndim == 2:
        return x
    L = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 0] / 100.0
    # LAB L is perceptual; convert back to sRGB-encoded gray value
    lab = np.zeros_like(x)
    lab[..., 0] = L * 100.0
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[..., 0].astype(np.float32)
