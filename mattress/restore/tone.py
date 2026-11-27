"""Deterministic fade / colour-cast correction (no generative model)."""
from __future__ import annotations

import cv2
import numpy as np


def fade_metrics(x: np.ndarray) -> dict:
    """x: float RGB [0,1] (gray replicated is fine)."""
    s = min(1.0, 1000 / max(x.shape[:2]))
    sm = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)
    L = lab[..., 0] / 100.0
    p1, p99 = np.percentile(L, [1, 99])
    chroma = np.hypot(lab[..., 1], lab[..., 2]) / 100.0
    mid = (L > 0.15) & (L < 0.85)
    means = sm[mid].mean(0) if mid.sum() > 100 else sm.reshape(-1, 3).mean(0)
    return {
        "range": round(float(p99 - p1), 4),
        "black": round(float(p1), 4),
        "white": round(float(p99), 4),
        "saturation": round(float(chroma.mean()), 4),
        "cast": round(float(means.max() - means.min()), 4),
        "channel_means": [round(float(m), 4) for m in means],
    }


def needs_tone(m: dict, cls: str, fc) -> list[str]:
    reasons = []
    if m["range"] < fc.min_range:
        reasons.append("low_range")
    if cls == "colour":
        if m["saturation"] < fc.min_saturation:
            reasons.append("desaturated")
        if m["cast"] > fc.max_cast:
            reasons.append("colour_cast")
    return reasons


def _stretch(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((ch - np.float32(lo)) / np.float32(max(hi - lo, 1e-3)), 0, 1).astype(np.float32)


def correct_tone(x: np.ndarray, cls: str, fc, reasons: list[str]) -> np.ndarray:
    """x: float32 RGB [0,1]. Returns corrected float32 RGB."""
    lo_p, hi_p = fc.stretch_pct
    y = x.astype(np.float32).copy()
    sample = y[:: max(1, y.shape[0] // 600), :: max(1, y.shape[1] // 600)].reshape(-1, 3)

    if cls == "colour" and "colour_cast" in reasons:
        # per-channel auto-levels (removes cast) blended with a common stretch (keeps intent)
        lo_c = np.percentile(sample, lo_p, axis=0)
        hi_c = np.percentile(sample, hi_p, axis=0)
        lum = sample.mean(1)
        lo_g, hi_g = np.percentile(lum, [lo_p, hi_p])
        a = float(fc.wb_strength)
        lo = a * lo_c + (1 - a) * lo_g
        hi = a * hi_c + (1 - a) * hi_g
        for c in range(3):
            y[..., c] = _stretch(y[..., c], lo[c], hi[c])
    elif "low_range" in reasons or "desaturated" in reasons or "colour_cast" in reasons:
        lum = sample.mean(1)
        lo_g, hi_g = np.percentile(lum, [lo_p, hi_p])
        y = _stretch(y, lo_g, hi_g)

    lab = cv2.cvtColor(y, cv2.COLOR_RGB2LAB)
    if cls == "colour" and "desaturated" in reasons:
        chroma = np.hypot(lab[..., 1], lab[..., 2]).mean() / 100.0
        target = max(fc.min_saturation * 1.2, chroma)
        k = float(np.clip(target / max(chroma, 1e-3), 1.0, fc.sat_boost_max))
        lab[..., 1] *= k
        lab[..., 2] *= k

    if fc.clahe_clip and fc.clahe_clip > 0:
        L16 = np.clip(lab[..., 0] / 100.0 * 65535, 0, 65535).astype(np.uint16)
        tiles = max(2, int(round(max(y.shape[:2]) / 600)))
        clahe = cv2.createCLAHE(clipLimit=float(fc.clahe_clip), tileGridSize=(tiles, tiles))
        Lc = clahe.apply(L16).astype(np.float32) / 65535 * 100.0
        lab[..., 0] = 0.5 * lab[..., 0] + 0.5 * Lc  # gentle
    if cls != "colour":
        lab[..., 1] = 0
        lab[..., 2] = 0
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    k = float(fc.get("strength", 1.0))  # gentle: mix the correction over the original
    base = x if cls == "colour" else np.repeat(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[..., None], 3, 2)
    return np.clip(base * (1 - k) + out * k, 0, 1).astype(np.float32)
