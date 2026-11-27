"""Tiled Real-ESRGAN upscaling (only for prints whose output would be small)."""
from __future__ import annotations

import numpy as np
import torch


def _ramp(n: int, overlap: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    if overlap > 0 and n > 2 * overlap:
        r = np.linspace(0.05, 1, overlap, dtype=np.float32)
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w


@torch.no_grad()
def upscale_tiled(desc, x: np.ndarray, device, tile: int = 512, overlap: int = 32) -> np.ndarray:
    """x float RGB [0,1] -> upscaled float RGB, processed tile by tile with feathered overlap."""
    s = int(desc.scale)
    H, W = x.shape[:2]
    out = np.zeros((H * s, W * s, 3), np.float32)
    acc = np.zeros((H * s, W * s, 1), np.float32)
    half = next(desc.model.parameters()).dtype == torch.float16
    step = tile - overlap
    ys = list(range(0, max(H - overlap, 1), step)) or [0]
    xs = list(range(0, max(W - overlap, 1), step)) or [0]
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            t = torch.from_numpy(x[y0c:y1, x0c:x1]).permute(2, 0, 1)[None].to(device)
            t = t.half() if half else t.float()
            r = desc(t)[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            th, tw = r.shape[:2]
            wgt = np.outer(_ramp(th, overlap * s), _ramp(tw, overlap * s))[..., None]
            out[y0c * s: y0c * s + th, x0c * s: x0c * s + tw] += r * wgt
            acc[y0c * s: y0c * s + th, x0c * s: x0c * s + tw] += wgt
    return out / np.maximum(acc, 1e-6)
