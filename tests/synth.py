"""Synthetic flatbed pages: photo-like prints on a white lid, with known geometry."""
from __future__ import annotations

import cv2
import numpy as np


def fake_photo(h: int, w: int, rng: np.random.Generator, colour: bool = True,
               white_border: int = 0) -> np.ndarray:
    """uint16 RGB photo-like content: gradients, blobs, lines and grain."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        a, b = rng.uniform(-1, 1, 2)
        base[..., c] = 0.45 + 0.25 * (a * xx / w + b * yy / h)
    for _ in range(12):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.05, 0.25) * min(h, w)
        col = rng.uniform(0.1, 0.9, 3)
        m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
        base = base * (1 - m) + col * m
    for _ in range(6):  # straight structure (horizon, frames)
        y = int(rng.uniform(0.1, 0.9) * h)
        cv2.line(base, (0, y), (w, y), tuple(float(v) for v in rng.uniform(0, 1, 3)), max(2, h // 150))
    base += rng.normal(0, 0.015, base.shape).astype(np.float32)
    if not colour:
        g = base.mean(2, keepdims=True)
        base = np.repeat(g, 3, 2) * np.array([1.0, 0.93, 0.8], np.float32)  # sepia-ish
    base = np.clip(base, 0.03, 0.97)
    if white_border:
        out = np.full((h + 2 * white_border, w + 2 * white_border, 3), 0.95, np.float32)
        out[white_border:-white_border, white_border:-white_border] = base
        base = out
    return (base * 65535).astype(np.uint16)


def make_page(rng: np.random.Generator, dpi: int = 300, n: int = 3, white_border: bool = False,
              max_tilt: float = 6.0) -> tuple[np.ndarray, list[dict]]:
    """Return (page uint16 RGB, truth list of {quad, angle, colour, k}).

    Prints are placed in a grid so they never overlap. Lid: off-white with slight noise.
    """
    W, H = int(8.5 * dpi), int(11.7 * dpi)
    page = np.full((H, W, 3), 0.965, np.float32)
    page += rng.normal(0, 0.004, page.shape).astype(np.float32)
    page = (np.clip(page, 0, 1) * 65535).astype(np.uint16)
    truth = []
    cell_h = H // n
    for i in range(n):
        ph, pw = int(rng.uniform(2.4, 2.8) * dpi), int(rng.uniform(3.5, 4.2) * dpi)
        colour = bool(i % 2 == 0)
        border = int(0.15 * dpi) if white_border else 0
        photo = fake_photo(ph, pw, rng, colour, border)
        ph, pw = photo.shape[:2]
        ang = float(rng.uniform(-max_tilt, max_tilt))
        cx = W / 2 + rng.uniform(-0.3, 0.3) * dpi
        cy = cell_h * (i + 0.5)
        # warp the photo onto the page
        src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
        R = cv2.getRotationMatrix2D((pw / 2, ph / 2), -ang, 1.0)
        dst = (np.c_[src, np.ones(4)] @ R.T) + np.array([cx - pw / 2, cy - ph / 2])
        dst = dst.astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(photo, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.ones((ph, pw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
        # thin lamp shadow along the edge (as real flatbeds produce)
        shadow = cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) - (mask > 0.5)
        page = np.where(shadow[..., None] > 0, (page * 0.85).astype(np.uint16), page)
        page = (page * (1 - mask[..., None]) + warped * mask[..., None]).astype(np.uint16)
        truth.append({"quad": dst, "angle": ang, "colour": colour})
    return page, truth


def add_dust(img: np.ndarray, rng: np.random.Generator, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Add white specks + a scratch. Returns (img, truth_mask)."""
    out = img.copy()
    m = np.zeros(img.shape[:2], np.uint8)
    h, w = m.shape
    for _ in range(n):
        cv2.circle(m, (int(rng.uniform(0.05, 0.95) * w), int(rng.uniform(0.05, 0.95) * h)),
                   int(rng.integers(1, 4)), 1, -1)
    x0 = int(w * 0.2)
    cv2.line(m, (x0, int(h * 0.15)), (x0 + int(w * 0.1), int(h * 0.8)), 1, 2)
    hi = np.iinfo(out.dtype).max if out.dtype != np.float32 else 1.0
    out[m > 0] = hi
    return out, m
