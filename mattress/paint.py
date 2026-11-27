"""Paint files: a copy of each photo the user can splodge magenta onto (in MS Paint or any
bitmap editor) to mark large damage - water stains, lost emulsion - for repair.

Layout of a paint file: an instruction strip (with a pure-magenta swatch for the colour
picker) on top, then the master image at `paint.scale`. Strip height depends only on the
image width so the reader can recompute it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .io import to_rgb, to_u8

MAGENTA = (255, 0, 255)
PAINT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TEXT = [
    "Paint damage with this colour (use the colour picker on this square).",
    "Rough splodges are fine. Don't paint over people. Save with the same name.",
]


def strip_height(width: int) -> int:
    return max(48, width // 14)


def template_size(master_shape, scale: float) -> tuple[int, int]:
    """(width, height) of a paint file for a master of this shape."""
    h, w = master_shape[:2]
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return tw, th + strip_height(tw)


def make_paint_file(master: np.ndarray, pid: str, out_dir: Path, scale: float = 0.5,
                    quality: int = 92) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = to_rgb(to_u8(master))
    tw, th_total = template_size(master.shape, scale)
    small = cv2.resize(img, (tw, th_total - strip_height(tw)), interpolation=cv2.INTER_AREA)
    sh = strip_height(tw)
    strip = np.full((sh, tw, 3), 255, np.uint8)
    pad = max(4, sh // 8)
    strip[pad: sh - pad, pad: sh - pad] = MAGENTA  # the swatch
    fs = (sh - 2 * pad) / 2 / 30.0
    avail = tw - sh - 2 * pad
    for line in TEXT:  # shrink to fit the width too
        (lw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        if lw > avail:
            fs *= avail / lw
    for i, line in enumerate(TEXT):
        y = pad + int((i + 0.8) * (sh - 2 * pad) / 2)
        cv2.putText(strip, line, (sh + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    max(1, int(fs * 1.5)), cv2.LINE_AA)
    out = np.vstack([strip, small])
    path = out_dir / f"{pid}_paint.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return path


def find_paint_file(out_dir: Path, pid: str) -> Path | None:
    cands = [p for p in Path(out_dir).glob(f"{pid}_paint.*") if p.suffix.lower() in PAINT_EXTS]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PaintFileError(ValueError):
    pass


def magenta_pixels(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (r > 170) & (b > 170) & (g < 100) & (np.abs(r - b) < 70)


def has_paint(path: Path) -> bool:
    """True if the file contains user paint (ignores the swatch in our instruction strip)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sh = strip_height(rgb.shape[1])
    strip = rgb[:sh]
    is_template = magenta_pixels(strip[:, : sh]).mean() > 0.3  # our swatch sits top-left
    body = rgb[sh:] if is_template else rgb
    return int(magenta_pixels(body).sum()) > 50


def read_paint_mask(path: Path, master_shape, scale: float = 0.5) -> np.ndarray:
    """Return a uint8 {0,1} mask at master resolution (all zero if nothing is painted)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PaintFileError(f"cannot read paint file {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tw, th = template_size(master_shape, scale)
    H, W = master_shape[:2]
    size = (rgb.shape[1], rgb.shape[0])
    if size == (tw, th):          # our paint file: drop the instruction strip
        img = rgb[strip_height(tw):]
    elif size == (W, H):          # user painted a full-size copy (e.g. the restored JPEG)
        img = rgb
    else:
        raise PaintFileError(
            f"{Path(path).name} is {size[0]}x{size[1]} but should be {tw}x{th} (paint file) or "
            f"{W}x{H} (full size) - it may have been cropped or resized. "
            "Delete it and rerun to get a fresh copy.")
    m = magenta_pixels(img).astype(np.uint8)
    if not m.any():
        return np.zeros(master_shape[:2], np.uint8)
    k = max(3, int(round(min(m.shape) * 0.006)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    min_area = m.size * 0.0005
    cv2.drawContours(filled, [c for c in cs if cv2.contourArea(c) >= min_area], -1, 1, cv2.FILLED)
    return cv2.resize(filled, (W, H), interpolation=cv2.INTER_NEAREST)
