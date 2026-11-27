"""Image I/O that preserves bit depth, DPI and ICC profile.

Internal conventions:
  * pages / masters are kept in their native dtype (uint8 or uint16), RGB order, HxWx3 or HxW
  * processing works on float32 RGB in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class ImageMeta:
    dpi: float | None = None
    icc: bytes | None = None
    src_dtype: str = "uint8"
    extra: dict = field(default_factory=dict)


SKIP_WORDS = ("mask", "paint", "repair_check")  # never treat these as page scans


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and p.is_file()
                  and not any(w in p.stem.lower() for w in SKIP_WORDS))


def _tiff_dpi(page: tifffile.TiffPage) -> float | None:
    tags = page.tags
    if "XResolution" not in tags:
        return None
    num, den = tags["XResolution"].value
    if not den:
        return None
    res = num / den
    unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
    unit = int(unit)
    if unit == 3:  # centimetre
        res *= 2.54
    elif unit == 1:  # no absolute unit
        return None
    return float(res) if res > 1 else None


def _normalise_channels(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.dtype not in (np.uint8, np.uint16):
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 65535 + 0.5).astype(np.uint16)
        else:
            arr = arr.astype(np.uint16)
    return np.ascontiguousarray(arr)


def read_image(path: str | Path) -> tuple[np.ndarray, ImageMeta]:
    path = Path(path)
    meta = ImageMeta()
    if path.suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            meta.dpi = _tiff_dpi(page)
            if "InterColorProfile" in page.tags:
                meta.icc = bytes(page.tags["InterColorProfile"].value)
            photometric = page.photometric
        if photometric == tifffile.PHOTOMETRIC.MINISWHITE:
            arr = np.iinfo(arr.dtype).max - arr
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"cannot read {path}")
        if arr.ndim == 3:
            code = cv2.COLOR_BGRA2RGBA if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB
            arr = cv2.cvtColor(arr, code)
        with Image.open(path) as im:
            d = im.info.get("dpi")
            if d and d[0] and d[0] > 1:
                meta.dpi = float(d[0])
            meta.icc = im.info.get("icc_profile")
    arr = _normalise_channels(arr)
    meta.src_dtype = str(arr.dtype)
    return arr, meta


def to_float(arr: np.ndarray) -> np.ndarray:
    """uint8/uint16 -> float32 [0,1]; gray stays 2D."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return np.clip(arr.astype(np.float32), 0, 1)


def to_u16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 65535.0 + 0.5).astype(np.uint16)


def to_u8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    if x.dtype == np.uint16:
        return (x >> 8).astype(np.uint8)
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def to_rgb(x: np.ndarray) -> np.ndarray:
    return np.repeat(x[..., None], 3, axis=2) if x.ndim == 2 else x


def write_tiff(path: str | Path, arr: np.ndarray, dpi: float | None, icc: bytes | None = None) -> None:
    """Write an RGB or gray TIFF (deflate-compressed) with dpi + optional ICC profile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype not in (np.uint8, np.uint16):
        arr = to_u16(arr)
    kw: dict = dict(
        photometric="rgb" if arr.ndim == 3 else "minisblack",
        compression="zlib",
        metadata=None,
    )
    if dpi:
        kw["resolution"] = (float(dpi), float(dpi))
        kw["resolutionunit"] = "INCH"
    if icc and arr.ndim == 3:  # an RGB profile is invalid on a gray image
        kw["iccprofile"] = icc
    tmp = path.with_suffix(path.suffix + ".part")
    tifffile.imwrite(tmp, arr, **kw)
    _replace(tmp, path)


def _replace(tmp: Path, path: Path, tries: int = 5) -> None:
    """Atomic replace; Windows refuses while another program (a viewer) has the file open."""
    import time

    for i in range(tries):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if i == tries - 1:
                tmp.unlink(missing_ok=True)
                raise PermissionError(f"cannot overwrite {path} - is it open in another program?")
            time.sleep(1.0)


def write_jpeg(path: str | Path, arr: np.ndarray, dpi: float | None, quality: int = 95,
               icc: bytes | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(to_u8(arr))
    kw: dict = dict(quality=int(quality), subsampling=0, optimize=True)
    if dpi:
        kw["dpi"] = (int(round(dpi)), int(round(dpi)))
    if icc and im.mode == "RGB":
        kw["icc_profile"] = icc
    tmp = path.with_suffix(path.suffix + ".part")
    im.save(tmp, "JPEG", **kw)
    _replace(tmp, path)


def write_npy(path: Path, arr: np.ndarray) -> None:
    """Atomic .npy write for intermediate stage results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    _replace(tmp, path)
