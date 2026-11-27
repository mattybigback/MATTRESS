"""Scratch / fold / crack detection with the U-Net from Microsoft's "Bringing Old Photos Back
to Life" (MIT). Weights are fetched once by scripts/fetch_models.py and stored as safetensors.

The model was trained on ~256-512 px images. At 600 dpi it is run at two scales: 512 px (short
side) finds folds, creases and larger scratches reliably; 1024 px adds fine scratches, which
are only kept where the model is very confident (it also fires on lips, fabric creases...).
"""
from __future__ import annotations

import io
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

ZIP_URL = ("https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/"
           "v1.0/global_checkpoints.zip")
MEMBER = "checkpoints/detection/FT_Epoch_latest.pt"
WEIGHTS = "bopb_scratch_detection.safetensors"


class _HttpFile(io.RawIOBase):
    """Seekable read-only HTTP file (Range requests) - lets zipfile pull one member out of a
    2 GB release archive without downloading the rest."""

    def __init__(self, url: str):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD",
                                                          headers={"User-Agent": "mattress"}))
        self.url, self.size, self.pos = r.geturl(), int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, o, w=0):
        self.pos = {0: o, 1: self.pos + o, 2: self.size + o}[w]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}",
                                                        "User-Agent": "mattress"})
        d = urllib.request.urlopen(req, timeout=60).read()
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def download(models_dir: Path) -> Path:
    import zipfile

    from safetensors.torch import save_file

    models_dir = Path(models_dir)
    dest = models_dir / WEIGHTS
    if dest.exists():
        return dest
    tmp = models_dir / "bopb_detection_checkpoint.part"
    z = zipfile.ZipFile(io.BufferedReader(_HttpFile(ZIP_URL), buffer_size=8 << 20))
    with z.open(MEMBER) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    ck = torch.load(tmp, map_location="cpu", weights_only=True)  # tensors only - no pickled code
    sd = {k.replace("module.", ""): v.contiguous() for k, v in ck["model_state"].items()}
    save_file(sd, str(dest))
    tmp.unlink()
    return dest


def load(models_dir: Path, device) -> torch.nn.Module:
    from safetensors.torch import load_file

    from .third_party.bopb_unet import UNet

    path = Path(models_dir) / WEIGHTS
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/fetch_models.py` once.")
    m = UNet()
    m.load_state_dict(load_file(str(path)))
    return m.eval().to(device)


@torch.no_grad()
def _prob(model, gray: np.ndarray, short: int, device) -> np.ndarray:
    H, W = gray.shape
    s = short / min(H, W)
    h, w = max(16, int(round(H * s / 16)) * 16), max(16, int(round(W * s / 16)) * 16)
    x = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    t = torch.from_numpy((x - 0.5) / 0.5)[None, None].to(device)
    p = torch.sigmoid(model(t))[0, 0].float().cpu().numpy()
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)


def detect(model, x: np.ndarray, device, sc, protect: list | None = None) -> tuple[np.ndarray, dict]:
    """x float RGB [0,1] -> uint8 {0,1} damage mask at full res + stats."""
    g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    p_lo = _prob(model, g, int(sc.scale_coarse), device)
    p_hi = _prob(model, g, int(sc.scale_fine), device)
    mask = (p_lo >= float(sc.thresh_coarse)) | (p_hi >= float(sc.thresh_fine))
    mask = mask.astype(np.uint8)
    H, W = mask.shape
    # drop tiny fragments, keep everything elongated or reasonably sized
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    min_area = (max(H, W) * 0.0015) ** 2
    keep[1:] = (st[1:, 4] >= min_area) | (np.maximum(st[1:, 2], st[1:, 3]) >= max(H, W) * 0.01)
    mask = keep[lab].astype(np.uint8)
    for (fx, fy, fw, fh) in protect or []:  # faces: never auto-repaired
        mx, my = fw * 0.25, fh * 0.25
        mask[max(0, int(fy - my)): int(fy + fh + my), max(0, int(fx - mx)): int(fx + fw + mx)] = 0
    r = max(1, int(round(max(H, W) * float(sc.dilate_frac))))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return mask, {"scratch_frac": round(float(mask.mean()), 5), "scratch_regions": int(keep[1:].sum())}
