"""Model weight registry, download (one-time) and lazy GPU loading.

After `python scripts/fetch_models.py` everything runs fully offline.
Checksums are recorded on first download (models/manifest.json) and verified on load.
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Weight:
    name: str
    filename: str
    urls: tuple[str, ...]
    licence: str


REGISTRY: dict[str, Weight] = {
    "lama": Weight(
        "lama", "big-lama.pt",
        ("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",),
        "Apache-2.0",
    ),
    "realesrgan_x2": Weight(
        "realesrgan_x2", "RealESRGAN_x2plus.pth",
        ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",),
        "BSD-3-Clause",
    ),
    "codeformer": Weight(
        "codeformer", "codeformer.pth",
        ("https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",),
        "S-Lab 1.0 (non-commercial)",
    ),
    "gfpgan": Weight(
        "gfpgan", "GFPGANv1.4.pth",
        ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",),
        "Apache-2.0",
    ),
    "yunet": Weight(
        "yunet", "face_detection_yunet_2023mar.onnx",
        (
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        "MIT",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(models_dir: Path) -> Path:
    return models_dir / "manifest.json"


def _load_manifest(models_dir: Path) -> dict:
    p = _manifest_path(models_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def download(key: str, models_dir: Path) -> Path:
    w = REGISTRY[key]
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / w.filename
    if dest.exists():
        return dest
    last_err: Exception | None = None
    for url in w.urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mattress-model-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            if tmp.stat().st_size < 100_000:  # an LFS pointer or HTML error page
                raise IOError(f"download too small ({tmp.stat().st_size} bytes)")
            tmp.replace(dest)
            break
        except Exception as e:  # try next mirror
            last_err = e
            tmp.unlink(missing_ok=True)
    else:
        raise RuntimeError(f"could not download {w.filename}: {last_err}")
    man = _load_manifest(models_dir)
    man[w.filename] = {"sha256": _sha256(dest), "licence": w.licence, "url": url}
    _manifest_path(models_dir).write_text(json.dumps(man, indent=2))
    return dest


def weight_path(key: str, models_dir: Path, verify: bool = True) -> Path:
    w = REGISTRY[key]
    p = Path(models_dir) / w.filename
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing - run `python scripts/fetch_models.py` once (needs internet)."
        )
    if verify:
        rec = _load_manifest(Path(models_dir)).get(w.filename)
        if rec and rec["sha256"] != _sha256(p):
            raise RuntimeError(f"checksum mismatch for {p}; delete it and re-run fetch_models.py")
    return p


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_spandrel(key: str, models_dir: Path, device: torch.device, half: bool = True):
    """Load a model through spandrel (Real-ESRGAN, CodeFormer, GFPGAN)."""
    import spandrel
    import spandrel_extra_arches

    spandrel_extra_arches.install(ignore_duplicates=True)
    desc = spandrel.ModelLoader(device=device).load_from_file(str(weight_path(key, models_dir)))
    desc.model.eval()
    if half and device.type == "cuda" and desc.supports_half:
        desc.model.half()
    return desc


def load_lama(models_dir: Path, device: torch.device):
    """Big-LaMa TorchScript archive (same file IOPaint uses): model(img[0..1], mask{0,1}) -> img."""
    m = torch.jit.load(str(weight_path("lama", models_dir)), map_location="cpu")
    m.eval()
    return m.to(device)


def free_gpu() -> None:
    """Release cached VRAM. Callers must drop their own model references first."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
