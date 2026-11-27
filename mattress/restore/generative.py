"""Generative repair of painted damage with SDXL inpainting.

Only pixels that are (a) under the user's paint, (b) judged damaged by refine_damage and
(c) not part of a person are regenerated. People are protected with a person-segmentation
model plus generously expanded face boxes, so the model can never redraw a person.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch

SDXL_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

DEFAULT_PROMPT = ("old colour photograph, 1990s family snapshot, film grain, plain natural "
                  "continuation of the surrounding background, same lighting")
DEFAULT_PROMPT_BW = ("old black and white photograph, film grain, plain natural continuation of "
                     "the surrounding background, same lighting")
NEGATIVE = ("person, people, face, hands, figure, animal, text, letters, watermark, logo, lamp, "
            "light fixture, chandelier, ceiling light, furniture, object, window, picture frame, "
            "decoration, ornament, star, sparkle, cartoon, painting, illustration, blurry, "
            "water stain, damage, mould, scratches, cracks")


def _hf_env(models_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str((Path(models_dir) / "hf").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _hub_dir(models_dir: Path) -> str:
    # explicit cache dir: huggingface_hub freezes HF_HOME at import time (open_clip imports it
    # earlier in a run), so an environment variable set here would be ignored
    return str((Path(models_dir) / "hf" / "hub").resolve())


def download_sdxl(models_dir: Path) -> None:
    _hf_env(models_dir)
    from huggingface_hub import snapshot_download

    snapshot_download(SDXL_INPAINT, cache_dir=_hub_dir(models_dir),
                      allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])


def load_sdxl(models_dir: Path, device: torch.device, low_vram: bool = True):
    _hf_env(models_dir)
    from diffusers import AutoPipelineForInpainting

    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            SDXL_INPAINT, torch_dtype=torch.float16, variant="fp16", local_files_only=True,
            cache_dir=_hub_dir(models_dir))
    except OSError as e:
        raise FileNotFoundError(
            "SDXL inpainting model missing - run `python scripts/fetch_models.py` once (needs internet)."
        ) from e
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    if low_vram:  # ~1.7 GB lower peak for ~20% more time
        pipe.vae.enable_tiling()
        pipe.enable_attention_slicing()
    return pipe


# ------------------------------------------------------------------ automatic prompt
SURFACES = [
    "dark varnished wooden panelled wall", "light wooden panelling", "painted plaster wall",
    "patterned wallpaper", "brick wall", "stone wall", "wooden fence", "curtains",
    "patterned carpet", "plain carpet", "wooden floorboards", "speckled terrazzo floor",
    "tiled floor", "paving slabs", "concrete", "tarmac road", "green lawn grass", "garden shrubs",
    "trees and foliage", "flower bed", "blue sky", "cloudy sky", "sandy beach", "sea water",
    "swimming pool water", "snow", "fields and countryside", "dark shadowy room",
    "kitchen cupboards", "sofa upholstery", "wooden furniture", "white door",
]


@torch.no_grad()
def auto_prompt(x: np.ndarray, regen: np.ndarray, prot: np.ndarray, models_dir: Path, device,
                cfg_orient, bw: bool) -> tuple[str, list]:
    """Describe the undamaged surroundings of the damage with CLIP.

    Patches (~12% of the photo) from a band of good image around the damage, people
    excluded, each vote for surfaces in SURFACES; close-up patches let CLIP see texture
    (wood grain vs fabric) that a whole downscaled photo hides.
    """
    import open_clip
    from PIL import Image

    H, W = regen.shape
    P = max(64, int(max(H, W) * 0.12))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * P + 1, 2 * P + 1))
    pr = cv2.dilate(prot.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    ring = cv2.dilate(regen.astype(np.uint8), k).astype(bool) & ~regen.astype(bool) & ~pr
    tiles = [x[y:y + P, x0:x0 + P] for y in range(0, H - P + 1, P // 3) for x0 in range(0, W - P + 1, P // 3)
             if ring[y:y + P, x0:x0 + P].mean() > 0.85]
    kind = "old black and white photograph" if bw else "old colour photograph"
    generic = DEFAULT_PROMPT_BW if bw else DEFAULT_PROMPT
    if not tiles:
        return generic, []
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg_orient.clip_model, pretrained=cfg_orient.clip_pretrained,
        cache_dir=str(Path(models_dir) / "clip"), device=device)
    tok = open_clip.get_tokenizer(cfg_orient.clip_model)
    text = model.encode_text(tok([f"a photo of {s}" for s in SURFACES]).to(device))
    text = text / text.norm(dim=-1, keepdim=True)
    probs = []
    for i in range(0, len(tiles), 32):
        b = torch.stack([preprocess(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)))
                         for t in tiles[i:i + 32]]).to(device)
        im = model.encode_image(b)
        im = im / im.norm(dim=-1, keepdim=True)
        probs.append((100 * im @ text.T).softmax(-1).float().cpu().numpy())
    del model, text
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    p = np.concatenate(probs).mean(0)
    order = np.argsort(p)[::-1]
    picked = [SURFACES[i] for i in order[:2] if p[i] >= 0.15]
    cands = [(SURFACES[i], round(float(p[i]), 3)) for i in order[:5]]
    if not picked:
        return generic, cands
    prompt = (f"{kind} of {' and '.join(picked)}, 1990s family snapshot, film grain, "
              "natural continuation of the surroundings, same lighting")
    return prompt, cands


# ------------------------------------------------------------------ people protection
_SEG = None


def person_mask(x: np.ndarray, models_dir: Path) -> np.ndarray:
    """uint8 {0,1} mask of people (DeepLabV3 person class), at x's resolution. CPU."""
    global _SEG
    os.environ.setdefault("TORCH_HOME", str((Path(models_dir) / "torch").resolve()))
    from torchvision.models.segmentation import (DeepLabV3_MobileNet_V3_Large_Weights,
                                                 deeplabv3_mobilenet_v3_large)

    if _SEG is None:
        w = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        _SEG = (deeplabv3_mobilenet_v3_large(weights=w).eval(), w.transforms())
    model, tf = _SEG
    H, W = x.shape[:2]
    s = 640 / max(H, W)
    sm = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(sm)).permute(2, 0, 1).float()
    with torch.no_grad():
        out = model(tf(t)[None])["out"][0]
    person = (out.argmax(0) == 15).numpy().astype(np.uint8)  # VOC class 15 = person
    return cv2.resize(person, (W, H), interpolation=cv2.INTER_NEAREST)


def protect_mask(x: np.ndarray, faces: np.ndarray, models_dir: Path, margin_frac: float) -> np.ndarray:
    """People (segmentation) + heads (face boxes expanded for hair/hats), dilated."""
    H, W = x.shape[:2]
    m = person_mask(x, models_dir)
    for f in faces:
        fx, fy, fw, fh = (float(v) for v in f[:4])
        x0, x1 = int(fx - 0.6 * fw), int(fx + 1.6 * fw)
        y0, y1 = int(fy - 1.0 * fh), int(fy + 1.5 * fh)  # generous above: hair, hats, tiaras
        m[max(0, y0): max(0, y1), max(0, x0): max(0, x1)] = 1
    r = max(3, int(max(H, W) * margin_frac))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


# ------------------------------------------------------------------ inpainting
def _seed(pid: str) -> int:
    return int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)


@torch.no_grad()
def generative_fill(pipe, x: np.ndarray, regen: np.ndarray, prompt: str, rc, pid: str) -> np.ndarray:
    """Inpaint `regen` pixels of float RGB x; returns the full-res composite."""
    H, W = regen.shape
    ys, xs = np.nonzero(regen)
    if len(ys) == 0:
        return x
    # crop around the damage with context, so small damage gets more resolution
    cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int(H * rc.gen_context), int(W * rc.gen_context)
    cy0, cy1 = max(0, cy0 - my), min(H, cy1 + my)
    cx0, cx1 = max(0, cx0 - mx), min(W, cx1 + mx)
    crop = x[cy0:cy1, cx0:cx1]
    mcrop = regen[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    s = (float(rc.gen_pixels) / (ch * cw)) ** 0.5
    w8, h8 = max(512, int(cw * s) // 8 * 8), max(512, int(ch * s) // 8 * 8)
    small = cv2.resize(crop, (w8, h8), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ms = (cv2.resize(mcrop.astype(np.float32), (w8, h8), interpolation=cv2.INTER_AREA) > 0.05)
    ms = cv2.dilate(ms.astype(np.uint8), np.ones((5, 5), np.uint8))

    from PIL import Image

    g = torch.Generator(device=pipe.device).manual_seed(_seed(pid))
    img = pipe(prompt=prompt, negative_prompt=NEGATIVE,
               image=Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8)),
               mask_image=Image.fromarray(ms * 255), width=w8, height=h8,
               strength=float(rc.gen_strength), guidance_scale=float(rc.gen_guidance),
               num_inference_steps=int(rc.gen_steps), generator=g).images[0]
    gen = np.asarray(img).astype(np.float32) / 255.0
    big = cv2.resize(gen, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # film grain to match the untouched image
    hf = crop - cv2.GaussianBlur(crop, (0, 0), 1.2)
    good = cv2.dilate(mcrop, np.ones((15, 15), np.uint8)) == 0
    sd = hf[good].std(0) if good.sum() > 100 else np.full(3, 0.01, np.float32)
    rng = np.random.default_rng(_seed(pid))
    noise = cv2.GaussianBlur(rng.normal(0, 1, crop.shape).astype(np.float32), (0, 0), 0.6)
    big = np.clip(big + noise * (sd / max(float(noise.std()), 1e-6)), 0, 1)

    a = cv2.GaussianBlur(mcrop.astype(np.float32), (0, 0), max(1.0, max(H, W) * 0.002))[..., None]
    a = np.maximum(a, mcrop[..., None].astype(np.float32))
    out = x.copy()
    out[cy0:cy1, cx0:cx1] = crop * (1 - a) + big * a
    return out.astype(np.float32)
