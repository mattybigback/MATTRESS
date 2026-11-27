"""Decide 0/90/180/270 rotation for a cropped print.

1. CLIP zero-shot: each rotation's similarity to "upright photo" prompts minus "rotated /
   upside-down" prompts, softmaxed across the four rotations.
2. Faces: YuNet on all four rotations; upright faces (eyes above mouth) vote. When faces
   are found the two signals are averaged.
A small prior favours 0 deg (prints are usually placed roughly upright).
Rotation k means np.rot90(img, k) (counter-clockwise k*90 deg).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .facedet import FaceDetector, upright_face_score
from .io import to_u8

UPRIGHT_PROMPTS = [
    "a photo", "an old photograph", "a family snapshot", "a landscape photo",
    "a portrait photo", "a photo taken the right way up",
]
ROTATED_PROMPTS = [
    "an upside-down photo", "a photo rotated sideways", "a photo turned 90 degrees",
    "an image flipped upside down",
]


def _rotations(img: np.ndarray) -> list[np.ndarray]:
    return [np.ascontiguousarray(np.rot90(img, k)) for k in range(4)]


class Orienter:
    def __init__(self, cfg, models_dir: Path, device):
        self.cfg = cfg
        self.models_dir = Path(models_dir)
        self.device = device
        self.faces = FaceDetector(models_dir, score=cfg.faces.min_score)
        self._clip = None

    def _load_clip(self):
        if self._clip is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import open_clip
            import torch

            oc = self.cfg.orient
            model, _, preprocess = open_clip.create_model_and_transforms(
                oc.clip_model, pretrained=oc.clip_pretrained,
                cache_dir=str(self.models_dir / "clip"), device=self.device)
            model.eval()
            tok = open_clip.get_tokenizer(oc.clip_model)
            with torch.no_grad():
                t = model.encode_text(tok(UPRIGHT_PROMPTS + ROTATED_PROMPTS).to(self.device))
                t = t / t.norm(dim=-1, keepdim=True)
            self._clip = (model, preprocess, t)
        return self._clip

    def _clip_scores(self, rots: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        model, preprocess, text = self._load_clip()
        batch = torch.stack([preprocess(Image.fromarray(to_u8(r))) for r in rots]).to(self.device)
        n_up = len(UPRIGHT_PROMPTS)
        t_up, t_rot = text[:n_up].mean(0), text[n_up:].mean(0)
        with torch.no_grad():
            im = model.encode_image(batch)
            im = im / im.norm(dim=-1, keepdim=True)
            # contrastive "upright minus rotated" similarity, compared ACROSS the 4 rotations
            d = 100.0 * (im @ t_up - im @ t_rot)
            return d.softmax(dim=0).float().cpu().numpy()

    def decide(self, img: np.ndarray) -> dict:
        oc = self.cfg.orient
        rgb = to_u8(img if img.ndim == 3 else np.repeat(img[..., None], 3, 2))
        h, w = rgb.shape[:2]
        s = min(1.0, 1600 / max(h, w))
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
        rots = _rotations(small)

        face_scores = np.array([upright_face_score(self.faces.detect(r)) for r in rots])
        clip = self._clip_scores(rots)
        if face_scores.max() >= 0.9:
            # faces are decisive for 90-degree errors, CLIP for upside-down (YuNet often
            # finds faces both ways up) - so combine them
            method = "faces+clip"
            scores = 0.5 * face_scores / (face_scores.sum() + 1e-6) + 0.5 * clip
        else:
            method = "clip"
            scores = clip.copy()
        scores[0] += oc.prior_upright
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        margin = float(scores[order[0]] - scores[order[1]])
        return {
            "k": best,
            "degrees_ccw": best * 90,
            "method": method,
            "scores": [round(float(x), 4) for x in scores],
            "face_scores": [round(float(x), 3) for x in face_scores],
            "margin": round(margin, 4),
            "confident": margin >= oc.min_margin,
        }

    def close(self):
        self._clip = None
