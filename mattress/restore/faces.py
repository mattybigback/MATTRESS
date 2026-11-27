"""Conservative face restoration: YuNet landmarks -> FFHQ alignment -> CodeFormer/GFPGAN ->
feathered paste-back, blended over the original to preserve identity."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from ..facedet import FaceDetector

# FFHQ 512 five-point template (image-left eye, image-right eye, nose, mouth-left, mouth-right)
FFHQ_512 = np.array([
    [192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
    [201.26117, 371.41043], [313.08905, 371.15118],
], np.float32)
MIN_FACE_PX = 24  # below this there is no real information to restore - leave alone


def select_faces(x: np.ndarray, det: FaceDetector, fc) -> list[dict]:
    faces = det.detect(x, work_long=2400)
    chosen = []
    g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    for f in faces:
        if f[14] < fc.min_score:
            continue
        fx, fy, fw, fh = f[:4]
        if fw < MIN_FACE_PX:
            continue
        x0, y0 = int(max(0, fx)), int(max(0, fy))
        crop = g[y0: int(fy + fh), x0: int(fx + fw)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
        blur = float(cv2.Laplacian(crop, cv2.CV_32F).var())
        small = fw < fc.small_px
        blurry = blur < fc.blur_thresh
        if small or blurry:
            chosen.append({
                "box": [round(float(v), 1) for v in f[:4]],
                "landmarks": f[4:14].reshape(5, 2).tolist(),
                "score": round(float(f[14]), 3),
                "blur": round(blur, 1),
                "reason": "small" if small else "blurry",
            })
    return chosen


def _run_face_model(desc, face: torch.Tensor, model_name: str, fidelity: float) -> torch.Tensor:
    """face: 1x3x512x512 in [0,1] -> restored, same shape/range.

    Both raw architectures (as loaded by spandrel) work in [-1, 1], like the originals.
    """
    x = face * 2 - 1
    if model_name == "codeformer":
        out = desc.model(x, weight=float(fidelity))  # fidelity weight w
    else:
        out = desc.model(x, return_rgb=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return ((out + 1) / 2).clamp(0, 1)


@torch.no_grad()
def restore_faces(desc, x: np.ndarray, faces: list[dict], fc, device) -> np.ndarray:
    out = x.copy()
    H, W = x.shape[:2]
    half = next(desc.model.parameters()).dtype == torch.float16
    for f in faces:
        lm = np.array(f["landmarks"], np.float32)
        M, _ = cv2.estimateAffinePartial2D(lm, FFHQ_512, method=cv2.LMEDS)
        if M is None:
            continue
        aligned = cv2.warpAffine(out, M, (512, 512), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        t = torch.from_numpy(aligned).permute(2, 0, 1)[None].to(device)
        t = t.half() if half else t.float()
        r = _run_face_model(desc, t, fc.model, fc.fidelity)[0].float().permute(1, 2, 0).cpu().numpy()

        Minv = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(r, Minv, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        # feathered mask: inner ellipse-ish region of the 512 crop
        m = np.zeros((512, 512), np.float32)
        cv2.rectangle(m, (40, 40), (472, 472), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 24)
        mback = cv2.warpAffine(m, Minv, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        a = mback * float(fc.blend)
        out = out * (1 - a) + np.clip(back, 0, 1) * a
    return out.astype(np.float32)
