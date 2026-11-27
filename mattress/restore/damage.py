"""Dust / scratch / small-tear detection and LaMa inpainting restricted to the mask."""
from __future__ import annotations

import cv2
import numpy as np
import torch


def _specks(weak: np.ndarray, strong: np.ndarray, k: int, max_area: float) -> tuple[np.ndarray, int]:
    """Compact blobs. Shape is judged on the WEAK component a strong seed belongs to, so a
    fragment of a long real edge is seen as part of that edge (and rejected)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(n, bool)
    seeds = np.unique(lab[strong])
    seeds = seeds[seeds > 0]
    dim = k * 1.5
    ok = (stats[seeds, 4] >= 2) & (stats[seeds, 4] <= max_area) & \
         (np.maximum(stats[seeds, 2], stats[seeds, 3]) <= dim)
    # isolation: dust is a lone blob; fragments of a real edge/line have strong neighbours
    win = int(k * 4) | 1
    integ = cv2.integral(strong.astype(np.uint8))
    H, W = strong.shape
    for j, i in enumerate(seeds):
        if not ok[j]:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = x + w // 2, y + h // 2
        x0, x1 = max(0, cx - win // 2), min(W, cx + win // 2 + 1)
        y0, y1 = max(0, cy - win // 2), min(H, cy + win // 2 + 1)
        total = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        own = int(strong[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].sum())
        if total - own > max(4, own * 0.5):
            ok[j] = False
    keep[seeds[ok]] = True
    return (keep[lab] & (weak > 0)).astype(np.uint8), int(ok.sum())


def _scratches(strong: np.ndarray, weak: np.ndarray, min_len: float, max_thick: float
               ) -> tuple[np.ndarray, int]:
    """Long, straight, thin lines (curved hair and wide edges are rejected)."""
    lines = cv2.HoughLinesP(strong.astype(np.uint8) * 255, 1, np.pi / 360, threshold=int(min_len * 0.6),
                            minLineLength=int(min_len), maxLineGap=int(max_thick * 2))
    out = np.zeros(strong.shape, np.uint8)
    if lines is None:
        return out, 0
    n = 0
    t_in = max(1, int(round(max_thick)))
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        core = np.zeros_like(out)
        cv2.line(core, (x1, y1), (x2, y2), 1, 3)
        inner = np.zeros_like(out)
        cv2.line(inner, (x1, y1), (x2, y2), 1, 2 * t_in + 1)
        outer = np.zeros_like(out)
        cv2.line(outer, (x1, y1), (x2, y2), 1, 6 * t_in + 1)
        ring = (outer > 0) & (inner == 0)
        fill_core = strong[core > 0].mean()
        fill_ring = strong[ring].mean() if ring.any() else 0.0
        length = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
        width = float(strong[outer > 0].sum()) / length  # mean strong-pixel width across the line
        # a real scratch: continuous along its centre line, thin, quiet a few px either side
        if fill_core > 0.5 and fill_ring < 0.15 and width <= max_thick:
            out |= (inner > 0) & (weak > 0)
            n += 1
    return out.astype(np.uint8), n


def damage_mask(x: np.ndarray, dpi: float, dc, protect: list | None = None) -> tuple[np.ndarray, dict]:
    """x: float RGB [0,1]. Returns uint8 mask {0,1} and stats.

    Specks: small high-contrast blobs from white/black top-hat.
    Scratches: thin, elongated top-hat components.
    Guards: components sitting in busy texture (hair, foliage, text) are rejected unless they
    stand out strongly from the local texture.
    """
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x
    g8 = np.clip(g * 255, 0, 255).astype(np.uint8)
    scale = dpi / 600.0
    k = max(5, int(round(15 * scale)) | 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bk = max(31, int(round(61 * scale)) | 1)
    mask = np.zeros(g8.shape, np.uint8)
    # Dust sits in focus on the glass, so it is crisper and smaller than print detail:
    # seeds come from a NARROW top-hat. It is also colourless (white/grey/black lint),
    # unlike pattern dots, berries, lights... -> neutrality test on local chroma.
    ks = max(3, int(round(7 * scale)) | 1)
    kern_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    if x.ndim == 3:
        lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
        # the speck pixel itself must be near-grey (dust is white/grey/black wherever it lands)
        neutral = np.hypot(lab[..., 1], lab[..., 2]) < float(dc.get("speck_max_chroma", 15.0))
    else:
        neutral = np.ones(g8.shape, bool)
    n_speck = n_scratch = 0
    # light damage (dust on the print, white scratches) and dark damage are handled
    # separately so a light scratch never merges with dark picture detail it crosses
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        resp = cv2.morphologyEx(g8, op, kern).astype(np.float32)
        texture = cv2.blur(resp, (bk, bk))            # local grain / detail level
        resp_s = cv2.morphologyEx(g8, op, kern_s).astype(np.float32)
        grain = cv2.blur(resp_s, (bk, bk))  # local film-grain / scanner-noise level
        strong = (resp_s > np.maximum(dc.speck_contrast, grain * float(dc.get("speck_snr", 5.0))))             & (resp > texture * 3.0) & neutral
        weak = (resp > np.maximum(dc.speck_contrast * 0.5, texture * 1.6)).astype(np.uint8)
        m1, ns = _specks(weak, strong, k, dc.speck_max_px * scale * scale)
        # scratches: scale-selective top-hat that only responds to features <= max_thick wide
        max_thick = 3 * scale + 1
        kt = int(2 * max_thick + 1) | 1
        rt = cv2.morphologyEx(g8, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt)))
        rt = rt.astype(np.float32)
        thin = (rt > dc.speck_contrast) & (rt > cv2.blur(rt, (bk, bk)) * 3.0) & neutral
        thin_weak = (rt > dc.speck_contrast * 0.5).astype(np.uint8)
        if op == cv2.MORPH_BLACKHAT and not dc.get("dark_scratches", False):
            m2, nl = np.zeros_like(mask), 0  # dark straight lines are usually picture content
        else:
            m2, nl = _scratches(thin, thin_weak, dc.scratch_min_len_px * scale, max_thick)
        mask |= m1 | m2
        n_speck += ns
        n_scratch += nl
    # never touch faces: eye catchlights and teeth look exactly like dust
    for (fx, fy, fw, fh) in protect or []:
        mx, my = fw * 0.25, fh * 0.25
        mask[max(0, int(fy - my)): int(fy + fh + my), max(0, int(fx - mx)): int(fx + fw + mx)] = 0
    # ignore a thin frame at the print edge (border/trim artefacts are not damage)
    b = max(2, int(round(min(mask.shape) * 0.01)))
    mask[:b] = 0
    mask[-b:] = 0
    mask[:, :b] = 0
    mask[:, -b:] = 0
    if dc.dilate_px:
        d = int(dc.dilate_px) * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    frac = float(mask.mean())
    return mask, {"mask_frac": round(frac, 6), "specks": n_speck, "scratches": n_scratch}


def _pad8(t: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    h, w = t.shape[-2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, h, w


def _tile_boxes(mask: np.ndarray, tile: int, pad: int) -> list[tuple[int, int, int, int]]:
    """Tiles (y0,y1,x0,x1) covering every masked pixel; each tile has `pad` context around its core."""
    H, W = mask.shape
    core = tile - 2 * pad
    boxes = []
    for cy in range(0, H, core):
        for cx in range(0, W, core):
            if not mask[cy: cy + core, cx: cx + core].any():
                continue
            y0 = max(0, min(cy - pad, H - tile))
            x0 = max(0, min(cx - pad, W - tile))
            boxes.append((y0, min(H, y0 + tile), x0, min(W, x0 + tile)))
    return boxes


@torch.no_grad()
def inpaint_lama(model, x: np.ndarray, mask: np.ndarray, device, tile: int = 512, pad: int = 128
                 ) -> np.ndarray:
    """Inpaint only masked pixels; everything outside the mask is returned bit-identical."""
    out = x.copy()
    for y0, y1, x0, x1 in _tile_boxes(mask, tile, pad):
        m = mask[y0:y1, x0:x1]
        if not m.any():
            continue
        # use the current state (earlier tiles already filled) as context
        img = torch.from_numpy(out[y0:y1, x0:x1]).permute(2, 0, 1)[None].float().to(device)
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(device)
        img_p, h, w = _pad8(img)
        mt_p, _, _ = _pad8(mt)
        mt_p = (mt_p > 0.5).float()
        res = model(img_p, mt_p)[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
        sel = m.astype(bool)
        region = out[y0:y1, x0:x1]
        region[sel] = res[sel]
    return out
