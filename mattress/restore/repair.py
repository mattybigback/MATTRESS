"""Repair of large damage (water stains, lost emulsion) inside rough user paint.

1. refine_damage: inside the paint, find pixels that do not look like the good image
   around the paint (different colour, drained chroma, lost texture). Pixels under the
   paint that DO look like their surroundings are left alone - so painting can be rough.
2. recolour: damaged pixels that still carry detail get their local LAB mean/spread matched
   to the nearby good image (deterministic, keeps grain and edges).
3. fill_lost: pixels where the image is gone (flat and near-black/near-white) are filled by
   Big-LaMa on a downscaled copy (large holes need the model's whole-image view), then
   blended back at full resolution with matched film grain.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .damage import _pad8


def _local_std(g: np.ndarray, k: int) -> np.ndarray:
    mu = cv2.blur(g, (k, k))
    return np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0))


def _drop_small(m: np.ndarray, min_area: float) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, 4] >= min_area
    return keep[lab].astype(np.uint8)


def refine_damage(x: np.ndarray, paint: np.ndarray, rc) -> tuple[np.ndarray, np.ndarray, dict]:
    """x float RGB [0,1] full-res, paint uint8 {0,1} full-res.
    Returns (recolour_mask, fill_mask) at full res (uint8) and stats."""
    H, W = paint.shape
    s = min(1.0, float(rc.work_long) / max(H, W))
    sm = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x
    pm = cv2.resize(paint, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    h, w = pm.shape
    lab = cv2.cvtColor(sm.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(A, B)
    tex = _local_std(L, 9)

    # reference: a ring of good image around the paint
    side = max(h, w)
    r_in = max(2, int(side * rc.ring_gap))
    r_out = max(r_in + 4, int(side * rc.ring_width))
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    p8 = pm.astype(np.uint8)
    ring = cv2.dilate(p8, ell(r_out)).astype(bool) & ~cv2.dilate(p8, ell(r_in)).astype(bool)
    if ring.sum() < 200:  # paint covers (almost) everything: fall back to all unpainted pixels
        ring = ~pm
    if ring.sum() < 200:
        return np.zeros_like(paint), np.zeros_like(paint), {"error": "no unpainted reference area"}

    # colour distance to the ring palette (k-means centres in LAB)
    ref = lab[ring].reshape(-1, 3).astype(np.float32)
    if len(ref) > 20000:
        ref = ref[np.random.default_rng(0).choice(len(ref), 20000, replace=False)]
    k = int(rc.palette_k)
    _, _, centres = cv2.kmeans(ref, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                               3, cv2.KMEANS_PP_CENTERS)
    px = lab[pm].reshape(-1, 3)
    dmin = np.min(np.linalg.norm(px[:, None, :] - centres[None, :, :], axis=2), axis=1)

    ring_c = float(np.median(C[ring])) + 1e-3
    ring_t = float(np.median(tex[ring])) + 1e-3
    d_col = np.zeros((h, w), np.float32)
    d_col[pm] = np.clip(dmin / rc.colour_dist, 0, 1)
    d_chr = np.clip((ring_c - C) / ring_c, 0, 1)
    d_tex = np.clip((ring_t - tex) / ring_t, 0, 1)
    score = rc.w_colour * d_col + rc.w_chroma * d_chr + rc.w_texture * d_tex
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 2)
    score[~pm] = 0

    vals = (np.clip(score[pm], 0, 1) * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu / 255.0, float(rc.min_score))
    dmg = ((score > thr) & pm).astype(np.uint8)
    dmg = cv2.morphologyEx(dmg, cv2.MORPH_OPEN, ell(1))
    dmg = cv2.morphologyEx(dmg, cv2.MORPH_CLOSE, ell(3))
    dmg = _drop_small(dmg, max(30, pm.sum() * 0.002))
    dmg = (cv2.dilate(dmg, ell(int(rc.grow_px))) & p8)  # cover the tide line, stay in paint

    # image gone: flat and near-black / near-white
    gone = dmg.astype(bool) & (tex < ring_t * rc.fill_texture) & ((L < rc.fill_dark_L) | (L > rc.fill_light_L))
    gone = cv2.morphologyEx(gone.astype(np.uint8), cv2.MORPH_CLOSE, ell(3))
    gone = _drop_small(gone & dmg, max(30, pm.sum() * 0.002))
    rec = (dmg & (1 - gone)).astype(np.uint8)

    up = lambda m: (cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    stats = {
        "paint_frac": round(float(pm.mean()), 4),
        "damage_frac_of_paint": round(float(dmg.sum() / max(pm.sum(), 1)), 3),
        "recolour_frac": round(float(rec.mean()), 4),
        "fill_frac": round(float(gone.mean()), 4),
        "threshold": round(thr, 3),
    }
    return up(rec) & paint, up(gone) & paint, stats


def _norm_conv(v: np.ndarray, wgt: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    num = cv2.GaussianBlur(v * wgt, (0, 0), sigma)
    den = cv2.GaussianBlur(wgt, (0, 0), sigma)
    return num / np.maximum(den, 1e-6), den


def _local_stats(ch: np.ndarray, wgt: np.ndarray, sigmas) -> tuple[np.ndarray, np.ndarray]:
    """Spatially varying mean/std of `ch` over pixels where wgt=1, with coarser fallbacks
    where there are no such pixels nearby."""
    mean = np.full(ch.shape, float((ch * wgt).sum() / max(wgt.sum(), 1)), np.float32)
    var0 = float(((ch - mean) ** 2 * wgt).sum() / max(wgt.sum(), 1))
    var = np.full(ch.shape, var0, np.float32)
    for sg in reversed(sigmas):  # coarse first, finer overwrites where supported
        m, den = _norm_conv(ch, wgt, sg)
        m2, _ = _norm_conv(ch * ch, wgt, sg)
        ok = den > 0.05
        mean[ok] = m[ok]
        var[ok] = np.maximum(m2[ok] - m[ok] ** 2, 0)
    return mean, np.sqrt(var)


def recolour(x: np.ndarray, rec: np.ndarray, dmg_all: np.ndarray, rc) -> np.ndarray:
    """Match local LAB statistics of `rec` pixels to the nearby good image."""
    if not rec.any():
        return x
    H, W = rec.shape
    s = min(1.0, float(rc.work_long) / max(H, W))
    sm = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    lab_s = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)
    rec_s = cv2.resize(rec.astype(np.float32), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_AREA)
    good_s = 1.0 - np.clip(cv2.resize(dmg_all.astype(np.float32), (sm.shape[1], sm.shape[0]),
                                      interpolation=cv2.INTER_AREA) * 4, 0, 1)
    rec_w = (rec_s > 0.5).astype(np.float32)
    side = max(sm.shape[:2])
    sigmas = [side * f for f in rc.stat_sigmas]
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    out = lab.copy()
    for c in range(3):
        mg, sg = _local_stats(lab_s[..., c], good_s, sigmas)
        md, sd = _local_stats(lab_s[..., c], rec_w, sigmas)
        up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        mg, sg, md, sd = up(mg), up(sg), up(md), up(sd)
        gain = np.clip(sg / np.maximum(sd, 0.5), 0.3, float(rc.max_gain))
        if c == 0:
            # lightness: stains mostly drain colour, and damaged areas are often genuinely
            # darker/lighter (flash fall-off, corners) - so only move part of the way
            k = float(rc.l_strength)
            full = (lab[..., 0] - md) * np.minimum(gain, 1.5) + mg
            out[..., 0] = lab[..., 0] + k * (full - lab[..., 0])
        else:
            out[..., c] = (lab[..., c] - md) * gain + mg
    res = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2RGB), 0, 1)
    a = cv2.GaussianBlur(rec.astype(np.float32), (0, 0), max(1.0, max(H, W) * 0.002))[..., None]
    return (x * (1 - a) + res * a).astype(np.float32)


@torch.no_grad()
def fill_lost(model, x: np.ndarray, fill: np.ndarray, good: np.ndarray, device, rc,
              seed: int = 0) -> np.ndarray:
    """LaMa on a downscaled copy for large holes; blended in at full res with film grain."""
    if not fill.any():
        return x
    H, W = fill.shape
    s = min(1.0, float(rc.lama_long) / max(H, W))
    sm = cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    ms = (cv2.resize(fill.astype(np.float32), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_AREA) > 0.01)
    ms = cv2.dilate(ms.astype(np.uint8), np.ones((5, 5), np.uint8))
    it, h, w = _pad8(torch.from_numpy(sm).permute(2, 0, 1)[None].to(device))
    mt, _, _ = _pad8(torch.from_numpy(ms.astype(np.float32))[None, None].to(device))
    r = model(it, (mt > 0.5).float())[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    big = cv2.resize(r, (W, H), interpolation=cv2.INTER_CUBIC)
    # LaMa tends to pull fills toward the image's dominant tone; match each filled region's
    # LAB mean to the pixels immediately around it (already recoloured where needed)
    rr = max(5, int(max(H, W) * 0.015))
    n, lab_cc = cv2.connectedComponents(fill.astype(np.uint8))
    blab = cv2.cvtColor(big.astype(np.float32), cv2.COLOR_RGB2LAB)
    xlab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    for i in range(1, n):
        comp = (lab_cc == i).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        y0, y1 = max(0, ys.min() - rr), min(H, ys.max() + rr + 1)
        x0, x1 = max(0, xs.min() - rr), min(W, xs.max() + rr + 1)
        c = comp[y0:y1, x0:x1]
        ring = cv2.dilate(c, ker).astype(bool) & ~fill[y0:y1, x0:x1].astype(bool)
        if ring.sum() < 20:
            continue
        inside = c.astype(bool)
        sub = blab[y0:y1, x0:x1]
        shift = xlab[y0:y1, x0:x1][ring].mean(0) - sub[inside].mean(0)
        sub[inside] += shift
    big = np.clip(cv2.cvtColor(blab, cv2.COLOR_LAB2RGB), 0, 1)
    # film grain matched to the good image's high-frequency level
    hf = x - cv2.GaussianBlur(x, (0, 0), 1.2)
    g = good.astype(bool)
    sd = hf[g].std(0) if g.sum() > 100 else np.full(3, 0.01)
    noise = np.random.default_rng(seed).normal(0, 1, x.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    noise *= sd / max(float(noise.std()), 1e-6)
    big = np.clip(big + noise, 0, 1)
    a = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), max(1.0, max(H, W) * 0.0015))[..., None]
    a = np.maximum(a, fill[..., None].astype(np.float32))
    return (x * (1 - a) + big * a).astype(np.float32)


def check_image(before: np.ndarray, after: np.ndarray, paint: np.ndarray, rec: np.ndarray,
                fill: np.ndarray, long_edge: int = 1400, protect: np.ndarray | None = None) -> np.ndarray:
    """Side-by-side RGB uint8: before with paint outline + recolour (blue) / fill (orange)
    tints, and the result."""
    H, W = paint.shape
    s = (long_edge / 2) / max(H, W)
    sz = (max(1, int(W * s)), max(1, int(H * s)))
    rs = lambda a: cv2.resize(a, sz, interpolation=cv2.INTER_AREA)
    b = (np.clip(rs(before), 0, 1) * 255).astype(np.uint8)
    a = (np.clip(rs(after), 0, 1) * 255).astype(np.uint8)
    rm = cv2.resize(rec, sz, interpolation=cv2.INTER_NEAREST).astype(bool)
    fm = cv2.resize(fill, sz, interpolation=cv2.INTER_NEAREST).astype(bool)
    pm = cv2.resize(paint, sz, interpolation=cv2.INTER_NEAREST)
    ov = b.copy()
    ov[rm] = (0.55 * ov[rm] + 0.45 * np.array([40, 120, 255])).astype(np.uint8)
    ov[fm] = (0.55 * ov[fm] + 0.45 * np.array([255, 140, 0])).astype(np.uint8)
    edge = cv2.morphologyEx(pm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ov[edge] = (255, 0, 255)
    if protect is not None and protect.any():  # painted but protected (people): green outline
        pr = cv2.resize(protect.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST)
        ov[cv2.morphologyEx(pr, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0] = (0, 220, 0)
    gap = np.full((sz[1], 8, 3), 255, np.uint8)
    return np.hstack([ov, gap, a])
