"""Find individual photo prints on a flatbed page scanned against a (white) lid.

Works on a downsampled copy (cfg.detect.work_dpi). Three cues are fused:
  1. difference from a lid/background model (blank-lid scan if given, else estimated from page)
  2. gradient edges incl. the thin shadow line a flatbed lamp casts along print edges
  3. local texture / saturation (photo content vs. smooth lid)
Contours of the fused mask are fitted with minAreaRect, sides are snapped to the strongest
nearby edge, and each detection gets a confidence score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .io import to_u8


@dataclass
class Detection:
    quad: list[list[float]]  # 4x2 corner points in FULL-RES page pixels, order tl,tr,br,bl
    angle: float             # tilt in degrees (print's rotation on the glass, -45..45)
    width: float             # full-res pixel size along the print's own axes
    height: float
    confidence: float
    rectangularity: float
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _gray_u8(img: np.ndarray) -> np.ndarray:
    g = to_u8(img)
    return cv2.cvtColor(g, cv2.COLOR_RGB2GRAY) if g.ndim == 3 else g


def _background_model(small: np.ndarray, blank_small: np.ndarray | None) -> np.ndarray:
    """Per-pixel lid colour estimate (float32, same shape as small)."""
    if blank_small is not None:
        return cv2.GaussianBlur(blank_small.astype(np.float32), (0, 0), 3)
    # No calibration scan: assume the lid dominates the page border and varies smoothly.
    # Large median filter over a downsampled copy, then upsample - prints are removed
    # because they are smaller than the kernel... except big prints, so blend with the
    # global border median for robustness.
    h, w = small.shape[:2]
    border = np.concatenate([
        small[: max(2, h // 40)].reshape(-1, *small.shape[2:]),
        small[-max(2, h // 40):].reshape(-1, *small.shape[2:]),
        small[:, : max(2, w // 40)].reshape(-1, *small.shape[2:]),
        small[:, -max(2, w // 40):].reshape(-1, *small.shape[2:]),
    ])
    med = np.median(border, axis=0).astype(np.float32)
    return np.broadcast_to(med, small.shape).astype(np.float32).copy()


def _lid_pixels(gray: np.ndarray, sd: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that look like bare scanner lid."""
    cand = (sd < 2.5) & (sat < 25) & (gray > 150)
    if cand.sum() < 0.005 * gray.size:
        return cand
    thr = np.percentile(gray[cand], 50)
    return cand & (gray >= thr)


def _fit_lid_surface(small: np.ndarray, lid: np.ndarray) -> np.ndarray:
    """Quadratic surface per channel fitted to lid pixels: models lamp fall-off / gradients."""
    h, w = lid.shape
    ys, xs = np.nonzero(lid)
    if len(ys) > 20000:
        sel = np.random.default_rng(0).choice(len(ys), 20000, replace=False)
        ys, xs = ys[sel], xs[sel]
    def basis(x, y):
        x = x / w - 0.5
        y = y / h - 0.5
        return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)
    A = basis(xs.astype(np.float32), ys.astype(np.float32))
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    B = basis(gx, gy)
    out = np.empty(small.shape, np.float32)
    for c in range(small.shape[2]):
        coef, *_ = np.linalg.lstsq(A, small[ys, xs, c].astype(np.float32), rcond=None)
        out[..., c] = B @ coef
    return out


def _rectangularity(mask: np.ndarray) -> tuple[float, int]:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0, 0
    c = max(cs, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    return cv2.contourArea(c) / max(rw * rh, 1), int(min(rw, rh))


def _cut_split(blob: np.ndarray, min_side: float, min_rect: float) -> list[np.ndarray]:
    """Split a blob of prints that share edges (L / staircase shapes) with guillotine cuts.

    Works in the blob's own axis frame. Cuts are placed at polygon vertex coordinates; a
    memoised recursive search picks the decomposition with the least wasted bounding-box
    area plus a small penalty per extra part.
    """
    r, _ = _rectangularity(blob)
    if r >= min_rect:
        return [blob]
    c = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    (cx, cy), _, ang = cv2.minAreaRect(c)
    ang = ((ang + 45) % 90) - 45          # smallest rotation that axis-aligns the blob
    H0, W0 = blob.shape
    P = int(max(H0, W0) * np.sin(np.deg2rad(abs(ang)))) + 2   # room so rotation never clips
    padded = cv2.copyMakeBorder(blob, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    H, W = padded.shape
    R = cv2.getRotationMatrix2D((cx + P, cy + P), ang, 1.0)
    rb = cv2.warpAffine(padded, R, (W, H), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(rb)
    if len(ys) == 0:
        return [blob]
    oy, ox = ys.min(), xs.min()
    sub = rb[oy: ys.max() + 1, ox: xs.max() + 1]
    sc = min(1.0, 160.0 / max(sub.shape))
    sm = cv2.resize(sub, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST) if sc < 1 else sub
    rc = max(cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    poly = cv2.approxPolyDP(rc, 0.015 * cv2.arcLength(rc, True), True).reshape(-1, 2)
    # edges are rarely perfectly straight after downscaling: try cuts a little either side
    xc = sorted({int(v) + d for v in poly[:, 0] for d in (-2, -1, 0, 1, 2)})
    yc = sorted({int(v) + d for v in poly[:, 1] for d in (-2, -1, 0, 1, 2)})
    ms = min_side * sc
    total = float(sm.sum())
    lam = 0.02 * total

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(x0: int, x1: int, y0: int, y1: int, depth: int):
        part = sm[y0:y1, x0:x1]
        a = float(part.sum())
        if a == 0:
            return 0.0, ()
        rr = np.flatnonzero(part.any(1))
        cc = np.flatnonzero(part.any(0))
        bx0, bx1 = x0 + int(cc[0]), x0 + int(cc[-1]) + 1
        by0, by1 = y0 + int(rr[0]), y0 + int(rr[-1]) + 1
        if min(bx1 - bx0, by1 - by0) < ms:
            return a, ()  # too small to be a print: discard (costs its area)
        out = ((bx1 - bx0) * (by1 - by0) - a + lam, ((bx0, bx1, by0, by1),))
        if depth >= 4:
            return out
        for v in xc:
            if bx0 + ms <= v <= bx1 - ms:
                c1, p1 = best(bx0, v, by0, by1, depth + 1)
                c2, p2 = best(v, bx1, by0, by1, depth + 1)
                if c1 + c2 < out[0]:
                    out = (c1 + c2, p1 + p2)
        for v in yc:
            if by0 + ms <= v <= by1 - ms:
                c1, p1 = best(bx0, bx1, by0, v, depth + 1)
                c2, p2 = best(bx0, bx1, v, by1, depth + 1)
                if c1 + c2 < out[0]:
                    out = (c1 + c2, p1 + p2)
        return out

    _, wins = best(0, sm.shape[1], 0, sm.shape[0], 0)
    if len(wins) <= 1:
        return [blob]
    Rinv = cv2.invertAffineTransform(R)
    parts = []
    for x0, x1, y0, y1 in wins:
        m = np.zeros_like(rb)
        X0, X1 = ox + int(round(x0 / sc)), ox + int(round(x1 / sc))
        Y0, Y1 = oy + int(round(y0 / sc)), oy + int(round(y1 / sc))
        m[Y0:Y1, X0:X1] = rb[Y0:Y1, X0:X1]
        back = cv2.warpAffine(m, Rinv, (W, H), flags=cv2.INTER_NEAREST)
        parts.append(np.ascontiguousarray(back[P:P + H0, P:P + W0]))
    return parts


def _contours_of(mask: np.ndarray) -> list:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [max(cs, key=cv2.contourArea)] if cs else []


def _erosion_split(blob: np.ndarray, min_side: float) -> list[np.ndarray] | None:
    """Erode until the blob falls apart into >=2 print-sized seeds, then grow each seed back
    over the blob (nearest seed wins). For prints touching at an angle / at a corner."""
    shape = blob.shape
    min_seed = (min_side * 0.5) ** 2
    for r in range(2, int(min_side * 0.6), 2):
        er = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(er)
        big = [i for i in range(1, n) if stats[i, 4] >= min_seed]
        if len(big) >= 2:
            seeds = np.zeros(shape, np.int32)
            for j, i in enumerate(big, 1):
                seeds[lab == i] = j
            _, nearest = cv2.distanceTransformWithLabels((seeds == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                         labelType=cv2.DIST_LABEL_PIXEL)
            idx = np.zeros(nearest.max() + 1, np.int32)
            ys, xs = np.nonzero(seeds)
            idx[nearest[ys, xs]] = seeds[ys, xs]
            owner = idx[nearest] * blob
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
            return [cv2.morphologyEx((owner == j).astype(np.uint8), cv2.MORPH_OPEN, k)
                    for j in range(1, len(big) + 1)]
        if not big:
            return None
    return None


def _xy_cut(blob: np.ndarray, lidlike: np.ndarray, min_side: float, depth: int = 0,
            maps: dict | None = None, max_long: float = 1e9, max_aspect: float = 1e9) -> list[np.ndarray]:
    """Split a blob along straight lanes of bare lid (prints laid in a grid with thin gaps).

    Works in the blob's own axis frame; a lane is a run of rows (or columns) where most of
    the blob's span is lid-coloured. Recurses so grids of any size are separated.
    """
    if depth > 6:
        return [blob]
    c = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    (cx, cy), _, ang = cv2.minAreaRect(c)
    ang = ((ang + 45) % 90) - 45
    H0, W0 = blob.shape
    P = int(max(H0, W0) * np.sin(np.deg2rad(abs(ang)))) + 2
    pb = cv2.copyMakeBorder(blob, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    pl = cv2.copyMakeBorder(lidlike, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    H, W = pb.shape
    R = cv2.getRotationMatrix2D((cx + P, cy + P), ang, 1.0)
    rb = cv2.warpAffine(pb, R, (W, H), flags=cv2.INTER_NEAREST)
    rl = cv2.warpAffine(pl, R, (W, H), flags=cv2.INTER_NEAREST) & rb
    ys, xs = np.nonzero(rb)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    oversized = max(bh, bw) > max_long or max(bh, bw) / max(min(bh, bw), 1) > max_aspect
    rot = lambda m: cv2.warpAffine(cv2.copyMakeBorder(m, P, P, P, P, cv2.BORDER_CONSTANT, value=0),
                                   R, (W, H), flags=cv2.INTER_NEAREST)[y0:y1, x0:x1]
    best = None
    inb0 = rb[y0:y1, x0:x1] > 0
    for axis in (0, 1):  # 0: horizontal lanes (cut rows), 1: vertical lanes
        span = rb[y0:y1, x0:x1].sum(axis=1 - axis).astype(np.float32)
        lid = rl[y0:y1, x0:x1].sum(axis=1 - axis).astype(np.float32)
        if maps is not None:  # edge profile: a real gap is bounded by print edges
            e = rot(maps["edge_h" if axis == 0 else "edge_v"]).astype(bool) & inb0
            ef = e.sum(axis=1 - axis) / np.maximum(span, 1)
        else:
            ef = None
        frac = np.where(span > 0, lid / np.maximum(span, 1), 0)
        frac = np.convolve(frac, np.ones(3) / 3, mode="same")
        n = len(frac)
        lanes = frac > 0.55
        i = 0
        while i < n:
            if lanes[i]:
                j = i
                while j < n and lanes[j]:
                    j += 1
                mid = (i + j) // 2
                bounded = ef is None or min(ef[max(0, i - 10): i + 3].max(initial=0),
                                            ef[max(0, j - 3): j + 10].max(initial=0)) >= 0.35
                if bounded and mid >= min_side and n - mid >= min_side:
                    score = float(frac[i:j].max())
                    if best is None or score > best[0]:
                        best = (score, axis, mid)
                i = j
            else:
                i += 1
    if best is None and oversized and maps is not None:
        # too big / too long to be one print: cut at the strongest straight boundary - a
        # full-width edge with a lighter (lid) strip beside it, i.e. a narrow gap
        inb = rb[y0:y1, x0:x1] > 0
        for axis, ek in ((0, "edge_h"), (1, "edge_v")):
            e = rot(maps[ek]).astype(bool) & inb
            br = rot(maps["bright"]).astype(bool) & inb
            span = inb.sum(axis=1 - axis).astype(np.float32)
            ef = e.sum(axis=1 - axis) / np.maximum(span, 1)
            bf = br.sum(axis=1 - axis) / np.maximum(span, 1)
            n = len(ef)
            k = 6
            for mid in range(int(min_side), int(n - min_side)):
                left = ef[max(0, mid - k): mid + 1].max()
                right = ef[mid: mid + k + 1].max()
                score = min(left, right) + 0.5 * bf[max(0, mid - 2): mid + 3].mean()
                if score > 0.6 and (best is None or score > best[0]):
                    best = (score, axis, mid)
    if best is None:
        return [blob]
    _, axis, mid = best
    a_, b_ = np.zeros_like(rb), np.zeros_like(rb)
    if axis == 0:
        a_[: y0 + mid] = rb[: y0 + mid]
        b_[y0 + mid:] = rb[y0 + mid:]
    else:
        a_[:, : x0 + mid] = rb[:, : x0 + mid]
        b_[:, x0 + mid:] = rb[:, x0 + mid:]
    Rinv = cv2.invertAffineTransform(R)
    out = []
    for part in (a_, b_):
        back = cv2.warpAffine(part, Rinv, (W, H), flags=cv2.INTER_NEAREST)[P:P + H0, P:P + W0]
        # trim the lid lane itself off each part before recursing
        back = cv2.morphologyEx(np.ascontiguousarray(back) & (1 - lidlike), cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        cs = _contours_of(back)
        if not cs:
            continue
        piece = np.zeros_like(blob)
        cv2.drawContours(piece, cs, -1, 1, cv2.FILLED)
        out.extend(_xy_cut(piece, lidlike, min_side, depth + 1, maps, max_long, max_aspect))
    return out or [blob]


def _find_seam(small: np.ndarray, blob: np.ndarray, min_side: float):
    """Look for the seam between two prints butted together inside `blob`: a straight line
    across the whole blob with an edge along (almost) all of it and a clear colour change
    across it. Returns two part masks or None. Interior lines of a single photo (horizons,
    door frames) are broken in places and rarely separate two different colour worlds."""
    c = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    (cx, cy), _, ang = cv2.minAreaRect(c)
    ang = ((ang + 45) % 90) - 45
    H0, W0 = blob.shape
    P = int(max(H0, W0) * np.sin(np.deg2rad(abs(ang)))) + 2
    R = cv2.getRotationMatrix2D((cx + P, cy + P), ang, 1.0)
    pad = lambda m, v=0: cv2.copyMakeBorder(m, P, P, P, P, cv2.BORDER_CONSTANT, value=v)
    H, W = H0 + 2 * P, W0 + 2 * P
    rb = cv2.warpAffine(pad(blob), R, (W, H), flags=cv2.INTER_NEAREST)
    ri = cv2.warpAffine(pad(small), R, (W, H), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(rb)
    y0, y1, x0, x1 = ys.min() + 3, ys.max() - 2, xs.min() + 3, xs.max() - 2
    crop = ri[y0:y1, x0:x1]
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    best = None
    for axis in (1, 0):  # 1: vertical seam (cut columns), 0: horizontal seam
        gr = cv2.Sobel(g, cv2.CV_32F, 1 if axis == 1 else 0, 0 if axis == 1 else 1, ksize=3)
        cont = (np.abs(gr) > 12).mean(axis=0 if axis == 1 else 1)
        n = len(cont)
        for k in range(max(int(min_side), 6), n - max(int(min_side), 6)):
            cc = cont[max(0, k - 2): k + 3].max()
            if cc < 0.85:
                continue
            Lh = lab[:, max(0, k - 25): k - 3] if axis == 1 else lab[max(0, k - 25): k - 3]
            Rh = lab[:, k + 3: k + 25] if axis == 1 else lab[k + 3: k + 25]
            d = float(np.linalg.norm(Lh.reshape(-1, 3).mean(0) - Rh.reshape(-1, 3).mean(0)))
            if d < 25:
                continue
            score = cc + d / 100.0
            if best is None or score > best[0]:
                best = (score, axis, k)
    if best is None:
        return None
    _, axis, k = best
    a_, b_ = np.zeros_like(rb), np.zeros_like(rb)
    if axis == 1:
        a_[:, : x0 + k] = rb[:, : x0 + k]
        b_[:, x0 + k:] = rb[:, x0 + k:]
    else:
        a_[: y0 + k] = rb[: y0 + k]
        b_[y0 + k:] = rb[y0 + k:]
    Rinv = cv2.invertAffineTransform(R)
    return [np.ascontiguousarray(cv2.warpAffine(m, Rinv, (W, H), flags=cv2.INTER_NEAREST)[P:P + H0, P:P + W0])
            for m in (a_, b_)]


def _split_outsized(contours: list, shape, small, min_side: float, ratio: float) -> list:
    areas = [cv2.contourArea(c) for c in contours]
    if sum(a >= min_side * min_side for a in areas) < 2:
        return contours
    out = []
    for i, (c, a) in enumerate(zip(contours, areas)):
        others = [b for j, b in enumerate(areas) if j != i and b >= min_side * min_side]
        if others and a > ratio * float(np.median(others)):
            blob = np.zeros(shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 1, cv2.FILLED)
            parts = _find_seam(small, blob, min_side)
            if parts:
                pcs = [pc for p in parts for pc in _contours_of(p)]

                def ok(pc):
                    (_, _), (w, h), _ = cv2.minAreaRect(pc)
                    return min(w, h) >= min_side and max(w, h) / max(min(w, h), 1) <= 2.0
                if len(pcs) == 2 and all(ok(pc) for pc in pcs):
                    out.extend(pcs)
                    continue
        out.append(c)
    return out


def _split_touching(c, shape, min_side: float, min_rect: float) -> list:
    """Split a blob made of several touching prints into one contour per print."""
    blob = np.zeros(shape, np.uint8)
    cv2.drawContours(blob, [c], -1, 1, cv2.FILLED)
    if _rectangularity(blob)[0] >= min_rect:
        return [c]
    out = []
    for part in _cut_split(blob, min_side, min_rect):  # shared straight edges (most common)
        if _rectangularity(part)[0] < min_rect:
            sub = _erosion_split(part, min_side)       # angled / corner contact
            if sub:
                for p_ in sub:
                    out.extend(_contours_of(p_))
                continue
        out.extend(_contours_of(part))
    return out or [c]


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points tl, tr, br, bl."""
    pts = pts.astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def _snap_rect(rect, edge_mag: np.ndarray, max_shift: int) -> tuple:
    """Move each side of a rotated rect (in/out) to where the edge response along it peaks."""
    (cx, cy), (rw, rh), ang = rect
    if max_shift <= 0:
        return rect
    t = np.deg2rad(ang)
    ux, uy = np.cos(t), np.sin(t)       # axis along rw
    vx, vy = -np.sin(t), np.cos(t)      # axis along rh
    h, w = edge_mag.shape

    def side_score(offset_u: float, offset_v: float, length: float, along_u: bool) -> float:
        n = max(8, int(length * 0.8))
        s = np.linspace(-length * 0.4, length * 0.4, n)
        if along_u:  # side parallel to u, located at v-offset
            xs = cx + ux * s + vx * offset_v
            ys = cy + uy * s + vy * offset_v
        else:
            xs = cx + vx * s + ux * offset_u
            ys = cy + vy * s + uy * offset_u
        xi = np.clip(np.round(xs).astype(int), 0, w - 1)
        yi = np.clip(np.round(ys).astype(int), 0, h - 1)
        return float(edge_mag[yi, xi].mean())

    shifts = range(-max_shift, max_shift + 1)
    # sides at +-rw/2 along u (parallel to v), +-rh/2 along v (parallel to u)
    best = {}
    for key, sign, along_u, base in (("r", 1, False, rw / 2), ("l", -1, False, rw / 2),
                                     ("b", 1, True, rh / 2), ("t", -1, True, rh / 2)):
        scores = []
        for d in shifts:
            off = sign * (base + d)
            if along_u:
                scores.append(side_score(0, off, rw, True))
            else:
                scores.append(side_score(off, 0, rh, False))
        best[key] = list(shifts)[int(np.argmax(scores))]
    new_rw = rw + best["r"] + best["l"]
    new_rh = rh + best["b"] + best["t"]
    du = (best["r"] - best["l"]) / 2.0
    dv = (best["b"] - best["t"]) / 2.0
    ncx = cx + ux * du + vx * dv
    ncy = cy + uy * du + vy * dv
    return ((ncx, ncy), (new_rw, new_rh), ang)


def _normalise_rect(rect) -> tuple:
    """Express rect with angle in (-45, 45]."""
    (cx, cy), (w, h), a = rect
    while a > 45:
        a -= 90
        w, h = h, w
    while a <= -45:
        a += 90
        w, h = h, w
    return ((cx, cy), (w, h), a)


def detect_prints(page: np.ndarray, page_dpi: float, cfg, blank: np.ndarray | None = None
                  ) -> tuple[list[Detection], dict]:
    """Return detections (full-res coordinates) and a debug/info dict."""
    dc = cfg.detect
    scale = float(dc.work_dpi) / float(page_dpi)
    H, W = page.shape[:2]
    sw, sh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    rgb = to_u8(page if page.ndim == 3 else np.repeat(page[..., None], 3, 2))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    blank_small = None
    if blank is not None:
        b = to_u8(blank if blank.ndim == 3 else np.repeat(blank[..., None], 3, 2))
        blank_small = cv2.resize(b, (sw, sh), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    f = gray.astype(np.float32)
    mu = cv2.blur(f, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (7, 7)) - mu * mu, 0))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gblur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gblur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gblur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Lid pixels: bright, flat, unsaturated. Prints often fill the page edges, so the lid
    # is estimated from wherever such pixels are (brightest half of the flat candidates).
    lid = _lid_pixels(gray, sd, hsv[..., 1])
    lid_sd = float(np.median(sd[lid])) if lid.any() else float(np.median(sd))
    lid_mag = float(np.median(mag[lid])) if lid.any() else float(np.median(mag))

    # --- cue 1: colour difference to lid model
    if blank_small is not None:
        bg = _background_model(small, blank_small)
    elif lid.any():
        bg = _fit_lid_surface(small, lid)
    else:
        bg = _background_model(small, None)
    diff = np.abs(small.astype(np.float32) - bg).max(axis=2)
    cue_diff = (diff > dc.bg_diff_thresh).astype(np.uint8)

    # --- cue 2: edges (gradient magnitude, includes the lamp shadow line)
    edges = (mag > max(12.0, lid_mag * 6)).astype(np.uint8)

    # --- cue 3: local texture (std-dev) and saturation
    cue_tex = ((sd > max(3.0, lid_sd * 3)) | (hsv[..., 1] > 40)).astype(np.uint8)

    fused = cue_diff | cue_tex
    # edges close the outline of white-bordered prints
    k = max(3, int(round(dc.work_dpi * 0.04)) | 1)
    outline = cv2.morphologyEx(edges | fused, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # Scanner-bed edge shadows run along the page border and would link prints that both
    # reach the edge, enclosing bare lid between them. Clear a thin border band first...
    bb = max(2, int(round(dc.work_dpi * 0.10)))
    inner = outline.copy()
    inner[:bb] = 0
    inner[-bb:] = 0
    inner[:, :bb] = 0
    inner[:, -bb:] = 0
    # fill enclosed regions (a print's white border + interior)
    ext, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(outline)
    cv2.drawContours(filled, ext, -1, 1, thickness=cv2.FILLED)

    # remove thin junk: dust, scanner-bed edge shadows along the page border
    ko = max(3, int(round(dc.work_dpi * float(dc.get("cleanup_in", 0.25)))) | 1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))
    # ...then extend prints that run off the glass straight out to the page edge
    filled[:bb, :] |= filled[bb:bb + 1, :]
    filled[-bb:, :] |= filled[-bb - 1:-bb, :]
    filled[:, :bb] |= filled[:, bb:bb + 1]
    filled[:, -bb:] |= filled[:, -bb - 1:-bb]

    min_side = dc.min_print_in * dc.work_dpi
    # lid-like = close to the lid colour and flat (looser than the lid model's own pixels,
    # so slightly shadowed gaps between prints still count)
    lidlike = ((diff < dc.bg_diff_thresh * 1.5) & (sd < max(4.0, lid_sd * 4))).astype(np.uint8)
    ethr = max(12.0, lid_mag * 6)
    maps = {"edge_h": (np.abs(gy) > ethr).astype(np.uint8),   # horizontal edges (vertical gradient)
            "edge_v": (np.abs(gx) > ethr).astype(np.uint8),
            "bright": (diff < dc.bg_diff_thresh * 2.5).astype(np.uint8)}
    contours = []
    for c in cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        blob = np.zeros_like(filled)
        cv2.drawContours(blob, [c], -1, 1, cv2.FILLED)
        for piece in _xy_cut(blob, lidlike, min_side, 0, maps,
                             dc.get("max_print_long_in", 9.5) * dc.work_dpi,
                             dc.get("max_print_aspect", 1.9)):   # grids / narrow gaps
            for pc in _contours_of(piece):
                contours.extend(_split_touching(pc, filled.shape, min_side, dc.min_rectangularity))
    # Two prints butted together with no gap look like one bigger print. If a region is much
    # bigger than the other prints on the same page, look for a straight seam across it and
    # accept the cut only if both halves have print-like proportions.
    contours = _split_outsized(contours, filled.shape, small, min_side,
                               float(dc.get("outsized_ratio", 1.5)))
    dets: list[Detection] = []
    page_notes: list[str] = []
    edge_norm = mag / (mag.max() + 1e-6)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_side * min_side:
            continue
        rect = _normalise_rect(cv2.minAreaRect(c))
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < min_side:
            continue
        # a region that is mostly bare lid inside is a shadow / lid-gradient patch, not a print
        cm = np.zeros_like(gray)
        cv2.drawContours(cm, [c], -1, 1, cv2.FILLED)
        lid_inside = float(lidlike[cm > 0].mean())
        if lid_inside > 0.6:
            page_notes.append("ignored_lid_patch")
            continue
        rectangularity = float(area / max(rw * rh, 1))
        notes: list[str] = []
        # touching the page edge => probably clipped by the scanner
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + bw >= sw - 1 or y + bh >= sh - 1:
            notes.append("touches_page_edge")
        rect = _snap_rect(rect, edge_norm, int(dc.edge_snap_px))
        (cx, cy), (rw, rh), ang = rect

        # confidence: rectangularity + edge support along the fitted outline + aspect sanity
        box = cv2.boxPoints(rect).astype(np.int32)
        ring = np.zeros_like(gray)
        cv2.polylines(ring, [box], True, 1, 2)
        ring[:3] = ring[-3:] = 0  # sides lying on the page edge have no edge to find
        ring[:, :3] = ring[:, -3:] = 0
        edge_support = float(edges[ring > 0].mean()) if ring.any() else 0.0
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        aspect_ok = 1.0 if aspect < 4 else 0.5
        conf = float(np.clip(0.55 * min(1, rectangularity / 0.97) + 0.45 * min(1, edge_support / 0.6),
                             0, 1) * aspect_ok)
        if rectangularity < dc.min_rectangularity:
            notes.append("non_rectangular_maybe_touching_prints")
            conf = min(conf, 0.5)

        quad_full = _order_quad(cv2.boxPoints(rect) / scale)
        dets.append(Detection(
            quad=quad_full.round(2).tolist(), angle=float(ang),
            width=float(rw / scale), height=float(rh / scale),
            confidence=round(conf, 3), rectangularity=round(rectangularity, 3), notes=notes,
        ))

    # reading order: top-to-bottom rows, left-to-right
    def key(d: Detection):
        q = np.array(d.quad)
        cy, cx = q[:, 1].mean(), q[:, 0].mean()
        return (round(cy / (page_dpi * 1.5)), cx)

    dets.sort(key=key)
    if not dets:
        page_notes.append("no_prints_found")
    lid_rgb = (np.median(small[lid], axis=0) if lid.any() else bg.reshape(-1, 3).mean(0))
    info = {
        "lid_rgb": [round(float(v), 1) for v in lid_rgb],
        "work_scale": scale,
        "n_found": len(dets),
        "min_confidence": min((d.confidence for d in dets), default=0.0),
        "notes": page_notes,
        "_debug_mask": filled,  # not serialised
        "_debug_small": small,
    }
    return dets, info


def debug_overlay(info: dict, dets: list[Detection]) -> np.ndarray:
    """Small RGB preview with detected quads drawn (for review sheets)."""
    small = info["_debug_small"].copy()
    s = info["work_scale"]
    for i, d in enumerate(dets):
        q = (np.array(d.quad) * s).astype(np.int32)
        col = (0, 200, 0) if d.confidence >= 0.6 else (230, 120, 0)
        cv2.polylines(small, [q], True, col, 2)
        cv2.putText(small, f"{i + 1}:{d.confidence:.2f}", tuple(q[0] + [4, 18]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return small
