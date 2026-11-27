"""Crop + deskew a detected print from the full-resolution page in a single resample."""
from __future__ import annotations

import cv2
import numpy as np

from .io import to_u8


def _quad_size(q: np.ndarray) -> tuple[int, int]:
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return int(round(w)), int(round(h))


def warp_quad(page: np.ndarray, quad: np.ndarray, trim_pct: float = 0.0) -> np.ndarray:
    """Perspective-warp `quad` (tl,tr,br,bl) out of `page` into an upright rectangle.

    The trim is applied in the transform itself, so pixels are only resampled once.
    """
    q = np.asarray(quad, np.float32)
    w, h = _quad_size(q)
    t = trim_pct / 100.0
    tw, th = w * t, h * t
    ow, oh = max(1, int(round(w - 2 * tw))), max(1, int(round(h - 2 * th)))
    src_rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    M_rect2page = cv2.getPerspectiveTransform(src_rect, q)
    # output pixel (x, y) -> rect coord (x + tw, y + th) -> page
    shift = np.array([[1, 0, tw], [0, 1, th], [0, 0, 1]], np.float64)
    M = M_rect2page @ shift
    return cv2.warpPerspective(page, M, (ow, oh), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def rotate_quad(quad: np.ndarray, deg: float) -> np.ndarray:
    q = np.asarray(quad, np.float64)
    c = q.mean(0)
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return ((q - c) @ R.T + c).astype(np.float32)


def residual_tilt(crop: np.ndarray, dpi: float, max_deg: float) -> float | None:
    """Estimate remaining tilt (deg) of an already-cropped print from straight lines.

    Uses long Hough segments (print border, horizons, door frames...) within +-max_deg of
    horizontal/vertical. Returns None if there is not enough evidence.
    """
    g = to_u8(crop)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    s = min(1.0, 150.0 / dpi)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    min_len = int(min(g.shape) * 0.25)
    lines = cv2.HoughLinesP(e, 1, np.pi / 1800, threshold=60, minLineLength=min_len, maxLineGap=5)
    if lines is None:
        return None
    angs, wts = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 4: Nx1x4, 5: Nx4
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # map to deviation from nearest axis
        dev = ((a + 45) % 90) - 45
        if abs(dev) <= max_deg:
            angs.append(dev)
            wts.append(np.hypot(x2 - x1, y2 - y1))
    if len(angs) < 2:
        return None
    angs, wts = np.array(angs), np.array(wts)
    order = np.argsort(angs)
    cum = np.cumsum(wts[order])
    med = float(angs[order][np.searchsorted(cum, cum[-1] / 2)])  # weighted median
    spread = float(np.average(np.abs(angs - med), weights=wts))
    if spread > 0.6:  # lines disagree - unreliable
        return None
    return med


def _intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.asarray(p2, np.float64) - np.asarray(p1, np.float64))
    return np.asarray(p1, np.float64) + t[0] * np.asarray(d1, np.float64)


def refine_quad(page: np.ndarray, quad, lid_rgb, dpi: float, search_in: float = 0.12,
                lid_tol: float = 28.0) -> tuple[np.ndarray, dict]:
    """Snap each side of a detected quad to the exact lid/print boundary at full resolution.

    For each side ~60 short profiles are sampled across the edge (+-search_in inches). On
    each, the print edge is the first non-lid pixel run met coming in from the lid. A line is
    fitted through those points (robustly). Sides with no lid outside them (prints touching,
    page border) fall back to the strongest edge or are left alone.
    """
    q = np.asarray(quad, np.float64)
    H, W = page.shape[:2]
    rgb = to_u8(page)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, 2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lid = np.asarray(lid_rgb, np.float32)
    c = q.mean(0)
    R = max(8, int(search_in * dpi))          # search outward
    Ri = max(R, int(0.4 * dpi))               # search further inward: coarse detection can
    offs = np.arange(R, -Ri - 1, -1, dtype=np.float32)  # over-reach by a few tenths of an inch
    lines, how = [], []
    border = max(4.0, dpi * 0.015)
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        d = p1 - p0
        L = float(np.hypot(*d))
        u = d / max(L, 1e-6)
        n = np.array([u[1], -u[0]])
        if np.dot((p0 + p1) / 2 - c, n) < 0:
            n = -n  # outward normal
        # near the edge of the glass: a thin lid strip may still separate print and border,
        # so search for lid as usual - but never fall back to an edge inside the picture
        on_border = all(min(pt[0], pt[1], W - 1 - pt[0], H - 1 - pt[1]) < border for pt in (p0, p1))
        ts = np.linspace(0.08, 0.92, 60)
        base = p0[None, :] + ts[:, None] * d[None, :]
        pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
        mx, my = pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32)
        prof = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        inside_img = (mx >= 0) & (my >= 0) & (mx <= W - 1) & (my <= H - 1)
        lidlike = (np.abs(prof - lid).max(-1) < lid_tol) & inside_img
        edge_pts, edge_t = [], []
        mid = R  # index of offset 0 (the detected side)
        for j in range(len(ts)):
            ll = lidlike[j]
            # lid runs along the profile (index 0 = farthest outside); a neighbouring print
            # may lie beyond the gap, so use the lid run NEAREST the detected side
            runs, k0 = [], None
            for k in range(len(ll) + 1):
                if k < len(ll) and ll[k]:
                    k0 = k if k0 is None else k0
                elif k0 is not None:
                    if k - k0 >= 2:
                        runs.append((k0, k))
                    k0 = None
            # a lid run must reach the detected side or lie outside it: white areas that
            # start well inside the photo (walls, sky) are picture, not lid
            tol = max(3, int(0.03 * dpi))
            runs = [r for r in runs if r[0] <= mid + tol]
            if not runs:
                continue                       # no lid near this point (touching print)
            a_, b_ = min(runs, key=lambda r: 0 if r[0] <= mid < r[1] else min(abs(r[0] - mid), abs(r[1] - mid)))
            if b_ >= len(ll) - 1:
                continue                       # lid all the way in: not this print's edge
            edge_pts.append(base[j] + offs[b_] * n)  # first non-lid pixel inside the run
            edge_t.append(ts[j])
        # enough support: 30% of samples, or 15% spread along >=25% of the side (a tilted
        # print crossing the edge of the glass only shows lid along part of that side)
        spread = (max(edge_t) - min(edge_t)) if edge_t else 0.0
        if len(edge_pts) >= 0.3 * len(ts) or (len(edge_pts) >= 0.15 * len(ts) and spread >= 0.25):
            ep = np.array(edge_pts, np.float32)
            off_along = (ep - p0) @ n
            med = np.median(off_along)
            keep = np.abs(off_along - med) < max(3.0, dpi * 0.01)
            if keep.sum() >= 0.12 * len(ts):
                vx, vy, x0, y0 = cv2.fitLine(ep[keep], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                dirv = np.array([vx, vy], np.float64)
                if np.dot(dirv, u) < 0:
                    dirv = -dirv
                lines.append((np.array([x0, y0], np.float64), dirv))
                how.append("lid")
                continue
        if on_border:  # print runs off the glass: keep the detected (page) edge
            lines.append((p0, u))
            how.append("page_edge")
            continue
        # fallback: strongest step edge in the averaged gray profile (touching prints)
        gp = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(0)
        grad = np.abs(np.gradient(cv2.GaussianBlur(gp.reshape(1, -1), (0, 0), 1.0).ravel()))
        win = slice(mid - R // 2, mid + R // 2 + 1)
        kk = int(np.argmax(grad[win])) + win.start
        if grad[kk] > 4.0:
            lines.append((p0 + offs[kk] * n, u))
            how.append("edge")
        else:
            lines.append((p0, u))
            how.append("unchanged")
    corners = []
    for i in range(4):
        pa, da = lines[(i - 1) % 4]
        pb, db = lines[i]
        x = _intersect(pa, da, pb, db)
        corners.append(x if x is not None else q[i])
    nq = np.array(corners, np.float64)
    if np.max(np.linalg.norm(nq - q, axis=1)) > 1.6 * (R + Ri):  # implausible: keep detection
        return q.astype(np.float32), {"sides": how, "rejected": True}
    return nq.astype(np.float32), {"sides": how}


def crop_print(page: np.ndarray, quad, dpi: float, trim_pct: float, refine_max_deg: float,
               lid_rgb=None) -> tuple[np.ndarray, dict]:
    q = np.asarray(quad, np.float32)
    info = {"refine_deg": 0.0}
    snapped = False
    if lid_rgb is not None:
        q, rinfo = refine_quad(page, q, lid_rgb, dpi)
        info["edges"] = rinfo
        snapped = not rinfo.get("rejected") and sum(h in ("lid", "edge") for h in rinfo["sides"]) >= 2
    if not snapped:  # no precise edges: straighten from lines in the picture instead
        crop = warp_quad(page, q, 0.0)
        tilt = residual_tilt(crop, dpi, refine_max_deg)
        if tilt is not None and 0.05 < abs(tilt) <= refine_max_deg:
            q = rotate_quad(q, tilt)
            info["refine_deg"] = round(tilt, 3)
    crop = warp_quad(page, q, trim_pct)
    info["final_quad"] = np.round(q, 2).tolist()
    return crop, info
