"""manifest.csv + _review/index.html contact sheet."""
from __future__ import annotations

import csv
import html
import json
import logging
import shutil
from pathlib import Path

import cv2

from .io import read_image, remove_tree, to_u8

log = logging.getLogger(__name__)

FLAG_TEXT = {
    "low_detection_confidence": "Crop may be wrong (low detection confidence)",
    "non_rectangular_maybe_touching_prints": "Outline not rectangular - prints touching or torn",
    "touches_page_edge": "Print touches scan edge - may be clipped",
    "orientation_uncertain": "Unsure which way is up",
    "heavy_damage": "Large damaged area was inpainted - check for invented detail",
}


def _thumb(src: Path, dst: Path, px: int) -> None:
    arr, _ = read_image(src)
    arr = to_u8(arr)
    h, w = arr.shape[:2]
    s = px / max(h, w)
    t = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    if t.ndim == 3:
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), t, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_reports(out: Path, rows: list[dict], pages_dir: Path, cfg) -> None:
    out = Path(out)
    # ---------------- manifest
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "page", "index", "class", "orientation_deg_ccw", "orientation_method",
                    "orientation_margin", "tilt_deg", "detect_conf", "applied", "flags",
                    "repaired_from_paint", "master", "restored_tif", "restored_jpg"])
        for st in rows:
            o = st.get("orientation", {})
            d = st["detection"]
            outp = st.get("outputs", {})
            w.writerow([
                st["id"], st["page"], st["index"], st.get("colour", {}).get("class", ""),
                o.get("degrees_ccw", 0), o.get("method", ""), o.get("margin", ""),
                round(d["angle"] + st["geometry"].get("refine_deg", 0.0), 3), d["confidence"],
                ";".join(st.get("applied", [])), ";".join(st.get("flags", [])),
                "yes" if any(a.startswith("repair") for a in st.get("applied", [])) else "",
                outp.get("master", ""), outp.get("restored_tif", ""), outp.get("restored_jpg", ""),
            ])

    # ---------------- review sheet
    rev = out / "_review"
    if not remove_tree(rev):
        # OneDrive / Explorer holding the folder: carry on, leftovers get overwritten.
        log.warning("could not fully clear %s (in use by another program?) - continuing", rev)
    flagged = [st for st in rows if st.get("flags")]
    repaired = [st for st in rows if any(a.startswith("repair") for a in st.get("applied", []))]
    page_flags = []
    for pj in sorted(Path(pages_dir).glob("*.json")):
        info = json.loads(pj.read_text())
        if info.get("flags"):
            page_flags.append((info, pj.with_name(pj.stem + "_detect.jpg")))
    if not flagged and not page_flags and not repaired:
        return
    rev.mkdir(parents=True, exist_ok=True)
    px = int(cfg.review.thumb_px)
    parts = [
        "<!doctype html><meta charset=utf-8><title>MATTRESS review</title>",
        "<style>body{font:14px system-ui;margin:16px;background:#f6f6f4;color:#222}"
        ".c{display:inline-block;vertical-align:top;margin:8px;padding:8px;background:#fff;"
        "border-radius:6px;box-shadow:0 1px 3px #0002;max-width:760px}img{max-height:%dpx;margin:2px}"
        "li{color:#a33}</style>" % px,
        f"<h1>MATTRESS review - {len(flagged)} photo(s), {len(page_flags)} page(s)</h1>",
    ]
    if page_flags:
        parts.append("<h2>Pages</h2>")
        for info, ov in page_flags:
            if ov.exists():
                shutil.copy(ov, rev / ov.name)
            reasons = "".join(f"<li>{html.escape(r)}</li>" for r in info["flags"])
            parts.append(f"<div class=c><b>{html.escape(info['page'])}</b> - {info['n']} print(s) found"
                         f"<ul>{reasons}</ul><img src='{html.escape(ov.name)}'></div>")
    if repaired:
        parts.append("<h2>Repaired from your paint</h2><p>Left: your paint outline (magenta); orange = "
                     "regenerated / filled, blue = recoloured, green outline = person, left alone. "
                     "Right: result. Not happy? Repaint and rerun.</p>")
        for st in repaired:
            chk = out / "paint" / f"{st['id']}_repair_check.jpg"
            if chk.is_file():
                shutil.copy(chk, rev / chk.name)
                parts.append(f"<div class=c><b>{html.escape(st['id'])}</b><br>"
                             f"<img style='max-height:520px' src='{html.escape(chk.name)}'></div>")
    if flagged:
        parts.append("<h2>Photos</h2>")
        for st in flagged:
            outs = st.get("outputs", {})
            m = out / outs.get("master", "")
            r = out / outs.get("restored_jpg", "")
            imgs = ""
            for label, src in (("before", m), ("after", r)):
                if src.is_file():
                    t = rev / "thumbs" / f"{st['id']}_{label}.jpg"
                    _thumb(src, t, px)
                    imgs += f"<img title={label} src='thumbs/{t.name}'>"
            if cfg.review.copy_flagged and r.is_file():
                shutil.copy(r, rev / r.name)
            reasons = "".join(f"<li>{html.escape(FLAG_TEXT.get(f, f))}</li>" for f in st["flags"])
            applied = ", ".join(st.get("applied", [])) or "none"
            parts.append(f"<div class=c><b>{html.escape(st['id'])}</b> ({html.escape(st['page'])})"
                         f"<ul>{reasons}</ul><small>applied: {html.escape(applied)}</small><br>{imgs}</div>")
    (rev / "index.html").write_text("\n".join(parts), encoding="utf-8")
