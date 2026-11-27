"""Stage-major orchestration with per-photo on-disk state (resumable).

Each GPU model is loaded once per run, used for every photo that needs it (one image per
forward pass), then unloaded before the next model - only one model is ever in VRAM.

Layout of OUT:
  masters/<id>_master.tif        untouched crop + deskew + orientation, 16-bit, dpi/ICC kept
  restored/<id>_<colour|bw>.tif  restored, 16-bit (gray = single channel)
  restored/<id>_<colour|bw>.jpg  access copy
  restored/<id>.json             sidecar: everything decided for this photo
  manifest.csv                   one row per photo
  _review/index.html             flagged photos/pages with reasons
  _work/                         intermediate state (safe to delete after a run)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import __version__
from .colorclass import classify, to_neutral_gray
from .detect import debug_overlay, detect_prints
from .geometry import crop_print
from .io import _replace, list_images, read_image, to_float, to_u16, to_rgb, write_jpeg, write_npy, write_tiff

log = logging.getLogger("mattress")

# dust removal (inpaint) runs BEFORE paint repair: dust under a stain is removed first, so
# recolouring neither measures nor amplifies it; filled areas are regenerated anyway
STAGES = ["split", "orient", "classify", "assess", "tone", "inpaint", "repair", "upscale", "faces",
          "finalize"]
REDO_FROM = "tone"  # stages from here on are redone when a paint file changes


def _people_lines_only(mask: np.ndarray, person: np.ndarray) -> tuple[np.ndarray, int]:
    """On people, keep only long thin damage (real scratches). Specks and blobs there are
    too often clothing patterns (eyelet knit, polka dots), buttons or catchlights."""
    H, W = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1 or not person.any():
        return mask, 0
    on_person = np.bincount(lab.ravel(), weights=person.ravel().astype(np.float64), minlength=n)
    frac = on_person / np.maximum(st[:, 4], 1)
    keep = np.ones(n, bool)
    keep[0] = False
    min_len = max(H, W) * 0.02
    dropped = 0
    for i in range(1, n):
        if frac[i] < 0.3:
            continue
        ys, xs = np.nonzero(lab[st[i, 1]: st[i, 1] + st[i, 3], st[i, 0]: st[i, 0] + st[i, 2]] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        long_, short = max(rw, rh), max(min(rw, rh), 1.0)
        if not (long_ >= min_len and long_ / short >= 6):
            keep[i] = False
            dropped += 1
    return keep[lab].astype(np.uint8), dropped


class Run:
    def __init__(self, cfg, in_dir: Path, out_dir: Path, blank: Path | None = None,
                 force: bool = False, keep_work: bool = False):
        self.cfg = cfg
        self.in_dir = Path(in_dir)
        self.out = Path(out_dir)
        self.work = self.out / "_work"
        self.blank_path = blank
        self.force = force
        self.keep_work = keep_work
        self.models_dir = Path(cfg.models_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self._device = None

    # ------------------------------------------------------------------ state helpers
    @property
    def device(self):
        if self._device is None:
            from .restore.models import resolve_device
            self._device = resolve_device(self.cfg.device)
            log.info("device: %s", self._device)
        return self._device

    def _state_path(self, pid: str) -> Path:
        return self.work / pid / "state.json"

    def load_state(self, pid: str) -> dict:
        return json.loads(self._state_path(pid).read_text())

    def save_state(self, st: dict) -> None:
        p = self._state_path(st["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_text(json.dumps(st, indent=2))
        _replace(tmp, p)  # retries: antivirus / indexer can hold the file for a moment

    def photos(self) -> list[dict]:
        out = []
        for p in sorted(self.work.glob("*/state.json")):
            out.append(json.loads(p.read_text()))
        return out

    def _img(self, st: dict) -> np.ndarray:
        return to_float(np.load(self.work / st["id"] / st["current"]))

    def _put(self, st: dict, stage: str, x: np.ndarray) -> None:
        name = f"{stage}.npy"
        write_npy(self.work / st["id"] / name, to_u16(x))
        old = st.get("current")
        st["current"] = name
        if old and old != name and old not in ("crop.npy", "base.npy"):
            (self.work / st["id"] / old).unlink(missing_ok=True)

    def _done(self, st: dict, stage: str) -> bool:
        return stage in st.get("done", []) and not self.force

    def _mark(self, st: dict, stage: str) -> None:
        st.setdefault("done", [])
        if stage not in st["done"]:
            st["done"].append(stage)
        self.save_state(st)

    @staticmethod
    def _flag(st: dict, reason: str) -> None:
        st.setdefault("flags", [])
        if reason not in st["flags"]:
            st["flags"].append(reason)

    # ------------------------------------------------------------------ stages
    def stage_split(self) -> None:
        pages = list_images(self.in_dir)
        blank = None
        if self.blank_path:
            blank, _ = read_image(self.blank_path)
            pages = [p for p in pages if p.resolve() != Path(self.blank_path).resolve()]
        pages_dir = self.work / "_pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for page_path in tqdm(pages, desc="split", unit="page"):
            marker = pages_dir / f"{page_path.stem}.json"
            fp = {"size": page_path.stat().st_size, "mtime": round(page_path.stat().st_mtime, 3)}
            if marker.exists() and not self.force:
                if json.loads(marker.read_text()).get("source") == fp:
                    continue
                log.info("%s changed since it was processed - redoing its photos", page_path.name)
                self._forget_page(page_path.name)
            page, meta = read_image(page_path)
            dpi = self._sane_dpi(meta.dpi, page.shape, page_path.name)
            if blank is not None and blank.shape[:2] != page.shape[:2]:
                log.warning("blank scan size differs from %s - ignoring blank", page_path.name)
                b = None
            else:
                b = blank
            dets, info = detect_prints(page, dpi, self.cfg, b)
            page_flags = list(info["notes"])
            if any(d.confidence < self.cfg.detect.min_confidence for d in dets):
                page_flags.append("low_detection_confidence")
            overlay_path = pages_dir / f"{page_path.stem}_detect.jpg"
            write_jpeg(overlay_path, debug_overlay(info, dets), None, 85)
            for i, d in enumerate(dets, 1):
                pid = f"{page_path.stem}_p{i:02d}"
                crop, ginfo = crop_print(page, d.quad, dpi, self.cfg.geometry.trim_pct,
                                         self.cfg.geometry.refine_max_deg, info["lid_rgb"])
                st = {
                    "id": pid, "page": page_path.name, "index": i, "dpi": dpi,
                    "icc": meta.icc.hex() if meta.icc else None, "src_dtype": meta.src_dtype,
                    "detection": d.to_dict(), "geometry": ginfo, "flags": [], "done": [],
                    "page_overlay": str(overlay_path.relative_to(self.out)),
                }
                if d.confidence < self.cfg.detect.min_confidence:
                    self._flag(st, "low_detection_confidence")
                for n in d.notes:
                    self._flag(st, n)
                write_npy(self.work / pid / "crop.npy", crop if crop.dtype == np.uint16 else to_u16(to_float(crop)))
                st["current"] = "crop.npy"
                self._mark(st, "split")
            marker.write_text(json.dumps({"page": page_path.name, "source": fp, "dpi": dpi, "n": len(dets),
                                          "flags": page_flags,
                                          "detections": [d.to_dict() for d in dets]}, indent=2))
            if not dets:
                log.warning("%s: no prints found", page_path.name)

    def _retire_paint(self, pid: str, reason: str) -> None:
        """Move a photo's paint file aside: unpainted ones are dropped, painted ones are kept
        in paint/_old so the user's work is never lost."""
        from .paint import find_paint_file, has_paint

        path = find_paint_file(self.paint_dir, pid)
        if path is None:
            return
        if has_paint(path):
            old = self.paint_dir / "_old"
            old.mkdir(parents=True, exist_ok=True)
            path.replace(old / path.name)
            log.warning("%s: %s - your paint was moved to paint/_old/%s; please repaint the new "
                        "paint file", pid, reason, path.name)
        else:
            path.unlink()
        for extra in (f"{pid}_repair_check.jpg",):
            (self.paint_dir / extra).unlink(missing_ok=True)

    def _forget_page(self, page_name: str) -> None:
        """Remove every photo (state + outputs) that came from this page."""
        import shutil

        for st in self.photos():
            if st.get("page") != page_name:
                continue
            pid = st["id"]
            self._retire_paint(pid, "its page scan was replaced")
            shutil.rmtree(self.work / pid, ignore_errors=True)
            for f in list((self.out / "masters").glob(f"{pid}_*")) + list((self.out / "restored").glob(f"{pid}_*")) \
                    + list((self.out / "restored").glob(f"{pid}.json")):
                f.unlink(missing_ok=True)

    def _sane_dpi(self, tagged: float | None, shape, name: str) -> float:
        """Scanner software often writes a bogus dpi tag (72/96/144). A flatbed bed is at most
        ~9 x 14 in, so a tag implying a bigger page is ignored in favour of assumed_dpi."""
        assumed = float(self.cfg.detect.assumed_dpi)
        if not tagged:
            return assumed
        long_in = max(shape[:2]) / tagged
        if long_in > 15.0:
            log.warning("%s: dpi tag %.0f implies a %.0f-inch page - using %.0f dpi",
                        name, tagged, long_in, assumed)
            return assumed
        return float(tagged)

    def stage_orient(self) -> None:
        todo = [s for s in self.photos() if not self._done(s, "orient")]
        if not todo:
            return
        orienter = None
        if self.cfg.orient.enabled:
            from .orient import Orienter
            orienter = Orienter(self.cfg, self.models_dir, self.device)
        for st in tqdm(todo, desc="orient", unit="photo"):
            raw = np.load(self.work / st["id"] / "crop.npy")
            if orienter is not None:
                dec = orienter.decide(raw)
                if not dec["confident"]:
                    self._flag(st, "orientation_uncertain")
            else:
                dec = {"k": 0, "degrees_ccw": 0, "method": "disabled", "confident": True}
            st["orientation"] = dec
            up = np.ascontiguousarray(np.rot90(raw, dec["k"]))
            write_npy(self.work / st["id"] / "base.npy", up)
            st["current"] = "base.npy"
            icc = bytes.fromhex(st["icc"]) if st.get("icc") else None
            write_tiff(self.out / "masters" / f"{st['id']}_master.tif", up, st["dpi"], icc)
            self._ensure_paint_file(st, up)
            self._mark(st, "orient")
        if orienter is not None:
            from .restore.models import free_gpu
            orienter.close()
            orienter = None
            free_gpu()

    # ------------------------------------------------------------------ paint files
    @property
    def paint_dir(self) -> Path:
        return self.out / "paint"

    def _ensure_paint_file(self, st: dict, master: np.ndarray | None = None) -> None:
        """Write a fresh paint file if none exists (never overwrite the user's)."""
        from .paint import file_sha, find_paint_file, make_paint_file

        from .paint import template_size

        pc = self.cfg.paint
        if not pc.enabled:
            return
        if master is None:
            master, _ = read_image(self.out / "masters" / f"{st['id']}_master.tif")
        existing = find_paint_file(self.paint_dir, st["id"])
        if existing is not None:
            im = cv2.imread(str(existing))
            ok = im is not None and (im.shape[1], im.shape[0]) in (
                template_size(master.shape, float(pc.scale)), (master.shape[1], master.shape[0]))
            if ok:
                return
            self._retire_paint(st["id"], "the photo's crop changed size")
            st.setdefault("paint", {}).pop("sha", None)
        path = make_paint_file(master, st["id"], self.paint_dir, float(pc.scale), int(pc.jpeg_quality))
        st.setdefault("paint", {})["sha"] = file_sha(path)
        self.save_state(st)

    def _check_paint(self) -> None:
        """Pick up paint files edited since the last run; queue those photos for
        reprocessing from the tone stage."""
        from .paint import PaintFileError, file_sha, find_paint_file, read_paint_mask

        if not self.cfg.paint.enabled:
            return
        for st in self.photos():
            if "orient" not in st.get("done", []):
                continue  # master not written yet
            self._ensure_paint_file(st)
            st = self.load_state(st["id"])
            path = find_paint_file(self.paint_dir, st["id"])
            if path is None:
                continue
            sha = file_sha(path)
            pinfo = st.setdefault("paint", {})
            if pinfo.get("sha") == sha:
                continue
            master, _ = read_image(self.out / "masters" / f"{st['id']}_master.tif")
            try:
                mask = read_paint_mask(path, master.shape, float(self.cfg.paint.scale))
            except PaintFileError as e:
                log.error("%s", e)
                continue
            has = bool(mask.any())
            pinfo["sha"] = sha
            if has or pinfo.get("has_mask"):
                if has:
                    cv2.imwrite(str(self.work / st["id"] / "paint_mask.png"), mask * 255)
                else:
                    (self.work / st["id"] / "paint_mask.png").unlink(missing_ok=True)
                pinfo["has_mask"] = has
                pinfo["file"] = path.name
                write_npy(self.work / st["id"] / "base.npy", master)
                old = st.get("current")
                st["current"] = "base.npy"
                if old and old not in ("base.npy", "crop.npy"):
                    (self.work / st["id"] / old).unlink(missing_ok=True)
                cut = STAGES.index(REDO_FROM)
                st["done"] = [d for d in st.get("done", []) if STAGES.index(d) < cut]
                st["applied"] = []
                st.pop("out_dpi", None)
                if st.get("plan", {}).get("faces") is not None:
                    st["plan"]["faces"] = {"run": bool(self.cfg.faces.enabled)}
                log.info("%s: paint %s - reprocessing", st["id"], "changed" if has else "removed")
            self.save_state(st)

    def stage_classify(self) -> None:
        for st in tqdm([s for s in self.photos() if not self._done(s, "classify")],
                       desc="classify", unit="photo"):
            st["colour"] = classify(np.load(self.work / st["id"] / "base.npy"), self.cfg)
            self._mark(st, "classify")

    def stage_assess(self) -> None:
        from .facedet import FaceDetector
        from .restore.damage import damage_mask
        from .restore.tone import fade_metrics, needs_tone

        det = FaceDetector(self.models_dir, score=0.6)
        sdet = None
        todo_assess = [s for s in self.photos() if not self._done(s, "assess")]
        if todo_assess and self.cfg.scratches.enabled:
            from .restore import scratch_detect
            sdet = scratch_detect.load(self.models_dir, self.device)
        for st in tqdm([s for s in self.photos() if not self._done(s, "assess")],
                       desc="assess", unit="photo"):
            x = to_rgb(to_float(np.load(self.work / st["id"] / "base.npy")))
            cls = st["colour"]["class"]
            plan: dict = {}
            fm = fade_metrics(x)
            reasons = needs_tone(fm, cls, self.cfg.fade) if self.cfg.fade.enabled else []
            plan["tone"] = {"run": bool(reasons) or cls != "colour", "reasons": reasons, "metrics": fm}
            if self.cfg.damage.enabled:
                faces = det.detect(x, work_long=2400)
                boxes = [tuple(float(v) for v in f[:4]) for f in faces]
                mask, ds = damage_mask(x, st["dpi"], self.cfg.damage, protect=boxes)
                ds["protected_faces"] = len(boxes)
                if sdet is not None:  # scratches, folds, cracks (learned detector)
                    from .restore import scratch_detect
                    smask, sstats = scratch_detect.detect(sdet, x, self.device, self.cfg.scratches, boxes)
                    mask = mask | smask
                    ds.update(sstats)
                if self.cfg.damage.get("people_lines_only", True):
                    from .restore.generative import person_mask
                    mask, dropped = _people_lines_only(mask, person_mask(x, self.models_dir))
                    ds["dropped_on_people"] = dropped
                ds["mask_frac"] = round(float(mask.mean()), 6)
                run = ds["mask_frac"] > self.cfg.damage.min_area
                if ds["mask_frac"] > self.cfg.damage.max_area_review:
                    self._flag(st, "heavy_damage")
                if run:
                    cv2.imwrite(str(self.work / st["id"] / "damage_mask.png"), mask * 255)
                plan["inpaint"] = {"run": run, **ds}
            else:
                plan["inpaint"] = {"run": False}
            long_edge = max(x.shape[:2])
            up = self.cfg.upscale
            plan["upscale"] = {"run": bool(up.enabled and long_edge < up.max_long_edge),
                               "long_edge": long_edge}
            plan["faces"] = {"run": bool(self.cfg.faces.enabled)}  # faces are chosen at final res
            st["plan"] = plan
            self._mark(st, "assess")
        if sdet is not None:  # release the detector before the next GPU stage
            from .restore.models import free_gpu
            sdet = None
            free_gpu()

    def stage_tone(self) -> None:
        from .restore.tone import correct_tone

        for st in tqdm([s for s in self.photos() if not self._done(s, "tone")], desc="tone", unit="photo"):
            p = st["plan"]["tone"]
            if p["run"]:
                cls = st["colour"]["class"]
                x = to_rgb(self._img(st))
                if cls != "colour":
                    x = to_rgb(to_neutral_gray(x))
                self._put(st, "tone", correct_tone(x, cls, self.cfg.fade, p["reasons"]))
                st.setdefault("applied", []).append("tone")
            self._mark(st, "tone")

    def _gpu_stage(self, stage: str, loader, fn) -> None:
        todo = [s for s in self.photos() if not self._done(s, stage)]
        need = [s for s in todo if s["plan"][stage]["run"]]
        model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if need:
                t0 = time.time()
                model = loader()
                log.info("%s model loaded in %.1fs", stage, time.time() - t0)
            for st in tqdm(todo, desc=stage, unit="photo"):
                if st["plan"][stage]["run"]:
                    fn(model, st)
                self._mark(st, stage)
        finally:
            if model is not None:
                from .restore.models import free_gpu
                model = None
                free_gpu()
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    log.info("%s: peak VRAM used by models %.2f GB (GPU now %.2f of %.1f GB used)",
                             stage, torch.cuda.max_memory_allocated() / 1e9,
                             (total - free) / 1e9, total / 1e9)

    def stage_repair(self) -> None:
        """Repair large damage marked by the user in a paint file."""
        from .restore.models import load_lama
        from .restore.repair import check_image, fill_lost, recolour, refine_damage

        rc = self.cfg.repair
        gen = rc.method == "generative"
        todo = [s for s in self.photos() if not self._done(s, "repair")]
        if not todo:
            return
        det = None
        # CPU first: refine the rough paint into recolour / fill (or regenerate) masks
        for st in todo:
            pm_path = self.work / st["id"] / "paint_mask.png"
            if not (st.get("paint", {}).get("has_mask") and pm_path.exists()):
                st["plan"]["repair"] = {"run": False}
                self.save_state(st)
                continue
            x = to_rgb(self._img(st))
            paint = self._load_mask(pm_path, x.shape)
            rec, fill, stats = refine_damage(x, paint, rc)
            cv2.imwrite(str(self.work / st["id"] / "repair_recolour.png"), rec * 255)
            cv2.imwrite(str(self.work / st["id"] / "repair_fill.png"), fill * 255)
            stats["method"] = rc.method
            if gen:
                from .facedet import FaceDetector
                from .restore.generative import protect_mask

                det = det or FaceDetector(self.models_dir, score=0.6)
                prot = protect_mask(x, det.detect(x, work_long=2400), self.models_dir,
                                    float(rc.protect_margin))
                r = max(1, int(max(x.shape[:2]) * float(rc.gen_grow)))
                if rc.get("gen_area", "damage") == "paint":   # regenerate everything painted
                    regen = paint & (1 - prot)
                else:                                          # only what looks damaged
                    regen = cv2.dilate(rec | fill, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) & paint & (1 - prot)
                cv2.imwrite(str(self.work / st["id"] / "repair_regen.png"), regen * 255)
                cv2.imwrite(str(self.work / st["id"] / "repair_protect.png"), prot * 255)
                stats["regen_frac"] = round(float(regen.mean()), 4)
                if regen.any():
                    stats.update(self._repair_prompt(st, x, regen, prot))
                stats["protected_overlap_frac"] = round(float((paint & prot).mean()), 4)
                st["plan"]["repair"] = {"run": bool(regen.any()), **stats}
            else:
                st["plan"]["repair"] = {"run": bool(rec.any() or fill.any()), **stats}
            log.info("%s: paint covers %.1f%% of photo; damage found in %.0f%% of the paint "
                     "(recolour %.1f%%, fill %.1f%% of photo)", st["id"],
                     stats.get("paint_frac", 0) * 100, stats.get("damage_frac_of_paint", 0) * 100,
                     stats.get("recolour_frac", 0) * 100, stats.get("fill_frac", 0) * 100)
            self.save_state(st)

        def fn_gen(pipe, st):
            from .restore.generative import DEFAULT_PROMPT, DEFAULT_PROMPT_BW, generative_fill

            d = self.work / st["id"]
            x = to_rgb(self._img(st))
            regen = self._load_mask(d / "repair_regen.png", x.shape)
            prot = self._load_mask(d / "repair_protect.png", x.shape)
            paint = self._load_mask(d / "paint_mask.png", x.shape)
            prompt = st["plan"]["repair"].get("prompt") or (
                DEFAULT_PROMPT if st["colour"]["class"] == "colour" else DEFAULT_PROMPT_BW)
            y = generative_fill(pipe, x, regen, prompt, rc, st["id"])
            cv2.imwrite(str(d / "repair_mask.png"), regen * 255)
            write_jpeg(self.paint_dir / f"{st['id']}_repair_check.jpg",
                       check_image(x, y, paint, np.zeros_like(regen), regen, protect=prot & paint), None, 90)
            self._put(st, "repair", y)
            st.setdefault("applied", []).append("repair_generative")

        def fn(model, st):
            if gen:
                return fn_gen(model, st)
            d = self.work / st["id"]
            x = to_rgb(self._img(st))
            rec = self._load_mask(d / "repair_recolour.png", x.shape)
            fill = self._load_mask(d / "repair_fill.png", x.shape)
            paint = self._load_mask(d / "paint_mask.png", x.shape)
            dmg = rec | fill
            y = recolour(x, rec, dmg, rc)
            good = (1 - cv2.dilate(dmg, np.ones((9, 9), np.uint8))).astype(np.uint8)
            y = fill_lost(model, y, fill, good, self.device, rc)
            cv2.imwrite(str(d / "repair_mask.png"), dmg * 255)
            write_jpeg(self.paint_dir / f"{st['id']}_repair_check.jpg",
                       check_image(x, y, paint, rec, fill), None, 90)
            self._put(st, "repair", y)
            st.setdefault("applied", []).append("repair")

        if gen:
            from .restore.generative import load_sdxl
            loader = lambda: load_sdxl(self.models_dir, self.device, bool(rc.get("low_vram", True)))
        else:
            loader = lambda: load_lama(self.models_dir, self.device)
        self._gpu_stage("repair", loader, fn)

    def _repair_prompt(self, st: dict, x: np.ndarray, regen: np.ndarray, prot: np.ndarray) -> dict:
        """User prompt file if present, else describe the undamaged surroundings with CLIP."""
        pfile = self.paint_dir / f"{st['id']}_prompt.txt"
        if pfile.exists():
            return {"prompt": pfile.read_text(encoding="utf-8").strip(), "prompt_source": pfile.name}
        from .restore.generative import auto_prompt

        prompt, cands = auto_prompt(x, regen, prot, self.models_dir, self.device, self.cfg.orient,
                                    st["colour"]["class"] != "colour")
        log.info("%s: repair prompt: %s", st["id"], prompt)
        return {"prompt": prompt, "prompt_source": "auto", "prompt_candidates": cands}

    @staticmethod
    def _load_mask(path: Path, shape) -> np.ndarray:
        m = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if m.shape != tuple(shape[:2]):
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return m

    def stage_inpaint(self) -> None:
        from .restore.damage import inpaint_lama
        from .restore.models import load_lama

        def fn(model, st):
            mask = cv2.imread(str(self.work / st["id"] / "damage_mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            x = to_rgb(self._img(st))
            y = inpaint_lama(model, x, mask, self.device,
                             tile=int(self.cfg.tile_size))
            self._put(st, "inpaint", y)
            st.setdefault("applied", []).append("inpaint")

        self._gpu_stage("inpaint", lambda: load_lama(self.models_dir, self.device), fn)

    def stage_upscale(self) -> None:
        from .restore.models import load_spandrel
        from .restore.upscale import upscale_tiled

        def fn(desc, st):
            x = to_rgb(self._img(st))
            y = upscale_tiled(desc, x, self.device, tile=int(self.cfg.tile_size))
            self._put(st, "upscale", y)
            st["out_dpi"] = st["dpi"] * desc.scale
            st.setdefault("applied", []).append(f"upscale_x{desc.scale}")

        self._gpu_stage("upscale", lambda: load_spandrel("realesrgan_x2", self.models_dir, self.device), fn)

    def stage_faces(self) -> None:
        from .facedet import FaceDetector
        from .restore.faces import restore_faces, select_faces
        from .restore.models import load_spandrel

        fc = self.cfg.faces
        todo = [s for s in self.photos() if not self._done(s, "faces")]
        if not todo:
            return
        det = FaceDetector(self.models_dir, score=fc.min_score) if fc.enabled else None
        # choose faces first (CPU), so the GPU model is only loaded if something needs it
        for st in todo:
            if st["plan"]["faces"]["run"] and det is not None:
                faces = select_faces(to_rgb(self._img(st)), det, fc)
                st["plan"]["faces"] = {"run": bool(faces), "faces": faces}
                self.save_state(st)

        def fn(desc, st):
            x = to_rgb(self._img(st))
            y = restore_faces(desc, x, st["plan"]["faces"]["faces"], fc, self.device)
            self._put(st, "faces", y)
            st.setdefault("applied", []).append(f"faces_{fc.model}x{len(st['plan']['faces']['faces'])}")

        key = "codeformer" if fc.model == "codeformer" else "gfpgan"
        self._gpu_stage("faces", lambda: load_spandrel(key, self.models_dir, self.device, half=False), fn)

    def stage_finalize(self) -> None:
        from .report import write_reports

        rows = []
        failed = []
        for st in tqdm(self.photos(), desc="finalize", unit="photo"):
            if not self._done(st, "finalize"):
                try:
                    self._finalize_one(st)
                except PermissionError as e:
                    log.error("%s: %s - close it and rerun to finish this photo", st["id"], e)
                    failed.append(st["id"])
                    continue
            rows.append(st)
        write_reports(self.out, rows, self.work / "_pages", self.cfg)
        if failed:
            log.error("%d photo(s) not written: %s", len(failed), ", ".join(failed))
        if not self.keep_work:
            for st in rows:
                for f in (self.work / st["id"]).glob("*.npy"):
                    if f.name != st.get("current"):
                        f.unlink(missing_ok=True)

    def _finalize_one(self, st: dict) -> None:
        x = self._img(st)
        cls = st["colour"]["class"]
        tag = "colour" if cls == "colour" else "bw"
        if tag == "bw":
            x = to_neutral_gray(to_rgb(x))
        dpi = st.get("out_dpi", st["dpi"])
        icc = bytes.fromhex(st["icc"]) if st.get("icc") else None
        base = self.out / "restored" / f"{st['id']}_{tag}"
        write_tiff(base.with_suffix(".tif"), to_u16(x), dpi, icc)
        write_jpeg(base.with_suffix(".jpg"), x, dpi, int(self.cfg.jpeg_quality), icc)
        st["outputs"] = {
            "master": f"masters/{st['id']}_master.tif",
            "restored_tif": str(base.with_suffix(".tif").relative_to(self.out)).replace("\\", "/"),
            "restored_jpg": str(base.with_suffix(".jpg").relative_to(self.out)).replace("\\", "/"),
        }
        st["mattress_version"] = __version__
        side = {k: v for k, v in st.items() if k not in ("current", "icc")}
        (self.out / "restored" / f"{st['id']}.json").write_text(json.dumps(side, indent=2))
        self._mark(st, "finalize")

    # ------------------------------------------------------------------ driver
    def redo_from(self, stage: str) -> None:
        """Forget `stage` and every later stage for all photos (e.g. after changing settings)."""
        cut = STAGES.index(stage)
        if cut <= STAGES.index("orient"):
            raise ValueError("use --force to redo cropping/orientation")
        for st in self.photos():
            st["done"] = [d for d in st.get("done", []) if STAGES.index(d) < cut]
            if cut <= STAGES.index(REDO_FROM):
                old = st.get("current")
                st["current"] = "base.npy"
                if old and old not in ("base.npy", "crop.npy"):
                    (self.work / st["id"] / old).unlink(missing_ok=True)
                st["applied"] = []
                st.pop("out_dpi", None)
            self.save_state(st)

    def run(self, stages: list[str]) -> None:
        self._check_paint()
        for s in STAGES:
            if s in stages:
                log.info("== stage %s", s)
                getattr(self, f"stage_{s}")()
                if s == "orient":
                    self._check_paint()  # paint files for newly cropped photos
