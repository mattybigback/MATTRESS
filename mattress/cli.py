"""mattress run IN OUT [options]"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from .config import load_config
from .pipeline import STAGES, Run


def main(argv: list[str] | None = None) -> int:
    # fully offline: model files come from ./models (fetch_models.py); huggingface_hub reads
    # this at import time, so set it before anything imports it
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    ap = argparse.ArgumentParser(prog="mattress", description="Offline photo-scan splitter and restorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a folder of page scans")
    r.add_argument("input", type=Path, help="folder of page scans (TIFF/PNG/JPEG)")
    r.add_argument("output", type=Path)
    r.add_argument("--config", type=Path, help="YAML overriding the packaged defaults")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one config value, e.g. --set faces.enabled=false")
    r.add_argument("--blank", type=Path, help="scan of the empty scanner lid (improves detection)")
    r.add_argument("--stages", default="all",
                   help=f"comma list or 'all' ({','.join(STAGES)}); later stages need earlier ones")
    r.add_argument("--force", action="store_true", help="discard previous progress and start over")
    r.add_argument("--redo-from", metavar="STAGE",
                   help="redo this stage and all later ones for every photo (after changing settings)")
    r.add_argument("--keep-work", action="store_true", help="keep intermediate images in OUT/_work")
    r.add_argument("--no-inpaint", action="store_true", help="shortcut for --set damage.enabled=false")
    r.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    overrides = list(a.set)
    if a.no_inpaint:
        overrides.append("damage.enabled=false")
    cfg = load_config(a.config, overrides)

    stages = STAGES if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s): {bad}")
    if not a.input.is_dir():
        ap.error(f"input folder not found: {a.input}")
    if a.force and (a.output / "_work").exists():
        shutil.rmtree(a.output / "_work")

    run = Run(cfg, a.input, a.output, blank=a.blank, keep_work=a.keep_work)
    if a.redo_from:
        if a.redo_from not in STAGES:
            ap.error(f"unknown stage {a.redo_from!r}")
        run.redo_from(a.redo_from)
    run.run(stages)
    print(f"Done. Results in {a.output.resolve()}")
    review = a.output / "_review" / "index.html"
    if review.exists():
        print(f"Some photos need a look: {review.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
