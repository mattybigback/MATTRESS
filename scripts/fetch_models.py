"""One-time download of all model weights into ./models (then the pipeline runs offline).

    python scripts/fetch_models.py [--models-dir models] [--skip-clip]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from mattress.config import load_config
from mattress.restore.models import REGISTRY, download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)

    for key, w in REGISTRY.items():
        p = download(key, models_dir)
        print(f"  ok  {w.filename:40s} {p.stat().st_size / 1e6:8.1f} MB  [{w.licence}]")

    if not args.skip_clip:
        cfg = load_config()
        os.environ.setdefault("HF_HOME", str((models_dir / "hf").resolve()))
        import open_clip

        open_clip.create_model_and_transforms(
            cfg.orient.clip_model, pretrained=cfg.orient.clip_pretrained,
            cache_dir=str(models_dir / "clip"),
        )
        print(f"  ok  CLIP {cfg.orient.clip_model}/{cfg.orient.clip_pretrained}")
    from mattress.restore import scratch_detect

    scratch_detect.download(models_dir)
    print("  ok  scratch/fold detector (Bringing Old Photos Back to Life, MIT)")
    # generative repair: SDXL inpainting (~7 GB) + person segmentation for protecting people
    from mattress.restore.generative import download_sdxl, person_mask
    import numpy as np

    download_sdxl(models_dir)
    print("  ok  SDXL inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)")
    person_mask(np.zeros((64, 64, 3), np.float32), models_dir)
    print("  ok  DeepLabV3 person segmentation")
    print("All models present - pipeline can now run offline.")


if __name__ == "__main__":
    main()
