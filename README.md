# MATTRESS

**M**ulti-photo **A**utomatic **T**rim, **T**idy, **R**estore & **E**nhance for **S**canned **S**naps.

Lay your prints on the scanner, scan the pages, and lie back while it does the rest.

MATTRESS is an offline, batch splitter and light restorer for flatbed photo scans. Each page can hold several loose prints. For every print, the pipeline:

1. finds it on the page;
2. crops and straightens it;
3. turns it upright;
4. decides whether it's colour or B&W;
5. restores it lightly, only where needed.

It runs locally on an NVIDIA GPU.

## Setup (once, needs internet)

```powershell
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python scripts\fetch_models.py      # ~8.5 GB of weights into .\models (incl. SDXL inpainting)
```

After that, the pipeline runs fully offline.

## Run

```powershell
.\.venv\Scripts\mattress run D:\scans\batch1 D:\scans\batch1_out
.\.venv\Scripts\mattress run IN OUT --blank D:\scans\empty_lid.tif   # better detection on white lid
.\.venv\Scripts\mattress run IN OUT --set faces.enabled=false --set upscale.enabled=false
.\.venv\Scripts\mattress run IN OUT --stages split,orient            # crop/rotate only, no restoration
```

If a run is interrupted, re-run the same command and it resumes where it stopped. To start over from scratch, add `--force`.

## Output

| Path | Contents |
|---|---|
| `masters/<page>_pNN_master.tif` | Crop that has only been straightened and turned upright, with no restoration. 16-bit, with the DPI and ICC profile kept. |
| `restored/<page>_pNN_colour.tif` / `_bw.tif` | Restored image, 16-bit. B&W photos are saved as true single-channel grayscale. |
| `restored/*.jpg` | JPEG access copy of each restored image. |
| `restored/*.json` | Everything that was measured and decided for that photo. |
| `manifest.csv` | One row per photo. |
| `_review/index.html` | Photos and pages to check by eye, with before/after thumbnails and the reason each was flagged. |

## What runs when

| Step | Method | Runs when |
|---|---|---|
| Fade / cast fix | Gentle deterministic levels, partial white balance and a small saturation lift, mixed 60/40 over the original | Range, saturation or cast is past its threshold. Mono photos always get it. |
| Dust | Top-hat filter, keeping only small, crisp, colourless specks | Always checked |
| Scratches, folds, cracks | The learned detector from Microsoft's *Bringing Old Photos Back to Life* (MIT), run at two scales | Always checked; faces are excluded |
| Repair of the above | Big-LaMa generative inpainting at full resolution, applied only inside the damage mask | Damaged area is more than 0.02% of the image |
| Large damage you paint | SDXL inpainting; people are protected (see below) | You painted it |
| Upscale x2 | Real-ESRGAN x2plus | Long edge is under 2000 px |
| Faces | CodeFormer at w=0.7, blended at 60% over the original | Face is under 160 px wide or blurry |

All thresholds are in [mattress/defaults.yaml](mattress/defaults.yaml). To change them, copy that file and pass it with `--config my.yaml`.

CodeFormer is licensed for non-commercial use only. For other uses, set `faces.model=gfpgan` (Apache-2.0).

## Fixing water damage and stains

Large damage, such as water stains, tide marks and lost emulsion, can't be told apart from real picture content automatically, so you mark it yourself.

1. After a run, open `OUT/paint/<photo>_paint.jpg` in Paint or any other bitmap editor.
2. Use the colour picker on the magenta square at the top. The colour must be pure magenta, RGB 255,0,255.
3. Paint over the damage. It doesn't need to be precise, but everything you paint is redrawn, so keep the paint mostly on the damage. People are detected and never redrawn, even if you paint over them; they show as a green outline on the check image.
4. Save with the same name. PNG, JPEG or BMP are all fine.
5. Rerun the same `mattress run` command. Only photos whose paint changed are reprocessed.
6. Check `OUT/paint/<photo>_repair_check.jpg`, which is also on the review page. Orange shows what was regenerated and green outlines show people that were left alone; the result is on the right.

**How it's repaired:** SDXL inpainting (`repair.method: generative`) redraws the painted area at about 15 s per photo, using about 10 GB of GPU memory. The app describes the undamaged surroundings automatically, for example "dark varnished wooden panelled wall and speckled terrazzo floor", and the photo's log records the words it chose. If it guesses wrong, put your own description in `OUT/paint/<photo>_prompt.txt` and rerun. This invents plausible detail where the original is lost, so always look at the check image.

Two settings change this:
- `--set repair.gen_area=damage` regenerates only the pixels under the paint that look damaged, keeping more of the original.
- `--set repair.method=recolour` uses the older approach, which recolours and fills without inventing new detail but is weaker on large damage.

Delete a paint file and rerun to undo a repair. A fresh, clean paint file is written in its place.

You can also paint on a full-size copy of the restored image, as long as it isn't resized or cropped. Never put paint files in the input folder; files with "mask" or "paint" in their name are ignored there anyway.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
