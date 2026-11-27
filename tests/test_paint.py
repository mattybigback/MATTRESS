import cv2
import numpy as np
import pytest

from mattress.config import load_config
from mattress.io import to_float
from mattress.paint import PaintFileError, make_paint_file, read_paint_mask, strip_height
from mattress.restore.repair import recolour, refine_damage

from synth import fake_photo

CFG = load_config()


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


@pytest.mark.parametrize("ext,quality", [(".png", None), (".jpg", 75)])
def test_paint_roundtrip(tmp_path, ext, quality):
    rng = np.random.default_rng(1)
    master = fake_photo(1200, 1800, rng)
    path = make_paint_file(master, "p", tmp_path, 0.5)
    img = cv2.imread(str(path))
    sh = strip_height(img.shape[1])
    truth = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(truth, (500, sh + 300), (220, 120), 20, 0, 360, 1, -1)
    img[truth > 0] = (255, 0, 255)  # BGR magenta
    out = tmp_path / f"p_paint_user{ext}"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
    cv2.imwrite(str(out), img, params)
    m = read_paint_mask(out, master.shape, 0.5)
    t_full = cv2.resize(truth[sh:], (master.shape[1], master.shape[0]), interpolation=cv2.INTER_NEAREST)
    assert _iou(m, t_full) > 0.95


def test_unpainted_is_empty_and_resized_rejected(tmp_path):
    rng = np.random.default_rng(2)
    master = fake_photo(900, 1300, rng)
    path = make_paint_file(master, "q", tmp_path, 0.5)
    assert not read_paint_mask(path, master.shape, 0.5).any()
    img = cv2.imread(str(path))
    cv2.imwrite(str(tmp_path / "q_paint.png"), cv2.resize(img, None, fx=0.9, fy=0.9))
    with pytest.raises(PaintFileError):
        read_paint_mask(tmp_path / "q_paint.png", master.shape, 0.5)


def test_refine_finds_stain_inside_rough_paint():
    rng = np.random.default_rng(3)
    x = to_float(fake_photo(1000, 1400, rng))
    # synthetic water stain: irregular blob, drained of colour, cooled and blurred
    stain = np.zeros(x.shape[:2], np.uint8)
    pts = np.array([[300, 200], [700, 150], [900, 300], [800, 500], [450, 520], [280, 380]], np.int32)
    cv2.fillPoly(stain, [pts], 1)
    damaged = x.copy()
    g = cv2.GaussianBlur(x.mean(2, keepdims=True).repeat(3, 2), (0, 0), 3)
    stained = np.clip(g * 0.6 + np.array([0.05, 0.08, 0.14], np.float32), 0, 1)
    damaged[stain > 0] = stained[stain > 0]
    # rough paint: the stain grown by ~40 px
    paint = cv2.dilate(stain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
    rec, fill, stats = refine_damage(damaged, paint, CFG.repair)
    found = rec | fill
    assert _iou(found, stain) > 0.8, stats
    y = recolour(damaged, rec, found, CFG.repair)
    far = cv2.dilate(found, np.ones((25, 25), np.uint8)) == 0
    assert np.abs(y[far] - damaged[far]).max() < 1e-4  # untouched outside the damage
