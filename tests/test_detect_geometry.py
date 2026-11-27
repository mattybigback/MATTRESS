import numpy as np
import pytest

from mattress.colorclass import classify
from mattress.config import load_config
from mattress.detect import detect_prints
from mattress.geometry import crop_print, residual_tilt
from mattress.io import to_float
from mattress.restore.damage import damage_mask

from synth import add_dust, fake_photo, make_page

CFG = load_config()


def _poly_iou(a, b, shape):
    import cv2
    ma = np.zeros(shape, np.uint8)
    mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.asarray(a, np.int32)], 1)
    cv2.fillPoly(mb, [np.asarray(b, np.int32)], 1)
    return (ma & mb).sum() / max((ma | mb).sum(), 1)


def _match(dets, truth):
    pairs = []
    for t in truth:
        tc = t["quad"].mean(0)
        d = min(dets, key=lambda d, tc=tc: np.linalg.norm(np.array(d.quad).mean(0) - tc))
        pairs.append((d, t))
    return pairs


@pytest.mark.parametrize("seed,white_border", [(1, False), (2, False), (3, True), (4, True)])
def test_detect_and_deskew(seed, white_border):
    rng = np.random.default_rng(seed)
    page, truth = make_page(rng, dpi=300, n=3, white_border=white_border)
    dets, _ = detect_prints(page, 300, CFG)
    assert len(dets) == len(truth), f"found {len(dets)} expected {len(truth)}"
    for d, t in _match(dets, truth):
        iou = _poly_iou(d.quad, t["quad"], page.shape[:2])
        assert iou > 0.97, iou
        assert abs(d.angle - t["angle"]) < 0.5, (d.angle, t["angle"])
        crop, g = crop_print(page, d.quad, 300, 0.3, 2.0)
        final = d.angle + g["refine_deg"]
        assert abs(final - t["angle"]) < 0.3, (final, t["angle"])
        # the crop must not contain lid: its border rows should be photo, not ~0.965 white
        if not white_border:
            edge = to_float(crop)[2:6].mean()
            assert edge < 0.93


def test_residual_tilt_sign():
    import cv2
    rng = np.random.default_rng(7)
    img = fake_photo(900, 1300, rng)
    M = cv2.getRotationMatrix2D((650, 450), -1.2, 1.0)  # content rotated +1.2deg (y-down)
    rot = cv2.warpAffine(img, M, (1300, 900), borderMode=cv2.BORDER_REFLECT)
    est = residual_tilt(rot, 300, 2.0)
    assert est is not None and abs(est - 1.2) < 0.25, est


def test_colour_classification():
    rng = np.random.default_rng(5)
    assert classify(fake_photo(600, 800, rng, colour=True), CFG)["class"] == "colour"
    assert classify(fake_photo(600, 800, rng, colour=False), CFG)["class"] in ("toned-mono", "mono")
    g = fake_photo(600, 800, rng, colour=True).mean(2).astype(np.uint16)
    assert classify(np.repeat(g[..., None], 3, 2), CFG)["class"] == "mono"


def test_damage_mask_finds_specks():
    import cv2
    rng = np.random.default_rng(9)
    img = to_float(fake_photo(1800, 2400, rng))
    dirty, truth = add_dust(img, rng)
    mask, _ = damage_mask(dirty, 600, CFG.damage)
    hit = (mask[truth > 0] > 0).mean()
    assert hit > 0.8, hit
    near = cv2.dilate(truth, np.ones((15, 15), np.uint8))
    stray = ((mask > 0) & (near == 0)).mean()
    assert stray < 0.0005, stray  # mask must not fire away from real damage
    clean_mask, _ = damage_mask(img, 600, CFG.damage)
    assert clean_mask.mean() < CFG.damage.min_area * 5
