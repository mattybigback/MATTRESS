"""YuNet face detector (OpenCV, MIT licence) shared by orientation and face restoration."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io import to_u8
from .restore.models import weight_path


class FaceDetector:
    def __init__(self, models_dir: Path, score: float = 0.8):
        self.path = str(weight_path("yunet", models_dir))
        self.score = score
        self.det = cv2.FaceDetectorYN.create(self.path, "", (320, 320), score, 0.3, 500)

    def detect(self, img: np.ndarray, work_long: int = 1600) -> np.ndarray:
        """Return Nx15 array in img pixel coords: x,y,w,h, 5 landmarks (x,y), score.

        Landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner.
        """
        rgb = to_u8(img)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, 2)
        h, w = rgb.shape[:2]
        s = min(1.0, work_long / max(h, w))
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        self.det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.det.detect(bgr)
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= s
        return faces


def upright_face_score(faces: np.ndarray) -> float:
    """Sum of scores of faces whose landmarks look upright (eyes above mouth)."""
    total = 0.0
    for f in faces:
        eyes_y = (f[5] + f[7]) / 2
        mouth_y = (f[11] + f[13]) / 2
        eye_dx = f[6] - f[4]  # left eye is to the right of the right eye in image coords
        if mouth_y > eyes_y and eye_dx > 0:
            total += float(f[14])
    return total
