"""Small image helpers shared by the pipeline."""
from __future__ import annotations

import cv2
import numpy as np


def downscale(img: np.ndarray, width: int) -> np.ndarray:
    """Fast, good-looking downscale to `width` pixels wide (aspect kept).

    A plain INTER_AREA resize of a 4K frame touches all 8 million pixels. Here
    the frame is first decimated by an integer step (a free strided view) to no
    less than twice the target size, and only that is area-filtered.
    """
    h, w = img.shape[:2]
    if w <= width:
        return img
    step = w // (width * 2)
    if step >= 2:
        img = np.ascontiguousarray(img[::step, ::step])
        h, w = img.shape[:2]
    height = max(1, int(round(h * width / w)))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
