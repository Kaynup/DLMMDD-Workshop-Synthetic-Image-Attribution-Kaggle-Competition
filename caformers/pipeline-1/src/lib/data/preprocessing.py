from __future__ import annotations

import numpy as np


def transpose_hwc_to_chw(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.transpose(2, 0, 1))
