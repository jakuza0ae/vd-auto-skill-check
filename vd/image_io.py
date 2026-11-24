"""Load images through Python paths, including non-ASCII Windows paths."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
