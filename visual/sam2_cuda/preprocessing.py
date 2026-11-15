"""CPU JPEG preprocessing for the SAM2 CUDA stream."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def preprocess_rgb(rgb, size):
    """Encode and decode JPEG, then resize and normalize to batched CHW."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise OSError("could not encode the SAM2 JPEG")
    with Image.open(BytesIO(encoded.tobytes())) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.transpose((pixels - mean) / std, (2, 0, 1))[None, ...]
