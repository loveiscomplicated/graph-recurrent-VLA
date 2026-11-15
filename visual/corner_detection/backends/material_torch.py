"""Device-side RGB classification with OpenCV's uint8 HSV quantization."""

from functools import lru_cache
from importlib.util import find_spec

import numpy as np
import torch


@lru_cache(maxsize=4)
def _hsv_divisors(device):
    axis = np.arange(1, 256, dtype=np.float64)
    saturation = np.r_[0, np.rint((255 << 12) / axis)].astype(np.int32)
    hue = np.r_[0, np.rint((180 << 12) / (6 * axis))].astype(np.int32)
    return tuple(torch.as_tensor(table, device=device) for table in (saturation, hue))


def material_colors(rgb):
    """Return raw Peg and Box masks and the neutral-color mask."""
    saturation_table, hue_table = _hsv_divisors(rgb.device)
    red, green, blue = rgb.to(torch.int32).unbind(dim=-1)
    value = torch.maximum(torch.maximum(red, green), blue)
    delta = value - torch.minimum(torch.minimum(red, green), blue)
    saturation = (delta * saturation_table[value.long()] + 2048) >> 12
    hue_numerator = torch.where(
        value == red, green - blue,
        torch.where(value == green, blue - red + 2 * delta, red - green + 4 * delta),
    )
    hue = (hue_numerator * hue_table[delta.long()] + 2048) >> 12
    hue = torch.where(hue < 0, hue + 180, hue)
    orange = (hue <= 8) & (saturation >= 85) & (saturation <= 190) & (value >= 60)
    white = (hue >= 90) & (hue <= 105) & (saturation >= 8) & (saturation <= 22) & (value >= 65)
    box = (hue >= 14) & (hue <= 30) & (saturation >= 65) & (saturation <= 195) & (value >= 50)
    return torch.stack((orange | white, box)), saturation < 8


def rgb_to_gray(rgb):
    """Convert uint8 RGB with the integer rounding used by OpenCV."""
    red, green, blue = rgb.to(torch.int32).unbind(dim=-1)
    luminance = 9798 * red + 19235 * green + 3735 * blue
    return ((luminance + 16384) >> 15).to(torch.uint8)


@lru_cache(maxsize=2)
def component_filter(device_type):
    """Resolve an available GPU connected-component implementation."""
    if device_type == "mps":
        from .material_components_mps import filter_components

        return filter_components
    if device_type == "cuda" and find_spec("triton") is not None:
        from .material_components_cuda import filter_components

        return filter_components
    return None
