import enum

import numpy as np


class RegionType(enum.Enum):
    SPECIAL = 0
    RECTANGLE = 1
    MASK = 3


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class Mask:
    def __init__(self, mask, offset=(0, 0)):
        self.mask = np.asarray(mask)

    def rasterize(self, bounds):
        x0, y0, x1, y1 = bounds
        return self.mask[y0:y1 + 1, x0:x1 + 1]

    def convert(self, rtype):
        assert rtype == RegionType.RECTANGLE
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return Rectangle(0, 0, 0, 0)
        return Rectangle(xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
