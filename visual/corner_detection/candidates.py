"""RGB material masks, subpixel features and contour candidate descriptors."""

from dataclasses import replace
import cv2
import numpy as np
from .backends import torch_ops as K


def material_masks(hsv):
    """One CPU topology pass, preserving the two original component cutoffs."""
    h, s, v = np.moveaxis(hsv, -1, 0)
    orange = (h <= 8) & (s >= 85) & (s <= 190) & (v >= 60)
    white = (h >= 90) & (h <= 105) & (s >= 8) & (s <= 22) & (v >= 65)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (orange | white).astype(np.uint8), 8
    )
    peg = (
        np.isin(labels, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 35))
        if n > 1
        else np.zeros(h.shape, bool)
    )
    box = (h >= 14) & (h <= 30) & (s >= 65) & (s <= 195) & (v >= 50)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(box.astype(np.uint8), 8)
    boxes = [
        np.isin(labels, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= threshold))
        if n > 1
        else np.zeros(h.shape, bool)
        for threshold in (35, 8)
    ]
    return np.stack((peg, boxes[0])), np.stack((peg, boxes[1]))


def _corner_features(rgb, mask):
    region = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    images = [
        mask.astype(np.float32),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0,
    ]
    result = []
    for kind, im in enumerate(images):
        found = cv2.goodFeaturesToTrack(im, 70, 0.025, 4.0, mask=region, blockSize=3)
        if found is None:
            continue
        cv2.cornerSubPix(
            im,
            found,
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        for uv in found[:, 0]:
            if 4 <= uv[0] < 508 and 4 <= uv[1] < 508:
                result.append((uv.astype(float), "mask" if kind == 0 else "rgb"))
    return result


class ContourProposals:
    """CPU topology descriptors, one device batch for every line fit in a frame."""

    def __init__(self):
        self.parts = []
        self.specs = []
        self.local_id = 0

    def line(self, points):
        index = len(self.parts)
        self.parts.append(points)
        return index

    def polygon(self, points, data, plane, hole, params, method):
        if len(points) < 16:
            return
        approx = cv2.approxPolyDP(
            points.astype(np.float32), params.polygon_epsilon_px, True
        )[:, 0]
        if not 3 <= len(approx) <= 30:
            return
        indices = [int(((points - p) ** 2).sum(1).argmin()) for p in approx]
        lines = []
        for i, start in enumerate(indices):
            end = indices[(i + 1) % len(indices)]
            segment = (
                points[start : end + 1]
                if end >= start
                else np.concatenate((points[start:], points[: end + 1]))
            )
            if (
                len(segment) < 5
                or np.linalg.norm(approx[i] - approx[(i + 1) % len(approx)])
                < params.minimum_edge_px
            ):
                lines.append(None)
            else:
                lines.append(self.line(segment))
        center = points.mean(0)
        for i, q in enumerate(approx):
            if lines[i - 1] is not None and lines[i] is not None:
                self.specs.append(
                    dict(
                        data=data,
                        plane=plane,
                        hole=hole,
                        method=method,
                        lines=(lines[i - 1], lines[i]),
                        center=center,
                        q=q,
                        adjacent=(approx[i - 1], approx[(i + 1) % len(approx)]),
                        det=0.18,
                        residual=params.maximum_line_residual_px,
                        distance=3.0,
                        local=None,
                    )
                )

    def local(self, points, data, plane, hole):
        if len(points) < 18:
            return
        rough = cv2.approxPolyDP(points.astype(np.float32), 0.8, True)[:, 0]
        center = points.mean(0)
        for q in rough:
            index = int(((points - q) ** 2).sum(1).argmin())
            key = self.local_id
            self.local_id += 1
            for span in (7, 11):
                if len(points) < 2 * span + 2:
                    continue
                previous = points[(index - np.arange(2, span + 1)) % len(points)]
                following = points[(index + np.arange(2, span + 1)) % len(points)]
                self.specs.append(
                    dict(
                        data=data,
                        plane=plane,
                        hole=hole,
                        method="local_boundary",
                        lines=(self.line(previous), self.line(following)),
                        center=center,
                        q=q,
                        adjacent=(previous[-1], following[-1]),
                        det=0.15,
                        residual=0.4,
                        distance=2.0,
                        local=key,
                    )
                )

    def add(self, data, params):
        for pi, mask in enumerate(data.masks_cpu):
            contours, hierarchy = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            for ci, contour in enumerate(contours):
                if abs(cv2.contourArea(contour)) < 22:
                    continue
                hole = bool(hierarchy[0, ci, 3] >= 0)
                points = contour[:, 0].astype(float)
                self.polygon(points, data, pi, hole, params, "boundary")
                if data.recovery:
                    self.local(points, data, pi, hole)
                    if data.oid == 1:
                        self.polygon(
                            points,
                            data,
                            pi,
                            hole,
                            replace(
                                params, polygon_epsilon_px=1.0, minimum_edge_px=4.0
                            ),
                            "short_boundary",
                        )

    def finish(self, device):
        fitted = K.fit_lines(self.parts, device)
        locals_ = {}
        for spec in self.specs:
            rows = fitted[list(spec["lines"])].astype(float).copy()
            rows[:, 2] += (
                (-1 if spec["hole"] else 1)
                * 0.5
                * np.sign(rows[:, :2] @ spec["center"] + rows[:, 2])
            )
            a, b = rows[:, :2]
            determinant = a[0] * b[1] - a[1] * b[0]
            if abs(determinant) < spec["det"]:
                continue
            d, e = -rows[:, 2]
            uv = np.array([d * b[1] - a[1] * e, a[0] * e - d * b[0]]) / determinant
            residual = float(rows[:, 3].max())
            if (
                residual > spec["residual"]
                or np.linalg.norm(uv - spec["q"]) > spec["distance"]
            ):
                continue
            p = dict(
                uv=uv,
                plane=spec["plane"],
                adjacent=spec["adjacent"],
                contour_hole=spec["hole"],
                line_residual_px=residual,
                method=spec["method"],
            )
            if spec["local"] is None:
                spec["data"].proposals.append(p)
            else:
                locals_.setdefault(spec["local"], []).append((spec["data"], p))
        for choices in locals_.values():
            if (
                len(choices) >= 2
                and np.linalg.norm(choices[0][1]["uv"] - choices[1][1]["uv"]) <= 0.8
            ):
                data, p = min(choices, key=lambda c: c[1]["line_residual_px"])
                data.proposals.append(p)
