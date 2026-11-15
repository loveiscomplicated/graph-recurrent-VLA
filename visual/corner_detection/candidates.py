"""RGB material masks, subpixel features and contour candidate descriptors."""

from dataclasses import replace
import cv2
import numpy as np
from .backends import torch_ops as K


def material_masks(hsv, *, optimized=False):
    """One CPU topology pass, preserving the two original component cutoffs."""
    if optimized:
        orange = cv2.inRange(hsv, (0, 85, 60), (8, 190, 255))
        white = cv2.inRange(hsv, (90, 8, 65), (105, 22, 255))
        box = cv2.inRange(hsv, (14, 65, 50), (30, 195, 255))
        return filter_material_components(cv2.bitwise_or(orange, white), box, optimized=True)

    h, s, v = np.moveaxis(hsv, -1, 0)
    orange = (h <= 8) & (s >= 85) & (s <= 190) & (v >= 60)
    white = (h >= 90) & (h <= 105) & (s >= 8) & (s <= 22) & (v >= 65)
    box = (h >= 14) & (h <= 30) & (s >= 65) & (s <= 195) & (v >= 50)
    return filter_material_components(orange | white, box, optimized=optimized)


def filter_material_components(peg_colors, box_colors, *, optimized=False):
    """Apply the original contour component area thresholds to color masks."""
    if optimized:
        # Component labels are dense indices into their statistics array.
        # Index a boolean table directly instead of searching the label set.
        def components(colors, thresholds):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                colors.astype(np.uint8, copy=False), 8
            )
            masks = []
            for threshold in thresholds:
                keep = stats[:, cv2.CC_STAT_AREA] >= threshold
                keep[0] = False
                masks.append(keep[labels])
            return masks

        peg, = components(peg_colors, (35,))
        core, recovery = components(box_colors, (35, 8))
        return np.stack((peg, core)), np.stack((peg, recovery))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        peg_colors.astype(np.uint8), 8
    )
    peg = (
        np.isin(labels, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 35))
        if n > 1
        else np.zeros(peg_colors.shape, bool)
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(box_colors.astype(np.uint8), 8)
    boxes = [
        np.isin(labels, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= threshold))
        if n > 1
        else np.zeros(box_colors.shape, bool)
        for threshold in (35, 8)
    ]
    return np.stack((peg, boxes[0])), np.stack((peg, boxes[1]))


def _corner_features(rgb, mask, *, optimized=False):
    region = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    images = [
        mask.astype(np.float32),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0,
    ]
    # The response and nonmaximum suppression only need pixels near the
    # search mask. Keep a halo and align to the min-distance grid (4 px).
    x0 = y0 = 0
    x1, y1 = mask.shape[1], mask.shape[0]
    if optimized:
        x, y, width, height = cv2.boundingRect(region)
        x0, y0 = max(0, x // 4 * 4 - 8), max(0, y // 4 * 4 - 8)
        x1, y1 = min(x1, x + width + 8), min(y1, y + height + 8)
    crop = np.s_[y0:y1, x0:x1]
    result = []
    for kind, im in enumerate(images):
        found = cv2.goodFeaturesToTrack(
            im[crop], 70, 0.025, 4.0, mask=region[crop], blockSize=3
        )
        if found is None:
            continue
        if x0 or y0:
            found += [x0, y0]
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

    @staticmethod
    def _vertex_indices(points, vertices, lookup):
        if lookup is None:
            return [int(((points - p) ** 2).sum(1).argmin()) for p in vertices]
        # approxPolyDP returns vertices from the input contour. Preserve the
        # first index when a contour visits the same pixel more than once.
        keys, first = lookup
        vertex_keys = vertices[:, 0] + 512 * vertices[:, 1]
        return first[np.searchsorted(keys, vertex_keys)]

    def polygon(self, points, data, plane, hole, params, method, *, lookup=None):
        if len(points) < 16:
            return
        approx = cv2.approxPolyDP(
            points.astype(np.float32), params.polygon_epsilon_px, True
        )[:, 0]
        if not 3 <= len(approx) <= 30:
            return
        indices = self._vertex_indices(points, approx, lookup)
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

    def local(self, points, data, plane, hole, *, lookup=None):
        if len(points) < 18:
            return
        rough = cv2.approxPolyDP(points.astype(np.float32), 0.8, True)[:, 0]
        center = points.mean(0)
        indices = self._vertex_indices(points, rough, lookup)
        for q, index in zip(rough, indices, strict=True):
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

    def add(self, data, params, *, reuse_contours=False):
        for pi, mask in enumerate(data.masks_cpu):
            contours, hierarchy = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            if reuse_contours:
                data.contours[pi] = (contours, hierarchy)
            for ci, contour in enumerate(contours):
                if abs(cv2.contourArea(contour)) < 22:
                    continue
                hole = bool(hierarchy[0, ci, 3] >= 0)
                points = contour[:, 0].astype(float)
                lookup = None
                if reuse_contours:
                    pixels = contour[:, 0]
                    lookup = np.unique(pixels[:, 0] + 512 * pixels[:, 1], return_index=True)
                self.polygon(points, data, pi, hole, params, "boundary", lookup=lookup)
                if data.recovery:
                    self.local(points, data, pi, hole, lookup=lookup)
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
                            lookup=lookup,
                        )

    def _intersections(self, fitted):
        """Evaluate the existing line intersection gates in one host batch."""
        line_ids = np.array([spec["lines"] for spec in self.specs])
        rows = fitted[line_ids].astype(float)
        centers = np.array([spec["center"] for spec in self.specs])
        targets = np.array([spec["q"] for spec in self.specs])
        signs = np.array([-0.5 if spec["hole"] else 0.5 for spec in self.specs])
        sidedness = (rows[:, :, :2] @ centers[:, :, None]).squeeze(-1)
        rows[:, :, 2] += signs[:, None] * np.sign(sidedness + rows[:, :, 2])

        a, b = rows[:, 0, :2], rows[:, 1, :2]
        determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        eligible = ~(np.abs(determinant) < [spec["det"] for spec in self.specs])
        indices = np.flatnonzero(eligible)
        a, b = a[indices], b[indices]
        d, e = -rows[indices, :, 2].T
        uv = np.column_stack((d * b[:, 1] - a[:, 1] * e, a[:, 0] * e - d * b[:, 0]))
        uv /= determinant[indices, None]
        residual = rows[indices, :, 3].max(axis=1)
        residual_limit = np.array([spec["residual"] for spec in self.specs])[indices]
        distance_limit = np.array([spec["distance"] for spec in self.specs])[indices]
        distance = np.linalg.norm(uv - targets[indices], axis=1)
        keep = ~((residual > residual_limit) | (distance > distance_limit))
        for index, point, error in zip(indices[keep], uv[keep], residual[keep], strict=True):
            yield self.specs[index], point, float(error)

    def _scalar_intersections(self, fitted):
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
            yield spec, uv, residual

    def finish(self, device, *, batched=False):
        if not self.specs:
            return
        fitted = K.fit_lines(self.parts, device)
        locals_ = {}
        intersections = self._intersections if batched else self._scalar_intersections
        for spec, uv, residual in intersections(fitted):
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
