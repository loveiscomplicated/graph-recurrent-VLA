"""Collect frame candidate descriptors and submit them in their merge order."""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

from . import metal_ops

if TYPE_CHECKING:
    from ..detector import PlaneData


class CandidateKind(IntEnum):
    """Candidate tags shared with the Metal frame validator."""

    BOUNDARY = 0
    JUNCTION = 1
    ENVELOPE = 2
    BOX_RGB = 3
    BOX_CUBE = 4


BOUNDARY_METHOD_CODES = {"boundary": 0, "local_boundary": 1, "short_boundary": 2}
MAX_PLANE_DATA = 3
_RECORD_WIDTH = 16


@dataclass
class CandidateGroup:
    """Keep one group's CPU metadata alongside its Metal descriptor rows."""

    kind: CandidateKind
    data: "PlaneData"
    metadata: list[dict]
    rows: np.ndarray


class FrameCandidates:
    """Submit the independent geometry checks together on the MPS device."""

    def __init__(self, data_list, core_masks, recovery_masks):
        self.slots = {id(data): index for index, data in enumerate(data_list)}
        self.plane_data = [
            (data, (core_masks if data.core else recovery_masks)[data.oid])
            for data in data_list
        ]
        self.groups = []
        self.cube_geometry = None

    def _rows(self, kind, data, metadata):
        rows = np.zeros((len(metadata), _RECORD_WIDTH), dtype=np.float32)
        rows[:, :4] = kind, self.slots[id(data)], len(data.planes), data.recovery
        self.groups.append(CandidateGroup(kind, data, metadata, rows))
        return rows

    def boundary(self, data, proposals):
        rows = self._rows(CandidateKind.BOUNDARY, data, proposals)
        rows[:, 4] = [proposal["plane"] for proposal in proposals]
        rows[:, 5] = [BOUNDARY_METHOD_CODES[proposal["method"]] for proposal in proposals]
        rows[:, 6:8] = [proposal["uv"] for proposal in proposals]
        rows[:, 8:12] = np.asarray([proposal["adjacent"] for proposal in proposals]).reshape(-1, 4)

    def junctions(self, data, proposals):
        rows = self._rows(CandidateKind.JUNCTION, data, proposals)
        rows[:, 6:9] = [proposal["point"] for proposal in proposals]
        rows[:, 9:12] = [proposal["indices"] for proposal in proposals]

    def envelopes(self, data, proposals):
        rows = self._rows(CandidateKind.ENVELOPE, data, proposals)
        rows[:, 4] = [proposal["plane"] for proposal in proposals]
        rows[:, 6:9] = [proposal["point"] for proposal in proposals]
        rows[:, 9:11] = [proposal["uv"] for proposal in proposals]
        rows[:, 11:15] = np.asarray([proposal["adjacent"] for proposal in proposals]).reshape(-1, 4)

    def box_rgb(self, data, proposals):
        rows = self._rows(CandidateKind.BOX_RGB, data, proposals)
        rows[:, 4] = [proposal["plane"] for proposal in proposals]
        rows[:, 6:8] = [proposal["uv"] for proposal in proposals]
        rows[:, 8:14] = np.asarray([proposal["axes"] for proposal in proposals]).reshape(-1, 6)

    def box_cube(self, data, metadata, points, rotation, signs, features):
        rows = self._rows(CandidateKind.BOX_CUBE, data, metadata)
        rows[:, 5] = len(features)
        rows[:, 6:9] = points
        rows[:, 9:12] = signs
        feature_pixels = np.asarray([item[0] for item in features], dtype=np.float32)
        geometry = np.concatenate((rotation.astype(np.float32).ravel(), feature_pixels.ravel()))
        self.cube_geometry = (rows, geometry)

    def validate(self, depth, valid, xyz, intrinsic, neutral, parameters):
        """Yield views of each candidate group's output in submission order."""
        if not self.groups:
            return
        count = sum(len(group.rows) for group in self.groups)
        chunks = [group.rows.ravel() for group in self.groups]
        if self.cube_geometry is not None:
            rows, geometry = self.cube_geometry
            rows[:, 4] = count * _RECORD_WIDTH
            chunks.append(geometry)
        records = torch.as_tensor(np.concatenate(chunks), device=xyz.device)

        # Pad the fixed Metal buffer slots with real storage. No row refers to
        # those extra slots, and each actual slot keeps its own recovery mask.
        plane_data = self.plane_data + [self.plane_data[0]] * (MAX_PLANE_DATA - len(self.plane_data))
        buffers = [
            tensor
            for data, object_mask in plane_data
            for tensor in (data.planes, data.eligible, data.masks, data.cores, data.support, object_mask)
        ]
        points = xyz.new_empty((count, 3))
        uv = xyz.new_empty((count, 2))
        supporting = torch.empty(count, dtype=torch.int64, device=xyz.device)
        primary = torch.empty(count, dtype=torch.bool, device=xyz.device)
        secondary = torch.empty_like(primary)
        metal_ops.library().validate_frame_candidates(
            records,
            *buffers,
            depth,
            valid,
            xyz,
            intrinsic,
            neutral,
            points,
            uv,
            supporting,
            primary,
            secondary,
            float(parameters.maximum_angle_cosine),
            float(parameters.maximum_support_gap_px),
            threads=count,
            group_size=128,
        )
        offset = 0
        for group in self.groups:
            end = offset + len(group.rows)
            span = slice(offset, end)
            yield group, points[span], uv[span], supporting[span], primary[span], secondary[span]
            offset = end
