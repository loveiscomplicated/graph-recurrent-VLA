"""Frame-level CUDA validation for the existing contour proposals."""

from dataclasses import dataclass

import numpy as np
import torch

from . import validation_torch


@dataclass
class ContourGroup:
    kind: str
    data: object
    proposals: list
    object_mask: torch.Tensor | None
    arrays: tuple[np.ndarray, ...]


class CudaContourBatch:
    """Upload all contour inputs together and register results in proposal order."""

    def __init__(self, detector):
        self.detector = detector
        self.groups: list[ContourGroup] = []

    def boundary(self, data, proposals):
        arrays = (
            np.array([p["uv"] for p in proposals], dtype=np.float32),
            np.array([p["adjacent"] for p in proposals], dtype=np.float32),
            np.array([p["plane"] for p in proposals], dtype=np.int64),
            np.array([p["method"] == "boundary" for p in proposals], dtype=np.bool_),
            np.array([p["method"] == "short_boundary" for p in proposals], dtype=np.bool_),
        )
        mask = self.detector.current_masks[data.oid]
        self.groups.append(ContourGroup("boundary", data, proposals, mask, arrays))

    def junctions(self, data, proposals):
        arrays = (
            np.array([p["point"] for p in proposals], dtype=np.float32),
            np.array([p["indices"] for p in proposals], dtype=np.int64),
        )
        self.groups.append(ContourGroup("junctions", data, proposals, None, arrays))

    def envelopes(self, data, proposals):
        arrays = (
            np.array([p["point"] for p in proposals], dtype=np.float32),
            np.array([p["uv"] for p in proposals], dtype=np.float32),
            np.array([p["adjacent"] for p in proposals], dtype=np.float32),
            np.array([p["plane"] for p in proposals], dtype=np.int64),
        )
        mask = self.detector.current_masks[0]
        self.groups.append(ContourGroup("envelopes", data, proposals, mask, arrays))

    def _upload_inputs(self):
        # Wider elements go first so the typed views start on aligned offsets.
        ordered = sorted(
            (
                (index, slot, array)
                for index, group in enumerate(self.groups)
                for slot, array in enumerate(group.arrays)
            ),
            key=lambda entry: -entry[2].itemsize,
        )
        types = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.bool_): torch.bool,
        }
        packed = np.concatenate([array.ravel().view(np.uint8) for _, _, array in ordered])
        uploaded = torch.as_tensor(packed, device=self.detector.device)
        inputs = [[None] * len(group.arrays) for group in self.groups]
        offset = 0
        for index, slot, array in ordered:
            view = uploaded.narrow(0, offset, array.nbytes).view(types[array.dtype])
            inputs[index][slot] = view.reshape(array.shape)
            offset += array.nbytes
        return inputs

    def validate(self):
        if not self.groups:
            return
        det = self.detector
        inputs = self._upload_inputs()
        geometry = (det.depth, det.valid, det.xyz, det.k, det.neutral)
        boundary_options = (
            det.parameters.maximum_support_gap_px,
            det.parameters.maximum_angle_cosine,
            det._validation_edges,
        )
        outputs = {}
        for kind in ("boundary", "junctions", "envelopes"):
            matching = [
                (index, (tuple(inputs[index]), group.data, group.object_mask))
                for index, group in enumerate(self.groups) if group.kind == kind
            ]
            if not matching:
                continue
            results = self._validate_kind(kind, matching, geometry, boundary_options)
            outputs.update(zip((index for index, _ in matching), results, strict=True))

        for index, group in enumerate(self.groups):
            data, proposals, kind = group.data, group.proposals, group.kind
            if kind == "boundary":
                points, uv, counts, primary, recovered = outputs[index]
            elif kind == "junctions":
                uv, primary, recovered = outputs[index]
                points, counts = inputs[index][0], None
            else:
                points, uv, counts, selected = outputs[index]
                det.add(data, points, uv, selected, proposals, counts)
                continue
            if data.core:
                det.add(data, points, uv, primary, proposals, counts, stage=0)
            if data.recovery:
                det.add(data, points, uv, recovered, proposals, counts, stage=1)

    def _validate_kind(self, kind, matching, geometry, boundary_options):
        from torch._dynamo.exc import FailOnRecompileLimitHit

        groups = tuple(item for _, item in matching)
        if len(groups) > 1:
            try:
                validate_frame = validation_torch.contour_frame_function(kind)
                return validate_frame(groups, *geometry, *boundary_options)
            except FailOnRecompileLimitHit:
                # Unseen combinations can still use the existing per-group graphs.
                pass
        det = self.detector
        results = []
        for tensors, data, mask in groups:
            if kind == "boundary":
                result = det._validate_boundary(
                    *tensors, data, mask, *geometry, *boundary_options
                )
            elif kind == "junctions":
                result = det._validate_junctions(*tensors, data, *geometry[:4])
            else:
                result = det._validate_envelopes(
                    *tensors, data, mask, *geometry, boundary_options[-1]
                )
            results.append(result)
        return results
