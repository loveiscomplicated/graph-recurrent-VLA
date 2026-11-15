"""Torch/CUDA/MPS geometry backend for the independent visible-corner probe.

OpenCV remains responsible for connected components, contour topology and
subpixel RGB features. Dense XYZ never returns to NumPy. Small contour/plane
descriptors and final candidates cross the CPU boundary explicitly. This is a
hybrid detector, not an assertion of an entirely GPU-resident pipeline.
"""

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import combinations, product

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from .backends import material_torch, rear_end_torch
from .backends import torch_ops as K
from .candidates import (
    ContourProposals,
    _corner_features,
    filter_material_components,
    material_masks,
)
from .geometry import Parameters, _face_envelopes
from .hole_rim import opening_corners
from .peg_prism import prism_corners, rear_end_corners


@dataclass
class PlaneData:
    oid: int
    support: torch.Tensor
    planes: torch.Tensor
    masks: torch.Tensor
    cores: torch.Tensor
    eligible: torch.Tensor
    planes_cpu: list
    masks_cpu: np.ndarray
    cores_cpu: np.ndarray | None
    support_cpu: np.ndarray
    eligible_cpu: set
    proposals: list
    core: bool = False
    recovery: bool = False
    points: torch.Tensor | None = None
    contours: dict = field(default_factory=dict)


@dataclass
class TensorResult:
    object_id: torch.Tensor
    uv: torch.Tensor
    xyz: torch.Tensor
    camera_xyz: torch.Tensor
    masks: torch.Tensor
    metadata: list
    diagnostics: list
    elapsed_ms: float
    parameters: dict

    def as_dict(self):
        """Explicit small-output export; native tensor results stay on device."""
        rows = (
            torch.cat(
                (
                    self.object_id[:, None].to(self.xyz.dtype),
                    self.uv,
                    self.xyz,
                    self.camera_xyz,
                ),
                1,
            )
            .cpu()
            .tolist()
        )
        corners = [
            dict(
                meta,
                object_id=int(row[0]),
                uv=row[1:3],
                xyz=row[3:6],
                camera_xyz=row[6:9],
            )
            for row, meta in zip(rows, self.metadata)
        ]
        return dict(
            corners=corners,
            masks=self.masks.cpu().numpy(),
            diagnostics=self.diagnostics,
            elapsed_ms=self.elapsed_ms,
            parameters=self.parameters,
            backend="torch",
            device=str(self.xyz.device),
            cpu_stages=[
                "OpenCV components/contours/RGB features",
                "small descriptor assembly/merge",
            ],
        )


class TorchCornerDetector:
    # The paired evaluator overrides this to replay the first opening implementation.
    _refine_opening = True
    _opening_square_span = "mean"

    def __init__(
        self,
        device="auto",
        parameters=None,
        *,
        hole_rgb_merge_radius_m=None,
        recover_hole_rim=False,
        recover_small_geometry=False,
        recover_fragmented_rim=False,
        verify_rejected_rim_squares=False,
        correct_box_exterior_pixel_center=False,
        correct_peg_position=False,
        center_peg_plane_bounds=False,
        recover_peg_rear_end=False,
        peg_rear_end_device="cpu",
        reject_peg_box_cut=False,
        optimized=False,
        compile_cuda_edges=False,
        timer=None,
    ):
        self.device = K.resolve_device(device)
        if compile_cuda_edges and self.device.type != "cuda":
            raise ValueError("compile_cuda_edges requires a CUDA detector")
        self._physical_edges = (
            K.compiled_physical_edges() if compile_cuda_edges else K.physical_edges
        )
        self.parameters = parameters or Parameters()
        if hole_rgb_merge_radius_m is not None and (
            self.device.type != "cpu"
            or not np.isfinite(hole_rgb_merge_radius_m)
            or hole_rgb_merge_radius_m <= 0
        ):
            raise ValueError(
                "hole_rgb_merge_radius_m must be positive and finite on CPU"
            )
        self.hole_rgb_merge_radius_m = hole_rgb_merge_radius_m
        self.correct_box_exterior_pixel_center = correct_box_exterior_pixel_center
        if correct_peg_position and not recover_small_geometry:
            raise ValueError("correct_peg_position requires recover_small_geometry")
        if center_peg_plane_bounds and not recover_small_geometry:
            raise ValueError("center_peg_plane_bounds requires recover_small_geometry")
        if recover_peg_rear_end and not recover_small_geometry:
            raise ValueError("recover_peg_rear_end requires recover_small_geometry")
        requested_rear_device = torch.device(peg_rear_end_device)
        if requested_rear_device.type not in ("cpu", "mps", "cuda"):
            raise ValueError("peg_rear_end_device must be cpu, mps or cuda[:index]")
        if requested_rear_device.type != "cpu" and not recover_peg_rear_end:
            raise ValueError("Accelerated rear-end recovery requires recover_peg_rear_end")
        self.peg_rear_end_device = K.resolve_device(requested_rear_device)
        if (
            self.device.type != "cpu"
            and recover_peg_rear_end
            and self.peg_rear_end_device != self.device
        ):
            raise ValueError("With GPU geometry, rear-end recovery must use the same device")
        # Historical device-resident replays use the original OpenCV color path.
        self._accelerate_rgb = self.device.type != "cpu" and recover_peg_rear_end
        self.reject_peg_box_cut = reject_peg_box_cut
        self.optimized = optimized
        self.timer = timer
        self.recover_peg_rear_end = recover_peg_rear_end
        self.center_peg_plane_bounds = center_peg_plane_bounds
        self.correct_peg_position = correct_peg_position
        self.recover_small_geometry = recover_small_geometry
        if recover_fragmented_rim and not (recover_hole_rim and recover_small_geometry):
            raise ValueError(
                "recover_fragmented_rim requires both geometry recovery options"
            )
        self.recover_hole_rim = recover_hole_rim
        self.recover_fragmented_rim = recover_fragmented_rim
        if verify_rejected_rim_squares and not recover_fragmented_rim:
            raise ValueError("verify_rejected_rim_squares requires recover_fragmented_rim")
        self.verify_rejected_rim_squares = verify_rejected_rim_squares

    @classmethod
    def main(
        cls,
        parameters=None,
        *,
        device="auto",
        peg_rear_end_device=None,
        reject_peg_box_cut=True,
        optimized=True,
        compile_cuda_edges=False,
        timer=None,
    ):
        """Build the main frame detector, optionally retaining geometry on GPU."""
        if str(device) == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and callable(
                getattr(torch.mps, "compile_shader", None)
            ):
                device = "mps"
            else:
                device = "cpu"
        device = K.resolve_device(device)
        return cls(
            device,
            parameters,
            recover_hole_rim=True,
            recover_small_geometry=True,
            recover_fragmented_rim=True,
            verify_rejected_rim_squares=True,
            correct_box_exterior_pixel_center=True,
            correct_peg_position=True,
            center_peg_plane_bounds=True,
            recover_peg_rear_end=True,
            peg_rear_end_device=peg_rear_end_device or device,
            reject_peg_box_cut=reject_peg_box_cut,
            optimized=optimized and device.type != "cpu",
            compile_cuda_edges=compile_cuda_edges,
            timer=timer,
        )

    def _measure(self, name):
        """Enable named timing without synchronizing ordinary inference calls."""
        return self.timer.measure(name) if self.timer is not None else nullcontext()

    def tensor(self, value, dtype=torch.float32):
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def prepare(self, oid, mask, support_cpu=None):
        p = self.parameters
        with self._measure("point_cloud_and_plane_fit"):
            support = mask & self.valid
            if support_cpu is None:
                points = self.xyz[support]
            else:
                pixels = self.tensor(np.flatnonzero(support_cpu), torch.int64)
                points = self.xyz.reshape(-1, 3).index_select(0, pixels)
            planes = K.fit_planes(
                points, p, 42 + oid, 4 if oid == 0 else 8, optimized=self.optimized
            )
        with self._measure("plane_masks"):
            masks, cores = K.plane_masks(self.xyz, support, planes, p.plane_tolerance_m)
            eligible = cores.flatten(1).sum(1) >= 20
        with self._measure("plane_descriptor_readback"):
            small = planes.cpu().numpy().astype(float)
            eligible_cpu = set(np.flatnonzero(eligible.cpu().numpy()))
            masks_cpu = masks.cpu().numpy()
            cores_cpu = None if self.optimized else cores.cpu().numpy()
            if support_cpu is None:
                support_cpu = support.cpu().numpy()
        return PlaneData(
            oid,
            support,
            planes,
            masks,
            cores,
            eligible,
            [(row[:3], float(row[3])) for row in small],
            masks_cpu,
            cores_cpu,
            support_cpu,
            eligible_cpu,
            [],
            points=points,
        )

    def add(
        self,
        data,
        points,
        uv,
        selected,
        metadata,
        counts=None,
        stage=1,
        feature_ids=None,
    ):
        if not len(points):
            return
        block = len(self.blocks)
        # Keep fixed-size GPU blocks until all validation is queued. Compacting
        # each stage separately forced nonzero/indices/counts readback barriers.
        self.blocks.append((points, uv, data.oid, selected, counts))
        for i in range(len(points)):
            meta = {
                k: v
                for k, v in metadata[i].items()
                if k not in ("uv", "adjacent", "point")
            }
            meta.setdefault("sources", 1)
            meta.setdefault("supporting_planes", 1)
            if stage == 0:
                meta["method"] = "v2_core"
            self.rows.append(
                dict(
                    block=block,
                    row=i,
                    stage=stage,
                    meta=meta,
                    feature=None if feature_ids is None else feature_ids[i],
                )
            )

    def boundary(self, data):
        ps = data.proposals
        if not ps or not len(data.planes):
            return
        uv = self.tensor(np.array([p["uv"] for p in ps]))
        pi = self.tensor([p["plane"] for p in ps], torch.int64)
        adjacent = self.tensor(np.array([p["adjacent"] for p in ps]))
        if self.device.type == "mps":
            from .backends import metal_ops

            methods = self.tensor(
                [
                    {"boundary": 0, "local_boundary": 1, "short_boundary": 2}[
                        p["method"]
                    ]
                    for p in ps
                ],
                torch.int32,
            )
            points, image, counts, core, recovered = metal_ops.boundary_batch(
                uv,
                adjacent,
                pi,
                methods,
                data,
                self.current_masks[data.oid],
                self.depth,
                self.valid,
                self.xyz,
                self.k,
                self.neutral,
                self.parameters,
            )
            if data.core:
                self.add(data, points, image, core, ps, counts, stage=0)
            if data.recovery:
                self.add(data, points, image, recovered, ps, counts, stage=1)
            return
        normal, offset = data.planes[pi, :3], data.planes[pi, 3]
        ray = K.rays(uv, self.k)
        denom = (ray * normal).sum(-1)
        points = ray * (-offset / denom)[:, None]
        aray = K.rays(adjacent, self.k)
        arms = (
            aray * (-offset[:, None] / (aray * normal[:, None]).sum(-1))[..., None]
            - points[:, None]
        )
        cosine = (arms[:, 0] * arms[:, 1]).sum(-1).abs() / (
            torch.linalg.vector_norm(arms[:, 0], dim=-1)
            * torch.linalg.vector_norm(arms[:, 1], dim=-1)
            + 1e-12
        )
        plane_patch, inside, pix = K.sample_patch(data.masks[0], uv, 3)
        plane_patch = data.masks[pi[:, None, None], pix[..., 1], pix[..., 0]] & inside
        distance = (pix.to(uv.dtype) - uv[:, None, None]).square().sum(-1)
        gap = (
            torch.where(plane_patch, distance, torch.inf)
            .flatten(1)
            .min(1)
            .values.sqrt()
        )
        patch_valid = self.valid[pix[..., 1], pix[..., 0]] & inside
        occluded = (
            patch_valid
            & (self.depth[pix[..., 1], pix[..., 0]] < points[:, 2, None, None] - 0.020)
        ).sum((1, 2)) >= 3
        cloud = self.xyz[pix[..., 1], pix[..., 0]]
        contact = (
            patch_valid
            & self.neutral[pix[..., 1], pix[..., 0]]
            & ~self.current_masks[data.oid][pix[..., 1], pix[..., 0]]
        )
        contact &= (cloud - points[:, None, None]).square().sum(-1) < 0.006**2
        pixel = torch.round(uv)
        base = (
            (denom.abs() >= 0.08)
            & (points[:, 2] > 0)
            & ((pixel >= 3) & (pixel < 510)).all(1)
        )
        base &= (
            (plane_patch.sum((1, 2)) >= 5)
            & (gap <= self.parameters.maximum_support_gap_px)
            & (contact.sum((1, 2)) < 3)
        )
        snapped, supporting = K.snap(points, pi, data.planes, data.eligible)
        image = K.project(snapped, self.k)
        boundary = self.tensor([p["method"] == "boundary" for p in ps], torch.bool)
        if data.core:
            core = base & boundary & (cosine <= 0.28) & ~occluded
            self.add(data, snapped, image, core, ps, supporting, stage=0)
        if data.recovery:
            physical, evidence = self._physical_edges(
                uv,
                adjacent,
                data.planes,
                pi,
                data.masks,
                self.current_masks[data.oid],
                self.depth,
                self.valid,
                self.xyz,
                self.k,
                self.neutral,
            )
            observed = K.visible(points, uv, data.support, self.xyz)
            selected = base & (cosine <= self.parameters.maximum_angle_cosine)
            selected &= (boundary & (cosine <= 0.28)) | physical
            selected &= ~occluded | (physical & observed)
            selected &= boundary | (
                ((image - uv).square().sum(-1) <= 4)
                & K.visible(snapped, image, data.support, self.xyz)
            )
            short = self.tensor(
                [p["method"] == "short_boundary" for p in ps], torch.bool
            )
            selected &= ~short | ((uv >= 8) & (uv < 504)).all(1)
            selected &= K.unoccluded(snapped, image, self.depth, self.valid)
            self.add(data, snapped, image, selected, ps, supporting, stage=1)

    def junctions(self, data):
        ps = []
        for indices in combinations(sorted(data.eligible_cpu), 3):
            normals = np.stack([data.planes_cpu[i][0] for i in indices])
            if (
                abs(np.linalg.det(normals)) < 0.85
                or np.max(np.abs(normals @ normals.T - np.eye(3))) > 0.25
            ):
                continue
            point = np.linalg.solve(
                normals, -np.array([data.planes_cpu[i][1] for i in indices])
            )
            ps.append(
                dict(
                    point=point,
                    indices=indices,
                    plane=indices[0],
                    method="three_visible_planes",
                    line_residual_px=0.0,
                    contour_hole=False,
                    supporting_planes=3,
                )
            )
        if not ps:
            return
        points = self.tensor(np.array([p["point"] for p in ps]))
        if self.device.type == "mps":
            from .backends import metal_ops

            indices = self.tensor([p["indices"] for p in ps], torch.int64)
            uv, selected, recovered = metal_ops.junction_batch(
                points, indices, data, self.xyz, self.depth, self.valid, self.k
            )
            if data.core:
                self.add(data, points, uv, selected, ps, stage=0)
            if data.recovery:
                self.add(data, points, uv, recovered, ps, stage=1)
            return
        uv = K.project(points, self.k)
        selected = (points[:, 2] > 0) & (
            (torch.round(uv) >= 6) & (torch.round(uv) < 506)
        ).all(1)
        support, inside, pix = K.sample_patch(data.support, uv, 4)
        nearby = (
            self.xyz[pix[..., 1], pix[..., 0]] - points[:, None, None]
        ).square().sum(-1) < 0.006**2
        for i in range(3):
            ids = self.tensor([p["indices"][i] for p in ps], torch.int64)
            selected &= (
                data.masks[ids[:, None, None], pix[..., 1], pix[..., 0]]
                & support
                & inside
                & nearby
            ).sum((1, 2)) >= 3
            _, _, corepix = K.sample_patch(data.support, uv, 6)
            selected &= (
                data.cores[ids[:, None, None], corepix[..., 1], corepix[..., 0]].sum(
                    (1, 2)
                )
                >= 2
            )
        depth, _, pix = K.sample_patch(self.depth, uv, 1)
        selected &= (
            self.valid[pix[..., 1], pix[..., 0]]
            & (depth < points[:, 2, None, None] - 0.004)
        ).sum((1, 2)) < 2
        if data.core:
            self.add(data, points, uv, selected, ps, stage=0)
        if data.recovery:
            self.add(
                data,
                points,
                uv,
                selected & K.unoccluded(points, uv, self.depth, self.valid),
                ps,
                stage=1,
            )

    def envelopes(self, data):
        if data.oid != 0 or not data.recovery:
            return
        ps = []
        cached_contours = getattr(data, "contours", {})
        for pi in sorted(data.eligible_cpu):
            for uv, point, adjacent, hole in _face_envelopes(
                data.masks_cpu[pi].astype(np.uint8) if pi not in cached_contours else None,
                pi,
                data.planes_cpu,
                data.eligible_cpu,
                self.k_cpu,
                contours=cached_contours.get(pi),
            ):
                ps.append(
                    dict(
                        uv=uv,
                        point=point,
                        adjacent=adjacent,
                        plane=pi,
                        contour_hole=hole,
                        line_residual_px=0.4,
                        method="observed_face_envelope",
                    )
                )
        if not ps:
            return
        points = self.tensor(np.array([p["point"] for p in ps]))
        uv = self.tensor(np.array([p["uv"] for p in ps]))
        pi = self.tensor([p["plane"] for p in ps], torch.int64)
        adjacent = self.tensor(np.array([p["adjacent"] for p in ps]))
        if self.device.type == "mps":
            from .backends import metal_ops

            points, image, counts, selected = metal_ops.envelope_batch(
                points,
                uv,
                adjacent,
                pi,
                data,
                self.current_masks[0],
                self.depth,
                self.valid,
                self.xyz,
                self.k,
                self.neutral,
            )
            self.add(data, points, image, selected, ps, counts)
            return
        physical, _ = self._physical_edges(
            uv,
            adjacent,
            data.planes,
            pi,
            data.masks,
            self.current_masks[0],
            self.depth,
            self.valid,
            self.xyz,
            self.k,
            self.neutral,
        )
        selected = physical & K.visible(points, uv, data.support, self.xyz)
        _, _, pix = K.sample_patch(self.depth, uv, 2)
        contact = (
            self.valid[pix[..., 1], pix[..., 0]]
            & self.neutral[pix[..., 1], pix[..., 0]]
            & ~self.current_masks[0][pix[..., 1], pix[..., 0]]
        )
        contact &= (
            self.xyz[pix[..., 1], pix[..., 0]] - points[:, None, None]
        ).square().sum(-1) < 0.006**2
        points, count = K.snap(points, pi, data.planes, data.eligible)
        image = K.project(points, self.k)
        selected &= (contact.sum((1, 2)) < 3) & ((image >= 8) & (image < 504)).all(1)
        selected &= ((image - uv).square().sum(-1) <= 4) & K.visible(
            points, image, data.support, self.xyz
        )
        selected &= K.unoccluded(points, image, self.depth, self.valid)
        self.add(data, points, image, selected, ps, count)

    def box_rgb(self, data, features):
        ps = []
        signs = np.array(list(product((-1, 1), repeat=2)))
        distances = np.array(list(product((0.002, 0.004), repeat=2)))
        bases = []
        for pi in sorted(data.eligible_cpu):
            n, d = data.planes_cpu[pi]
            other = next(
                (
                    j
                    for j in sorted(data.eligible_cpu)
                    if j != pi and abs(n @ data.planes_cpu[j][0]) < 0.2
                ),
                None,
            )
            if other is None:
                continue
            axis = data.planes_cpu[other][0] - n * (n @ data.planes_cpu[other][0])
            axis /= np.linalg.norm(axis)
            bases.append((pi, np.stack((axis, np.cross(n, axis)))))
        for fi, (uv, feature) in enumerate(features):
            for pi, basis in bases:
                ps.append(
                    dict(
                        uv=uv,
                        plane=pi,
                        axes=basis,
                        fi=fi,
                        feature=feature,
                        method="box_rgb_quadrants",
                        line_residual_px=0.3,
                    )
                )
        if not ps:
            return
        uv = self.tensor(np.array([p["uv"] for p in ps]))
        pi = self.tensor([p["plane"] for p in ps], torch.int64)
        axes = self.tensor(np.array([p["axes"] for p in ps]))
        if self.device.type == "mps":
            from .backends import metal_ops

            points, selected, holes = metal_ops.box_rgb_batch(
                uv,
                pi,
                axes,
                data,
                self.depth,
                self.valid,
                self.xyz,
                self.k,
                self.neutral,
            )
            self._add_box_rgb(data, points, uv, selected, holes, ps)
            return
        n, d = data.planes[pi, :3], data.planes[pi, 3]
        ray = K.rays(uv, self.k)
        denom = (ray * n).sum(-1)
        points = ray * (-d / denom)[:, None]
        selected = (
            (denom.abs() > 0.1)
            & (points[:, 2] > 0)
            & K.visible(points, uv, data.support, self.xyz)
        )
        selected &= K.unoccluded(points, uv, self.depth, self.valid)
        s = self.tensor(signs)
        distances = self.tensor(distances)
        delta = s[:, None] * distances[None]
        probe = points[:, None, None] + torch.einsum("sdt,ntc->nsdc", delta, axes)
        pix0 = K.project(probe, self.k).round().to(torch.int64)
        in_image = ((pix0 >= 1) & (pix0 <= 510)).all((2, 3))
        pix = pix0.clamp(0, 511)
        cloud = self.xyz[pix[..., 1], pix[..., 0]]
        evidence = data.support[pix[..., 1], pix[..., 0]] & (
            (cloud * n[:, None, None]).sum(-1) + d[:, None, None]
        ).abs().lt(0.0012)
        fractions = torch.where(in_image, evidence.float().mean(-1), 0.5)
        occupied = fractions >= 0.75
        empty = fractions <= 0.25
        exterior = (occupied.sum(1) == 1) & (empty.sum(1) == 3)
        interior = (occupied.sum(1) == 3) & (empty.sum(1) == 1)
        selected &= exterior | interior
        sign_id = torch.where(
            exterior,
            occupied.to(torch.int64).argmax(1),
            empty.to(torch.int64).argmax(1),
        )
        chosen = s[sign_id]
        adjacent = K.project(points[:, None] + chosen[..., None] * 0.015 * axes, self.k)
        physical, _ = self._physical_edges(
            uv,
            adjacent,
            data.planes,
            pi,
            data.masks,
            data.support,
            self.depth,
            self.valid,
            self.xyz,
            self.k,
            self.neutral,
        )
        selected &= physical
        _, _, patch = K.sample_patch(self.depth, uv, 2)
        contact = (
            self.valid[patch[..., 1], patch[..., 0]]
            & self.neutral[patch[..., 1], patch[..., 0]]
            & ~data.support[patch[..., 1], patch[..., 0]]
        )
        contact &= (
            self.xyz[patch[..., 1], patch[..., 0]] - points[:, None, None]
        ).square().sum(-1) < 0.006**2
        selected &= contact.sum((1, 2)) < 3
        self._add_box_rgb(data, points, uv, selected, interior, ps)

    def _add_box_rgb(self, data, points, uv, selected, holes, proposals):
        feature_ids = [p.pop("fi") for p in proposals]
        if self.optimized:
            # These flags are first consumed by merge. Reading them here would
            # wait for the GPU before the remaining candidates are submitted.
            self.box_rgb_holes = (len(self.blocks), holes)
            flags = [False] * len(proposals)
        else:
            flags = holes.cpu().tolist()
        for proposal, hole in zip(proposals, flags, strict=True):
            proposal["contour_hole"] = bool(hole)
            proposal.pop("axes")
        self.add(data, points, uv, selected, proposals, stage=2, feature_ids=feature_ids)

    def box_cube(self, data, features):
        y, x = np.nonzero(data.support_cpu)
        if not len(x) or x.min() < 2 or y.min() < 2 or x.max() > 509 or y.max() > 509:
            return
        n, d = data.planes_cpu[0]
        other = next(
            (
                i
                for i in sorted(data.eligible_cpu)
                if i != 0 and abs(n @ data.planes_cpu[i][0]) < 0.15
            ),
            None,
        )
        if other is None:
            return
        axis = data.planes_cpu[other][0] - n * (n @ data.planes_cpu[other][0])
        axis /= np.linalg.norm(axis)
        rotation = np.stack((n, axis, np.cross(n, axis)), 1)
        observed_points = getattr(data, "points", None)
        if observed_points is None:
            observed_points = self.xyz[data.support]
        coords = observed_points @ self.tensor(rotation)
        lo, hi = (
            torch.quantile(coords, self.tensor([0.001, 0.999]), dim=0)
            .cpu()
            .numpy()
            .astype(float)
        )
        widths = hi - lo
        size = float(max(widths[1:])) + 0.0015
        if not 0.16 <= size <= 0.27 or min(widths[1:]) < 0.8 * size:
            return
        center = (lo + hi) / 2
        center[0] = -d + (size / 2 if hi[0] + d > -d - lo[0] else -size / 2)
        signs = np.array(list(product((-1, 1), repeat=3)))
        points = self.tensor((center + signs * size / 2) @ rotation.T)
        if self.device.type == "mps":
            from .backends import metal_ops

            feature_tensor = self.tensor(
                np.array([f[0] for f in features], np.float32).reshape(-1, 2)
            )
            points, uv, selected = metal_ops.box_cube_batch(
                points,
                self.tensor(rotation),
                self.tensor(signs),
                feature_tensor,
                data,
                self.depth,
                self.valid,
                self.xyz,
                self.k,
                self.neutral,
            )
            metadata = [
                dict(
                    plane=0,
                    line_residual_px=0.25,
                    contour_hole=False,
                    supporting_planes=2,
                    method="box_observed_cube",
                    estimated_size_m=size,
                )
                for _ in range(8)
            ]
            self.add(data, points, uv, selected, metadata, stage=2)
            return
        eligible = sorted(data.eligible_cpu)
        distances = (
            points @ data.planes[eligible, :3].T + data.planes[eligible, 3]
        ).abs()
        nearest = self.tensor(eligible, torch.int64)[distances.argmin(1)]
        snapped, _ = K.snap(points, nearest, data.planes, data.eligible)
        points = torch.where(
            (distances.min(1).values < 0.003)[:, None], snapped, points
        )
        uv = K.project(points, self.k)
        if features:
            f = self.tensor(np.array([q[0] for q in features]))
            diff = (uv[:, None] - f[None]).square().sum(-1)
            closest = f[diff.argmin(1)]
            close = diff.min(1).values < 2.5**2
            distances = (
                points @ data.planes[eligible, :3].T + data.planes[eligible, 3]
            ).abs()
            pi = self.tensor(eligible, torch.int64)[distances.argmin(1)]
            normal = data.planes[pi, :3]
            ray = K.rays(closest, self.k)
            denom = (ray * normal).sum(-1)
            refined = ray * (-data.planes[pi, 3] / denom)[:, None]
            refine = (
                close
                & (denom.abs() > 0.1)
                & ((refined - points).square().sum(-1) < 0.005**2)
            )
            points = torch.where(refine[:, None], refined, points)
            uv = torch.where(refine[:, None], closest, uv)
            image_corner = (uv[:, None] - f[None]).square().sum(-1).min(
                1
            ).values < 2.5**2
        else:
            image_corner = torch.zeros(8, dtype=torch.bool, device=self.device)
        visible = K.visible(points, uv, data.support, self.xyz)
        observed, inside, pix = K.sample_patch(data.support, uv, 3)
        nearby = (
            self.xyz[pix[..., 1], pix[..., 0]] - points[:, None, None]
        ).square().sum(-1) < 0.006**2
        selected = (
            visible | (image_corner & (observed & inside & nearby).any((1, 2)))
        ) & K.unoccluded(points, uv, self.depth, self.valid)
        physical_any = torch.zeros(8, dtype=torch.bool, device=self.device)
        # Build all plane-distance masks once, rather than per cube vertex.
        actual_masks = (
            (self.xyz.reshape(-1, 3) @ data.planes[:, :3].T + data.planes[:, 3]).abs()
            < 0.0012
        ).T.reshape(-1, 512, 512) & data.support
        for ai in range(3):
            normals = data.planes[:, :3]
            target = self.tensor(rotation[:, ai])
            match = (
                ((normals @ target).abs() > 0.97)[None]
                & ((points @ normals.T + data.planes[:, 3]).abs() < 0.0025)
                & data.eligible[None]
            )
            pi = match.to(torch.int64).argmax(1)
            tangent = [j for j in range(3) if j != ai]
            adjacent = (
                points[:, None]
                - self.tensor(signs[:, tangent])[..., None]
                * 0.015
                * self.tensor(rotation[:, tangent].T)[None]
            )
            physical, _ = self._physical_edges(
                uv,
                K.project(adjacent, self.k),
                data.planes,
                pi,
                actual_masks,
                data.support,
                self.depth,
                self.valid,
                self.xyz,
                self.k,
                self.neutral,
            )
            physical_any |= match.any(1) & physical
        selected &= physical_any | image_corner
        metadata = [
            dict(
                plane=0,
                line_residual_px=0.25,
                contour_hole=False,
                supporting_planes=2,
                method="box_observed_cube",
                estimated_size_m=size,
            )
            for _ in range(8)
        ]
        self.add(data, points, uv, selected, metadata, stage=2)

    def correct_box_exterior(self, points, uv, object_id, metadata):
        """Apply the observed half-pixel offset only near proposed cube vertices."""
        cube_blocks = dict.fromkeys(
            row["block"]
            for row in self.rows
            if row["meta"]["method"] == "box_observed_cube"
        )
        exterior_ids = [
            i
            for i, meta in enumerate(metadata)
            if meta["method"] != "box_opening" and not meta.get("contour_hole")
        ]
        if not cube_blocks or not exterior_ids:
            return points, uv
        anchors = torch.cat([self.blocks[block][0] for block in cube_blocks])
        indices = self.tensor(exterior_ids, torch.int64)
        near_cube = torch.cdist(points[indices], anchors).amin(dim=1) < 0.012
        if self.optimized:
            correct_ids = [
                i for i, near in zip(exterior_ids, near_cube.tolist(), strict=True)
                if near and self.merged_object_ids[i] == 1
            ]
            if not correct_ids:
                return points, uv
            indices = self.tensor(correct_ids, torch.int64)
        else:
            indices = indices[near_cube & (object_id[indices] == 1)]
            correct_ids = indices.tolist()
        corrected = points.clone()
        corrected_uv = uv.clone()
        corrected[indices, :2] += 0.5 * points[indices, 2:3] / self.k.diagonal()[:2]
        corrected_uv[indices] += 0.5
        for i in correct_ids:
            metadata[i]["box_exterior_pixel_center_corrected"] = True
        return corrected, corrected_uv

    def apply_peg_position_correction(self, points, uv, object_id):
        """Apply the benchmark camera's fixed Peg ray and depth offsets."""
        if self.optimized:
            peg_ids = [i for i, oid in enumerate(self.merged_object_ids) if oid == 0]
            if not peg_ids:
                return points, uv
            is_peg = self.tensor(peg_ids, torch.int64)
        else:
            is_peg = object_id == 0
        peg_points = points[is_peg]
        # Shift the original ray before increasing depth so the two coordinates
        # in the result describe the same corrected 3-D point.
        corrected_uv = uv.clone()
        shifted_uv = K.project(peg_points, self.k) + 0.25
        corrected_uv[is_peg] = shifted_uv
        corrected = points.clone()
        corrected[is_peg] = K.rays(shifted_uv, self.k) * (
            peg_points[:, 2:3] + 0.0005
        )
        return corrected, corrected_uv

    def peg_vertices(self, data):
        """Fit supported Peg vertices from this frame."""
        return prism_corners(
            data.points.numpy(),
            data.planes_cpu,
            center_plane_bounds=self.center_peg_plane_bounds,
        )

    def peg_prism(self, data):
        """Queue locally supported Peg prism vertices for the final merge."""
        if self.device.type == "cpu":
            points = self.tensor(self.peg_vertices(data))
            selected = torch.ones(len(points), dtype=torch.bool, device=self.device)
        else:
            if len(data.points) < 40 or len(data.planes) < 2:
                return
            pairs, sides = self._peg_geometry_indices(self.device, len(data.planes), 3)
            if self.optimized:
                self.peg_prism_basis = rear_end_torch.prism_basis(data.planes, pairs)
            points, selected = rear_end_torch.prism_candidates(
                data.points, data.planes, pairs, sides,
                centered=self.center_peg_plane_bounds,
                optimized=self.optimized,
                frame_basis=getattr(self, "peg_prism_basis", None),
            )
        metadata = [{"method": "peg_prism", "contour_hole": False} for _ in points]
        self.add(data, points, K.project(points, self.k), selected, metadata, stage=4)

    def peg_box_cut_indices(self, points, metadata, peg, box):
        """Find merged prism outputs formed by the Peg disappearing behind Box."""
        if not len(box.points):
            return []
        indices = [i for i, meta in enumerate(metadata) if meta["method"] == "peg_prism"]
        if len(indices) < 2:
            return []
        pairs, _ = self._peg_geometry_indices(self.device, len(peg.planes), 3)
        frame_basis = getattr(self, "peg_prism_basis", None)
        if frame_basis is None:
            frame_basis = rear_end_torch.prism_basis(peg.planes, pairs)
        basis, _ = frame_basis
        cut = rear_end_torch.box_cut_vertices(
            points[indices],
            peg.planes,
            basis,
            box.points,
            self.depth,
            peg.support,
            box.support,
            self.k,
        )
        return [index for index, reject in zip(indices, cut.tolist(), strict=True) if reject]

    def peg_rear_end(self, peg, box):
        """Queue supported rear corners through the existing Peg replacement step."""
        if self.peg_rear_end_device.type != "cpu":
            if len(peg.points) < 40 or len(peg.planes) < 2 or not len(box.points):
                return
            points, uv, selected = self._peg_rear_end_accelerated(peg, box)
        else:
            with self._measure("rear_candidates"):
                corners = rear_end_corners(
                    peg.points.numpy(), peg.planes_cpu, box.points.numpy()
                )
                points = self.tensor(corners)
            if not len(points):
                return
            with self._measure("rear_projection"):
                uv, checked_points, checked_uv = self._project_rear_end(points, self.k)
            with self._measure("rear_depth_support"):
                distances, _ = cKDTree(peg.points.numpy()).query(checked_points.numpy())
                supported = self.tensor(distances < 0.005, torch.bool)
            with self._measure("rear_box_occlusion"):
                selected = supported & rear_end_torch.unoccluded(
                    checked_points, checked_uv, self.depth, box.support
                )

        metadata = [{"method": "peg_rear_end", "contour_hole": False} for _ in points]
        self.add(peg, points, uv, selected, metadata, stage=4)

    def _project_rear_end(self, points, intrinsic):
        """Validate the same calibrated camera position that will be emitted."""
        uv = K.project(points, intrinsic)
        if not self.correct_peg_position:
            return uv, points, uv
        shifted = uv + 0.25
        corrected = K.rays(shifted, intrinsic) * (points[:, 2:3] + 0.0005)
        return uv, corrected, shifted

    @staticmethod
    @lru_cache(maxsize=24)
    def _peg_geometry_indices(device, planes, dimensions):
        """Reuse small constant index tables on the compute device."""
        pairs = torch.tensor(list(combinations(range(planes), 2)), device=device)
        sides = torch.tensor(list(product((False, True), repeat=dimensions)), device=device)
        return pairs, sides

    def _peg_rear_end_accelerated(self, peg, box):
        """Verify rear corners on the same device as the observed geometry."""
        device = self.peg_rear_end_device
        cloud, planes, box_cloud = peg.points, peg.planes, box.points
        depth, box_support, intrinsic = self.depth, box.support, self.k
        if device != self.device:
            with self._measure("rear_upload"):
                cloud, planes, box_cloud, depth, box_support, intrinsic = (
                    value.to(device)
                    for value in (cloud, planes, box_cloud, depth, box_support, intrinsic)
                )
        pairs, sides = self._peg_geometry_indices(device, len(planes), 2)
        with self._measure("rear_candidates"):
            points, plausible = rear_end_torch.rear_candidates(
                cloud, planes, box_cloud, pairs, sides, optimized=self.optimized
            )
            supported = plausible & rear_end_torch.depth_supported(points, cloud, 0.006)
        with self._measure("rear_projection"):
            uv, checked_points, checked_uv = self._project_rear_end(points, intrinsic)
        with self._measure("rear_depth_support"):
            supported = supported & rear_end_torch.depth_supported(checked_points, cloud, 0.005)
        with self._measure("rear_box_occlusion"):
            selected = supported & rear_end_torch.unoccluded(
                checked_points, checked_uv, depth, box_support
            )
        if device != self.device:
            with self._measure("rear_download"):
                return points.to(self.device), uv.to(self.device), selected.to(self.device)
        return points, uv, selected

    def box_openings(self, data):
        """Queue verified opening corners after all existing Box candidates."""
        with self._measure("hole_depth_readback"):
            depth_cpu = getattr(self, "depth_cpu", None)
            if depth_cpu is None:
                depth_cpu = self.depth.cpu().numpy()
        for plane, corners, rectangular, verified in opening_corners(
            depth_cpu,
            data.support_cpu,
            data.planes_cpu,
            data.eligible_cpu,
            self.k_cpu,
            small_opening=self.recover_small_geometry,
            fragmented=self.recover_fragmented_rim,
            square_span=self._opening_square_span,
            verify_rejected_square=self.verify_rejected_rim_squares,
        ):
            points = self.tensor(corners)
            if self._refine_opening:
                # Counter the observed subpixel raster bias at fitted rim edges.
                # Keep the adjustment local to the opening recovery path.
                correction_px = 0.375 if self.recover_fragmented_rim else 0.25
                pixel_shift = correction_px / self.k.diag()[:2]
                points[:, :2] += points[:, 2:] * pixel_shift
            uv = K.project(points, self.k)
            metadata = [
                {
                    "method": "box_opening",
                    "plane": plane,
                    "contour_hole": True,
                    "rectangular": rectangular,
                }
                for _ in corners
            ]
            selected = torch.full((len(corners),), verified, dtype=torch.bool, device=self.device)
            self.add(data, points, uv, selected, metadata, stage=3)

    def merge(self):
        self.merged_object_ids = []
        if not self.blocks:
            return (
                self.xyz.new_empty((0, 3)),
                self.xyz.new_empty((0, 2)),
                torch.empty(0, dtype=torch.int64, device=self.device),
                [],
            )
        points = torch.cat([b[0] for b in self.blocks])
        image = torch.cat([b[1] for b in self.blocks])
        offsets = np.cumsum([0] + [len(b[0]) for b in self.blocks])
        # Three aggregate readbacks replace up to three barriers per block.
        with self._measure("merge_readback"):
            flag_blocks = [b[3] for b in self.blocks]
            rgb_holes = getattr(self, "box_rgb_holes", None)
            if rgb_holes is not None:
                rgb_block, holes = rgb_holes
                flag_blocks.append(holes)
            flags = torch.cat(flag_blocks).cpu().numpy()
            if rgb_holes is not None:
                hole_flags = flags[offsets[-1]:]
            count_blocks = [b[4] for b in self.blocks if b[4] is not None]
            count_cpu = torch.cat(count_blocks).cpu().numpy() if count_blocks else None
            camera_cpu = points.cpu().numpy()
        count_offsets = {}
        start = 0
        for i, block in enumerate(self.blocks):
            if block[4] is not None:
                count_offsets[i] = start
                start += len(block[0])
        rows_valid = []
        for row in self.rows:
            if not flags[int(offsets[row["block"]] + row["row"])]:
                continue
            if rgb_holes is not None and row["block"] == rgb_block:
                row["meta"]["contour_hole"] = bool(hole_flags[row["row"]])
            if row["block"] in count_offsets:
                row["meta"]["supporting_planes"] = int(
                    count_cpu[count_offsets[row["block"]] + row["row"]]
                )
            rows_valid.append(row)
        chosen = []
        # Stable greedy order matches the V2 -> V3 -> V4 accuracy contract.
        for stage in (0, 1, 2, 3, 4):
            rows = [r for r in rows_valid if r["stage"] == stage]
            if stage < 2:
                rows.sort(
                    key=lambda r: (
                        -r["meta"]["supporting_planes"],
                        r["meta"]["line_residual_px"],
                    )
                )
            else:
                front = {}
                for r in rows:
                    if r["feature"] is not None:
                        index = int(offsets[r["block"]] + r["row"])
                        key = r["feature"]
                        if (
                            key not in front
                            or camera_cpu[index, 2]
                            < camera_cpu[
                                int(offsets[front[key]["block"]] + front[key]["row"]), 2
                            ]
                        ):
                            front[key] = r
                rows = [
                    r for r in rows if r["feature"] is None or front[r["feature"]] is r
                ]
            if stage == 3 and self._refine_opening:
                # A jointly fitted opening replaces nearby legacy concave points.
                # Outer corners and points without concavity evidence keep priority.
                opening_indices = [int(offsets[r["block"]] + r["row"]) for r in rows]
                opening_points = camera_cpu[opening_indices]
                chosen = [
                    entry
                    for entry in chosen
                    if entry["oid"] != 1
                    or not entry["meta"].get("contour_hole")
                    or not (
                        np.linalg.norm(
                            opening_points - camera_cpu[entry["index"]], axis=1
                        )
                        < 0.008
                    ).any()
                ]
            if stage == 4 and rows:
                indices = [int(offsets[row["block"]] + row["row"]) for row in rows]
                peg_vertices = camera_cpu[indices]
                chosen = [
                    entry
                    for entry in chosen
                    if entry["oid"] != 0
                    or not (
                        np.linalg.norm(
                            peg_vertices - camera_cpu[entry["index"]], axis=1
                        )
                        <= 0.006
                    ).any()
                ]
            for row in rows:
                index = int(offsets[row["block"]] + row["row"])
                oid = self.blocks[row["block"]][2]
                radius = (
                    self.parameters.merge_radius_m
                    if stage == 0
                    else self.parameters.additional_merge_radius_m
                )
                if stage == 3:
                    radius = 0.003
                if (
                    self.hole_rgb_merge_radius_m is not None
                    and row["meta"].get("method") == "box_rgb_quadrants"
                    and row["meta"].get("contour_hole")
                ):
                    radius = self.hole_rgb_merge_radius_m
                near = next(
                    (
                        r
                        for r in chosen
                        if r["oid"] == oid
                        and np.linalg.norm(camera_cpu[r["index"]] - camera_cpu[index])
                        < radius
                    ),
                    None,
                )
                if near is None:
                    chosen.append(dict(index=index, oid=oid, meta=dict(row["meta"])))
                elif stage < 2:
                    near["meta"]["sources"] += 1
        ids = self.tensor([r["index"] for r in chosen], torch.int64)
        self.merged_object_ids = [r["oid"] for r in chosen]
        return (
            points[ids],
            image[ids],
            self.tensor(self.merged_object_ids, torch.int64),
            [r["meta"] for r in chosen],
        )

    @torch.inference_mode()
    def __call__(self, rgb, depth_raw_mm, intrinsic, extrinsic_cv, masks=None):
        start = time.perf_counter()
        if tuple(rgb.shape) != (512, 512, 3) or tuple(depth_raw_mm.squeeze().shape) != (
            512,
            512,
        ):
            raise ValueError("native 512 RGB-D is required")
        with self._measure("prepare_rgbd_and_planes"):
            with self._measure("input_depth_and_calibration"):
                self.k = self.tensor(intrinsic)
                self.ext = self.tensor(extrinsic_cv)
                if self.k.shape != (3, 3) or self.ext.shape != (3, 4):
                    raise ValueError(
                        "intrinsic [3,3] and CV world-to-camera extrinsic [3,4] required"
                    )
                raw = self.tensor(depth_raw_mm).squeeze()
            if self.optimized and not torch.is_tensor(intrinsic):
                self.k_cpu = np.asarray(intrinsic, dtype=np.float32).astype(float)
            else:
                self.k_cpu = self.k.cpu().numpy().astype(float)
            self.depth_cpu = None
            with self._measure("depth_unprojection"):
                self.depth = raw * 0.001
                self.valid = (raw > 0) & (self.depth < 4) & torch.isfinite(self.depth)
                self.xyz = K.unproject(self.depth, self.k)
            with self._measure("rgb_host_view"):
                # OpenCV RGB corner features still need a host image.
                rgb_cpu = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
                rgb_cpu = np.asarray(rgb_cpu, np.uint8)
            # OpenCV already needs host RGB. For host inputs the same color
            # masks can be formed before crossing the device boundary.
            classify_on_device = self._accelerate_rgb and (
                not self.optimized or torch.is_tensor(rgb)
            )
            if classify_on_device:
                with self._measure("rgb_upload"):
                    rgb_device = self.tensor(rgb, torch.uint8)
                with self._measure("rgb_color_classification"):
                    color_masks, neutral = material_torch.material_colors(rgb_device)
                if masks is None:
                    with self._measure("rgb_color_readback"):
                        peg_colors, box_colors = color_masks.cpu().numpy()
                    with self._measure("rgb_material_masks"):
                        core_cpu, recovery_cpu = filter_material_components(
                            peg_colors, box_colors, optimized=self.optimized
                        )
            else:
                with self._measure("rgb_color_classification"):
                    hsv = cv2.cvtColor(rgb_cpu, cv2.COLOR_RGB2HSV)
                    neutral = hsv[..., 1] < 8
                if masks is None:
                    with self._measure("rgb_material_masks"):
                        core_cpu, recovery_cpu = material_masks(hsv, optimized=self.optimized)
            if masks is not None:
                if tuple(masks.shape) != (2, 512, 512):
                    raise ValueError("masks must have shape [2,512,512]")
                core_cpu = (
                    masks.detach().cpu().numpy()
                    if isinstance(masks, torch.Tensor)
                    else np.asarray(masks, bool)
                )
                recovery_cpu = core_cpu
            with self._measure("mask_upload"):
                self.neutral = self.tensor(neutral, torch.bool)
                core = self.tensor(core_cpu, torch.bool)
                recovery = (
                    core if self.optimized and core_cpu is recovery_cpu
                    else self.tensor(recovery_cpu, torch.bool)
                )
            self.current_masks = core
            self.blocks = []
            self.rows = []
            self.box_rgb_holes = None
            self.peg_prism_basis = None
            data_list = []
            recovery_data = []
            valid_cpu = None
            if (
                self.optimized
                and self.device.type != "cpu"
                and not torch.is_tensor(depth_raw_mm)
            ):
                # The host already owns depth. Build row-major gather indices
                # there and keep the unprojected geometry on the accelerator.
                raw_cpu = np.asarray(depth_raw_mm, dtype=np.float32).squeeze()
                self.depth_cpu = raw_cpu * np.float32(0.001)
                valid_cpu = (
                    (raw_cpu > 0) & (self.depth_cpu < 4) & np.isfinite(self.depth_cpu)
                )
            for oid in (0, 1):
                support_cpu = (
                    core_cpu[oid].astype(bool, copy=False) & valid_cpu
                    if valid_cpu is not None else None
                )
                data = self.prepare(oid, core[oid], support_cpu)
                data.core = True
                data_list.append(data)
                if np.array_equal(core_cpu[oid], recovery_cpu[oid]):
                    data.recovery = True
                    recovery_data.append(data)
                else:
                    support_cpu = (
                        recovery_cpu[oid].astype(bool, copy=False) & valid_cpu
                        if valid_cpu is not None else None
                    )
                    extra = self.prepare(oid, recovery[oid], support_cpu)
                    extra.recovery = True
                    data_list.append(extra)
                    recovery_data.append(extra)
        with self._measure("contour_candidates"):
            contours = ContourProposals()
            for data in data_list:
                contours.add(data, self.parameters, reuse_contours=self.optimized)
            contours.finish(self.device, batched=self.optimized)
            for data in data_list:
                self.current_masks = core if data.core else recovery
                self.boundary(data)
                self.junctions(data)
                self.envelopes(data)
        with self._measure("box_exterior"):
            self.current_masks = recovery
            box = recovery_data[1]
            features = []
            if len(box.planes) and box.support_cpu.sum() >= 80:
                features = _corner_features(
                    rgb_cpu, box.support_cpu, optimized=self.optimized
                )
                self.box_cube(box, features)
                self.box_rgb(box, features)
        if self.recover_hole_rim:
            with self._measure("hole_rim"):
                self.box_openings(box)
        if self.recover_small_geometry:
            with self._measure("peg_prism"):
                self.peg_prism(recovery_data[0])
            if self.recover_peg_rear_end:
                with self._measure("peg_rear_end"):
                    self.peg_rear_end(recovery_data[0], box)
        with self._measure("merge"):
            points, uv, object_id, metadata = self.merge()
        with self._measure("position_correction"):
            if self.correct_box_exterior_pixel_center:
                points, uv = self.correct_box_exterior(points, uv, object_id, metadata)
            if self.correct_peg_position:
                points, uv = self.apply_peg_position_correction(points, uv, object_id)
            world = (points - self.ext[:, 3]) @ self.ext[:, :3]
        if self.reject_peg_box_cut:
            with self._measure("peg_box_cut"):
                removed = self.peg_box_cut_indices(points, metadata, recovery_data[0], box)
                if removed:
                    removed = set(removed)
                    remaining = [i for i in range(len(metadata)) if i not in removed]
                    points, uv, object_id, world = (
                        value[remaining] for value in (points, uv, object_id, world)
                    )
                    metadata = [metadata[i] for i in remaining]
        diagnostics = [
            dict(
                object_id=d.oid,
                pixels=int(d.support_cpu.sum()),
                planes=len(d.planes),
                core=d.core,
                recovery=d.recovery,
            )
            for d in data_list
        ]
        diagnostics.append(
            dict(
                box_recovery=dict(features=len(features), planes=len(box.planes)),
                shared_depth=True,
                unique_plane_preparations=len(data_list),
            )
        )
        K.synchronize(self.device)
        return TensorResult(
            object_id,
            uv,
            world,
            points,
            recovery,
            metadata,
            diagnostics,
            (time.perf_counter() - start) * 1000,
            asdict(self.parameters),
        )


@lru_cache(maxsize=4)
def default_detector(device="auto"):
    return TorchCornerDetector.main(device=device)


def detect_tensors(
    rgb,
    depth_raw_mm,
    intrinsic,
    extrinsic_cv,
    masks=None,
    *,
    device="auto",
    parameters=None,
):
    detector = (
        default_detector(str(device))
        if parameters is None
        else TorchCornerDetector.main(parameters, device=device)
    )
    return detector(rgb, depth_raw_mm, intrinsic, extrinsic_cv, masks)


def detect(
    rgb,
    depth_raw_mm,
    intrinsic,
    extrinsic_cv,
    masks=None,
    parameters=None,
    *,
    device="auto",
):
    return detect_tensors(
        rgb,
        depth_raw_mm,
        intrinsic,
        extrinsic_cv,
        masks,
        device=device,
        parameters=parameters,
    ).as_dict()
