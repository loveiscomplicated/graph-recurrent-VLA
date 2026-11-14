"""Torch/CUDA/MPS geometry backend for the independent visible-corner probe.

OpenCV remains responsible for connected components, contour topology and
subpixel RGB features. Dense XYZ never returns to NumPy. Small contour/plane
descriptors and final candidates cross the CPU boundary explicitly. This is a
hybrid detector, not an assertion of an entirely GPU-resident pipeline.
"""

import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations, product

import cv2
import numpy as np
import torch

from .backends import torch_ops as K
from .candidates import ContourProposals, _corner_features, material_masks
from .geometry import Parameters, _face_envelopes
from .hole_rim import opening_corners


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
    cores_cpu: np.ndarray
    support_cpu: np.ndarray
    eligible_cpu: set
    proposals: list
    core: bool = False
    recovery: bool = False
    points: torch.Tensor | None = None


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

    def __init__(
        self,
        device="auto",
        parameters=None,
        *,
        hole_rgb_merge_radius_m=None,
        recover_hole_rim=False,
    ):
        self.device = K.resolve_device(device)
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
        if recover_hole_rim and self.device.type != "cpu":
            raise ValueError("recover_hole_rim currently requires the CPU detector")
        self.recover_hole_rim = recover_hole_rim

    def tensor(self, value, dtype=torch.float32):
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def prepare(self, oid, mask):
        p = self.parameters
        support = mask & self.valid
        points = self.xyz[support]
        planes = K.fit_planes(points, p, 42 + oid, 4 if oid == 0 else 8)
        masks, cores = K.plane_masks(self.xyz, support, planes, p.plane_tolerance_m)
        eligible = cores.flatten(1).sum(1) >= 20
        small = planes.cpu().numpy().astype(float)
        eligible_cpu = set(eligible.nonzero()[:, 0].cpu().tolist())
        return PlaneData(
            oid,
            support,
            planes,
            masks,
            cores,
            eligible,
            [(row[:3], float(row[3])) for row in small],
            masks.cpu().numpy(),
            cores.cpu().numpy(),
            support.cpu().numpy(),
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
            physical, evidence = K.physical_edges(
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
        for pi in sorted(data.eligible_cpu):
            for uv, point, adjacent, hole in _face_envelopes(
                data.masks_cpu[pi].astype(np.uint8),
                pi,
                data.planes_cpu,
                data.eligible_cpu,
                self.k_cpu,
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
        physical, _ = K.physical_edges(
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
            flags = holes.cpu().tolist()
            feature_ids = [p.pop("fi") for p in ps]
            for i, p in enumerate(ps):
                p["contour_hole"] = bool(flags[i])
                p.pop("axes")
            self.add(data, points, uv, selected, ps, stage=2, feature_ids=feature_ids)
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
        physical, _ = K.physical_edges(
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
        flags = interior.cpu().tolist()
        feature_ids = [p.pop("fi") for p in ps]
        for i, p in enumerate(ps):
            p["contour_hole"] = bool(flags[i])
            p.pop("axes")
        self.add(data, points, uv, selected, ps, stage=2, feature_ids=feature_ids)

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
            physical, _ = K.physical_edges(
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

    def box_openings(self, data):
        """Queue verified opening corners after all existing Box candidates."""
        for plane, corners, rectangular, verified in opening_corners(
            self.depth.numpy(),
            data.support_cpu,
            data.planes_cpu,
            data.eligible_cpu,
            self.k_cpu,
        ):
            points = self.tensor(corners)
            if self._refine_opening:
                # Counter the observed subpixel raster bias at fitted rim edges.
                # Keep the adjustment local to the opening recovery path.
                pixel_shift = 0.25 / self.k.diag()[:2]
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
            selected = torch.full((len(corners),), verified, dtype=torch.bool)
            self.add(data, points, uv, selected, metadata, stage=3)

    def merge(self):
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
        flags = torch.cat([b[3] for b in self.blocks]).cpu().numpy()
        count_blocks = [b[4] for b in self.blocks if b[4] is not None]
        count_cpu = torch.cat(count_blocks).cpu().numpy() if count_blocks else None
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
            if row["block"] in count_offsets:
                row["meta"]["supporting_planes"] = int(
                    count_cpu[count_offsets[row["block"]] + row["row"]]
                )
            rows_valid.append(row)
        camera_cpu = points.cpu().numpy()
        chosen = []
        # Stable greedy order matches the V2 -> V3 -> V4 accuracy contract.
        for stage in (0, 1, 2, 3):
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
        return (
            points[ids],
            image[ids],
            self.tensor([r["oid"] for r in chosen], torch.int64),
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
        self.k = self.tensor(intrinsic)
        self.ext = self.tensor(extrinsic_cv)
        if self.k.shape != (3, 3) or self.ext.shape != (3, 4):
            raise ValueError(
                "intrinsic [3,3] and CV world-to-camera extrinsic [3,4] required"
            )
        self.k_cpu = self.k.cpu().numpy().astype(float)
        raw = self.tensor(depth_raw_mm).squeeze()
        self.depth = raw * 0.001
        self.valid = (raw > 0) & (self.depth < 4) & torch.isfinite(self.depth)
        self.xyz = K.unproject(self.depth, self.k)
        # Explicit topology bridge: the images go to OpenCV, not dense XYZ.
        rgb_cpu = (
            rgb.detach().cpu().numpy()
            if isinstance(rgb, torch.Tensor)
            else np.asarray(rgb)
        )
        rgb_cpu = np.asarray(rgb_cpu, np.uint8)
        hsv = cv2.cvtColor(rgb_cpu, cv2.COLOR_RGB2HSV)
        self.neutral = self.tensor(hsv[..., 1] < 8, torch.bool)
        if masks is None:
            core_cpu, recovery_cpu = material_masks(hsv)
        else:
            if tuple(masks.shape) != (2, 512, 512):
                raise ValueError("masks must have shape [2,512,512]")
            core_cpu = (
                masks.detach().cpu().numpy()
                if isinstance(masks, torch.Tensor)
                else np.asarray(masks, bool)
            )
            recovery_cpu = core_cpu
        core = self.tensor(core_cpu, torch.bool)
        recovery = self.tensor(recovery_cpu, torch.bool)
        self.current_masks = core
        self.blocks = []
        self.rows = []
        data_list = []
        recovery_data = []
        for oid in (0, 1):
            data = self.prepare(oid, core[oid])
            data.core = True
            data_list.append(data)
            if np.array_equal(core_cpu[oid], recovery_cpu[oid]):
                data.recovery = True
                recovery_data.append(data)
            else:
                extra = self.prepare(oid, recovery[oid])
                extra.recovery = True
                data_list.append(extra)
                recovery_data.append(extra)
        contours = ContourProposals()
        for data in data_list:
            contours.add(data, self.parameters)
        contours.finish(self.device)
        for data in data_list:
            self.current_masks = core if data.core else recovery
            self.boundary(data)
            self.junctions(data)
            self.envelopes(data)
        self.current_masks = recovery
        box = recovery_data[1]
        features = []
        if len(box.planes) and box.support_cpu.sum() >= 80:
            features = _corner_features(rgb_cpu, box.support_cpu)
            self.box_cube(box, features)
            self.box_rgb(box, features)
        if self.recover_hole_rim:
            self.box_openings(box)
        points, uv, object_id, metadata = self.merge()
        world = (points - self.ext[:, 3]) @ self.ext[:, :3]
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
    return TorchCornerDetector(device)


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
        else TorchCornerDetector(device, parameters)
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
