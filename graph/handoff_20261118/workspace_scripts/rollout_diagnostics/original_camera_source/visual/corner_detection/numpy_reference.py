"""Explicit NumPy reference detector for regression checks; never an implicit fallback."""

from dataclasses import asdict, replace
from itertools import combinations
import time
import cv2
import numpy as np
from .geometry import (
    Parameters,
    color_masks,
    unproject,
    _planes,
    _snap_planes,
    _visible_at_corner,
    _unoccluded_footprint,
    _physical_edges,
    _local_boundary_corners,
    _face_envelopes,
    _polygon_corners,
)


def detect(rgb, depth_raw_mm, intrinsic, extrinsic_cv, masks=None, parameters=None):
    """Return only current, image-supported candidates in world coordinates."""
    started = time.perf_counter()
    from .planar_reference import detect as baseline_detect

    baseline = baseline_detect(rgb, depth_raw_mm, intrinsic, extrinsic_cv, masks=masks)
    p = parameters or Parameters()
    rgb = np.asarray(rgb)
    raw = np.asarray(depth_raw_mm).squeeze()
    if rgb.shape != (512, 512, 3) or raw.shape != (512, 512):
        raise ValueError("native 512 RGB-D is required")
    masks = color_masks(rgb, 8) if masks is None else np.asarray(masks, bool)
    if masks.shape != (2, 512, 512):
        raise ValueError("masks must be [peg,box, H,W] = [2,512,512]")
    depth = raw.astype(np.float64) * 0.001
    valid = (raw > 0) & (depth < 4)
    # Current RGB-D evidence of contact/occlusion, independent of robot GT/FK.
    neutral = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1] < 8
    xyz = unproject(depth, intrinsic)
    ext = np.asarray(extrinsic_cv)
    rot, translation = ext[:, :3], ext[:, 3]
    candidates = []
    diagnostics = []
    for oid in (0, 1):
        support = masks[oid] & valid
        vv, uu = np.nonzero(support)
        pts = xyz[vv, uu]
        if len(pts) < p.minimum_plane_pixels:
            diagnostics.append(dict(object_id=oid, pixels=len(pts), planes=0))
            continue
        planes = _planes(pts, p, np.random.default_rng(42 + oid), 4 if oid == 0 else 8)
        distances = (
            np.stack([np.abs(pts @ n + d) for n, d in planes], -1)
            if planes
            else np.empty((len(pts), 0))
        )
        labels = distances.argmin(1) if planes else np.zeros(len(pts), int)
        plane_masks = []
        for pi, (normal, offset) in enumerate(planes):
            plane_mask = np.zeros(raw.shape, np.uint8)
            selected = (labels == pi) & (distances[:, pi] < p.plane_tolerance_m)
            plane_mask[vv[selected], uu[selected]] = 1
            # Close only one-pixel sampling cracks, not a whole missing face.
            plane_mask = cv2.morphologyEx(
                plane_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )
            plane_masks.append(plane_mask)
        plane_cores = [cv2.erode(m, np.ones((3, 3), np.uint8)) for m in plane_masks]
        eligible_planes = {i for i, m in enumerate(plane_cores) if m.sum() >= 20}
        for pi, (normal, offset) in enumerate(planes):
            plane_mask = plane_masks[pi]
            contours, hierarchy = cv2.findContours(
                plane_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            for ci, contour in enumerate(contours):
                if abs(cv2.contourArea(contour)) < 22:
                    continue
                is_hole = bool(hierarchy[0, ci, 3] >= 0)
                proposals = [
                    (*c, "boundary") for c in _polygon_corners(contour, p, is_hole)
                ]
                proposals += [
                    (*c, "local_boundary")
                    for c in _local_boundary_corners(contour, p, is_hole)
                ]
                short = replace(p, polygon_epsilon_px=1.0, minimum_edge_px=4.0)
                if oid == 1:
                    proposals += [
                        (*c, "short_boundary")
                        for c in _polygon_corners(contour, short, is_hole)
                    ]
                for uv, line_residual, adjacent, origin in proposals:
                    original_uv = uv.copy()
                    if origin == "short_boundary" and not (
                        8 <= uv[0] < 504 and 8 <= uv[1] < 504
                    ):
                        continue
                    ray = np.array(
                        [
                            (uv[0] - intrinsic[0, 2]) / intrinsic[0, 0],
                            (uv[1] - intrinsic[1, 2]) / intrinsic[1, 1],
                            1.0,
                        ]
                    )
                    denom = normal @ ray
                    if abs(denom) < 0.08:
                        continue
                    point = ray * (-offset / denom)
                    adjacent_3d = []
                    for xy in adjacent:
                        r = np.array(
                            [
                                (xy[0] - intrinsic[0, 2]) / intrinsic[0, 0],
                                (xy[1] - intrinsic[1, 2]) / intrinsic[1, 1],
                                1.0,
                            ]
                        )
                        adjacent_3d.append(r * (-offset / (normal @ r)) - point)
                    aa, bb = adjacent_3d
                    cosine = abs(aa @ bb) / (
                        np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12
                    )
                    if cosine > p.maximum_angle_cosine:
                        continue
                    physical, edge_evidence = _physical_edges(
                        uv,
                        adjacent,
                        normal,
                        offset,
                        plane_mask,
                        masks[oid],
                        depth,
                        valid,
                        xyz,
                        intrinsic,
                        neutral,
                    )
                    if (origin != "boundary" or cosine > 0.28) and not physical:
                        continue
                    u, v = np.round(uv).astype(int)
                    if not 2 <= u < 510 or not 2 <= v < 510 or point[2] <= 0:
                        continue
                    patch = plane_mask[v - 3 : v + 4, u - 3 : u + 4]
                    y, x = np.nonzero(patch)
                    if (
                        len(x) < 5
                        or np.min(np.hypot(x + u - 3 - uv[0], y + v - 3 - uv[1]))
                        > p.maximum_support_gap_px
                    ):
                        continue
                    # A foreground surface in front of the proposed corner is occlusion.
                    occluder = valid[v - 3 : v + 4, u - 3 : u + 4] & (
                        depth[v - 3 : v + 4, u - 3 : u + 4] < point[2] - 0.020
                    )
                    if occluder.sum() >= 3 and not (
                        physical and _visible_at_corner(point, uv, support, xyz)
                    ):
                        continue
                    contact = (
                        valid[v - 3 : v + 4, u - 3 : u + 4]
                        & neutral[v - 3 : v + 4, u - 3 : u + 4]
                        & ~masks[oid, v - 3 : v + 4, u - 3 : u + 4]
                        & (
                            np.linalg.norm(
                                xyz[v - 3 : v + 4, u - 3 : u + 4] - point, axis=-1
                            )
                            < 0.006
                        )
                    )
                    if contact.sum() >= 3:
                        continue
                    point, supporting_planes = _snap_planes(
                        point, pi, planes, eligible_planes
                    )
                    uv = (
                        point[:2]
                        / point[2]
                        * np.array([intrinsic[0, 0], intrinsic[1, 1]])
                        + intrinsic[:2, 2]
                    )
                    if origin != "boundary" and np.linalg.norm(uv - original_uv) > 2.0:
                        continue
                    if origin != "boundary" and not _visible_at_corner(
                        point, uv, support, xyz
                    ):
                        continue
                    world = (point - translation) @ rot
                    candidates.append(
                        dict(
                            object_id=oid,
                            uv=uv.tolist(),
                            xyz=world.tolist(),
                            camera_xyz=point.tolist(),
                            plane=pi,
                            line_residual_px=line_residual,
                            contour_hole=bool(hierarchy[0, ci, 3] >= 0),
                            sources=1,
                            supporting_planes=supporting_planes,
                            method=origin,
                            edge_evidence=edge_evidence,
                        )
                    )
        # The visible trihedral junction of three faces is inside the silhouette.
        # It can be missed by contour simplification; require current support on
        # all three faces rather than completing an unseen cuboid.
        for indices in combinations(range(len(planes)), 3):
            if not set(indices).issubset(eligible_planes):
                continue
            normals = np.stack([planes[i][0] for i in indices])
            if abs(np.linalg.det(normals)) < 0.85:
                continue
            if np.max(np.abs(normals @ normals.T - np.eye(3))) > 0.25:
                continue
            point = np.linalg.solve(normals, -np.array([planes[i][1] for i in indices]))
            if point[2] <= 0:
                continue
            uv = (
                point[:2] / point[2] * np.array([intrinsic[0, 0], intrinsic[1, 1]])
                + intrinsic[:2, 2]
            )
            u, v = np.round(uv).astype(int)
            if not 6 <= u < 506 or not 6 <= v < 506:
                continue
            current = support[v - 4 : v + 5, u - 4 : u + 5]
            local_xyz = xyz[v - 4 : v + 5, u - 4 : u + 5]
            nearby = np.linalg.norm(local_xyz - point, axis=-1) < 0.006
            if not all(
                np.count_nonzero(
                    plane_masks[i][v - 4 : v + 5, u - 4 : u + 5] & current & nearby
                )
                >= 3
                for i in indices
            ):
                continue
            if not all(
                np.count_nonzero(plane_cores[i][v - 6 : v + 7, u - 6 : u + 7]) >= 2
                for i in indices
            ):
                continue
            foreground = valid[v - 1 : v + 2, u - 1 : u + 2] & (
                depth[v - 1 : v + 2, u - 1 : u + 2] < point[2] - 0.004
            )
            if foreground.sum() >= 2:
                continue
            candidates.append(
                dict(
                    object_id=oid,
                    uv=uv.tolist(),
                    xyz=((point - translation) @ rot).tolist(),
                    camera_xyz=point.tolist(),
                    plane=int(indices[0]),
                    line_residual_px=0.0,
                    contour_hole=False,
                    sources=1,
                    supporting_planes=3,
                    method="three_visible_planes",
                )
            )
        for pi in sorted(eligible_planes) if oid == 0 else []:
            for uv, point, adjacent, hole in _face_envelopes(
                plane_masks[pi], pi, planes, eligible_planes, intrinsic
            ):
                if not _visible_at_corner(point, uv, support, xyz):
                    continue
                physical, evidence = _physical_edges(
                    uv,
                    adjacent,
                    *planes[pi],
                    plane_masks[pi],
                    masks[oid],
                    depth,
                    valid,
                    xyz,
                    intrinsic,
                    neutral,
                )
                if not physical:
                    continue
                u, v = np.round(uv).astype(int)
                contact = (
                    valid[v - 2 : v + 3, u - 2 : u + 3]
                    & neutral[v - 2 : v + 3, u - 2 : u + 3]
                    & ~masks[oid, v - 2 : v + 3, u - 2 : u + 3]
                    & (
                        np.linalg.norm(
                            xyz[v - 2 : v + 3, u - 2 : u + 3] - point, axis=-1
                        )
                        < 0.006
                    )
                )
                if contact.sum() >= 3:
                    continue
                original_uv = uv.copy()
                point, count = _snap_planes(point, pi, planes, eligible_planes)
                uv = (
                    point[:2] / point[2] * np.array([intrinsic[0, 0], intrinsic[1, 1]])
                    + intrinsic[:2, 2]
                )
                if not (8 <= uv[0] < 504 and 8 <= uv[1] < 504):
                    continue
                if np.linalg.norm(uv - original_uv) > 2.0:
                    continue
                if not _visible_at_corner(point, uv, support, xyz):
                    continue
                candidates.append(
                    dict(
                        object_id=oid,
                        uv=uv.tolist(),
                        xyz=((point - translation) @ rot).tolist(),
                        camera_xyz=point.tolist(),
                        plane=pi,
                        line_residual_px=0.4,
                        contour_hole=hole,
                        sources=1,
                        supporting_planes=count,
                        method="observed_face_envelope",
                        edge_evidence=evidence,
                    )
                )
        diagnostics.append(dict(object_id=oid, pixels=len(pts), planes=len(planes)))
    # Nearby candidates supported by adjacent faces become one node.
    # Keep the verified core detector's candidates; recovery may add genuinely
    # new observed junctions, but cannot replace a core point by a noisier fit.
    merged = [dict(c, method="v2_core") for c in baseline["corners"]]
    for c in sorted(
        candidates, key=lambda c: (-c["supporting_planes"], c["line_residual_px"])
    ):
        if not _unoccluded_footprint(
            np.array(c["camera_xyz"]), np.array(c["uv"]), depth, valid
        ):
            continue
        near = next(
            (
                m
                for m in merged
                if m["object_id"] == c["object_id"]
                and np.linalg.norm(np.array(m["xyz"]) - c["xyz"])
                < p.additional_merge_radius_m
            ),
            None,
        )
        if near is None:
            merged.append(c.copy())
        else:
            # Keep the better line fit, don't average incompatible depth faces.
            near["sources"] += 1
    from .box_recovery import recover

    extra, box_diagnostics = recover(rgb, raw, intrinsic, ext, merged, masks=masks)
    merged.extend(extra)
    diagnostics.append(dict(box_recovery=box_diagnostics))
    return dict(
        corners=merged,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        masks=masks,
        diagnostics=diagnostics,
        parameters=asdict(p),
    )
