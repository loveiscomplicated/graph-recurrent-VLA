"""Evaluation-only corner geometry; never imported by detector.py."""

from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from .geometry import unproject


def transform(local, pose):
    q = pose[3:]
    return (
        np.asarray(local) @ Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T
        + pose[:3]
    )


def corners(gt):
    half = np.asarray(gt["peg_half_sizes"]).reshape(3)
    length = half[0]
    radius = float(np.asarray(gt["hole_radius"]).ravel()[0])
    rows = []
    for kind, pose, scale, oid in (
        ("peg", gt["peg_pose"], half, 0),
        ("box_outer", gt["box_pose"], [length] * 3, 1),
        ("hole_rim", gt["hole_pose"], [length, radius, radius], 1),
    ):
        pose = np.asarray(pose).reshape(7)
        for i, sign in enumerate(product((-1, 1), repeat=3)):
            local = np.array(sign) * scale
            rows.append(
                dict(
                    kind=kind,
                    object_id=oid,
                    corner_id=i,
                    xyz=transform(local[None], pose)[0],
                )
            )
    return rows


def visible_reference(gt, sensor, ids):
    """Visibility fixed independently of detector: nearby actor depth support.

    A physical vertex is supported if inside the image and an observed surface
    of the correct actor is within 6 mm in a 7x7 pixel neighbourhood. This is a
    raster visibility approximation, recorded explicitly rather than treating
    every 3-D box vertex as visible. No detector output enters this rule.
    """
    ext, k = sensor["extrinsic_cv"], sensor["intrinsic_cv"]
    depth = np.asarray(sensor["depth"]).squeeze().astype(float) * 0.001
    segmentation = np.asarray(sensor["segmentation"]).squeeze()
    xyz = unproject(depth, k)
    result = []
    for row in corners(gt):
        point = np.asarray(row["xyz"]) @ ext[:, :3].T + ext[:, 3]
        if point[2] <= 0:
            continue
        uv = point[:2] / point[2] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]
        u, v = np.round(uv).astype(int)
        if not 3 <= u < 509 or not 3 <= v < 509:
            continue
        sub = xyz[v - 3 : v + 4, u - 3 : u + 4]
        actor = segmentation[v - 3 : v + 4, u - 3 : u + 4] == ids[row["object_id"]]
        valid = (depth[v - 3 : v + 4, u - 3 : u + 4] > 0) & actor
        if valid.any() and np.min(np.linalg.norm(sub[valid] - point, axis=1)) <= 0.006:
            result.append({**row, "uv": uv})
    return result


def score(predictions, refs, all_refs, threshold=0.003):
    result = []
    for oid in (0, 1):
        pred = [p for p in predictions if p["object_id"] == oid]
        target = [r for r in refs if r["object_id"] == oid]
        a = np.asarray([p["xyz"] for p in pred]).reshape(-1, 3)
        b = np.asarray([r["xyz"] for r in target]).reshape(-1, 3)
        hits = []
        errors = []
        if len(a) and len(b):
            distance = np.linalg.norm(a[:, None] - b[None], axis=-1)
            # Maximize 3-mm matches first, then distance. One-to-one, no duplicates.
            cost = distance + 100 * (distance > threshold)
            pi, ri = linear_sum_assignment(cost)
            for p, r in zip(pi, ri):
                errors.append(float(distance[p, r] * 1000))
                if distance[p, r] <= threshold:
                    hits.append(
                        dict(
                            prediction=int(p),
                            reference=int(r),
                            kind=target[r]["kind"],
                            error_mm=float(distance[p, r] * 1000),
                        )
                    )
        nearest = []
        if len(a) and len(b):
            nearest = (
                np.min(np.linalg.norm(a[:, None] - b[None], axis=-1), axis=1) * 1000
            ).tolist()
        result.append(
            dict(
                object_id=oid,
                predictions=len(a),
                visible_reference=len(b),
                matched_3mm=len(hits),
                precision_3mm=len(hits) / len(a) if len(a) else None,
                recall_3mm=len(hits) / len(b) if len(b) else None,
                nearest_visible_errors_mm=nearest,
                assignment_errors_mm=errors,
                matches=hits,
                by_kind={
                    kind: dict(
                        visible=sum(r["kind"] == kind for r in target),
                        matched=sum(h["kind"] == kind for h in hits),
                    )
                    for kind in ("peg", "box_outer", "hole_rim")
                    if any(r["kind"] == kind for r in target)
                },
            )
        )
    return result
