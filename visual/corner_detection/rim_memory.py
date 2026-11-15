"""Causal Hole rim recovery for a stationary Box within one episode."""

import numpy as np

from .geometry import unproject


class StaticRimMemory:
    """Recall earlier rim observations only where current Box depth supports them."""

    def __init__(self):
        self.points = []

    def update(self, result, depth_mm, intrinsic, extrinsic, alternatives=()):
        """Return current predictions plus supported, spatially distinct old rims."""
        predictions = result.as_dict()["corners"]
        box = result.masks[1].cpu().numpy()
        depth = np.asarray(depth_mm).squeeze() * 0.001
        pixels = unproject(depth, intrinsic)
        focal = np.diag(intrinsic)[:2]
        rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
        current = [np.asarray(p["xyz"]) for p in predictions if p["object_id"] == 1]
        candidates = [
            np.asarray(row["xyz"])
            for row in alternatives
            if row["method"] == "box_opening"
        ]
        for point in [*candidates, *self.points]:
            if any(np.linalg.norm(point - other) < 0.002 for other in current):
                continue
            camera_point = rotation @ point + translation
            if camera_point[2] <= 0:
                continue
            uv = focal * camera_point[:2] / camera_point[2] + intrinsic[:2, 2]
            u, v = np.rint(uv).astype(int)
            height, width = box.shape
            if not (3 <= u < width - 3 and 3 <= v < height - 3):
                continue
            patch = np.s_[v - 3 : v + 4, u - 3 : u + 4]
            nearby = np.linalg.norm(pixels[patch] - camera_point, axis=-1) < 0.006
            if not (nearby & box[patch] & (depth[patch] > 0)).any():
                continue
            predictions.append(
                {
                    "xyz": point.tolist(),
                    "uv": uv.tolist(),
                    "object_id": 1,
                    "method": "rim_memory",
                    "contour_hole": True,
                }
            )
            current.append(point)
        for prediction in [*predictions, *alternatives]:
            if prediction["method"] != "box_opening":
                continue
            point = np.asarray(prediction["xyz"])
            if not any(np.linalg.norm(point - other) < 0.0005 for other in self.points):
                self.points.append(point)
        return predictions


def replace_rim_square(predictions, square, intrinsic, extrinsic):
    """Substitute a verified square only near the current emitted rim."""
    if square is None:
        return predictions
    camera = square.copy()
    camera[:, :2] += 0.375 * camera[:, 2:] / intrinsic.diagonal()[:2]
    world = (camera - extrinsic[:, 3]) @ extrinsic[:, :3]
    rim_methods = ("box_opening", "rim_memory")
    rim = np.array(
        [p["xyz"] for p in predictions if p["method"] in rim_methods]
    ).reshape(-1, 3)
    if len(rim):
        separation = np.linalg.norm(world[:, None] - rim, axis=-1)
        if separation.min(axis=1).max() > 0.007:
            return predictions
    kept = [p for p in predictions if p["method"] not in rim_methods]
    box = np.array([p["xyz"] for p in kept if p["object_id"] == 1]).reshape(-1, 3)
    uv = camera[:, :2] / camera[:, 2:] * intrinsic.diagonal()[:2] + intrinsic[:2, 2]
    for point, pixel, camera_point in zip(world, uv, camera, strict=True):
        if np.any(np.linalg.norm(box - point, axis=1) < 0.003):
            continue
        kept.append(
            {
                "xyz": point.tolist(),
                "uv": pixel.tolist(),
                "camera_xyz": camera_point.tolist(),
                "object_id": 1,
                "method": "box_opening",
                "contour_hole": True,
            }
        )
    return kept
