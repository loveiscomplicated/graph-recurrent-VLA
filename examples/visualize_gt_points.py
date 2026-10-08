"""Save camera images with projected PegInsertionSide-v1 GT points."""

from __future__ import annotations

import argparse
from itertools import product
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw

from envs.maniskill import make_peg_insertion_env


COLORS = {
    "TCP": (255, 60, 60),
    "L_tip": (60, 255, 60),
    "R_tip": (60, 160, 255),
    "peg": (255, 210, 40),
    "hole": (255, 70, 220),
}


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] wxyz quaternions to rotation matrices."""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _box_corners(half_sizes: torch.Tensor) -> torch.Tensor:
    signs = torch.tensor(
        list(product((-1.0, 1.0), repeat=3)),
        device=half_sizes.device,
        dtype=half_sizes.dtype,
    )
    return signs * half_sizes


def _local_to_world(local_points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    rotation = _quat_wxyz_to_matrix(pose[3:])
    return local_points @ rotation.T + pose[:3]


def _collect_points(state) -> tuple[np.ndarray, list[str]]:
    """Return world XYZ points and color keys for the first env."""
    points = [state.ee_pose[0, :3], state.left_fingertip_pos[0], state.right_fingertip_pos[0]]
    groups = ["TCP", "L_tip", "R_tip"]

    peg_pose = state.peg_pose[0]
    peg_corners = _local_to_world(_box_corners(state.peg_half_sizes[0]), peg_pose)
    points.extend(peg_corners.unbind(0))
    groups.extend(["peg"] * 8)

    # The task's square aperture lies in the hole frame's yz plane; x is its axis.
    radius = state.hole_radius[0]
    hole_local = torch.tensor(
        [[0.0, sy * radius, sz * radius] for sy, sz in product((-1.0, 1.0), repeat=2)],
        device=radius.device,
        dtype=radius.dtype,
    )
    hole_corners = _local_to_world(hole_local, state.hole_pose[0])
    points.extend(hole_corners.unbind(0))
    groups.extend(["hole"] * 4)

    return torch.stack(points).detach().cpu().numpy(), groups


def _draw_projected_points(
    rgb: torch.Tensor,
    points_world: np.ndarray,
    groups: list[str],
    intrinsic_cv: torch.Tensor,
    extrinsic_cv: torch.Tensor,
    point_radius: int,
) -> Image.Image:
    image = Image.fromarray(rgb.detach().cpu().numpy().astype(np.uint8))
    draw = ImageDraw.Draw(image)
    intrinsic = intrinsic_cv.detach().cpu().numpy()
    extrinsic = extrinsic_cv.detach().cpu().numpy()
    points_camera = points_world @ extrinsic[:, :3].T + extrinsic[:, 3]
    projected = points_camera @ intrinsic.T

    width, height = image.size
    for point, pixel, group in zip(points_camera, projected, groups):
        if point[2] <= 0 or pixel[2] <= 0:
            continue
        u, v = pixel[0] / pixel[2], pixel[1] / pixel[2]
        if not (0 <= u < width and 0 <= v < height):
            continue
        x, y = int(round(u)), int(round(v))
        color = COLORS[group]
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--output", type=Path, default=Path("outputs/gt_points"))
    parser.add_argument("--point-radius", type=int, default=2, help="Marker radius in pixels (default: 2)")
    args = parser.parse_args()
    if args.point_radius < 1:
        parser.error("--point-radius must be at least 1")

    env = make_peg_insertion_env(seed=0, obs_mode="rgb", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        if args.camera not in obs["sensor_data"]:
            parser.error(f"unknown camera {args.camera!r}; choose from {list(obs['sensor_data'])}")

        from envs.maniskill import get_gt_state

        state = get_gt_state(env)
        points_world, groups = _collect_points(state)
        camera = obs["sensor_data"][args.camera]
        camera_params = obs["sensor_param"][args.camera]
        image = _draw_projected_points(
            camera["rgb"][0],
            points_world,
            groups,
            camera_params["intrinsic_cv"][0],
            camera_params["extrinsic_cv"][0],
            args.point_radius,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / f"{args.camera}_gt_points.png"
        image.save(output_path)
        print(f"Saved projected GT points to {output_path.resolve()}")
        print(f"Camera: {args.camera}; points: {len(points_world)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
