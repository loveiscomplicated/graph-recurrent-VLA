"""Camera overlays and box-local aperture view for geometry inspection."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .common import config, DEFAULT_CONFIG
from .maniskill_gt import GTAdapter, validate_live_collision
from .visualization import plt


def overlay(rgb, positions, colors, intrinsic, extrinsic):
    image = Image.fromarray(rgb.cpu().numpy())
    draw = ImageDraw.Draw(image)
    camera = positions @ extrinsic[:, :3].T + extrinsic[:, 3]
    pixels = camera @ intrinsic.T
    for point, pixel, color in zip(camera, pixels, colors):
        if point[2] <= 0:
            continue
        x, y = pixel[:2]/pixel[2]
        if 0 <= x < image.width and 0 <= y < image.height:
            radius = 2 if color in ("red", "lime", "cyan", "white") else 1
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("outputs/graph/camera_diagnostics"))
    args = p.parse_args()
    cfg = config(args.config)
    from envs.maniskill import make_peg_insertion_env, get_gt_state
    env = make_peg_insertion_env(seed=args.seed, obs_mode="rgb", control_mode=cfg["control_mode"], render_mode="rgb_array")
    try:
        obs = env.unwrapped.get_obs()
        g = GTAdapter(cfg["peg_points"], cfg["hole_points"], cfg["sampling_seed"])(env)
        print(validate_live_collision(env, g))
        s = get_gt_state(env)
        positions = torch.cat((g.ee_pose[:, None, :3], g.left_fingertip[:, None], g.right_fingertip[:, None],
            s.left_finger_pose[:, None, :3], s.right_finger_pose[:, None, :3], g.peg_points, g.hole_points), 1)[0].cpu().numpy()
        colors = ["red", "lime", "cyan", "white", "white"] + ["orange"]*cfg["peg_points"]+["magenta"]*cfg["hole_points"]
        args.output.mkdir(parents=True, exist_ok=True)
        for name, camera in obs["sensor_data"].items():
            params = obs["sensor_param"][name]
            image = overlay(camera["rgb"][0], positions, colors,
                            params["intrinsic_cv"][0].cpu().numpy(), params["extrinsic_cv"][0].cpu().numpy())
            image.resize((512, 512), Image.Resampling.NEAREST).save(args.output/f"{name}_overlay.png")
        Image.fromarray(env.render()[0].cpu().numpy()).save(args.output/"scene.png")
        # Looking along the hole axis makes an incorrectly filled aperture clear.
        points = g.hole_local[0].cpu().numpy()
        cy, cz = g.hole_offset[0, 1:].tolist()
        radius = float(g.hole_radius[0, 0])
        half = float(g.box_half_size[0, 0])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(points[:, 1], points[:, 2], c=points[:, 0], cmap="coolwarm", s=18)
        for ylo, yhi, zlo, zhi, color in ((-half, half, -half, half, "black"),
                                        (cy-radius, cy+radius, cz-radius, cz+radius, "green")):
            ax.plot([ylo, yhi, yhi, ylo, ylo], [zlo, zlo, zhi, zhi, zlo], c=color)
        ax.set(xlabel="box-local y (m)", ylabel="box-local z (m)", title="Projected collision surface: aperture interior stays empty")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(args.output/"aperture.png", dpi=150)
        plt.close(fig)
    finally:
        env.close()


if __name__ == "__main__":
    main()
