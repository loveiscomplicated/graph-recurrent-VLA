"""Can the lateral peg-to-hole error be read from the policy's input images at all?

Closed-loop failures stall at insertion depth with ~2 cm lateral error even after
DART recovery data. This probe trains a small ResNet18 regressor on the same
inputs the policies get, predicting the peg head position in the hole frame
(y, z lateral; x depth) for frames near the hole. Large probe error means the
inputs (resolution / viewpoint) cannot support mm-level alignment.

Example:
    python -m visual.baseline.probe_lateral --output outputs/visual_baseline/probe_lateral.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EPISODES = Path("data/visual_baseline/episodes")
NEAR_HOLE = (-0.25, -0.05)  # head depth along the hole axis (m) where alignment happens
HELD_ONLY = False  # set by --held-only
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by wxyz quaternions (both batched along axis 0)."""
    w, xyz = quat[:, :1], quat[:, 1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector + w * t + np.cross(xyz, t)


def head_in_hole(episode) -> np.ndarray:
    """Peg head position expressed in the hole frame, per frame (T, 3) in metres."""
    peg, hole, half = episode["gt_peg_pose"], episode["gt_hole_pose"], episode["gt_peg_half_sizes"]
    head_local = np.stack([half[:, 0], np.zeros(len(half)), np.zeros(len(half))], axis=1)
    head = peg[:, :3] + quat_rotate(peg[:, 3:], head_local)
    hole_conjugate = hole[:, 3:] * np.array([1, -1, -1, -1])
    return quat_rotate(hole_conjugate, head - hole[:, :3])


def load_frames(files: list[str]):
    """Images of both cameras and head-in-hole targets for frames near the hole."""
    def load(name):
        with np.load(EPISODES / name) as episode:
            target = head_in_hole(episode)
            keep = (target[:, 0] >= NEAR_HOLE[0]) & (target[:, 0] <= NEAR_HOLE[1])
            if HELD_ONLY:
                lift = episode["gt_peg_pose"][:, 2] - episode["gt_peg_pose"][0, 2]
                keep &= lift > 0.03  # peg in the gripper, not lying on the table
            return (episode["main_rgb"][keep], episode["wrist_rgb"][keep], target[keep],
                    episode["main_depth"][keep])

    with ThreadPoolExecutor(16) as pool:
        parts = list(pool.map(load, files))
    main = torch.from_numpy(np.concatenate([p[0] for p in parts]))
    wrist = torch.from_numpy(np.concatenate([p[1] for p in parts]))
    target = torch.from_numpy(np.concatenate([p[2] for p in parts])).float()
    depth = torch.from_numpy(np.concatenate([p[3] for p in parts]))  # int16 mm, N1HW at 128 px
    return main, wrist, target, depth


class Probe(nn.Module):
    """One ImageNet ResNet18 per camera, concatenated features → (x, y, z)."""

    def __init__(self, cameras: int):
        super().__init__()
        self.encoders = nn.ModuleList()
        for _ in range(cameras):
            backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            backbone.fc = nn.Identity()
            self.encoders.append(backbone)
        self.head = nn.Sequential(nn.Linear(512 * cameras, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        return self.head(torch.cat([encoder(x) for encoder, x in zip(self.encoders, images)], dim=1))


def prepare(images: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 RGB NCHW → resized, ImageNet-normalized float; int16 depth (mm) → 3-channel metres."""
    if images.dtype == torch.int16:
        depth = images.cuda(non_blocking=True).float().clamp(min=0) / 1000.0
        depth = F.interpolate(depth, size=(size, size), mode="nearest")
        return (depth - 1.0).repeat(1, 3, 1, 1)  # centred around ~1 m
    x = images.cuda(non_blocking=True).float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="area")
    return (x - IMAGENET_MEAN.cuda()) / IMAGENET_STD.cuda()


def run_probe(name, cameras, size, train, validation, steps, batch_size=64) -> dict:
    """Train one probe variant and return validation errors in millimetres."""
    torch.manual_seed(0)
    # Tensor positions in the loaded tuple: 0 main RGB, 1 wrist RGB, 3 main depth.
    picks = {"main": [0], "wrist": [1], "both": [0, 1], "main_rgbd": [0, 3], "all": [0, 1, 3]}[cameras]
    model = Probe(len(picks)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scale = 100.0  # predict centimetres for a well-conditioned loss
    started = time.time()
    model.train()
    for step in range(steps):
        index = torch.randint(len(train[2]), (batch_size,))
        images = [prepare(train[i][index], size) for i in picks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.smooth_l1_loss(model(images).float(), train[2][index].cuda() * scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    errors = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(validation[2]), 256):
            sl = slice(start, start + 256)
            prediction = model([prepare(validation[i][sl], size) for i in picks]).float() / scale
            errors.append((prediction.cpu() - validation[2][sl]).abs())
    errors = torch.cat(errors) * 1000.0
    lateral = errors[:, 1:].norm(dim=1)
    result = dict(name=name, cameras=cameras, size=size, steps=steps,
                  mae_mm_xyz=[round(float(v), 2) for v in errors.mean(0)],
                  lateral_norm_mm_median=round(float(lateral.median()), 2),
                  lateral_norm_mm_p90=round(float(lateral.quantile(0.9)), 2),
                  target_lateral_std_mm=round(float(validation[2][:, 1:].std(0).norm() * 1000), 2),
                  seconds=round(time.time() - started))
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--variants", default="both_256,main_256,wrist_256,both_128",
                        help="comma list of <cameras>_<size>[_x<steps multiplier>], e.g. main_rgbd_256")
    parser.add_argument("--held-only", action="store_true", help="only frames with the peg lifted > 3 cm")
    args = parser.parse_args()
    global HELD_ONLY
    HELD_ONLY = args.held_only

    files = sorted(row["file"] for row in json.loads((EPISODES / "index.json").read_text())["rows"])
    train = load_frames(files[:-90])
    validation = load_frames(files[-90:])
    print(f"frames near hole: train {len(train[2])}, validation {len(validation[2])}", flush=True)

    results = []
    for name in args.variants.split(","):
        # e.g. "main_rgbd_256" or "both_256_x3" (3x the training steps)
        parts = name.split("_")
        multiplier = int(parts.pop()[1:]) if parts[-1].startswith("x") else 1
        size = int(parts.pop())
        cameras = "_".join(parts)
        results.append(run_probe(name, cameras, size, train, validation, args.steps * multiplier))
    args.output.write_text(json.dumps(dict(near_hole_depth_m=NEAR_HOLE, results=results), indent=2))


if __name__ == "__main__":
    main()
