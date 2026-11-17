"""Train an ACT or Diffusion baseline on the prepared expert episodes.

Run through the shared entrypoint:
    python main.py --config visual/configs/baseline_act.yaml

All episodes are held in RAM as uint8 tensors (~36 GB at 256 px for both
cameras); random batches are gathered on the CPU and normalized on the GPU.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.experiment_logger import ExperimentLogger
from visual.baseline.policy import CAMERA_NAMES, DEPTH_KEY, Normalizer, build_policy, compute_stats, image_key


class EpisodeData:
    """All prepared episodes concatenated along time, with episode boundaries."""

    def __init__(self, directory: Path, files: list[str], steps_by_file: dict[str, int],
                 use_depth: bool = False):
        lengths = torch.tensor([steps_by_file[name] for name in files])
        self.ends = lengths.cumsum(0)
        self.starts = self.ends - lengths
        # episode_of[t] gives the episode containing global step t.
        self.episode_of = torch.repeat_interleave(torch.arange(len(files)), lengths)

        # Preallocate and fill in place so peak RAM stays ~1x the dataset (the
        # container limit is 125 GB; concatenating a list doubled it).
        total = int(lengths.sum())
        with np.load(directory / files[0]) as first:
            image_shape = first["main_rgb"].shape[1:]
            depth_shape = first["main_depth"].shape[1:]
        self.rgb = {camera: torch.empty((total, *image_shape), dtype=torch.uint8) for camera in CAMERA_NAMES}
        self.depth = torch.empty((total, *depth_shape), dtype=torch.int16) if use_depth else None
        self.robot15 = torch.empty((total, 15), dtype=torch.float32)
        self.actions = torch.empty((total, 7), dtype=torch.float32)

        def load(position):
            name, start, end = files[position], int(self.starts[position]), int(self.ends[position])
            with np.load(directory / name) as episode:
                if len(episode["actions"]) != end - start:
                    raise ValueError(f"{name}: index.json steps disagree with the file")
                for camera in CAMERA_NAMES:
                    self.rgb[camera][start:end] = torch.from_numpy(episode[f"{camera}_rgb"])
                if self.depth is not None:
                    self.depth[start:end] = torch.from_numpy(episode["main_depth"])
                self.robot15[start:end] = torch.from_numpy(episode["robot15"])
                self.actions[start:end] = torch.from_numpy(episode["actions"])

        with ThreadPoolExecutor(16) as pool:
            list(pool.map(load, range(len(files))))

    def __len__(self) -> int:
        return len(self.actions)

    def sample(self, steps: torch.Tensor, n_obs: int, horizon: int) -> dict[str, torch.Tensor]:
        """Gather observation history and action window around global steps.

        Observation k uses step ``t - (n_obs-1) + k`` clamped to the episode start
        (repeat padding). Actions cover ``t - (n_obs-1)`` .. ``+horizon``; entries
        outside the episode are clamped and flagged in ``action_is_pad``.
        """
        episode = self.episode_of[steps]
        start, end = self.starts[episode, None], self.ends[episode, None]
        first = steps[:, None] - (n_obs - 1)

        obs_index = torch.maximum(first + torch.arange(n_obs), start)
        action_index = first + torch.arange(horizon)
        action_is_pad = (action_index < start) | (action_index >= end)
        action_index = action_index.clamp(start, end - 1)

        batch = {image_key(c): self.rgb[c][obs_index] for c in CAMERA_NAMES}
        if self.depth is not None:
            batch[DEPTH_KEY] = self.depth[obs_index]
        batch["observation.state"] = self.robot15[obs_index]
        batch["action"] = self.actions[action_index]
        batch["action_is_pad"] = action_is_pad
        return batch


def random_shift(images: torch.Tensor, pixels: int) -> torch.Tensor:
    """Translate each image by up to ``pixels`` (border-replicated), DrQ-style."""
    flat = images.flatten(0, -4)
    size = flat.shape[-1]
    shift = (torch.rand(len(flat), 2, device=flat.device) * 2 - 1) * (2 * pixels / size)
    theta = torch.zeros(len(flat), 2, 3, device=flat.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    shifted = F.grid_sample(flat, grid, padding_mode="border", align_corners=False)
    return shifted.view_as(images)


def to_policy_batch(raw, normalizer, n_obs, shift_pixels, training, state_noise=0.0) -> dict[str, torch.Tensor]:
    """Move a raw batch to the GPU, normalize it, and drop the obs time axis for ACT.

    ``state_noise`` adds Gaussian noise (in normalized units) to robot15 during
    training so the policy cannot rely on proprioception alone.
    """
    batch = {key: value.cuda(non_blocking=True) for key, value in raw.items()}
    for camera in CAMERA_NAMES:
        images = normalizer.image(batch[image_key(camera)])
        if training and shift_pixels:
            images = random_shift(images, shift_pixels)
        batch[image_key(camera)] = images
    if DEPTH_KEY in batch:
        # No random shift on depth: absolute pixel position carries the hole location.
        batch[DEPTH_KEY] = normalizer.depth(batch[DEPTH_KEY], batch[image_key("main")].shape[-1])
    batch["observation.state"] = normalizer.state(batch["observation.state"])
    if training and state_noise:
        batch["observation.state"] += torch.randn_like(batch["observation.state"]) * state_noise
    batch["action"] = normalizer.action(batch["action"])
    if n_obs == 1:
        for key in [image_key(c) for c in CAMERA_NAMES] + [DEPTH_KEY, "observation.state"]:
            if key in batch:
                batch[key] = batch[key][:, 0]
    return batch


def split_episodes(index: dict, validation_episodes: int) -> tuple[list[str], list[str]]:
    """Hold out the last episodes (in sorted file order) for validation loss."""
    files = sorted(row["file"] for row in index["rows"])
    return files[:-validation_episodes], files[-validation_episodes:]


def windows(policy_config: dict) -> tuple[int, int]:
    """(observation steps, action window length) used by the policy type."""
    if policy_config["type"] == "act":
        return 1, policy_config["chunk_size"]
    return policy_config["n_obs_steps"], policy_config["horizon"]


def learning_rate(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay, as a multiplier of the base rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validation_loss(policy, data, normalizer, n_obs, horizon, batch_size) -> float:
    """Mean training loss on a fixed set of held-out steps."""
    policy.train()  # ACT needs train mode for its VAE loss; no optimizer step here.
    generator = torch.Generator().manual_seed(0)
    steps = torch.randint(len(data), (batch_size * 8,), generator=generator)
    losses = []
    for chunk in steps.split(batch_size):
        batch = to_policy_batch(data.sample(chunk, n_obs, horizon), normalizer, n_obs, 0, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        losses.append(loss.item())
    return float(np.mean(losses))


def save_checkpoint(path: Path, policy, stats, config, step) -> None:
    torch.save(dict(model=policy.state_dict(), stats=stats, config=config, step=step), path)


def train(config: dict) -> None:
    """Entry point for ``main.py``; see visual/configs/baseline_*.yaml."""
    data_config, policy_config, train_config = config["data"], config["policy"], config["train"]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(train_config["seed"])

    directory = Path(data_config["episodes"])
    index = json.loads((directory / "index.json").read_text())
    train_files, validation_files = split_episodes(index, data_config["validation_episodes"])
    started = time.time()
    steps_by_file = {row["file"]: row["steps"] for row in index["rows"]}
    use_depth = data_config.get("use_depth", False)
    train_data = EpisodeData(directory, train_files, steps_by_file, use_depth)
    validation_data = EpisodeData(directory, validation_files, steps_by_file, use_depth)
    print(f"loaded {len(train_data)} train / {len(validation_data)} val steps in {time.time() - started:.0f}s", flush=True)

    stats = compute_stats(train_data.robot15, train_data.actions, train_config["action_normalization"])
    normalizer = Normalizer(stats)
    policy = build_policy(policy_config, index["rgb_size"], data_config.get("use_state", True), use_depth).cuda()
    n_obs, horizon = windows(policy_config)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=train_config["lr"], weight_decay=train_config["weight_decay"]
    )
    base_rates = [group["lr"] for group in optimizer.param_groups]
    total_steps, batch_size = train_config["steps"], train_config["batch_size"]

    with ExperimentLogger(team="visual", config=config, seed=train_config["seed"],
                          model=f"baseline_{policy_config['type']}") as logger:
        logger.record(output_dir=str(output), train_episodes=len(train_files),
                      validation_episodes=len(validation_files))
        policy.train()
        started = time.time()
        for step in range(total_steps):
            scale = learning_rate(step, total_steps, train_config["warmup_steps"])
            for group, base in zip(optimizer.param_groups, base_rates):
                group["lr"] = base * scale

            steps = torch.randint(len(train_data), (batch_size,))
            batch = to_policy_batch(train_data.sample(steps, n_obs, horizon), normalizer,
                                    n_obs, train_config["random_shift"], True,
                                    train_config.get("state_noise", 0.0))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = policy.forward(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), train_config["grad_clip"])
            optimizer.step()

            done = step + 1
            if done % train_config["log_every"] == 0:
                rate = done / (time.time() - started)
                logger.log(dict(loss=loss.item(), lr_scale=scale, steps_per_s=rate, **(parts or {})), step=done)
                print(f"step {done} loss {loss.item():.4f} {rate:.1f} it/s", flush=True)
            if done % train_config["eval_every"] == 0 or done == total_steps:
                value = validation_loss(policy, validation_data, normalizer, n_obs, horizon, batch_size)
                logger.log(dict(validation_loss=value), step=done)
                print(f"step {done} validation_loss {value:.4f}", flush=True)
                policy.train()
            if done % train_config["save_every"] == 0 or done == total_steps:
                path = output / f"checkpoint_{done:07d}.pt"
                save_checkpoint(path, policy, stats, config, done)
                logger.link_output(path, checkpoint=True)
