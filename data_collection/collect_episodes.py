"""Collect raw episodes whose first Main frame passes peg visibility.

Run from the repository root: python -m data_collection.collect_episodes --help.
With zero steps, save approved initial observations. For trajectories, supply a
policy factory via --policy module:factory, or --zero-actions for smoke checks.
"""

import argparse
import importlib
import importlib.metadata
import subprocess
from pathlib import Path

import torch

from envs.maniskill import get_gt_state, make_peg_insertion_env
from .storage import append_attempt, save_frame, to_numpy, write_json
from .visibility import CameraObservation, evaluate


def initial_visibility(env, observation):
    gt = get_gt_state(env)
    image = observation["sensor_data"]["base_camera"]
    calibration = observation["sensor_param"]["base_camera"]
    tcp = to_numpy(gt.ee_pose)[0]
    camera = CameraObservation(
        "initial",
        0,
        "base_camera",
        0.0,
        to_numpy(image["rgb"])[0],
        to_numpy(image["depth"])[0, ..., 0],
        to_numpy(calibration["intrinsic_cv"])[0],
        to_numpy(calibration["extrinsic_cv"])[0],
        to_numpy(calibration["cam2world_gl"])[0],
        tcp,
    )
    result, _ = evaluate(
        camera, to_numpy(gt.peg_pose)[0], to_numpy(gt.peg_half_sizes)[0], tcp
    )
    return result, vars(gt)


def collect_episodes(
    env,
    output,
    *,
    num_episodes,
    max_attempts,
    start_seed=0,
    steps=0,
    policy=None,
    visibility_mode="ends_and_grasp",
    metadata=None,
):
    """A policy has reset(env, seed) and act(observation) -> native action.

    Reset runs only after an accepted first observation. The same environment
    continues from that observation; no second reset precedes the first action.
    Only committed episode.json receipts appear in manifest.json.
    """
    if num_episodes < 1 or max_attempts < 1 or steps < 0 or start_seed < 0:
        raise ValueError(
            "positive collection counts and nonnegative steps/seed required"
        )
    if steps and policy is None:
        raise ValueError("trajectory collection requires an explicit policy")
    keys = {"ends_and_grasp": "requested_pass", "full_surface": "strict_100_pass"}
    if visibility_mode not in keys:
        raise ValueError("unknown visibility mode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        num_episodes=num_episodes,
        max_attempts=max_attempts,
        start_seed=start_seed,
        steps=steps,
        visibility_mode=visibility_mode,
        metadata=metadata or {},
    )
    write_json(output / "config.json", config)
    episodes = []
    manifest = dict(status="RUNNING", attempted=0, episodes=episodes)
    write_json(output / "manifest.json", manifest)
    try:
        for seed in range(start_seed, start_seed + max_attempts):
            observation, info = env.reset(seed=seed)
            visibility, gt = initial_visibility(env, observation)
            manifest["attempted"] += 1
            passed = bool(visibility[keys[visibility_mode]])
            append_attempt(
                output, dict(seed=seed, accepted=passed, visibility=visibility)
            )
            if not passed:
                continue
            name = f"episode_{len(episodes):06d}"
            pending = output / (name + ".partial")
            pending.mkdir()
            write_json(pending / "visibility.json", visibility)
            save_frame(
                pending / "frame_000000.npz", observation, info=info, ground_truth=gt
            )
            length = 0
            terminated = truncated = False
            if steps:
                policy.reset(env, seed)
            for step in range(steps):
                action = torch.as_tensor(
                    policy.act(observation), device=env.unwrapped.device
                )
                if (
                    tuple(action.shape) != tuple(env.action_space.shape)
                    or not torch.isfinite(action).all()
                ):
                    raise ValueError(
                        "policy must return finite batched native actions matching action_space"
                    )
                # Snapshot the applied command before stepping or requesting another.
                command = to_numpy(action)
                observation, reward, term, trunc, info = env.step(action)
                length = step + 1
                save_frame(
                    pending / f"frame_{length:06d}.npz",
                    observation,
                    action=command,
                    reward=reward,
                    terminated=term,
                    truncated=trunc,
                    info=info,
                )
                terminated, truncated = bool(term.item()), bool(trunc.item())
                if terminated or truncated:
                    break
            receipt = dict(
                seed=seed,
                path=name,
                steps=length,
                observations=length + 1,
                terminated=terminated,
                truncated=truncated,
                visibility_mode=visibility_mode,
            )
            write_json(pending / "episode.json", receipt)
            pending.rename(output / name)
            episodes.append(receipt)
            write_json(output / "manifest.json", manifest)
            print(f"Accepted {len(episodes)}/{num_episodes}: seed={seed}", flush=True)
            if len(episodes) == num_episodes:
                break
        manifest["status"] = (
            "COMPLETE" if len(episodes) == num_episodes else "EXHAUSTED"
        )
    except BaseException:
        manifest["status"] = "ERROR"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest


class ZeroActions:
    """Explicit simulator smoke behavior, not an expert demonstration."""

    def reset(self, env, seed):
        self.action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    def act(self, observation):
        return self.action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument(
        "--policy", help="Python module:factory; factory() returns reset/act policy"
    )
    behavior.add_argument(
        "--zero-actions", action="store_true", help="explicit smoke trajectory"
    )
    parser.add_argument(
        "--visibility-mode",
        choices=["ends_and_grasp", "full_surface"],
        default="ends_and_grasp",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="physics device"
    )
    parser.add_argument("--render-backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--observation-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--control-mode", default=None)
    args = parser.parse_args()
    if (
        min(args.num_episodes, args.max_attempts, args.resolution) < 1
        or min(args.steps, args.start_seed) < 0
    ):
        parser.error("counts/resolution must be positive; steps/seed nonnegative")
    if args.steps and not (args.policy or args.zero_actions):
        parser.error("--steps requires --policy or --zero-actions")
    policy = ZeroActions() if args.zero_actions else None
    if args.policy:
        module, attribute = args.policy.split(":", 1)
        policy = getattr(importlib.import_module(module), attribute)()
    env = make_peg_insertion_env(
        device=args.device,
        render_backend=args.render_backend,
        observation_device=args.observation_device,
        obs_mode="rgb+depth",
        control_mode=args.control_mode,
        sensor_configs={
            name: dict(width=args.resolution, height=args.resolution)
            for name in ("base_camera", "hand_camera")
        },
    )
    try:
        metadata = {
            **vars(args),
            "output": str(args.output),
            "sim_backend_actual": env.unwrapped.backend.sim_backend,
            "render_backend_actual": env.unwrapped.backend.render_backend,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "mani-skill",
                    "sapien",
                    "gymnasium",
                    "torch",
                    "numpy",
                    "scipy",
                    "trimesh",
                )
            },
        }
        source = Path(__file__).resolve().parents[1]
        metadata["source_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        metadata["source_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            ).strip()
        )
        result = collect_episodes(
            env,
            args.output,
            num_episodes=args.num_episodes,
            max_attempts=args.max_attempts,
            start_seed=args.start_seed,
            steps=args.steps,
            policy=policy,
            visibility_mode=args.visibility_mode,
            metadata=metadata,
        )
        if result["status"] != "COMPLETE":
            raise SystemExit(
                "Attempt budget exhausted before collecting requested episodes"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
