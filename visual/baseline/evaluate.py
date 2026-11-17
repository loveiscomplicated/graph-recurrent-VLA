"""Closed-loop evaluation of a baseline checkpoint on the fixed evaluation seeds.

Success means ``info['success']`` became true within the 250-step horizon, the
same criterion as the expert reference in eval_seeds.json.

Example:
    python -m visual.baseline.evaluate \
        --checkpoint outputs/visual_baseline/act_main/checkpoint_0100000.pt \
        --seeds outputs/visual_baseline/eval_seeds.json \
        --output outputs/visual_baseline/act_main/eval_0100000.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml

from envs.maniskill import get_gt_state
from visual.baseline.common import CAMERAS, HORIZON, live_robot15, make_env, resize_depth, resize_rgb
from visual.baseline.policy import CAMERA_NAMES, DEPTH_KEY, Normalizer, build_policy, image_key
from visual.baseline.prepare_data import DEPTH_SIZE


def load_policy(checkpoint_path: str, overrides: dict | None = None):
    """Rebuild the policy and its normalizer from a training checkpoint.

    ``overrides`` replaces inference-only policy settings, e.g.
    ``{"temporal_ensemble_coeff": None, "n_action_steps": 10}`` for ACT.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    config["policy"].update(overrides or {})
    image_size = json.loads((Path(config["data"]["episodes"]) / "index.json").read_text())["rgb_size"]
    policy = build_policy(config["policy"], image_size, config["data"].get("use_state", True),
                          config["data"].get("use_depth", False)).cuda()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, Normalizer(checkpoint["stats"]), image_size


def observation_batch(env, observation, normalizer, image_size, use_depth=False) -> dict[str, torch.Tensor]:
    """Convert one simulator observation into the normalized single-step policy input."""
    batch = {}
    for name, camera in zip(CAMERA_NAMES, CAMERAS):
        rgb = observation["sensor_data"][camera]["rgb"].cpu()
        batch[image_key(name)] = normalizer.image(resize_rgb(rgb, image_size).cuda())
    if use_depth:
        # Same path as training: 512 → 128 px nearest (prepare_data), then → image_size.
        depth = resize_depth(observation["sensor_data"]["base_camera"]["depth"].cpu(), DEPTH_SIZE)
        batch[DEPTH_KEY] = normalizer.depth(depth.cuda(), image_size)
    state = torch.from_numpy(live_robot15(env)).cuda()[None]
    batch["observation.state"] = normalizer.state(state)
    return batch


# The expert starts inserting with the peg head ~10–12 cm in front of the hole along
# the hole axis (peg_head_pos_at_hole x ≈ -0.10…-0.12) once lateral error < 1.5 mm.
PRE_INSERT_DEPTH = -0.13
ALIGNED_LATERAL = 0.005
STAGES = ("success", "aligned", "misaligned", "lifted", "no_lift")


def failure_stage(success: bool, lift: float, max_depth: float, min_lateral: float) -> str:
    """Furthest task stage reached.

    no_lift (peg < 2 cm up) → lifted (never at pre-insert depth) → misaligned (at
    depth, lateral error ≥ 5 mm) → aligned (at depth, lateral < 5 mm) → success.
    """
    if success:
        return "success"
    if lift < 0.02:
        return "no_lift"
    if max_depth < PRE_INSERT_DEPTH:
        return "lifted"
    return "aligned" if min_lateral < ALIGNED_LATERAL else "misaligned"


def video_frame(observation) -> np.ndarray:
    """Main and wrist RGB side by side at 256 px, uint8 HWC."""
    images = [resize_rgb(observation["sensor_data"][camera]["rgb"].cpu(), 256)[0] for camera in CAMERAS]
    return torch.cat(images, dim=-1).permute(1, 2, 0).numpy()


def evaluate_seeds(task: tuple) -> list[dict]:
    """Roll out the policy once per seed in a worker process.

    ``task`` is (checkpoint path, seeds, seeds to record as video, video directory).
    """
    checkpoint_path, seeds, video_seeds, video_dir, overrides, binary_gripper = task
    # nproc reports 256 host cores but the pod gets 36; default threading oversubscribes badly.
    torch.set_num_threads(2)
    policy, normalizer, image_size = load_policy(checkpoint_path, overrides)
    env = make_env()
    low = torch.as_tensor(env.action_space.low, device="cuda")
    high = torch.as_tensor(env.action_space.high, device="cuda")
    results = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            policy.reset()
            success_step = None
            step_times = []
            start_height = float(get_gt_state(env).peg_pose[0, 2])
            lift, max_depth, min_lateral = 0.0, -1.0, float("inf")
            frames = [video_frame(observation)] if seed in video_seeds else None
            for step in range(HORIZON):
                # Time observation → action (preprocessing + policy), excluding the simulator step.
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    action = policy.select_action(observation_batch(
                        env, observation, normalizer, image_size, DEPTH_KEY in policy.config.input_features))
                action = normalizer.unnormalize_action(action.float()).clamp(low, high)
                if binary_gripper:
                    # The expert only commands ±1; regression averages it toward 0 (weak grip).
                    action[..., -1] = torch.where(action[..., -1] >= 0, 1.0, -1.0)
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - started)
                observation, _, terminated, truncated, info = env.step(action.cpu())
                lift = max(lift, float(get_gt_state(env).peg_pose[0, 2]) - start_height)
                head = info["peg_head_pos_at_hole"].reshape(-1)
                max_depth = max(max_depth, float(head[0]))
                if head[0] >= PRE_INSERT_DEPTH:
                    min_lateral = min(min_lateral, float(head[1:].norm()))
                if frames is not None:
                    frames.append(video_frame(observation))
                if bool(info["success"].any()):
                    success_step = step + 1
                    break
                if bool(terminated.any() or truncated.any()):
                    break
            success = success_step is not None
            stage = failure_stage(success, lift, max_depth, min_lateral)
            if frames is not None:
                imageio.mimsave(Path(video_dir) / f"seed_{seed}_{stage}.mp4", frames, fps=20)
            results.append(dict(seed=seed, success=success, steps=success_step, stage=stage,
                                max_lift_m=round(lift, 4), max_head_depth_m=round(max_depth, 4),
                                min_lateral_at_depth_m=round(min_lateral, 4) if min_lateral < 1 else None,
                                step_ms=[round(t * 1000, 2) for t in step_times]))
            print(f"seed {seed} {stage}", flush=True)
    finally:
        env.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--videos", type=int, default=0, help="record the first N seeds as mp4")
    parser.add_argument("--binary-gripper", action="store_true", help="execute sign(gripper) instead of the raw value")
    parser.add_argument("--policy-override", action="append", default=[], metavar="KEY=YAML",
                        help="inference setting, e.g. temporal_ensemble_coeff=null (repeatable)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new output path")

    reference = json.loads(args.seeds.read_text())
    seeds = reference["seeds"]
    video_seeds = set(seeds[: args.videos])
    video_dir = args.output.with_suffix("")  # e.g. eval_0100000.json → eval_0100000/
    if video_seeds:
        video_dir.mkdir(parents=True, exist_ok=True)
    parts = [seeds[i::args.workers] for i in range(args.workers)]
    overrides = {}
    for item in args.policy_override:
        if "=" not in item:
            raise SystemExit(f"--policy-override expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    tasks = [(args.checkpoint, part, video_seeds, str(video_dir), overrides, args.binary_gripper)
             for part in parts]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [row for part in pool.map(evaluate_seeds, tasks) for row in part]
    rows.sort(key=lambda row: row["seed"])

    successes = sum(row["success"] for row in rows)
    stages = {name: sum(row["stage"] == name for row in rows) for name in STAGES}
    # Workers share one GPU, so these latencies are upper bounds of single-process inference.
    step_ms = np.array([t for row in rows for t in row["step_ms"]])
    latency = dict(mean=step_ms.mean(), p50=np.percentile(step_ms, 50),
                   p95=np.percentile(step_ms, 95), max=step_ms.max(), workers=args.workers)
    record = dict(checkpoint=args.checkpoint, seeds_file=str(args.seeds), episodes=len(rows),
                  policy_overrides=overrides, binary_gripper=args.binary_gripper,
                  success=successes, success_rate=successes / len(rows), stages=stages,
                  step_latency_ms={k: round(float(v), 2) for k, v in latency.items()},
                  expert_success_rate=reference["expert_success_rate"], results=rows)
    args.output.write_text(json.dumps(record, indent=2))
    print(f"success={successes}/{len(rows)} stages={stages} (expert {reference['expert_success_rate']:.2f}) "
          f"step latency p50 {latency['p50']:.1f} ms p95 {latency['p95']:.1f} ms")


if __name__ == "__main__":
    main()
