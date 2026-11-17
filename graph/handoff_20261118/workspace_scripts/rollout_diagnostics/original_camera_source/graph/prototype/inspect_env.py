"""Inspect the installed simulator and validate randomized live collision geometry."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .common import config, DEFAULT_CONFIG, versions, write_json, seed_all
from .maniskill_gt import GTAdapter, validate_live_collision
from .visualization import plot_geometry


def structure(x):
    if isinstance(x, dict):
        return {k: structure(v) for k, v in x.items()}
    return {"shape": list(x.shape), "dtype": str(x.dtype)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=Path("outputs/graph/inspection"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    cfg = config(args.config)
    seed_all(cfg["seed"], cfg["threads"])
    from envs.maniskill import make_peg_insertion_env, get_gt_state
    env = make_peg_insertion_env(control_mode=cfg["control_mode"], obs_mode="state_dict")
    adapter = GTAdapter(cfg["peg_points"], cfg["hole_points"], cfg["sampling_seed"])
    report = {"versions": versions(), "episodes": []}
    try:
        task = env.unwrapped
        report.update(env_id=env.spec.id, robot=task.agent.uid,
                      backend=str(task.backend), control_mode=task.control_mode,
                      controller=repr(task.agent.controller.configs),
                      action_space=str(env.action_space), max_episode_steps=env.spec.max_episode_steps)
        for seed in args.seeds:
            obs, _ = env.reset(seed=seed, options={"reconfigure": True})
            g = adapter(env)
            state = get_gt_state(env)
            episode = {"seed": seed, "observation": structure(obs),
                       "raw_gt": {k: v.tolist() for k, v in asdict(state).items()},
                       "geometry_shapes": {k: list(v.shape) for k, v in g.as_dict().items()},
                       "collision_check": validate_live_collision(env, g)}
            report["episodes"].append(episode)
            plot_geometry(g, args.output / f"surface_seed{seed}.png", title=f"seed {seed}; half-size {g.peg_half_size[0].tolist()}")
            torch.save(g.as_dict(), args.output / f"geometry_seed{seed}.pt")
            env.step(torch.zeros(1, 7))
        write_json(args.output / "inspection.json", report)
        print(report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
