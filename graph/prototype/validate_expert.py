"""Positive control: paired expert actions from full and alignment snapshots."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import seed_all, write_json
from .data import load_dataset, reset_episode, state_at
from .maniskill_gt import GTAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("outputs/graph/dataset.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/graph/expert_control.json"))
    p.add_argument("--horizon", type=int, default=250)
    args = p.parse_args()
    data, _ = load_dataset(args.dataset)
    manifest = data["manifest"]
    cfg = manifest["config"]
    seed_all(cfg["seed"], cfg["threads"])
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    env = make_peg_insertion_env(control_mode=cfg["control_mode"])
    env = TimeLimitWrapper(env.unwrapped, args.horizon)
    episodes = [e for e in manifest["episodes"] if e["episode_id"] in manifest["splits"]["test"]]
    checks = {c["episode_id"]: c for c in manifest["snapshot_checks"]}
    report = {"horizon": args.horizon, "full": [], "alignment": []}
    try:
        with h5py.File(manifest["source"], "r") as trajectories, h5py.File(manifest["initial_state_source"], "r") as original:
            for setting in ("full", "alignment"):
                for ep in episodes:
                    eid, t = ep["episode_id"], 0
                    if setting == "alignment":
                        rows = torch.where((data["episode_id"] == eid) & (data["phase"] == 3))[0]
                        if not len(rows):
                            rows = torch.where((data["episode_id"] == eid) & (data["phase"] == 4))[0]
                        if not len(rows):
                            continue
                        t = int(data["timestep"][rows[0]])
                    env = TimeLimitWrapper(env.unwrapped, args.horizon+t)
                    reset_episode(env, ep)
                    snapshot = state_at(original[f"traj_{checks[eid]['source_episode_id']}/env_states"], 0)
                    env.unwrapped.set_state_dict(snapshot)
                    env.unwrapped.agent.controller.reset()
                    prefix_error = 0.
                    for prefix_t, past_action in enumerate(trajectories[f"traj_{eid}/actions"][:t]):
                        obs, _, _, _, _ = env.step(torch.from_numpy(past_action)[None])
                        expected = trajectories[f"traj_{eid}/obs"][prefix_t+1]
                        prefix_error = max(prefix_error, float(np.max(np.abs(obs.cpu().numpy()[0]-expected))))
                    actions = trajectories[f"traj_{eid}/actions"][t:t+args.horizon]
                    saved_obs = trajectories[f"traj_{eid}/obs"][t+1:t+1+len(actions)]
                    success, error, steps = False, 0., 0
                    for action, target_obs in zip(actions, saved_obs):
                        obs, _, _, _, info = env.step(torch.from_numpy(action)[None])
                        steps += 1
                        error = max(error, float(np.max(np.abs(obs.cpu().numpy()[0]-target_obs))))
                        success |= bool(info["success"].item())
                        if success:
                            break
                    record = {"episode_id": eid, "seed": ep["episode_seed"], "start_timestep": t,
                              "success": success, "steps": steps, "max_observation_error": error,
                              "prefix_observation_max_error": prefix_error}
                    report[setting].append(record)
                    print(setting, record, flush=True)
        for setting in ("full", "alignment"):
            records = report[setting]
            report[setting+"_success_rate"] = sum(r["success"] for r in records)/len(records) if records else None
        write_json(args.output, report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
