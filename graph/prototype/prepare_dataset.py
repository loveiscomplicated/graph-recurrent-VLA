"""Restore replay snapshots and extract Geometry_t paired with Action_t."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from .common import config, DEFAULT_CONFIG, seed_all, versions, write_json
from .data import episode_split, state_at, validate_trajectory, reset_episode, phase_label, PHASES, sha256
from .geometry import concatenate
from .maniskill_gt import GTAdapter, validate_live_collision
from .visualization import plot_geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj-path", type=Path, required=True)
    p.add_argument("--initial-state-source", type=Path, required=True, help="Original pre-conversion H5, repairs ManiSkill 3.0.1 first-state bug")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=Path("outputs/graph/dataset.pt"))
    p.add_argument("--count", type=int, help="limit successful replay episodes")
    p.add_argument("--validate-replays", type=int, default=3)
    args = p.parse_args()
    cfg = config(args.config)
    seed_all(cfg["seed"], cfg["threads"])
    meta = json.loads(args.traj_path.with_suffix(".json").read_text())
    if meta["env_info"]["env_id"] != "PegInsertionSide-v1":
        raise ValueError("wrong task")
    episodes = meta["episodes"][:args.count]
    if any(e["control_mode"] != cfg["control_mode"] or not e.get("success", False) for e in episodes):
        raise ValueError("dataset requires successful replays in pd_ee_delta_pose")
    splits = episode_split([e["episode_id"] for e in episodes], cfg["seed"],
                           [e["episode_seed"] for e in episodes])
    source = h5py.File(args.initial_state_source, "r") if args.initial_state_source else None
    source_map = {}
    if source:
        source_meta = json.loads(args.initial_state_source.with_suffix(".json").read_text())
        for ep in source_meta["episodes"]:
            source_map.setdefault(ep["episode_seed"], []).append(ep["episode_id"])
    from envs.maniskill import make_peg_insertion_env
    env = make_peg_insertion_env(control_mode=cfg["control_mode"])
    adapter = GTAdapter(cfg["peg_points"], cfg["hole_points"], cfg["sampling_seed"])
    states, actions_all, ids, steps, phases = [], [], [], [], []
    checks, replay_checks = [], []
    start = perf_counter()
    try:
        with h5py.File(args.traj_path, "r") as trajectories:
            for epi, episode in enumerate(episodes):
                eid = episode["episode_id"]
                group = trajectories[f"traj_{eid}"]
                actions, observations = validate_trajectory(group)
                reset_episode(env, episode)
                collision = validate_live_collision(env, adapter(env))
                initial = state_at(group["env_states"], 0)
                original_initial = initial
                source_id = None
                if source:
                    candidates = []
                    for candidate_id in source_map[episode["episode_seed"]]:
                        candidate = state_at(source[f"traj_{candidate_id}/env_states"], 0)
                        env.unwrapped.set_state_dict(candidate)
                        error = float(np.max(np.abs(env.unwrapped.get_obs().cpu().numpy()[0]-observations[0])))
                        candidates.append((error, candidate_id, candidate))
                    error, source_id, original_initial = min(candidates, key=lambda c: c[0])
                    if error > 2e-5:
                        raise ValueError(f"no original initial state matches episode {eid}: {error}")
                initial_drift = 0.
                env.unwrapped.set_state_dict(initial)
                initial_drift = float(np.max(np.abs(env.unwrapped.get_obs().cpu().numpy()[0]-observations[0])))
                # Source t=0 always wins when explicitly supplied. Converted
                # replay 3.0.1 writes source t=1 into env_states[0].
                max_error = 0.
                saved_phases = set()
                for t in range(len(actions)+1):
                    snapshot = original_initial if t == 0 else state_at(group["env_states"], t)
                    env.unwrapped.set_state_dict(snapshot)
                    live = env.unwrapped.get_obs().cpu().numpy()[0]
                    error = float(np.max(np.abs(live-observations[t])))
                    max_error = max(max_error, error)
                    if error > 2e-5:
                        raise ValueError(f"episode {eid} t={t}: restored obs mismatch {error}; check reset geometry and --initial-state-source")
                    if t == len(actions):
                        continue
                    g = adapter(env)
                    phase = phase_label(env, g)
                    states.append(g)
                    ids.append(eid)
                    steps.append(t)
                    phases.append(phase)
                    if epi == 0 and phase not in saved_phases:
                        from .graph_builder import GraphBuilder
                        graph = GraphBuilder(cfg["surface_k"], cfg["interaction_k"])(g)
                        out = args.output.parent/"demonstration_graphs"/f"{PHASES[phase]}_t{t}.png"
                        plot_geometry(g, out, graph, title=f"Expert {PHASES[phase]}, episode {eid}, t={t}")
                        torch.save(g.as_dict(), out.with_suffix(".pt"))
                        saved_phases.add(phase)
                actions_all.append(torch.from_numpy(actions))
                checks.append({"episode_id": eid, "seed": episode["episode_seed"], "timesteps": len(actions),
                    "max_observation_error": max_error, "stored_initial_obs_error": initial_drift,
                    "initial_source_used": bool(source), "source_episode_id": source_id, "collision": collision,
                    "peg_half_size": states[-1].peg_half_size[0].tolist(),
                    "hole_offset": states[-1].hole_offset[0].tolist()})
                if epi < args.validate_replays:
                    reset_episode(env, episode)
                    env.unwrapped.set_state_dict(original_initial)
                    max_replay_error, ever_success = 0., False
                    for t, action in enumerate(actions):
                        obs, _, _, _, info = env.step(torch.from_numpy(action)[None])
                        max_replay_error = max(max_replay_error, float(np.max(np.abs(obs.cpu().numpy()[0]-observations[t+1]))))
                        ever_success |= bool(info["success"].item())
                    replay_checks.append({"episode_id": eid, "success": ever_success,
                                          "max_obs_error": max_replay_error, "steps": len(actions)})
                    if not ever_success or max_replay_error > .001:
                        raise ValueError(f"independent replay failed: {replay_checks[-1]}")
                print(f"extracted {epi+1}/{len(episodes)}: episode={eid}, T={len(actions)}, observation max error={max_error:.3g}", flush=True)
        geometry = concatenate(states)
        manifest = {"schema_version": 1, "source": str(args.traj_path.resolve()),
                    "source_sha256": sha256(args.traj_path), "versions": versions(), "config": cfg,
                    "control_mode": cfg["control_mode"], "splits": splits,
                    "episodes": episodes, "snapshot_checks": checks,
                    "replay_checks": replay_checks, "extraction_seconds": perf_counter()-start,
                    "initial_state_source": str(args.initial_state_source.resolve()) if source else None,
                    "initial_state_source_sha256": sha256(args.initial_state_source) if source else None,
                    "samples": len(ids), "phase_counts": {name: phases.count(i) for i, name in enumerate(PHASES)},
                    "phase_definition": "geometry-only proximity/opening/height/head-in-hole heuristic; contact caches are not restored",
                    "normalization": "train-only feature statistics; normalized expert actions unchanged",
                    "alignment": "sample t uses restored state[t] and actions[t]; final state[T] excluded"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"geometry": geometry.as_dict(), "actions": torch.cat(actions_all),
                    "episode_id": torch.tensor(ids), "timestep": torch.tensor(steps),
                    "phase": torch.tensor(phases), "manifest": manifest}, args.output)
        write_json(args.output.with_suffix(".json"), manifest)
        print(f"saved {args.output}: {len(ids)} samples; splits { {k: len(v) for k, v in splits.items()} }")
    finally:
        env.close()
        if source:
            source.close()


if __name__ == "__main__":
    main()
