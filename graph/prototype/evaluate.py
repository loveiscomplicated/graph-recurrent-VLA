"""Held-out action prediction and paired simulator rollouts, separately reported."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from .common import seed_all, write_json, versions
from .data import load_dataset, split_indices, reset_episode, state_at, sha256, PHASES, phase_label
from .geometry import transform, inverse_transform, rotation
from .graph_builder import GraphBuilder
from .maniskill_gt import GTAdapter
from .metrics import action_metrics
from .models import make_policy
from .normalization import FeatureNormalizer


def diagnostic(env, g, contacts_valid=True):
    local_head = torch.zeros_like(g.peg_pose[:, None, :3])
    local_head[:, 0, 0] = g.peg_half_size[:, 0]
    head = inverse_transform(transform(local_head, g.peg_pose), g.hole_pose)[0, 0]
    peg_axis = rotation(g.peg_pose[:, 3:])[0, :, 0]
    hole_axis = rotation(g.hole_pose[:, 3:])[0, :, 0]
    angle = torch.acos((peg_axis*hole_axis).sum().clamp(-1, 1))
    depth = float(g.box_half_size[0, 0])
    return {"transverse_head_error_m": float(head[1:].norm()),
            "axis_alignment_error_rad": float(angle), "head_x_in_hole_frame_m": float(head[0]),
            "insertion_progress": float(((head[0]+depth)/(2*depth)).clamp(0, 1)),
            "grasped": bool(env.unwrapped.agent.is_grasping(env.unwrapped.peg).item()) if contacts_valid else None,
            "phase": PHASES[phase_label(env, g)]}


def rollout(model, name, cfg, norm, builder, data, geometry, args, warm=False):
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    env = make_peg_insertion_env(control_mode=cfg["control_mode"], render_mode="rgb_array")
    # Installed TimeLimitWrapper supports batched Torch flags; use the same
    # wrapper when an explicit evaluation horizon overrides registered 100.
    env = TimeLimitWrapper(env.unwrapped, args.horizon)
    adapter = GTAdapter(cfg["peg_points"], cfg["hole_points"], cfg["sampling_seed"])
    episodes = [e for e in data["manifest"]["episodes"] if e["episode_id"] in data["manifest"]["splits"]["test"]]
    if args.episode_ids:
        if not set(args.episode_ids).issubset({e["episode_id"] for e in episodes}):
            raise ValueError("rollout episode IDs must belong to held-out test split")
        episodes = [e for e in episodes if e["episode_id"] in args.episode_ids]
    if args.rollout_episodes:
        episodes = episodes[:args.rollout_episodes]
    checks = {e["episode_id"]: e for e in data["manifest"]["snapshot_checks"]}
    trajectory = h5py.File(data["manifest"]["source"], "r")
    original_path = data["manifest"].get("initial_state_source")
    original = h5py.File(original_path, "r") if original_path else None
    records = []
    try:
        for i, ep in enumerate(episodes):
            eid = ep["episode_id"]
            t = 0
            if warm:
                rows = torch.where((data["episode_id"] == eid) & (data["phase"] == 3))[0]
                if not len(rows):
                    rows = torch.where((data["episode_id"] == eid) & (data["phase"] == 4))[0]
                if not len(rows):
                    records.append({"episode_id": eid, "seed": ep["episode_seed"], "skipped": "no alignment or insertion snapshot"})
                    continue
                t = int(data["timestep"][rows[0]])
            env = TimeLimitWrapper(env.unwrapped, args.horizon+t)
            reset_episode(env, ep)
            snapshot = state_at(trajectory[f"traj_{eid}/env_states"], 0)
            if original:
                snapshot = state_at(original[f"traj_{checks[eid]['source_episode_id']}/env_states"], 0)
            env.unwrapped.set_state_dict(snapshot)
            env.unwrapped.agent.controller.reset()
            prefix_error = 0.
            if warm:
                # Past expert actions reconstruct hidden controller/contact
                # state. No expert action is supplied to the policy itself.
                for prefix_t, past_action in enumerate(trajectory[f"traj_{eid}/actions"][:t]):
                    obs, _, _, _, _ = env.step(torch.from_numpy(past_action)[None])
                    expected = trajectory[f"traj_{eid}/obs"][prefix_t+1]
                    prefix_error = max(prefix_error, float(np.max(np.abs(obs.cpu().numpy()[0]-expected))))
                if prefix_error > .001:
                    raise ValueError(f"expert prefix diverged: episode {eid}, max error {prefix_error}")
            initial = adapter(env)
            record = {"episode_id": eid, "seed": ep["episode_seed"], "start_timestep": t,
                      "expert_prefix_observation_max_error": prefix_error,
                      "initial_geometry_sha256": hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in initial.as_dict().values())).hexdigest(),
                      "initial": diagnostic(env, initial, contacts_valid=warm), "trace": []}
            writer = None
            if i < args.videos:
                import imageio.v2 as imageio
                path = args.output.parent/"videos"/f"{name}_{'alignment' if warm else 'full'}_episode{eid}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(path, fps=5)
                record["video"] = str(path)
            ever_success, ever_grasped, lost, previous_grasp = False, bool(record["initial"]["grasped"]), 0, bool(record["initial"]["grasped"])
            try:
                for step in range(args.horizon):
                    g = adapter(env)
                    prediction = model(norm.apply(name, g, builder))
                    action = prediction.clamp(torch.as_tensor(env.action_space.low), torch.as_tensor(env.action_space.high))
                    _, _, terminated, truncated, info = env.step(action)
                    d = diagnostic(env, adapter(env))
                    record["trace"].append({"step": step+1, **d})
                    ever_grasped |= d["grasped"]
                    lost += int(previous_grasp and not d["grasped"])
                    previous_grasp = d["grasped"]
                    ever_success |= bool(info["success"].item())
                    if writer and step % 4 == 0:
                        writer.append_data(env.render()[0].cpu().numpy())
                    if ever_success or bool(terminated.item()) or bool(truncated.item()):
                        break
            finally:
                if writer:
                    writer.close()
            final = record["trace"][-1]
            record.update(success=ever_success, episode_length=len(record["trace"]), final=final,
                          ever_grasped=ever_grasped, grasp_losses=lost,
                          failure_case=None if ever_success else ("grasp_lost" if lost else "no_grasp" if not ever_grasped else "alignment_or_insertion"))
            records.append(record)
            print(f"{name} {'alignment' if warm else 'full'} episode {eid}: success={ever_success}, steps={record['episode_length']}, final transverse={final['transverse_head_error_m']:.4f}m", flush=True)
    finally:
        env.close()
        trajectory.close()
        if original:
            original.close()
    complete = [r for r in records if "success" in r]
    return {"horizon": args.horizon, "registered_task_horizon": 100, "initialization": "expert prefix replay to held-out alignment state" if warm else "held-out original initial snapshot",
            "episodes": records, "success_rate": sum(r["success"] for r in complete)/len(complete) if complete else None,
            "success_rate_by_100_steps": sum(r["success"] and r["episode_length"] <= 100 for r in complete)/len(complete) if complete else None,
            "evaluated_episodes": len(complete), "skipped_episodes": len(records)-len(complete),
            "mean_episode_length": float(np.mean([r["episode_length"] for r in complete])) if complete else None,
            "mean_final_transverse_error_m": float(np.mean([r["final"]["transverse_head_error_m"] for r in complete])) if complete else None,
            "backend": "physx_cpu", "device": "cpu"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("outputs/graph/dataset.pt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rollout", action="store_true")
    p.add_argument("--warmstart", action="store_true", help="also evaluate paired alignment/insertion snapshots")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--rollout-episodes", type=int, help="default: all test episodes")
    p.add_argument("--episode-ids", type=int, nargs="+", help="specific held-out rollout episodes, e.g. a video example")
    p.add_argument("--videos", type=int, default=0, help="save first N episodes per rollout setting")
    p.add_argument("--latency-repeats", type=int, default=50)
    p.add_argument("--refresh-latency", action="store_true", help="refresh offline/latency metrics and retain existing paired rollout results")
    args = p.parse_args()
    if args.horizon < 1 or args.latency_repeats < 1:
        p.error("horizon and latency-repeats must be positive")
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    cfg, name = checkpoint["config"], checkpoint["model_name"]
    seed_all(cfg["seed"], cfg["threads"])
    data, geometry = load_dataset(args.dataset)
    dataset_hash = sha256(args.dataset)
    if checkpoint["dataset_sha256"] != dataset_hash or checkpoint["splits"] != data["manifest"]["splits"]:
        raise ValueError("checkpoint dataset/split mismatch")
    model = make_policy(name, cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    norm = FeatureNormalizer(**checkpoint["normalizer"])
    builder = GraphBuilder(cfg["surface_k"], cfg["interaction_k"], center=name == "center")
    test = split_indices(data, "test")
    with torch.inference_mode():
        predictions = torch.cat([model(norm.apply(name, geometry.select(ids), builder)) for ids in test.split(cfg["batch_size"])])
        targets = data["actions"][test]
        metrics = action_metrics(predictions, targets)
        by_phase = {}
        for phase, label in enumerate(PHASES):
            mask = data["phase"][test] == phase
            by_phase[label] = {"samples": int(mask.sum()), **action_metrics(predictions[mask], targets[mask])} if mask.any() else {"samples": 0}
        benchmark = test[torch.linspace(0, len(test)-1, min(16, len(test))).long()]
        g = geometry.select(benchmark[:1])
        for _ in range(10):
            model(norm.apply(name, g, builder))
        construction, inference = [], []
        for repeat in range(args.latency_repeats):
            g = geometry.select(benchmark[repeat % len(benchmark):repeat % len(benchmark)+1])
            start = perf_counter()
            inputs = norm.apply(name, g, builder)
            construction.append((perf_counter()-start)*1000)
            start = perf_counter()
            model(inputs)
            inference.append((perf_counter()-start)*1000)
        report = {"model": name, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
                  "dataset_sha256": dataset_hash, "test_episode_ids": data["manifest"]["splits"]["test"],
                  "test_samples": len(test), "parameters": checkpoint["parameters"], "config": cfg,
                  "versions": versions(), "offline": metrics, "phase_offline": by_phase,
                  "latency": {"batch_size": 1, "states": len(benchmark), "repeats": args.latency_repeats, "device": "cpu", "threads": cfg["threads"],
                              "feature_or_graph_construction_ms": float(np.mean(construction)),
                              "network_inference_ms": float(np.mean(inference)),
                              "total_ms": float(np.mean(construction)+np.mean(inference)),
                              "graph_stats": None if name == "flat" else inputs.stats},
                  "rollout": None, "alignment_rollout": None}
        if args.refresh_latency:
            previous = json.loads(args.output.read_text())
            if any(previous[k] != report[k] for k in ("model", "checkpoint", "dataset_sha256")):
                raise ValueError("existing result does not match checkpoint/dataset")
            for setting in ("rollout", "alignment_rollout"):
                report[setting] = previous[setting]
        if args.rollout:
            report["rollout"] = rollout(model, name, cfg, norm, builder, data, geometry, args)
        if args.warmstart:
            report["alignment_rollout"] = rollout(model, name, cfg, norm, builder, data, geometry, args, warm=True)
    write_json(args.output, report)
    print(json.dumps({"model": name, "offline": metrics, "latency": report["latency"]}, indent=2))


if __name__ == "__main__":
    main()
