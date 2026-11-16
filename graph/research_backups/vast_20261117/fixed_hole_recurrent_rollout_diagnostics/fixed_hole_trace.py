"""Record pre-action geometry around the existing exp2 rollout."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json

import cv2
import numpy as np
import torch

import exp2_eval as exp2
from utils.experiment_logger import ExperimentLogger


def read_rows(root):
    rows = [json.loads(line) for path in sorted(root.glob("results_*.jsonl"))
            for line in path.read_text().splitlines()]
    return {row["env_seed"]: row["none"] for row in rows
            if row["cand"] == "ens_mamba5@step" and "none" in row}


def lateral_trace(row, t, label):
    sample = row[t]
    vectors = sample["relative_hole_mm"]
    actual = np.asarray(vectors["gt"])
    raw = np.asarray(vectors["raw"])
    filtered = np.asarray(vectors["filtered"])
    return dict(anchor=label, t=t, gt_axial=round(float(actual[0]), 2),
                actual_lateral=round(float(np.linalg.norm(actual[1:])), 2),
                raw_error=round(float(np.linalg.norm((raw-actual)[1:])), 2),
                filtered_error=round(float(np.linalg.norm((filtered-actual)[1:])), 2))


def landmarks(rows, success):
    grasp = [row["grasp"] is True for row in rows]
    sustained = next((t for t in range(len(rows)-2) if all(grasp[t:t+3])), None)
    result = []
    if sustained is not None:
        result.append(lateral_trace(rows, sustained, "first_3_step_grasp"))
    if success:
        result.append(lateral_trace(rows, len(rows)-1, "before_success_action"))
    elif any(grasp):
        closest = min((t for t, held in enumerate(grasp) if held),
                      key=lambda t: np.linalg.norm(rows[t]["relative_hole_mm"]["gt"]))
        result.append(lateral_trace(rows, closest, "closest_while_grasped"))
    return result


def evaluate(config):
    source = Path(config["source"])
    source_exp2 = source / "recurrent/exp2"
    history = read_rows(source_exp2 / "eval/final_ens")
    assert len(history) == 200 and sum(row["success"] for row in history.values()) == 179
    seeds = config["seeds"]
    paths = json.loads((source / "graph_candidate_paths.json").read_text())
    checkpoints = [source_exp2 / f"checkpoints/runs_r6_mix_filter/mamba_step_s{i}/last.pt"
                   for i in range(5)]
    input_paths = [*map(Path, paths.values()), *checkpoints]
    hashes = {str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for p in input_paths}
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    opts = SimpleNamespace(no_move=True, motion="none", oracle=False, trigger_m=0.06)
    with ExperimentLogger(team="recurrent", model="ens_mamba5", job_type="evaluate",
                          config={**config, "checkpoint_hashes": hashes},
                          root=config["experiment_root"]) as run:
        graph = exp2.Graph(paths["gnn"], paths["corner"], "cuda")
        decoder = exp2.Ensemble([exp2.Decoder(exp2.load_ck(p), "step") for p in checkpoints])
        env = exp2.make_env(16)
        try:
            for seed in seeds:
                rows = []
                original_act = decoder.act

                def record(state36, counts=None):
                    gt = exp2.gt_object21(env)
                    raw = np.asarray(state36[15:]).copy()
                    q_inv = exp2.q_conj(gt[13:17])
                    task = exp2.peg_task(env)
                    grasp = (bool(task.agent.is_grasping(task.peg).item()) if rows else None)
                    action = original_act(state36, counts)
                    filtered = decoder.last[15:].copy()
                    if not all(np.array_equal(d.last, decoder.last) for d in decoder.decs):
                        raise ValueError("Ensemble members received different filtered states")
                    vectors = {key: (1000 * exp2.q_rot(q_inv, value[7:10])).tolist()
                               for key, value in (("raw", raw), ("filtered", filtered), ("gt", gt))}
                    rows.append(dict(raw_object21=raw.tolist(), filtered_object21=filtered.tolist(),
                                     gt_object21=gt.tolist(), relative_hole_mm=vectors,
                                     grasp=grasp, action=action.tolist()))
                    return action

                with patch.object(decoder, "act", record):
                    result, _ = exp2.run_job(env, graph, decoder, seed, opts)
                old = history[seed]
                count = min(len(old["err_mm"]), len(result["err_mm"]))
                raw_gap = float(np.max(np.abs(np.asarray(old["err_mm"][:count]) -
                                              np.asarray(result["err_mm"][:count]))))
                filt_gap = float(np.max(np.abs(np.asarray(old["fxyz_mm"][:count]) -
                                               np.asarray(result["fxyz_mm"][:count]))))
                summary = dict(seed=seed, archived_success=old["success"],
                               replay_success=result["success"], archived_steps=old["steps"],
                               replay_steps=result["steps"], max_raw_3d_log_gap_mm=raw_gap,
                               max_filtered_xyz_log_gap_mm=filt_gap,
                               landmarks=landmarks(rows, result["success"]))
                path = run.directory / f"seed_{seed}.json"
                with path.open("x") as stream:
                    json.dump(dict(summary=summary, trace=rows), stream, allow_nan=False)
                run.link_output(path)
                print(json.dumps(summary), flush=True)
        finally:
            env.close()
