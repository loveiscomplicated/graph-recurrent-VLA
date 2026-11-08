"""Measure cold and warm full-episode training steps on a real prepared episode."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from graph.temporal.corner_tracking.tracker import TrackingSettings
from tools.corner_tracking_inputs import load_training_episode, require_cuda_device
from training.gnn.temporal.corner_tracking import build_models, resolve_config
from training.gnn.temporal.corner_tracking_calibration import bootstrap_gates
from training.gnn.temporal.corner_tracking_rollout import replay_episode
from utils.timer import Timer


def benchmark(config, phase, frames, repeats):
    """Restore identical initial weights before every timed optimizer step."""
    device = require_cuda_device(config["device"])
    torch.set_num_threads(1)
    episode, manifest = load_training_episode(config["data"]["root"])
    if frames:
        episode.frames = episode.frames[:frames]
    motion, gnn = build_models(config, config["architectures"][0], config["seeds"][0], device)
    motion.train()
    gnn.train(phase == "joint")
    gnn.requires_grad_(phase == "joint")
    initial_motion, initial_gnn = deepcopy(motion.state_dict()), deepcopy(gnn.state_dict())
    gate, _ = bootstrap_gates([episode])
    settings = TrackingSettings(**config["tracking"])
    report = []
    timer = Timer()
    for iteration in range(repeats + 1):
        motion.load_state_dict(initial_motion)
        gnn.load_state_dict(initial_gnn)
        execution = config.get("execution", {})
        optimizer = torch.optim.AdamW([
            dict(params=motion.parameters(), lr=config["training"]["motion_lr"]),
            dict(params=gnn.parameters(), lr=config["training"]["gnn_lr"]),
        ], weight_decay=config["training"]["weight_decay"], fused=execution.get("optimizer_fused", False))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        measurement = f"training_step_{iteration}"
        with timer.measure(measurement):
            result = replay_episode(motion, gnn, episode, dict(mode="natural", start=0, stop=0),
                                    gate, settings, config["loss"], collect_diagnostics=False,
                                    compute_object_loss=phase == "joint")
            loss = result["node_loss"]
            if phase == "joint":
                loss = result["object_loss"] + config["training"]["node_weight"] * loss
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(list(motion.parameters()) + list(gnn.parameters()),
                                                     config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
        elapsed = timer.summary()[measurement]["total_seconds"]
        record = dict(iteration=iteration, cold=iteration == 0, seconds=elapsed,
                      loss=float(loss.detach().cpu()), gradient_norm=float(gradient.cpu()),
                      supervised_nodes=result["node_supervision_count"],
                      peak_bytes=torch.cuda.max_memory_allocated(device))
        report.append(record)
        print(json.dumps(record), flush=True)
    return dict(architecture=config["architectures"][0], phase=phase, frames=len(episode.frames),
                episode=episode.name, dataset_sha256=manifest["dataset_sha256"],
                execution=config.get("execution", {}), measurements=report,
                median_warm_seconds=float(np.median([row["seconds"] for row in report[1:]])),
                context="Single real episode; other training jobs share the GPU; identical weights each repeat")


def main():
    parser = argparse.ArgumentParser(description=__doc__ + " Requires CUDA; CPU/MPS are unsupported.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("pretrain", "joint"), default="pretrain")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 0 or args.repeats < 1:
        parser.error("frames must be nonnegative and repeats positive")
    config = resolve_config(yaml.safe_load(Path(args.config).read_text()))
    report = benchmark(config, args.phase, args.frames, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
