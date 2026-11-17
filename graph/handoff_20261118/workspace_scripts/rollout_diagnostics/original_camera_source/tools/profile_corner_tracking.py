"""Profile an actual prepared episode without changing or resuming training runs."""

import argparse
import cProfile
import json
from pathlib import Path
import pstats

import numpy as np
import torch
import yaml

from graph.temporal.corner_tracking.tracker import TrackingSettings
from tools.corner_tracking_inputs import load_training_episode, require_cuda_device
from training.gnn.temporal.corner_tracking import build_models, resolve_config
from training.gnn.temporal.corner_tracking_calibration import bootstrap_gates
from training.gnn.temporal.corner_tracking_rollout import replay_episode
from utils.timer import Timer


def main():
    parser = argparse.ArgumentParser(description=__doc__ + " Requires CUDA; CPU/MPS are unsupported.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--architecture", choices=("mamba", "transformer"), required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Use the optimized training-only path without diagnostic exports")
    parser.add_argument("--node-only", action="store_true", help="Profile the frozen-GNN pretraining loss")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    if options.frames < 3:
        parser.error("--frames must be at least 3")
    config = resolve_config(yaml.safe_load(Path(options.config).read_text()))
    device = require_cuda_device(config["device"])
    episode, manifest = load_training_episode(config["data"]["root"])
    source_frames = len(episode.frames)
    episode.frames = episode.frames[:options.frames]
    torch.set_num_threads(1)
    motion, gnn = build_models(config, options.architecture, 0, device)
    if options.node_only:
        gnn.requires_grad_(False)
    calibration, _ = bootstrap_gates([episode])
    calibration.variance_scale[:] = 0.1  # Exercise completed-graph path as well.
    settings = TrackingSettings(**config["tracking"])
    plan = dict(mode="natural", start=0, stop=0)
    optimizer = torch.optim.AdamW(list(motion.parameters()) + list(gnn.parameters()), lr=0.001)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    profiler = cProfile.Profile()
    timer = Timer()
    with timer.measure("training_step"):
        profiler.enable()
        replay_options = {"collect_diagnostics": False} if options.no_diagnostics else {}
        if options.node_only and options.no_diagnostics:
            replay_options["compute_object_loss"] = False
        result = replay_episode(motion, gnn, episode, plan, calibration, settings, config["loss"], **replay_options)
        loss = (result["node_loss"] if options.node_only else
                result["object_loss"] + 0.1 * result["node_loss"])
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        profiler.disable()
    elapsed = timer.summary()["training_step"]["total_seconds"]
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    profiler.dump_stats(str(output / "profile.pstats"))
    if result["prediction"] is not None:
        np.savez_compressed(output / "predictions.npz", prediction=result["prediction"],
                            target=result["target"])
    (output / "tracking.json").write_text(json.dumps(result["trace"], indent=2) + "\n")
    torch.save(dict(motion={name: value.detach().cpu() for name, value in motion.state_dict().items()},
                    gnn={name: value.detach().cpu() for name, value in gnn.state_dict().items()}),
               output / "updated_weights.pt")
    with (output / "profile.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(80)
    report = dict(architecture=options.architecture, frames=len(episode.frames),
                  source_frames=source_frames, complete_episode=len(episode.frames) == source_frames,
                  episode=episode.name, dataset_sha256=manifest["dataset_sha256"],
                  elapsed_seconds=elapsed, seconds_per_frame=elapsed / len(episode.frames),
                  loss=float(loss.detach().cpu()), node_supervision_count=result["node_supervision_count"],
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  diagnostics=not options.no_diagnostics,
                  node_only=options.node_only,
                  context="One real episode or prefix under concurrent production GPU load; cProfile wall times overlap")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
