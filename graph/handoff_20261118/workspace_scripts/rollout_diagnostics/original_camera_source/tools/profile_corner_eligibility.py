"""Time causal mask eligibility on a single checksummed real episode."""

import argparse
import json
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash
from data_collection.temporal.corner_tracks import confirmed_eligibility
from graph.temporal.corner_tracking.tracker import TrackingSettings
from tools.corner_tracking_inputs import load_training_episode
from training.gnn.temporal.corner_tracking_calibration import bootstrap_gates
from utils.timer import Timer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    episode, manifest = load_training_episode(options.data_root)
    calibration, _ = bootstrap_gates([episode])
    torch.set_num_threads(1)
    timings, digests = [], []
    timer = Timer()
    for repeat in range(2):
        episode.eligibility_cache.clear()
        measurement = f"eligibility_{repeat}"
        with timer.measure(measurement):
            eligible = confirmed_eligibility(episode, calibration, TrackingSettings())
        timings.append(timer.summary()[measurement]["total_seconds"])
        digests.append(canonical_hash([sorted(ids) for ids in eligible]))
    if len(set(digests)) != 1:
        raise ValueError("Repeated eligibility results differ")
    result = dict(episode=episode.name, frames=len(episode.frames),
                  dataset_sha256=manifest["dataset_sha256"], seconds=timings,
                  eligibility_sha256=digests[0], context="CPU; concurrent production jobs")
    Path(options.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
