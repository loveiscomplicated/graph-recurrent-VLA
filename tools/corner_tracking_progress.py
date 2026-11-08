"""Observe corner-training progress without changing checkpointed model sources.

Run `python -m tools.corner_tracking_progress --config recipe.yaml`. The wrappers
only report completed work; model data, random state and optimizer are untouched.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch

from data_collection.episode_dataset import file_sha256
from main import load_config, load_trainer
import training.gnn.temporal.corner_tracking as workflow
import training.gnn.temporal.corner_tracking_calibration as calibration


class Progress:
    """Report completed episodes with a monotonic, stage-local throughput."""

    def __init__(self, every_episodes=10, every_seconds=30):
        self.every_episodes = every_episodes
        self.every_seconds = every_seconds
        self.identity = "unselected"
        self.stage = None
        self.completed = 0
        self.total = 0
        self.started = 0.0
        self.last_report = 0.0

    def begin(self, stage, total):
        self.stage, self.total, self.completed = stage, total, 0
        self.started = self.last_report = time.monotonic()
        self.report()

    def report(self, loss=None, supervised_nodes=None, position_error=None):
        elapsed = time.monotonic() - self.started
        loss_text = f" episode_loss={loss():.6g}" if loss is not None else ""
        if supervised_nodes is not None:
            error_text = "N/A" if position_error is None else f"{float(position_error.cpu()) * 1000:.2f}"
            loss_text += f" position_error_mm={error_text}"
            loss_text += f" supervised_nodes={supervised_nodes}"
        eta = f"{elapsed / self.completed * (self.total - self.completed):.0f}s" if self.completed else "N/A"
        print(f"[progress] {self.identity} stage={self.stage} "
              f"episodes={self.completed}/{self.total} elapsed_s={elapsed:.1f} stage_eta={eta}{loss_text}", flush=True)
        self.last_report = time.monotonic()

    def advance(self, loss=None, supervised_nodes=None, position_error=None):
        self.completed += 1
        if (self.completed == self.total or self.completed % self.every_episodes == 0
                or time.monotonic() - self.last_report >= self.every_seconds):
            self.report(loss, supervised_nodes, position_error)

    @contextmanager
    def scoped(self, stage, total):
        """Restore the previous training counter after calibration/evaluation."""
        previous = (self.stage, self.total, self.completed, self.started, self.last_report)
        self.begin(stage, total)
        try:
            yield
        finally:
            self.stage, self.total, self.completed, self.started, self.last_report = previous


def run(config_path, every_episodes=10, every_seconds=30):
    """Instrument the existing root dispatcher, including checkpoint resume."""
    if every_episodes < 1 or every_seconds <= 0:
        raise ValueError("Progress intervals must be positive")
    progress = Progress(every_episodes, every_seconds)
    original_train = workflow.train_one
    original_plans = workflow.plans_for
    original_eligibility = workflow.confirmed_eligibility
    original_replay = workflow.replay_episode
    original_calibration_replay = calibration.replay_episode
    original_calibrate = workflow.calibrate_uncertainty
    original_evaluate = workflow.evaluate_episodes
    training_settings = None

    def train_one(config, groups, identity, architecture, seed):
        nonlocal training_settings
        progress.identity = f"{architecture} seed={seed}"
        training_settings = config["training"]
        return original_train(config, groups, identity, architecture, seed)

    def eligibility(*args, **kwargs):
        result = original_eligibility(*args, **kwargs)
        progress.advance()
        return result

    def plans_for(episodes, seed, tag, config, gate):
        with progress.scoped(f"mask_planning:{tag}", len(episodes)):
            result = original_plans(episodes, seed, tag, config, gate)
        if tag.startswith("train_epoch_"):
            epoch = int(tag.removeprefix("train_epoch_")) + 1
            progress.begin(f"train_epoch:{epoch} forward_completed", len(episodes))
        return result

    def replay(*args, **kwargs):
        result = original_replay(*args, **kwargs)
        loss = None
        if progress.stage and progress.stage.startswith("train_epoch:"):
            def episode_loss():
                node_loss = float(result["node_loss"].detach().cpu())
                if result["object_loss"] is None:
                    return node_loss
                object_loss = float(result["object_loss"].detach().cpu())
                return object_loss + training_settings["node_weight"] * node_loss

            loss = episode_loss
        progress.advance(loss, result["node_supervision_count"] if loss is not None else None,
                         result["position_error_m"] if loss is not None else None)
        return result

    def calibration_replay(*args, **kwargs):
        result = original_calibration_replay(*args, **kwargs)
        progress.advance()
        return result

    def calibrate(*args, **kwargs):
        episodes = args[2] if len(args) > 2 else kwargs["episodes"]
        with progress.scoped("calibration", len(episodes)):
            return original_calibrate(*args, **kwargs)

    def evaluate(*args, **kwargs):
        episodes = args[2] if len(args) > 2 else kwargs["episodes"]
        with progress.scoped("evaluation", len(episodes)):
            return original_evaluate(*args, **kwargs)

    config = load_config(config_path)
    config["progress_observer"] = dict(sha256=file_sha256(Path(__file__)),
                                       every_episodes=every_episodes, every_seconds=every_seconds)
    with patch.object(workflow, "train_one", train_one), patch.object(workflow, "plans_for", plans_for), \
         patch.object(workflow, "confirmed_eligibility", eligibility), \
         patch.object(workflow, "replay_episode", replay), \
         patch.object(calibration, "replay_episode", calibration_replay), \
         patch.object(workflow, "calibrate_uncertainty", calibrate), \
         patch.object(workflow, "evaluate_episodes", evaluate):
        return load_trainer(config["trainer"])(config)


def main():
    parser = argparse.ArgumentParser(description="Log causal corner-training episode progress")
    parser.add_argument("--config", required=True)
    parser.add_argument("--every-episodes", type=int, default=10)
    parser.add_argument("--every-seconds", type=float, default=30)
    options = parser.parse_args()
    run(options.config, options.every_episodes, options.every_seconds)


if __name__ == "__main__":
    main()
