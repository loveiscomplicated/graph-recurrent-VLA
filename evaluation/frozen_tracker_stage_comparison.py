"""Rescore three frozen Object21 pipelines on common GT insertion-depth slices."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.five_mode_corner_tracker_comparison import load_test
from evaluation.frozen_surface_tracker_gnn import position_errors_mm
from training.gnn.loss import object21_loss
from utils.experiment_logger import ExperimentLogger, write


def depth_stage_masks(depth):
    """Reuse the existing axial-depth bands; these are geometry, not contact labels."""
    depth = np.asarray(depth)
    if depth.ndim != 1 or not np.isfinite(depth).all():
        raise ValueError("Expected finite per-frame GT insertion depth in mm")
    return {
        "all": np.ones(len(depth), dtype=bool),
        "alignment_pre_insertion": depth <= 0,
        "insertion": depth > 0,
        "entry": (depth > 0) & (depth <= 20),
        "deep": depth > 20,
    }


def score_slice(prediction, residual, target, mask, loss_settings):
    """Calculate the unchanged Object21 loss and full 3D errors on one frame slice."""
    count = int(mask.sum())
    if not count:
        return None
    output = {"object21": torch.from_numpy(prediction[mask]),
              "endpoint_residual": torch.from_numpy(residual[mask])}
    terms = object21_loss(output, torch.from_numpy(target[mask]), **loss_settings)
    errors = position_errors_mm(prediction[mask], target[mask])
    return {
        "object21_loss": float(terms["total"]),
        **{name + "_mm": float(values.mean()) for name, values in errors.items()},
    }


def summarize_stages(prediction, residual, target, modes, masks, loss_settings):
    """Retain pooled and equal-mode scores; use only nonempty modes for each slice."""
    if (residual.shape != (len(target), 3) or not np.isfinite(residual).all()
            or residual.dtype != np.float32 or prediction.dtype != np.float32
            or target.dtype != np.float32):
        raise ValueError("Object21 scoring requires finite FP32 predictions, targets and [frames,3] residuals")
    position_errors_mm(prediction, target)
    result = {}
    for stage, mask in masks.items():
        by_mode = {}
        for mode in sorted(set(modes)):
            selected = mask & (modes == mode)
            score = score_slice(prediction, residual, target, selected, loss_settings)
            if score is not None:
                by_mode[mode] = score
        balanced = {
            metric: float(np.mean([score[metric] for score in by_mode.values()]))
            for metric in next(iter(by_mode.values()))
        } if by_mode else None
        result[stage] = dict(mode_balanced=balanced, pooled=score_slice(
            prediction, residual, target, mask, loss_settings), by_mode=by_mode)
    return result


def evaluate(config):
    """Verify saved inference receipts before using GT to define shared score-only slices."""
    torch.set_num_threads(config.get("cpu_threads", 4))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    episodes, manifest = load_test(config["data_root"], config["test_manifest_sha256"])
    depth = np.concatenate([episode["depth_mm"] for episode in episodes])
    episode_index = np.repeat(np.arange(len(episodes)), [len(episode["tcp"]) for episode in episodes])
    modes = np.asarray([episodes[index]["mode"] for index in episode_index])
    masks = depth_stage_masks(depth)
    stage_counts = {
        stage: dict(frames=int(mask.sum()), episodes=int(np.unique(episode_index[mask]).size),
                    modes={mode: int((mask & (modes == mode)).sum()) for mode in sorted(set(modes))})
        for stage, mask in masks.items()
    }
    reports = {}
    reference_target = None
    with ExperimentLogger(team="graph", model="frozen_tracker_stage_comparison", job_type="evaluate",
                          config=config, root=output / "experiments") as logger:
        result = dict(run_id=logger.run_id, episodes=len(episodes), frames=len(depth),
                      aggregation="equal mean of nonempty mode frame-means in each slice",
                      stage_definition="GT axial insertion depth: pre<=0mm; insertion>0mm; entry(0,20]mm; deep>20mm",
                      caveat="Pre-insertion includes approach/distant frames; axial insertion is not a contact/inside-opening label",
                      stage_counts=stage_counts, models={}, config=config)
        for name, settings in config["models"].items():
            report_path = Path(settings["report"])
            if file_sha256(report_path) != settings["report_sha256"]:
                raise ValueError(f"{name} inference report checksum mismatch")
            report = json.loads(report_path.read_text())
            if (report["config"]["test_manifest_sha256"] != config["test_manifest_sha256"]
                    or report["episodes"] != 450 or report["frames"] != len(depth)):
                raise ValueError(f"{name} inference did not use the common complete test split")
            path = Path(settings["predictions"])
            model_receipt = report["models"][settings["report_model"]]
            if file_sha256(path) != model_receipt["predictions_sha256"]:
                raise ValueError(f"{name} inference payload checksum mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                prediction, target = arrays["object21"], arrays["target"]
                residual = arrays["endpoint_residual"]
                np.testing.assert_array_equal(arrays["episode_index"], episode_index)
            if len(target) != manifest["frames"]:
                raise ValueError(f"{name} target count mismatch")
            if reference_target is None:
                reference_target = target
                truth = np.concatenate([episode["gt"][:, :8].mean(axis=1) - episode["tcp"] for episode in episodes])
                np.testing.assert_allclose(target[:, :3], truth, atol=2e-6, rtol=0)
            else:
                np.testing.assert_array_equal(target, reference_target)
            scores = summarize_stages(prediction, residual, target, modes, masks, config["loss"])
            result["models"][name] = dict(source_run_id=report["run_id"],
                prediction_sha256=model_receipt["predictions_sha256"], stages=scores)
            reports[name] = report
            for stage, values in scores.items():
                if values["mode_balanced"]:
                    logger.log(values["mode_balanced"], step=0, prefix=f"test/{name}/{stage}")
        result["reduction_percent"] = {}
        baseline = result["models"][config["baseline"]]["stages"]
        for name, model in result["models"].items():
            if name == config["baseline"]:
                continue
            result["reduction_percent"][name] = {}
            for stage in masks:
                original, candidate = baseline[stage]["mode_balanced"], model["stages"][stage]["mode_balanced"]
                result["reduction_percent"][name][stage] = {
                    metric: (value - candidate[metric]) / value * 100 if value != 0 else None
                    for metric, value in original.items()
                } if original is not None else None
        mask_path = output / "stage_membership.npz"
        np.savez_compressed(mask_path, depth_mm=depth, episode_index=episode_index, mode=modes, **masks)
        result["stage_membership_sha256"] = file_sha256(mask_path)
        write(output / "results.json", result)
        logger.link_output(mask_path)
        logger.link_output(output / "results.json")
        logger.record(dataset_sha256=config["test_manifest_sha256"], device="cpu", inference_reused=True,
                      episodes=len(episodes), frames=len(depth), source_run_ids={name: report["run_id"] for name, report in reports.items()})
        print(json.dumps({name: {stage: value["mode_balanced"] for stage, value in model["stages"].items()}
                          for name, model in result["models"].items()}, indent=2), flush=True)
