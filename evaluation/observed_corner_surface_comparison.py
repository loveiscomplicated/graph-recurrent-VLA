"""Compare frozen corner-only/surface trackers on identical post-observation targets."""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_surface_context import (
    attach_surface_context,
    surface_sequence_arrays,
)
from evaluation.five_mode_corner_tracker_comparison import (
    BODIES,
    load_test,
    observation_history,
    predict,
    summarize,
)
from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker
from utils.experiment_logger import ExperimentLogger, write


@torch.inference_mode()
def predict_surface(episodes, checkpoint, batch_size, device):
    """Use the unchanged surface encoder, causal inputs and checkpoint-fitted prior."""
    model = (
        SurfaceSpatialCornerTracker(**checkpoint["config"]["model"]).to(device).eval()
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    arrays, lengths, _ = surface_sequence_arrays(
        episodes, checkpoint["fallback"].cpu().numpy()
    )
    names = (
        "features",
        "anchor",
        "detected",
        "observed",
        "valid",
        "surface_features",
        "surface_valid",
    )
    tensors = {name: torch.as_tensor(arrays[name], device=device) for name in names}
    predictions = []
    for start in range(0, len(episodes), batch_size):
        stop = min(start + batch_size, len(episodes))
        length = int(lengths[start:stop].max())
        batch = [tensors[name][start:stop, :length] for name in names]
        output = model(*batch)
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite surface-model test predictions")
        torch.testing.assert_close(output[batch[3]], batch[2][batch[3]], rtol=0, atol=0)
        for index, row in enumerate(output):
            predictions.append(row[: lengths[start + index]].cpu().numpy())
    return np.concatenate(predictions)


def paired_metrics(error, eligible, episodes, history, surface_present):
    """Pool identical eligible point-frames for means and percentiles of both models."""
    result = {
        "body": {
            name: summarize(error[:, slots][eligible[:, slots]])
            for name, slots in BODIES.items()
        },
    }
    result["mode"] = {}
    for mode in sorted({episode["mode"] for episode in episodes}):
        frames = np.asarray(
            [episodes[index]["mode"] == mode for index in history["episode_index"]]
        )
        result["mode"][mode] = {
            name: summarize(error[frames, slots][eligible[frames, slots]])
            for name, slots in BODIES.items()
        }
    result["peg_surface_support"] = {
        name: summarize(error[:, :8][eligible[:, :8] & frames[:, None]])
        for name, frames in (("present", surface_present), ("absent", ~surface_present))
    }
    result["occlusion_age"] = {}
    for name, lower, upper in (
        ("1_10", 1, 10),
        ("11_40", 11, 40),
        ("41_100", 41, 100),
        ("over_100", 101, 100000),
    ):
        mask = eligible & (history["age"] >= lower) & (history["age"] <= upper)
        result["occlusion_age"][name] = {
            body: summarize(error[:, slots][mask[:, slots]])
            for body, slots in BODIES.items()
        }
    return result


def evaluate(config):
    """Evaluate both validation-selected checkpoints once; never tune on test results."""
    device = torch.device(config["device"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    episodes, manifest = load_test(config["data_root"], config["test_manifest_sha256"])
    development_path = Path(config["training_data_root"]) / "manifest.json"
    development_hash = file_sha256(development_path)
    development = json.loads(development_path.read_text())
    development_seeds = {row["seed"] for row in development["records"]}
    test_seeds = {episode["seed"] for episode in episodes}
    if development_seeds & test_seeds:
        raise ValueError("Test reset seeds overlap the training/validation dataset")
    if (
        file_sha256(Path(config["surface_context"]) / "manifest.json")
        != config["surface_manifest_sha256"]
    ):
        raise ValueError("Frozen test-surface manifest checksum mismatch")
    surface_hash = attach_surface_context(
        {"test": episodes}, config["surface_context"], config["data_root"]
    )
    history = observation_history(episodes)
    observed = np.concatenate([episode["observed"] for episode in episodes])
    truth = np.concatenate([episode["gt"] for episode in episodes])
    eligible = history["ever"] & ~observed
    surface_present = np.concatenate(
        [episode["surface_valid"].any(axis=1) for episode in episodes]
    )
    saved = {
        "gt": truth,
        "observed": observed,
        "ever_seen": history["ever"],
        "eligible": eligible,
        "episode_index": history["episode_index"],
        "age": history["age"],
        "surface_present": surface_present,
        "episode_offsets": np.r_[
            0, np.cumsum([len(episode["gt"]) for episode in episodes])
        ],
    }
    result = {
        "episodes": len(episodes),
        "seeds": manifest["seeds"],
        "frames": len(truth),
        "eligibility": "actual corner observed at or before t, currently undetected; first detections excluded",
        "eligible_point_frames": int(eligible.sum()),
        "excluded_before_first_observation": int((~history["ever"] & ~observed).sum()),
        "test_manifest_sha256": config["test_manifest_sha256"],
        "surface_manifest_sha256": surface_hash,
        "models": {},
        "config": config,
    }
    train_data_hash = None
    fallback_hash = None
    with ExperimentLogger(
        team="graph",
        model="observed_corner_surface_test_comparison",
        job_type="evaluate",
        config=config,
        root=output / "experiments",
    ) as logger:
        result["run_id"] = logger.run_id
        logger.record(
            runtime={
                "torch": str(torch.__version__),
                "device": str(device),
                "precision": "FP32; TF32 disabled; identical batch size",
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            }
        )
        for name, settings in config["models"].items():
            path = Path(settings["checkpoint"])
            if file_sha256(path) != settings["sha256"]:
                raise ValueError(f"Frozen checkpoint checksum mismatch: {name}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            training = checkpoint["config"]
            if training["training"].get("supervision") != "after_first_observation":
                raise ValueError(
                    f"Checkpoint {name} was not trained with first-observation exclusion"
                )
            is_surface = bool(training.get("surface_context"))
            if is_surface != (settings["kind"] == "surface"):
                raise ValueError(
                    f"Checkpoint architecture differs from configured kind: {name}"
                )
            current_hash = training["dataset_manifest_sha256"]
            if current_hash != development_hash:
                raise ValueError(
                    f"Checkpoint training dataset identity differs from the audited manifest: {name}"
                )
            if train_data_hash is not None and current_hash != train_data_hash:
                raise ValueError(
                    "The checkpoints were trained on different corner datasets"
                )
            train_data_hash = current_hash
            current_fallback_hash = hashlib.sha256(
                checkpoint["fallback"].cpu().numpy().tobytes()
            ).hexdigest()
            if fallback_hash is not None and current_fallback_hash != fallback_hash:
                raise ValueError(
                    "The models have different train-fitted priors, so their corner inputs differ"
                )
            fallback_hash = current_fallback_hash
            function = predict_surface if is_surface else predict
            prediction = function(episodes, checkpoint, config["batch_size"], device)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"Invalid prediction shape or values for {name}")
            error = np.linalg.norm(prediction - truth, axis=-1) * 1000
            if not np.isfinite(error[eligible]).all():
                raise RuntimeError(f"Nonfinite eligible errors for {name}")
            metrics = paired_metrics(
                error, eligible, episodes, history, surface_present
            )
            result["models"][name] = {
                "epoch": checkpoint["epoch"],
                "run_id": checkpoint["run_id"],
                "sha256": settings["sha256"],
                "training_batch_size": training["batch_size"],
                "training_dataset_manifest_sha256": current_hash,
                **metrics,
                "fallback_sha256": current_fallback_hash,
            }
            saved[f"prediction_{name}"] = prediction
            saved[f"error_{name}_mm"] = error
            logger.link_output(path, checkpoint=True)
            logger.log(metrics["body"], prefix=f"test/{name}")
            print(name, json.dumps(metrics["body"]), flush=True)
            del checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        np.savez_compressed(output / "predictions.npz", **saved)
        result["predictions_sha256"] = file_sha256(output / "predictions.npz")
        write(output / "results.json", result)
        for path in (output / "results.json", output / "predictions.npz"):
            logger.link_output(path)
        print("COMPARISON_COMPLETE", logger.run_id, flush=True)
