"""Evaluate frozen Mamba/Gate composition on one identified validation snapshot."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.hybrid_features import replay_hybrid
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from evaluation.gnn.temporal.hybrid_diagnostics import (
    analyze_hybrid,
    check_reproduction,
    save_hybrid_plots,
)
from evaluation.gnn.temporal.hybrid_inputs import load_reference, restore_branches
from evaluation.gnn.temporal.insertion import (
    replay_insertion,
    score_insertion_methods,
)
from evaluation.gnn.temporal_gate import replay_gate
from graph.pose_geometry import geometric_endpoint
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def evaluate(config):
    """Audit, smoke-check, replay all validation frames, then save paired diagnostics."""
    device = set_device(config.get("device", "cpu"))
    torch.set_num_threads(config.get("math_threads", 1))
    torch.manual_seed(config.get("seed", 0))
    data, baselines, reference, previous, provenance, identity, frames = load_reference(config)
    mamba, gate, audit = restore_branches(config, reference, provenance, identity, device)
    if mamba.with_robot != ("tcp_quaternion" in data):
        raise ValueError("Observed robot fields differ from Mamba training schema")
    with ExperimentLogger(
        team="graph",
        model="hybrid_temporal_estimation",
        config=config,
        seed=config.get("seed", 0),
        job_type="evaluate",
    ) as experiment:
        output = Path(config["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        dataset = InsertionEpisodeDataset(data)
        smoke_rows = np.concatenate(dataset.indices[:2])
        smoke = {name: value[smoke_rows] for name, value in data.items()}
        print("Checking independent branch replay on two complete episodes", flush=True)
        solo_mamba, _, parity = replay_insertion(mamba, smoke, device)
        solo_gate = replay_gate(gate, smoke, device)["object21"]
        combined, _, _ = replay_hybrid(mamba, gate, smoke, device)
        np.testing.assert_allclose(combined["mamba"], solo_mamba, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(combined["learned_gate"], solo_gate, atol=1e-6, rtol=1e-5)
        audit["smoke"] = {
            "episodes": 2,
            "frames": len(smoke_rows),
            "standalone_mamba_sequence_stream_max_difference": parity,
            "standalone_gate_and_mamba_equal_integrated_replay": True,
        }
        print(
            f"Replaying {len(dataset)} episodes / {len(data['frame'])} frames on {device}",
            flush=True,
        )
        predictions, alpha, control_geometry = replay_hybrid(mamba, gate, data, device)
        audit["reproduction"] = check_reproduction(
            predictions, baselines, previous, reference["evaluation"], data
        )
        for name in ("hybrid_h1", "hybrid_h2"):
            difference = float(np.max(np.abs(predictions[name + "_a"] - predictions[name + "_b"])))
            audit[name + "_method_a_b_max_difference_m"] = difference
            np.testing.assert_allclose(
                predictions[name + "_a"], predictions[name + "_b"], atol=1e-6, rtol=1e-5
            )
        # Keep the historical GNN/Gate vector assemblies, and label fair geometry ablations separately.
        for name in ("gnn", "learned_gate"):
            geometric = predictions[name].copy()
            geometric[:, 7:10] = geometric_endpoint(torch.from_numpy(geometric)).numpy()
            predictions[name + "_geometric"] = geometric
        summary, errors = score_insertion_methods(data, predictions, reference["evaluation"])
        diagnostics = analyze_hybrid(
            data,
            predictions,
            errors,
            reference["temporal_split_manifest"],
            seed=config.get("seed", 0),
            resamples=config.get("bootstrap_resamples", 10000),
        )
        summary.update(
            split="validation",
            independent_test=False,
            excluded_frames=0,
            primary_method="hybrid_h1_a",
            inference_context="full_episode_streaming",
        )
        audit["frame_identity_sha256"] = hashlib.sha256(
            json.dumps(frames, sort_keys=True).encode()
        ).hexdigest()
        audit["inputs"] = {
            name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
            for name, path in (
                ("prediction_snapshot", Path(config["reference_evaluation"]) / "predictions.npz"),
                ("data_identity", config["data_identity"]),
                ("source_summary", Path(config["reference_evaluation"]) / "summary.json"),
            )
        }
        np.savez_compressed(
            output / "predictions.npz",
            **data,
            gate_alpha=alpha,
            hybrid_predicted_hole_frame=control_geometry,
            **{name + "_object21": value for name, value in predictions.items()},
        )
        np.savez_compressed(
            output / "frame_errors.npz",
            sample_ids=data["sample_ids"],
            **{
                name + "__" + metric: values
                for name, metrics in errors.items()
                for metric, values in metrics.items()
            },
        )
        write_json(output / "summary.json", summary)
        write_json(output / "diagnostics.json", diagnostics)
        write_json(output / "audit.json", audit)
        write_json(output / "frame_manifest.json", frames)
        write_json(output / "effective_config.json", config)
        save_hybrid_plots(output, data, predictions, errors, diagnostics)
        for name, method in summary["methods"].items():
            experiment.log(method["groups"]["all"], prefix=name)
        experiment.record(
            dataset_sha256=identity["dataset_sha256"],
            device=str(device),
            split="validation",
            independent_test=False,
            frame_identity_sha256=audit["frame_identity_sha256"],
        )
        for name in ("gnn", "mamba", "gate"):
            experiment.link_output(config[name + "_checkpoint"], checkpoint=True)
        for path in output.iterdir():
            experiment.link_output(path)
        print(f"Hybrid evaluation saved: {output}", flush=True)
        return output
