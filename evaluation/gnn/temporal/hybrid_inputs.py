"""Validate frozen hybrid snapshots, checkpoints and original source identity."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from evaluation.gnn.temporal.insertion import check_development_provenance
from graph.temporal.insertion import feature_names
from graph.temporal.observations import QUALITY_NAMES
from graph.temporal.pose_observations import INSERTION_CONTRACT
from graph.temporal_gate import LearnedTemporalGate
from training.gnn.data import canonical_hash
from training.gnn.temporal.insertion import build_insertion_model

DATA_FIELDS = (
    "sample_ids",
    "episode",
    "frame",
    "prediction",
    "target",
    "tcp_world",
    "features",
    "quality",
    "pose_quality",
    "tcp_quaternion",
    "gripper_qpos",
)
BASELINE_FIELDS = {
    "gnn": "gnn_object21",
    "learned_gate": "learned_gate_object21",
    "mamba": "primary_object21",
}


def verify_model_sources(reference_hashes, *, repository=None):
    """Require exact source bytes, allowing only this cleanup's import relocations.

    Historical runs keep their original paths and hashes. For those runs, undo
    the known import spelling changes in memory and compare the resulting bytes;
    a change to model calculations still fails the original SHA256 check.
    """
    source_paths = {
        "graph/temporal_mamba.py": "graph/temporal/insertion.py",
        "graph/temporal_gate.py": "graph/temporal_gate.py",
        "graph/pose_geometry.py": "graph/pose_geometry.py",
        "graph/gated_mpnn.py": "graph/gated_mpnn.py",
        "graph/temporal_pose_observations.py": "graph/temporal/pose_observations.py",
    }
    import_relocations = {
        "graph.temporal.observations": "graph.temporal_observations",
        "graph.temporal.pose_gru": "graph.temporal_pose",
        "graph.temporal.pose_observations": "graph.temporal_pose_observations",
    }
    if repository is None:
        repository = Path(__file__).resolve().parents[3]
    else:
        repository = Path(repository)
    audit = {}
    for original_path, current_path in source_paths.items():
        source = (repository / current_path).read_bytes()
        current_hash = hashlib.sha256(source).hexdigest()
        reference_path = current_path if current_path in reference_hashes else original_path
        expected_hash = reference_hashes.get(reference_path)
        matched_hash = current_hash
        verification = "exact_bytes"
        if matched_hash != expected_hash:
            for current_import, original_import in import_relocations.items():
                source = source.replace(
                    f"from {current_import} import ".encode(),
                    f"from {original_import} import ".encode(),
                )
            matched_hash = hashlib.sha256(source).hexdigest()
            verification = "original_bytes_after_import_relocation"
        if matched_hash != expected_hash and source.endswith(b"\n") and not source.endswith(b"\n\n"):
            matched_hash = hashlib.sha256(source + b"\n").hexdigest()
            verification = "original_bytes_after_import_and_final_blank_line_relocation"
        if matched_hash != expected_hash:
            raise ValueError(f"Model source differs from the recorded evaluation: {current_path}")
        audit[current_path] = {
            "sha256": current_hash,
            "reference_path": reference_path,
            "reference_sha256": expected_hash,
            "verification": verification,
        }
    return audit


def load_reference(config):
    """Validate complete frame identities against the original published inventory."""
    root = Path(config["reference_evaluation"])
    reference = json.loads((root / "effective_config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    if summary["split"] != "validation" or summary.get("independent_test"):
        raise ValueError("Hybrid diagnosis requires the reused development validation")
    if file_sha256(root / "predictions.npz") != config["reference_predictions_sha256"]:
        raise ValueError("Reference prediction snapshot SHA256 differs from the audited source")
    with np.load(root / "predictions.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in DATA_FIELDS if name in saved}
        baselines = {name: saved[field].copy() for name, field in BASELINE_FIELDS.items()}
    dataset = InsertionEpisodeDataset(data)
    manifest = reference["temporal_split_manifest"]
    expected = set(manifest["splits"]["validation"])
    if set(dataset.episodes) != expected:
        raise ValueError("Snapshot episodes differ from the validation manifest")
    if (
        len(dataset) != config["expected_episodes"]
        or len(data["frame"]) != config["expected_frames"]
        or len(dataset) != summary["episodes"]
        or len(data["frame"]) != summary["frames"]
    ):
        raise ValueError("Snapshot episode/frame counts differ from the requested full validation")
    identity = json.loads(Path(config["data_identity"]).read_text())
    selection = {
        "source_dataset_sha256": identity["source_dataset_sha256"],
        "inventory_sha256": identity["cache_inventory_sha256"],
        "empty_inputs": identity["empty_inputs"],
    }
    if canonical_hash(selection) != identity["dataset_sha256"]:
        raise ValueError("Published cache selection identity checksum differs")
    validation_ids = {
        row["id"]: row["sha256"]
        for row in identity["frames"]
        if row["id"].rsplit("/", 1)[0] in expected
    }
    if set(data["sample_ids"]) != set(validation_ids):
        raise ValueError("Snapshot frame IDs differ from the published frame inventory")
    np.testing.assert_array_equal(baselines["gnn"], data["prediction"])
    for name, prediction in baselines.items():
        if prediction.shape != (len(data["frame"]), 21) or not np.isfinite(prediction).all():
            raise ValueError(f"{name}: expected a finite frame-aligned Object21 snapshot")
    # The 17D contract is a named subset, not a substitute for Mamba's 25D input.
    np.testing.assert_array_equal(data["features"], data["pose_quality"][:, 8:])
    frame_manifest = [
        dict(
            sample_id=str(sample),
            episode=str(episode),
            frame=int(frame),
            parent=manifest["parent_by_episode"][str(episode)],
            raw_sha256=validation_ids[str(sample)],
        )
        for sample, episode, frame in zip(data["sample_ids"], data["episode"], data["frame"])
    ]
    return data, baselines, reference, summary, provenance, identity, frame_manifest


def restore_branches(config, reference, provenance, identity, device):
    """Restore exact weights and each trained schema; explicitly select runtime backend."""
    payloads = {}
    paths = {name: Path(config[name + "_checkpoint"]) for name in ("gnn", "mamba", "gate")}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected = {
        "gnn": reference["source_checkpoint_sha256"],
        "mamba": provenance["checkpoints"]["primary"]["sha256"],
        "gate": provenance["checkpoints"]["learned_gate"]["sha256"],
    }
    if hashes != expected:
        raise ValueError("One or more checkpoints differ from the recorded baseline evaluation")
    for name, path in paths.items():
        payloads[name] = torch.load(path, weights_only=True, map_location="cpu")
        if payloads[name]["dataset_sha256"] != identity["dataset_sha256"]:
            raise ValueError(f"{name}: checkpoint and published dataset identity differ")
    mamba_payload, gate_payload = payloads["mamba"], payloads["gate"]
    for payload in (mamba_payload, gate_payload):
        check_development_provenance(payload, reference, identity)
    trained = mamba_payload["config"]
    if (
        mamba_payload.get("kind") != "temporal_insertion"
        or mamba_payload["epoch"] != 19
        or trained["model"]["architecture"] != "mamba"
    ):
        raise ValueError("Expected the selected epoch-19 Mamba temporal_insertion checkpoint")
    if (
        trained["feature_names"] != list(feature_names(trained["with_robot"]))
        or trained["observation_contract"] != INSERTION_CONTRACT
    ):
        raise ValueError("Mamba checkpoint input feature contract is unsupported")
    if trained["normalization"]["source_split"] != "train":
        raise ValueError("Mamba normalization must be fitted on train only")
    runtime = deepcopy(trained)
    runtime["model"]["backend"] = config["mamba_backend"]
    mamba = build_insertion_model(runtime).to(device)
    mamba.load_state_dict(mamba_payload["model"], strict=True)
    for buffer, field in ((mamba.input_mean, "mean"), (mamba.input_std, "std")):
        torch.testing.assert_close(buffer.cpu(), torch.tensor(trained["normalization"][field]))
    if gate_payload.get("kind") != "learned_temporal_gate" or gate_payload["config"][
        "feature_names"
    ] != list(QUALITY_NAMES):
        raise ValueError("Expected the existing Learned Gate 17D quality contract")
    gate = LearnedTemporalGate(**gate_payload["config"]["gate"]).to(device)
    gate.load_state_dict(gate_payload["gate"], strict=True)
    source_audit = verify_model_sources(reference["source_sha256"])
    audit = {
        "checkpoints": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashes[name],
                "kind": payloads[name]["kind"],
                "epoch": payloads[name].get("epoch"),
                "config": payloads[name]["config"],
            }
            for name, path in paths.items()
        },
        "mamba_training_backend": trained["model"]["backend"],
        "mamba_runtime_backend": runtime["model"]["backend"],
        "device": str(device),
        "torch_version": torch.__version__,
        "mamba_parameters": sum(value.numel() for value in mamba.parameters()),
        "gate_parameters": sum(value.numel() for value in gate.parameters()),
        "trainable_parameters": 0,
        "optimizer_created": False,
        "backpropagation": False,
        "mamba_relative_vector": "pose-derived; no separate relative head",
        "gnn_execution": "reuse identified original Frozen GNN prediction cache",
        "normalization": "Mamba train mean/std; Gate metre coordinates / saved position_scale_m",
        "inference_context": "full episode; cold reset at frame 0; no warm-up exclusion",
        "timestamp": "original contiguous saved frame index; no independent timestamp field in snapshot",
        "raw_proprioception": "reuse original checksum-joined observed TCP quaternion and finger qpos",
        "excluded_frames": 0,
        "independent_test": False,
        "unchanged_model_source_sha256": {
            details["reference_path"]: details["reference_sha256"]
            for details in source_audit.values()
        },
        "model_source_compatibility": source_audit,
        "evaluation_source_sha256": {
            name: file_sha256(name)
            for name in (
                "graph/temporal/hybrid.py",
                "evaluation/gnn/temporal/hybrid.py",
                "evaluation/gnn/temporal/hybrid_diagnostics.py",
                "evaluation/gnn/temporal/hybrid_inputs.py",
                "data_collection/temporal/hybrid_features.py",
            )
        },
    }
    return mamba.eval().requires_grad_(False), gate.eval().requires_grad_(False), audit
