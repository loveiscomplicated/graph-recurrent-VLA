"""Validate the Graph team's temporal recipes before opening source frames.

Each resolver retains its workflow's existing defaults and output schema. Shared
validation covers frozen inputs and diagnostic settings, not a repository schema.
"""

import importlib.metadata
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from graph.temporal.hole_gru import TemporalHoleGRU
from graph.temporal.insertion import TemporalInsertionEstimator, feature_names
from graph.temporal.pose_gru import MODES, TemporalPoseGRU
from graph.temporal.pose_observations import INSERTION_CONTRACT
from graph.utils.hole_memory import MemorySettings
from training.gnn.temporal.pose_loss import INSERTION_ABLATIONS

LOSS_TERMS = (
    "peg_position",
    "peg_rotation",
    "hole_position",
    "hole_rotation",
    "relative_position",
    "lateral_position",
    "relative_orientation",
)


def resolve_common_config(config):
    """Validate the team-owned recipe before loading any source frames."""
    effective = deepcopy(config)
    defaults = {
        "training": {
            "epochs": 20,
            "lr": 0.001,
            "loss": "huber",
            "huber_delta_m": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "evaluation": {
            "batch_size": 64,
            "visibility_labels": None,
            "bad_error_mm": 10.0,
            "plot_episodes": 3,
            "long_occlusion_frames": 20,
            "latency_warmup_steps": 10,
            "latency_steps": 100,
        },
        "rule_memory": {},
        "output": {"root": "outputs/temporal_gru"},
    }
    effective.setdefault("device", "cpu")
    effective.setdefault("seed", 0)
    if type(effective["seed"]) is not int or not 0 <= effective["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for section, settings in defaults.items():
        effective.setdefault(section, {})
        for name, value in settings.items():
            effective[section].setdefault(name, value)
    for name in ("checkpoint", "split_manifest"):
        path = Path(effective[name]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        effective[name] = str(path)
    cache = effective.get("data", {}).get("cache_dir")
    if not isinstance(cache, str) or not cache:
        raise ValueError("data.cache_dir must point to the extracted published graph cache")
    effective["data"] = {"cache_dir": str(Path(cache).expanduser().resolve())}
    for section, names in (
        ("training", ("epochs",)),
        (
            "evaluation",
            (
                "batch_size",
                "plot_episodes",
                "long_occlusion_frames",
                "latency_warmup_steps",
                "latency_steps",
            ),
        ),
    ):
        for name in names:
            value = effective[section][name]
            if type(value) is not int or value < 1:
                raise ValueError(f"{section}.{name} must be an integer >= 1")
    for section, names in (
        ("training", ("lr", "huber_delta_m", "gradient_clip_norm")),
        ("evaluation", ("bad_error_mm",)),
    ):
        for name in names:
            value = effective[section][name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} must be finite and positive")
    if effective["training"]["loss"] not in ("huber", "l1"):
        raise ValueError("training.loss must be huber or l1")
    labels = effective["evaluation"]["visibility_labels"]
    if labels is not None:
        labels = Path(labels).expanduser().resolve()
        if not labels.is_file():
            raise FileNotFoundError(f"Missing diagnostic visibility_labels: {labels}")
        effective["evaluation"]["visibility_labels"] = str(labels)
    effective["output"]["root"] = str(Path(effective["output"]["root"]).expanduser().resolve())
    MemorySettings(**effective["rule_memory"])
    return effective


def resolve_gru_config(config):
    """Validate Hole-only GRU settings without consuming the training RNG."""
    effective = resolve_common_config(config)
    settings = effective.setdefault("gru", {})
    for name, value in {"hidden": 64, "layers": 1, "position_scale_m": 0.1}.items():
        settings.setdefault(name, value)
    with torch.random.fork_rng(devices=[]):
        TemporalHoleGRU(**settings)
    return effective


def resolve_pose_defaults(config):
    """Resolve shared pose data, objectives and diagnostic settings."""
    common = deepcopy(config)
    requested_gru = common.pop("gru", {})
    common.setdefault("output", {}).setdefault("root", "outputs/temporal_pose")
    effective = resolve_common_config(common)
    effective["gru"] = {
        "hidden": 128,
        "projection": 64,
        "head_hidden": 64,
        "layers": 1,
        "position_mode": "direct",
        "position_scale_m": 0.1,
        **requested_gru,
    }
    effective.setdefault("mode", "joint_peg_hole")
    raw = config.get("data", {}).get("raw_root")
    effective["data"]["raw_root"] = (
        str(Path(raw).expanduser().resolve()) if raw is not None else None
    )
    for name in ("independent_cache_dir", "independent_raw_root"):
        value = config.get("data", {}).get(name)
        effective["data"][name] = (
            str(Path(value).expanduser().resolve()) if value is not None else None
        )
    effective["with_robot"] = raw is not None
    effective["training"].setdefault("episode_batch_size", 4)
    batch_size = effective["training"]["episode_batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("training.episode_batch_size must be an integer >= 1")
    loss = effective.setdefault("loss", {})
    loss.setdefault("position_unit_m", 0.01)
    loss.setdefault("rotation_unit_rad", 0.1)
    weights = loss.setdefault("weights", {})
    for name in ("peg_position", "peg_rotation", "hole_position", "hole_rotation"):
        weights.setdefault(name, 1.0)
    for name in ("position_unit_m", "rotation_unit_rad"):
        if type(loss[name]) not in (float, int) or not np.isfinite(loss[name]) or loss[name] <= 0:
            raise ValueError(f"loss.{name} must be finite and positive")
    expected_weights = {"peg_position", "peg_rotation", "hole_position", "hole_rotation"}
    if "ablation" in loss:
        if loss["ablation"] not in INSERTION_ABLATIONS:
            raise ValueError(f"loss.ablation must be one of {tuple(INSERTION_ABLATIONS)}")
        loss.setdefault("rotation_objective", "pose_surrogate")
        if loss["rotation_objective"] not in ("pose_surrogate", "geodesic_huber"):
            raise ValueError("loss.rotation_objective must be pose_surrogate or geodesic_huber")
        for name, value in (
            ("relative_position", 0.25),
            ("lateral_position", 0.25),
            ("relative_orientation", 0.1),
        ):
            weights.setdefault(name, value)
            expected_weights.add(name)
    if set(weights) != expected_weights:
        raise ValueError(f"loss.weights must contain exactly {sorted(expected_weights)}")
    if any(
        type(value) not in (float, int) or not np.isfinite(value) or value < 0
        for value in weights.values()
    ):
        raise ValueError("loss.weights must be finite nonnegative numbers")
    settings = effective["evaluation"]
    for name, value in (
        ("motion_threshold_m", 0.001),
        ("motion_threshold_rad", 0.01),
        ("recovery_rotation_deg", 10.0),
        ("lag_max_frames", 10),
    ):
        settings.setdefault(name, value)
        if (
            type(settings[name]) not in (int, float)
            or not np.isfinite(settings[name])
            or settings[name] <= 0
        ):
            raise ValueError(f"evaluation.{name} must be finite and positive")
    if type(settings["lag_max_frames"]) is not int:
        raise ValueError("evaluation.lag_max_frames must be an integer")
    for name, value in (
        ("co_motion_tolerance_m", 0.002),
        ("direction_change_cosine", 0.0),
        ("alignment_axial_min_m", -0.05),
        ("alignment_axial_max_m", 0.02),
        ("alignment_lateral_m", 0.01),
    ):
        settings.setdefault(name, value)
        if type(settings[name]) not in (int, float) or not np.isfinite(settings[name]):
            raise ValueError(f"evaluation.{name} must be finite")
    if not -1 <= settings["direction_change_cosine"] <= 1:
        raise ValueError("evaluation.direction_change_cosine must be in [-1,1]")
    if (
        settings["co_motion_tolerance_m"] <= 0
        or settings["alignment_lateral_m"] <= 0
        or settings["alignment_axial_min_m"] > settings["alignment_axial_max_m"]
    ):
        raise ValueError("Invalid co-motion tolerance or insertion-alignment diagnostic bounds")
    if "ablation" in loss:
        loss["active_terms"] = [
            name + "_" + term
            for name in ("peg", "hole")
            if effective["mode"] in ("joint_peg_hole", name + "_only")
            for term in ("position", "rotation")
        ] + list(INSERTION_ABLATIONS[loss["ablation"]])
        loss["effective_weights"] = {
            name: weights[name] if name in loss["active_terms"] else 0.0 for name in weights
        }
        selection = effective.setdefault("selection", {})
        selection.setdefault(
            "components",
            {
                "relative_3d_mm_p90": {"scale": 3.0, "weight": 1.0},
                "lateral_mm_p90": {"scale": 3.0, "weight": 1.0},
                "peg_moving_position_mm_mean": {"scale": 3.0, "weight": 1.0},
                "peg_moving_rotation_deg_mean": {"scale": 5.0, "weight": 0.25},
                "peg_position_mm_mean": {"scale": 3.0, "weight": 0.25},
                "hole_position_mm_mean": {"scale": 3.0, "weight": 0.25},
                "peg_rotation_deg_mean": {"scale": 5.0, "weight": 0.1},
                "hole_rotation_deg_mean": {"scale": 5.0, "weight": 0.1},
            },
        )
        for name, specification in selection["components"].items():
            if name not in {
                "relative_3d_mm_p90",
                "lateral_mm_p90",
                "peg_moving_position_mm_mean",
                "peg_moving_rotation_deg_mean",
                "peg_position_mm_mean",
                "hole_position_mm_mean",
                "peg_rotation_deg_mean",
                "hole_rotation_deg_mean",
            }:
                raise ValueError(f"Unknown checkpoint selection metric: {name}")
            for field in ("scale", "weight"):
                value = specification[field]
                if (
                    type(value) not in (int, float)
                    or not np.isfinite(value)
                    or value < 0
                    or (field == "scale" and value == 0)
                ):
                    raise ValueError(f"selection {name}.{field} must be finite and valid")
        if not selection["components"] or not any(
            value["weight"] > 0 for value in selection["components"].values()
        ):
            raise ValueError("Checkpoint selection needs at least one positive component weight")
    settings.setdefault("phase_labels", None)
    comparison = effective.setdefault("comparison", {})
    comparison.setdefault("learned_gate", None)
    checkpoints = comparison.setdefault("gru_checkpoints", {})
    for mode in ("hole_only", "peg_only", "joint_peg_hole"):
        checkpoints.setdefault(mode, None)
    return effective


def resolve_pose_config(config):
    """Validate the Pose GRU after resolving its data and evaluation settings."""
    effective = resolve_pose_defaults(config)
    with torch.random.fork_rng(devices=[]):
        TemporalPoseGRU(
            mode=effective["mode"], with_robot=effective["with_robot"], **effective["gru"]
        )
    return effective


def dependency_versions():
    """Record installed versions without importing optional CUDA extensions."""
    versions = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
    }
    distributions = (
        "mamba-ssm",
        "causal-conv1d",
        "triton",
        "transformers",
        "einops",
        "numpy",
        "matplotlib",
        "mani_skill",
        "sapien",
    )
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def resolve_insertion_config(config):
    """Reuse existing source/proprioception validation; add explicit insertion choices."""
    common = deepcopy(config)
    requested_loss = common.pop("loss", {})
    effective = resolve_pose_defaults(common)
    effective["data"]["independent_cache_dir"] = config.get("data", {}).get("independent_cache_dir")
    effective["data"]["independent_raw_root"] = config.get("data", {}).get("independent_raw_root")
    for name in ("independent_cache_dir", "independent_raw_root"):
        value = effective["data"][name]
        if value is not None:
            effective["data"][name] = str(Path(value).expanduser().resolve())
    require_robot = config.get("data", {}).get("require_robot", True)
    if type(require_robot) is not bool:
        raise ValueError("data.require_robot must be boolean")
    if require_robot and not effective["with_robot"]:
        raise ValueError(
            "Full robot features require data.raw_root; explicitly set require_robot=false for cache-only ablation"
        )
    effective["data"]["require_robot"] = require_robot
    model = effective.setdefault("model", {})
    defaults = dict(
        architecture="mamba",
        backend="torch_reference",
        d_model=128,
        n_layers=1,
        d_state=16,
        d_conv=4,
        expand=2,
        head_hidden=64,
        position_mode="residual",
        position_scale_m=0.01,
    )
    if model.get("architecture") == "transformer":
        defaults.update(n_heads=4, feedforward_dim=256, dropout=0.0)
    if set(model) - set(defaults):
        raise ValueError(f"Unknown insertion model settings: {set(model) - set(defaults)}")
    for name, value in defaults.items():
        model.setdefault(name, value)
    if model["architecture"] in ("gru", "transformer") and model["backend"] != "torch_reference":
        raise ValueError("GRU and Transformer require backend=torch_reference")
    loss = dict(
        position_unit_m=0.01,
        rotation_unit_rad=0.1,
        ablation="pose_relative_full",
        rotation_objective="geodesic_huber",
        weights={name: 1.0 for name in LOSS_TERMS},
    )
    if set(requested_loss) - set(loss):
        raise ValueError("Unknown insertion loss settings")
    loss.update({name: value for name, value in requested_loss.items() if name != "weights"})
    loss["weights"].update(requested_loss.get("weights", {}))
    if loss["ablation"] not in INSERTION_ABLATIONS:
        raise ValueError(f"loss.ablation must be one of {tuple(INSERTION_ABLATIONS)}")
    if loss["rotation_objective"] != "geodesic_huber":
        raise ValueError("Insertion workflow requires loss.rotation_objective=geodesic_huber")
    for name in ("position_unit_m", "rotation_unit_rad"):
        value = loss[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"loss.{name} must be finite and positive")
    if set(loss["weights"]) != set(LOSS_TERMS) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in loss["weights"].values()
    ):
        raise ValueError(f"loss.weights must contain finite nonnegative values for {LOSS_TERMS}")
    active = [
        obj + "_" + term for obj in MODES[effective["mode"]] for term in ("position", "rotation")
    ]
    active.extend(INSERTION_ABLATIONS[loss["ablation"]])
    if not any(loss["weights"][name] > 0 for name in active):
        raise ValueError("At least one active loss weight must be positive")
    effective["loss"] = loss
    effective["training"].setdefault("sequence_length", 64)
    effective["training"].setdefault("context_policy", "reset_windows")
    effective["training"].setdefault("cuda_memory_fraction", None)
    early = effective["training"].get("early_stopping")
    if early is not None:
        if not isinstance(early, dict) or set(early) != {"patience", "min_epochs", "min_delta"}:
            raise ValueError("training.early_stopping requires patience, min_epochs, and min_delta")
        for name in ("patience", "min_epochs"):
            if type(early[name]) is not int or early[name] < 1:
                raise ValueError(f"training.early_stopping.{name} must be a positive integer")
        delta = early["min_delta"]
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
            raise ValueError("training.early_stopping.min_delta must be finite and nonnegative")
    memory_fraction = effective["training"]["cuda_memory_fraction"]
    if memory_fraction is not None and (
        type(memory_fraction) not in (int, float)
        or not math.isfinite(memory_fraction)
        or not 0 < memory_fraction <= 1
    ):
        raise ValueError("training.cuda_memory_fraction must be null or a number in (0,1]")
    if effective["training"]["context_policy"] != "reset_windows":
        raise ValueError("Only explicit reset_windows context policy is implemented")
    length = effective["training"]["sequence_length"]
    if type(length) is not int or length < 1:
        raise ValueError("training.sequence_length must be a positive integer")
    effective["evaluation"].setdefault("split", "validation")
    if effective["evaluation"]["split"] not in ("validation", "test"):
        raise ValueError("evaluation.split must be validation or test")
    effective["evaluation"].setdefault("benchmark_training", False)
    if type(effective["evaluation"]["benchmark_training"]) is not bool:
        raise ValueError("evaluation.benchmark_training must be boolean")
    effective["comparison"] = deepcopy(config.get("comparison", {}))
    effective["comparison"].setdefault("checkpoints", {})
    effective["comparison"].setdefault("learned_gate", None)
    effective["feature_names"] = list(feature_names(effective["with_robot"]))
    effective["observation_contract"] = INSERTION_CONTRACT
    effective["dependencies"] = dependency_versions()
    effective["implementation_reference"] = (
        "state-spaces/mamba v2.2.6 Mamba-1" if model["architecture"] == "mamba"
        else f"PyTorch {model['architecture']}"
    )
    # Validate before loading datasets, without advancing the training RNG.
    with torch.random.fork_rng(devices=[]):
        TemporalInsertionEstimator(
            mode=effective["mode"], with_robot=effective["with_robot"], **effective["model"]
        )
    return effective


def resolve_joint_job_config(config):
    """Combine a model-specific base with the four seed-specific settings.

    Base paths are relative to the repository root, like other Graph config paths.
    The returned dictionary is the complete config consumed and logged by training.
    """
    override_names = ("seed", "checkpoint", "temporal_checkpoint", "output")
    expected_names = {"trainer", "base_config", *override_names}
    if set(config) != expected_names:
        raise ValueError(f"Joint job config must contain exactly: {', '.join(sorted(expected_names))}")

    base_path = Path(config["base_config"])
    with base_path.open(encoding="utf-8") as file:
        base = yaml.safe_load(file)
    if not isinstance(base, dict):
        raise ValueError(f"Joint base config must be a YAML mapping: {base_path}")
    if base.get("trainer") != "training.gnn.temporal.joint_finetuning:train":
        raise ValueError(f"Joint base config has an unexpected trainer: {base_path}")
    if any(name in base for name in override_names):
        raise ValueError(f"Joint base config must not contain seed-specific settings: {base_path}")

    effective = deepcopy(base)
    for name in override_names:
        effective[name] = deepcopy(config[name])
    return effective
