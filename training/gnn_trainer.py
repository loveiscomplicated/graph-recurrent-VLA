"""Initialize a config-selected Graph model and tracking run without training."""

from copy import deepcopy
import math

import torch
from torch import nn

from graph.basic_gnn import BasicGNN
from graph.gated_mpnn import VertexEdgeGatedMPNN
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def validate_integer_settings(settings: dict, minimums: dict[str, int]) -> None:
    """Check constructor dimensions before constructing a model."""
    for name, minimum in minimums.items():
        value = settings.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"model.{name} must be an integer >= {minimum}")


def validate_positive_scale(name: str, value: object) -> None:
    """Reject nonnumeric, nonfinite and nonpositive physical scales."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def resolve_config(config: dict) -> dict:
    """Validate the selected model and record all defaults in a config copy."""
    effective = deepcopy(config)
    effective.setdefault("seed", 0)
    effective.setdefault("device", "auto")
    if type(effective["seed"]) is not int or not 0 <= effective["seed"] < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")

    model_config = effective.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("model must be a YAML mapping of model settings")

    model_name = model_config.setdefault("name", "basic_gnn")
    if model_name == "basic_gnn":
        minimums = dict(n_layers=1, in_dim=1, h_dim=1, out_dim=1, edge_dim=1)
        defaults = dict(dropout=0.1, use_confidence=False)
    elif model_name == "vertex_edge_gated_mpnn":
        minimums = dict(num_semantic_ids=1, hidden=8, layers=1, edge_embedding_dim=1)
        defaults = dict(
            hidden=128,
            layers=3,
            edge_embedding_dim=16,
            dropout=0.0,
            use_semantic_ids=False,
            use_confidence=False,
            coord_scale_m=0.05,
            size_scales_m=[0.04, 0.01, 0.01, 0.02],
        )
    else:
        raise ValueError(
            "model.name must be 'basic_gnn' or 'vertex_edge_gated_mpnn'; "
            f"got {model_name!r}"
        )

    unknown = set(model_config) - set(minimums) - set(defaults) - {"name"}
    if unknown:
        raise ValueError(f"Unknown {model_name} model settings: {sorted(unknown)}")
    for name, value in defaults.items():
        model_config.setdefault(name, value)
    validate_integer_settings(model_config, minimums)

    dropout = model_config["dropout"]
    if (type(dropout) not in (int, float) or not math.isfinite(dropout)
            or not 0 <= dropout < 1):
        raise ValueError("model.dropout must be a finite number in [0, 1)")

    for name in ("use_semantic_ids", "use_confidence"):
        if name in model_config and type(model_config[name]) is not bool:
            raise ValueError(f"model.{name} must be bool")

    if model_name == "vertex_edge_gated_mpnn":
        validate_positive_scale("model.coord_scale_m", model_config["coord_scale_m"])
        scales = model_config["size_scales_m"]
        if not isinstance(scales, (list, tuple)) or len(scales) != 4:
            raise ValueError("model.size_scales_m must contain exactly four positive scales")
        for index, scale in enumerate(scales):
            validate_positive_scale(f"model.size_scales_m[{index}]", scale)

    return effective


def build_model(model_config: dict, device: torch.device) -> nn.Module:
    """Construct a model from resolved settings without changing their contents."""
    constructor_settings = dict(model_config)
    model_name = constructor_settings.pop("name", "basic_gnn")
    if model_name == "basic_gnn":
        model = BasicGNN(**constructor_settings)
    elif model_name == "vertex_edge_gated_mpnn":
        model = VertexEdgeGatedMPNN(**constructor_settings)
    else:
        raise ValueError(f"Unsupported model.name: {model_name!r}")
    return model.to(device)


def train(config: dict) -> None:
    """Initialize model, device and logger; data and optimization remain TODOs."""
    effective = resolve_config(config)
    device = set_device(effective["device"])
    torch.manual_seed(effective["seed"])
    model_name = effective["model"]["name"]

    with ExperimentLogger(
        team="graph",
        config=effective,
        seed=effective["seed"],
        model=model_name,
        job_type="initialize",
    ) as experiment:
        model = build_model(effective["model"], device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        experiment.record(
            initialization_only=True,
            device=str(device),
            parameter_count=parameter_count,
        )
        print(f"{type(model).__name__} initialized on {device} ({parameter_count:,} parameters)")
        print(f"Experiment: {experiment.directory}")
        print("Initialization only; no training has been run.")

        # TODO: Prepare team-selected data and define the training objective.
        # TODO: Add optimizer, training/validation, and checkpoint registration.
