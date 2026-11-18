"""Current-frame detector preprocessing and hard routing for Five-mode GNNs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from data_collection.detected_corner_visibility import (
    classify_frame,
    coplanarity_tolerance,
)
from data_collection.episode_dataset import file_sha256
from graph.build_graph import NodeType
from training.detected_corner_gnn import graph_batch
from training.gnn.trainer import build_model
from utils.device_set import set_device


ROUTED_CONDITIONS = ("condition_1", "condition_2")


@dataclass(frozen=True)
class ConditionEvidence:
    """Plane evidence from detector corners; RMS is undefined below four Peg points."""

    peg_plane_rms_m: float | None
    peg_noncoplanar: bool | None
    coplanarity_tolerance_m: float


@dataclass(frozen=True)
class RoutedObject21Prediction:
    """One frame's routing decision and optional TCP-relative Object21 output."""

    condition: str
    selected_model: str | None
    object21: torch.Tensor | None
    peg_corner_count: int
    hole_rim_count: int
    peg_surface_count: int
    condition_evidence: ConditionEvidence
    prediction_status: str


def _as_numpy(value) -> np.ndarray:
    """Detach a small detector tensor without retaining accelerator state."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _float_coordinates(value, name) -> np.ndarray:
    """Validate real coordinates, including their float32 representation for the GNN."""
    try:
        array = _as_numpy(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite real coordinates") from error
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain finite real coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        coordinates = array.astype(np.float32, copy=False)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain finite coordinates representable as float32")
    return coordinates


def _rim_flags(detection, count) -> np.ndarray:
    """Read the detector's post-merge Hole-rim labels in corner order."""
    metadata = detection.metadata
    if not isinstance(metadata, (list, tuple)) or len(metadata) != count:
        raise ValueError("Detector metadata must have one row per merged corner")
    flags = []
    for row in metadata:
        if not isinstance(row, Mapping):
            raise ValueError("Each detector metadata row must be a mapping")
        flag = row.get("contour_hole", False)
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("Detector contour_hole flags must be boolean")
        flags.append(flag)
    return np.asarray(flags, dtype=np.bool_)


def _condition_evidence(corner_xyz, corner_kinds, tolerance_m) -> ConditionEvidence:
    """Expose the same best-fit-plane convention used by the visibility classifier."""
    peg = corner_xyz[corner_kinds == NodeType.PEG].astype(np.float64)
    plane_rms = None
    noncoplanar = None
    if len(peg) >= 4:
        singular_values = np.linalg.svd(peg - peg.mean(axis=0), compute_uv=False)
        plane_rms = float(singular_values[-1] / np.sqrt(len(peg)))
        noncoplanar = plane_rms > tolerance_m
    return ConditionEvidence(plane_rms, noncoplanar, tolerance_m)


def load_surface_model(
    settings: Mapping,
    device: torch.device,
    condition: str,
) -> nn.Module:
    """Load one checksum-pinned Peg-surface GNN checkpoint."""
    if condition not in ROUTED_CONDITIONS:
        raise ValueError(f"Unknown routed condition: {condition}")
    if not isinstance(settings, Mapping) or not {"path", "sha256"} <= settings.keys():
        raise ValueError("Each routed checkpoint needs path and sha256")
    checksum = settings["sha256"]
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)):
        raise ValueError("Each routed checkpoint sha256 must be 64 lowercase hex characters")
    path = Path(settings["path"]).resolve()
    if file_sha256(path) != settings["sha256"]:
        raise ValueError(f"Checkpoint checksum does not match the routing config: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model_config = checkpoint["config"]["model"]
        data_config = checkpoint["config"]["data"]
        state = checkpoint["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Checkpoint lacks the Five-mode model contract: {path}") from error
    if not isinstance(data_config, Mapping) or not isinstance(model_config, Mapping):
        raise ValueError(f"Checkpoint model and data configurations must be mappings: {path}")
    if data_config.get("include_surface") is not True:
        raise ValueError(f"Hard routing requires a Peg-surface checkpoint: {path}")
    if data_config.get("condition") != condition:
        raise ValueError(f"Checkpoint is not trained for routed {condition}: {path}")
    if data_config.get("factory") != "training.gnn.packed_five_mode_data:build_data":
        raise ValueError(f"Hard routing requires the packed Five-mode input contract: {path}")
    if model_config.get("name") != "vertex_edge_gated_mpnn" or "factory" in model_config:
        raise ValueError(f"Hard routing requires a VertexEdgeGatedMPNN checkpoint: {path}")

    model = build_model(model_config, device)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


class FiveModeHardRoutingPipeline:
    """Detect, classify, build one graph, and run the matching C1/C2 expert."""

    def __init__(
        self,
        models,
        *,
        detector=None,
        device="auto",
        coplanarity_tolerance_m=0.001,
    ):
        if not isinstance(models, Mapping) or set(models) != set(ROUTED_CONDITIONS):
            raise ValueError("models must contain exactly condition_1 and condition_2")
        if any(not isinstance(model, nn.Module) for model in models.values()):
            raise TypeError("Every routed model must be a torch.nn.Module")
        if detector is not None and not callable(detector):
            raise TypeError("detector must be callable or None")
        self.device = set_device(device)
        self.models = {
            condition: model.to(self.device).eval().requires_grad_(False)
            for condition, model in models.items()
        }
        self.detector = detector
        self.coplanarity_tolerance_m = coplanarity_tolerance(coplanarity_tolerance_m)

    @torch.inference_mode()
    def predict_detection(self, detection, peg_surface_xyz, tcp_xyz) -> RoutedObject21Prediction:
        """Route one already-detected frame and run at most one expert model.

        ``peg_surface_xyz`` must contain only the current frame's visible,
        depth-valid, non-corner TAPNext++ Peg surface points. Condition selection
        always uses ``detection`` alone.
        """
        corner_xyz = _float_coordinates(detection.xyz, "detection.xyz")
        corner_kinds = _as_numpy(detection.object_id)
        if corner_kinds.ndim != 1:
            raise ValueError("detection.object_id must have shape [N]")
        rim = _rim_flags(detection, len(corner_kinds))
        condition = classify_frame(
            corner_xyz, corner_kinds, rim,
            tolerance_m=self.coplanarity_tolerance_m,
        )
        peg_corner_count = int(np.count_nonzero(corner_kinds == NodeType.PEG))
        hole_rim_count = int(np.count_nonzero((corner_kinds == NodeType.BOX) & rim))
        evidence = _condition_evidence(
            corner_xyz, corner_kinds, self.coplanarity_tolerance_m,
        )

        surface_xyz = _float_coordinates(peg_surface_xyz, "peg_surface_xyz")
        tcp = _float_coordinates(tcp_xyz, "tcp_xyz")
        if surface_xyz.ndim != 2 or surface_xyz.shape[1:] != (3,):
            raise ValueError(
                "peg_surface_xyz must be finite world coordinates with shape [S,3]"
            )
        if tcp.shape != (3,) or not np.isfinite(tcp).all():
            raise ValueError("tcp_xyz must be finite world coordinates with shape [3]")

        if condition == "condition_3":
            return RoutedObject21Prediction(
                condition=condition,
                selected_model=None,
                object21=None,
                peg_corner_count=peg_corner_count,
                hole_rim_count=hole_rim_count,
                peg_surface_count=len(surface_xyz),
                condition_evidence=evidence,
                prediction_status="unavailable",
            )

        nodes = np.concatenate(
            (corner_xyz.astype(np.float32, copy=False), surface_xyz), axis=0,
        )
        kinds = np.concatenate((
            corner_kinds.astype(np.int64, copy=False),
            np.full(len(surface_xyz), NodeType.PEG, dtype=np.int64),
        ))
        xyz_tensor = torch.as_tensor(
            nodes, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        kind_tensor = torch.as_tensor(
            kinds, dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        valid = torch.ones_like(kind_tensor, dtype=torch.bool)
        tcp_tensor = torch.as_tensor(
            tcp, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)
        graph, _ = graph_batch(xyz_tensor, kind_tensor, valid, valid, tcp_tensor)
        graph.validate()

        output = self.models[condition](graph)
        if not isinstance(output, dict) or "object21" not in output:
            raise ValueError(f"{condition} model must return an object21 tensor")
        object21 = output["object21"]
        if (not torch.is_tensor(object21) or object21.shape != (1, 21)
                or not object21.is_floating_point() or not torch.isfinite(object21).all()
                or object21.device != xyz_tensor.device):
            raise ValueError(f"{condition} model returned an invalid Object21 prediction")
        return RoutedObject21Prediction(
            condition=condition,
            selected_model=condition,
            object21=object21[0],
            peg_corner_count=peg_corner_count,
            hole_rim_count=hole_rim_count,
            peg_surface_count=len(surface_xyz),
            condition_evidence=evidence,
            prediction_status="available",
        )

    def predict_rgbd(
        self,
        rgb,
        depth_raw_mm,
        intrinsic_cv,
        extrinsic_cv,
        peg_surface_xyz,
        tcp_xyz,
        *,
        masks=None,
    ) -> RoutedObject21Prediction:
        """Run the current-frame detector once, then classify and route its result."""
        if self.detector is None:
            raise RuntimeError(
                "predict_rgbd requires a detector when the pipeline is constructed"
            )
        detection = self.detector(
            rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv, masks=masks,
        )
        return self.predict_detection(detection, peg_surface_xyz, tcp_xyz)


def build_hard_routing_pipeline(config, *, detector=None) -> FiveModeHardRoutingPipeline:
    """Construct the configured detector and the two checksum-pinned experts."""
    tolerance_m = coplanarity_tolerance(config.get("coplanarity_tolerance_m", 0.001))
    device = set_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config.get("matmul_precision", "high"))
    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(ROUTED_CONDITIONS):
        raise ValueError("checkpoints must contain exactly condition_1 and condition_2")
    models = {
        condition: load_surface_model(checkpoints[condition], device, condition)
        for condition in ROUTED_CONDITIONS
    }
    if detector is None:
        from visual.corner_detection import TorchCornerDetector

        detector = TorchCornerDetector.main(
            device=config.get("detector_device", str(device)),
        )
    return FiveModeHardRoutingPipeline(
        models,
        detector=detector,
        device=device,
        coplanarity_tolerance_m=tolerance_m,
    )


__all__ = [
    "ConditionEvidence",
    "FiveModeHardRoutingPipeline",
    "RoutedObject21Prediction",
    "build_hard_routing_pipeline",
    "load_surface_model",
]
