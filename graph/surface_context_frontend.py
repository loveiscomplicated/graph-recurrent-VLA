"""Online frozen Surface-Context Tracker--GNN inference for recurrent control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.corner_association import CornerMatcher
from graph.frozen_tracker_graph import build_tracker_graph
from graph.temporal.causal_transformer import sinusoidal_positions
from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker
from training.gnn.trainer import build_model
from visual.corner_detection import TorchCornerDetector


TRACKER_SHA256 = "f98c4ef1c77d25fa059e9f9cf048fb1f190bcf74302a0fe1367bfe1341236afc"
GNN_SHA256 = "0f4d9d6fd6ce615125698e243b259fad92557a3f5c5848aafe24a85277720241"
SLOT_COUNT = 20
SURFACE_POINT_COUNT = 32


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def detector_families(result) -> np.ndarray:
    """Map detector candidates to Peg, Box exterior, and Hole-rim families."""
    object_ids = result.object_id.detach().cpu().numpy()
    hole_hint = np.asarray(
        [
            bool(metadata.get("contour_hole", False))
            or metadata.get("method") == "box_opening"
            for metadata in result.metadata
        ],
        dtype=bool,
    )
    return np.where(object_ids == 0, 0, np.where(hole_hint, 2, 1)).astype(np.int64)


def camera_to_world(camera_xyz: np.ndarray, extrinsic_cv: np.ndarray) -> np.ndarray:
    """Transform row-vector CV camera coordinates to world coordinates."""
    camera_xyz = np.asarray(camera_xyz, dtype=np.float32)
    extrinsic_cv = np.asarray(extrinsic_cv, dtype=np.float32)
    if camera_xyz.ndim != 2 or camera_xyz.shape[1] != 3 or extrinsic_cv.shape != (3, 4):
        raise ValueError("Expected camera_xyz [N,3] and extrinsic_cv [3,4]")
    rotation = extrinsic_cv[:, :3]
    translation = extrinsic_cv[:, 3]
    return ((camera_xyz - translation) @ rotation).astype(np.float32)


@dataclass
class SurfaceFrontendFrame:
    """One causal frontend result and its exportable intermediate arrays."""

    object21: np.ndarray
    detected: np.ndarray
    observed: np.ndarray
    surface_xyz: np.ndarray
    surface_valid: np.ndarray
    node_valid: np.ndarray
    endpoint_residual: np.ndarray
    counts: np.ndarray
    timing_ms: np.ndarray


class SurfaceTrackerStream:
    """Streaming adapter that is numerically equivalent to full causal inference."""

    def __init__(
        self,
        model: SurfaceSpatialCornerTracker,
        fallback: np.ndarray,
        device: torch.device,
    ):
        fallback = np.asarray(fallback, dtype=np.float32)
        if fallback.shape != (SLOT_COUNT, 3) or not np.isfinite(fallback).all():
            raise ValueError("Surface tracker fallback must be finite [20,3]")
        self.model = model
        self.fallback = fallback
        self.surface_center = fallback[:8].mean(axis=0)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.frame_index = 0
        self.last_observed_frame = np.full(SLOT_COUNT, -1, dtype=np.int64)
        self.anchor = self.fallback.copy()
        width = self.model.ids.embedding_dim
        self.histories = [
            torch.empty((SLOT_COUNT, 0, width), device=self.device)
            for _ in self.model.temporal
        ]

    def _corner_features(
        self, detected: np.ndarray, observed: np.ndarray, tcp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.last_observed_frame[observed] = self.frame_index
        self.anchor[observed] = detected[observed]
        seen = self.last_observed_frame >= 0
        age = self.frame_index - self.last_observed_frame
        features = np.concatenate(
            (
                (self.anchor - self.fallback) / 0.1,
                (tcp - self.anchor) / 0.1,
                observed[:, None],
                seen[:, None],
                (np.minimum(age, 40) / 40)[:, None],
                np.full((SLOT_COUNT, 1), self.frame_index / 200, dtype=np.float32),
            ),
            axis=-1,
        ).astype(np.float32)
        return features, seen

    @torch.inference_mode()
    def step(
        self,
        detected: np.ndarray,
        observed: np.ndarray,
        tcp: np.ndarray,
        surface_xyz: np.ndarray,
        surface_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complete twenty slots from the current measurements and surface points."""
        detected = np.asarray(detected, dtype=np.float32)
        observed = np.asarray(observed, dtype=bool)
        tcp = np.asarray(tcp, dtype=np.float32)
        surface_xyz = np.asarray(surface_xyz, dtype=np.float32)
        surface_valid = np.asarray(surface_valid, dtype=bool)
        expected = (
            detected.shape == (SLOT_COUNT, 3)
            and observed.shape == (SLOT_COUNT,)
            and tcp.shape == (3,)
            and surface_xyz.shape == (SURFACE_POINT_COUNT, 3)
            and surface_valid.shape == (SURFACE_POINT_COUNT,)
        )
        if not expected or not np.isfinite(detected).all() or not np.isfinite(surface_xyz).all():
            raise ValueError("Invalid Surface tracker step arrays")

        features, seen = self._corner_features(detected, observed, tcp)
        model = self.model
        hidden = model.input(torch.as_tensor(features, device=self.device)) + model.ids.weight
        hidden = hidden + sinusoidal_positions(
            1,
            hidden.shape[-1],
            offset=self.frame_index,
            device=self.device,
            dtype=hidden.dtype,
        )
        hidden, self.histories[0] = model.temporal[0].step(
            hidden[:, None], self.histories[0]
        )
        queries = model.spatial_norm(hidden[:, 0])[None]
        surface_features = np.concatenate(
            (
                (surface_xyz - self.surface_center) / 0.1,
                (tcp - surface_xyz) / 0.1,
            ),
            axis=-1,
        ).astype(np.float32)
        surface_features[~surface_valid] = 0
        surfaces = model.surface_input(
            torch.as_tensor(surface_features, device=self.device)
        )[None]
        keys = torch.cat((queries, surfaces), dim=1)
        peg_queries = torch.arange(SLOT_COUNT, device=self.device) < 8
        allowed = torch.as_tensor(surface_valid, device=self.device)
        surface_blocked = ~(peg_queries[None, :, None] & allowed[None, None, :])
        blocked = torch.cat((model.spatial_mask[None], surface_blocked), dim=-1)
        blocked = blocked.repeat_interleave(model.heads, dim=0)
        update, _ = model.spatial(
            queries, keys, keys, attn_mask=blocked, need_weights=False
        )
        hidden = hidden + update.transpose(0, 1)
        hidden, self.histories[1] = model.temporal[1].step(
            hidden, self.histories[1]
        )
        prediction = self.anchor + model.output(hidden[:, 0]).float().cpu().numpy() * 0.1
        prediction[observed] = detected[observed]
        if not np.isfinite(prediction).all():
            raise ValueError("Surface tracker returned nonfinite coordinates")
        self.frame_index += 1
        return prediction.astype(np.float32), seen


class SurfaceContextFrontend:
    """Detector, physical association, TAPNext++, Surface tracker and Object21 GNN."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = torch.device(config["device"])
        tracker_path = Path(config["tracker"]["path"])
        gnn_path = Path(config["gnn"]["path"])
        self._verify_checkpoint(tracker_path, config["tracker"].get("sha256", TRACKER_SHA256), "tracker")
        self._verify_checkpoint(gnn_path, config["gnn"].get("sha256", GNN_SHA256), "gnn")

        tracker_checkpoint = torch.load(tracker_path, map_location="cpu", weights_only=False)
        if (
            tracker_checkpoint.get("epoch") != 443
            or tracker_checkpoint["config"]["training"].get("supervision")
            != "after_first_observation"
        ):
            raise ValueError("Expected the selected epoch-443 observed-only Surface tracker")
        tracker = SurfaceSpatialCornerTracker(
            **tracker_checkpoint["config"]["model"]
        ).to(self.device).eval().requires_grad_(False)
        tracker.load_state_dict(tracker_checkpoint["model"], strict=True)
        fallback = _numpy(tracker_checkpoint["fallback"])
        self.tracker = SurfaceTrackerStream(tracker, fallback, self.device)

        gnn_checkpoint = torch.load(gnn_path, map_location="cpu", weights_only=False)
        gnn_tracker_hash = (
            gnn_checkpoint.get("config", {}).get("data", {}).get("tracker_sha256")
        )
        if gnn_tracker_hash != config["tracker"].get("sha256", TRACKER_SHA256):
            raise ValueError("Frozen GNN was not trained from the configured Surface tracker")
        self.gnn = build_model(gnn_checkpoint["config"]["model"], self.device)
        self.gnn.load_state_dict(gnn_checkpoint["model"], strict=True)
        self.gnn.eval().requires_grad_(False)

        detector_config = config.get("detector", {})
        self.detector = TorchCornerDetector.main(
            device=detector_config.get("device", config["device"]),
            optimized=detector_config.get("optimized", True),
        )
        from evaluation.tapnextpp_surface_tracking import load_tracker

        self.tap_config = {
            "tap_input_resolution": 256,
            "tap_cuda_autocast": False,
            "tap_cache_cuda_autocast": False,
            "tap_mps_fp16_vit": False,
            "tap_mps_fp16_recurrent_mlp": False,
            "tap_adaptive_crop": False,
            "tap_cuda_graph": False,
            **config["tapnextpp"],
        }
        self.tap = load_tracker(self.tap_config)
        self.reset()

    @staticmethod
    def _verify_checkpoint(path: Path, expected: str, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {label} checkpoint not found: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen {label} checkpoint SHA-256 differs: expected {expected}, got {actual}"
            )

    def reset(self) -> None:
        self.matcher = CornerMatcher()
        self.tracker.reset()
        self.tap_state = None
        self.query = None

    @staticmethod
    def observation_arrays(observation: dict[str, Any]):
        """Read the native Main camera arrays from a ManiSkill observation."""
        sensor = observation["sensor_data"]["base_camera"]
        parameters = observation["sensor_param"]["base_camera"]
        return (
            _numpy(sensor["rgb"])[0],
            _numpy(sensor["depth"])[0].squeeze(),
            _numpy(parameters["intrinsic_cv"])[0],
            _numpy(parameters["extrinsic_cv"])[0],
        )

    def _surface_points(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        peg_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        from evaluation.tapnextpp_surface_tracking import (
            choose_points,
            lift,
            track_surface_frame,
        )

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.query is None:
            self.query = choose_points(peg_mask, depth, SURFACE_POINT_COUNT)
        tracked, visible, self.tap_state = track_surface_frame(
            self.tap,
            self.tap_config,
            bgr,
            self.query if self.tap_state is None else None,
            self.tap_state,
            observation_mask=peg_mask,
        )
        valid_indices, camera_xyz = lift(
            tracked, depth, peg_mask, intrinsic, visible
        )
        dense = np.zeros((SURFACE_POINT_COUNT, 3), dtype=np.float32)
        valid = np.zeros(SURFACE_POINT_COUNT, dtype=bool)
        dense[valid_indices] = camera_to_world(camera_xyz, extrinsic)
        valid[valid_indices] = True
        return dense, valid

    @torch.inference_mode()
    def step_arrays(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        tcp: np.ndarray,
    ) -> SurfaceFrontendFrame:
        """Run one causal frame from raw RGB-D and TCP world coordinates."""
        start = perf_counter()
        result = self.detector(rgb, depth, intrinsic, extrinsic)
        points = result.xyz.detach().cpu().numpy().astype(np.float32)
        assignment = self.matcher.step(points, detector_families(result), tcp)
        observed = assignment >= 0
        detected = np.zeros((SLOT_COUNT, 3), dtype=np.float32)
        detected[observed] = points[assignment[observed]]
        after_detector = perf_counter()

        peg_mask = result.masks[0].detach().cpu().numpy().astype(bool)
        surface_xyz, surface_valid = self._surface_points(
            rgb, depth, intrinsic, extrinsic, peg_mask
        )
        if self.device.type == "cuda":
            # Match the selected model's published evaluation: tracker FP32 with
            # TF32 disabled, followed by GNN FP32 with TF32 enabled.
            torch.backends.cuda.matmul.allow_tf32 = False
        tracked, node_valid = self.tracker.step(
            detected, observed, tcp, surface_xyz, surface_valid
        )
        after_tracker = perf_counter()

        graph, flags = build_tracker_graph(
            torch.as_tensor(tracked[None], device=self.device),
            torch.as_tensor(observed[None], device=self.device),
            torch.as_tensor(node_valid[None], device=self.device),
            torch.as_tensor(np.asarray(tcp, dtype=np.float32)[None], device=self.device),
        )
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        output = self.gnn(graph, flags)
        object21 = output["object21"][0].float().cpu().numpy()
        endpoint = output.get("endpoint_residual")
        endpoint_residual = (
            np.zeros(3, dtype=np.float32)
            if endpoint is None
            else endpoint[0].float().cpu().numpy().astype(np.float32)
        )
        after_gnn = perf_counter()
        counts = np.asarray((observed[:8].sum(), observed[8:].sum()), dtype=np.int16)
        timing = np.asarray(
            (
                (after_detector - start) * 1000,
                (after_tracker - after_detector) * 1000,
                (after_gnn - after_tracker) * 1000,
            ),
            dtype=np.float32,
        )
        return SurfaceFrontendFrame(
            object21=object21.astype(np.float32),
            detected=detected,
            observed=observed,
            surface_xyz=surface_xyz,
            surface_valid=surface_valid,
            node_valid=node_valid,
            endpoint_residual=endpoint_residual,
            counts=counts,
            timing_ms=timing,
        )

    def step(self, observation: dict[str, Any], tcp: np.ndarray) -> SurfaceFrontendFrame:
        """Run one causal frame directly from a ManiSkill observation."""
        return self.step_arrays(*self.observation_arrays(observation), tcp)
