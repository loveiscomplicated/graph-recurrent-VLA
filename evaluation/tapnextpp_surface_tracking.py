"""Replay detector corners and TAPNext++ surface-only rigid motion from frame zero."""

import json
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.tapnextpp_mps import (
    compile_mps_recurrent_blocks,
    crop_native_patches,
    enable_mps_fp16_vit,
    run_block_in_fp16,
    use_past_query_mask_tokens,
)
from evaluation.tapnextpp_support import RGB_KEY, synchronize


def choose_points(mask, depth, count):
    """Spread queries over interior Peg pixels with valid depth."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    candidates = np.argwhere((core > 0) & (depth > 0))[:, ::-1].astype(np.float32)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} interior surface pixels for {count} queries")
    index = np.linalg.norm(candidates - candidates.mean(0), axis=1).argmin()
    distance = np.full(len(candidates), np.inf)
    selected = []
    for _ in range(count):
        point = candidates[index]
        selected.append(point)
        distance = np.minimum(distance, np.sum((candidates - point) ** 2, axis=1))
        index = distance.argmax()
    return np.asarray(selected)


def lift(xy, depth, mask, intrinsic, visible):
    """Recover camera XYZ for track coordinates supported by current RGB-D."""
    finite = np.isfinite(xy).all(axis=1)
    uv = np.rint(np.where(finite[:, None], xy, 0)).astype(int)
    u, v = uv.T
    h, w = depth.shape
    usable = finite & visible & (u >= 0) & (v >= 0) & (u < w) & (v < h)
    ids = np.flatnonzero(usable)
    ids = ids[(depth[v[ids], u[ids]] > 0) & mask[v[ids], u[ids]]]
    rays = np.column_stack((xy[ids], np.ones(len(ids)))) @ np.linalg.inv(intrinsic).T
    return ids, rays * (depth[v[ids], u[ids]] * .001)[:, None]


def fit_rigid(reference, current):
    """Fit a batch of proper rotations and translations for row-vector XYZ."""
    ref_mean = reference.mean(axis=-2, keepdims=True)
    cur_mean = current.mean(axis=-2, keepdims=True)
    covariance = (reference - ref_mean).swapaxes(-1, -2) @ (current - cur_mean)
    u, singular, vh = np.linalg.svd(covariance)
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    rotation = u @ vh
    return rotation, cur_mean - ref_mean @ rotation, singular


def estimate_motion(reference, current, generator, config):
    """Choose a rigid consensus among current surface correspondences."""
    minimum = config["minimum_inliers"]
    if len(reference) < minimum:
        return None, None, 0
    samples = np.array(list(combinations(range(len(reference)), 3)))
    budget = config["ransac_samples"]
    if len(samples) > budget:
        samples = samples[generator.choice(len(samples), budget, replace=False)]
    rotation, translation, singular = fit_rigid(reference[samples], current[samples])
    residual = np.linalg.norm(reference[None] @ rotation + translation - current, axis=-1)
    inlier = residual <= config["inlier_threshold_m"]
    inlier[singular[:, 1] < 1e-6] = False
    count = inlier.sum(axis=1)
    best = np.lexsort((np.where(inlier, residual, 0).sum(axis=1), -count))[0]
    if count[best] < minimum:
        return None, None, 0
    keep = inlier[best]
    rotation, translation, singular = fit_rigid(reference[keep], current[keep])
    if singular[1] < 1e-6:
        return None, None, 0
    return rotation, translation, int(count[best])


def project(xyz, intrinsic):
    """Project camera-space XYZ to source-image XY."""
    uvw = xyz @ intrinsic.T
    return uvw[:, :2] / uvw[:, 2:]


def associate(source, target, gate):
    """Assign nearby 3D points one to one and return accepted index pairs."""
    if not len(source) or not len(target):
        return []
    distance = np.linalg.norm(source[:, None] - target[None], axis=-1)
    rows, cols = linear_sum_assignment(distance + (distance > gate))
    return [(i, j) for i, j in zip(rows, cols) if distance[i, j] <= gate]


def read_frames(config, episode):
    """Read checksum-verified RGB-D and calibration without loading GT for inference."""
    source = Path(config["source"])
    frames = []
    for index, digest in enumerate(episode["frame_sha256"]):
        path = source / f"frame_{index:06d}.npz"
        if file_sha256(path) != digest:
            raise ValueError(f"Frame checksum differs: {path}")
        with np.load(path) as archive:
            depth = archive["observation/sensor_data/base_camera/depth"][0].squeeze()
            frames.append((
                archive[RGB_KEY][0], depth,
                archive["observation/sensor_param/base_camera/intrinsic_cv"][0],
                archive["observation/sensor_param/base_camera/extrinsic_cv"][0],
            ))
    return frames


def detector_inputs(frame, device, mode):
    """Stage detector inputs while retaining the host frame for tracking and fusion."""
    if mode in ("mps", "cuda"):
        rgb, depth, intrinsic, extrinsic = frame
        return (
            torch.as_tensor(rgb, device=device),
            torch.as_tensor(depth, device=device),
            torch.as_tensor(intrinsic, device=device, dtype=torch.float32),
            torch.as_tensor(extrinsic, device=device, dtype=torch.float32),
        )
    return frame


def resolve_detector_input_mode(mode, device):
    """Select the input representation and validate explicit accelerator modes."""
    if mode not in ("auto", "numpy", "mps", "cuda"):
        raise ValueError("detector_inputs must be auto, numpy, mps or cuda")
    if mode == "auto":
        return device.type if device.type in ("mps", "cuda") else "numpy"
    if mode != "numpy" and mode != device.type:
        raise ValueError(f"{mode} detector inputs require a {mode} detector")
    return mode


def track_surface_frame(tracker, config, bgr, query=None, state=None, observation_mask=None):
    """Track one frame and return positions in source image pixels."""
    crop = config.get("tap_crop_xyxy")
    if crop is not None:
        if not isinstance(crop, (list, tuple)) or len(crop) != 4 or any(
            type(edge) is not int for edge in crop
        ):
            raise ValueError("tap_crop_xyxy must be four integer pixel edges [left, top, right, bottom]")
        left, top, right, bottom = crop
        height, width = bgr.shape[:2]
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError(f"tap_crop_xyxy {crop} is outside the {width}x{height} frame")
        bgr = bgr[top:bottom, left:right]
        offset = np.array([left, top], dtype=np.float32)
        if query is not None:
            query = query - offset
            if np.any(query < 0) or np.any(query >= [right - left, bottom - top]):
                raise ValueError("tap_crop_xyxy must contain every initial query")
    options = {"observation_mask": observation_mask} if config.get("tap_adaptive_crop") else {}
    tracked, visible, state = tracker.track_frame(
        bgr, query_points_xy=query, state=state,
        autocast=config.get("tap_cuda_autocast", False),
        **options,
    )
    if crop is not None:
        tracked += offset
    return tracked, visible, state


def replay(config, frames, tracker, detector):
    """Initialize once, track surface points only, and fuse current detections."""
    generator = np.random.default_rng(config["ransac_seed"])
    skip_query_embeddings = config.get("tap_mps_skip_past_query_embeddings", False)
    input_mode = config.get("detector_inputs", "numpy")
    state = None
    rows = []
    for index, frame in enumerate(frames):
        rgb, depth, intrinsic, _ = frame
        begin = perf_counter()
        inputs = detector_inputs(frame, detector.device, input_mode)
        detection = detector(*inputs)
        peg = detection.object_id == 0
        corners_xyz = detection.camera_xyz[peg].cpu().numpy()
        corners_xy = detection.uv[peg].cpu().numpy()
        mask = detection.masks[0].cpu().numpy().astype(bool)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector_end = perf_counter()
        if index == 0:
            query = choose_points(mask, depth, config["surface_queries"])
            _, reference = lift(query, depth, mask, intrinsic, np.ones(len(query), bool))
            template = corners_xyz.copy()
            if not len(template):
                raise ValueError("Detector returned no initial Peg corners")
        start = perf_counter()
        query_scope = nullcontext()
        if skip_query_embeddings and state is not None:
            # This replay only injects queries at step zero. TAPNext++ subtracts
            # the current step, so all queries select the mask token from here on.
            query_scope = use_past_query_mask_tokens(tracker)
        with query_scope:
            tracked, visible, state = track_surface_frame(
                tracker, config, bgr, query if index == 0 else None, state, observation_mask=mask,
            )
        synchronize(tracker.device)
        track_end = perf_counter()
        valid, current = lift(tracked, depth, mask, intrinsic, visible)
        if index == 0:
            rotation, translation, inliers = np.eye(3), np.zeros((1, 3)), len(reference)
        else:
            rotation, translation, inliers = estimate_motion(reference[valid], current, generator, config)
        projected = np.full((len(template), 2), np.nan)
        direct = projected.copy()
        if rotation is not None:
            moved = template @ rotation + translation
            projected[:] = project(moved, intrinsic)
            for slot, candidate in associate(moved, corners_xyz, config["detection_gate_m"]):
                direct[slot] = corners_xy[candidate]
        fused = np.where(np.isfinite(direct), direct, projected)
        end = perf_counter()
        rows.append(dict(
            surface_xy=tracked, surface_visible=visible, surface_rgbd_valid=len(valid),
            pose_inliers=inliers, projected_xy=projected, detected_xy=direct,
            fused_xy=fused, detector_ms=(detector_end - begin) * 1000,
            initialization_ms=(start - detector_end) * 1000,
            tap_ms=(track_end - start) * 1000, fusion_ms=(end - track_end) * 1000,
            total_ms=(end - begin) * 1000, raw_corners_xy=corners_xy,
            raw_corners_xyz=corners_xyz,
        ))
        if config.get("tap_adaptive_crop"):
            rows[-1]["tap_patch_xyxy"] = state.crop
        if index % 25 == 0:
            print(f"Frame {index}: {inliers} rigid inliers", flush=True)
    return query, template, rows


def accuracy(prediction, truth, mask):
    """Count unresolved forecasts as failures and summarize finite errors."""
    error = np.linalg.norm(prediction[mask] - truth[mask], axis=-1)
    finite = np.isfinite(error)
    return dict(samples=len(error), available=int(finite.sum()), within_4px=int((error <= 4).sum()),
                median_error_px=float(np.median(error[finite])) if finite.any() else None)


def report(config, frames, rows, template):
    """Use frozen references only after replay to assign initial identities and score."""
    with np.load(config["frozen_queries"]) as archive:
        ref = {k: archive[k] for k in ("observed_xy", "gt_xy", "requested_anchor", "phases")}
    visible = np.isfinite(ref["observed_xy"]).all(axis=-1)
    initial_ids = np.flatnonzero(visible[0])
    initial_pairs = associate(project(template, frames[0][2]), ref["gt_xy"][0, initial_ids], 4.0)
    # The gate above is in source pixels, and serves only to establish evaluation identities.
    physical_ids = np.full(len(template), -1)
    for slot, target in initial_pairs:
        physical_ids[slot] = initial_ids[target]
    xy = {key: np.full_like(ref["gt_xy"], np.nan) for key in
          ("projected_xy", "detected_xy", "fused_xy")}
    for slot, identity in enumerate(physical_ids):
        if identity >= 0:
            for key in xy:
                xy[key][:, identity] = [row[key][slot] for row in rows]
    # Score current-frame detections against visible references without requiring
    # an identity to have been available in frame zero.
    detector_visible_xy = np.full_like(ref["gt_xy"], np.nan)
    for index, row in enumerate(rows):
        ids = np.flatnonzero(visible[index])
        for slot, target in associate(row["raw_corners_xy"], ref["gt_xy"][index, ids], 4.0):
            detector_visible_xy[index, ids[target]] = row["raw_corners_xy"][slot]
    visible[0] = False
    hidden = ref["requested_anchor"] >= 0
    corner6 = hidden & (np.arange(hidden.shape[1]) == 6) & (ref["requested_anchor"] == 83)
    result = dict(
        initial_corner_ids=physical_ids.tolist(),
        visible_detector=accuracy(detector_visible_xy, ref["gt_xy"], visible),
        hidden_fused=accuracy(xy["fused_xy"], ref["gt_xy"], hidden),
        hidden_rigid_only=accuracy(xy["projected_xy"], ref["gt_xy"], hidden),
        corner6_hidden_fused=accuracy(xy["fused_xy"], ref["gt_xy"], corner6),
        corner6_hidden_rigid=accuracy(xy["projected_xy"], ref["gt_xy"], corner6),
        hidden_by_phase={
            str(phase): accuracy(xy["fused_xy"], ref["gt_xy"], hidden & (ref["phases"] == phase)[:, None])
            for phase in dict.fromkeys(ref["phases"])
        },
        pose_available_frames=sum(row["pose_inliers"] >= config["minimum_inliers"] for row in rows),
        timing={key: dict(init_ms=rows[0][key], mean_ms=float(np.mean([r[key] for r in rows[1:]])),
                          p95_ms=float(np.percentile([r[key] for r in rows[1:]], 95)))
                for key in ("detector_ms", "tap_ms", "fusion_ms", "total_ms")},
    )
    xy.update(gt_xy=ref["gt_xy"], visible_mask=visible, hidden_mask=hidden, corner6_mask=corner6,
              phases=ref["phases"], detector_scored_xy=detector_visible_xy)
    return result, xy


def load_tracker(config):
    """Load the configured online implementation and its inference optimizations."""
    native_crop = config.get("tap_native_patch_crop_xyxy")
    adaptive = config.get("tap_adaptive_crop", False)
    graph = config.get("tap_cuda_graph", False)
    if graph and (not adaptive or config["device"] == "mlx" or torch.device(config["device"]).type != "cuda"):
        raise ValueError("tap_cuda_graph requires adaptive CUDA tracking")
    if adaptive:
        supported = config["device"] == "mlx" or torch.device(config["device"]).type == "cuda"
        if not supported or native_crop is not None or config.get("tap_crop_xyxy") is not None:
            raise ValueError("tap_adaptive_crop requires MLX or CUDA and selects its own patch window")
    if native_crop is not None and config.get("tap_crop_xyxy") is not None:
        raise ValueError("Choose either tap_native_patch_crop_xyxy or tap_crop_xyxy")
    if config["device"] == "mlx":
        from visual.tapnextpp_mlx import OnlineTracker

        if config.get("tap_crop_xyxy") is not None:
            raise ValueError("TAPNext++ MLX requires a full frame and optional native patch crop")
        torch_options = (
            "tap_cuda_autocast", "tap_cache_cuda_autocast", "tap_mps_fp16_vit",
            "tap_mps_fp16_recurrent_mlp", "tap_mps_skip_past_query_embeddings",
            "tap_mps_compile_recurrent",
        )
        if any(config[key] for key in torch_options):
            raise ValueError("TAPNext++ MLX requires the PyTorch-specific flags to be disabled")
        if adaptive:
            from visual.tapnextpp_adaptive import AdaptiveOnlineTracker

            tracker_type = AdaptiveOnlineTracker
        else:
            tracker_type = OnlineTracker
        return tracker_type.from_checkpoint(
            config["checkpoint"], config["tap_input_resolution"], native_crop,
        )
    if adaptive:
        from visual.tapnextpp_cuda import AdaptiveCudaTracker

        tracker_type = AdaptiveCudaTracker
    else:
        tracker_type = TAPNextPP
    tracker = tracker_type.from_checkpoint(
        config["checkpoint"],
        device=config["device"],
        input_resolution=config["tap_input_resolution"],
    )
    if adaptive:
        tracker.cuda_graph_enabled = graph
    if config["tap_mps_fp16_vit"]:
        enable_mps_fp16_vit(tracker)
    if config["tap_mps_skip_past_query_embeddings"] and tracker.device.type != "mps":
        raise ValueError("tap_mps_skip_past_query_embeddings requires an MPS tracker")
    if config["tap_mps_fp16_recurrent_mlp"]:
        if tracker.device.type != "mps":
            raise ValueError("tap_mps_fp16_recurrent_mlp requires an MPS tracker")
        for block in tracker._model.blocks:
            run_block_in_fp16(block.ssm_block.mlp_block)
    if config["tap_mps_compile_recurrent"]:
        compile_mps_recurrent_blocks(tracker)
    if native_crop is not None:
        crop_native_patches(tracker, native_crop)
    return tracker


def evaluate(config):
    """Measure the RGB-D detector plus one continuous surface-only TAPNext++ stream."""
    config = {
        "tap_input_resolution": 256,
        "tap_cuda_autocast": False,
        "tap_cache_cuda_autocast": False,
        "tap_mps_fp16_vit": False,
        "tap_mps_fp16_recurrent_mlp": False,
        "tap_mps_skip_past_query_embeddings": False,
        "tap_mps_compile_recurrent": False,
        "detector_inputs": "auto",
        **config,
    }
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    receipt = Path(config["source"]) / "episode.json"
    episode = json.loads(receipt.read_text())
    frames = read_frames(config, episode)
    tracker = load_tracker(config)
    detector = TorchCornerDetector.main(device=config["detector_device"])
    mode = resolve_detector_input_mode(config["detector_inputs"], detector.device)
    config["detector_inputs"] = mode
    warmup_frames = config.get("detector_warmup_frames", 1)
    if type(warmup_frames) is not int or warmup_frames < 1:
        raise ValueError("detector_warmup_frames must be a positive integer")
    warmup_frames = min(warmup_frames, len(frames))
    with ExperimentLogger(team="visual", model="tapnextpp_surface_tracking", config=config,
                          seed=episode["seed"], job_type="evaluate") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        logger.record(episode_sha256=file_sha256(receipt), checkpoint_sha256=file_sha256(Path(config["checkpoint"])),
                      frozen_queries_sha256=file_sha256(Path(config["frozen_queries"])),
                      torch_version=str(torch.__version__), tracker_device=str(tracker.device),
                      resolved_detector_device=str(detector.device))
        if config["device"] == "mlx":
            logger.record(mlx_version=tracker.runtime_version)
        print(f"Output directory: {output}", flush=True)
        # CUDA compilation can specialize on shapes that appear after frame zero.
        for index in np.linspace(0, len(frames) - 1, warmup_frames, dtype=int)[1:]:
            detector(*detector_inputs(frames[index], detector.device, mode))
        # Warm up on frame zero with disposable state before the measured replay.
        rgb, depth, _, _ = frames[0]
        warm = detector(*detector_inputs(frames[0], detector.device, mode))
        warm_mask = warm.masks[0].cpu().numpy()
        query = choose_points(warm_mask, depth, config["surface_queries"])
        scope = nullcontext()
        if config["tap_cache_cuda_autocast"]:
            if tracker.device.type != "cuda" or not config["tap_cuda_autocast"]:
                raise ValueError("tap_cache_cuda_autocast requires CUDA FP16 autocast")
            # Preserve the weight cache across nested tracker calls while
            # leaving the detector and preprocessing outside mixed precision.
            scope = torch.autocast("cuda", enabled=False)
        with scope:
            warm_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            warm_start = perf_counter()
            _, _, warm_state = track_surface_frame(tracker, config, warm_frame, query, observation_mask=warm_mask)
            if config["tap_mps_compile_recurrent"] or config["device"] == "mlx" or config.get("tap_adaptive_crop"):
                # The initial and continuation calls have different cache signatures.
                track_surface_frame(tracker, config, warm_frame, state=warm_state, observation_mask=warm_mask)
            del warm_state
            synchronize(tracker.device)
            if config["tap_mps_compile_recurrent"] or config["device"] == "mlx" or config.get("tap_adaptive_crop"):
                logger.record(tracker_compile_warmup_ms=(perf_counter() - warm_start) * 1000)
            query, template, rows = replay(config, frames, tracker, detector)
        result, xy = report(config, frames, rows, template)
        path = output / "summary.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        logger.link_output(path)
        trace = output / "tracks.npz"
        scalars = {key: np.asarray([row[key] for row in rows]) for key in
                   ("pose_inliers", "surface_rgbd_valid", "surface_xy", "surface_visible",
                    "detector_ms", "initialization_ms", "tap_ms", "fusion_ms", "total_ms")}
        if config.get("tap_adaptive_crop"):
            scalars["tap_patch_xyxy"] = np.asarray([row["tap_patch_xyxy"] for row in rows])
        np.savez_compressed(trace, query_xy=query, frame_indices=np.arange(len(rows)), **xy, **scalars)
        logger.link_output(trace)
        logger.log(result, prefix="eval")
        print(json.dumps(result, indent=2), flush=True)
