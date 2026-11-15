"""Audit TAPNext++ support geometry on the frozen Peg tracking workload."""

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from cotracker.models.core.model_utils import get_points_on_a_grid
from tapnet.tapnextpp.votsp2026 import utils
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger

VARIANTS = {"a": (False, False), "b": (True, False),
            "c": (False, True), "d": (True, True)}
RGB_KEY = "observation/sensor_data/base_camera/rgb"


def load_workload(config, episode):
    """Load only the frozen baseline inputs and verify every source RGB frame."""
    frames, phases = [], []
    source = Path(config["source"])
    for index, checksum in enumerate(episode["frame_sha256"]):
        path = source / f"frame_{index:06d}.npz"
        if file_sha256(path) != checksum:
            raise ValueError(f"Source frame checksum differs: {path}")
        with np.load(path, allow_pickle=False) as frame:
            frames.append(frame[RGB_KEY][0])
            phases.append(str(frame["collection/phase"].item()))
    keys = ("requested_anchor", "observed_xy", "gt_xy", "gt_occluded", "phases")
    with np.load(config["frozen_queries"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in keys}
    if not np.array_equal(phases, arrays["phases"]):
        raise ValueError("Frozen phase labels differ from the source episode")
    if any(array.shape[0] != len(frames) for array in arrays.values()):
        raise ValueError("Frozen workload length differs from the source episode")
    return frames, arrays


def make_streams(arrays):
    """Keep each physical corner and its original anchor independent."""
    requests = arrays["requested_anchor"]
    streams = []
    for anchor in np.unique(requests[requests >= 0]):
        for slot in np.flatnonzero((requests == anchor).any(axis=0)):
            scored = np.flatnonzero(requests[:, slot] == anchor)
            point = arrays["observed_xy"][anchor, slot]
            if not np.isfinite(point).all() or anchor > scored.min():
                raise ValueError(f"Invalid forward query at frame {anchor}, corner {slot}")
            streams.append((int(anchor), int(slot), scored, point))
    if not streams:
        raise ValueError("Frozen workload contains no scored queries")
    return streams


def query_points(point, shape, resolution, extent, variant):
    """Generate CoTracker grid geometry directly in TAPNext++ model pixels."""
    h, w = shape[:2]
    target = np.asarray(point, dtype=np.float32).reshape(1, 2)
    model_xy = utils.display_to_model(target, h, w, resolution)
    x, y = model_xy[0]
    parts = [model_xy]
    local, global_grid = VARIANTS[variant]
    if local:
        xy = get_points_on_a_grid(9, (extent, extent), (float(y), float(x)))[0]
        xy[40] = torch.from_numpy(model_xy[0])
        parts.append(xy.numpy())
    if global_grid:
        xy = get_points_on_a_grid(4, (resolution, resolution))[0]
        parts.append(xy.numpy())
    model_xy = np.concatenate(parts)
    display_xy = utils.model_to_display(model_xy, h, w, resolution)
    display_xy[0] = target[0]
    if local:
        display_xy[41] = target[0]
    return display_xy, model_xy


def synchronize(device):
    """Flush the selected accelerator before reading the timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def trace_stream(engine, frames, indices, display_xy, model_xy, *, audit_attention=False):
    """Keep full tracks and capture the actual inner model initialization input."""
    inputs, attention = [], []

    def capture(_module, _args, kwargs):
        query = kwargs["query_points"]
        state = kwargs["state"]
        if not inputs:
            if state is not None or query is None:
                raise RuntimeError("Stream must start with fresh state and a query tensor")
            inputs.append(query)
        elif query is not None or state is None or state.query_points is not inputs[0]:
            raise RuntimeError("Continuation must use the same stream's cached queries")

    def capture_attention(_module, args, kwargs):
        if not attention:
            q, k, v = args[:3]
            attention.append({
                "shape": list(q.shape), "shared_qkv": q is k and k is v,
                "has_mask": any(kwargs.get(key) is not None
                                for key in ("attn_mask", "key_padding_mask")),
            })

    with_points = engine._model.register_forward_pre_hook(capture, with_kwargs=True)
    attn_hook = None
    if audit_attention:
        block = engine._model.blocks[0].vit_block
        # The EncoderBlock fast path can bypass the MultiheadAttention hook.
        attn_hook = block.self_attention.register_forward_pre_hook(
            capture_attention, with_kwargs=True
        )
    state = None
    positions, visible, durations = [], [], []
    try:
        synchronize(engine.device)
        for index in indices:
            frame = cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR)
            start = perf_counter()
            xy, vis, state = engine.track_frame(
                frame, query_points_xy=display_xy if state is None else None,
                state=state, autocast=False,
            )
            synchronize(engine.device)
            durations.append((perf_counter() - start) * 1000)
            positions.append(xy)
            visible.append(vis)
    finally:
        with_points.remove()
        if attn_hook is not None:
            attn_hook.remove()

    tensor = inputs[0].cpu().numpy()
    np.testing.assert_array_equal(tensor[..., 0], np.zeros((1, len(model_xy))))
    np.testing.assert_allclose(tensor[0, :, 1:], model_xy[:, ::-1], rtol=0, atol=4e-5)
    if state.step != len(indices) or np.shape(positions) != (len(indices), len(model_xy), 2):
        raise RuntimeError("Model did not retain all query tracks through the stream")
    if audit_attention:
        patch = engine._model.patch_size[0]
        tokens = (engine.input_resolution // patch) ** 2 + len(model_xy)
        if (not attention or attention[0]["shape"][:2] != [1, tokens]
                or not attention[0]["shared_qkv"] or attention[0]["has_mask"]):
            raise RuntimeError(f"Unexpected joint attention inputs: {attention}")
    return {
        "frame_indices": np.asarray(indices), "query_source_xy": display_xy,
        "model_query_tyx": tensor, "trajectory_source_xy": np.asarray(positions),
        "visibility": np.asarray(visible), "inference_ms": np.asarray(durations),
    }, attention


def draw_stream(frame, trace, variant, path):
    """Draw every query and its trajectory in a two-panel source-pixel image."""
    initial = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = initial.shape[:2]
    points = np.rint(trace["query_source_xy"]).astype(int)
    # Leave room to draw CoTracker grid points outside the source image.
    left, top = np.maximum(0, 8 - points.min(axis=0))
    right, bottom = np.maximum(0, points.max(axis=0) + 9 - [width, height])
    initial = cv2.copyMakeBorder(initial, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(48, 48, 48))
    offset = np.array([left, top])
    points += offset
    trajectory = initial.copy()
    colors = [(0, 215, 255)]
    local, global_grid = VARIANTS[variant]
    colors.extend([(210, 195, 25)] * (81 if local else 0))
    colors.extend([(240, 120, 205)] * (16 if global_grid else 0))
    for index in reversed(range(len(colors))):
        color = colors[index]
        radius = 6 if index == 0 else 3
        cv2.circle(initial, tuple(points[index]), radius, color, 1, cv2.LINE_AA)
        track = trace["trajectory_source_xy"][:, index] + offset
        finite = np.isfinite(track).all(axis=-1)
        for start in np.flatnonzero(finite[:-1] & finite[1:]):
            first = tuple(np.rint(track[start]).astype(int))
            last = tuple(np.rint(track[start + 1]).astype(int))
            cv2.line(trajectory, first, last, color, 2 if index == 0 else 1, cv2.LINE_AA)
    for image, title in ((initial, "Anchor queries"), (trajectory, "All predicted trajectories")):
        cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4)
        cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    if not cv2.imwrite(str(path), np.concatenate((initial, trajectory), axis=1)):
        raise OSError(f"Could not write {path}")


def metrics(errors, durations):
    """Summarize GT-corner errors and synchronized timings."""
    return {
        "targets": len(errors), "within_4px": int((errors <= 4).sum()),
        "success_at_4px": float(np.mean(errors <= 4)) if len(errors) else None,
        "median_error_px": float(np.median(errors)) if len(errors) else None,
        "timed_calls": len(durations),
        "mean_inference_ms": float(np.mean(durations)) if len(durations) else None,
        "p95_inference_ms": float(np.percentile(durations, 95)) if len(durations) else None,
    }


def run_variant(engine, frames, arrays, streams, config, variant, output, logger):
    """Run all frozen streams and score only their primary slots."""
    phases = arrays["phases"]
    errors, times, score_phases, inference_phases = [], [], [], []
    for anchor, slot, scored, point in streams:
        display_xy, model_xy = query_points(
            point, frames[anchor].shape, engine.input_resolution,
            config["local_extent_model_px"], variant,
        )
        indices = range(anchor, int(scored[-1]) + 1)
        trace, _ = trace_stream(engine, frames, indices, display_xy, model_xy)
        prediction = trace["trajectory_source_xy"][scored - anchor, 0]
        error = np.linalg.norm(prediction - arrays["gt_xy"][scored, slot], axis=-1)
        if not np.isfinite(error).all():
            raise RuntimeError(f"Missing primary positions at anchor {anchor}, corner {slot}")
        errors.extend(error)
        score_phases.extend(phases[scored])
        times.extend(trace["inference_ms"])
        inference_phases.extend(phases[trace["frame_indices"]])
        stem = output / f"{variant}_frame_{anchor:03d}_corner_{slot}"
        archive = stem.with_suffix(".npz")
        np.savez_compressed(archive, **trace, scored_frames=scored, corner_id=slot,
                            anchor_frame=anchor, primary_error_source_px=error)
        image = stem.with_suffix(".png")
        draw_stream(frames[anchor], trace, variant, image)
        logger.link_output(archive)
        logger.link_output(image)
    errors, times = np.asarray(errors), np.asarray(times)
    score_phases, inference_phases = np.asarray(score_phases), np.asarray(inference_phases)
    return {
        "all": metrics(errors, times),
        **{phase: metrics(errors[score_phases == phase], times[inference_phases == phase])
           for phase in dict.fromkeys(phases)},
    }


def evaluate(config):
    """Warm up, verify joint attention and run the matched four-way comparison."""
    device = torch.device(config["device"])
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    receipt = Path(config["source"]) / "episode.json"
    episode = json.loads(receipt.read_text())
    frames, arrays = load_workload(config, episode)
    streams = make_streams(arrays)
    if config["tap_resolution"] != TAPNextPP.MODEL_SIZE:
        raise ValueError("This comparison requires the 256 × 256 checkpoint and coordinate space")
    checkpoint = Path(config["tap_checkpoint"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    with ExperimentLogger(team="visual", model="tapnextpp_support", config=config,
                          seed=episode["seed"], job_type="evaluate") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        print(f"Output directory: {output}", flush=True)
        logger.record(device=str(device), torch_version=str(torch.__version__),
                      gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      cuda_version=torch.version.cuda,
                      checkpoint_sha256=file_sha256(checkpoint),
                      episode_sha256=file_sha256(receipt),
                      frozen_queries_sha256=file_sha256(Path(config["frozen_queries"])))
        engine = TAPNextPP.from_checkpoint(checkpoint, device=device,
                                          input_resolution=config["tap_resolution"])
        anchor, slot, _, point = streams[0]
        for variant in VARIANTS:
            display, model = query_points(point, frames[anchor].shape,
                                          engine.input_resolution,
                                          config["local_extent_model_px"], variant)
            # A reverse smoke check uses a new state starting at the same anchor.
            direction = range(anchor, anchor + 2)
            trace, attention = trace_stream(engine, frames, direction, display, model,
                                            audit_attention=True)
            path = output / f"{variant}_forward_smoke.npz"
            np.savez_compressed(path, **trace)
            logger.link_output(path)
            if variant == "d":
                reverse, _ = trace_stream(engine, frames, range(anchor, anchor - 2, -1),
                                          display, model)
                np.testing.assert_array_equal(reverse["model_query_tyx"], trace["model_query_tyx"])
                path = output / "d_reverse_smoke.npz"
                np.savez_compressed(path, **reverse)
                logger.link_output(path)
            logger.record(**{f"attention_{variant}": attention[0]})
            print(f"Verified {variant.upper()}: {len(model)} queries, {attention[0]}", flush=True)
        summary = {}
        for variant in VARIANTS:
            summary[variant] = run_variant(
                engine, frames, arrays, streams, config, variant, output, logger,
            )
            logger.log(summary[variant], prefix=f"eval/{variant}")
            print(f"Finished {variant.upper()}: {summary[variant]['all']}", flush=True)
        path = output / "comparison.json"
        path.write_text(json.dumps(summary, indent=2) + "\n")
        logger.link_output(path)
