"""Replay the same saved frames with the reference and native MPS RANSAC."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.corner_detection_holdout import INPUTS, detector_source_hashes, hardware, load_episode
from evaluation.hole_rim_ransac_fallback import _extract_frames, _verified_path
from utils.experiment_logger import ExperimentLogger
from utils.timer import Timer
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.backends.native_ransac import NativeRansac

_COORDINATES = ("uv", "camera_xyz", "xyz")


def output_digest(result):
    """Identify the ordered detector output, excluding its measured duration."""
    digest = hashlib.sha256()
    masks = result["masks"]
    digest.update(masks.tobytes())
    digest.update(json.dumps((masks.shape, masks.dtype.str)).encode())
    digest.update(json.dumps((result["diagnostics"], result["parameters"]), sort_keys=True).encode())
    for corner in result["corners"]:
        metadata = {key: value for key, value in corner.items() if key not in _COORDINATES}
        digest.update(json.dumps(metadata, sort_keys=True, default=np.generic.item).encode())
        for key in _COORDINATES:
            digest.update(np.asarray(corner[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def _timed(detector, extension, inputs):
    scope = nullcontext() if extension is None else extension.activate()
    with scope:
        started = perf_counter()
        result = detector(*inputs).as_dict()
        return result, (perf_counter() - started) * 1000


def _compare(detectors, native, inputs, order):
    results = [None, None]
    for index in (order, 1 - order):
        results[index] = _timed(detectors[index], native if index else None, inputs)
    (reference, reference_ms), (candidate, native_ms) = results
    before, after = output_digest(reference), output_digest(candidate)
    return dict(reference_ms=reference_ms, native_ms=native_ms,
                reference_sha256=before, native_sha256=after, identical=before == after)


def _fit_timing(detectors, native, inputs, order):
    """Measure the plane-fit subset separately from ordinary detector latency."""
    times = [0.0, 0.0]
    for index in (order, 1 - order):
        detector = detectors[index]
        timer = Timer(synchronize=torch.mps.synchronize)
        detector.timer = timer
        try:
            _timed(detector, native if index else None, inputs)
        finally:
            detector.timer = None
        stats = timer.summary().get("plane_fit")
        if stats:
            times[index] = stats["total_seconds"] * 1000
    return times


def _warmup(detectors, native, inputs, count):
    for _ in range(count):
        for index, detector in enumerate(detectors):
            _timed(detector, native if index else None, inputs)


def _holdout(config, episodes, source, detectors, native):
    """Replay host frames and one independently timed resident frame per episode."""
    host, resident, fit = [], [], []
    limit = config.get("frames_per_episode")
    for episode_index, episode in enumerate(episodes):
        arrays = load_episode(source.parent, episode)
        count = len(episode["frames"])
        indexes = range(count) if limit is None else np.linspace(0, count - 1, min(count, limit), dtype=int)
        for index in indexes:
            inputs = tuple(arrays[key][index] for key in INPUTS)
            if not host:
                _warmup(detectors, native, inputs, config["warmup_frames"])
                device_inputs = tuple(torch.as_tensor(value, device="mps") for value in inputs)
                _warmup(detectors, native, device_inputs, config["warmup_frames"])
            order = len(host) % 2
            host.append(dict(
                episode=episode["file"], step=episode["frames"][index]["step"],
                **_compare(detectors, native, inputs, order),
            ))

        index = count // 2
        inputs = tuple(arrays[key][index] for key in INPUTS)
        device_inputs = tuple(torch.as_tensor(value, device="mps") for value in inputs)
        order = episode_index % 2
        resident.append(dict(
            episode=episode["file"], step=episode["frames"][index]["step"],
            **_compare(detectors, native, device_inputs, order),
        ))
        fit.append(_fit_timing(detectors, native, inputs, order))
        if (episode_index + 1) % 8 == 0 or episode_index + 1 == len(episodes):
            print(f"{episode_index + 1}/{len(episodes)} holdout episodes", flush=True)
    return host, resident, fit


def _moving(config, detectors, native):
    """Optionally replay the pre-existing checksum-pinned moving-Hole case."""
    source = config.get("moving_case_result")
    if source is None:
        return None, []
    case = json.loads(Path(source).read_text())["config"]
    receipt = json.loads((Path(case["case_root"]) / "receipt.json").read_text())
    archive_hash = receipt["raw_archive"]["sha256"]
    archive = _verified_path(case["raw_archive"], archive_hash, "moving-Hole RGB-D archive")
    records = []
    with TemporaryDirectory() as temporary:
        frames = _extract_frames(archive, Path(temporary), receipt["frames"])
        for index, path in enumerate(frames):
            with np.load(path, allow_pickle=False) as row:
                rgb = row["observation/sensor_data/base_camera/rgb"][0]
                depth = row["observation/sensor_data/base_camera/depth"][0]
                intrinsic = row["observation/sensor_param/base_camera/intrinsic_cv"][0]
                extrinsic = row["observation/sensor_param/base_camera/extrinsic_cv"][0]
            records.append(dict(
                frame=path.name,
                **_compare(detectors, native, (rgb, depth, intrinsic, extrinsic), index % 2),
            ))
    return dict(case=receipt["case_name"], archive_sha256=archive_hash), records


def _timing(values):
    return dict(mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)))


def _summary(rows):
    return dict(
        frames=len(rows),
        identical_frames=sum(row["identical"] for row in rows),
        reference_ms=_timing([row["reference_ms"] for row in rows]),
        native_ms=_timing([row["native_ms"] for row in rows]),
    )


def evaluate(config):
    """Compare exact output and steady-state latency without changing public APIs."""
    torch.set_num_threads(config["threads"])
    cv2.setNumThreads(config["threads"])
    source = Path(config["source"])
    manifest = json.loads(source.read_text())
    episodes = manifest["episodes"][:config.get("max_episodes")]
    if not episodes or config["warmup_frames"] < 1 or config.get("frames_per_episode", 1) < 1:
        raise ValueError("Native RANSAC benchmark requires frames and at least one warmup")
    output_root = Path(config["output_root"])
    with NativeRansac(output_root / "build") as native:
        native_sources = Path(__file__).resolve().parents[1] / "visual/corner_detection/backends/native"
        provenance = dict(
            source_sha256=file_sha256(source), detector_sources=detector_source_hashes(),
            native_sources={path.name: file_sha256(path) for path in sorted(native_sources.iterdir())},
            native_library_sha256=file_sha256(native.library_path),
            platform=platform.platform(), hardware=hardware(torch.device("mps")),
            torch=str(torch.__version__), torch_git=torch.version.git_version,
            numpy=np.__version__, opencv=cv2.__version__,
        )
        with ExperimentLogger(
            team="visual", model="native_mps_ransac", job_type="evaluate",
            config={**config, **provenance}, seed=manifest["config"]["protocol_seed"],
        ) as logger:
            detectors = tuple(TorchCornerDetector.main(device="mps") for _ in range(2))
            host, resident, fits = _holdout(config, episodes, source, detectors, native)
            moving_case, moving = _moving(config, detectors, native)
            cohorts = {"host_rgbd": host, "mps_rgbd": resident}
            if moving:
                cohorts["moving_hole"] = moving
            summary = {name: _summary(rows) for name, rows in cohorts.items()}
            report = dict(
                config=config, provenance=provenance, moving_case=moving_case,
                summary=summary, rows=cohorts, native_calls=native.calls,
                plane_fit_ms=dict(frames=len(fits), reference=_timing([row[0] for row in fits]),
                                  native=_timing([row[1] for row in fits])),
            )
            output = output_root / logger.run_id / "results.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, report)
            logger.link_output(output)
            logger.log(summary, prefix="eval")
            print(json.dumps(summary, indent=2), flush=True)
            print(output, flush=True)
            if not native.calls or any(row["frames"] != row["identical_frames"] for row in summary.values()):
                raise RuntimeError(f"Native RANSAC parity failed; see {output}")
