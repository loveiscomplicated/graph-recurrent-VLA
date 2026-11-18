"""Cache the main RGB-D detector's observed corners by raw episode and frame."""

import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import scipy
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger
from visual import corner_detection


CAMERA = "base_camera"
FRAME_INPUTS = (
    f"observation/sensor_data/{CAMERA}/rgb",
    f"observation/sensor_data/{CAMERA}/depth",
    f"observation/sensor_param/{CAMERA}/intrinsic_cv",
    f"observation/sensor_param/{CAMERA}/extrinsic_cv",
)


def load_source(config):
    """Check the pinned dataset and retain manifest order for selected episodes."""
    manifest_path = Path(config["source_manifest"])
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != config["source_manifest_sha256"]:
        raise ValueError(f"Source manifest SHA-256 mismatch: {manifest_path}")
    source = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    if source["status"] != "COMPLETE" or file_sha256(root / "protocol.json") != source["protocol_sha256"]:
        raise ValueError(f"Expected a complete collection with its original protocol: {root}")
    episodes = source["episodes"]
    paths = {row["path"] for row in episodes}
    cases = {(row["seed"], row["mode"]) for row in episodes}
    if not episodes or len(paths) != len(episodes) or len(cases) != len(episodes):
        raise ValueError("Source manifest must contain unique episode paths and seed/mode pairs")
    requested = config.get("episodes")
    if requested is not None:
        selected = set(requested)
        if not selected or len(selected) != len(requested) or not selected <= paths:
            raise ValueError("episodes must list unique paths from the source manifest")
        episodes = [row for row in episodes if row["path"] in selected]
    return root, source["protocol_sha256"], episodes


def read_inputs(path, expected_hash, device):
    """Check raw frame bytes, then load only RGB-D and CV calibration."""
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_hash:
        raise ValueError(f"Raw frame SHA-256 mismatch: {path}")
    with np.load(io.BytesIO(contents), allow_pickle=False) as frame:
        return tuple(torch.as_tensor(frame[key][0], device=device) for key in FRAME_INPUTS)


@torch.inference_mode()
def cache_episode(detector, root, row, *, frame_reader=read_inputs):
    """Preserve native detector order using offsets, including frames with no corners."""
    name = row["path"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"Expected a single relative episode directory name: {name!r}")
    episode = root / name
    count = row["observations"]
    hashes = row["frame_sha256"]
    if (json.loads((episode / "episode.json").read_text()) != row
            or len(hashes) != count or count != row["steps"] + 1):
        raise ValueError(f"Invalid source episode receipt or frame count: {episode}")

    offsets = np.zeros(count + 1, dtype=np.int64)
    coordinates, object_ids, methods, rim_flags = [], [], [], []
    for index, digest in enumerate(hashes):
        path = episode / f"frame_{index:06d}.npz"
        result = detector(*frame_reader(path, digest, detector.device))
        points = torch.cat((result.uv, result.xyz, result.camera_xyz), dim=1).cpu().numpy()
        ids = result.object_id.cpu().numpy()
        if (len(ids) != len(points) or len(ids) != len(result.metadata)
                or not np.isfinite(points).all() or not np.isin(ids, (0, 1)).all()):
            raise ValueError(f"Invalid detector output: {path}")
        coordinates.append(points)
        object_ids.append(ids)
        methods.extend(meta["method"] for meta in result.metadata)
        rim_flags.extend(meta.get("contour_hole", False) for meta in result.metadata)
        offsets[index + 1] = offsets[index] + len(points)

    points = np.concatenate(coordinates)
    return dict(
        frame_indices=np.arange(count, dtype=np.int64),
        corner_offsets=offsets,
        corner_object_id=np.concatenate(object_ids).astype(np.int8),
        corner_uv=points[:, :2],
        corner_world_xyz=points[:, 2:5],
        corner_camera_xyz=points[:, 5:8],
        corner_method=np.asarray(methods, dtype=str),
        corner_contour_hole=np.asarray(rim_flags, dtype=bool),
    )


def write_episode(path, arrays):
    """Round-trip the NPZ before exposing it to downstream readers."""
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(temporary, allow_pickle=False) as saved:
        for name, expected in arrays.items():
            actual = saved[name]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError(f"Cache readback mismatch in {path.name}: {name}")
    temporary.replace(path)


def build(config):
    """Read and write only detector data and assemble a checksum-linked index."""
    root, protocol_hash, episodes = load_source(config)
    torch.set_num_threads(config.get("threads", 1))
    cv2.setNumThreads(1)
    detector = corner_detection.TorchCornerDetector.main(device=config["device"])
    detector_root = Path(corner_detection.__file__).parent
    source_hashes = {str(path.relative_to(detector_root)): file_sha256(path)
                     for path in sorted(detector_root.rglob("*.py"))}
    device = detector.device
    runtime = dict(device=str(device), torch=str(torch.__version__), numpy=np.__version__,
                   opencv=cv2.__version__, scipy=scipy.__version__)
    if device.type == "cuda":
        runtime.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
    provenance = dict(
        source_manifest_sha256=config["source_manifest_sha256"], source_protocol_sha256=protocol_hash,
        detector="TorchCornerDetector.main", camera=CAMERA, plane_fit_method=detector.plane_fit_method,
        detector_parameters=asdict(detector.parameters), detector_source_sha256=source_hashes,
        cache_source_sha256=file_sha256(Path(__file__)), runtime=runtime,
        source_revision=config.get("source_revision"), source_worktree_dirty=config.get("source_worktree_dirty"),
    )

    with ExperimentLogger(team="visual", model="corner_detection_cache", config=config,
                          job_type="prepare") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        logger.record(**provenance)
        print(f"Output directory: {output}", flush=True)
        started = perf_counter()
        records = []
        frames, corners = 0, 0
        for index, row in enumerate(episodes, start=1):
            arrays = cache_episode(detector, root, row)
            path = output / f"{row['path']}.npz"
            write_episode(path, arrays)
            frame_count = len(arrays["frame_indices"])
            kinds = arrays["corner_object_id"]
            offsets = arrays["corner_offsets"]
            records.append(dict(
                source=row["path"], seed=row["seed"], mode=row["mode"], source_split=row["split"],
                frames=frame_count, corners=len(kinds), peg_corners=int(np.count_nonzero(kinds == 0)),
                box_corners=int(np.count_nonzero(kinds == 1)), empty_frames=int(np.count_nonzero(np.diff(offsets) == 0)),
                file=path.name, sha256=file_sha256(path),
            ))
            frames += frame_count
            corners += len(kinds)
            elapsed = perf_counter() - started
            logger.log({"frames": frames, "corners": corners, "elapsed_seconds": elapsed}, step=index)
            print(f"{index}/{len(episodes)} {row['path']} frames={frames} corners={corners} "
                  f"rate={frames / elapsed:.2f} frames/s", flush=True)

        manifest = output / "manifest.json"
        write_json(manifest, dict(status="COMPLETE", run_id=logger.run_id, **provenance,
                                  frames=frames, corners=corners, episodes=records))
        logger.link_output(manifest)
    return manifest

