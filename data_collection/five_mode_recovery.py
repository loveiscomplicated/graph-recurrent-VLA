"""Recover aligned Five-mode detector, surface and supervised training arrays."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import io
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
import subprocess

import numpy as np
import torch

from data_collection.corner_detection_cache import FRAME_INPUTS, cache_episode, write_episode
from data_collection.detected_corner_visibility import classify_episode
from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_cache import build_surface_arrays, read_episode_index
from data_collection.five_mode_collection import collect_batch, collection_lock, persist_batch
from data_collection.recurrent_bc_dataset import robot15
from data_collection.storage import write_json
from graph.utils.object21_targets import object21_from_gt


GT_FIELDS = ("ee_pose", "peg_pose", "hole_pose", "peg_half_sizes", "hole_radius")
CACHE_TYPES = ("corners", "surface", "targets")


def verified_frame(path, digest):
    """Load one checksum-pinned frame, with NumPy decoding keys on demand."""
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError(f"Frame checksum differs: {path}")
    return np.load(io.BytesIO(contents), allow_pickle=False)


def collect_target(frame):
    """Read the robot state and the inputs for the existing Object21 transform."""
    tcp = frame["observation/extra/tcp_pose"][0, :3]
    gt = {name: frame[f"ground_truth/{name}"] for name in GT_FIELDS}
    if tcp.shape != (3,) or not np.allclose(tcp, gt["ee_pose"][0, :3], rtol=0, atol=1e-6):
        raise ValueError("Observed TCP and Object21 target origin disagree")
    robot = robot15(frame)
    if not np.array_equal(robot[13:], frame["observation/agent/qpos"][0, -2:]):
        raise ValueError("Robot15 gripper coordinates differ from Panda qpos")
    return tcp, gt, robot


def pack_targets(rows):
    """Convert an episode's world poses to Object21 in one call."""
    tcp = np.stack([row[0] for row in rows]).astype(np.float32, copy=False)
    gt = {name: np.concatenate([row[1][name] for row in rows]) for name in GT_FIELDS}
    robot = np.stack([row[2] for row in rows]).astype(np.float32, copy=False)
    return dict(frame_indices=np.arange(len(rows), dtype=np.int64),
                tcp=tcp, robot15=robot, object21=object21_from_gt(gt).numpy())


def read_targets(path, episode):
    """Recover labels from the original frames of an existing cache."""
    rows = []
    for index, digest in enumerate(episode["frame_sha256"]):
        with verified_frame(path / f"frame_{index:06d}.npz", digest) as frame:
            rows.append(collect_target(frame))
    return pack_targets(rows)


def episode_arrays(path, episode, detector, tracker, settings, read_mode):
    """Decode raw frames once for detector, tracker and supervised targets."""
    from evaluation.tapnextpp_surface_tracking import read_frames

    frames, targets = [], []

    def read_once(frame_path, digest, device):
        with verified_frame(frame_path, digest) as frame:
            rgb, depth, intrinsic, extrinsic = (frame[key][0] for key in FRAME_INPUTS)
            if read_mode != "reference":
                frames.append((rgb, depth.squeeze(), intrinsic, extrinsic))
            targets.append(collect_target(frame))
        return tuple(torch.as_tensor(value, device=device)
                     for value in (rgb, depth, intrinsic, extrinsic))

    corners = cache_episode(detector, path.parent, episode, frame_reader=read_once)
    if read_mode == "reference":
        frames = read_frames({"source": path}, episode)
    surface = build_surface_arrays(frames, corners, tracker, detector, settings)
    equal = True
    if read_mode == "check":
        reference = cache_episode(detector, path.parent, episode)
        reference_frames = read_frames({"source": path}, episode)
        reference_surface = build_surface_arrays(reference_frames, reference, tracker, detector, settings)
        for first, second in ((corners, reference), (surface, reference_surface)):
            equal &= first.keys() == second.keys() and all(
                first[key].dtype == second[key].dtype and np.array_equal(first[key], second[key])
                for key in first
            )
        corners, surface = reference, reference_surface
    return (corners, surface, pack_targets(targets)), equal


def store_case(batch_output, case, episode, arrays):
    """Write the three aligned NPZ files and classify each frame from detector output."""
    count = case["observations"]
    expected = np.arange(count)
    for kind, table in zip(CACHE_TYPES, arrays, strict=True):
        if not np.array_equal(table["frame_indices"], expected):
            raise ValueError(f"{kind} frame identity differs for {case}")
    corners, surface, targets = arrays
    for field, width in (("tcp", 3), ("object21", 21), ("robot15", 15)):
        values = targets[field]
        if values.shape != (count, width) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field} rows for {case}")
    if (not np.allclose(targets["tcp"], targets["robot15"][:, :3], rtol=0, atol=1e-6)
            or np.any(targets["robot15"][:, 3] < 0)):
        raise ValueError(f"Robot15 TCP position or quaternion convention differs for {case}")
    groups = classify_episode(corners, count, tolerance_m=0.001)
    files = {}
    for kind, table in zip(CACHE_TYPES, arrays, strict=True):
        relative = Path(kind) / f'{case["seed"]}_{case["policy_mode"]}.npz'
        path = batch_output / relative
        path.parent.mkdir(exist_ok=True)
        write_episode(path, table)
        files[kind] = dict(file=relative.as_posix(), sha256=file_sha256(path))
    return dict(case, files=files, conditions=groups,
                frame_sha256=episode["frame_sha256"],
                surface_anchor_frame=int(surface["surface_anchor_frame"]))


def pack_cache(output):
    """Put only the three validated cache directories in the publishable tar."""
    target = output / "learning.tar"
    temporary = output / "learning.partial"
    with tarfile.open(temporary, "w") as archive:
        for kind in CACHE_TYPES:
            archive.add(output / kind, arcname=kind)
    temporary.replace(target)
    for kind in CACHE_TYPES:
        shutil.rmtree(output / kind)
    return dict(file=target.name, sha256=file_sha256(target), bytes=target.stat().st_size)


def recover_batch(job, output, scratch, collector, detector, tracker, settings, read_mode):
    """Collect one frozen batch and overlap CPU compression with feature extraction."""
    output.mkdir(parents=True)
    seed_range = tuple(job["raw_archive"]["record_range"])
    protocol = job["protocol"]
    config = dict(job["collection"], collection_output=str(output), compress_frames=False,
                  minimum_free_gib=1)
    records = protocol["records"][slice(*seed_range)]
    started = perf_counter()
    with TemporaryDirectory(prefix="five_mode_recovery_", dir=scratch) as temporary:
        staging = Path(temporary)
        collect_batch(seed_range, staging, config=config, protocol=protocol, executor=collector)
        collected = perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            archived = pool.submit(persist_batch, staging, seed_range, target=output, config=config)
            cases = []
            exact = True
            for case, episode, path in read_episode_index(staging, records):
                arrays, equal = episode_arrays(path, episode, detector, tracker, settings, read_mode)
                cases.append(store_case(output, case, episode, arrays))
                exact &= equal
            raw_receipt = archived.result()
    learning = pack_cache(output)
    return dict(key=job["key"], episodes=cases, raw=raw_receipt, learning=learning,
                original_archive_sha256=job["raw_archive"]["sha256"],
                reference_exact=bool(exact) if read_mode == "check" else None,
                collect_seconds=collected - started, total_seconds=perf_counter() - started)


def read_retained_cache(archive, receipt):
    """Read only the retained episode member pinned by the published manifest."""
    with archive.extractfile(receipt["file"]) as stream:
        contents = stream.read()
    if hashlib.sha256(contents).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Retained cache checksum differs: {receipt['file']}")
    with np.load(io.BytesIO(contents), allow_pickle=False) as arrays:
        return dict(arrays)


def recover_retained(job, output, release, scratch, protocol):
    """Attach Object21/TCP to 520 intact episodes without running a detector."""
    source = Path(job["retained_path"]).resolve()
    raw = job["raw_archive"]
    if file_sha256(source) != raw["sha256"]:
        raise ValueError(f"Retained raw archive checksum differs: {source}")
    expected = {(row["seed"], row["policy_mode"]): row for row in job["original_episodes"]}
    seed_range = raw["record_range"]
    records = protocol["records"][slice(*seed_range)]
    cases = []
    output.mkdir(parents=True)
    with TemporaryDirectory(prefix="five_mode_retained_", dir=scratch) as temporary:
        subprocess.run(["tar", "--zstd", "-xf", str(source), "-C", temporary], check=True)
        pin = job["original_caches"]
        with tarfile.open(release / pin["corners"]["file"]) as corner_tar, \
                tarfile.open(release / pin["surface"]["file"]) as surface_tar:
            for case, episode, path in read_episode_index(Path(temporary), records):
                old = expected.pop((case["seed"], case["policy_mode"]))
                if old["observations"] != case["observations"] or old["split"] != case["split"]:
                    raise ValueError(f"Retained cache frame or split differs: {case}")
                corners = read_retained_cache(corner_tar, old["files"]["corners"])
                surface = read_retained_cache(surface_tar, old["files"]["surface"])
                labels = read_targets(path, episode)
                cases.append(store_case(output, case, episode, (corners, surface, labels)))
    if expected:
        raise ValueError("Retained raw archive did not reproduce the saved episode index")
    learning = pack_cache(output)
    return dict(key=job["key"], episodes=cases, raw=dict(raw, retained_path=str(source)),
                learning=learning, original_archive_sha256=raw["sha256"], source="retained")


def worker(config):
    """Serve assigned batch requests using persistent simulator and model processes."""
    import cv2

    from evaluation.tapnextpp_surface_tracking import load_tracker
    from visual.corner_detection import TorchCornerDetector

    root = Path(config["root"]).resolve()
    inbox = root / "inbox"
    results = root / "results"
    scratch = Path(config["scratch"])
    for directory in (inbox, results, scratch):
        directory.mkdir(parents=True, exist_ok=True)
    settings = config["tracking"]
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch._dynamo.config.recompile_limit = 64
    read_mode = "check"
    detector = tracker = None
    with collection_lock(root), ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as collector:
        while not (root / "stop").exists():
            pending = sorted(inbox.glob("*.json"))
            if not pending:
                sleep(1)
                continue
            request = pending[0]
            job = json.loads(request.read_text())
            output = results / job["key"]
            receipt_path = output / "receipt.json"
            if receipt_path.exists():
                request.unlink()
                continue
            if output.exists():
                raise ValueError(f"Unverified output requires a new attempt directory: {output}")
            if detector is None:
                detector = TorchCornerDetector.main(device=settings["detector_device"])
                tracker = load_tracker(settings)
            print("Starting", job["key"], read_mode, flush=True)
            receipt = recover_batch(job, output, scratch, collector, detector, tracker, settings, read_mode)
            if read_mode == "check":
                read_mode = "single" if receipt["reference_exact"] else "reference"
            receipt["worker_read_mode"] = read_mode
            write_json(receipt_path, receipt)
            request.unlink()
            print(job["key"], receipt["collect_seconds"], receipt["total_seconds"], flush=True)
