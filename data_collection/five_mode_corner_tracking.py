"""Build causal tracker sequences from checksum-pinned Five-mode observations."""

import io
import json
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.episode_dataset import file_sha256
from graph.corner_association import CornerMatcher
from utils.experiment_logger import write

SIGNS = np.asarray(list(product((-1, 1), repeat=3)))


def corner_truth(poses):
    """Use the existing evaluation convention: Peg 8, Box 8, front Hole rim 4."""
    half = poses["peg_half_sizes"]
    length = half[0]
    radius = poses["hole_radius"].item()
    groups = (
        (poses["peg_pose"], SIGNS * half),
        (poses["box_pose"], SIGNS * length),
        (poses["hole_pose"], SIGNS[:4] * [length, radius, radius]),
    )
    points = []
    for pose, local in groups:
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
        points.append(rotation.apply(local) + pose[:3])
    peg = poses["peg_pose"]
    hole = poses["hole_pose"]
    head = Rotation.from_quat(peg[[4, 5, 6, 3]]).apply([length, 0, 0]) + peg[:3]
    local_head = Rotation.from_quat(hole[[4, 5, 6, 3]]).inv().apply(head - hole[:3])
    return np.concatenate(points).astype(np.float32), float(
        (local_head[0] + length) * 1000
    )


def read_raw_truth(archive, expected_hash, zstd):
    """Decode only GT arrays, verifying raw archive and every original frame hash."""
    if file_sha256(archive) != expected_hash:
        raise ValueError(f"Raw archive checksum mismatch: {archive}")
    process = subprocess.Popen(
        [zstd, "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    episodes = {}
    index = None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as source:
            for member in source:
                name = member.name.removeprefix("./")
                if name.endswith("episode_index.json"):
                    index = json.load(source.extractfile(member))
                elif name.endswith("episode.json"):
                    metadata = json.load(source.extractfile(member))
                    parent = str(Path(name).parent)
                    episode = episodes.setdefault(parent, {"rows": {}, "digests": {}})
                    episode["metadata"] = metadata
                elif name.endswith(".npz"):
                    parent = str(Path(name).parent)
                    frame_index = int(Path(name).stem.removeprefix("frame_"))
                    contents = source.extractfile(member).read()
                    episode = episodes.setdefault(parent, {"rows": {}, "digests": {}})
                    # TAR metadata can follow its frames, so compare hashes after reading.
                    episode["digests"][frame_index] = sha256(contents).hexdigest()
                    with np.load(io.BytesIO(contents), allow_pickle=False) as frame:
                        names = (
                            "peg_pose",
                            "box_pose",
                            "hole_pose",
                            "peg_half_sizes",
                            "hole_radius",
                        )
                        poses = {key: frame[f"ground_truth/{key}"][0] for key in names}
                        episode["rows"][frame_index] = corner_truth(poses)
        if process.wait() != 0 or index is None:
            raise ValueError(f"Unable to decode raw archive: {archive}")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    result = {}
    for record in index:
        episode = episodes[record["path"]]
        rows = episode["rows"]
        if sorted(rows) != list(range(record["observations"])):
            raise ValueError(f"Incomplete frame sequence: {archive}:{record['path']}")
        for frame_index, digest in episode["digests"].items():
            if digest != episode["metadata"]["frame_sha256"][frame_index]:
                raise ValueError(
                    f"Raw frame checksum mismatch: {archive}:{record['path']}:{frame_index}"
                )
        result[record["seed"], record["policy_mode"]] = (
            np.stack([rows[i][0] for i in range(len(rows))]),
            np.asarray([rows[i][1] for i in range(len(rows))], dtype=np.float32),
        )
    return result


def read_member(source, receipt):
    """Load an explicitly named, checksum-verified detector or target NPZ."""
    contents = source.extractfile(receipt["member"]).read()
    if sha256(contents).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Release member checksum mismatch: {receipt['member']}")
    with np.load(io.BytesIO(contents), allow_pickle=False) as arrays:
        return dict(arrays)


def associate_corners(corners, tcp):
    """Assign physical slots from current detections and TCP; never receive GT."""
    length = len(tcp)
    if not np.array_equal(corners["frame_indices"], np.arange(length)):
        raise ValueError(
            "Detector frame indices must be contiguous and aligned with TCP"
        )
    matcher = CornerMatcher()
    detected = np.zeros((length, 20, 3), dtype=np.float32)
    observed = np.zeros((length, 20), dtype=bool)
    for frame_index, (start, stop) in enumerate(
        zip(corners["corner_offsets"][:-1], corners["corner_offsets"][1:], strict=True)
    ):
        points = corners["corner_world_xyz"][start:stop]
        objects = corners["corner_object_id"][start:stop]
        hints = corners["corner_contour_hole"][start:stop]
        hints = hints | (corners["corner_method"][start:stop] == "box_opening")
        family = np.where(objects == 0, 0, np.where(hints, 2, 1))
        assignment = matcher.step(points, family, tcp[frame_index])
        mask = assignment >= 0
        detected[frame_index, mask] = points[assignment[mask]]
        observed[frame_index] = mask
    return detected, observed


def prepare_batch(job):
    """Produce one portable NPZ per episode without decoding RGB-D pixels."""
    release, raw_root, output = (
        Path(job[key]) for key in ("release", "raw_root", "output")
    )
    records = job["records"]
    raw_archive = raw_root / records[0]["batch"]
    candidates = list(raw_archive.glob("*.tar.zst"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one raw archive in {raw_archive}")
    truth = read_raw_truth(candidates[0], records[0]["raw_sha256"], job["zstd"])
    summaries = []
    opened = {}
    with ExitStack() as archive_stack:
        for record in records:
            seed, mode = record["seed"], record["policy_mode"]
            arrays = {}
            for kind in ("corners", "targets"):
                receipt = record["files"][kind]
                archive_name = receipt["archive"]
                if archive_name not in opened:
                    opened[archive_name] = archive_stack.enter_context(
                        tarfile.open(release / archive_name)
                    )
                arrays[kind] = read_member(opened[archive_name], receipt)
            target = arrays["targets"]
            gt, depth = truth[seed, mode]
            if len(gt) != record["frames"] or len(gt) != len(target["tcp"]):
                raise ValueError(f"Raw/target length mismatch for {seed}/{mode}")
            detected, observed = associate_corners(arrays["corners"], target["tcp"])
            relative_peg = gt[:, :8].mean(axis=1) - target["tcp"]
            np.testing.assert_allclose(
                relative_peg, target["object21"][:, :3], atol=2e-6, rtol=0
            )
            destination = output / record["split"] / f"{seed}_{mode}.npz"
            np.savez_compressed(
                destination,
                detected=detected,
                observed=observed,
                gt=gt,
                tcp=target["tcp"],
                depth_mm=depth,
            )
            summaries.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "split": record["split"],
                    "frames": len(gt),
                    "path": str(destination.relative_to(output)),
                    "sha256": file_sha256(destination),
                    "missing": int((~observed).sum()),
                    "raw_sha256": record["raw_sha256"],
                    "detector_sha256": record["files"]["corners"]["sha256"],
                }
            )
    return summaries


def prepare(config):
    """Preserve release seed splits and prepare train/validation, excluding test bytes."""
    release = Path(config["release"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = release / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != config["manifest_sha256"]:
        raise ValueError("Five-mode manifest does not match the configured SHA-256")
    manifest = json.loads(manifest_path.read_text())
    records = [
        row for row in manifest["records"] if row["split"] in ("train", "validation")
    ]
    split_seeds = {
        split: {r["seed"] for r in manifest["records"] if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(
        split_seeds[a] & split_seeds[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Release reset seeds overlap between dataset splits")
    for split in ("train", "validation"):
        (output / split).mkdir(exist_ok=True)
    batches = {}
    for row in records:
        batches.setdefault(row["batch"], []).append(row)
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError(
            "Install the zstd executable before preparing raw Five-mode archives"
        )
    summaries = []
    started = monotonic()
    with ProcessPoolExecutor(max_workers=config.get("workers", 8)) as pool:
        futures = [
            pool.submit(
                prepare_batch,
                {
                    "release": str(release),
                    "raw_root": config["raw_root"],
                    "output": str(output),
                    "records": rows,
                    "zstd": zstd,
                },
            )
            for rows in batches.values()
        ]
        for index, future in enumerate(as_completed(futures), 1):
            summaries.extend(future.result())
            print(
                f"Prepared {index}/{len(futures)} batches, {len(summaries)} episodes, {monotonic() - started:.1f}s",
                flush=True,
            )
    summaries.sort(key=lambda row: (row["split"], row["seed"], row["mode"]))
    write(
        output / "manifest.json",
        {
            "format": "five_mode_causal_corner_sequences_v1",
            "source_manifest_sha256": manifest_hash,
            "source_hf": config["source_hf"],
            "episodes": len(summaries),
            "frames": sum(row["frames"] for row in summaries),
            "records": summaries,
            "preparation_config": config,
            "test_payloads_read": False,
        },
    )


def load_training_episodes(root):
    """Read only prepared train/validation episodes and validate their identities."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "five_mode_causal_corner_sequences_v1":
        raise ValueError("Expected a prepared causal Five-mode corner dataset")
    groups = {"train": [], "validation": []}
    seeds = {split: set() for split in groups}
    identities = set()
    for record in manifest["records"]:
        split = record["split"]
        if split not in groups:
            raise ValueError("Prepared training dataset must exclude test payloads")
        identity = (record["seed"], record["mode"])
        if identity in identities:
            raise ValueError(f"Duplicate episode identity: {identity}")
        identities.add(identity)
        seeds[split].add(record["seed"])
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Prepared episode checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as source:
            arrays = dict(source)
        length = record["frames"]
        for key in ("gt", "detected"):
            if (
                arrays[key].shape != (length, 20, 3)
                or not np.isfinite(arrays[key]).all()
            ):
                raise ValueError(f"Expected finite {key} coordinates [T,20,3]: {path}")
        if arrays["observed"].shape != (length, 20) or arrays["observed"].dtype != bool:
            raise ValueError(f"Expected a boolean observed mask [T,20]: {path}")
        if arrays["tcp"].shape != (length, 3) or not np.isfinite(arrays["tcp"]).all():
            raise ValueError(f"Expected finite TCP coordinates [T,3]: {path}")
        groups[split].append(dict(arrays, seed=record["seed"], mode=record["mode"]))
    if seeds["train"] & seeds["validation"]:
        raise ValueError("Training and validation reset seeds must be disjoint")
    if [len(groups[key]) for key in groups] != [3600, 450]:
        raise ValueError(
            "Expected 3600 train and 450 validation episodes from the 4500-episode release"
        )
    return groups, manifest


def training_fallback(episodes):
    """Fit the unobserved-slot prior exclusively from training GT coordinates."""
    total = np.zeros((20, 3), dtype=np.float64)
    count = 0
    for episode in episodes:
        total += episode["gt"].sum(axis=0, dtype=np.float64)
        count += len(episode["gt"])
    return (total / count).astype(np.float32)


def sequence_arrays(episodes, fallback):
    """Build the handoff model's ten causal features without prediction feedback."""
    lengths = np.asarray([len(episode["gt"]) for episode in episodes])
    shape = (len(episodes), int(lengths.max()), 20)
    arrays = {
        "features": np.zeros((*shape, 10), dtype=np.float32),
        "anchor": np.zeros((*shape, 3), dtype=np.float32),
        "detected": np.zeros((*shape, 3), dtype=np.float32),
        "gt": np.zeros((*shape, 3), dtype=np.float32),
        "observed": np.zeros(shape, dtype=bool),
        "valid": np.zeros(shape[:2], dtype=bool),
        "age": np.zeros(shape, dtype=np.int32),
        "depth_mm": np.zeros(shape[:2], dtype=np.float32),
    }
    for index, episode in enumerate(episodes):
        detected, observed = episode["detected"], episode["observed"]
        length = len(detected)
        time = np.arange(length)[:, None]
        last = np.maximum.accumulate(np.where(observed, time, -1), axis=0)
        seen = last >= 0
        last_xyz = detected[np.maximum(last, 0), np.arange(20)]
        anchor = np.where(seen[..., None], last_xyz, fallback[None])
        age = time - last
        arrays["features"][index, :length] = np.concatenate(
            (
                (anchor - fallback[None]) / 0.1,
                (episode["tcp"][:, None] - anchor) / 0.1,
                observed[..., None],
                seen[..., None],
                (np.minimum(age, 40) / 40)[..., None],
                np.broadcast_to(time[..., None] / 200, (length, 20, 1)),
            ),
            axis=-1,
        )
        for key, values in (
            ("anchor", anchor),
            ("detected", detected),
            ("gt", episode["gt"]),
            ("observed", observed),
            ("age", age),
            ("depth_mm", episode["depth_mm"]),
        ):
            arrays[key][index, :length] = values
        arrays["valid"][index, :length] = True
    return arrays, lengths, [episode["mode"] for episode in episodes]
