"""Read Five-mode pose labels and identify recently detected physical corners."""

from io import BytesIO
import json
from pathlib import PurePosixPath
import subprocess
import tarfile

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.spatial_corner_tracking import VERTICES
from data_collection.state21_vertex_targets import POSE_FIELDS, corners_from_pose, match_vertices


def read_raw_poses(path, digest, episodes):
    """Stream the pinned RGB-D archive, decoding only requested GT pose fields."""
    if file_sha256(path) != digest:
        raise ValueError(f"Raw Five-mode archive checksum differs: {path}")
    expected = {(row['seed'], row['policy_mode']): row for row in episodes}
    poses, directories = {}, {}
    seen = {key: set() for key in expected}
    with subprocess.Popen(["zstd", "-qdc", str(path)], stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name == PurePosixPath("episode_index.json") and member.isfile():
                    index = json.load(archive.extractfile(member))
                    for row in index:
                        key = row["seed"], row["policy_mode"]
                        if key not in expected:
                            continue
                        directory = PurePosixPath(row["path"])
                        count = row["observations"]
                        if (directory.is_absolute() or ".." in directory.parts
                                or directory in directories or key in poses
                                or count != expected[key]["frames"] or row["split"] != "validation"):
                            raise ValueError(f"Invalid raw episode index for {key}: {path}")
                        poses[key] = np.empty((count, sum(size for _, size in POSE_FIELDS)), np.float32)
                        directories[directory] = key
                    continue
                key = directories.get(name.parent)
                if key is None or not member.isfile() or not name.name.startswith("frame_"):
                    continue
                frame = int(name.stem.removeprefix("frame_"))
                if name.suffix != ".npz" or not 0 <= frame < len(poses[key]) or frame in seen[key]:
                    raise ValueError(f"Unexpected raw Five-mode frame: {name}")
                with np.load(BytesIO(archive.extractfile(member).read()), allow_pickle=False) as raw:
                    poses[key][frame] = np.concatenate([
                        raw[f"ground_truth/{field}"][0].reshape(size) for field, size in POSE_FIELDS
                    ])
                seen[key].add(frame)
        # Consume the tar padding so zstd can finish writing and verify the stream.
        while process.stdout.read(65536):
            pass
        if process.wait() != 0:
            raise ValueError(f"Could not decompress raw Five-mode archive: {path}")
    if poses.keys() != expected.keys() or any(len(seen[key]) != len(values) for key, values in poses.items()):
        raise ValueError(f"Raw Five-mode frames are incomplete: {path}")
    return poses


def recent_corner_candidates(corners, poses, *, max_age, gate_m, margin_m):
    """Rank missing IDs by age and previous match distance, without downstream labels."""
    last_seen = np.full(len(VERTICES), -max_age - 1, dtype=np.int32)
    last_error = np.full(len(VERTICES), np.inf)
    vertex_index = {vertex: index for index, vertex in enumerate(VERTICES)}
    vertex_kinds = np.asarray([vertex[0] for vertex in VERTICES])
    offsets = corners["corner_offsets"]
    for frame, pose in enumerate(poses):
        references = corners_from_pose(pose)
        reference_xyz = np.asarray([references[key] for key in VERTICES])
        rows = slice(*offsets[frame:frame + 2])
        xyz, kinds = corners["corner_world_xyz"][rows], corners["corner_object_id"][rows]
        observed = {(int(kind), index): point for index, (kind, point)
                    in enumerate(zip(kinds, xyz, strict=True))}
        matches = match_vertices(observed, references, gate_m=gate_m, margin_m=margin_m)
        for detection, reference in matches.items():
            vertex = vertex_index[reference]
            last_seen[vertex] = frame
            last_error[vertex] = np.linalg.norm(observed[detection] - references[reference])
        ages = frame - last_seen
        eligible = (ages > 0) & (ages <= max_age)
        # An ambiguous nearby detection could already represent this corner.
        if eligible.any() and len(xyz):
            nearby = np.linalg.norm(xyz[:, None] - reference_xyz[None], axis=-1) <= gate_m
            eligible &= ~(nearby & (kinds[:, None] == vertex_kinds[None])).any(axis=0)
        ranked = np.lexsort((np.arange(len(VERTICES)), last_error, ages))
        selected = ranked[eligible[ranked]]
        yield selected, ages[selected], reference_xyz[selected]
