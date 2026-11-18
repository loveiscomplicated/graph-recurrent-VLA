"""Load a verified supervised Five-mode cohort into device-resident frame arrays."""

from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from graph.build_graph import NodeType
from training.detected_corner_gnn import graph_batch
from training.gnn.data import DataBundle, TrainingBatch
from utils.device_set import set_device


def read_archive_member(archive, receipt, kind):
    """Read only the checksum-pinned detector, surface or target episode."""
    member = PurePosixPath(receipt["member"])
    # A full release prefixes each member with its collection batch.
    if (member.is_absolute() or ".." in member.parts or len(member.parts) not in (2, 3)
            or member.parent.name != kind
            or member.suffix != ".npz" or not archive.getmember(str(member)).isfile()):
        raise ValueError(f"Invalid {kind} cache member: {member}")
    with archive.extractfile(str(member)) as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Training cache checksum differs: {member}")
    with np.load(BytesIO(data), allow_pickle=False) as array:
        return dict(array)


def pack_episode(episode, tables, include_surface, *, include_box_surface=False):
    """Keep detector corners and append Peg tracks, or explicitly typed Peg/Box tracks."""
    if include_box_surface and not include_surface:
        raise ValueError("Box surface input also requires include_surface=True")
    frames = episode["frames"]
    if any(not np.array_equal(table["frame_indices"], np.arange(frames))
           for table in tables.values()):
        raise ValueError(f"Unaligned targets and observations for seed {episode['seed']}")
    corners, targets = tables["corners"], tables["targets"]
    corner_offsets = corners["corner_offsets"]
    counts = np.diff(corner_offsets)
    if include_surface:
        surface = tables["surface"]
        surface_offsets = surface["surface_offsets"]
        surface_xyz = surface["surface_world_xyz"]
        object_ids = surface.get("surface_object_id")
        if include_box_surface:
            if (object_ids is None or object_ids.shape != (len(surface_xyz),)
                    or object_ids.dtype.kind not in "iu" or not np.isin(object_ids, (0, 1)).all()):
                raise ValueError("Peg+Box experiments require explicit surface_object_id for every point")
        elif object_ids is not None:
            peg = object_ids == NodeType.PEG
            frame_ids = np.repeat(np.arange(frames), np.diff(surface_offsets))
            surface_xyz = surface_xyz[peg]
            surface_offsets = np.r_[0, np.bincount(frame_ids[peg], minlength=frames).cumsum()]
        counts = counts + np.diff(surface_offsets)
    width = max(1, int(counts.max()))
    xyz = np.zeros((frames, width, 3), dtype=np.float32)
    kinds = np.full((frames, width), NodeType.PAD, dtype=np.int8)
    for frame in range(frames):
        start, stop = corner_offsets[frame:frame + 2]
        length = stop - start
        xyz[frame, :length] = corners["corner_world_xyz"][start:stop]
        kinds[frame, :length] = corners["corner_object_id"][start:stop]
        if include_surface:
            left, right = surface_offsets[frame:frame + 2]
            xyz[frame, length:counts[frame]] = surface_xyz[left:right]
            kinds[frame, length:counts[frame]] = (
                object_ids[left:right] if include_box_surface else NodeType.PEG
            )
    tcp, object21 = targets["tcp"], targets["object21"]
    if (tcp.shape != (frames, 3) or object21.shape != (frames, 21)
            or not all(np.isfinite(value).all() for value in (xyz, tcp, object21))
            or not np.isin(kinds, (NodeType.PAD, NodeType.PEG, NodeType.BOX)).all()):
        raise ValueError(f"Invalid supervised frame arrays for seed {episode['seed']}")
    return xyz, kinds, tcp, object21


class PackedFrameBatches:
    """Shuffle frame indices on the CPU and build graph batches on the device."""

    def __init__(self, episodes, device, batch_size, generator):
        count = sum(len(frames) for _, _, frames in episodes)
        width = max(arrays[0].shape[1] for _, arrays, _ in episodes)
        xyz = np.zeros((count, width, 3), dtype=np.float32)
        kinds = np.full((count, width), NodeType.PAD, dtype=np.int8)
        ids = []
        start = 0
        for episode, (points, types, _, _), frames in episodes:
            stop = start + len(points)
            xyz[start:stop, :points.shape[1]] = points
            kinds[start:stop, :points.shape[1]] = types
            prefix = f"{episode['seed']}:{episode['policy_mode']}:"
            ids.extend(prefix + str(frame) for frame in frames)
            start = stop
        self.xyz = torch.as_tensor(xyz, device=device)
        self.kinds = torch.as_tensor(kinds, device=device)
        self.tcp = torch.as_tensor(np.concatenate([row[1][2] for row in episodes]), device=device)
        self.target = torch.as_tensor(np.concatenate([row[1][3] for row in episodes]), device=device)
        self.ids = ids
        self.device = device
        self.batch_size = batch_size
        self.generator = generator

    def __len__(self):
        return (len(self.ids) + self.batch_size - 1) // self.batch_size

    def frame_order(self):
        if self.generator is None:
            return torch.arange(len(self.ids))
        return torch.randperm(len(self.ids), generator=self.generator)

    def __iter__(self):
        for host_rows in self.batch_rows():
            rows = host_rows.to(self.device)
            kinds = self.kinds[rows].long()
            valid = kinds != NodeType.PAD
            graph, _ = graph_batch(self.xyz[rows], kinds, valid, valid, self.tcp[rows])
            yield TrainingBatch(graph, {"object21": self.target[rows]},
                                [self.ids[i] for i in host_rows.tolist()])

    def batch_rows(self):
        """Yield CPU row indices; specialized samplers may emit smaller dispersed batches."""
        return self.frame_order().split(self.batch_size)


class BalancedFrameBatches(PackedFrameBatches):
    """Sample two disjoint conditions in a fixed ratio in every batch."""

    def __init__(self, primary, secondary, device, batch_size, generator,
                 *, ratio=(1, 1), frames_per_epoch=None):
        if len(ratio) != 2 or any(type(weight) is not int or weight < 1 for weight in ratio):
            raise ValueError("Condition ratio needs two positive integer weights")
        self.ratio = ratio
        group_size = sum(ratio)
        if batch_size % group_size:
            raise ValueError("Batch size must be divisible by the sum of condition weights")
        self.primary_count = sum(len(frames) for _, _, frames in primary)
        self.secondary_count = sum(len(frames) for _, _, frames in secondary)
        available = min(self.primary_count // ratio[0], self.secondary_count // ratio[1])
        if frames_per_epoch is not None and (
            type(frames_per_epoch) is not int or frames_per_epoch % group_size
            or not 0 < frames_per_epoch <= available * group_size
        ):
            raise ValueError("Epoch frame count must fit both conditions without replacement")
        groups = available if frames_per_epoch is None else frames_per_epoch // group_size
        if not groups:
            raise ValueError("Both conditions need enough training frames for one group")
        self.primary_frames = groups * ratio[0]
        self.secondary_frames = groups * ratio[1]
        super().__init__(primary + secondary, device, batch_size, generator)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Balanced condition training requires disjoint frame IDs")

    def __len__(self):
        frames = self.primary_frames + self.secondary_frames
        return (frames + self.batch_size - 1) // self.batch_size

    def frame_order(self):
        primary = torch.randperm(self.primary_count, generator=self.generator)[:self.primary_frames]
        secondary = torch.randperm(self.secondary_count, generator=self.generator)[:self.secondary_frames]
        return torch.cat((primary.reshape(-1, self.ratio[0]),
                          secondary.reshape(-1, self.ratio[1]) + self.primary_count), dim=1).flatten()


def indexed_frames(condition_index, condition, splits, records):
    """Resolve an external visibility index against the manifest's episode splits."""
    frames = {}
    for split in splits:
        for row in condition_index[condition][split]:
            key = row["seed"], row["policy_mode"]
            if key in frames or records.get(key) != split:
                raise ValueError(f"Unexpected or duplicate {condition}/{split} episode: {key}")
            frames[key] = row["frames"]
    expected = {key for key, split in records.items() if split in splits}
    if frames.keys() != expected:
        raise ValueError(f"Incomplete episode coverage in the {condition} index")
    return frames


def read_splits(path, manifest, splits_to_read, condition, include_surface, *, condition_index=None,
                include_box_surface=False):
    """Decode only the requested frames from the pinned supervised cohort."""
    records = manifest["records"]
    episode_splits = {(episode["seed"], episode["policy_mode"]): episode["split"]
                      for episode in records}
    if (manifest["format"] != "five_mode_supervised_v1"
            or len(episode_splits) != len(records) or len(records) != manifest["episodes"]):
        raise ValueError(f"Expected a unique Five-mode supervised manifest: {path}")
    if condition not in ("all", "condition_1", "condition_2", "condition_3"):
        raise ValueError(f"Unknown Five-mode visibility selection: {condition}")
    selections = None
    if condition_index is not None and condition != "all":
        selections = indexed_frames(condition_index, condition, splits_to_read, episode_splits)
    names = ("corners", "targets", "surface") if include_surface else ("corners", "targets")
    archive_names = {row["file"] for row in manifest["archives"]}
    splits = {split: [] for split in splits_to_read}
    with ExitStack() as stack:
        archives = {}
        for episode in records:
            split = episode["split"]
            if split not in splits:
                continue
            if condition == "all":
                frames = range(episode["frames"])
            elif selections is None:
                frames = episode["conditions"][condition]
            else:
                frames = selections[episode["seed"], episode["policy_mode"]]
            if list(frames) != sorted(set(frames)) or any(
                frame < 0 or frame >= episode["frames"] for frame in frames
            ):
                raise ValueError(f"Invalid {condition} frame index for seed {episode['seed']}")
            if not frames:
                continue
            arrays = {}
            for kind in names:
                receipt = episode["files"][kind]
                name = receipt["archive"]
                archive = (path.parent / name).resolve()
                if name not in archive_names or not archive.is_relative_to(path.parent):
                    raise ValueError(f"Invalid supervised archive: {name}")
                if name not in archives:
                    archives[name] = stack.enter_context(tarfile.open(archive, "r:"))
                arrays[kind] = read_archive_member(archives[name], receipt, kind)
            packed = pack_episode(episode, arrays, include_surface,
                                  include_box_surface=include_box_surface)
            selected = tuple(array[frames] for array in packed) if condition != "all" else packed
            splits[split].append((episode, selected, frames))
    return splits


def build_data(config, shuffle_generator):
    """Read the frozen train/validation splits and leave the test split reserved."""
    settings = config["data"]
    path = Path(settings["manifest"]).resolve()
    manifest = json.loads(path.read_text())
    include_surface = settings["include_surface"]
    condition = settings.get("condition", "all")
    index_path = settings.get("condition_index")
    index = json.loads(Path(index_path).read_text()) if index_path is not None else None
    splits = read_splits(path, manifest, ("train", "validation"), condition, include_surface,
                         condition_index=index)
    for split, episodes in splits.items():
        actual = sum(len(frames) for _, _, frames in episodes)
        expected = settings.get("expected_frames", {}).get(split)
        if condition == "all":
            expected = sum(row["frames"] for row in manifest["records"] if row["split"] == split)
        if not actual or (expected is not None and actual != expected):
            raise ValueError(f"Unexpected {split} frame count: {actual}")
        print(f"Loaded {condition}/{split}: {actual:,} frames from {len(episodes):,} episodes", flush=True)

    device = set_device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(settings.get("matmul_precision", "highest"))
    batch_size = config["training"]["batch_size"]
    identity = dict(manifest_sha256=file_sha256(path), include_surface=include_surface,
                    surface_object="peg" if include_surface else None, condition=condition)
    if index_path is not None:
        identity["condition_index_sha256"] = file_sha256(Path(index_path))
    balanced_condition = settings.get("balanced_train_condition")
    if balanced_condition is None:
        train = PackedFrameBatches(splits["train"], device, batch_size, shuffle_generator)
    else:
        if "all" in (condition, balanced_condition) or balanced_condition == condition:
            raise ValueError("Balanced training needs two distinct visibility conditions")
        secondary = read_splits(path, manifest, ("train",), balanced_condition, include_surface,
                                condition_index=index)["train"]
        mix = settings.get("train_mix")
        options = {} if mix is None else dict(
            ratio=(mix["weights"][condition], mix["weights"][balanced_condition]),
            frames_per_epoch=mix["frames_per_epoch"],
        )
        train = BalancedFrameBatches(splits["train"], secondary, device, batch_size,
                                     shuffle_generator, **options)
        identity["balanced_train_condition"] = balanced_condition
        if mix is None:
            identity["frames_per_condition"] = train.primary_frames
        else:
            identity["train_mix"] = mix
        print(f"Mixed train per epoch: {condition}={train.primary_frames:,}, "
              f"{balanced_condition}={train.secondary_frames:,}; per-batch ratio={train.ratio}", flush=True)
    return DataBundle(
        train=train,
        validation=PackedFrameBatches(splits["validation"], device, batch_size, None),
        identity=dict(identity, dataset_sha256=canonical_hash(identity)),
    )
