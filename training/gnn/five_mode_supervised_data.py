"""Read the recovered Five-mode targets with the existing observed-graph collator."""

from bisect import bisect_right
from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_collection.episode_dataset import file_sha256
from training.gnn.data import DataBundle
from training.gnn.visibility_data import collate


class FiveModeSupervisedDataset(Dataset):
    """Read a selected frame, keeping a bounded cache of decoded episodes."""

    def __init__(self, manifest_path, index_path, split, condition, *, include_surface):
        self.root = Path(manifest_path).resolve().parent
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["format"] != "five_mode_supervised_v1":
            raise ValueError("Expected the recovered Five-mode supervised manifest")
        self.archives = {row["file"] for row in manifest["archives"]}
        self.include_surface = include_surface
        records = {(row["seed"], row["policy_mode"]): row for row in manifest["records"]}
        if len(records) != manifest["episodes"]:
            raise ValueError("Duplicate Five-mode episode identity")
        selections = json.loads(Path(index_path).read_text())[condition][split]
        self.rows = []
        self.offsets = [0]
        for entry in selections:
            key = entry["seed"], entry["policy_mode"]
            episode = records[key]
            frames = entry["frames"]
            if (episode["split"] != split or frames != sorted(set(frames))
                    or any(frame < 0 or frame >= episode["frames"] for frame in frames)):
                raise ValueError(f"Invalid {condition} frame selection for {key}")
            if frames:
                self.rows.append((episode, frames))
                self.offsets.append(self.offsets[-1] + len(frames))
        self._stack = None
        self._handles = {}
        self._decoded = OrderedDict()

    def __len__(self):
        return self.offsets[-1]

    def close(self):
        """Close all shard files opened by this dataset instance."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._handles.clear()
        self._decoded.clear()

    def _read(self, receipt):
        name = receipt["archive"]
        member = PurePosixPath(receipt["member"])
        path = (self.root / name).resolve()
        if (name not in self.archives or not path.is_relative_to(self.root)
                or member.is_absolute() or ".." in member.parts or member.suffix != ".npz"):
            raise ValueError("Invalid recovered cache location")
        if self._stack is None:
            self._stack = ExitStack()
        if name not in self._handles:
            self._handles[name] = self._stack.enter_context(tarfile.open(path, "r:"))
        with self._handles[name].extractfile(member.as_posix()) as stream:
            content = stream.read()
        if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
            raise ValueError(f"Recovered cache checksum differs: {member}")
        with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
            return dict(arrays)

    def _episode(self, position):
        if position not in self._decoded:
            episode = self.rows[position][0]
            names = ("corners", "targets", "surface") if self.include_surface else ("corners", "targets")
            data = {name: self._read(episode["files"][name]) for name in names}
            indices = np.arange(episode["frames"])
            if any(not np.array_equal(table["frame_indices"], indices) for table in data.values()):
                raise ValueError("Recovered cache frame indices differ")
            self._decoded[position] = data
            if len(self._decoded) > 32:
                self._decoded.popitem(last=False)
        self._decoded.move_to_end(position)
        return self._decoded[position]

    def __getitem__(self, item):
        if not 0 <= item < len(self):
            raise IndexError(item)
        position = bisect_right(self.offsets, item) - 1
        episode, frames = self.rows[position]
        frame = frames[item - self.offsets[position]]
        data = self._episode(position)
        corners, targets = data["corners"], data["targets"]
        rows = slice(*corners["corner_offsets"][frame:frame + 2])
        xyz, kinds = corners["corner_world_xyz"][rows], corners["corner_object_id"][rows]
        if self.include_surface:
            surface = data["surface"]
            rows = slice(*surface["surface_offsets"][frame:frame + 2])
            points = surface["surface_world_xyz"][rows]
            point_kinds = surface.get("surface_object_id")
            point_kinds = (np.zeros(len(points), dtype=kinds.dtype) if point_kinds is None
                           else point_kinds[rows].astype(kinds.dtype, copy=False))
            xyz = np.concatenate((xyz, points))
            kinds = np.concatenate((kinds, point_kinds))
        identity = f'{episode["seed"]}:{episode["policy_mode"]}:{frame}'
        return tuple(torch.from_numpy(value) for value in (xyz, kinds, targets["tcp"][frame],
                                                           targets["object21"][frame])) + (identity,)


def build_data(config, shuffle_generator):
    """Use paired labels with detector corners and optional surface tracks."""
    settings = config["data"]
    batch_size = config["training"]["batch_size"]
    inputs = settings["manifest"], settings["condition_index"]
    condition = settings.get("condition", "condition_1")
    use_surface = settings.get("include_surface", False)
    train = FiveModeSupervisedDataset(*inputs, "train", condition, include_surface=use_surface)
    validation = FiveModeSupervisedDataset(*inputs, "validation", condition, include_surface=use_surface)
    if not len(train) or not len(validation):
        raise ValueError(f"Empty training or validation selection for {condition}")
    return DataBundle(
        train=DataLoader(train, batch_size=batch_size, shuffle=True,
                         generator=shuffle_generator, collate_fn=collate),
        validation=DataLoader(validation, batch_size=batch_size, collate_fn=collate),
        identity=dict(manifest_sha256=file_sha256(Path(inputs[0])),
                      visibility_index_sha256=file_sha256(Path(inputs[1])),
                      include_surface=use_surface, condition=condition),
    )
