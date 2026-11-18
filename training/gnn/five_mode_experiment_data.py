"""Pinned C2 experiment inputs and capped, episode-dispersed training replay."""

from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import tarfile

import numpy as np
import torch

from data_collection.detected_corner_visibility import classify_episode
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.five_mode_surface_audit import pinned_json
from evaluation.gnn.five_mode_endpoint_diagnostics import corner_measurements
from evaluation.gnn.five_mode_subtype_diagnostics import subtype_masks
from training.gnn.packed_five_mode_data import PackedFrameBatches, indexed_frames, read_archive_member, read_splits


def condition_groups(path, manifest, index, split, frame_ids, tolerance_m):
    """Verify C2 membership from detector corners and align subtype masks to model rows."""
    assignments = {(row["seed"], row["policy_mode"]): row["split"] for row in manifest["records"]}
    selected = indexed_frames(index, "condition_2", (split,), assignments)
    names = {row["file"] for row in manifest["archives"]}
    ids, measurements = [], []
    with ExitStack() as stack:
        archives = {}
        for episode in manifest["records"]:
            if episode["split"] != split:
                continue
            key = episode["seed"], episode["policy_mode"]
            frames = selected[key]
            if not frames:
                continue
            receipt = episode["files"]["corners"]
            archive_path = (path.parent / receipt["archive"]).resolve()
            if receipt["archive"] not in names or not archive_path.is_relative_to(path.parent):
                raise ValueError(f"Unexpected experiment corner archive: {archive_path}")
            if archive_path not in archives:
                archives[archive_path] = stack.enter_context(tarfile.open(archive_path, "r:"))
            corners = read_archive_member(archives[archive_path], receipt, "corners")
            if classify_episode(corners, episode["frames"], tolerance_m=tolerance_m)["condition_2"] != frames:
                raise ValueError(f"Detector C2 index differs for {key}")
            measurements.extend(corner_measurements(corners, frames))
            ids.extend(f"{key[0]}:{key[1]}:{frame}" for frame in frames)
    if ids != frame_ids:
        raise ValueError(f"Detector subtype order differs from {split} graph frame IDs")
    return subtype_masks(measurements, tolerance_m)


class DispersedFrameBatches(PackedFrameBatches):
    """Use every train frame once, then replay a bounded uniform or C2-E-only subset."""

    def __init__(self, episodes, device, batch_size, generator, groups, sampling):
        super().__init__(episodes, device, batch_size, generator)
        self.groups = groups
        self.sampling = sampling
        if sampling["mode"] not in ("uniform", "focus"):
            raise ValueError("sampling.mode must be uniform or focus")
        for name in ("max_episode_frames_per_batch", "max_seed_frames_per_batch"):
            if type(sampling[name]) is not int or sampling[name] < 1:
                raise ValueError(f"sampling.{name} must be an integer >= 1")
        self.episode_ids = [frame_id.rsplit(":", 1)[0] for frame_id in self.ids]
        self.seeds = [frame_id.split(":", 1)[0] for frame_id in self.ids]

    def frame_order(self):
        focus = torch.from_numpy(np.flatnonzero(self.groups["C2-E"]))
        if self.sampling["mode"] == "uniform":
            extra = torch.randperm(len(self.ids), generator=self.generator)[:len(focus)]
        else:
            extra = focus
        # Every frame has at most two exposures; both arms have exactly N + N_E draws.
        rows = torch.cat((torch.arange(len(self.ids)), extra))
        return rows[torch.randperm(len(rows), generator=self.generator)]

    def batch_rows(self):
        pending = deque(self.frame_order().tolist())
        while pending:
            batch, seen_rows = [], set()
            episode_counts, seed_counts = Counter(), Counter()
            for _ in range(len(pending)):
                row = pending.popleft()
                episode, seed = self.episode_ids[row], self.seeds[row]
                eligible = (
                    row not in seen_rows
                    and episode_counts[episode] < self.sampling["max_episode_frames_per_batch"]
                    and seed_counts[seed] < self.sampling["max_seed_frames_per_batch"]
                )
                if not eligible:
                    pending.append(row)
                    continue
                batch.append(row)
                seen_rows.add(row)
                episode_counts[episode] += 1
                seed_counts[seed] += 1
                if len(batch) == self.batch_size:
                    break
            yield torch.tensor(batch, dtype=torch.long)

    def __len__(self):
        state = self.generator.get_state()
        try:
            return sum(1 for _ in self.batch_rows())
        finally:
            self.generator.set_state(state)

    def sampling_summary(self):
        """Describe the next epoch without advancing its shuffle RNG."""
        state = self.generator.get_state()
        try:
            batches = list(self.batch_rows())
        finally:
            self.generator.set_state(state)
        rows = torch.cat(batches).numpy()
        exposure = np.bincount(rows, minlength=len(self.ids))
        policy = [self.ids[row].split(":")[1] for row in rows]
        return dict(
            unique_frames=len(self.ids), draws=len(rows), batches=len(batches),
            maximum_exposures_per_frame=int(exposure.max()),
            subtype_draws={name: int(exposure[mask].sum()) for name, mask in self.groups.items()},
            policy_draws=dict(Counter(policy)),
            seed_count=len(set(self.seeds)), episode_count=len(set(self.episode_ids)),
        )


@dataclass
class ExpertData:
    train: DispersedFrameBatches
    validation: PackedFrameBatches
    reference_validation: PackedFrameBatches
    validation_groups: dict
    identity: dict


def verify_same_rows(reference, candidate):
    """Require identical corners, Peg points, TCP, targets and frame IDs before adding Box."""
    if len(reference) != len(candidate):
        raise ValueError("Augmented release changed the selected episodes")
    for (original, original_arrays, original_frames), (augmented, augmented_arrays, augmented_frames) in zip(
        reference, candidate, strict=True,
    ):
        keys = ("seed", "policy_mode", "split", "raw_sha256")
        original_corner_hash = original["files"]["corners"]["sha256"]
        augmented_corner_hash = augmented["files"]["corners"]["sha256"]
        if (any(original.get(key) != augmented.get(key) for key in keys)
                or original_corner_hash != augmented_corner_hash
                or list(original_frames) != list(augmented_frames)
                or any(not np.array_equal(left, right) for left, right in zip(original_arrays, augmented_arrays, strict=True))):
            raise ValueError("Box augmentation must preserve detector corners, Peg tracks, TCP, targets and frame order")


def build_data(config, generator):
    """Load train/validation only, keeping a separate unchanged Peg-only reference."""
    settings = config["data"]
    mode = settings["surface_mode"]
    if mode not in ("peg", "peg_box"):
        raise ValueError("data.surface_mode must be peg or peg_box")
    path, manifest = pinned_json(settings["manifest"])
    reference_path, reference = pinned_json(settings["reference_manifest"])
    _, index = pinned_json(settings["condition_index"])
    if mode == "peg_box":
        _, audit = pinned_json(settings["surface_audit"])
        if (audit["config"]["manifest"]["sha256"] != settings["manifest"]["sha256"]
                or audit["config"]["condition_index"]["sha256"] != settings["condition_index"]["sha256"]):
            raise ValueError("Box-surface audit must describe the candidate release and detector index")
        for split in ("train", "validation"):
            cohort = audit["counts"][split]["condition_2"]
            if not cohort["box_points"] or cohort["episodes_with_object_ids"] != cohort["episodes"]:
                raise ValueError(f"Box surface unavailable for C2/{split}: regenerate typed surface data before this arm")
    seed_splits = {}
    for dataset in (reference, manifest):
        for episode in dataset["records"]:
            if seed_splits.setdefault(episode["seed"], episode["split"]) != episode["split"]:
                raise ValueError("Experiment release changes reset-seed split membership")
    splits = ("train", "validation")
    reference_rows = read_splits(reference_path, reference, splits, "condition_2", True, condition_index=index)
    if path == reference_path:
        peg_rows = reference_rows
    else:
        peg_rows = read_splits(path, manifest, splits, "condition_2", True, condition_index=index)
        for split in splits:
            verify_same_rows(reference_rows[split], peg_rows[split])
    rows = peg_rows if mode == "peg" else read_splits(
        path, manifest, splits, "condition_2", True, condition_index=index, include_box_surface=True,
    )
    device = torch.device(config["device"])
    loaders, groups = {}, {}
    identity = dict(surface_mode=mode, manifest_sha256=settings["manifest"]["sha256"],
                    reference_manifest_sha256=settings["reference_manifest"]["sha256"],
                    condition_index_sha256=settings["condition_index"]["sha256"],
                    sampling=config["sampling"], frame_ids_sha256={},
                    loader_sha256=file_sha256(Path(__file__)))
    for split in splits:
        batch_size = config["training"]["batch_size"] if split == "train" else config["evaluation"]["batch_size"]
        loader = PackedFrameBatches(rows[split], device, batch_size, None)
        cohort = settings["cohorts"][split]
        frame_hash = canonical_hash(loader.ids)
        if len(loader.ids) != cohort["frames"] or frame_hash != cohort["frame_ids_sha256"]:
            raise ValueError(f"Experiment {split} frames differ from the pinned C2 cohort")
        groups[split] = condition_groups(path, manifest, index, split, loader.ids, config["coplanarity_tolerance_m"])
        identity["frame_ids_sha256"][split] = frame_hash
        loaders[split] = loader
    train = DispersedFrameBatches(rows["train"], device, config["training"]["batch_size"],
                                  generator, groups["train"], config["sampling"])
    reference_validation = PackedFrameBatches(reference_rows["validation"], device,
                                              config["evaluation"]["batch_size"], None)
    identity["dataset_sha256"] = canonical_hash(identity)
    return ExpertData(train, loaders["validation"], reference_validation, groups["validation"], identity)
