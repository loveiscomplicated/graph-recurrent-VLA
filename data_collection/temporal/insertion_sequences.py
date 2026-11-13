"""Parent-safe insertion splits and shared causal GRU/Mamba sequence windows."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data_collection.temporal.pose_sequences import TemporalPoseDataset
from data_collection.temporal_gate_data import make_gate_manifest, validate_gate_splits
from graph.temporal.insertion import feature_names
from graph.temporal.pose_observations import observation_age_step
from training.gnn.cached_data import build_data


class InsertionEpisodeDataset(TemporalPoseDataset):
    """Append observed-support ages without exposing labels to the feature builder."""

    def __init__(self, data):
        super().__init__(data)
        self.feature_names = feature_names(self.with_robot)

    def __getitem__(self, index):
        observations, target = super().__getitem__(index)
        age = None
        ages = []
        for quality in self.data["pose_quality"][self.indices[index]]:
            age = observation_age_step(torch.as_tensor(quality)[None], age)
            ages.append(age[0])
        return torch.cat((observations, torch.stack(ages)), dim=-1), target


class InsertionWindowDataset:
    """Nonoverlapping causal windows; recurrent state resets at each window.

    Proprioceptive deltas and observation ages are constructed on complete episodes
    first. No future sample enters an input, and no window crosses an episode.
    This is explicitly truncated context, not state-carrying TBPTT. Both backbones
    use this same reset policy; full streaming evaluation retains episode context.
    """

    def __init__(self, episodes, sequence_length):
        if type(sequence_length) is not int or sequence_length < 1:
            raise ValueError("training.sequence_length must be a positive integer")
        self.episodes = episodes
        # Frozen observations never change during training. Build complete causal
        # features once rather than rebuilding an episode for each of its windows.
        self.sequences = [episodes[index] for index in range(len(episodes))]
        self.windows = [
            (index, start, min(start + sequence_length, len(rows)))
            for index, rows in enumerate(episodes.indices)
            for start in range(0, len(rows), sequence_length)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start, stop = self.windows[index]
        observations, target = self.sequences[episode]
        return observations[start:stop], target[start:stop]


def validate_insertion_splits(manifest, source_splits, independent_splits=None):
    """Reserve all previous GNN evaluation parents and reject hidden sibling leakage."""
    role = manifest.get("test_role")
    if role == "gnn_validation_reference":
        return validate_gate_splits(manifest, source_splits)
    if role != "independent" or independent_splits is None:
        raise ValueError(
            "Independent test requires test_role=independent and independent_cache_dir"
        )
    splits = manifest["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Insertion splits must contain train, validation and test")
    parents = manifest["parent_by_episode"]
    source_episodes = {episode for values in source_splits.values() for episode in values}
    external_episodes = {episode for values in independent_splits.values() for episode in values}
    if source_episodes & external_episodes:
        raise ValueError("Independent cache episode IDs must be distinct from all source episodes")
    for episode in source_episodes | external_episodes:
        if not isinstance(parents.get(episode), str) or not parents[episode]:
            raise ValueError(f"Missing explicit parent identity: {episode}")
    source_parents = {parents[episode] for episode in source_episodes}
    if source_parents & {parents[episode] for episode in external_episodes}:
        raise ValueError("Independent test has a parent previously used by the GNN")
    # Reuse all existing development/reserved-parent validation, substituting only
    # the historical reference test. This does not materialize or read its frames.
    development = dict(manifest, splits=dict(splits, test=source_splits["validation"]))
    validate_gate_splits(development, source_splits)
    allowed_test = set(independent_splits["train"]) | set(independent_splits["validation"])
    test = splits["test"]
    if (
        not isinstance(test, list)
        or not test
        or len(set(test)) != len(test)
        or not set(test) <= allowed_test
    ):
        raise ValueError(
            "Independent test needs unique episodes materialized in the independent cache"
        )
    return splits


def select_insertion_loaders(
    source_config, manifest_path, requested, generator, independent_cache_dir=None
):
    """Check source/cache inventories first; only requested split frames are read later."""
    from copy import deepcopy

    source = build_data(source_config, generator)
    manifest = json.loads(Path(manifest_path).read_text())
    external = None
    if manifest.get("test_role") == "independent":
        if independent_cache_dir is None:
            raise ValueError("data.independent_cache_dir is required for an independent test")
        # Metadata is sufficient to validate parents during training; the external
        # cache inventory and tensors are opened only for an explicit test run.
        metadata = json.loads(
            (Path(independent_cache_dir) / "metadata/data_identity.json").read_text()
        )
        independent_splits = metadata["splits"]
        if "test" in requested:
            external_config = deepcopy(source_config)
            external_config["data"]["cache_dir"] = independent_cache_dir
            external = build_data(external_config, generator)
    else:
        independent_splits = None
    splits = validate_insertion_splits(manifest, source.identity["splits"], independent_splits)
    loaders = {}
    for split in requested:
        candidates = [source.train] if split != "test" else [source.validation]
        if split == "test" and external is not None:
            candidates = [external.train, external.validation]
        selected = set(splits[split])
        subsets = []
        found = set()
        for candidate in candidates:
            indices = []
            for index, sample in enumerate(candidate.dataset.samples):
                episode = str(Path(sample.sample_id).parent)
                if episode in selected:
                    indices.append(index)
                    found.add(episode)
            if indices:
                subsets.append(Subset(candidate.dataset, indices))
        if found != selected:
            raise ValueError(f"{split} episodes have missing cache entries: {selected - found}")
        dataset = torch.utils.data.ConcatDataset(subsets)
        template = candidates[0]
        loaders[split] = DataLoader(
            dataset,
            shuffle=False,
            batch_size=template.batch_size,
            collate_fn=template.collate_fn,
            num_workers=template.num_workers,
        )
    identity = dict(source.identity)
    if external is not None:
        identity["independent_dataset_sha256"] = external.identity["dataset_sha256"]
        identity["independent_frames"] = external.identity["frames"]
    return loaders, identity, manifest


def main():
    """Create an explicit split manifest from metadata and user-supplied actual parents."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--independent-cache-dir")
    parser.add_argument("--parent-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = json.loads((Path(args.cache_dir) / "metadata/data_identity.json").read_text())
    parents = json.loads(Path(args.parent_map).read_text())
    manifest = make_gate_manifest(source["splits"], parents, args.validation_fraction, args.seed)
    manifest["test_role"] = "gnn_validation_reference"
    independent = None
    if args.independent_cache_dir:
        metadata = json.loads(
            (Path(args.independent_cache_dir) / "metadata/data_identity.json").read_text()
        )
        independent = metadata["splits"]
        manifest["splits"]["test"] = sorted(independent["train"] + independent["validation"])
        manifest["test_role"] = "independent"
    validate_insertion_splits(manifest, source["splits"], independent)
    with Path(args.output).open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
