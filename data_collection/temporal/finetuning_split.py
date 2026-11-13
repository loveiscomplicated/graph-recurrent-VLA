"""Parent-disjoint fine-tuning partitions inside the original Mamba train set."""

import numpy as np


def make_finetuning_split(original, *, seed, development_fraction):
    """Reserve the old validation/test; never describe pretraining data as held out."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be in (0, 1)")
    parents = original["parent_by_episode"]
    original_sets = {name: set(values) for name, values in original["splits"].items()}
    if not {"train", "validation", "test"} <= set(original_sets):
        raise ValueError("Original manifest must contain train, validation and test")
    parent_sets = {}
    for name, episodes in original_sets.items():
        if len(episodes) != len(original["splits"][name]) or not episodes:
            raise ValueError(f"Original {name} episodes must be nonempty and unique")
        for episode in episodes:
            if not isinstance(parents.get(episode), str) or not parents[episode]:
                raise ValueError(f"Missing explicit parent: {episode}")
        parent_sets[name] = {parents[episode] for episode in episodes}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if original_sets[left] & original_sets[right] or parent_sets[left] & parent_sets[right]:
            raise ValueError(f"Original {left}/{right} episode or parent overlap")
    candidates = sorted(parent_sets["train"])
    if len(candidates) < 2:
        raise ValueError("Fine-tuning requires at least two original training parents")
    shuffled = np.random.default_rng(seed).permutation(candidates)
    count = min(len(candidates) - 1, max(1, round(len(candidates) * development_fraction)))
    development_parents = set(shuffled[:count])
    development = sorted(
        episode for episode in original_sets["train"] if parents[episode] in development_parents
    )
    train = sorted(original_sets["train"] - set(development))
    return {
        "splits": {
            "train": train,
            "development": development,
            "benchmark": sorted(original_sets["validation"]),
            "reserved_reference": sorted(original_sets["test"]),
        },
        "parent_by_episode": dict(parents),
        "development_role": "fine-tuning development; previously seen by pretrained Mamba/Gate/GNN",
        "benchmark_role": "reused 126-episode development benchmark; never gradient or selection data",
        "independent_test": False,
        "seed": seed,
        "development_fraction": development_fraction,
    }


def select_episode_data(data, episodes):
    """Select whole episodes with the same row mask for every observation and label."""
    selected = np.isin(data["episode"], episodes)
    if set(data["episode"][selected]) != set(episodes):
        raise ValueError("Requested fine-tuning episodes are missing from the source data")
    return {name: values[selected].copy() for name, values in data.items()}
