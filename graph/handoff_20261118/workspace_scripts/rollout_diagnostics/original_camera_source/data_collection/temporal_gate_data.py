"""Explicit parent-disjoint temporal-gate selection over published graph caches."""

import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from training.gnn.cached_data import build_data


def validate_gate_splits(manifest, source_splits):
    """Keep gate development inside GNN train; reserve GNN validation for comparison.

    parent_by_episode is explicit provenance: use the episode ID itself only when
    the source collection confirms each episode is an independent parent.
    """
    splits = manifest['splits']
    parents = manifest['parent_by_episode']
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Gate splits must contain train, validation and test')
    seen, parent_split = set(), {}
    for split, episodes in splits.items():
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f'Gate {split} needs a nonempty episode list')
        allowed = set(source_splits['validation' if split == 'test' else 'train'])
        for episode in episodes:
            if episode in seen or episode not in allowed:
                raise ValueError(f'Overlapping or forbidden GNN split episode: {episode}')
            seen.add(episode)
            parent = parents.get(episode)
            if not isinstance(parent, str) or not parent:
                raise ValueError(f'Missing source parent identity for {episode}')
            if parent in parent_split and parent_split[parent] != split:
                raise ValueError(f'Parent leakage across gate splits: {parent}')
            parent_split[parent] = split
    if set(splits['test']) != set(source_splits['validation']):
        raise ValueError('Gate test must include the complete original GNN validation split')
    # Even unused siblings cannot hide a GNN validation/test parent in development.
    for split in ('validation', 'test'):
        for episode in source_splits[split]:
            parent = parents.get(episode)
            if not isinstance(parent, str) or not parent:
                raise ValueError(f'Missing reserved episode parent identity: {episode}')
            if parent_split.get(parent) in ('train', 'validation'):
                raise ValueError(f'Reserved GNN parent appears in gate development: {parent}')
    return splits


def selected_loaders(source_config, generator, manifest_path, requested):
    """Index/check the original cache and select episodes before reading any frame."""
    bundle = build_data(source_config, generator)
    manifest = json.loads(Path(manifest_path).read_text())
    splits = validate_gate_splits(manifest, bundle.identity['splits'])
    loaders = {}
    for split in requested:
        source = bundle.validation if split == 'test' else bundle.train
        selected = set(splits[split])
        indices = [index for index, sample in enumerate(source.dataset.samples)
                   if str(Path(sample.sample_id).parent) in selected]
        observed = {str(Path(source.dataset.samples[index].sample_id).parent) for index in indices}
        if observed != selected:
            raise ValueError(f'Gate {split} episodes are missing cached frames')
        loaders[split] = DataLoader(Subset(source.dataset, indices), shuffle=False,
                                   batch_size=source.batch_size, collate_fn=source.collate_fn,
                                   num_workers=source.num_workers)
    return loaders, bundle.identity, manifest


def make_gate_manifest(source_splits, parents, validation_fraction=0.2, seed=0):
    """Divide eligible GNN-train parents; exclude siblings of reserved GNN episodes."""
    import random

    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be in (0,1)')
    for episodes in source_splits.values():
        for episode in episodes:
            if not isinstance(parents.get(episode), str) or not parents[episode]:
                raise ValueError(f'Missing explicit source parent identity: {episode}')
    reserved_parents = {parents[episode] for split in ('validation', 'test')
                        for episode in source_splits[split]}
    eligible = sorted({parents[episode] for episode in source_splits['train']} - reserved_parents)
    if len(eligible) < 2:
        raise ValueError('Need at least two independent GNN-train parents outside reserved parents')
    random.Random(seed).shuffle(eligible)
    validation_count = max(1, min(len(eligible) - 1, round(len(eligible) * validation_fraction)))
    validation_parents = set(eligible[:validation_count])
    training_parents = set(eligible[validation_count:])
    manifest = {'splits': {
        'train': sorted(episode for episode in source_splits['train'] if parents[episode] in training_parents),
        'validation': sorted(episode for episode in source_splits['train'] if parents[episode] in validation_parents),
        'test': sorted(source_splits['validation']),
    }, 'parent_by_episode': parents, 'seed': seed, 'validation_fraction': validation_fraction}
    validate_gate_splits(manifest, source_splits)
    return manifest


def main():
    """Write a new, reviewable split manifest without reading frames or fitting models."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--parent-map', required=True,
                        help='JSON episode-ID -> actual collection parent-ID mapping')
    parser.add_argument('--output', required=True)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()
    metadata = json.loads((Path(arguments.cache_dir) / 'metadata/data_identity.json').read_text())
    parents = json.loads(Path(arguments.parent_map).read_text())
    manifest = make_gate_manifest(metadata['splits'], parents, arguments.validation_fraction, arguments.seed)
    with Path(arguments.output).open('x', encoding='utf-8') as destination:
        json.dump(manifest, destination, indent=2)
        destination.write('\n')


if __name__ == '__main__':
    main()
