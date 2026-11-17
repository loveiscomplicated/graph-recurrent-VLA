"""Use a selected-frame M0 with the original contiguous graph episodes."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data_collection.episode_dataset import file_sha256
from training.gnn.cached_data import build_data
from training.gnn.data import canonical_hash
from training.gnn.trainer import build_model, resolve_config
from data_collection.vertex_family_manifest import read_manifest


def prepare_selected_sequences(effective, device, requested):
    """Freeze M0 and select disjoint complete episodes from its original cache."""
    settings = effective['selected_m0']
    checkpoint = torch.load(effective['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint.get('kind') not in ('final_weights', 'best_weights'):
        raise ValueError('Expected selected M0 final_weights or best_weights checkpoint')
    selected = read_manifest(settings['dataset'])
    identity_parts = [selected['dataset_sha256'], 'baseline']
    validation_selection = checkpoint['config']['data'].get('validation_selection')
    if validation_selection:
        identity_parts.append({'validation_selection_sha256': file_sha256(validation_selection),
                               'rule': 'both_pass'})
    if canonical_hash(identity_parts) != checkpoint['dataset_sha256']:
        raise ValueError('M0 checkpoint and selected dataset identities differ')
    reference = torch.load(settings['cache_source_checkpoint'], map_location='cpu', weights_only=True)
    source = resolve_config(reference['config'])
    if source['data']['factory'] != 'training.gnn.cached_data:build_data':
        raise ValueError('Expected a published cache source checkpoint')
    source['data']['cache_dir'] = effective['data']['cache_dir']
    source['training']['batch_size'] = effective['evaluation']['batch_size']
    bundle = build_data(source, torch.Generator().manual_seed(effective['seed']))
    identity = bundle.identity
    if identity['dataset_sha256'] != reference['dataset_sha256']:
        raise ValueError('Cache does not match its source checkpoint')
    if selected['source_dataset_sha256'] != identity['source_dataset_sha256']:
        raise ValueError('Selected M0 dataset and contiguous graph cache have different origins')

    manifest = json.loads(Path(effective['split_manifest']).read_text())
    parents = manifest['parent_by_episode']
    splits = manifest['splits']
    source_episodes = set(identity['splits']['train'])
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation and test episode splits')
    seen_episodes, seen_parents = set(), set()
    for split in ('train', 'validation', 'test'):
        episodes = splits[split]
        if not episodes or len(set(episodes)) != len(episodes) or not set(episodes) <= source_episodes:
            raise ValueError(f'Invalid {split} episode list')
        split_parents = {parents[episode] for episode in episodes}
        if seen_episodes.intersection(episodes) or seen_parents.intersection(split_parents):
            raise ValueError('Temporal splits overlap by episode or collection parent')
        seen_episodes.update(episodes)
        seen_parents.update(split_parents)
    expected_test = {row['parent_id'] for row in selected['samples'] if row['split'] == 'validation'}
    if {parents[episode] for episode in splits['test']} != expected_test:
        raise ValueError('Temporal test does not match the selected M0 validation parents')

    loaders = {}
    for split in requested:
        wanted = set(splits[split])
        indices = [i for i, sample in enumerate(bundle.train.dataset.samples)
                   if str(Path(sample.sample_id).parent) in wanted]
        found = {str(Path(bundle.train.dataset.samples[i].sample_id).parent) for i in indices}
        if found != wanted:
            raise ValueError(f'Cached episodes missing from {split}')
        loaders[split] = DataLoader(Subset(bundle.train.dataset, indices), shuffle=False,
                                    batch_size=bundle.train.batch_size,
                                    collate_fn=bundle.train.collate_fn, num_workers=0)
    model = build_model(checkpoint['config']['model'], device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    repository = Path(__file__).resolve().parents[3]
    source_files = ('training/gnn/temporal/selected_m0.py',
                    'training/gnn/temporal/insertion.py',
                    'training/gnn/temporal/selected_estimator.py',
                    'evaluation/gnn/temporal/insertion.py',
                    'graph/temporal/insertion.py',
                    'graph/temporal/causal_transformer.py')
    effective.update(source_checkpoint_sha256=file_sha256(effective['checkpoint']),
                     split_manifest_sha256=file_sha256(effective['split_manifest']),
                     temporal_split_manifest=manifest, source_training_config=source,
                     source_sha256={name: file_sha256(repository / name) for name in source_files})
    return model, loaders, identity
