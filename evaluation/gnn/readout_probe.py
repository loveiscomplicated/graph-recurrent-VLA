"""Paired, parent-clustered analysis of completed frozen-readout experiments."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_diagnostics import group_masks
from evaluation.gnn.vertex_metrics import (
    cluster_interval, load_aligned_predictions, statistics,
)
from utils.experiment_logger import ExperimentLogger


VARIANTS = ('attention', 'multiseed', 'attention_capacity', 'multiseed_one', 'multiseed_eight')


def compare(config):
    """Compare retrained heads separately from the original pretrained baseline."""
    root = Path(config['runs'])
    summaries = {}
    arrays = {}
    for path in root.rglob('summary.json'):
        summary = json.loads(path.read_text())
        variant = summary.get('variant')
        if variant not in VARIANTS:
            continue
        key = (variant, summary['seed'])
        if key in summaries:
            raise ValueError(f'Duplicate completed probe: {key}')
        with np.load(path.parent/'predictions.npz',allow_pickle=False) as archive:
            data = dict(archive)
        summaries[key] = dict(summary, local_path=str(path.parent.resolve()),
                              prediction_sha256=file_sha256(path.parent/'predictions.npz'))
        arrays[key] = data
    if not summaries:
        raise ValueError('No completed frozen-readout runs found')
    if config.get('expected_seeds'):
        expected = set(config['expected_seeds'])
        for variant in config.get('required_variants', ['attention', 'multiseed']):
            actual = {seed for name, seed in arrays if name == variant}
            if actual != expected:
                raise ValueError(f'Incomplete {variant} seed set: expected {expected}, got {actual}')
    original, original_hashes = load_aligned_predictions(config['original_predictions'])
    baseline = original['baseline']
    parents = json.loads(Path(config['parent_manifest']).read_text())['parent_by_episode']
    episodes = np.asarray([str(Path(identifier).parent) for identifier in baseline['sample_ids']])
    parent_ids = np.asarray([parents[episode] for episode in episodes])
    groups = group_masks(baseline['counts'])
    metrics = [key for key in baseline if key.endswith(('_error_mm','_error_deg'))]
    comparisons = {}
    per_model = {}
    for key,data in arrays.items():
        for field in ('sample_ids','target','counts'):
            if not np.array_equal(data[field],baseline[field]):
                raise ValueError(f'Probe frame, label or count mismatch: {key}/{field}')
        per_model[f'{key[0]}_seed_{key[1]}'] = {
            group: {'frames':int(mask.sum()),'episodes':len(set(episodes[mask])),
                    'parents':len(set(parent_ids[mask])),
                    'metrics':{metric:statistics(data[metric][mask]) for metric in metrics}}
            for group,mask in groups.items()
        }
    pairs = [(variant, 'attention') for variant in VARIANTS[1:]]
    pairs.append(('multiseed', 'attention_capacity'))
    for variant, reference_variant in pairs:
        seeds = sorted(seed for name,seed in arrays
                       if name == variant and (reference_variant, seed) in arrays)
        if not seeds:
            continue
        records = {}
        differences = defaultdict(list)
        for seed in seeds:
            reference = summaries[(reference_variant,seed)]
            candidate = summaries[(variant,seed)]
            for field in ('feature_sha256','training_order_sha256','updates'):
                if reference[field] != candidate[field]:
                    raise ValueError(f'Controlled probe mismatch: {variant}/seed={seed}/{field}')
            if reference.get('initial_head_sha256') and candidate.get('initial_head_sha256'):
                if reference['initial_head_sha256'] != candidate['initial_head_sha256']:
                    raise ValueError(f'Fresh head initialization differs: {variant}/seed={seed}')
            record = {}
            for group,mask in groups.items():
                values = {}
                for metric in metrics:
                    delta = arrays[(variant,seed)][metric] - arrays[(reference_variant,seed)][metric]
                    values[metric] = {'candidate_minus_reference':statistics(delta[mask])}
                    if group in ('all','high_support_proxy','low_support_proxy','missing_target'):
                        values[metric]['parent_ci95'] = cluster_interval(
                            delta[mask],parent_ids[mask],config.get('bootstrap_repetitions',2000))
                    if group=='all':
                        differences[metric].append(delta)
                record[group] = values
            records[str(seed)] = record
        averaged = {}
        for metric,values in differences.items():
            means = np.asarray([value.mean() for value in values])
            averaged[metric] = {
                'seed_mean_delta':float(means.mean()),
                'seed_sd_delta':float(means.std(ddof=1)) if len(means)>1 else None,
                'seed_deltas':means.tolist(),
                'parent_ci95_conditional_on_seeds':cluster_interval(
                    np.mean(values,axis=0),parent_ids,config.get('bootstrap_repetitions',2000)),
            }
        comparison_name = variant if reference_variant == 'attention' else f'{variant}_vs_{reference_variant}'
        comparisons[comparison_name] = dict(reference_variant=reference_variant,
                                             candidate_variant=variant, seeds=seeds,
                                             per_seed=records, averaged=averaged)
    means = {}
    disagreement = {}
    for variant in VARIANTS:
        seeds = sorted(seed for name,seed in arrays if name==variant)
        if not seeds:
            continue
        stacked = np.stack([arrays[(variant, seed)]['object21'][:, 17] * 1000 for seed in seeds])
        disagreement[variant] = {
            group: statistics(stacked.std(axis=0, ddof=1)[mask])
            for group, mask in groups.items()
        } if len(seeds) > 1 else None
        means[variant] = {'seeds':seeds,'trainable_parameters':summaries[(variant,seeds[0])]['trainable_parameters'],
                          'metrics':{}}
        for metric in metrics:
            values=np.asarray([arrays[(variant,seed)][metric].mean() for seed in seeds])
            means[variant]['metrics'][metric] = {
                'seed_mean':float(values.mean()),
                'seed_sd':float(values.std(ddof=1)) if len(values)>1 else None,
                'per_seed':values.tolist(),
            }
    result = dict(
        runs={f'{name}_seed_{seed}':value for (name,seed),value in summaries.items()},
        original_prediction_sha256=original_hashes, models=per_model,
        seed_summary=means, paired_comparisons=comparisons,
        size_prediction_sd_across_training_seeds_mm=disagreement,
        pretrained_reference={metric:statistics(baseline[metric]) for metric in metrics},
        uncertainty='Whole-parent bootstrap; intervals condition on the listed fixed encoder and training seeds.',
    )
    with ExperimentLogger(team='graph',config=config,seed=0,
                          model='frozen_readout_comparison',job_type='evaluate') as experiment:
        output=Path(config['output']['root']).resolve()/experiment.run_id
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'summary.json',result)
        write_json(output/'effective_config.json',config)
        experiment.link_output(output/'summary.json',kind='metrics')
        experiment.link_output(output/'effective_config.json',kind='metrics')
        print(f'Frozen readout comparison: {output}',flush=True)
        return output
