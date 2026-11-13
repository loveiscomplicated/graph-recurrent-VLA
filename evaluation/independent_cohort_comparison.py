"""Compare the frozen unfiltered test and the separate matched-visibility follow-up."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_collection.episode_dataset import file_sha256
from evaluation.independent_test_report import reference_summary
from utils.experiment_logger import ExperimentLogger


def model_identities(protocol):
    """Return the ordered model-seed and checkpoint identities for a test."""
    return [(model['seed'], model['gnn_sha256'], model['temporal_sha256'])
            for model in protocol['models']]


def compare(config):
    """Check independent seed/model identities and plot precomputed held-out metrics."""
    results = {name: json.loads(Path(config[f'{name}_results']).read_text())
               for name in ('unfiltered', 'matched')}
    protocols = {name: json.loads(Path(config[f'{name}_protocol']).read_text())
                 for name in ('unfiltered', 'matched')}
    for name in results:
        if results[name]['protocol_sha256'] != file_sha256(config[f'{name}_protocol']):
            raise ValueError(f'{name} results refer to another test protocol')
    if set(protocols['unfiltered']['seeds']) & set(protocols['matched']['seeds']):
        raise ValueError('Unfiltered and matched test seeds must be disjoint')
    if model_identities(protocols['unfiltered']) != model_identities(protocols['matched']):
        raise ValueError('Test cohorts use different frozen models')
    reference = reference_summary(config['references'])
    with ExperimentLogger(team='graph', model='independent_cohort_comparison', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
        names = ('expert', 'random_after_lift')
        positions = np.arange(len(names))
        for axis, metric, title in zip(axes, ('peg_position_error_mm', 'hole_position_error_mm'), ('Peg', 'Hole')):
            for offset, cohort, color, label in ((-.18, 'unfiltered', '#d2564a', 'Unfiltered test'),
                                                  (.18, 'matched', '#2d9a66', 'Matched-visibility follow-up')):
                means = [results[cohort]['summary'][name]['post_lift']['metrics'][metric]['mean'] for name in names]
                axis.bar(positions + offset, means, width=.36, color=color, label=label)
            axis.axhline(reference[metric]['mean'], linestyle='--', color='#345b9c',
                         label='Development mean, all frames')
            axis.set_xticks(positions, ['Expert', 'Random after lift'])
            axis.set_ylabel('Frame-weighted mean position error (mm)')
            axis.set_title(title)
            axis.grid(axis='y', alpha=.2)
        axes[1].legend(fontsize=8)
        figure.suptitle('Three-model mean errors: test after lift versus development reference')
        image = output / 'cohort_comparison.png'
        figure.savefig(image, dpi=150)
        plt.close(figure)
        summary = dict(config=config, reference_development=reference,
                       cohorts={name: results[name]['summary'] for name in results},
                       test_seed_overlap=0,
                       caveat='The matched cohort was preregistered after observing the first test; development uses all frames, tests show post-lift frames.')
        path = output / 'comparison.json'
        path.write_text(json.dumps(summary, indent=2))
        logger.record(test_seed_overlap=0)
        logger.link_output(path)
        logger.link_output(image)
        print(f'Cohort comparison: {output}', flush=True)
