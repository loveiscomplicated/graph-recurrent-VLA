"""Compare saved Object21 errors by observed PEG, target and joint node counts."""

import csv
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_diagnostics import cluster_interval
from evaluation.gnn.vertex_family import METRICS, MODES
from evaluation.gnn.vertex_metrics import statistics
from utils.experiment_logger import ExperimentLogger


def count_groups(counts):
    """Keep object counts separate and include observed joint-count cells."""
    peg = counts[:, 0]
    target = counts[:, 1] + counts[:, 2]
    groups = [('all', -1, -1, np.ones(len(counts), dtype=bool)),
              ('target_present', -1, -1, target > 0)]
    for name, values in [('peg', peg), ('target', target), ('total', counts.sum(1))]:
        for value in np.unique(values):
            groups.append((name, int(value), -1, values == value))
    for peg_count, target_count in np.unique(np.column_stack((peg, target)), axis=0):
        groups.append(('joint', int(peg_count), int(target_count),
                       (peg == peg_count) & (target == target_count)))
    return groups


def read_predictions(root, sample_ids, seeds):
    """Require the same ordered frames and GT across all saved natural predictions."""
    predictions = {}
    hashes = {}
    reference = None
    for mode in MODES:
        collected = {metric: [] for metric in METRICS}
        for seed in seeds:
            path = root / f'{mode}_seed_{seed}_natural.npz'
            hashes[path.name] = file_sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['sample_ids'], sample_ids)
                target = archive['target']
                if reference is None:
                    reference = target.copy()
                np.testing.assert_array_equal(target, reference)
                for metric in METRICS:
                    values = archive[metric]
                    if values.shape != (len(sample_ids),) or not np.isfinite(values).all():
                        raise ValueError(f'Invalid per-frame metric {metric}: {path}')
                    collected[metric].append(values.copy())
        predictions[mode] = {metric: np.stack(values) for metric, values in collected.items()}
    return predictions, hashes


def draw_count_curves(rows, output):
    """Plot descriptive means while marking the number of frames at each count."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [('target', 'hole_position_error_mm', 'Hole position (mm)'),
              ('target', 'relative_3d_error_mm', 'Endpoint–Hole relative error (mm)'),
              ('peg', 'hole_position_error_mm', 'Hole position (mm)'),
              ('peg', 'peg_half_size_x_error_mm', 'Peg half-size X (mm)'),
              ('peg', 'peg_position_error_mm', 'Peg position (mm)'),
              ('peg', 'peg_rotation_error_deg', 'Peg rotation (degrees)')]
    names = dict(baseline='M0', detector='M1 Detector', oracle='M2 Oracle', type_control='Capacity control')
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, (count_type, metric, label) in zip(axes.flat, panels):
        for mode in MODES:
            selected = [row for row in rows if row['count_type'] == count_type
                        and row['metric'] == metric and row['mode'] == mode and row['frames'] >= 20]
            axis.plot([row['count'] for row in selected], [row['mean'] for row in selected],
                      marker='o', markersize=4, label=names[mode])
        selected = [row for row in rows if row['count_type'] == count_type
                    and row['metric'] == metric and row['mode'] == 'baseline' and row['frames'] >= 20]
        for row in selected:
            axis.text(row['count'], .97, f"n={row['frames']}", transform=axis.get_xaxis_transform(),
                      ha='center', va='top', fontsize=7)
        axis.set(xlabel=f'Observed {count_type.upper()} nodes', ylabel=label)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.suptitle('Validation: 4,102 frames / 40 parents; mean errors over 3 trained seeds\n'
                    'Groups with n < 20 are omitted from curves and retained in CSV; comparisons are descriptive')
    figure.savefig(output / 'vertex_count_errors.png', dpi=170)
    plt.close(figure)


def draw_object21_curves(rows, output):
    """Export every physical Object21 component against each object's count."""
    import matplotlib.pyplot as plt

    components = [
        ('peg_position_error_mm', 'Peg position (mm)'),
        ('peg_rotation_error_deg', 'Peg rotation (degrees)'),
        ('relative_3d_error_mm', 'Endpoint–Hole relative vector (mm)'),
        ('hole_position_error_mm', 'Hole position (mm)'),
        ('hole_rotation_error_deg', 'Hole rotation (degrees)'),
        ('peg_half_size_x_error_mm', 'Peg half-size X (mm)'),
        ('peg_half_size_y_error_mm', 'Peg half-size Y (mm)'),
        ('peg_half_size_z_error_mm', 'Peg half-size Z (mm)'),
        ('hole_radius_error_mm', 'Hole radius (mm)'),
    ]
    names = dict(baseline='M0', detector='M1 Detector', oracle='M2 Oracle', type_control='Capacity control')
    for count_type in ('target', 'peg'):
        figure, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, components):
            for mode in MODES:
                selected = [row for row in rows if row['count_type'] == count_type
                            and row['metric'] == metric and row['mode'] == mode and row['frames'] >= 20]
                axis.plot([row['count'] for row in selected], [row['mean'] for row in selected],
                          marker='o', markersize=3, label=names[mode])
            axis.set(xlabel=f'Observed {count_type.upper()} nodes', ylabel=label)
            axis.grid(alpha=.25)
            axis.legend(fontsize=7)
        figure.suptitle(f'All Object21 components vs {count_type.upper()} count | 4,102 validation frames / 3 seeds\n'
                        'Groups with fewer than 20 frames are retained in CSV; counts are observational')
        figure.savefig(output / f'{count_type}_object21_errors.png', dpi=170)
        plt.close(figure)


def analyze(config):
    """Aggregate saved predictions without model inference or checkpoint updates."""
    root = Path(config['predictions'])
    count_path = Path(config['counts'])
    seeds = config.get('seeds', [0, 1, 2])
    with np.load(count_path, allow_pickle=False) as archive:
        sample_ids = archive['sample_ids'].copy()
        parents = archive['parent_ids'].copy()
        counts = archive['counts'].copy()
    if counts.shape != (len(sample_ids), 3) or (counts < 0).any():
        raise ValueError('counts must be nonnegative [frame, PEG/BOX/HOLE]')
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError('Count sample IDs must be unique')
    predictions, hashes = read_predictions(root, sample_ids, seeds)
    source_summary = json.loads((root / 'summary.json').read_text())
    if source_summary['groups']['all']['frames'] != len(sample_ids):
        raise ValueError('Count and evaluation frame totals differ')
    with ExperimentLogger(team='graph', config=config, seed=seeds[0],
                          model='vertex_family_count_analysis', job_type='evaluation') as logger:
        output = Path(config['output']['root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        comparisons = []
        for count_type, count, target_count, mask in count_groups(counts):
            context = dict(count_type=count_type, count=count, target_count=target_count,
                           frames=int(mask.sum()), parents=len(set(parents[mask])))
            for metric in METRICS:
                averages = {}
                for mode in MODES:
                    values = predictions[mode][metric].mean(0)
                    averages[mode] = values
                    intervals = cluster_interval(values[mask], parents[mask], 2000)
                    stats = statistics(values[mask])
                    # Recover the published whole-set result as a consistency check.
                    if count_type == 'all':
                        expected = source_summary['groups']['all']['modes'][mode][metric]['mean']
                        np.testing.assert_allclose(stats['mean'], expected, rtol=1e-6, atol=1e-5)
                    rows.append(dict(**context, metric=metric, mode=mode, **stats,
                                     parent_ci_low=intervals['low'] if context['parents'] >= 2 else None,
                                     parent_ci_high=intervals['high'] if context['parents'] >= 2 else None,
                                     **{f'seed_{seed}_mean': float(values[mask].mean())
                                        for seed, values in zip(seeds, predictions[mode][metric])}))
                for reference, candidate in [('baseline', 'detector'), ('baseline', 'oracle'),
                                             ('type_control', 'detector'), ('type_control', 'oracle')]:
                    delta = (averages[reference] - averages[candidate])[mask]
                    intervals = cluster_interval(delta, parents[mask], 2000)
                    comparisons.append(dict(**context, metric=metric, reference=reference, candidate=candidate,
                                            gain_mean=float(delta.mean()), parent_ci_low=intervals['low'] if context['parents'] >= 2 else None,
                                            parent_ci_high=intervals['high'] if context['parents'] >= 2 else None))
        for filename, records in [('count_metrics.csv', rows), ('paired_count_gains.csv', comparisons)]:
            with (output / filename).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        draw_count_curves(rows, output)
        draw_object21_curves(rows, output)
        write_json(output / 'provenance.json', dict(config=config, counts_sha256=file_sha256(count_path),
                   predictions_sha256=hashes, source_sha256=file_sha256(Path(__file__)), source_summary_sha256=file_sha256(root / 'summary.json'),
                   frames=len(sample_ids), parents=len(set(parents)), seeds=seeds,
                   averaging='Average per-frame errors across seeds, then average frames; no ensemble',
                   intervals='2000 Parent-cluster draws conditional on fixed trained seeds',
                   limitations='Count groups are observational; joint count cells do not control geometry/quality'))
        for path in output.iterdir():
            logger.link_output(path)
        logger.record(frames=len(sample_ids), parents=len(set(parents)), metric_rows=len(rows))
        print(f'Vertex family count analysis: {output}', flush=True)
    return output
