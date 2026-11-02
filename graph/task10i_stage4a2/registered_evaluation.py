"""Apply the already registered adoption rules to completed paired artifacts.

This evaluator cannot run a model, change a threshold, or select frames. Its
output is a statistical gate receipt, not an authorization to replace production.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .aggregate_evaluation import csv_rows
from .evaluation_metrics import hierarchical_paired
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read, write, sha


def composite(parts, metric, criteria):
    """Equal weight supported slots/objects *within* each physical parent."""
    parents = defaultdict(list)
    for part in parts:
        for row in part['parent_values']:
            parents[row['parent_id']].append((row['baseline'], row['candidate']))
    if not parents:
        return dict(metric=metric, paired_parents=0, status='NO_SUPPORT')
    keys = sorted(parents)
    values = np.stack([np.mean(parents[p], axis=0) for p in keys])
    delta = values[:, 1] - values[:, 0]
    boot = criteria['bootstrap']
    rng = np.random.default_rng(boot['seed'])
    draws = delta[rng.integers(len(delta), size=(boot['replicates'], len(delta)))].mean(axis=1)
    return dict(metric=metric, baseline=float(values[:, 0].mean()),
        candidate=float(values[:, 1].mean()), delta=float(delta.mean()),
        ci95=np.quantile(draws, boot['interval']).tolist(), paired_parents=len(keys),
        parent_values=[dict(parent_id=p, baseline=float(v[0]), candidate=float(v[1]),
                            supported_components=len(parents[p])) for p, v in zip(keys, values)])


def derived_pixel_rows(rows, candidate):
    """Empty predictions retain undefined precision; damage gets its own rate."""
    lookup = {(r['episode'], r['timestep'], r['camera'], r['object_id'], r['condition']): r for r in rows}
    assert len(lookup) == len(rows), 'duplicate physical pixel observation'
    result = []
    for row in rows:
        r = dict(row)
        if r['reference_kind'] != 'EXACT_GT':
            result.append(r)
            continue
        n, v = r['predicted_pixels'], r['visible_pixels']
        r['visible_wrong_mask_rate'] = float(n > 0 and r['precision'] < .8) if v >= 8 else None
        r['false_pixels_per_visible_GT_pixel'] = r['fp'] / v if v >= 8 else None
        r['precision_defined'] = float(n > 0)
        b = lookup[r['episode'], r['timestep'], r['camera'], r['object_id'], 'B0']
        r['healthy_tracking_damage'] = (float(r['condition'] == candidate and r['correct_mask_availability'] != 1)
            if b['correct_mask_availability'] == 1 else None)
        result.append(r)
    return result


def t0_both_missing(rows, candidate):
    lookup = {(r['episode'], r['camera'], r['condition']): r for r in rows
              if r['timestep'] == 0 and r['object_id'] == 0}
    output = []
    for episode in sorted({r['episode'] for r in rows}):
        for condition in ('B0', candidate):
            pair = [lookup[episode, c, condition] for c in ('base_camera', 'hand_camera')]
            if all(r['reference_kind'] == 'EXACT_GT' and r['visible_pixels'] >= 8 for r in pair):
                output.append(dict(episode=episode, parent_id=pair[0]['parent_id'], timestep=0,
                    condition=condition, reference_kind='EXACT_GT',
                    both_visible_camera_t0_peg_missing=float(all(r['tp'] == 0 for r in pair))))
    return output


def statistics(pixel, graph, candidate, criteria):
    kwargs = dict(candidate=candidate, bootstrap=criteria['bootstrap']['replicates'],
                  seed=criteria['bootstrap']['seed'])
    pixel = derived_pixel_rows(pixel, candidate)
    pixel_lookup = {(r['episode'], r['timestep'], r['condition'], r['camera'], r['object_id']): r for r in pixel}
    graph = [dict(r) for r in graph]
    for r in graph:
        cameras = ('base_camera', 'hand_camera') if r['object_id'] == 0 else ('base_camera',)
        sources = [pixel_lookup[r['episode'], r['timestep'], r['condition'], c, r['object_id']] for c in cameras]
        visible = sum(v['visible_pixels'] for v in sources) if r['reference_kind'] == 'EXACT_GT' else None
        r['graph_false_object_presence'] = float(r['geometry_points'] > 0) if visible == 0 else None
    slots = {tuple(v) for v in criteria['primary']['slots']}
    relevant = [r for r in pixel if (r['camera'], r['object_id']) in slots]
    summaries = []
    for metric in ('correct_mask_availability', 'precision', 'visible_wrong_mask_rate',
                   'false_pixels_per_visible_GT_pixel', 'healthy_tracking_damage', 'false_presence'):
        summaries += [dict(scope='pixel', **r) for r in hierarchical_paired(relevant, metric, **kwargs)]
    for metric in ('correct_source_coverage', 'robot_point_fraction', 'background_point_fraction',
                   'wrong_object_point_fraction', 'correct_representative_fraction',
                   'peg_ends_coverage', 'box_observed_rim_coverage', 'correct_boundary_coverage', 'graph_false_object_presence'):
        summaries += [dict(scope='graph', **r) for r in hierarchical_paired(graph, metric, ('object_id',), **kwargs)]
    summaries += [dict(scope='t0', **r) for r in hierarchical_paired(t0_both_missing(relevant, candidate),
        'both_visible_camera_t0_peg_missing', (), **kwargs)]
    primary = composite([r for r in summaries if r['metric'] == 'correct_mask_availability'],
                        criteria['primary']['name'], criteria)
    downstream = composite([r for r in summaries if r['metric'] == 'correct_source_coverage'],
                           criteria['downstream']['name'], criteria)
    counts = []
    for camera, oid in sorted(slots):
        for condition in ('B0', candidate):
            selected = [r for r in relevant if r['camera'] == camera and r['object_id'] == oid
                        and r['condition'] == condition and r['reference_kind'] == 'EXACT_GT']
            n = sum(r['tp'] + r['fp'] for r in selected)
            tp = sum(r['tp'] for r in selected)
            counts.append(dict(camera=camera, object_id=oid, condition=condition,
                exact_frames=len(selected), visible_frames=sum(r['visible_pixels'] >= 8 for r in selected),
                undefined_precision_frames=sum(r['precision'] is None for r in selected),
                visible_wrong_masks=sum(r['visible_wrong_mask_rate'] == 1 for r in selected),
                healthy_B0_frames=sum(r.get('healthy_tracking_damage') is not None for r in selected),
                damaged_healthy_frames=sum(r.get('healthy_tracking_damage') == 1 for r in selected),
                pooled_precision=tp/n if n else None, tp=tp, fp=n-tp))
    return summaries, primary, downstream, counts


def gates(summaries, primary, downstream, criteria):
    output = []
    def check(name, value, op, limit, **scope):
        valid = value is not None and np.isfinite(value)
        passed = valid and {'>': lambda: value > limit, '>=': lambda: value >= limit,
                           '<=': lambda: value <= limit}[op]()
        output.append(dict(name=name, value=value, comparison=op, limit=limit,
            status=('PASS' if passed else 'FAIL') if valid else 'INCONCLUSIVE', **scope))
    pc, dc, sc = criteria['primary'], criteria['downstream'], criteria['safety']
    check('primary_absolute_gain', primary.get('delta'), '>=', pc['minimum_absolute_improvement'])
    headroom = 1-primary['baseline'] if 'baseline' in primary else 0
    check('primary_relative_error_reduction', primary.get('delta', 0)/headroom if headroom > 0 else None,
          '>=', pc['minimum_relative_error_reduction'])
    check('primary_lower_CI', primary.get('ci95', [None])[0], '>', pc['ci95_lower_delta_must_exceed'])
    check('downstream_absolute_gain', downstream.get('delta'), '>=', dc['minimum_absolute_improvement'])
    check('downstream_lower_CI', downstream.get('ci95', [None])[0], '>=', dc['ci95_lower_delta_minimum'])
    rules = {
        'precision': ('precision_conditional_paired_defined_ci95_lower_delta_minimum', 0, '>='),
        'visible_wrong_mask_rate': ('visible_wrong_mask_rate_ci95_upper_delta_maximum', 1, '<='),
        'false_pixels_per_visible_GT_pixel': ('false_pixels_per_visible_GT_pixel_ci95_upper_delta_maximum', 1, '<='),
        'robot_point_fraction': ('graph_robot_point_fraction_ci95_upper_delta_maximum', 1, '<='),
        'background_point_fraction': ('graph_background_point_fraction_ci95_upper_delta_maximum', 1, '<='),
        'wrong_object_point_fraction': ('graph_wrong_object_point_fraction_ci95_upper_delta_maximum', 1, '<='),
        'correct_representative_fraction': ('graph_correct_representative_fraction_ci95_lower_delta_minimum', 0, '>='),
        'peg_ends_coverage': ('peg_end_coverage_ci95_lower_delta_minimum', 0, '>='),
        'box_observed_rim_coverage': ('box_observed_rim_coverage_ci95_lower_delta_minimum', 0, '>='),
        'correct_boundary_coverage': ('correct_boundary_coverage_ci95_lower_delta_minimum', 0, '>='),
        'false_presence': ('false_object_presence_ci95_upper_delta_maximum', 1, '<='),
        'graph_false_object_presence': ('false_object_presence_ci95_upper_delta_maximum', 1, '<='),
        'both_visible_camera_t0_peg_missing': ('both_visible_camera_t0_peg_missing_ci95_upper_delta_maximum', 1, '<='),
        'healthy_tracking_damage': ('healthy_tracking_damage_ci95_upper_rate_maximum', 1, '<='),
    }
    expected = []
    for metric in rules:
        if metric in ('peg_ends_coverage',): expected.append((metric, None, 0))
        elif metric == 'box_observed_rim_coverage': expected.append((metric, None, 1))
        elif metric == 'both_visible_camera_t0_peg_missing': expected.append((metric, None, None))
        elif metric in ('robot_point_fraction', 'background_point_fraction', 'wrong_object_point_fraction',
                        'correct_representative_fraction', 'correct_boundary_coverage', 'graph_false_object_presence'):
            expected.extend((metric, None, oid) for oid in (0, 1))
        else: expected.extend((metric, cam, oid) for cam, oid in map(tuple, pc['slots']))
    for metric, camera, oid in expected:
        matches = [r for r in summaries if r['metric'] == metric and r.get('camera') == camera and r.get('object_id') == oid]
        assert len(matches) <= 1
        key, side, comparison = rules[metric]
        check(key, matches[0]['ci95'][side] if matches else None, comparison, sc[key],
              metric=metric, camera=camera, object_id=oid, paired_parents=matches[0]['paired_parents'] if matches else 0)
    return output


def run(output, paired, expected, criteria_path, candidate, scope):
    output.mkdir(parents=True, exist_ok=False)
    criteria = read(criteria_path)
    assert criteria['status'] == 'FROZEN'
    assert read(paired/'pipeline_state.json')['status'] == 'PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
    manifest = read(expected)
    pixel, graph, inputs = [], [], []
    frames = exact = 0
    for e in manifest['entries']:
        folder = paired/'episodes'/str(e['episode_id'])
        receipt = read(folder/'receipt.json')
        assert receipt.get('candidate', 'A') == candidate
        assert receipt['raw_sha256'] == e['raw_sha256']
        assert receipt['frames'] == e['actions']+1
        assert all(sha(folder/p) == h for p, h in receipt['files'].items()), 'artifact readback failed'
        pixel.extend(csv_rows(folder/'pixel_metrics.csv'))
        graph.extend(csv_rows(folder/'graph_metrics.csv'))
        frames += receipt['frames']; exact += receipt['exact_frames']
        inputs.append(dict(path=str(folder/'receipt.json'), sha256=sha(folder/'receipt.json')))
    assert len(inputs) == manifest['episodes'] and frames == manifest['frames']
    assert {r['parent_id'] for r in pixel} == {e['parent_id'] for e in manifest['entries']}
    summaries, primary, downstream, counts = statistics(pixel, graph, candidate, criteria)
    checks = gates(summaries, primary, downstream, criteria)
    write(output/'registered_parent_metrics.json', summaries)
    write(output/'composite_reliability_metrics.json', dict(primary=primary, downstream=downstream))
    csv_write(output/'false_positive_safety_audit.csv', counts)
    csv_write(output/'registered_gate_results.csv', checks)
    write(output/'statistical_gate_receipt.json', dict(status='REGISTERED_STATISTICAL_GATES_EVALUATED',
        scope=scope, candidate=candidate, parents=manifest['parents'], episodes=len(inputs), frames=frames,
        exact_frames=exact, unsupported_frames=frames-exact, criteria_sha256=sha(criteria_path),
        source_sha256=sha(Path(__file__)), expected_sha256=sha(expected), inputs=inputs,
        primary={k:v for k,v in primary.items() if k != 'parent_values'},
        downstream={k:v for k,v in downstream.items() if k != 'parent_values'},
        failures=[r for r in checks if r['status']=='FAIL'],
        unsupported=[r for r in checks if r['status']=='INCONCLUSIVE'],
        passed=sum(r['status']=='PASS' for r in checks), total=len(checks),
        runtime='NOT_EVALUATED_BY_THIS_STATISTICAL_MODULE', hard_contracts='SEPARATE_SOURCE_AND_GRAPH_RECEIPTS_REQUIRED',
        decision='NOT_A_FINAL_DECISION', production_replacement_authorized=False, whole_goal_complete=False,
        precision_empty_imputed=False, population_rare_error_bound_certified=False,
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    from utils.experiment_logger import ExperimentLogger
    parser = argparse.ArgumentParser()
    for name in ('output', 'paired', 'expected', 'criteria'): parser.add_argument(name, type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--scope', required=True, choices=('development', 'internal', 'final_normal', 'final_stress'))
    args = parser.parse_args()
    with ExperimentLogger(team='integration', model='Task10I_Stage4A2_registered_statistical_evaluation',
            seed=1011202643, root=args.output.parent/'experiments', job_type='evaluation',
            config=dict(candidate=args.candidate, scope=args.scope, criteria_sha256=sha(args.criteria),
                        source_sha256=sha(Path(__file__)), GT_posthoc_only=True)) as logger:
        run(args.output, args.paired, args.expected, args.criteria, args.candidate, args.scope)
        receipt=read(args.output/'statistical_gate_receipt.json');receipt['run_id']=logger.run_id
        write(args.output/'statistical_gate_receipt.json', receipt)
        logger.link_output(args.output/'statistical_gate_receipt.json', kind='metrics')
