"""Task03-D Phase 1: diagnose ONLY saved frames 0/120 and S0/S2/S3.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_rim_bottleneck

No sampler, normal estimator, candidate extraction, FPS or simulator is called.
Results go to a new ExperimentLogger UUID directory; inputs remain read-only.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace

import numpy as np

from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import build_organized_geometry
from visual.runtime.rgbd import BOX, PEG, attach_rgbd
from visual.runtime.test.normal_diagnostics import endpoints
from visual.runtime.test.sampling_evaluation import (
    CoverageMatcher, count_ratio, evaluation_context, staged_reference_coverage,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE_RUN = '2551448b0fae4c6bbce3caf48774275f'
CONDITIONS = dict(S0='S0_n256_r0_t20_s0', S2='S2_n256_r0.25_t20_s0',
                  S3='S3_n256_r0.25_t20_s0')
FRAMES = (0, 120)
TARGETS = {0: (103, 103, (6, 16, 3)), 120: (41, 8, (3, 0, 2))}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: {actual!r} != {expected!r}')


def source_contract(source, config):
    """Fail closed on historical evaluation changes; permit additive diagnostics.

    Every original top-level function and every original CoverageMatcher method
    must retain its AST. Evaluation imports/constants remain unchanged too.
    """
    records = {}
    for relative, expected in config['source_sha256'].items():
        saved, current = source/'evaluated_source'/relative, REPO/relative
        require_equal(digests(saved)[1], expected, f'source snapshot {relative}')
        saved_tree, current_tree = ast.parse(saved.read_text()), ast.parse(current.read_text())
        if relative.endswith('/sampling_evaluation.py'):
            functions = {n.name: n for n in current_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for node in saved_tree.body:
                if isinstance(node, ast.ClassDef):
                    replacement = functions[node.name]
                    methods = {n.name: n for n in replacement.body if isinstance(n, ast.FunctionDef)}
                    replacement = ast.ClassDef(name=node.name, bases=replacement.bases,
                        keywords=replacement.keywords, body=[methods[n.name] if isinstance(n, ast.FunctionDef)
                            else n for n in node.body], decorator_list=replacement.decorator_list)
                elif isinstance(node, ast.FunctionDef):
                    replacement = functions[node.name]
                else:
                    if ast.dump(node) not in [ast.dump(n) for n in current_tree.body]:
                        raise ValueError(f'historical evaluation import/constant changed: {relative}')
                    replacement = node
                require_equal(ast.dump(replacement), ast.dump(node), f'historical evaluation definition {relative}')
            status = 'original definitions unchanged; additive diagnostics only'
        elif relative.endswith('/test_sampling_evaluation.py'):
            status = 'diagnostic tests extended; not an evaluation dependency'
        else:
            require_equal(ast.dump(current_tree), ast.dump(saved_tree), f'historical source AST {relative}')
            status = 'byte identical' if digests(current)[1] == expected else 'AST identical (formatting only)'
        records[relative] = dict(original_sha256=expected, current_sha256=digests(current)[1], status=status)
    return records


def geometry_evidence(frame):
    """Only the mask/validity fields read by the unchanged evaluation_context.

    endpoints(support, ~valid_depth) is the original in-image four-neighbour
    depth_validity_boundary rule. No normals/features/candidates are extracted.
    """
    geometry = build_organized_geometry(frame)
    objects = {oid: SimpleNamespace(object_mask=geometry.object_masks[oid],
        depth_validity_boundary=endpoints(geometry.observed_mask(oid), ~geometry.valid_depth))
        for oid in (PEG, BOX)}
    return SimpleNamespace(frame=SimpleNamespace(geometry=geometry, objects=objects))


def validate_saved_selection(archive, key, frame, geometry):
    """Exact source raster, object support and float32 XYZ equality, no epsilon."""
    require_equal(int(archive['source_frame']), frame.source_frame, 'source_frame')
    np.testing.assert_array_equal(archive['shape'], frame.depth.shape)
    flat, vu = archive[key+'/flat_index'], archive[key+'/pixels_vu']
    oid, points = archive[key+'/object_id'], archive[key+'/points']
    priority = archive[key+'/priority_selected']
    if flat.ndim != 1 or flat.dtype.kind not in 'iu' or len(np.unique(flat)) != len(flat):
        raise ValueError('invalid/duplicate saved source pixels')
    if ((flat < 0) | (flat >= frame.depth.size)).any():
        raise ValueError('source pixel outside raster')
    np.testing.assert_array_equal(vu, np.column_stack(np.unravel_index(flat, frame.depth.shape)))
    np.testing.assert_array_equal(flat, vu[:, 0]*frame.depth.shape[1]+vu[:, 1])
    require_equal(len(oid), len(flat), 'object ID length')
    if not np.isin(oid, [PEG, BOX]).all() or priority.shape != flat.shape or priority.dtype != bool:
        raise ValueError('invalid object IDs/selection roles')
    np.testing.assert_array_equal(points, geometry.xyz.reshape(-1, 3)[flat])
    for object_id in (PEG, BOX):
        if not geometry.observed_mask(object_id).ravel()[flat[oid == object_id]].all():
            raise ValueError('saved selection outside valid unambiguous object support')
    if archive[key+'/depth_priority_selected'].any():
        raise ValueError('unexpected depth priority in S0/S2/S3')
    return dict(source_frame=frame.source_frame, selected_nodes=len(flat),
                xyz_exact=True, max_abs_xyz_error_m=0., raster_and_object_support=True)


def depth_summary(values):
    values = np.asarray(values)
    if len(values) and not (np.isfinite(values) & (values > 0)).all():
        raise ValueError('nonpositive/nonfinite depth in observed support')
    return dict(count=len(values), min=float(values.min()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def occupancy(frame, node_sets, negatives):
    """Per-path node occupancy; overlapping ROIs and their deduplicated union."""
    masks = dict(negatives)
    masks['known_negative_union'] = np.logical_or.reduce(list(negatives.values()))
    x = node_sets['X']
    roi = {}
    for name, mask in masks.items():
        observed = int(mask.ravel()[x].sum())
        roi[name] = {}
        for stage, flat in node_sets.items():
            if flat is None:
                roi[name][stage] = None
                continue
            count = int(mask.ravel()[flat].sum())
            roi[name][stage] = dict(count=count, fraction_of_path=count_ratio(count, len(flat)),
                                   fraction_of_observed_roi=count_ratio(count, observed))
    depth = {name: depth_summary(frame.depth.ravel()[flat]) if flat is not None else None
             for name, flat in node_sets.items()}
    # Observation-derived quantile bands describe depth occupancy without an
    # arbitrary far cutoff or reclassifying ordinary surface nodes as errors.
    bounds = depth['X']
    bands = {}
    if len(x):
        for stage, flat in node_sets.items():
            if flat is None:
                bands[stage] = None
                continue
            z = frame.depth.ravel()[flat]
            counts = (int((z <= bounds['median']).sum()),
                      int(((z > bounds['median']) & (z <= bounds['p95'])).sum()),
                      int((z > bounds['p95']).sum()))
            bands[stage] = {name: count_ratio(n, len(z)) for name, n in
                zip(('z_le_X_median', 'X_median_lt_z_le_X_p95', 'z_gt_X_p95'), counts)}
    return dict(negative_roi=roi, depth_m=depth, depth_quantile_occupancy=bands,
                depth_band_bounds_m=dict(median=bounds['median'], p95=bounds['p95']))


def compare_ids(first, second):
    first, second = set(first), set(second)
    return {name: dict(count=len(ids), reference_ids=sorted(ids)) for name, ids in
            [('intersection', first & second), ('S2_only', first-second), ('S3_only', second-first)]}


def diagnose_frame(frame, gt, robot_ids, source):
    evidence = geometry_evidence(frame)
    geometry = evidence.frame.geometry
    context = evaluation_context(frame, evidence, gt, frame.camera, robot_ids)
    metrics = json.loads((source/f'f{frame.source_frame:04d}_metrics.json').read_text())
    results, alignments = {}, {}
    with np.load(source/f'f{frame.source_frame:04d}_samples.npz', allow_pickle=False) as archive:
        # Recomputed Task02 references must be bit-exact, including ID order.
        # Actual matching below always uses the saved arrays.
        for suffix, recomputed in zip(('uv', 'xyz'), context['references']['box']):
            np.testing.assert_array_equal(archive[f'evaluation/box/reference_{suffix}'], recomputed)
        reference = tuple(archive['evaluation/box/reference_'+s] for s in ('uv', 'xyz'))
        matcher = CoverageMatcher(geometry, BOX, reference)
        for method, key in CONDITIONS.items():
            require_equal(metrics[key]['source_frame'], frame.source_frame, 'metric source row')
            alignments[method] = validate_saved_selection(archive, key, frame, geometry)
            choose = archive[key+'/object_id'] == BOX
            f = archive[key+'/flat_index'][choose]
            role = archive[key+'/priority_selected'][choose]
            p, s = f[role], f[~role]
            c = None if method == 'S0' else archive[f'evidence/{"fd2" if method == "S2" else "pca5"}/1/20/orientation_flat']
            if method == 'S0' and len(p):
                raise ValueError('General FPS unexpectedly has priority nodes')
            row = staged_reference_coverage(matcher, c, p if c is not None else None, s, f)
            original = metrics[key]['objects']['box']
            rescored, _ = matcher.score(f)
            require_equal(rescored['tolerance'], original['structural']['tolerance'], 'frame-level coverage')
            counts = original['counts']
            for name, n in [('observed', len(matcher.flat)), ('budget', len(f)), ('priority', len(p)),
                            ('surface_fps', len(s)), ('selected', len(f))]:
                require_equal(n, counts[name], 'Box '+name)
            if c is not None:
                require_equal(len(c), counts['candidates'], 'Box candidates')
                require_equal(int(np.isin(f, c).sum()), counts['selected_candidates'], 'selected candidate nodes')
            nodes = dict(X=matcher.flat, C=c, P=p if c is not None else None, S=s, F=f)
            row.update(occupancy(frame, nodes, context['negatives']['box']))
            for name, oid in [('peg', PEG), ('box', BOX)]:
                obj = archive[key+'/object_id'] == oid
                obj_flat = archive[key+'/flat_index'][obj]
                obj_priority = archive[key+'/priority_selected'][obj]
                estimator = 'fd2' if method == 'S2' else 'pca5'
                obj_candidates = archive[f'evidence/{estimator}/{oid}/20/orientation_flat']
                saved_counts = metrics[key]['objects'][name]['counts']
                expected_counts = dict(observed=int(geometry.observed_mask(oid).sum()),
                    budget=len(obj_flat), candidates=len(obj_candidates), priority=int(obj_priority.sum()),
                    orientation_priority=int(obj_priority.sum()), depth_priority=0,
                    surface_fps=int((~obj_priority).sum()), selected=len(obj_flat),
                    selected_candidates=int(np.isin(obj_flat, obj_candidates).sum()))
                require_equal(saved_counts, expected_counts, name+' allocation counters')
            # Prove existing ROI occupancy is reproduced, not a new ROI definition.
            for name, saved in original['negative'].items():
                actual = row['negative_roi'][name]
                for old, stage in [('observed', 'X'), ('selected', 'F'), ('priority', 'P')]:
                    require_equal(actual[stage]['count'] if actual[stage] is not None else 0,
                                  saved[old], 'saved negative ROI '+name+'/'+old)
            row.update(condition=key, control='General FPS; C/P not applicable' if c is None else None,
                       allocation={name: metrics[key]['objects'][name]['counts'] for name in ('box', 'peg')},
                       original_frame_coverage_exact=True)
            results[method] = row
    expected_v, expected_o, expected_final = TARGETS[frame.source_frame]
    require_equal(len(results['S0']['reference_ids']['V']), expected_v, 'target GT')
    require_equal(len(results['S0']['reference_ids']['O']), expected_o, 'target observed')
    for method, n in zip(CONDITIONS, expected_final):
        require_equal(results[method]['stages']['F']['unique_reference_count'], n, 'target '+method)
    # Allocation targets requested explicitly, separate from reference counts.
    box = {m: results[m]['allocation']['box'] for m in CONDITIONS}
    if frame.source_frame == 0:
        for m in CONDITIONS:
            require_equal((box[m]['observed'], box[m]['budget']), (11665, 127), 'frame 0 allocation')
        expected = {'S2': (297, 32), 'S3': (599, 32)}
    else:
        expected = {'S2': (24, 24), 'S3': (65, 49)}
    for m, pair in expected.items():
        require_equal((box[m]['candidates'], box[m]['priority']), pair, 'candidate/priority target '+m)
    comparison = {stage: compare_ids(results['S2']['reference_ids'][stage],
                                    results['S3']['reference_ids'][stage]) for stage in ('C', 'P', 'F')}
    return dict(source_frame=frame.source_frame, conditions=results, S2_vs_S3=comparison,
                alignment=alignments, saved_reference_order_and_geometry_exact=True,
                target_coverage_exact=True, saved_negative_roi_counts_exact=True)


def diagnose(source, output_root):
    if os.environ.get('WANDB_MODE') != 'disabled':
        raise ValueError('set WANDB_MODE=disabled for this local-only diagnostic')
    source = Path(source).resolve()
    comparison_path = source/'comparison.json'
    original = json.loads(comparison_path.read_text())
    require_equal(original['run_id'], SOURCE_RUN, 'Task03 source run')
    config = original['config']
    source_checks = source_contract(source, config)
    episode, masks = Path(config['input']), Path(config['masks'])
    inputs = [comparison_path, episode, masks]
    require_equal(digests(episode)[1], config['input_sha256'], 'episode SHA256')
    require_equal(digests(masks)[1], config['masks_sha256'], 'tracker mask SHA256')
    require_equal(config['tolerances_px_m'], [[1., .002], [2., .003], [3., .005]], 'evaluation tolerances')
    segmap = episode.with_name('segmentation_id_map.json')
    require_equal(digests(segmap)[1], config['segmentation_map_sha256'], 'segmentation map SHA256')
    inputs.append(segmap)
    tracking_path = masks.with_name('tracking.json')
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text())
        require_equal(tracking['camera'], config['camera'], 'tracker camera')
        require_equal(Path(tracking['input']).resolve(), episode.resolve(), 'tracker source episode')
        inputs.append(tracking_path)
    for f in FRAMES:
        inputs.extend(source/f'f{f:04d}_{suffix}' for suffix in ('samples.npz', 'metrics.json'))
    before = {str(p): digests(p)[1] for p in inputs}
    current_sources = {str(p.relative_to(REPO)): digests(p)[1] for p in
                       (Path(__file__), Path(__file__).with_name('sampling_evaluation.py'),
                        Path(__file__).with_name('test_sampling_evaluation.py'))}
    diagnostic_config = dict(task='03-D', phase=1, source_run_id=SOURCE_RUN, source_directory=str(source),
        frames=FRAMES, conditions=CONDITIONS, tolerance_px_m=[2., .003],
        input_sha256=before, source_sha256=current_sources, device='cpu', gpu_executed=False,
        gt_diagnostic_only=True, sampler_executed=False, candidate_extraction_executed=False,
        normal_estimation_executed=False, python=platform.python_version(), numpy=np.__version__)
    with ExperimentLogger(team='visual', config=diagnostic_config, seed=0, model='task03d_rim_bottleneck',
                          job_type='evaluate', root=output_root) as logger:
        output = logger.directory
        logger.record(source_run_id=SOURCE_RUN, evaluation_device='cpu', gpu_executed=False)
        print('Output:', output, flush=True)
        with np.load(episode, allow_pickle=False) as ep, np.load(masks, allow_pickle=False) as mask:
            gt_keys = [f'{config["camera"]}/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose',
                       'box_hole_pose', 'peg_half_sizes', 'box_hole_radii']
            gt = {k: ep[k] for k in gt_keys}
            sources = mask['source_frame']
            np.testing.assert_array_equal(sources, np.arange(len(gt[gt_keys[0]])))
            peg_masks, box_masks = mask['peg'], mask['box']
            selected_rows = [int(np.flatnonzero(sources == f).item()) for f in FRAMES]
            tracking_rows = ((i, {1: peg_masks[row], 2: box_masks[row]}) for i, row in enumerate(selected_rows))
            frames = attach_rgbd(tracking_rows, ep, camera=config['camera'], source_frames=sources[selected_rows])
            rows = [diagnose_frame(frame, gt, config['robot_ids'], source) for frame in frames]
        require_equal(tuple(r['source_frame'] for r in rows), FRAMES, 'diagnosed sources')
        after = {str(p): digests(p)[1] for p in inputs}
        require_equal(after, before, 'input preservation SHA256')
        result = dict(run_id=logger.run_id, source_run_id=SOURCE_RUN, config=diagnostic_config,
                      source_contract=source_checks, inputs_unchanged=True, frames=rows,
                      notes=['Reference IDs index saved evaluation/box/reference_uv and reference_xyz rows.',
                             'All coverage uses the same pair within 2px AND 3mm.',
                             'Overlapping set differences and ROI counts are not additive causal losses.',
                             'S0 saved candidate diagnostics appear only in allocation; C/P are inapplicable.',
                             'Depth bands are empirical X quantiles, not filtering or far/error labels.',
                             'No FPS trace or causal counterfactual was executed.'])
        write_json(output/'diagnostic.json', result)
        for relative in current_sources:
            dest = output/'diagnostic_source'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/relative, dest)
            logger.link_output(dest, kind='reports')
        logger.link_output(output/'diagnostic.json', kind='metrics')
        for row in rows:
            logger.log({m: r['coverage'] for m, r in row['conditions'].items()},
                       step=row['source_frame'], prefix='eval/task03d')
        logger.record(inputs_unchanged=True, evaluated_frames=len(rows), all_target_coverage_exact=True,
                      source_contract_verified=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO/'visual/results/sampling'/SOURCE_RUN)
    parser.add_argument('--out', type=Path, default=REPO/'experiments')
    args = parser.parse_args()
    print(diagnose(args.source, args.out))


if __name__ == '__main__':
    main()
