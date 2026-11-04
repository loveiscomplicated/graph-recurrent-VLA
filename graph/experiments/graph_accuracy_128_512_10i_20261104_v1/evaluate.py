"""Post-inference GT scoring with common physical surface references."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .common import CAMERAS, PARENTS, SIZES, entries, read, write, sha, sensor_frame
from graph.task10i_stage4a2.evaluation_metrics import pixel_metrics, correct_source_coverage, hierarchical_paired
from graph.task10i_stage4a2.paired_evaluation import reference_points
from graph.task10i_stage4a2.dinov2_matching_audit import csv_write
from utils.experiment_logger import ExperimentLogger


def actor_labels(segmentation, object_ids, object_id, cameras, pixels):
    """Classify the real representative source; merged sources do not vote."""
    labels = []
    target, other = ('peg', 'box') if object_id == 0 else ('box', 'peg')
    for c, p in zip(cameras, pixels):
        actor = int(segmentation[int(c)].ravel()[int(p)])
        ids = object_ids[int(c)]
        labels.append('correct' if actor in ids[target] else 'wrong_object' if actor in ids[other]
                      else 'robot' if actor in ids['robot'] else 'background')
    return np.asarray(labels, dtype=str)


def main(capture, inference, output):
    done = read(inference/'inference_complete.json')
    if done['status'] != 'COMPLETE' or done['graph_frames'] != 168:
        raise ValueError('inference not complete')
    if sha(inference/'protocol.json') != done['protocol_sha256']:
        raise ValueError('protocol changed')
    for s in done['sequences']:
        for rel, h in s['files'].items():
            if sha(inference/rel) != h:
                raise ValueError('prediction changed before scoring')
    output.mkdir(parents=True, exist_ok=False)
    index = entries(capture)
    point_rows, pixel_rows = [], []
    config = dict(inference=str(inference), capture=str(capture),
        inference_receipt_sha256=sha(inference/'inference_complete.json'),
        common_reference='same current native1024 visible GT depth, Peg both views, Box Main only',
        visible_threshold_1024_pixels=1024, correct_source_only=True, radius_m=.004)
    with ExperimentLogger(team='integration', model='task10i_current_graph_native128_512_scoring',
            seed=42, job_type='evaluate', root=output/'experiments', config=config) as log:
        for parent in PARENTS:
            for t in range(21):
                ref_cameras, _ = sensor_frame(capture, index, parent, t, 1024, parent)
                reference_segs, ids = [], []
                native = {}
                for size in (*SIZES, 1024):
                    segs, idlist = [], []
                    for camera in CAMERAS:
                        e = index[parent, t, size, camera]
                        p = capture/e['gt_file']
                        if sha(p) != e['gt_sha256']:
                            raise ValueError('GT reference changed')
                        with np.load(p, allow_pickle=False) as z:
                            seg = z['actor_segmentation'].copy()
                        if seg.shape != (size, size):
                            raise ValueError('GT/native size mismatch')
                        segs.append(seg)
                        idlist.append(e['object_ids'])
                    native[size] = (segs, idlist)
                reference_segs, ids = native[1024]
                if ids[0] != ids[1]:
                    raise ValueError('mixed actor mapping')
                rr = dict(object_ids={k: ids[0][k] for k in ('peg', 'box')})
                for size in SIZES:
                    condition = 'B0' if size == 128 else 'N512'
                    meta = dict(parent_id=parent, episode=parent, timestep=t, size=size,
                                condition=condition, reference_kind='EXACT_GT')
                    with np.load(inference/parent/str(size)/f't{t:04d}.npz', allow_pickle=False) as z:
                        a = {k: z[k].copy() for k in z.files}
                    segs, mappings = native[size]
                    for ci, camera in enumerate(CAMERAS):
                        for oid, name in enumerate(('peg', 'box')):
                            target = np.isin(segs[ci], mappings[ci][name])
                            row = dict(meta, camera=camera, object_id=oid,
                                graph_eligible=oid == 0 or ci == 0,
                                **pixel_metrics(a['raw_masks'][ci, oid], target,
                                    np.isin(segs[ci], mappings[ci]['robot']),
                                    np.isin(segs[ci], mappings[ci][('box', 'peg')[oid]])))
                            pixel_rows.append(row)
                    for oid in (0, 1):
                        visible, boundary = reference_points(ref_cameras, reference_segs, rr, oid)
                        take = a['object_id'] == oid
                        labels = actor_labels(segs, mappings, oid, a['camera'][take], a['pixel'][take])
                        correct = labels == 'correct'
                        xyz = a['world_xyz'][take]
                        n = len(xyz)
                        supported = len(visible) >= 1024
                        stratum = 'visible' if supported else 'tiny' if len(visible) else 'absent'
                        row = dict(meta, object_id=oid, common_visible_samples=len(visible), stratum=stratum,
                            points=n, correct_points=int(correct.sum()),
                            correct_representative_fraction=float(correct.mean()) if n else None,
                            coverage_4mm=correct_source_coverage(xyz, correct, visible),
                            boundary_coverage_4mm=correct_source_coverage(xyz, correct, boundary),
                            correct_point_recall=float(correct.any()) if supported else None,
                            false_presence=float(n > 0) if not len(visible) else None,
                            node_count=n, edge_count=int(a['edge_index'].shape[1]),
                            wrist_gate_accepted=read(inference/parent/str(size)/f't{t:04d}.json')['gate']['accepted'])
                        for label in ('robot', 'wrong_object', 'background'):
                            row[label+'_points'] = int((labels == label).sum())
                            row[label+'_fraction'] = float((labels == label).mean()) if n else None
                        point_rows.append(row)
            print('SCORED', parent, flush=True)
        metrics = ('correct_representative_fraction', 'coverage_4mm', 'boundary_coverage_4mm',
                   'correct_point_recall', 'robot_fraction', 'wrong_object_fraction', 'background_fraction', 'false_presence')
        paired = []
        for metric in metrics:
            paired.extend(hierarchical_paired(point_rows, metric, group_keys=('object_id',),
                bootstrap=10000, seed=42, candidate='N512'))
        pooled = []
        for oid in (0, 1):
            for size in SIZES:
                rows = [r for r in point_rows if r['object_id'] == oid and r['size'] == size]
                n = sum(r['points'] for r in rows)
                eligible = [r for r in rows if r['stratum'] == 'visible']
                pooled.append(dict(object_id=oid, size=size, frames=len(rows),
                    strata=dict(Counter(r['stratum'] for r in rows)), predicted_points=n,
                    correct_points=sum(r['correct_points'] for r in rows),
                    point_precision=sum(r['correct_points'] for r in rows)/n if n else None,
                    visible_frames=len(eligible),
                    visible_correct_presence=sum(r['correct_point_recall'] for r in eligible)/len(eligible) if eligible else None,
                    empty_frames=sum(r['points'] == 0 for r in rows),
                    mean_coverage=float(np.mean([r['coverage_4mm'] for r in rows if r['coverage_4mm'] is not None]))))
        csv_write(output/'point_metrics.csv', point_rows)
        csv_write(output/'pixel_metrics.csv', pixel_rows)
        write(output/'paired_metrics.json', paired)
        write(output/'summary.json', dict(status='COMPLETE', run_id=log.run_id, pooled=pooled,
            graph_frames=168, object_frames=336, paired=paired, config=config,
            source_run_id=done['run_id'], native_grounding=done['grounding'],
            limitations=['4 known Train parents; t0..20 only', 'recorded CPU-rendered native inputs',
                         'no end-pose or GT edge accuracy', 'single stochastic Terra realization per native input',
                         'conditional precision/coverage paired available denominators; missing reported separately']))
        log.log(dict(graph_frames=168, object_frames=336))
        for name in ('summary.json', 'point_metrics.csv', 'pixel_metrics.csv', 'paired_metrics.json'):
            log.link_output(output/name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.capture, a.inference, a.output)
