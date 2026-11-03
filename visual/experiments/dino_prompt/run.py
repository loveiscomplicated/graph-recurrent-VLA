"""Bounded two-round prompt experiment using existing DINO and initialization APIs.

python -m visual.experiments.dino_prompt.run --inputs INPUTS --out OUTPUTS --model MODEL_DIR
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw
from visual.runtime import initialization as init
from visual.experiments.dino_prompt.prepare import sha, write

PROMPTS = {
    'peg': ['red and white block', 'peg', 'white block', 'red and white peg',
            'white rectangular block', 'white stick', 'white peg with red tip',
            'red and white rectangular block', 'white rectangular peg',
            'red tipped white peg', 'red and white stick',
            'white rectangular block with a red end'],
    'box': ['cardboard box', 'box', 'brown box', 'brown cardboard box',
            'rectangular box', 'brown block', 'box with hole', 'brown block with a hole',
            'brown rectangular box', 'cardboard block', 'wooden box', 'brown box with a hole'],
}
OBJECTS = ('peg', 'box')
BASELINE = (PROMPTS['peg'][0], PROMPTS['box'][0])


def gt_bbox(mask):
    y, x = np.nonzero(mask)
    return [float(x.min()), float(y.min()), float(x.max()+1), float(y.max()+1)] if len(x) else None


def overlap(box, gt):
    return init.box_iou(box, gt) if box is not None and gt is not None else 0.0


class CachedDetector:
    """Consumer-facing adapter: only RGB and text enter candidate selection."""
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, rgb, prompt):
        record = self.cache[(hashlib.sha256(rgb.tobytes()).hexdigest(), prompt)]
        return record['boxes'], record['scores']


def evaluate(frame, rgb, segmentation, detector, pair):
    candidates = [init.candidates(detector, rgb, prompt) for prompt in pair]
    selected = init.assign(*candidates)
    rows = []
    for k, (obj, prompt) in enumerate(zip(OBJECTS, pair)):
        raw_boxes, scores = detector(rgb, prompt)
        boxes_gt = [gt_bbox(np.isin(segmentation, frame['object_ids'][name]))
                    for name in OBJECTS] if segmentation is not None else [None, None]
        gt = boxes_gt[k]
        visible = frame.get('gt_pixels', {}).get(obj, 0)
        supported = segmentation is not None and visible >= 8
        ious = [overlap(b, gt) for b in raw_boxes]
        good = [i for i, value in enumerate(ious) if value >= .5] if supported else []
        chosen = selected[k]
        chosen_box = chosen[0] if chosen else None
        chosen_iou = overlap(chosen_box, gt)
        area_good = [i for i in good if np.prod(np.subtract(raw_boxes[i][2:], raw_boxes[i][:2]))
                     < init.MAX_BOX_FRAC * rgb.shape[0] * rgb.shape[1]]
        good_filtered = [c for c in candidates[k] if overlap(c[0], gt) >= .5] if supported else []
        correct = supported and chosen_iou >= .5
        if not supported:
            drop = 'GT_UNAVAILABLE' if segmentation is None else 'SMALL_OR_INVISIBLE'
        elif correct:
            drop = 'SUCCESS'
        elif not good:
            drop = 'NO_GOOD_PROPOSAL_AT_THRESHOLD'
        elif not area_good:
            drop = 'AREA'
        elif not good_filtered:
            drop = 'TOP5'
        elif max(c[1] for c in good_filtered) <= init.MISSING_SCORE:
            drop = 'NULL_SCORE'
        else:
            drop = 'JOINT_OR_RANKING'
        rows.append(dict(frame=frame['id'], parent=frame['parent'], split=frame['split'],
                         control=frame['control'], object=obj, prompt=prompt,
                         supported=supported, gt_pixels=visible, gt_bbox=gt,
                         proposal_hit=bool(good), oracle_iou=max(ious, default=0.0),
                         top1_iou=ious[0] if ious else 0.0,
                         good_rank=good[0]+1 if good else None,
                         good_score=float(scores[good[0]]) if good else None,
                         selected_box=chosen_box, selected_score=chosen[1] if chosen else None,
                         selected_iou=chosen_iou, correct=correct,
                         wrong_object=bool(supported and chosen is not None and chosen_iou < .5
                                           and overlap(chosen_box, boxes_gt[1-k]) >= .5),
                         unsupported_selection=bool(not supported and chosen is not None),
                         drop=drop))
    return rows


def metrics(rows):
    supported = [r for r in rows if r['supported']]
    return dict(supported=len(supported), correct=sum(r['correct'] for r in supported),
                wrong_object=sum(r['wrong_object'] for r in supported),
                proposal_hits=sum(r['proposal_hit'] for r in supported),
                mean_selected_iou=float(np.mean([r['selected_iou'] for r in supported])) if supported else 0.0)


def choose(evaluations, pairs, lengths):
    """Preserve every successful baseline control before maximizing recovery."""
    baseline = evaluations[BASELINE]
    protected = {(r['frame'], r['object']) for r in baseline if r['control'] and r['correct']}
    def key(pair):
        rows = evaluations[pair]
        controls = sum((r['frame'], r['object']) in protected and r['correct'] for r in rows)
        m = metrics(rows)
        return (controls == len(protected), m['correct'], -m['wrong_object'],
                m['mean_selected_iou'], -sum(lengths[p] for p in pair), -pairs.index(pair))
    return max(pairs, key=key)


def transitions(before, after):
    old = {(r['frame'], r['object']): r for r in before}
    return [dict(frame=r['frame'], object=r['object'],
                 before=old[(r['frame'], r['object'])]['drop'], after=r['drop'],
                 recovered=not old[(r['frame'], r['object'])]['correct'] and r['correct'],
                 regressed=old[(r['frame'], r['object'])]['correct'] and not r['correct'],
                 new_confusion=not old[(r['frame'], r['object'])]['wrong_object'] and r['wrong_object'])
            for r in after if r['supported']]


def temporal(frames, images, detector, pair, rows):
    scans = sorted([f for f in frames if f['parent'] == '0355' and f['camera'] == 'base_camera'],
                   key=lambda f: f['timestep'])
    # detect_initial expects frame indices, including intervening unqueried frames.
    assert [f['timestep'] for f in scans] == list(range(0, 31, 5))
    sequence = [images[scans[0]['id']]] * 31
    for f in scans:
        sequence[f['timestep']] = images[f['id']]
    result = init.detect_initial(sequence, detector, dict(zip(OBJECTS, pair)))
    for obj, record in result.items():
        frame = record['frame']
        ref = next((r for r in rows if r['frame'] == f'0355_base_camera_{frame:03d}'
                    and r['object'] == obj), None) if frame is not None else None
        record['gt_iou_at_commit'] = overlap(record['box'], ref['gt_bbox']) if ref and ref['supported'] else None
        record['correct_at_commit'] = (record['gt_iou_at_commit'] >= .5
                                        if record['gt_iou_at_commit'] is not None else None)
    return result


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def gallery(out, frames, images, baseline, winner):
    originals = {(r['frame'], r['object']): r for r in baseline}
    new = {(r['frame'], r['object']): r for r in winner}
    canvas = Image.new('RGB', (900, len(frames)*330), 'white')
    for i, frame in enumerate(frames):
        rgb = images[frame['id']]
        for column, label in enumerate(('GT', 'baseline', 'selected')):
            tile = Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST)
            d = ImageDraw.Draw(tile)
            for obj, color in [('peg', '#ff3b30'), ('box', '#00cfff')]:
                row = originals[(frame['id'], obj)] if column < 2 else new[(frame['id'], obj)]
                box = row['gt_bbox'] if column == 0 else row['selected_box']
                if box:
                    coords = [float(np.clip(v*2, 0, 255)) for v in box]
                    d.rectangle(coords, outline=color, width=2)
            canvas.paste(tile, (column*300, i*330+45))
            draw = ImageDraw.Draw(canvas)
            draw.text((column*300+4, i*330+3), frame['id'], fill='black')
            draw.text((column*300+4, i*330+20), label, fill='black')
            statuses = ' / '.join((originals if column < 2 else new)[(frame['id'], o)]['drop'] for o in OBJECTS)
            draw.text((column*300+4, i*330+305), statuses, fill='black')
    canvas.save(out / 'comparison.png')


def run(args):
    import torch
    import transformers
    from visual.runtime.grounding import GDino
    from utils.experiment_logger import ExperimentLogger
    os.environ['WANDB_MODE'] = 'disabled'
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    assert sha(args.inputs / 'frames.npz') == manifest['bundle_sha256']
    with np.load(args.inputs / 'frames.npz') as z:
        images = {f['id']: z[f['id']+'_rgb'].copy() for f in manifest['frames'] if f['id']+'_rgb' in z}
        segs = {f['id']: z[f['id']+'_seg'].copy() for f in manifest['frames'] if f['id']+'_seg' in z}
    frames = [f for f in manifest['frames'] if f['id'] in images]
    search = [f for f in frames if f['split'] == 'search']
    confirm = [f for f in frames if f['split'] == 'confirm']
    assert len(search) == 10 and len(confirm) == 4, 'Missing mandatory RGB inputs'
    for f in frames:
        assert hashlib.sha256(images[f['id']].tobytes()).hexdigest() == f['rgb_sha256']
    model_path = args.model.resolve()
    model_identity = {str(p.relative_to(model_path)): sha(p) for p in sorted(model_path.iterdir()) if p.is_file()}
    source_identity = {str(p): sha(p) for p in (Path(__file__), Path(init.__file__),
                      Path(__file__).with_name('prepare.py'), Path(__file__).parents[2]/'runtime/grounding.py')}
    protocol = dict(task='10j', prompts=PROMPTS, baseline=BASELINE, search=[f['id'] for f in search],
                    confirm=[f['id'] for f in confirm], min_gt_pixels=8, hit_iou=.5,
                    max_calls=300, device=args.device, model_identity=model_identity,
                    source_identity=source_identity, manifest_sha256=sha(args.inputs/'manifest.json'),
                    initialization=dict(area=init.MAX_BOX_FRAC, null=init.MISSING_SCORE,
                                        exclusion=init.EXCLUSION_IOU, scan_every=init.SCAN_EVERY))
    write(args.out / 'protocol.json', protocol)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    with ExperimentLogger(team='visual', model='grounding-dino-tiny-prompt-feasibility', seed=42,
                          job_type='evaluate', root=args.out/'experiments', config=protocol) as log:
        log.record(input_manifest=manifest, execution_environment=dict(python=platform.python_version(),
                   torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                   device=torch.cuda.get_device_name() if args.device.startswith('cuda') else args.device))
        t0 = time.perf_counter()
        detector = GDino(device=args.device, model_id=str(model_path), threshold=.1)
        load_seconds = time.perf_counter()-t0
        cache, calls = {}, []
        def infer(frame, prompt, phase, reuse=True):
            key = (frame['rgb_sha256'], prompt)
            if reuse and key in cache:
                return cache[key]
            if len(calls) >= 300:
                raise RuntimeError('300-call budget exhausted')
            rgb = images[frame['id']]
            boxes, scores = detector(rgb, prompt)
            row = dict(call=len(calls)+1, frame=frame['id'], rgb_sha256=key[0], prompt=prompt,
                       phase=phase, boxes=boxes, scores=scores, ms=detector.last_ms.copy())
            calls.append(row)
            if reuse:
                cache[key] = row
            write(args.out/'calls.json', calls)
            return row
        for obj in OBJECTS:
            infer(search[0], PROMPTS[obj][0], 'warmup', reuse=False)
        baseline_records = {}
        for frame in search:
            for prompt in BASELINE:
                baseline_records[(frame['id'], prompt)] = infer(frame, prompt, 'baseline')
        # Repeat a real input to check numerical stability on this host.
        repeat = infer(search[0], BASELINE[0], 'repeat', reuse=False)
        expected = baseline_records[(search[0]['id'], BASELINE[0])]
        stable = (np.allclose(repeat['boxes'], expected['boxes'], atol=1e-4, rtol=0)
                  and np.allclose(repeat['scores'], expected['scores'], atol=1e-5, rtol=0))
        assert stable, 'Repeated baseline changed beyond tolerance'
        cached = CachedDetector(cache)
        def evaluate_pair(subset, pair):
            return [r for f in subset for r in evaluate(f, images[f['id']], segs.get(f['id']), cached, pair)]
        baseline_rows = evaluate_pair(search, BASELINE)
        # Historical fingerprints are validated before search. They identify the
        # frozen 0355/0645 cases; tolerance allows tiny CUDA-version differences.
        fingerprints = [('0355_base_camera_000', 'peg', .2916409373283386),
                        ('0355_base_camera_030', 'peg', .4480925500392914),
                        ('0645_base_camera_000', 'peg', .4812784194946289)]
        checks = []
        for fid, obj, score in fingerprints:
            rec = baseline_records[(fid, BASELINE[OBJECTS.index(obj)])]
            error = min((abs(s-score) for s in rec['scores']), default=1.)
            checks.append(dict(frame=fid, object=obj, expected_score=score, min_score_error=error,
                               pass_check=error < 1e-4))
        baseline_temporal = temporal(search, images, cached, BASELINE, baseline_rows)
        write(args.out/'baseline_verification.json', dict(repeat_stable=stable, historical=checks,
                    temporal=baseline_temporal))
        assert all(c['pass_check'] for c in checks), 'Historical baseline fingerprint mismatch'
        assert baseline_temporal['peg']['frame'] == 30, '0355 baseline commit mismatch'
        assert baseline_temporal['peg']['correct_at_commit'] is False, '0355 wrong owner not reproduced'
        evaluations = {}
        for round_no, bounds in [(1, (0, 8)), (2, (8, 12))]:
            if round_no == 2:
                write(args.out/'round2_rationale.json', dict(
                    premise='Fixed complementary phrases; one bounded refinement, no confirm peeking',
                    hypotheses={'peg': 'object name, red tip, elongated shape vs block confusion',
                                'box': 'material, shape and hole specificity'},
                    round1_best=round1_pair,
                    round1_baseline_transitions=transitions(baseline_rows, evaluations[tuple(round1_pair)])))
            for frame in search:
                for obj in OBJECTS:
                    for prompt in PROMPTS[obj][bounds[0]:bounds[1]]:
                        infer(frame, prompt, f'round{round_no}')
            current = {o: PROMPTS[o][:bounds[1]] for o in OBJECTS}
            pairs = list(itertools.product(current['peg'], current['box']))
            for pair in pairs:
                evaluations[pair] = evaluate_pair(search, pair)
            lengths = {p: len(detector.proc.tokenizer(p.lower().strip()+'.')['input_ids'])
                       for o in OBJECTS for p in current[o]}
            best = choose(evaluations, pairs, lengths)
            summaries = [dict(peg=p[0], box=p[1], **metrics(evaluations[p])) for p in pairs]
            csv_write(args.out/f'round{round_no}_combinations.csv', summaries)
            write(args.out/f'round{round_no}_best.json', dict(pair=best, metrics=metrics(evaluations[best])))
            print('ROUND', round_no, best, metrics(evaluations[best]), 'calls', len(calls), flush=True)
            if round_no == 1:
                round1_pair = list(best)
        selected = best
        # Freeze selection on disk BEFORE any confirmation forward.
        write(args.out/'selection_frozen.json', dict(pair=selected, search_metrics=metrics(evaluations[selected]),
              calls_before_confirmation=len(calls), confirmation_used_for_selection=False))
        for frame in confirm:
            for prompt in dict.fromkeys((*BASELINE, *selected)):
                infer(frame, prompt, 'confirm')
        confirm_before = evaluate_pair(confirm, BASELINE)
        confirm_after = evaluate_pair(confirm, selected)
        before = baseline_rows + confirm_before
        after = evaluations[selected] + confirm_after
        change_search = transitions(baseline_rows, evaluations[selected])
        change_confirm = transitions(confirm_before, confirm_after)
        search_recoveries = sum(r['recovered'] for r in change_search)
        confirm_failure_ids = {f['id'] for f in confirm if not f['control']}
        confirm_recoveries = sum(r['recovered'] for r in change_confirm if r['frame'] in confirm_failure_ids)
        control_ids = {f['id'] for f in frames if f['control']}
        control_regressions = sum(r['regressed'] for r in change_search+change_confirm if r['frame'] in control_ids)
        new_confusions = sum(r['new_confusion'] for r in change_confirm)
        if search_recoveries and confirm_recoveries and not control_regressions and not new_confusions:
            verdict = 'FOLLOWUP_WORTHWHILE'
        elif search_recoveries:
            verdict = 'CASE_LIMITED_IMPROVEMENT'
        elif metrics(after)['proposal_hits'] > metrics(before)['proposal_hits']:
            verdict = 'PROPOSAL_ONLY_IMPROVEMENT'
        else:
            verdict = 'NO_RECOVERY_IN_THIS_SEARCH'
        all_details = []
        for obj in OBJECTS:
            for prompt in PROMPTS[obj]:
                pair = (prompt, BASELINE[1]) if obj == 'peg' else (BASELINE[0], prompt)
                all_details.extend(r for r in evaluations[pair] if r['object'] == obj)
        csv_write(args.out/'prompt_details.csv', all_details)
        csv_write(args.out/'baseline_selected.csv', [dict(condition=c, **r) for c, rs in
                  [('baseline', before), ('selected', after)] for r in rs])
        csv_write(args.out/'transitions.csv', change_search+change_confirm)
        selected_temporal = temporal(search, images, cached, selected, evaluations[selected])
        latencies = [sum(c['ms'].values()) for c in calls if c['phase'] not in ('warmup', 'repeat')]
        summary = dict(status='PARTIAL_INPUT_GT' if manifest['missing'] else 'COMPLETE', verdict=verdict,
                       run_id=log.run_id, selected=dict(zip(OBJECTS, selected)), baseline=dict(zip(OBJECTS, BASELINE)),
                       round1_pair=round1_pair, model_load_s=load_seconds, total_calls=len(calls),
                       calls_by_phase={p: sum(c['phase']==p for c in calls) for p in sorted({c['phase'] for c in calls})},
                       search=dict(baseline=metrics(baseline_rows), selected=metrics(evaluations[selected])),
                       confirm=dict(baseline=metrics(confirm_before), selected=metrics(confirm_after)),
                       search_recoveries=search_recoveries, confirm_failure_recoveries=confirm_recoveries,
                       control_regressions=control_regressions, new_confirm_confusions=new_confusions,
                       baseline_temporal=baseline_temporal, selected_temporal=selected_temporal,
                       latency_total_ms=dict(median=float(np.median(latencies)), p95=float(np.percentile(latencies,95))),
                       missing=manifest['missing'],
                       not_executed=['SAM2', 'DINOv2', 'simulator', 'training', 'MPS', 'GPU PhysX'])
        gallery(args.out, frames, images, before, after)
        write(args.out/'summary.json', summary)
        # Verify original model files were not modified by this experiment.
        assert model_identity == {str(p.relative_to(model_path)): sha(p) for p in sorted(model_path.iterdir()) if p.is_file()}
        for source, digest in manifest['sources'].items():
            assert sha(source) == digest, source
        write(args.out/'verification.json', dict(baseline_pass=True, source_files_unchanged=True,
              model_files_unchanged=True, budget_pass=len(calls)<=300, selection_frozen_before_confirm=True,
              GT_only_in_evaluation=True, source_identity=source_identity))
        log.log(dict(calls=len(calls), search_recoveries=search_recoveries,
                     confirm_recoveries=confirm_recoveries, control_regressions=control_regressions))
        for p in sorted(args.out.iterdir()):
            if p.is_file() and not p.name.endswith('.tracking.json'):
                log.link_output(p)
        print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='Existing local HF model snapshot; no download')
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
