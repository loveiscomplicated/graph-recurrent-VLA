"""Read-only forensic analysis of the frozen verifier pilot; no model inference."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/task10i/stage4a2'
RUN = BASE / 'dino_qwen_candidate_verifier_20261102_v1'
OUT = BASE / 'dino_qwen_verifier_failure_audit_20261102_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((RUN / name).read_text())


def rows(name):
    with (RUN / name).open() as f:
        return list(csv.DictReader(f))


def counts(values):
    return dict(collections.Counter(values))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    indexed = read('artifact_index.json')['files']
    assert all(sha(RUN / name) == item['sha256'] for name, item in indexed.items())
    candidates = rows('candidate_evaluation.csv')
    selected = rows('hybrid_selected_boxes.csv')
    raw = read('verifier_raw_responses.json')['responses']
    requests = read('request_manifest.json')['requests']
    by_id = {r['id']: r for r in candidates}
    responses = {r['id']: r for r in raw}
    assert len(candidates) == len(raw) == len(requests) == 251
    assert [r['id'] for r in raw] == [r['id'] for r in requests]
    assert all(r['raw_text'].strip() == r['strict_label'] for r in raw)
    assert all(len(r['image_sha256']) == 3 for r in raw)
    cache = json.loads((OUT / 'installed_cache_audit.json').read_text())
    infer = ROOT / 'graph/experiments/dino_qwen_candidate_verifier_20261102_v1/infer.py'
    assert sha(infer) == cache['inference_source_sha256']
    assert cache['cpu_cache_probe']['distinct_objects']
    assert cache['cpu_cache_probe']['first_seq_len'] == cache['cpu_cache_probe']['second_seq_len'] == 0
    assert cache['generation_defaults']['cache_implementation'] is None
    assert cache['cuda_initialized'] is False

    groups = []
    for frame in ['F0', 'F1']:
        for camera in ['ALL', 'base_camera', 'hand_camera']:
            cs = [r for r in candidates if r['frame_kind'] == frame and
                  (camera == 'ALL' or r['camera'] == camera)]
            good = [r for r in cs if r['query_object'] == 'peg' and r['target_good'] == 'True']
            groups.append(dict(frame=frame, camera=camera, candidates=len(cs),
                               labels=counts(r['strict_label'] for r in cs),
                               good_peg_boxes=len(good), good_peg_labels=counts(r['strict_label'] for r in good),
                               role_confusion={role: counts(r['strict_label'] for r in cs if r['GT_role'] == role)
                                               for role in ['PEG', 'BOX', 'ROBOT', 'AMBIGUOUS']}))
    regressions = []
    for t in rows('hybrid_transitions.csv'):
        if t['baseline'] != 'raw_top1' or t['regressed'] != 'True':
            continue
        old = next(r for r in selected if r['method'] == 'raw_top1' and
                   r['image_id'] == t['image_id'] and r['object'] == t['object'])
        c = by_id[old['candidate_id']]
        p = responses[c['id']]
        pix = json.loads(c['actor_pixels'])
        fg = sum(pix[k] for k in ['peg', 'box', 'robot'])
        same = [r for r in candidates if r['image_id'] == c['image_id'] and r['query_object'] == 'peg']
        regressions.append(dict(candidate_id=c['id'], image_id=c['image_id'],
                                parent_id=c['parent_id'], frame=c['frame_kind'], camera=c['camera'],
                                old_IoU=float(t['old_IoU']), new_IoU=float(t['new_IoU']),
                                Qwen_label=c['strict_label'], actor_pixels=pix,
                                peg_fraction_foreground=pix['peg']/fg,
                                peg_fraction_rectangle=pix['peg']/sum(pix.values()),
                                request_index=p['timing']['index'], input_tokens=p['timing']['input_tokens'],
                                same_query_candidates=len(same), same_query_labels=counts(r['strict_label'] for r in same),
                                box=json.loads(c['box']), crop_box=json.loads(c['crop_box'])))
    assert len(regressions) == 2
    assert all(r['Qwen_label'] == 'BOX' and r['old_IoU'] > .97 and r['new_IoU'] == 0 for r in regressions)
    duplicates = collections.defaultdict(list)
    for r in raw:
        duplicates[tuple(r['image_sha256'])].append(r)
    duplicate_groups = [g for g in duplicates.values() if len(g) > 1]
    order = []
    for start, end in [(0, 63), (63, 126), (126, 189), (189, 251)]:
        rs = raw[start:end]
        order.append(dict(start_zero_based=start, end_exclusive=end, n=len(rs),
                          labels=counts(r['strict_label'] for r in rs)))
    from PIL import Image
    crop_checks = 0
    for r in requests:
        c = by_id[r['id']]
        local = [BASE / Path(p).parts[2].removeprefix('task10i_stage4a2_') /
                 Path(*Path(p).parts[3:]) for p in r['images']]
        assert all(sha(p) == h for p, h in zip(local, r['image_sha256']))
        with Image.open(local[0]) as im, Image.open(local[2]) as crop:
            assert im.convert('RGB').crop(tuple(json.loads(c['crop_box']))).tobytes() == crop.convert('RGB').tobytes()
        crop_checks += 1
    result = dict(status='COMPLETE_VERIFIED', scope='Posthoc frozen-artifact diagnosis, not new model evaluation',
                  context=dict(messages_per_request='Fresh list containing one user message', images_per_request=3,
                               previous_answers_passed=False, explicit_cache_passed=False,
                               input_tokens_min=min(r['timing']['input_tokens'] for r in raw),
                               input_tokens_max=max(r['timing']['input_tokens'] for r in raw),
                               vision_forward_count_per_request=1,
                               cache_evidence='Installed generation preparation and Qwen M-RoPE source; CPU empty-cache probe',
                               limitation='Original run did not log entry cache identity/length hooks'),
                  time_camera=groups, recall_regressions=regressions, execution_order=order,
                  identical_input_groups=len(duplicate_groups),
                  identical_input_inconsistent_labels=sum(len({r['strict_label'] for r in g}) > 1 for g in duplicate_groups),
                  verification=dict(original_indexed_artifacts_verified=len(indexed), crop_pixel_equal=crop_checks,
                                    input_image_hashes_checked=len(requests)*3, source_matches_remote=True,
                                    original_artifacts_unchanged=True, new_Qwen_loads=0, new_Qwen_generations=0,
                                    new_DINO_SAM2_forwards=0, new_simulator_calls=0),
                  source_sha256=sha(Path(__file__)),
                  inputs={n:sha(RUN/n) for n in ['artifact_index.json','verifier_raw_responses.json',
                          'candidate_evaluation.csv','hybrid_transitions.csv','hybrid_selected_boxes.csv',
                          'request_manifest.json','prompt_identity.json']})
    assert all(sha(RUN / name) == item['sha256'] for name, item in indexed.items())
    (OUT/'audit_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],verified=result['verification'],regressions=regressions),indent=2))


if __name__ == '__main__':
    main()
