"""Strict post-run receipt inventory and SHA readback, without model calls.

This verifies the independent evidence pipeline, not the scientific conclusion
or the full task. Report review, gallery QA and the final completion audit remain
separate. Run only after the registered evaluations and diagnostics finish.
"""
import argparse
import hashlib
import json
from pathlib import Path


INDEPENDENT_STAGES = (
    'internal_manifest', 'internal_B0', 'internal_challenger',
    'internal_reference', 'internal_masks', 'internal_graph',
    'internal_consistency', 'internal_summary', 'internal_registered',
    'new_capture', 'final_manifest', 'final_manifest_normal',
    'final_manifest_stress', 'final_B0', 'final_challenger', 'final_runtime',
    'final_reference', 'final_matching_audit', 'final_masks',
    'final_temporal_masks', 'final_graph', 'final_consistency',
    'final_summary_normal', 'final_registered_normal',
    'final_summary_stress', 'final_registered_stress',
)
SUPPLEMENTAL_STAGES = tuple('internal_' + suffix for suffix in
    ('matching', 'temporal', 'geometry', 'recovery', 'phase')) + tuple(
    scope + '_' + suffix for scope in ('mini', 'final') for suffix in
    ('raw', 'oracle', 'geometry', 'recovery', 'phase'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_stages(root, expected, cache=None):
    """Missing receipts fail even if all receipts that happen to exist pass."""
    expected = set(expected)
    actual = {p.stem for p in (root / 'stages').glob('*.json')}
    assert actual == expected, dict(root=str(root), missing=sorted(expected-actual),
                                   unexpected=sorted(actual-expected))
    cache = {} if cache is None else cache
    verified = []
    for name in sorted(expected):
        receipt = root / 'stages' / (name + '.json')
        data = read(receipt)
        assert data['status'] == 'PASS', str(receipt)
        assert data['files'], 'empty stage evidence: ' + str(receipt)
        for relative, wanted in data['files'].items():
            path = (root / relative).resolve()
            assert path.is_relative_to(root.resolve()), str(path)
            assert path.is_file(), str(path)
            if path not in cache:
                cache[path] = sha(path)
            assert cache[path] == wanted, str(path)
        verified.append(dict(stage=name, receipt_sha256=sha(receipt),
                             file_references=len(data['files'])))
    return verified


def verify_counts(base, parents):
    expected = read(base / 'expected_counts.json')
    entries = expected['entries']
    assert len(entries) == expected['episodes']
    assert len({e['parent_id'] for e in entries}) == expected['parents'] == parents
    assert sum(e['actions'] + 1 for e in entries) == expected['frames']
    by_id = {str(e['episode_id']): e for e in entries}
    assert len(by_id) == len(entries)
    for condition in ('baseline', 'candidate'):
        folder = base / condition
        state = read(folder / 'pipeline_state.json')
        assert state['status'] == 'FROZEN_LIVE_CONDITION_COMPLETE'
        rows = read(folder / 'progress.json')
        assert len(rows) == expected['episodes'] == state['episodes']
        assert sum(r['frames'] for r in rows) == expected['frames'] == state['frames']
        assert {str(r['episode']) for r in rows} == set(by_id)
        for row in rows:
            e = by_id[str(row['episode'])]
            assert row['raw_sha256'] == e['raw_sha256']
            assert row['frames'] == e['actions'] + 1
            assert row['GT_forward_input'] is False
    consistency = read(base / 'live_graph_consistency.json')
    assert consistency['status'] == 'LIVE_AND_POSTHOC_GRAPH_BYTE_EQUAL'
    for eid in by_id:
        isolation = read(base / 'masks/perception' / eid / 'baseline_memory_isolation.json')
        assert isolation['status'] == 'PASS' and isolation['all_frames_byte_equal'] is True
    return {key: expected[key] for key in ('parents', 'episodes', 'frames')}


def run(independent, reporting, finalization, supplemental, strata, output):
    terminals = (
        (independent, 'FROZEN_INDEPENDENT_RESULTS_READY_FOR_FINAL_REPORT'),
        (reporting, 'INDEPENDENT_DECISION_GALLERY_READBACK_COMPLETE'),
        (finalization, 'NAMED_DELIVERABLES_READY_FOR_FULL_MARKDOWN_AND_LOCAL_READBACK'),
        (supplemental, 'POST_TEST_SUPPLEMENTAL_ROOT_CAUSE_DIAGNOSTICS_COMPLETE'),
        (strata, 'ALL_DESCRIPTIVE_STRATA_COMPLETE'),
    )
    for root, wanted in terminals:
        assert read(root / 'pipeline_state.json')['status'] == wanted, str(root)
    cache = {}
    stages = {}
    for label, root, expected in (
        ('independent', independent / 'evaluation', INDEPENDENT_STAGES),
        ('reporting', reporting, ('decision', 'gallery')),
        ('supplemental', supplemental, SUPPLEMENTAL_STAGES),
        ('strata', strata, ('mini', 'internal', 'final')),
    ):
        stages[label] = verify_stages(root, expected, cache)
    evaluation = independent / 'evaluation'
    counts = {scope: verify_counts(evaluation / scope, parents)
              for scope, parents in (('internal', 30), ('final', 44))}
    for scope, minimum in (('normal', 32), ('stress', 12)):
        spec = read(evaluation / 'final' / (scope + '_expected_counts.json'))
        assert spec['parents'] >= minimum
        counts['final_' + scope] = {key: spec[key] for key in ('parents', 'episodes', 'frames')}
    original = read(supplemental / 'independent_metrics_before_diagnostics.json')
    assert original == {name: sha(evaluation / name / 'statistical_gate_receipt.json')
                        for name in original}
    named = finalization / 'named'
    index = read(named / 'artifact_index.json')
    for name, info in index['files'].items():
        assert sha(named / name) == info['sha256']
        assert (named / name).stat().st_size == info['bytes']
    result = dict(status='INDEPENDENT_EVIDENCE_INVENTORY_AND_SHA_READBACK_PASS',
        source_sha256=sha(Path(__file__)), stages=stages, counts=counts,
        distinct_stage_artifacts_read=len(cache),
        original_registered_metrics_unchanged_after_diagnostics=True,
        named_index_sha256=sha(named / 'artifact_index.json'),
        scientific_interpretation_and_full_completion_not_certified=True,
        report_gallery_QA_and_requirement_audit_still_required=True,
        whole_goal_complete=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'preserve prior readback receipt'
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('independent', 'reporting', 'finalization', 'supplemental', 'strata', 'output'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
