"""Conservative original numeric gates; no validation-dependent threshold changes."""
from pathlib import Path
from .run_pipeline import read,write,sha


def contract(protocol):
    return dict(numeric_gates=protocol['selection'],
        primary='parent-macro correct object presence in final Graph, GT-visible support>=8',
        primary_key='graph/0/object_recall',
        continuous_quality='visible_coverage_4mm and boundary coverage reported separately; not substituted for the recall gate',
        aggregation='same-frame -> episode mean -> parent mean; paired differences',
        zero_denominator='UNSUPPORTED, never counted as safety evidence',
        latency='observer-free fixed3 Development20-frame causal prefixes, all-frame p95 ratio',
        challenger_selection='fewest measurable safety failures, then largest visible surface coverage gain; deterministic name tie-break',
        no_candidate_meeting_gates='NO_GO; freeze best research challenger for independent falsification, not production adoption',
        internal_test='confirm chosen fixed challenger; do not retune or switch after test',
        validation='all registered16 Shared Validation parents and all their observations, paired once after final freeze')


def evaluate(comparison,latency,protocol):
    rows={r['metric']:r for r in read(comparison/'paired_macro.json') if r['scope']=='all'}
    cfg=protocol['selection'];checks=[]
    def lower(name,key,minimum):
        value=rows[key]['delta'] if key in rows else None
        checks.append(dict(name=name,metric=key,delta=value,minimum=minimum,
            status='UNSUPPORTED' if value is None else 'PASS' if value>=minimum else 'FAIL'))
    lower('primary_graph_recall_gain','graph/0/object_recall',cfg['minimum_parent_macro_visible_graph_recall_gain'])
    lower('Box_graph_recall','graph/1/object_recall',-cfg['maximum_box_graph_recall_loss'])
    for camera in ('base_camera','hand_camera'):
        for oid in (0,1):lower('pixel_precision_'+camera+'_'+str(oid),f'mask/{camera}/{oid}/precision',-cfg['maximum_pixel_precision_loss'])
    for oid in (0,1):
        for key,limit in [('robot_fraction',cfg['maximum_robot_contamination_increase']),('wrong_object_fraction',cfg['maximum_wrong_object_contamination_increase']),('false_presence',cfg['maximum_false_object_presence_increase'])]:
            r=rows.get(f'graph/{oid}/{key}',{});value=r.get('delta')
            checks.append(dict(name=f'graph_{oid}_{key}',metric=f'graph/{oid}/{key}',delta=value,maximum=limit,
                status='UNSUPPORTED' if value is None else 'PASS' if value<=limit else 'FAIL'))
    ratio=read(latency/'latency.json')['summaries']['all_60_frames']['candidate_to_baseline_p95_ratio']
    checks.append(dict(name='latency',ratio=ratio,maximum=cfg['maximum_p95_latency_ratio'],status='PASS' if ratio<=cfg['maximum_p95_latency_ratio'] else 'FAIL'))
    ref=read(comparison/'reference_support.json')
    checks.append(dict(name='complete_reference_support',unsupported=ref['unsupported'],status='PASS' if ref['unsupported']==0 else 'UNSUPPORTED'))
    passed=all(c['status']=='PASS' for c in checks)
    return dict(decision='GO_RESEARCH_ONLY' if passed else 'NO_GO',checks=checks,
        coverage_gain=rows.get('graph/0/visible_coverage_4mm',{}).get('delta'),
        primary_gain=rows.get('graph/0/object_recall',{}).get('delta'),
        primary_baseline=rows.get('graph/0/object_recall',{}).get('baseline'),
        measurable_safety_failures=sum(c['status']=='FAIL' for c in checks if c['name']!='primary_graph_recall_gain'),
        unsupported_constraints=[c['name'] for c in checks if c['status']=='UNSUPPORTED'],
        no_20hz_claim=True,no_production_adoption=True)


def freeze(source,output,stage,candidate,mode,decision,previous=None):
    from datetime import datetime,timezone
    original=read(Path('/workspace/task10i_stage4a_20261031_v5/run/source_freeze.json'))['files']
    files={name:sha(source/name) for name in original if not name.startswith('graph/task10i_stage4a/')}
    for name in ('reliable_reanchor.py','depth_component.py','depth_box_component.py','paired_candidate.py','perframe_cache.py','run_pipeline.py','instrumentation.py'):
        p=source/'graph/task10i_stage4a'/name
        if p.exists():files[str(p.relative_to(source))]=sha(p)
    cache_proof=Path('/workspace/task10i_stage4a_cache_equivalence_20261101_v1/equivalence.json')
    assert read(cache_proof)['status']=='REAL_DATA_PREPARE_CACHE_EQUIVALENCE_PASS'
    assert sha(source/'graph/task10i_stage4a/perframe_cache.py')==sha(cache_proof.parent/'source/graph/task10i_stage4a/perframe_cache.py')
    thread_proof=read(Path('/workspace/task10i_stage4a_blas_equivalence_20261101_v1/run/new_host_forward_equivalence.json'))
    assert thread_proof['status']=='PASS' and thread_proof['frames']==101
    payload=dict(stage=stage,candidate=candidate,mode=mode,inference_files=files,decision=decision,
        cpu_thread_environment={k:__import__('os').environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        cpu_thread_equivalence_proof='/workspace/task10i_stage4a_blas_equivalence_20261101_v1/run/new_host_forward_equivalence.json',
        exact_current_frame_memo=True,memo_equivalence_proof=str(cache_proof),memo_equivalence_sha256=sha(cache_proof),
        frozen_utc=datetime.now(timezone.utc).isoformat(),production_adoption=False,
        previous_freeze=str(previous) if previous else None,previous_freeze_sha256=sha(previous) if previous else None,
        validation_used_for_selection=False,configuration_change_after_internal_test=False)
    assert not output.exists();write(output,payload)
    return payload
