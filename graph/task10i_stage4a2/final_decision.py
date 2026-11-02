"""Conservative decision from immutable registered independent results only.

This is a post-evaluation report generator. It cannot tune, infer, publish,
replace perception, recache Stage4B or train a policy.
"""
import argparse
import csv
from pathlib import Path
from graph.task10i_stage4a.run_pipeline import read,write,sha


def classify(statistical,runtime,contracts_verified):
    if not contracts_verified:return 'INCONCLUSIVE'
    checks=[r for s in statistical for r in s['checks']]+runtime
    if any(r['status']=='FAIL' for r in checks):return 'NO_GO'
    if any(r['status']!='PASS' for r in checks):return 'INCONCLUSIVE'
    return 'CONDITIONAL_GO'  # Only the preregistered limited slots, never global GO.


def registered(folder):
    receipt=read(folder/'statistical_gate_receipt.json')
    with (folder/'registered_gate_results.csv').open() as stream:
        checks=list(csv.DictReader(stream))
    assert receipt['total']==len(checks)
    assert len(receipt['failures'])==sum(r['status']=='FAIL' for r in checks)
    assert all(sha(folder/p)==h for p,h in receipt['files'].items())
    return dict(receipt=receipt,checks=checks,path=str(folder/'statistical_gate_receipt.json'),
        sha256=sha(folder/'statistical_gate_receipt.json'))


def run(root,evaluation,freeze):
    from utils.experiment_logger import ExperimentLogger
    from .final_capture import freeze_barrier
    root.mkdir(parents=True,exist_ok=False)
    identity=freeze_barrier(freeze,None,None,scope='internal')
    config=read(freeze/'candidate_config_frozen.json')
    assert read(evaluation/'pipeline_state.json')['status']=='FROZEN_INTERNAL_AND_NEW_FINAL_EVALUATION_ARTIFACTS_COMPLETE'
    # The coordinator has verified every stage artifact before its atomic
    # receipt. Read all committed outputs again, including the actual masks,
    # segmentation sidecars, graph tensors, provenance and final capture.
    stages=[]
    for p in sorted((evaluation/'stages').glob('*.json')):
        r=read(p);assert r['status']=='PASS'
        assert all(sha(evaluation/name)==h for name,h in r['files'].items()),str(p)
        stages.append(dict(path=str(p),sha256=sha(p),files=len(r['files'])))
    internal=registered(evaluation/'internal/registered')
    normal=registered(evaluation/'final/registered_normal')
    stress=registered(evaluation/'final/registered_stress')
    criteria=read(freeze/'adoption_criteria_frozen.json')
    assert normal['receipt']['parents']>=criteria['sample_review']['minimum_normal_parents']
    assert stress['receipt']['parents']>=criteria['sample_review']['minimum_stress_parents']
    assert all(r['receipt']['criteria_sha256']==sha(freeze/'adoption_criteria_frozen.json') for r in (internal,normal,stress))
    runtime=read(evaluation/'final/runtime/runtime_gate_results.json')
    final_runtime=[r for r in runtime if r['stratum'] in ('normal','stress')]
    assert len(final_runtime)==6
    consistency={scope:read(evaluation/scope/'live_graph_consistency.json') for scope in ('internal','final')}
    capture=read(evaluation/'new_benchmark/independent_benchmark_manifest.json')
    assert capture['status']=='SYNCHRONOUS_BENCHMARK_CAPTURE_COMPLETE'
    decision=classify([normal,stress],final_runtime,True)
    payload=dict(status='FROZEN_INDEPENDENT_PERCEPTION_RELIABILITY_DECISION_COMPLETE',decision=decision,
        candidate=config['candidate'],correction_slots=config['correction_slots'],limited_scope_only=True,
        source_and_config=identity,Internal_used_for_tuning=False,Final_used_for_tuning=False,
        Internal30=internal,independent_normal=normal,independent_stress=stress,
        runtime=final_runtime,contracts_verified=True,live_Graph_five_tensor_consistency=consistency,
        source_files_readback=True,full_committed_artifact_readback=True,
        conservative_policy='Any registered benefit, safety or runtime FAIL prevents adoption; unsupported constraints remain INCONCLUSIVE; no retrospective threshold/criterion adjustment',
        sample_limit='Parent-bootstrap intervals do not certify a population rare-error probability below1%; normal/stress are separate and stress is synthetic',
        Stage4A_previous_NO_GO_preserved=True,Stage4B_full_recache_authorized=False,
        production_replacement_authorized=False,training_authorized=False,publishing_authorized=False,
        complete_report_and_gallery_required=True,whole_goal_complete=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_independent_reliability_decision',seed=42,
            root=root/'experiments',job_type='analysis',config=dict(source_sha256=sha(Path(__file__)),**identity)) as logger:
        payload['run_id']=logger.run_id
        write(root/'internal_test_metrics.json',dict(**internal,source_or_threshold_retuned=False))
        write(root/'independent_final_test_metrics.json',dict(normal=normal,stress=stress,runtime=final_runtime,
            normal_and_stress_pooled_as_primary=False,exact_reference_only=True))
        write(root/'perception_reliability_decision.json',payload)
        write(root/'stage4b_handoff_receipt.json',dict(status='RESEARCH_HANDOFF_ONLY_NO_REBUILD',decision=decision,
            candidate=config['candidate'],decision_sha256=sha(root/'perception_reliability_decision.json'),
            source_config=identity,production_changed=False,Stage4B_full_recache_started=False,
            Stage4B_full_recache_authorized=False,GNN_GRU_training_started=False,
            downstream_actions='Read the full research report and registered safety/runtime failures; no automatic recache, adoption or training'))
        write(root/'verification.json',dict(status='FULL_INDEPENDENT_ARTIFACT_SHA_READBACK_PASS',stages=stages,
            files_verified=sum(s['files'] for s in stages),freeze=identity,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','evaluation','freeze'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
