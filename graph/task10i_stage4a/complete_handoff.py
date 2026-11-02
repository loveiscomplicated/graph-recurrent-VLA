"""Finalize an inspected report only after every scientific/evidence gate finished.

Run locally after copying final receipts, reviewing actual gallery/plot images,
and saving the complete report to docs/visual-graph-integration. This records a
new common ExperimentLogger identity; it never adopts a perception candidate.
"""
import argparse
from pathlib import Path
from .run_pipeline import read,write,sha


def run(evidence,report,review,output):
    from utils.experiment_logger import ExperimentLogger
    required={
        'posthoc/report_evidence_ready.json':('status','REPORT_EVIDENCE_READY'),
        'posthoc/handoff_audit/protection_checks.json':('status','PASS'),
        'posthoc/handoff_audit/causal_graph_checks.json':('status','PASS'),
        'camera_contract/scope.json':('status','RECORDED_CAMERA_CONTRACT_PASS'),
        'oracle_graph/scope.json':('status','ORACLE_GRAPH_STRUCTURE_AUDITED'),
        'oracle_grouping/oracle_grouping_control.json':('status','NATIVE_GROUPED_GT_BOX_CONTROL_FINISHED'),
        'detector_outcomes/scope.json':('status','DETECTOR_AND_IDENTITY_OUTCOMES_FINISHED'),
        'initialization_bottlenecks/scope.json':('status','T0_ASSIGNMENT_RECONSTRUCTION_PASS'),
        'development_temporal/scope.json':('status','DEVELOPMENT_TEMPORAL_COMPARISONS_FINISHED'),
        'refinement_retention/scope.json':('status','RAW_REANCHOR_REFINEMENT_RETENTION_FINISHED'),
        'global_missing_oracles/scope.json':('status','GLOBAL_GRAPH_MISSING_ORACLES_FINISHED'),
        'approximate_reference/reference_accounting.json':('status','SEPARATE_APPROXIMATE_REFERENCE_SUPPLEMENT_FINISHED'),
        'forward_equivalence/new_host_forward_equivalence.json':('status','PASS'),
        'cache_equivalence/equivalence.json':('status','REAL_DATA_PREPARE_CACHE_EQUIVALENCE_PASS'),
    }
    for relative,(field,value) in required.items():assert read(evidence/relative)[field]==value,relative
    handoff=read(evidence/'finalization/handoff.json')
    assert handoff['status']=='SCIENTIFIC_EVALUATIONS_FINISHED_REPORT_REVIEW_PENDING'
    assert not handoff['production_adoption'] and not handoff['policy_training']
    approximate=read(evidence/'approximate_reference/reference_accounting.json')
    assert approximate['exact_reference_flags_unchanged'] and approximate['primary_decisions_unchanged']
    protection=read(evidence/'posthoc/handoff_audit/protection_checks.json')
    assert protection['original_raw_gt_files']==712
    causal=read(evidence/'posthoc/handoff_audit/causal_graph_checks.json')
    assert sum(r['frames'] for r in causal['partitions'])==13150
    for name,count in [('internal_test',5903),('validation',3085)]:
        p=read(evidence/('finalization/'+name+'/partition_summary.json'))
        assert p['frames']==count and p['complete_registered_partition'] and p['no_candidate_changes_after_freeze']
    reviewed=read(review)
    assert reviewed['performed_by']=='Codex' and reviewed['report_sha256']==sha(report)
    assert reviewed['report_links_checked'] and reviewed['numeric_and_scope_review_complete']
    assert report.parent.name=='visual-graph-integration' and report.parent.parent.name=='docs'
    text=report.read_text()
    assert '**상태: COMPLETED' in text and '## 남은 최종 검증' not in text
    gallery=read(evidence/'posthoc/gallery/case_manifest.json')
    for case in gallery['cases']:
        if case['status']=='EXACT_GT_CASE':
            name='posthoc/gallery/'+case['image']
            assert reviewed['inspected_images'][name]==sha(evidence/name),name
    for item in reviewed['additional_inspected_images']:
        assert sha(item['path'])==item['sha256']
    output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_final_report_review_handoff',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(role='artifact_review_only',
        scientific_root=handoff['final_source_root'],production_adoption=False,model_forwards=0)) as log:
        inputs={r:sha(evidence/r) for r in required}
        payload=dict(status='COMPLETED_RESEARCH_INVESTIGATION',whole_goal_complete=True,run_id=log.run_id,
            scientific_decision=handoff['overall_decision'],candidate=handoff['selected_challenger'],
            production_adoption=False,policy_training=False,external_publishing=False,
            report=str(report.resolve()),report_sha256=sha(report),review_receipt=str(review.resolve()),review_sha256=sha(review),
            scientific_handoff_sha256=sha(evidence/'finalization/handoff.json'),required_evidence=inputs,
            original_raw_gt_unchanged=712,evaluated_logical_sources=208,evaluated_observations=13150,
            reference_limitations_not_erased=True,unsupported_constraints_not_counted_as_pass=True,
            scientific_success_not_implied_by_completion=True)
        write(output/'completion_audit.json',payload)
        log.log(dict(evaluated_logical_sources=208,evaluated_observations=13150,protected_original_raw_gt_files=712))
        log.link_output(output/'completion_audit.json',kind='metrics');log.link_output(report,kind='reports')
        log.link_output(review,kind='metrics')
    print(output/'completion_audit.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('report',type=Path)
    p.add_argument('review',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.evidence,a.report,a.review,a.output)
