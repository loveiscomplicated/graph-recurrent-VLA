"""Collect named final deliverables without changing any scientific result.

Raw inputs, full query caches and per-frame Graphs stay in their original roots.
This review bundle copies metrics/protocols and records every source hash.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,independent,reporting):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False);sources=[];workspace=Path('/workspace')
    def old(name):return workspace/('task10i_stage4a2_'+name)/'run'
    def copy(name,source):
        assert source.is_file(),str(source)
        target=root/name;shutil.copyfile(source,target);assert sha(target)==sha(source)
        sources.append(dict(destination=name,source=str(source),sha256=sha(source)))
    def merged(name,inputs):
        rows=[];columns=[]
        for dataset,source in inputs:
            assert source.exists(),str(source)
            with source.open() as stream:
                for r in csv.DictReader(stream):rows.append(dict(evidence_dataset=dataset,**r))
            sources.append(dict(destination=name,source=str(source),sha256=sha(source),evidence_dataset=dataset))
        for r in rows:
            for k in r:
                if k not in columns:columns.append(k)
        assert columns,'cannot manufacture an empty metric table'
        with (root/name).open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(rows)
    freeze=independent/'freeze';evaluation=independent/'evaluation';final=evaluation/'final';internal=evaluation/'internal'
    assert read(reporting/'pipeline_state.json')['status']=='INDEPENDENT_DECISION_GALLERY_READBACK_COMPLETE'
    candidate=read(freeze/'candidate_config_frozen.json')['candidate']
    audit=old('20261101_v1')
    # The oldest identity accurately says the Small weights had not yet been
    # found then. Preserve that receipt; create a new completed identity here.
    identity=read(audit/'three_model_identity.json')
    identity['dinov2']=dict(model='facebook/dinov2-small',
        download_identity=read('/workspace/task10i_stage4a2_models/dinov2-small/download_identity.json'),
        live_identity=read(final/'candidate/dinov2_model_identity.json'),baseline_dependency=False)
    write(root/'three_model_identity.json',identity)
    for name in ('baseline_forward_equivalence.json','execution_call_graph.json','dataset_partition_manifest.json'):
        copy(name,audit/name)
    preprocessing=read(audit/'model_preprocessing_contract.json')
    preprocessing['Small_downloaded_default_processor_config']=read('/workspace/task10i_stage4a2_models/dinov2-small/preprocessor_config.json')
    preprocessing['Small_live_feature_preprocessing']=dict(input='currently observed uint8 RGB',
        resize='bilinear448x448, align_corners=False',center_crop=False,RGB_divisor=255,
        mean=[.485,.456,.406],std=[.229,.224,.225],autocast='CUDA bfloat16',
        extraction='last_hidden_state without CLS,1024x384 patch features',
        pooling='area-resized32x32 current predicted clean-mask weights, then L2 normalization',
        output_shape=[384],default_HF_processor_called_in_live_path=False,
        source='graph/task10i_stage4a2/dinov2_runtime.py:CurrentFeatures.embed')
    preprocessing['candidate_config']=read(freeze/'candidate_config_frozen.json')
    preprocessing['baseline_Small_NOT_EXECUTED']=True
    write(root/'model_preprocessing_contract.json',preprocessing)
    for name in ('candidate_source_frozen.json','candidate_config_frozen.json','adoption_criteria_frozen.json'):
        copy(name,freeze/name)
    copy('new_final_benchmark_protocol.json',old('benchmark_protocol_20261101_v1')/'new_final_benchmark_protocol.json')
    copy('final_collection_design.json',old('collection_protocol_20261101_v1')/'final_collection_design.json')
    copy('independent_benchmark_manifest.json',evaluation/'new_benchmark/independent_benchmark_manifest.json')
    copy('dino_raw_queries_audit.csv',old('dino_20261101_v2')/'dino_raw_queries_audit.csv')
    merged('dino_candidate_restriction_funnel.csv',[
        ('HISTORICAL_CONSERVATIVE_RECTANGLE_SUPPORT',old('dino_attribution_20261101_v1')/'dino_candidate_restriction_funnel.csv'),
        ('HISTORICAL_ACTUAL_SAM_OUTCOME',old('dino_outcomes_20261101_v1')/'dino_operational_outcome_funnel.csv')])
    copy('dino_prompt_comparison.csv',workspace/'task10i_stage4a2_controls_20261101_v1/prompt/dino_prompt_comparison.csv')
    copy('dino_assignment_error_cases.json',old('dino_outcomes_20261101_v1')/'dino_assignment_error_cases.json')
    merged('dino_candidate_selection_ablation.csv',[
        ('ORACLE_SAME_OUTPUT_NULL_EXCLUSION_DIAGNOSTIC',old('dino_attribution_20261101_v1')/'dino_same_output_assignment_counterfactual.csv'),
        ('DEVELOPMENT36_PRACTICAL_A',old('newdev_summary_20261101_v2')/'summary/tri_model_ablation_metrics.csv')])
    for name in ('dinov2_call_trace.json','dinov2_reference_provenance.json','dinov2_reference_contamination_audit.json'):
        copy(name,final/'matching'/name)
    merged('dinov2_positive_negative_pairs.csv',[
        ('HISTORICAL_PHYSICAL_ACTOR_DIAGNOSTIC',old('dinov2_20261101_v2')/'matching/dinov2_positive_negative_pairs.csv'),
        ('DEVELOPMENT_HYPOTHETICAL_VS_IDEAL_REFERENCE',old('reference_pairs_20261101_v1')/'dinov2_positive_negative_pairs.csv'),
        ('DEVELOPMENT_BACKGROUND_HARD_NEGATIVES',old('background_pairs_20261101_v1')/'dinov2_positive_negative_pairs.csv'),
        ('INDEPENDENT_FINAL_ACTUAL_COMPARISONS',final/'matching/dinov2_positive_negative_pairs.csv')])
    copy('dinov2_similarity_distributions.csv',root/'dinov2_positive_negative_pairs.csv')
    copy('dinov2_crop_feature_ablation.csv',old('component_controls_20261101_v1')/'matched_crop/dinov2_crop_feature_ablation.csv')
    merged('dinov2_reidentification_metrics.csv',[
        ('HISTORICAL_PHYSICAL_ACTOR_DIAGNOSTIC',old('dinov2_20261101_v2')/'matching/dinov2_reidentification_metrics.csv'),
        ('DEVELOPMENT_REFERENCE_CONTROL',old('reference_pairs_20261101_v1')/'dinov2_reidentification_metrics.csv'),
        ('DEVELOPMENT_BACKGROUND_NEGATIVES',old('background_pairs_20261101_v1')/'dinov2_reidentification_metrics.csv'),
        ('INDEPENDENT_FINAL_ACTUAL_SPARSE_MATCHING',final/'matching/dinov2_reidentification_metrics.csv')])
    temporal=read(final/'temporal/sam2_causal_recovery_metrics.json')
    drift=[]
    for r in temporal:
        for event in r['wrong_identity_onsets_after_earlier_correct_mask']:
            drift.append(dict(episode=r['episode'],parent_id=r['parent_id'],condition=r['condition'],camera=r['camera'],object_id=r['object_id'],
                event_kind='POSTHOC_WRONG_ACTOR_ONSET_AFTER_PRIOR_CORRECT_MASK',**event))
    with (root/'sam2_identity_drift_cases.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=['episode','parent_id','condition','camera','object_id','event_kind','start','end','frames']);w.writeheader();w.writerows(drift)
    with (old('sam_controls_20261101_v2')/'sam2_causal_recovery_metrics.csv').open() as stream:
        initialization=[r for r in csv.DictReader(stream) if r['timestep']=='0']
    with (root/'sam2_initialization_audit.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(initialization[0]));w.writeheader();w.writerows(initialization)
    write(root/'sam2_memory_state_trace.json',dict(baseline_official_outputs=read(audit/'sam2_memory_state_trace.json'),
        controlled_memory_API_trace=read(old('sam_controls_20261101_v2')/'sam2_memory_state_trace.json'),
        actual_Development_correction_trace=read(old('correction_diagnostics_20261101_v1')/'sam2_memory_state_trace.json'),
        Final_per_episode_state_machine={e['episode_id']:read(final/'candidate/perception'/str(e['episode_id'])/'state_machine.json') for e in read(final/'expected_counts.json')['entries']},
        official_signals_and_prediction_support_rules_are_distinct=True))
    copy('sam2_reanchor_event_log.csv',final/'matching/sam2_reanchor_event_log.csv')
    copy('sam2_causal_recovery_metrics.csv',final/'temporal/sam2_temporal_summary.csv')
    copy('sam2_memory_correction_ablation.csv',old('api_controls_20261101_v1')/'sam2_memory_correction_ablation.csv')
    write(root/'three_model_integration_protocol.json',dict(candidate=candidate,config=read(freeze/'candidate_config_frozen.json'),
        baseline_order='current RGB -> original DINO scan/confirmation/assignment -> original SAM -> current geometry/Graph; no DINOv2 call',
        candidate_order='current original SAM -> prediction-only support state -> event-driven current DINO -> separate current SAM candidate -> past-reference DINOv2 when available -> temporal/support validation -> fresh per-object memory commit -> unchanged current Graph; separately validated later original-initialization handoff',
        GT_evaluation='completed RGB-D-only predictions first; separate EXACT_GT evaluator thereafter',
        no_future_reference=True,no_cross_episode_bank=True,no_Production_changes=True,Graph_contract='Graph21/Edge4, current-only, cap128/object, Box Main-only'))
    studies=[('DEVELOPMENT36_A',old('newdev_summary_20261101_v2')/'summary'),('DEVELOPMENT36_P',old('newdev_P_20261101_v1')/'summary'),
        ('DEVELOPMENT36_C_KEEP',old('newdev_C_KEEP_20261101_v1')/'summary_C_KEEP'),
        ('DEVELOPMENT36_BC_KEEP',old('newdev_C_KEEP_20261101_v1')/'summary_BC_KEEP'),
        ('DEVELOPMENT36_MAIN_BOX',old('newdev_MAIN_BOX_20261101_v1')/'summary'),
        ('DEVELOPMENT36_WB_HANDOFF',old('newdev_WB_HANDOFF_20261101_v1')/'summary'),
        ('INTERNAL30',internal/'summary'),('FINAL_NORMAL32',final/'summary_normal'),('FINAL_SYNTHETIC_STRESS12',final/'summary_stress')]
    for name in ('tri_model_ablation_metrics.csv','final_graph_reliability_metrics.csv','per_camera_per_phase_results.csv'):
        merged(name,[(label,folder/name) for label,folder in studies])
    merged('false_positive_safety_audit.csv',[(label,folder/'false_positive_safety_audit.csv') for label,folder in [
        ('DEVELOPMENT36_WB_HANDOFF',old('newdev_WB_HANDOFF_20261101_v1')/'registered'),
        ('INTERNAL30',internal/'registered'),('FINAL_NORMAL32',final/'registered_normal'),('FINAL_SYNTHETIC_STRESS12',final/'registered_stress')]])
    copy('latency_and_gpu_metrics.csv',final/'runtime/latency_and_gpu_metrics.csv')
    for name in ('internal_test_metrics.json','independent_final_test_metrics.json','perception_reliability_decision.json','stage4b_handoff_receipt.json','verification.json'):
        copy(name,reporting/'decision'/name)
    copy('new_independent_gallery_index.json',reporting/'gallery/gallery_index.json')
    write(root/'source_artifact_map.json',sources)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_named_deliverables_collection',seed=42,
            root=root/'experiments',job_type='analysis',config=dict(source_sha256=sha(Path(__file__)),candidate=candidate)) as logger:
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.iterdir() if p.is_file()}
        assert all(sha(root/p)==v['sha256'] for p,v in files.items())
        write(root/'artifact_index.json',dict(status='NAMED_DELIVERABLES_SHA_READBACK_PASS',run_id=logger.run_id,files=files,
            raw_and_Graph_artifacts_retained_at_original_roots=True,normal_and_synthetic_stress_not_pooled_as_primary=True,
            full_report_static_HTML_local_readback_and_completion_audit_pending=True,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','independent','reporting'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
