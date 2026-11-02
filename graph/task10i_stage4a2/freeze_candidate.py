"""Freeze one concrete Development-selected challenger before Internal/Final.

Freezing permits independent evaluation only. It never grants adoption or
publishing, and records Development gate failures without relabelling them.
"""
import argparse
import importlib.metadata
from pathlib import Path
import shutil
from .reanchor_state_machine import Rules
from dataclasses import asdict
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,development,forward_source,evaluation_source,collector_source,profile,design,protocol,partition,candidate,rationale):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    status=read(development/'pipeline_state.json')['status']
    assert status in ('DEVELOPMENT_MAIN_BOX_SCOPE_EVALUATION_COMPLETE','DEVELOPMENT_WRIST_BOX_HANDOFF_EVALUATION_COMPLETE')
    assert candidate in ('BC_MAIN_BOX','BC_WB_HANDOFF')
    gate=development/'registered/statistical_gate_receipt.json';metrics=read(gate)
    assert metrics['candidate']==candidate and metrics['scope']=='development' and metrics['parents']==36
    smoke=read(profile/'scope.json');assert smoke['status']=='LIVE_PROFILE_WRAPPER_HANDOFF_GRAPH_EQUIVALENT'
    assert smoke['internal_original_baseline_frames_equal']==16
    development_files=read(development/'source_config_frozen.json')['files']
    for name in ('tri_model_integration.py','reanchor_state_machine.py','dinov2_runtime.py','development_correction.py','model_trace.py','frozen_inference.py'):
        assert sha(forward_source/'graph/task10i_stage4a2'/name)==next(h for p,h in smoke['source'].items() if Path(p).name==name)
        if name!='frozen_inference.py':
            expected_sha=next(h for p,h in development_files.items() if p.endswith('/graph/task10i_stage4a2/'+name))
            assert sha(forward_source/'graph/task10i_stage4a2'/name)==expected_sha,'candidate core differs from selected Development study'
    specification=read(design)
    assert specification['status']=='FROZEN' and specification['new_final_benchmark_protocol_sha256']==sha(protocol)
    registered=read(protocol)
    seeds=[e['seed'] for queue in registered['queues'].values() for e in queue]
    engineering_seeds={0,42,30080,71300001,71300002,71300003,71300004,42424242}
    assert len(seeds)==len(set(seeds)) and not set(seeds)&engineering_seeds
    assert sha(collector_source/'graph/task10i_stage4a2/final_capture.py')==specification['collector_source_sha256']
    criteria=Path('/workspace/task10i_stage4a2_criteria_20261101_v1/run/adoption_criteria_frozen.json')
    assert sha(criteria)==metrics['criteria_sha256']
    assert len(read(partition)['development'])==36 and len(read(partition)['internal_holdout'])==30
    model=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
    paths=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
    files={}
    for source in set((forward_source,evaluation_source,collector_source)):
        files.update({str(p):sha(p) for p in source.rglob('*.py')})
    for p,h in model['dino_files'].items():
        path=Path(paths['dino'])/p;assert sha(path)==h;files[str(path)]=h
    assert sha(paths['sam'])==model['sam_sha256'];files[paths['sam']]=model['sam_sha256']
    small=Path('/workspace/task10i_stage4a2_models/dinov2-small');identity=read(small/'download_identity.json')
    for p,h in identity['files'].items():
        assert sha(small/p)==h;files[str(small/p)]=h
    import sam2
    files.update({str(p):sha(p) for p in Path(sam2.__file__).parent.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')})
    files.update({str(p):sha(p) for p in Path('/workspace/task10g_f1_abc_20261031/full/inputs').rglob('*') if p.is_file()})
    normalizer=Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json');files[str(normalizer)]=sha(normalizer)
    shutil.copyfile(criteria,root/'adoption_criteria_frozen.json')
    source=dict(status='FROZEN',files=files,forward_source_root=str(forward_source),evaluation_source_root=str(evaluation_source),
        collector_source_root=str(collector_source),profile_engineering_sha256=sha(profile/'scope.json'),
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','transformers','tokenizers','numpy','scipy','mani_skill','sapien')})
    slots=[['base_camera',1]] if candidate=='BC_MAIN_BOX' else [['hand_camera',0],['base_camera',1]]
    config=dict(status='FROZEN',candidate=candidate,correction_slots=slots,geometry_cache=True,lazy_matching=True,
        late_baseline_handoff=candidate=='BC_WB_HANDOFF',rules=asdict(Rules(minimum_detection_score=.2)),
        baseline_handoff=dict(minimum_nonrobot_fraction=.9,minimum_old_support_overlap=.8,minimum_clean_area_ratio=1.,
            require_new_original_initialization_after_early_anchor=True,reid_threshold=.6,negative_margin=.05),
        source_root=str(evaluation_source),forward_source_root=str(forward_source),collector_source_root=str(collector_source),
        selection_rationale=rationale,selection_data='historical studies + new Development36 only',
        development_gate_receipt=str(gate),development_gate_receipt_sha256=sha(gate),
        development_failures_retained=metrics['failures'],development_unsupported_constraints=metrics['unsupported'],
        allow_internal_evaluation=True,allow_independent_final_capture=True,
        continue_to_final_without_retuning_after_internal=True,
        adoption_criteria_sha256=sha(root/'adoption_criteria_frozen.json'),
        new_final_benchmark_protocol_sha256=sha(protocol),final_collection_design_sha256=sha(design),
        partition_sha256=sha(partition),limited_scope_only=True,overall_pipeline_GO_authorized=False,
        registered_seed_count=len(seeds),engineering_or_neutral_reset_seed_collisions=0,
        validation_driven_tuning_permitted=False,production_replacement=False,Stage4B_full_recache=False,training=False,
        Independent_Internal_opened_before_freeze=False,new_Final_opened_before_freeze=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_independent_challenger_source_freeze',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        write(root/'candidate_source_frozen.json',source);write(root/'candidate_config_frozen.json',config)
        write(root/'candidate_selection_receipt.json',dict(status='ONE_CHALLENGER_FROZEN_FOR_INDEPENDENT_EVALUATION_ONLY',
            run_id=logger.run_id,candidate=candidate,source_sha256=sha(root/'candidate_source_frozen.json'),
            config_sha256=sha(root/'candidate_config_frozen.json'),criteria_sha256=sha(root/'adoption_criteria_frozen.json'),
            entire_goal_complete=False,GO_claim=False,other_attempts_preserved=True))
        logger.link_output(root/'candidate_selection_receipt.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','development','forward_source','evaluation_source','collector_source','profile','design','protocol','partition'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);p.add_argument('--rationale',required=True);a=p.parse_args()
    run(a.root,a.development,a.forward_source,a.evaluation_source,a.collector_source,a.profile,a.design,a.protocol,a.partition,a.candidate,a.rationale)
