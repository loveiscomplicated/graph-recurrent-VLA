"""Exercise fresh-collector serialization through live models and paired GT.

Uses the already captured engineering30080 sequence, never a registered Final
seed. This verifies the evaluation adapters before freezing the challenger.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .lazy_feature_control import wait_for
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,dependency,collector,candidate='BC_KEEP',profiled=False):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    receipt=read(collector/'normal/sequence/receipt.json')
    assert receipt['parent_id']=='engineering30080_normal' and receipt['frames']==26
    assert all(sha(collector/'normal/sequence'/p)==h for p,h in receipt['files'].items())
    entry={k:v for k,v in receipt.items() if k not in ('rows','files')}
    expected=root/'expected_counts.json';write(expected,dict(parents=1,episodes=1,frames=26,entries=[entry]))
    config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        expected_sha256=sha(expected),collector_receipt_sha256=sha(collector/'normal/sequence/receipt.json'),
        Final_registered_seeds_opened=False,engineering_only=True,condition=candidate,geometry_cache=True,lazy_matching=True,profiled=profiled)
    write(root/'source_config_frozen.json',config)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_final_path_engineering',seed=42,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        wait_for(root,dependency/'pipeline_state.json','LIVE_PROFILE_WRAPPER_HANDOFF_GRAPH_EQUIVALENT' if profiled else 'LAZY_ACTUAL_REID_EQUIVALENCE_COMPLETE',
                 'graph.task10i_stage4a2.profile_engineering' if profiled else 'graph.task10i_stage4a2.lazy_feature_control',dependency)
        prefix=[sys.executable,'-B','-u','-m']
        commands=[
            ('baseline',prefix+['graph.task10i_stage4a2.frozen_inference',str(root/'baseline'),str(expected),'--condition','B0','--engineering']),
            ('candidate',prefix+['graph.task10i_stage4a2.frozen_inference',str(root/'candidate'),str(expected),'--condition',candidate,'--engineering','--geometry-cache','--lazy-matching']),
            ('combine',prefix+['graph.task10i_stage4a2.evaluation_inputs','combine',str(root/'masks'),str(expected),
                '--baseline',str(root/'baseline'),'--challenger',str(root/'candidate'),'--candidate',candidate,
                *(['--require-baseline-isolation'] if profiled else [])]),
            ('references',prefix+['graph.task10i_stage4a2.evaluation_inputs','references',str(root/'references'),str(expected),
                '--baseline',str(root/'baseline'),'--challenger',str(root/'candidate')]),
            ('paired',prefix+['graph.task10i_stage4a2.paired_evaluation',str(root/'paired'),str(root/'masks'),str(root/'references'),str(expected),'--candidate',candidate]),
            ('consistency',prefix+['graph.task10i_stage4a2.live_graph_consistency',str(root/'live_graph_consistency.json'),
                str(expected),str(root/'paired'),str(root/'masks'),'--candidate',candidate]),
            ('summary',prefix+['graph.task10i_stage4a2.aggregate_evaluation',str(root/'summary'),str(root/'paired'),str(expected),'--candidate',candidate]),
            ('registered_gates',prefix+['graph.task10i_stage4a2.registered_evaluation',str(root/'registered'),str(root/'paired'),
                str(expected),'/workspace/task10i_stage4a2_criteria_20261101_v1/run/adoption_criteria_frozen.json','--candidate',candidate,'--scope','development'])]
        if profiled:
            commands.extend([
                ('matching',prefix+['graph.task10i_stage4a2.matching_outcomes',str(root/'matching'),str(expected),str(root/'baseline'),
                    str(root/'candidate'),str(root/'references'),'--candidate',candidate]),
                ('runtime',prefix+['graph.task10i_stage4a2.latency_report',str(root/'runtime'),str(expected),str(root/'baseline'),str(root/'candidate'),
                    '/workspace/task10i_stage4a2_criteria_20261101_v1/run/adoption_criteria_frozen.json','--candidate',candidate])])
        for name,command in commands:
            write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,command=command,pid=os.getpid()))
            for attempt in range(2):
                log=root/(name+f'_attempt{attempt}.log')
                with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                if not result.returncode:break
                if name not in ('baseline','candidate') or attempt or 'out of memory' not in log.read_text().lower():
                    raise RuntimeError('final-path engineering failed; inspect '+str(log))
        assert read(root/'live_graph_consistency.json')['total_condition_frames']==52
        rr=read(root/'paired/pipeline_state.json');assert rr['frames']==rr['exact_frames']==26
        write(root/'scope.json',dict(status='FINAL_SERIALIZATION_MODEL_GRAPH_GT_ENGINEERING_COMPLETE',run_id=logger.run_id,
            frames_per_condition=26,exact_frames=26,live_and_posthoc_Graph_five_tensors_equal=True,
            Final_registered_seeds_opened=False,GT_forward_input=False,decision_metrics_are_not_scientific=True,
            files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and 'experiments' not in p.parts},
            whole_goal_complete=False))
        write(root/'pipeline_state.json',dict(status='FINAL_SERIALIZATION_MODEL_GRAPH_GT_ENGINEERING_COMPLETE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','dependency','collector'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',default='BC_KEEP');p.add_argument('--profiled',action='store_true')
    a=p.parse_args()
    try:run(a.root,a.dependency,a.collector,a.candidate,a.profiled)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
