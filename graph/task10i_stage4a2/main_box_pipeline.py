"""Development-only Main-Box scope study; preserves failed full-scope results."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .lazy_feature_control import wait_for
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,baseline,partition,reference,dependency,candidate='BC_MAIN_BOX',handoff_engineering=None):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    assert candidate in ('BC_MAIN_BOX','BC_WB_HANDOFF')
    if candidate=='BC_WB_HANDOFF':
        assert handoff_engineering is not None
        proof=read(handoff_engineering/'scope.json')
        assert proof['causal_recovery_demonstrated'] is True
        for name in ('tri_model_integration.py','reanchor_state_machine.py','dinov2_runtime.py','development_correction.py','model_trace.py'):
            expected_sha=next(h for p,h in proof['source'].items() if Path(p).name==name)
            assert sha(Path(__file__).with_name(name))==expected_sha,'core code changed after actual handoff engineering'
    slots=[['base_camera',1]] if candidate=='BC_MAIN_BOX' else [['hand_camera',0],['base_camera',1]]
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        criteria=Path('/workspace/task10i_stage4a2_criteria_20261101_v1/run/adoption_criteria_frozen.json')
        identity=dict(files={str(p):sha(p) for folder in ('graph','visual/runtime','utils') for p in (source/folder).rglob('*.py')},
            baseline=str(baseline),partition_sha256=sha(partition),criteria_sha256=sha(criteria),
            reference=str(reference),candidate=candidate,correction_slots=slots,
            features='same first supported reference and .6/.05 matching; skip unused comparison embeddings',
            geometry_cache=True,Internal_and_Final_access=False,
            handoff_engineering_sha256=sha(handoff_engineering/'scope.json') if handoff_engineering else None,
            late_baseline_handoff=candidate=='BC_WB_HANDOFF',
            motivation='full C/BC damages100 healthy Wrist frames; Main Box damage0; separate limited-scope Development hypothesis, no retrospective repair of full-scope results')
        frozen=root/'source_config_frozen.json'
        if frozen.exists():assert read(frozen)==identity
        else:write(frozen,identity)
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        wait_for(root,dependency/'pipeline_state.json','FINAL_SERIALIZATION_MODEL_GRAPH_GT_ENGINEERING_COMPLETE',
            'graph.task10i_stage4a2.final_path_engineering',dependency)
        python=[sys.executable,'-B','-u','-m']
        commands=[('contracts',python+['unittest','graph.task10i_stage4a2.test_contracts','graph.task10i_stage4a2.test_registered_evaluation',
            'graph.task10i_stage4a2.test_dino_attribution','graph.task10i_stage4a2.test_evaluation_inputs','graph.task10i_stage4a2.test_temporal_metrics',
            'graph.task10i_stage4a.test_contracts','graph.task10i_stage4a.test_perframe_cache','-v'],None),
            ('engineering',python+['graph.task10i_stage4a2.development_correction',str(root/'engineering'),str(baseline),str(partition),
                '--engineering','--conditions',candidate,'--geometry-cache'],root/'engineering'),
            ('engineering_graph',python+['graph.task10i_stage4a2.paired_evaluation',str(root/'engineering_graph'),str(root/'engineering'),str(reference),
                str(root/'engineering/expected_counts.json'),'--engineering-only','--candidate',candidate],root/'engineering_graph'),
            ('development',python+['graph.task10i_stage4a2.development_correction',str(root/'development'),str(baseline),str(partition),
                '--conditions',candidate,'--geometry-cache'],root/'development'),
            ('paired_graph',python+['graph.task10i_stage4a2.paired_evaluation',str(root/'paired_graph'),str(root/'development'),str(reference),
                str(root/'development/expected_counts.json'),'--candidate',candidate,'--resume-existing'],root/'paired_graph'),
            ('summary',python+['graph.task10i_stage4a2.aggregate_evaluation',str(root/'summary'),str(root/'paired_graph'),
                str(root/'development/expected_counts.json'),'--candidate',candidate],root/'summary'),
            ('registered',python+['graph.task10i_stage4a2.registered_evaluation',str(root/'registered'),str(root/'paired_graph'),
                str(root/'development/expected_counts.json'),str(criteria),'--candidate',candidate,'--scope','development'],root/'registered')]
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_Development_scoped_'+candidate,seed=42,
                root=root/'coordinator_experiments',job_type='evaluation',config=identity) as logger:
            for name,command,child in commands:
                receipt=root/'stages'/(name+'.json')
                if receipt.exists():
                    r=read(receipt);assert r['signature']==signature and r['command']==command
                    assert all(sha(root/p)==h for p,h in r['files'].items());continue
                for attempt in range(2):
                    log=root/(name+f'_attempt{attempt}.log')
                    write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,attempt=attempt,command=command,pid=os.getpid()))
                    with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                    if not result.returncode:break
                    if name not in ('engineering','development') or attempt or 'out of memory' not in log.read_text().lower():raise RuntimeError('preserved failure: '+str(log))
                if name=='engineering_graph':assert read(child/'engineering_barrier.json')['receipt']['exact_frames']==3
                if name=='development':
                    assert read(child/'pipeline_state.json')['status']=='DEVELOPMENT_SCOPED_CORRECTION_INFERENCE_COMPLETE'
                    import numpy as np
                    for p in (child/'perception').glob('*/'+candidate+'.npz'):
                        with np.load(p) as a,np.load(p.with_name('B0.npz')) as b:
                            aa=np.unpackbits(a['mask_bits'],axis=1).reshape(-1,2,2,128,128)
                            bb=np.unpackbits(b['mask_bits'],axis=1).reshape(-1,2,2,128,128)
                            assert np.array_equal(aa[:,0,0],bb[:,0,0]) and np.array_equal(aa[:,1,1],bb[:,1,1]),'out-of-scope mask changed'
                            if candidate=='BC_MAIN_BOX':assert np.array_equal(aa[:,1,0],bb[:,1,0]),'Wrist Peg changed outside Main Box scope'
                artifacts=[p for p in child.rglob('*') if p.is_file()] if child else [log]
                write(receipt,dict(status='PASS',signature=signature,command=command,files={str(p.relative_to(root)):sha(p) for p in artifacts}))
            write(root/'pipeline_state.json',dict(status='DEVELOPMENT_MAIN_BOX_SCOPE_EVALUATION_COMPLETE' if candidate=='BC_MAIN_BOX' else 'DEVELOPMENT_WRIST_BOX_HANDOFF_EVALUATION_COMPLETE',run_id=logger.run_id,
                candidate_selected=False,Internal_and_Final_opened=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','baseline','partition','reference','dependency'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',default='BC_MAIN_BOX');p.add_argument('--handoff-engineering',type=Path)
    a=p.parse_args()
    try:run(a.root,a.baseline,a.partition,a.reference,a.dependency,a.candidate,a.handoff_engineering)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
