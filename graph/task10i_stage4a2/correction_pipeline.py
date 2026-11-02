"""Serialized full Development C_KEEP/BC_KEEP, exact paired Graphs and reports."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,baseline,partition,reference,dependency):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        rt=Path('/workspace/task10g_f1_abc_20261031/full');paths=read(rt/'runtime_paths.json')
        model=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
        v2=Path('/workspace/task10i_stage4a2_models/dinov2-small')
        identity=dict(source={str(p.relative_to(source)):sha(p) for folder in ('graph','visual/runtime','utils')
            for p in (source/folder).rglob('*.py')},partition_sha256=sha(partition),
            baseline_config_sha256=sha(baseline/'source_config_frozen.json'),
            reference_completion_sha256=sha(reference/'pipeline_state.json'),
            criteria_sha256=sha(Path(__file__).with_name('adoption_criteria.json')),
            model_files={str(Path(paths['dino'])/p):sha(Path(paths['dino'])/p) for p in model['dino_files']},
            sam_sha256=sha(paths['sam']),dinov2_files={p:sha(v2/p) for p in read(v2/'download_identity.json')['files']},
            graph_protocol_sha256=sha(source/'graph/plan_a_geometry/protocol.json'),
            schema_sha256=sha('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'),
            assets={str(p):sha(p) for p in (rt/'inputs').rglob('*') if p.is_file()},
            conditions=['C_KEEP','BC_KEEP'],Internal_or_Final_access=False)
        frozen=root/'source_config_frozen.json'
        if frozen.exists():assert read(frozen)==identity,'changed source/config requires new root'
        else:write(frozen,identity)
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        deadline=time.monotonic()+24*3600
        while True:
            file=dependency/'pipeline_state.json'
            if file.exists():
                state=read(file)
                if state['status']=='BACKGROUND_TILE_MATCHING_AUDIT_COMPLETE':break
                if state['status']=='FAILED':raise RuntimeError('background diagnostic dependency failed')
            live=[]
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    command=path.read_bytes()
                    if b'graph.task10i_stage4a2.background_matching_audit' in command and str(dependency).encode() in command:live.append(path.parent.name)
                except OSError:pass
            if not live:raise RuntimeError('background diagnostic handle missing without terminal completion')
            write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_BACKGROUND_AUDIT',pids=live,pid=os.getpid()))
            if time.monotonic()>deadline:raise TimeoutError('GPU serialization deadline')
            time.sleep(10)
        python=[sys.executable,'-B','-u','-m']
        commands=[('contracts',python+['unittest','graph.task10i_stage4a2.test_contracts','graph.task10i_stage4a.test_contracts','-v'],None,None,None),
            ('engineering',python+['graph.task10i_stage4a2.development_correction',str(root/'engineering'),str(baseline),str(partition),'--engineering'],
                root/'engineering','pipeline_state.json','DEVELOPMENT_C_BC_KEEP_INFERENCE_COMPLETE')]
        for condition in ('C_KEEP','BC_KEEP'):
            child=root/('engineering_graph_'+condition)
            commands.append(('engineering_graph_'+condition,python+['graph.task10i_stage4a2.paired_evaluation',str(child),str(root/'engineering'),
                str(reference),str(root/'engineering/expected_counts.json'),'--engineering-only','--candidate',condition],child,'engineering_barrier.json','PASS'))
        commands.append(('development',python+['graph.task10i_stage4a2.development_correction',str(root/'development'),str(baseline),str(partition)],
            root/'development','pipeline_state.json','DEVELOPMENT_C_BC_KEEP_INFERENCE_COMPLETE'))
        for condition in ('C_KEEP','BC_KEEP'):
            child=root/('paired_graph_'+condition)
            commands.append(('paired_graph_'+condition,python+['graph.task10i_stage4a2.paired_evaluation',str(child),str(root/'development'),str(reference),
                str(root/'development/expected_counts.json'),'--candidate',condition,'--resume-existing'],child,'pipeline_state.json','PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'))
            summary=root/('summary_'+condition)
            commands.append(('summary_'+condition,python+['graph.task10i_stage4a2.aggregate_evaluation',str(summary),str(child),
                str(root/'development/expected_counts.json'),'--candidate',condition],summary,'scope.json','NEW_DEVELOPMENT_PAIRED_SUMMARY_COMPLETE'))
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_Development_correction_pipeline',seed=42,
                root=root/'coordinator_experiments',job_type='evaluation',config=identity) as logger:
            for name,command,child,terminal,status in commands:
                receipt=root/'stages'/(name+'.json')
                if receipt.exists():
                    r=read(receipt);assert r['signature']==signature and r['command']==command
                    assert all(sha(root/p)==h for p,h in r['files'].items());continue
                complete=bool(child and (child/terminal).exists() and read(child/terminal)['status']==status)
                if complete:
                    # A crash after the child commit can precede coordinator commit.
                    for file in child.rglob('receipt.json'):
                        r=read(file)
                        if 'files' in r:assert all(sha(file.parent/p)==h for p,h in r['files'].items())
                else:
                    if child and child.exists() and name.startswith(('summary_','engineering_graph_')):
                        child.rename(child.with_name(child.name+'.partial_'+str(time.time_ns())))
                    for attempt in range(2):
                        write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,attempt=attempt,pid=os.getpid(),command=command,run_id=logger.run_id))
                        logfile=root/f'{name}_attempt{attempt}.log'
                        with logfile.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                        if result.returncode==0:break
                        if name not in ('engineering','development') or attempt==1 or 'out of memory' not in logfile.read_text().lower():
                            raise RuntimeError(f'{name} failed: preserved {logfile}')
                    if child:assert read(child/terminal)['status']==status
                artifacts=[p for p in child.rglob('*') if p.is_file()] if child else [root/(name+'_attempt0.log')]
                write(receipt,dict(status='PASS',signature=signature,command=command,run_id=logger.run_id,
                    files={str(p.relative_to(root)):sha(p) for p in artifacts},recovered_child_commit=complete))
            assert all(read(root/('summary_'+condition)/'scope.json')['parents']==36 for condition in ('C_KEEP','BC_KEEP'))
            write(root/'pipeline_state.json',dict(status='DEVELOPMENT_C_BC_KEEP_FORWARD_GRAPH_SUMMARY_COMPLETE',
                run_id=logger.run_id,candidate_selected=False,Internal_or_Final_opened=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','baseline','partition','reference','dependency'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.baseline,a.partition,a.reference,a.dependency)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,
            message=str(exc),partial_preserved=True,whole_goal_complete=False))
        raise
