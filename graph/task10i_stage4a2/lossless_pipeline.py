"""Crash-safe Development P inference, same Graph evaluation and summaries.

Runs independently in tmux. No Internal/Final stage or candidate adoption occurs.
Each stage skip requires immutable source/command and full artifact SHA readback.
"""
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


def wait_keep(root,dependency):
    deadline=time.monotonic()+24*3600
    while True:
        statefile=dependency/'pipeline_state.json'
        if statefile.exists():
            state=read(statefile)
            if state['status']=='KEEP_CURRENT_HISTORICAL_CONTROL_COMPLETE':
                assert read(dependency/'historical/scope.json')['uncertain_output_policy']=='preserve_current'
                return
            if state['status']=='FAILED':raise RuntimeError('keep-control failed; inspect preserved source/output')
        live=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                command=p.read_bytes()
                if b'graph.task10i_stage4a2.keep_control_runner' in command and str(dependency).encode() in command:live.append(p.parent.name)
            except OSError:pass
        if not live:raise RuntimeError('keep-control coordinator missing without completion receipt')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_KEEP_CONTROL',pids=live,pid=os.getpid()))
        if time.monotonic()>deadline:raise TimeoutError('GPU serialization deadline')
        time.sleep(10)


def run(root,baseline,partition,reference,dependency):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        files={str(p.relative_to(source)):sha(p) for directory in ('graph','visual/runtime','utils')
            for p in (source/directory).rglob('*.py')}
        runtime=Path('/workspace/task10g_f1_abc_20261031/full')
        paths=read(runtime/'runtime_paths.json')
        model=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
        v2=Path('/workspace/task10i_stage4a2_models/dinov2-small')
        model_files={str(Path(paths['dino'])/p):sha(Path(paths['dino'])/p) for p in model['dino_files']}
        model_files[paths['sam']]=sha(paths['sam'])
        model_files.update({str(v2/p):sha(v2/p) for p in read(v2/'download_identity.json')['files']})
        identity=dict(files=files,criteria_sha256=sha(Path(__file__).with_name('adoption_criteria.json')),
            partition_sha256=sha(partition),baseline_expected_sha256=sha(baseline/'expected_counts.json'),
            reference_completion_sha256=sha(reference/'pipeline_state.json'),
            baseline=str(baseline),reference=str(reference),candidate='P',Internal_and_Final_access=False,
            model_files=model_files,graph_protocol_sha256=sha(source/'graph/plan_a_geometry/protocol.json'),
            schema_sha256=sha('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'),
            assets={str(p):sha(p) for p in (runtime/'inputs').rglob('*') if p.is_file()})
        assert identity['criteria_sha256']==read('/workspace/task10i_stage4a2_criteria_20261101_v1/run/registration_receipt.json')['criteria_sha256']
        frozen=root/'source_config_frozen.json'
        if frozen.exists():assert read(frozen)==identity,'source/config changed; new root required'
        else:write(frozen,identity)
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        wait_keep(root,dependency)
        python=[sys.executable,'-B','-u','-m']
        commands=[
            ('contracts',python+['unittest','graph.task10i_stage4a2.test_contracts','graph.task10i_stage4a.test_contracts','-v'],None),
            ('engineering',python+['graph.task10i_stage4a2.development_lossless',str(root/'engineering'),str(baseline),str(partition),'--engineering'],root/'engineering'),
            ('engineering_graph',python+['graph.task10i_stage4a2.paired_evaluation',str(root/'engineering_graph'),str(root/'engineering'),str(reference),
                str(root/'engineering/expected_counts.json'),'--engineering-only','--candidate','P'],root/'engineering_graph'),
            ('development',python+['graph.task10i_stage4a2.development_lossless',str(root/'development'),str(baseline),str(partition)],root/'development'),
            ('paired_graph',python+['graph.task10i_stage4a2.paired_evaluation',str(root/'paired_graph'),str(root/'development'),str(reference),
                str(root/'development/expected_counts.json'),'--candidate','P'],root/'paired_graph'),
            ('summary',python+['graph.task10i_stage4a2.aggregate_evaluation',str(root/'summary'),str(root/'paired_graph'),
                str(root/'development/expected_counts.json'),'--candidate','P'],root/'summary')]
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_Development_P_pipeline',seed=42,
                root=root/'coordinator_experiments',job_type='evaluation',config=identity) as logger:
            for name,command,child in commands:
                receipt=root/'stages'/(name+'.json')
                if receipt.exists():
                    r=read(receipt)
                    assert r['signature']==signature and r['command']==command
                    assert all(sha(root/p)==h for p,h in r['files'].items())
                    continue
                for attempt in range(2):
                    write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,attempt=attempt,pid=os.getpid(),
                        command=command,run_id=logger.run_id))
                    logfile=root/(name+f'_attempt{attempt}.log')
                    with logfile.open('w') as stream:
                        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode==0:break
                    text=logfile.read_text()
                    if name not in ('engineering','development') or attempt==1 or 'out of memory' not in text.lower():
                        raise RuntimeError(f'{name} failed; log preserved in {logfile}')
                if name in ('engineering','development'):
                    assert read(child/'pipeline_state.json')['status']=='DEVELOPMENT_P_INFERENCE_COMPLETE'
                elif name=='engineering_graph':
                    r=read(child/'engineering_barrier.json');assert r['receipt']['frames']==r['receipt']['exact_frames']==3
                elif name=='paired_graph':
                    assert read(child/'pipeline_state.json')['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
                elif name=='summary':
                    assert read(child/'scope.json')['conditions']==['B0','P']
                artifacts=[logfile] if child is None else [p for p in child.rglob('*') if p.is_file()]
                write(receipt,dict(status='PASS',signature=signature,command=command,run_id=logger.run_id,
                    files={str(p.relative_to(root)):sha(p) for p in artifacts}))
            summary=read(root/'summary/scope.json')
            assert summary['parents']==36 and summary['episodes']==88
            write(root/'pipeline_state.json',dict(status='DEVELOPMENT_P_FORWARD_GRAPH_SUMMARY_COMPLETE',
                run_id=logger.run_id,frames=read(root/'development/expected_counts.json')['frames'],
                Internal_and_Final_opened=False,candidate_selected=False,whole_goal_complete=False))
            logger.link_output(root/'summary/scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','baseline','partition','reference','dependency'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.baseline,a.partition,a.reference,a.dependency)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,
            message=str(exc),partial_preserved=True,whole_goal_complete=False))
        raise
