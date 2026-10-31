"""tmux-owned state machine. No agent/callback dependency or manual stage switch."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from graph.plan_a_bc.common import read,write,sha,digest,now

STAGES=['PREFLIGHT','DATA_VERIFICATION','MODEL_CONTRACT_TEST','LAMBDA_CALIBRATION','PRELIMINARY_PROTOCOL_FREEZE',
    'TRAINING_SMOKE','RUNTIME_ROLLOUT_SMOKE','CONCURRENCY_BENCHMARK','TRAINING_CONFIG_FREEZE',
    'PARALLEL_TRAIN_A_B_C','CHECKPOINT_VERIFICATION','DEVELOPMENT_CLOSED_LOOP','MODEL_SELECTION_FREEZE',
    'FINAL_CLOSED_LOOP','GEOMETRY_EVALUATION','ACTION_CORRECTION_EVALUATION','ACTION_AND_FAILURE_ANALYSIS',
    'REPORT_GENERATION','ARTIFACT_HASH_VERIFICATION']


def verify_stage(root,row):
    if row.get('status')!='COMPLETED':raise ValueError('stage not completed '+row['stage_id'])
    if row['exit_code']!=0 or not row['outputs']:raise ValueError('stage missing output predicate')
    for p,h in row['outputs'].items():
        if not (root/p).is_file() or sha(root/p)!=h:raise ValueError('stage artifact changed '+p)
    return True


def verify_all_stages(root):
    state=read(root/'pipeline_state.json')
    if state['status']!='COMPLETED' or set(state['stages'])!=set(STAGES):raise ValueError('pipeline incomplete')
    for row in state['stages'].values():verify_stage(root,row)
    return True


def run(root,stop_after=None):
    root=root.resolve();cfg=read(root/'config.json')
    if not os.getenv('TMUX'):raise RuntimeError('orchestrator must run inside tmux')
    lock=(root/'orchestrator.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another orchestrator owns this task')
    state=read(root/'pipeline_state.json') if (root/'pipeline_state.json').exists() else dict(status='PENDING',stages={})
    if state['status']=='COMPLETED':verify_all_stages(root);return
    state.update(status='RUNNING',pid=os.getpid(),started_at=state.get('started_at',now()),config_sha256=sha(root/'config.json'))
    def persist():
        write(root/'pipeline_state.json',state);write(root/'stage_manifest.json',state['stages'])
    persist()
    child=None
    def interrupted(signum,frame):raise InterruptedError('orchestrator signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for index,stage in enumerate(STAGES):
            input_hash=digest(dict(config=sha(root/'config.json'),upstream={k:v.get('output_manifest_hash') for k,v in state['stages'].items() if k in STAGES[:index]}))
            previous=state['stages'].get(stage)
            if previous and previous['status']=='COMPLETED':
                verify_stage(root,previous)
                if previous['input_manifest_hash']!=input_hash:raise ValueError('stage inputs changed '+stage)
                print('SKIP verified',stage,flush=True);continue
            row=dict(stage_id=stage,status='RUNNING',input_manifest_hash=input_hash,output_manifest_hash=None,
                started_at=now(),finished_at=None,exit_code=None,actual_steps=0,actual_episodes=0,
                error=None,traceback=None,resumable=True,outputs={})
            state['stages'][stage]=row;state['current_stage']=stage;persist()
            logs=root/'logs';logs.mkdir(exist_ok=True);path=logs/(stage+'_'+str(time.time_ns())+'.log')
            print('STAGE',stage,'START',now(),flush=True)
            with path.open('w') as log:
                child=subprocess.Popen([sys.executable,'-u','-B','-m','graph.graph_recurrent10g.stages',
                    '--root',str(root),'--stage',stage],stdout=log,stderr=subprocess.STDOUT)
                row['worker_pid']=child.pid;persist();code=child.wait();child=None
            row.update(exit_code=code,status='VERIFYING');persist()
            if code!=0:
                row['error']=path.read_text()[-12000:]
                raise RuntimeError('stage failed '+stage+'; '+str(path))
            result=read(root/'stage_results'/(stage+'.json'))
            if result.get('status')!='PASS' or not result.get('files'):raise ValueError('stage completion evidence missing')
            outputs={p:sha(root/p) for p in result['files']}
            row.update(outputs=outputs,output_manifest_hash=digest(outputs),status='COMPLETED',finished_at=now(),
                actual_steps=result.get('steps',3*cfg['training']['updates'] if stage=='PARALLEL_TRAIN_A_B_C' else 0),
                actual_episodes=result.get('episodes',0))
            verify_stage(root,row);persist();print('STAGE',stage,'COMPLETE',now(),flush=True)
            if stop_after==stage:
                state.update(status='ENGINEERING_INTERRUPTED',intentional_restart_test=True);persist();return
        state.update(status='COMPLETED',completed_at=now(),current_stage=None);persist();verify_all_stages(root)
        write(root/'COMPLETED',dict(at=now(),completion_audit_sha256=sha(root/'completion_audit.json')))
        print('PIPELINE COMPLETED',root,flush=True)
    except BaseException as exc:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),blocked_at=now())
        if state.get('current_stage'):
            row=state['stages'][state['current_stage']];row.update(status='BLOCKED',traceback=state['traceback'],finished_at=now())
        persist();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stop-after');a=p.parse_args();run(a.root,a.stop_after)
