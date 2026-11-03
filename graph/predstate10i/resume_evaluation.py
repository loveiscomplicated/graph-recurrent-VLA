"""Standalone operational continuation. Copy OUTSIDE the frozen source tree.

Each real episode owns a fresh Python/CUDA/Vulkan/vision process. The original
policy, rollout engine, seeds, checkpoints and selection code are imported as-is.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import traceback


def bind(root):
    sys.path.insert(0,str(root/'source'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def verify_episode(root,folder):
    row=read(folder/'episode.json')
    assert row['status']=='COMPLETE'
    assert sha(folder/'trace.json')==row['trace_sha256']
    assert sha(folder/'rollout.mp4')==row['video_sha256']
    assert len(read(folder/'trace.json'))==row['actions']<=250
    return row


def episode_worker(root,ops,spec):
    import torch
    from utils.experiment_logger import ExperimentLogger
    from graph.predstate10i.evaluate import load_s,models
    from graph.predstate10i.rollout import episode
    from graph.graph_recurrent10g.eval_runner import vision
    from graph.plan_a_bc.model import normalizer
    from graph.integration_baseline.bc08_codec import ActionCodec
    from graph.plan_a_bc.common import tensor_identity
    torch.set_num_threads(1)
    cfg=read(root/'config.json');smoke=spec['kind']=='smoke'
    if spec['kind'] in ['development','smoke']:
        cp=root/'train_S'/f"step{spec['step']:06d}.pt"
        mm={'S':load_s(root,cp)}
        stage=f"development/step{spec['step']:06d}" if not smoke else 'resume_smoke/'+spec['name']
        label='S0';engine='B0';observers=None
    else:
        assert read(root/'development_selection.json')['status']=='FROZEN'
        mm=models(root,root/'best.pt');stage='final';label=spec['condition']
        engine='E' if label=='E' else {'0':'B0','1':'H1','2':'H2'}[label[-1]]
        observers={k:mm[k] for k in ['R','J']} if label=='E' else None
    m=mm['S' if label=='E' else label[0]]
    before={k:tensor_identity(v.state_dict()) for k,v in mm.items()}
    norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'));detector,tracker=vision(root)
    with ExperimentLogger(team='graph',model='Task10I_resume_'+spec['kind'],seed=42,config=cfg,
                         root=root/'experiments',job_type='evaluate') as log:
        log.record(resume_operation=str(ops),episode_spec=spec,process_isolation='one episode per child',
                   scientific_source_identity_sha256=sha(root/'execution_source_identity.json'))
        row=episode(root,root,m,norm,codec,detector,tracker,spec['seed'],engine,stage,label,
                    horizon=2 if smoke else 250,observers=observers)
        assert before=={k:tensor_identity(v.state_dict()) for k,v in mm.items()}
        verify_episode(root,root/row['folder'])
        log.link_output(root/row['folder']/'episode.json',kind='metrics')
        write(ops/'receipts'/(spec['name']+'.json'),dict(status='PASS',pid=os.getpid(),run_id=log.run_id,
            spec=spec,row=row,weights_unchanged=True,core_source_sha256=sha(root/'execution_source_identity.json')))


def aggregate(root,mode):
    from graph.graph_recurrent10g.eval_runner import csv_write
    from graph.predstate10i.evaluate import paired_verify
    from utils.experiment_logger import ExperimentLogger
    cfg=read(root/'config.json');rows=[]
    if mode=='development':
        for step in cfg['training']['checkpoint_steps']:
            cp=root/'train_S'/f'step{step:06d}.pt'
            for seed in cfg['evaluation']['dev_seeds']:
                row=verify_episode(root,root/'rollouts/development'/f'step{step:06d}'/f'seed{seed}'/'S0')
                row.update(step=step,checkpoint_sha256=sha(cp));rows.append(row)
        assert len(rows)==24
        path=root/'development_rollouts.csv'
        assert not path.exists()
        csv_write(path,rows)
    else:
        manifest=read(root/'final_seed_manifest.json')
        for seed in manifest['parents']:
            for label in manifest['conditions']:
                rows.append(verify_episode(root,root/'rollouts/final'/f'seed{seed}'/label))
        assert len(rows)==120
        csv_write(root/'final_functional_success.csv',[r for r in rows if r['condition'] in ['E','S0','R0','J0']])
        csv_write(root/'handoff_success.csv',[r for r in rows if r['condition'][-1:] in ['1','2']])
        paired_verify(root,rows,'final');path=root/'final_functional_success.csv'
    with ExperimentLogger(team='graph',model='Task10I_resume_aggregate_'+mode,seed=42,
                         config=cfg,root=root/'experiments',job_type='evaluate') as log:
        log.record(completed_episodes=len(rows),existing_episode_ownership_preserved=True)
        log.link_output(path,kind='metrics')


def prepare(root,ops):
    assert read(root/'pipeline_status.json')['status']=='FAILED'
    assert not (root/'development_selection.json').exists() and not (root/'rollouts/final').exists()
    assert read(root/'train_S/result.json')['updates']==11487
    assert read(root/'preflight_verification.json')['status']=='PASS'
    for name,h in read(root/'execution_source_identity.json')['hashes'].items():assert sha(root/'source'/name)==h,name
    for name,h in read(root/'source_checkpoint_audit.json')['protected'].items():assert sha(name)==h,name
    protected={}
    for p in list((root/'train_S').rglob('*'))+[root/'last.pt',root/'train_metrics.csv',root/'validation_metrics.csv']:
        if p.is_file():protected[str(p.relative_to(root))]=sha(p)
    for name in ['config.json','train_config.json','sampling_schedule.json','action_codec.json','normalizer.json',
                 'state38_normalizer.json','final_seed_manifest.json','comparison_policy_manifest.json',
                 'state38_feature_cache_manifest.json','execution_source_identity.json']:
        protected[name]=sha(root/name)
    cfg=read(root/'config.json');missing=[];reused=[]
    for step in cfg['training']['checkpoint_steps']:
        cp=root/'train_S'/f'step{step:06d}.pt';assert sha(cp)==read(cp.with_suffix('.json'))['sha256']
        for seed in cfg['evaluation']['dev_seeds']:
            folder=root/'rollouts/development'/f'step{step:06d}'/f'seed{seed}'/'S0'
            if (folder/'episode.json').exists():
                verify_episode(root,folder);reused.append(dict(step=step,seed=seed))
                for p in folder.rglob('*'):
                    if p.is_file():protected[str(p.relative_to(root))]=sha(p)
            else:missing.append(dict(kind='development',step=step,seed=seed,name=f'dev_{step}_{seed}'))
    assert len(reused)==23 and missing==[dict(kind='development',step=11487,seed=30042,name='dev_11487_30042')]
    final=[];manifest=read(root/'final_seed_manifest.json')
    for labels in [['E','S0','R0','J0'],['S1','S2','R1','R2','J1','J2']]:
        for seed in manifest['parents']:
            for label in labels:final.append(dict(kind='final',condition=label,seed=seed,name=f'final_{seed}_{label}'))
    assert len(final)==120
    write(ops/'resume_plan.json',dict(status='PREPARED',root=str(root),reused_development=reused,pending_development=missing,
        final=final,protected=protected,source_unchanged=True,training_rerun=False,automatic_retry=False,
        policy='one fresh child per episode; original episode/model/selection functions unchanged',runner_sha256=sha(Path(__file__))))


def fixture(ops):
    import tempfile
    with tempfile.TemporaryDirectory(dir=ops) as td:
        p=Path(td)/'lock';owner=p.open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cmd=[sys.executable,'-c','import fcntl,sys; f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',str(p)]
        assert subprocess.run(cmd,capture_output=True).returncode!=0
        assert subprocess.run([sys.executable,'-c','raise SystemExit(7)']).returncode==7
        # A completed result with a changed media/trace hash is rejected.
        folder=Path(td)/'episode';folder.mkdir();(folder/'trace.json').write_text('[{}]');(folder/'rollout.mp4').write_bytes(b'fixture')
        write(folder/'episode.json',dict(status='COMPLETE',actions=1,trace_sha256=sha(folder/'trace.json'),video_sha256=sha(folder/'rollout.mp4')))
        verify_episode(Path(td),folder);(folder/'trace.json').write_text('[{},{}]')
        try:verify_episode(Path(td),folder)
        except AssertionError:pass
        else:raise AssertionError('corrupt result accepted')
    write(ops/'fixture_verification.json',dict(status='PASS',nonzero_child_exit_checked=True,duplicate_lock_rejected=True,
        corrupted_completed_result_rejected=True,automatic_retry=False))


def run(root,ops):
    from graph.plan_a_bc.common import now
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(ops/'resume_plan.json')
    assert plan['runner_sha256']==sha(Path(__file__))
    assert read(ops/'fixture_verification.json')['status']=='PASS'
    assert read(ops/'fresh_process_equivalence.json')['status']=='PASS'
    assert os.getenv('CONTAINER_ID')=='58752018'
    assert not (root/'PIPELINE_COMPLETE').exists()
    for name,h in plan['protected'].items():assert sha(root/name)==h,name
    assert (root/'FAILED').exists()
    history=ops/'prior_attempt';history.mkdir(exist_ok=False)
    # Preserve canonical failure/status before resuming; no old logs are overwritten.
    import shutil
    for name in ['pipeline_status.json','heartbeat.json','FAILED']:
        if (root/name).exists():shutil.copy2(root/name,history/name)
    (root/'FAILED').rename(history/'FAILED_marker')
    stages=[];child=None;stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):write(root/'heartbeat.json',dict(at=now(),pid=os.getpid(),child_pid=child.pid if child else None,operation=str(ops)))
    def interrupted(signum,frame):raise InterruptedError('resume signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def status(state):write(root/'pipeline_status.json',dict(status=state,pid=os.getpid(),operation=str(ops),
        inherited_training='COMPLETE:11487; never rerun',reused_development=23,stages=stages))
    def execute(name,command):
        nonlocal child
        row=dict(stage=name,status='RUNNING',started_at=now(),command=command);stages.append(row);status('RUNNING')
        logfile=ops/'logs'/(name+'.log');logfile.parent.mkdir(exist_ok=True)
        with logfile.open('xb') as log:
            print('RESUME_STAGE_START',name,now(),flush=True)
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=root/'source')
            row['child_pid']=child.pid;status('RUNNING');code=child.wait();child=None
        row.update(status='PASS' if code==0 else 'FAILED',exit_code=code,completed_at=now(),log=str(logfile),log_sha256=sha(logfile));status('RUNNING' if code==0 else 'FAILED')
        if code:raise subprocess.CalledProcessError(code,command)
        print('RESUME_STAGE_COMPLETE',name,now(),flush=True)
    def self_cmd(*args):return [sys.executable,'-B','-u',str(Path(__file__).resolve()),'--root',str(root),'--ops',str(ops),*args]
    def module_cmd(module,stage):return [sys.executable,'-B','-u','-m','graph.predstate10i.'+module,'--root',str(root),'--stage',stage]
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    write(root/'heartbeat.json',dict(at=now(),pid=os.getpid(),operation=str(ops)));status('RUNNING')
    try:
        for spec in plan['pending_development']:
            path=ops/'jobs'/(spec['name']+'.json');write(path,spec)
            execute(spec['name'],self_cmd('--job',str(path)))
        execute('DEVELOPMENT_AGGREGATION',self_cmd('--aggregate','development'))
        execute('CHECKPOINT_SELECTION_FREEZE',module_cmd('evaluate','select'))
        for spec in plan['final']:
            path=ops/'jobs'/(spec['name']+'.json');write(path,spec)
            execute(spec['name'],self_cmd('--job',str(path)))
        execute('FINAL_AGGREGATION_AND_PAIRED_VERIFICATION',self_cmd('--aggregate','final'))
        for name,h in plan['protected'].items():assert sha(root/name)==h,name
        execute('REPORT_ANALYSIS',module_cmd('report','analyze'))
        execute('ARTIFACT_VERIFICATION',module_cmd('report','verify'))
        assert read(root/'verification.json')['status']==read(root/'completion_audit.json')['status']=='PASS'
        status('PIPELINE_COMPLETE')
        names=['pipeline_status.json','verification.json','completion_audit.json','artifact_index.json',
            'checkpoint_manifest.json','best.pt','last.pt','10i_predicted_state38_graph_recurrent_bc.md','report.html']
        hashes={n:sha(root/n) for n in names}
        write(root/'pipeline_receipt.json',dict(status='PASS',files=hashes,resume_plan_sha256=sha(ops/'resume_plan.json'),
            runner_sha256=sha(Path(__file__)),readback=True))
        write(root/'PIPELINE_COMPLETE',dict(completed_at=now(),verification_sha256=sha(root/'verification.json'),
            completion_audit_sha256=sha(root/'completion_audit.json'),receipt_sha256=sha(root/'pipeline_receipt.json')))
        print('PIPELINE_COMPLETE',now(),flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        record=dict(status='FAILED',at=now(),operation=str(ops),error=repr(exc),traceback=traceback.format_exc(),stages=stages)
        write(root/'pipeline_status.json',record);write(root/'FAILED',record);raise
    finally:stop.set();thread.join(timeout=15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ops',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--fixture',action='store_true');p.add_argument('--job',type=Path)
    p.add_argument('--aggregate',choices=['development','final']);a=p.parse_args();bind(a.root)
    if a.prepare:prepare(a.root,a.ops)
    elif a.fixture:fixture(a.ops)
    elif a.job:episode_worker(a.root,a.ops,read(a.job))
    elif a.aggregate:aggregate(a.root,a.aggregate)
    else:run(a.root,a.ops)
