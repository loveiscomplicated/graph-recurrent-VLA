"""Detached Task08 tmux supervisor with frozen two or three job policy."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from graph.integration_baseline.direct_runtime import atomic, append, read, lock, Events, now
from graph.integration_baseline.bc08_train import load_plan


def tmux_environment():
    # An existing tmux server predates this shell. Forward only nonsecret runtime
    # settings explicitly; never put webhook/API-token values in process args.
    names=['WANDB_MODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME','CUBLAS_WORKSPACE_CONFIG',
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MPLCONFIGDIR','VK_ICD_FILENAMES','PYTHONUNBUFFERED']
    return [part for name in names if name in os.environ for part in ['-e',name+'='+os.environ[name]]]


def resources():
    import shutil
    import psutil
    def query(fields,scope='gpu'):
        out=subprocess.check_output(['nvidia-smi',f'--query-{scope}='+fields,'--format=csv,noheader,nounits'],text=True)
        return [line.strip().split(', ') for line in out.splitlines() if line.strip()]
    return dict(time=now(),gpu=query('index,name,memory.total,memory.free,memory.used'),
        processes=query('pid,process_name,used_memory','compute-apps'),
        cpu_count=os.cpu_count(),cpu_percent=psutil.cpu_percent(),load_average=list(os.getloadavg()),
        ram_available=psutil.virtual_memory().available,ram_total=psutil.virtual_memory().total,
        storage_free=shutil.disk_usage(Path.cwd()).free,storage_io=psutil.disk_io_counters()._asdict())


def gate(snapshot,*,peak=0,ram_peak=0,slots=1,vision=False):
    if len(snapshot['gpu'])!=1 or '3090' not in snapshot['gpu'][0][1]:
        raise RuntimeError('expected single RTX3090')
    free=int(snapshot['gpu'][0][3])*1024**2
    required=max((6 if vision else 2)*1024**3,slots*int(peak)*1.5+2*1024**3)
    if free<required or snapshot['ram_available']<max(8*1024**3,slots*ram_peak*1.5+2*1024**3) or snapshot['storage_free']<8*1024**3:
        raise RuntimeError('resource gate insufficient; preserve other jobs')
    return dict(required_free_vram=int(required),actual_free_vram=free,slots=slots,vision=vision)


def wait_for_gpu_idle(*,timeout=30):
    """Allow CUDA teardown after an owned worker exits; never disturb other PIDs."""
    deadline=time.monotonic()+timeout
    while True:
        snapshot=resources()
        if not snapshot['processes']:return snapshot
        if time.monotonic()>=deadline:
            raise RuntimeError('GPU process active after worker exit; no training/vision overlap: '+
                ','.join(str(p[0]) for p in snapshot['processes']))
        time.sleep(.5)


def probe_residency(snapshot,probes):
    """Validate triple residency across container/host PID namespaces."""
    probe_pids={str(p['pid']) for p in probes}
    resident_pids={str(p[0]) for p in snapshot['processes']}
    if (len(probe_pids)!=3 or len(resident_pids)!=3 or
            len(snapshot['processes'])!=3 or not all(p['paired'] for p in probes)):
        raise RuntimeError('three distinct resident GPU probe processes required')
    for pid in probe_pids:os.kill(int(pid),0)
    if resident_pids==probe_pids:return 'same_namespace'
    # Vast's NVML sees host PIDs; the container sees its own PIDs. NVML
    # labels host PIDs [Not Found] when /proc is scoped to the container.
    if all(p[1]=='[Not Found]' for p in snapshot['processes']):return 'host_container_pid_namespace'
    raise RuntimeError('GPU PIDs do not match the owned probe processes')


def shell_job_command(command,log,channel,exit_path):
    """Notify only after the GPU-owning Python process has actually exited."""
    return (shlex.join(command)+' >> '+shlex.quote(str(log))+' 2>&1\n'
        'task08_worker_exit_code=$?\n'
        'printf "%s\\n" "$task08_worker_exit_code" > '+shlex.quote(str(exit_path))+'\n'
        'tmux wait-for -S '+shlex.quote(channel)+'\n'
        'exit "$task08_worker_exit_code"\n')


def job_command(plan_path,job,log):
    cmd=[sys.executable,'-m','graph.integration_baseline.bc08_entry','graph.integration_baseline.bc08_pipeline','--plan',str(plan_path),'--job',json.dumps(job,separators=(',',':'))]
    root=Path(plan_path).parent
    return shell_job_command(cmd,log,'grvla08-'+root.name[:8]+'-'+job['key']+'-exit',root/'jobs'/f'{job["key"]}.exit')


def run_tmux_jobs(plan_path,jobs):
    """Waits only inside detached supervisor. Training is owned by separate tmux."""
    limit=read(plan_path).get('maximum_parallel_training_jobs',2)
    if limit not in (2,3) or len(jobs)>limit:raise ValueError('job count exceeds frozen parallel resource policy')
    root=Path(plan_path).parent;waiters=[]
    for job in jobs:
        key=job['key'];channel='grvla08-'+root.name[:8]+'-'+key
        if subprocess.run(['tmux','has-session','-t',channel],capture_output=True).returncode==0:
            raise RuntimeError('duplicate tmux job '+channel)
        # Start wait before session to avoid exit-before-wait races.
        waiter=subprocess.Popen(['tmux','wait-for',channel+'-exit'])
        log=root/'logs'/f'{key}.log';log.parent.mkdir(exist_ok=True)
        (root/'jobs').mkdir(exist_ok=True)
        atomic(root/'jobs'/f'{key}.json',dict(status='LAUNCHING',job=job,session=channel,log=str(log)))
        command=job_command(plan_path,job,log)
        subprocess.run(['tmux','new-session','-d','-s',channel,'-c',str(Path.cwd()),*tmux_environment(),command],check=True)
        waiters.append((job,waiter))
    for job,waiter in waiters:
        while True:
            try:
                if waiter.wait(timeout=30)!=0:raise RuntimeError('tmux event wait failed')
                break
            except subprocess.TimeoutExpired:
                # Supervisor-only liveness fallback for SIGKILL/tmux deletion,
                # which cannot execute the worker's finally/event notification.
                channel='grvla08-'+root.name[:8]+'-'+job['key']
                if subprocess.run(['tmux','has-session','-t',channel],capture_output=True).returncode:
                    record=read(root/'jobs'/f'{job["key"]}.json')
                    waiter.terminate();waiter.wait()
                    if record['status'] in ('RUNNING','LAUNCHING'):
                        record.update(status='ERROR',error_type='TMUX_SESSION_LOST',exit_code=137,
                            explanation='session disappeared without worker completion; last atomic checkpoint retained')
                        atomic(root/'jobs'/f'{job["key"]}.json',record)
                    break
    outcomes={}
    for spec in jobs:
        path=root/'jobs'/f'{spec["key"]}.json';record=read(path)
        exit_path=path.with_suffix('.exit')
        if exit_path.exists():record['process_exit_code']=int(exit_path.read_text().strip())
        if record['status'] in ('RUNNING','LAUNCHING'):
            record.update(status='ERROR',error_type='WORKER_EXIT_WITHOUT_COMPLETION',exit_code=record.get('process_exit_code',137))
            marker=root/(f'{spec["phase"]}-{spec["method"]}-'+('probe' if spec.get('probe') else 'train')+'.json')
            if marker.exists():record['directory']=read(marker)['directory']
        atomic(path,record);outcomes[spec['key']]=record
    return outcomes


def full_resident_gate(plan_path,state):
    """Real ABC resident probes before the explicitly requested three-job run."""
    import threading
    from uuid import uuid4
    root=Path(plan_path).parent;methods=['A','B','C']
    gate(resources(),peak=state['resource_gate']['single_peak'],ram_peak=state['resource_gate'].get('single_ram_peak',0),slots=3)
    previous={m:read(root/f'full-{m}-probe.json')['run_id'] if (root/f'full-{m}-probe.json').exists() else None for m in methods}
    observed={};failures=[];generation=uuid4().hex[:8]
    def release():
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                markers=[root/f'full-{m}-probe.json' for m in methods]
                if all(p.exists() for p in markers):
                    records=[read(p) for p in markers]
                    if all(r['run_id']!=previous[m] for m,r in zip(methods,records)):
                        dirs=[Path(r['directory']) for r in records]
                        if all((d/'probe.json').exists() for d in dirs):
                            probes=[read(d/'probe.json') for d in dirs]
                            snap=resources();observed.update(snapshot=snap,probes=probes)
                            gate(snap,slots=1)
                            observed['pid_mapping']=probe_residency(snap,probes)
                            return
                time.sleep(.2)
            raise TimeoutError('ABC resident probes timed out')
        except BaseException as exc:failures.append(type(exc).__name__)
        finally:(root/'probe_release').touch()
    (root/'probe_release').unlink(missing_ok=True)
    thread=threading.Thread(target=release);thread.start()
    jobs=[dict(key=f'full_probe_{m}_{generation}',kind='train',phase='full',method=m,probe=True,paired=True) for m in methods]
    outcomes=run_tmux_jobs(plan_path,jobs);thread.join()
    if failures or any(v['status']!='PROBE_COMPLETE' for v in outcomes.values()):
        atomic(root/'triple_resource_failure.json',dict(failures=failures,outcomes=outcomes,observed=observed))
        raise RuntimeError('ABC resident resource/contract probe failed; no Full training launched')
    proof=dict(status='PASS',requested_parallel_training_jobs=3,**observed)
    atomic(root/'triple_resource_gate.json',proof,immutable=True)
    state['resource_gate']['triple']=proof
    state['resource_gate']['single_peak']=max([state['resource_gate']['single_peak']]+[p['memory']['cuda_peak_reserved_bytes'] for p in observed['probes']])
    state['resource_gate']['single_ram_peak']=max([state['resource_gate'].get('single_ram_peak',0)]+[p['memory']['process_highwater_bytes'] for p in observed['probes']])
    atomic(root/'status.json',state)


def job(plan_path,spec):
    root=Path(plan_path).parent;key=spec['key'];status=dict(status='RUNNING',pid=os.getpid(),started_at=now(),job=spec)
    atomic(root/'jobs'/f'{key}.json',status)
    try:
        if spec['kind']=='train':
            from graph.integration_baseline.bc08_train import worker
            result=worker(plan_path,spec['phase'],spec['method'],resume=spec.get('resume'),probe=spec.get('probe',False),paired=spec.get('paired',False))
        else:
            from graph.integration_baseline.bc08_eval import evaluate
            result=evaluate(plan_path,spec['phase'],spec['method'],spec['group'],spec.get('receipt'),smoke=spec.get('smoke',False))
        status.update(status=result['status'],directory=result['directory'],result=result,exit_code=0)
    except BaseException as exc:
        from tools.discord_notifier import Sanitizer
        status.update(status='ERROR',exit_code=1,error_type=type(exc).__name__,
            oom=isinstance(exc,__import__('torch').cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower(),
            traceback=Sanitizer()(traceback.format_exc()))
        marker=root/(f'{spec["phase"]}-{spec["method"]}-'+('probe' if spec.get('probe') else 'train')+'.json')
        if marker.exists():status['directory']=read(marker)['directory']
    finally:
        status['ended_at']=now();atomic(root/'jobs'/f'{key}.json',status)
        # The tmux shell sends the event after this Python process exits. Sending
        # it here lets the supervisor race with CUDA context teardown.
    return status


def train_jobs(plan_path,phase,methods,state,events):
    plan=read(plan_path);root=Path(plan_path).parent
    methods=[m for m in methods if state['stages'].get(phase+'_'+m,{}).get('status')!='COMPLETE']
    if not methods:return
    before=resources();gate(before,peak=state['resource_gate']['single_peak'],ram_peak=state['resource_gate'].get('single_ram_peak',0),slots=len(methods))
    append(root/'resources.jsonl',dict(stage=phase,when='before_training',snapshot=before))
    jobs=[dict(key=phase+'_'+m,kind='train',phase=phase,method=m) for m in methods]
    # Explicit resume creates new Logger UUID, reading the last immutable cursor.
    for spec in jobs:
        old=state['stages'].get(spec['key'],{})
        if old.get('directory'):
            pointer=Path(old['directory'])/'last_valid.json'
            if not pointer.exists():raise RuntimeError('previous job lacks recoverable checkpoint')
            spec['resume']=str(pointer)
    if state['resource_gate'].get('sequential') and len(jobs)>1:
        outcomes={}
        for spec in jobs:outcomes.update(run_tmux_jobs(plan_path,[spec]))
    else:outcomes=run_tmux_jobs(plan_path,jobs)
    for key,value in outcomes.items():state['stages'][key]=value
    atomic(root/'status.json',state)
    failed=[(key,x) for key,x in outcomes.items() if x['status']!='COMPLETE']
    # Retry OOM only after BOTH owned jobs have exited; no other process is killed.
    if failed and all(x.get('oom') for _,x in failed):
        state['resource_gate']['sequential']=True
        events.emit('parallel_to_sequential',reason='measured training CUDA OOM; valid checkpoints retained',jobs=[k for k,_ in failed])
        for key,old in failed:
            pointer=Path(old['directory'])/'last_valid.json'
            if not pointer.exists():raise RuntimeError('OOM before valid checkpoint; explicit recovery required')
            spec=dict(key=key+'_sequential',kind='train',phase=phase,method=old['job']['method'],resume=str(pointer))
            gate(resources(),peak=state['resource_gate']['single_peak'],slots=1)
            value=run_tmux_jobs(plan_path,[spec])[spec['key']];state['stages'][key]=value
            atomic(root/'status.json',state)
            if value['status']!='COMPLETE':raise RuntimeError('sequential fallback failed')
    elif failed:raise RuntimeError('training job failed/stopped; no subsequent stages')
    from graph.integration_baseline.bc08_train import chunks
    expected=plan['overfit_updates'] if phase=='overfit' else sum(len(chunks(plan,'full',e)) for e in range(1,plan['config']['training']['max_epochs']+1))
    for method in methods:
        outcome=state['stages'][phase+'_'+method]
        if outcome['result']['updates']!=expected:raise RuntimeError('optimizer update budget mismatch')


def supervise(plan_path,*,resume=False):
    plan_path=Path(plan_path).resolve();root=plan_path.parent;plan=load_plan(plan_path)
    with lock(Path(__file__).resolve().parents[2]/'.task08-pipeline.lock'),lock(root/'pipeline.lock'):
        old=read(root/'status.json') if (root/'status.json').exists() else {}
        if old.get('status')=='COMPLETE':raise RuntimeError('pipeline already complete')
        if old.get('stages') and not resume:raise RuntimeError('explicit --resume required')
        state=dict(status='RUNNING',stage='preflight',stages=old.get('stages',{}),resource_gate=old.get('resource_gate',{}),pid=os.getpid(),started_at=now())
        events=Events(root)
        try:
            # Protect the reference model, dataset and graph files; baseline inputs
            # are rehashed before prepare, each worker, and after the pipeline.
            initial=resources();gate(initial,slots=2)
            if initial['processes']:raise RuntimeError('existing GPU process: training/rollout exclusivity cannot be certified')
            append(root/'resources.jsonl',dict(stage='preflight',snapshot=initial))
            subprocess.run([sys.executable,'-m','graph.integration_baseline.bc08_eval','--plan',str(plan_path),'--freeze'],check=True)
            if not state['resource_gate']:
                # First separate single-process actual B2xT32 checks.
                peak=0;single={}
                for method in ['A','C']:
                    spec=dict(key='single_probe_'+method,kind='train',phase='overfit',method=method,probe=True)
                    value=run_tmux_jobs(plan_path,[spec])[spec['key']]
                    if value['status']!='PROBE_COMPLETE':raise RuntimeError('single real-data CUDA probe failed')
                    data=read(Path(value['directory'])/'probe.json');single[method]=data
                    peak=max(peak,data['memory']['cuda_peak_reserved_bytes'])
                gate(resources(),peak=peak,slots=2)
                # Two resident forward/backward/optimizer probes share a barrier.
                # Plan stays immutable: barrier request is a worker launch flag via
                # an environment variable local to these probe processes.
                jobs=[dict(key='paired_probe_'+m,kind='train',phase='overfit',method=m,probe=True,paired=True) for m in ['A','C']]
                # Release coordinator is a short thread in the supervisor; it only
                # observes these owned initial probes, never training progress.
                import threading
                observed={};failure=[]
                def release():
                    try:
                        deadline=time.monotonic()+180
                        while time.monotonic()<deadline:
                            markers=[root/f'overfit-{m}-probe.json' for m in ['A','C']]
                            if all(p.exists() and (Path(read(p)['directory'])/'probe.json').exists() for p in markers):
                                dirs=[Path(read(p)['directory']) for p in markers]
                                if all(read(d/'probe.json')['paired'] for d in dirs):
                                    snap=resources();gate(snap,peak=0,slots=1)
                                    if len(snap['processes'])!=2:raise RuntimeError('paired probe must show two actual GPU processes')
                                    observed.update(snapshot=snap,probes=[read(d/'probe.json') for d in dirs]);break
                            time.sleep(.2)
                        else:raise TimeoutError('paired real CUDA probe timed out')
                    except BaseException as exc:failure.append(type(exc).__name__)
                    finally:(root/'probe_release').touch()
                (root/'probe_release').unlink(missing_ok=True)
                thread=threading.Thread(target=release);thread.start()
                pair=run_tmux_jobs(plan_path,jobs);thread.join()
                if failure or any(v['status']!='PROBE_COMPLETE' for v in pair.values()):raise RuntimeError('paired GPU resource gate failed: '+str(failure))
                state['resource_gate']=dict(single_peak=peak,single_ram_peak=max(x['memory']['process_highwater_bytes'] for x in single.values()),single=single,paired=observed,sequential=False)
                atomic(root/'resource_gate.json',state['resource_gate'],immutable=True)
            atomic(root/'status.json',state)
            # Small A/C concurrently. No validation/test checkpoint selection.
            state['stage']='small_overfit';atomic(root/'status.json',state)
            train_jobs(plan_path,'overfit',['A','C'],state,events)
            from graph.integration_baseline.direct_runtime import sha
            if len({sha(Path(state['stages']['overfit_'+m]['directory'])/'batch_identity.jsonl') for m in ['A','C']})!=1:
                raise RuntimeError('overfit episode/batch/sequence identity mismatch')
            def evaluation(phase,method,group,smoke=False):
                key=('smoke_' if smoke else '')+group+'_'+method
                if state['stages'].get(key,{}).get('status') in ('COMPLETE','SMOKE_COMPLETE'):return
                # Owned training processes have joined; any remaining GPU work
                # belongs to somebody else and blocks an autonomous rollout.
                r=wait_for_gpu_idle();gate(r,vision=True)
                append(root/'resources.jsonl',dict(stage=key,snapshot=r))
                receipt=None
                if method!='reference':
                    train=state['stages'][phase+'_'+method]
                    receipt=str(Path(train['directory'])/('best.json' if phase=='full' else 'last_valid.json'))
                spec=dict(key=key,kind='eval',phase=phase,method=method,group=group,receipt=receipt,smoke=smoke)
                value=run_tmux_jobs(plan_path,[spec])[key];state['stages'][key]=value;atomic(root/'status.json',state)
                if value['status'] not in ('COMPLETE','SMOKE_COMPLETE'):raise RuntimeError('evaluation failed; preserve artifacts, no repeat-seed tuning')
            state['stage']='overfit_closed_loop';atomic(root/'status.json',state)
            evaluation('overfit','A','train_seed',smoke=True)
            for m in ['A','C']:evaluation('overfit',m,'train_seed')
            # Contract smoke failure blocks Full; manipulation failure is retained
            # evidence and deliberately does not change settings or hide results.
            state['stage']='full_training';atomic(root/'status.json',state)
            if plan.get('maximum_parallel_training_jobs',2)==3:
                if not state['resource_gate'].get('triple'):full_resident_gate(plan_path,state)
                train_jobs(plan_path,'full',['A','B','C'],state,events)
            else:
                train_jobs(plan_path,'full',['A','B'],state,events)
                train_jobs(plan_path,'full',['C'],state,events)
            if len({sha(Path(state['stages']['full_'+m]['directory'])/'batch_identity.jsonl') for m in ['A','B','C']})!=1:
                raise RuntimeError('Full ABC episode/batch/sequence identity mismatch')
            state['stage']='full_closed_loop';atomic(root/'status.json',state)
            for group,methods in [('development',['A','B','C']),('historical7',['reference','A','B','C']),('final',['A','B','C'])]:
                for m in methods:evaluation('full',m,group)
            load_plan(plan_path) # source/reference/dataset manifest recheck
            if plan.get('protection_manifest'):
                protected=read(plan['protection_manifest'])
                changed=[p for p,expected in protected.items() if not Path(p).is_file() or sha(p)!=expected]
                atomic(root/'protection_verified.json',dict(status='PASS' if not changed else 'FAIL',files=len(protected),changed=changed))
                if changed:raise RuntimeError('original source/artifact protection mismatch')
            state.update(status='COMPLETE',stage='report',finished_at=now());atomic(root/'status.json',state)
            from graph.integration_baseline.bc08_report import render
            render(plan_path,state);events.emit('task08_complete',artifacts=str(root),html=str(root/'bc_training_comparison.html'))
        except BaseException as exc:
            from tools.discord_notifier import Sanitizer
            state.update(status='ERROR',error_type=type(exc).__name__,last_normal_stage=state.get('stage'))
            atomic(root/'status.json',state)
            atomic(root/'pipeline_error_bundle.json',dict(error_type=type(exc).__name__,traceback=Sanitizer()(traceback.format_exc()),
                state=state,resume_command=shlex.join([sys.executable,'-m','graph.integration_baseline.bc08_pipeline','--plan',str(plan_path),'--resume']),
                callback='NOT_CONFIGURED; notifications and local error bundle only'))
            from graph.integration_baseline.bc08_report import render
            render(plan_path,state);events.emit('task08_error',bundle=str(root/'pipeline_error_bundle.json'));raise
        finally:events.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--launch',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--job')
    a=p.parse_args()
    if a.job:
        result=job(a.plan,json.loads(a.job));raise SystemExit(result['exit_code'])
    if a.launch:
        plan=load_plan(a.plan);root=a.plan.resolve().parent;session='grvla08-pipeline-'+root.name[:8]
        cmd=[sys.executable,'-m','graph.integration_baseline.bc08_pipeline','--plan',str(a.plan.resolve())]
        if a.resume:cmd+=['--resume']
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(Path.cwd()),*tmux_environment(),
            shlex.join(cmd)+' >> '+shlex.quote(str(root/'pipeline.log'))+' 2>&1'],check=True)
        print(json.dumps(dict(session=session,plan=str(a.plan.resolve()),status=str(root/'status.json'))))
    else:supervise(a.plan,resume=a.resume)


if __name__=='__main__':main()
