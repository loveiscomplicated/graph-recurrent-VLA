"""Measured CUDA process overlap and bounded checkpoint-only OOM recovery."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from graph.plan_a_bc.common import read,write,now


def gpu_memory():
    line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return dict(used_bytes=int(line[0])*1024**2,free_bytes=int(line[1])*1024**2)


def launch(root,conditions,folder,steps=None,benchmark=False):
    folder.mkdir(parents=True,exist_ok=True);workers={};logs={}
    barrier=folder/'GO' if benchmark else None
    if barrier and barrier.exists():raise FileExistsError('reuse benchmark attempt forbidden')
    for c in conditions:
        out=folder/c if benchmark else root/('train_'+c)
        out.mkdir(parents=True,exist_ok=True)
        logs[c]=(folder/(c+'_'+str(time.time_ns())+'.log')).open('w')
        args=[sys.executable,'-u','-B','-m','graph.graph_recurrent10g.train_worker',
            '--root',str(root),'--condition',c,'--out',str(out)]
        if benchmark:args+=['--mode','benchmark','--steps',str(steps),'--barrier',str(barrier)]
        workers[c]=subprocess.Popen(args,stdout=logs[c],stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+240;peak=0;minimum_free=10**15;started=time.perf_counter();samples=[]
    try:
        if benchmark:
            while not all((folder/c/'ready.json').exists() for c in conditions):
                if any(p.poll() is not None for p in workers.values()):raise RuntimeError('worker failed before barrier')
                if time.monotonic()>deadline:raise TimeoutError('worker ready timeout')
                time.sleep(.2)
            write(barrier,dict(at=now(),pids={c:p.pid for c,p in workers.items()}));started=time.perf_counter()
        while any(p.poll() is None for p in workers.values()):
            mem=gpu_memory();peak=max(peak,mem['used_bytes']);minimum_free=min(minimum_free,mem['free_bytes'])
            samples.append(dict(elapsed=time.perf_counter()-started,**mem))
            status={c:dict(pid=p.pid,returncode=p.poll(),alive=p.poll() is None,
                progress=read(root/('train_'+c)/'status.json') if not benchmark and (root/('train_'+c)/'status.json').exists() else None)
                for c,p in workers.items()}
            write(root/'worker_status.json',dict(at=now(),workers=status,mode='benchmark' if benchmark else 'training'))
            time.sleep(.5)
        codes={c:p.wait() for c,p in workers.items()}
        results={c:read((folder/c if benchmark else root/('train_'+c))/'result.json')
            for c in conditions if (folder/c if benchmark else root/('train_'+c)).joinpath('result.json').exists() and codes[c]==0}
        wall=time.perf_counter()-started
        result=dict(conditions=conditions,codes=codes,results=results,wall_seconds=wall,
            aggregate_updates_per_second=sum(r['steps'] for r in results.values())/wall,
            peak_device_used_bytes=peak,minimum_device_free_bytes=minimum_free,samples=samples,
            process_ids={c:p.pid for c,p in workers.items()},distinct_processes=len(set(p.pid for p in workers.values()))==len(conditions))
        write(folder/'group_result.json',result);return result
    finally:
        # Only children owned by this invocation can be terminated.
        for p in workers.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in logs.values():f.close()


def benchmark(root):
    cfg=read(root/'config.json');out=root/'concurrency';singles=[]
    for c in 'ABC':singles.append(launch(root,[c],out/('single_'+c),cfg['benchmark_steps'],True))
    if any(any(x!=0 for x in r['codes'].values()) for r in singles):raise RuntimeError('single worker benchmark failed')
    triple=launch(root,list('ABC'),out/'triple',cfg['benchmark_steps'],True)
    serial_rate=3/sum(1/r['aggregate_updates_per_second'] for r in singles)
    # triple rate is all three updates summed per wall second.
    safe=all(v==0 for v in triple['codes'].values()) and triple['minimum_device_free_bytes']>=cfg['minimum_free_vram_bytes']
    reasonable=triple['aggregate_updates_per_second']>=cfg['concurrency_throughput_floor']*serial_rate
    chosen=3 if safe and reasonable else 1;pairtests=[]
    if chosen!=3:
        for pair in (list('AB'),list('BC')):
            r=launch(root,pair,out/('pair_'+''.join(pair)),cfg['benchmark_steps'],True);pairtests.append(r)
        if all(all(x==0 for x in r['codes'].values()) and r['minimum_device_free_bytes']>=cfg['minimum_free_vram_bytes']
                and r['aggregate_updates_per_second']>=cfg['concurrency_throughput_floor']*serial_rate for r in pairtests):chosen=2
    result=dict(status='PASS',chosen_concurrency=chosen,singles=singles,triple=triple,pairs=pairtests,
        serial_effective_updates_per_second=serial_rate,three_actual_CUDA_workers_tested=True,
        rule=dict(minimum_free_bytes=cfg['minimum_free_vram_bytes'],throughput_floor=cfg['concurrency_throughput_floor']),
        reason='highest measured safe concurrency with reasonable aggregate throughput; batch/objective unchanged')
    write(root/'concurrency_benchmark.json',result)
    return dict(status='PASS',steps=sum(r['steps'] for group in singles+[triple]+pairtests for r in group['results'].values()),files=['concurrency_benchmark.json'])


def train(root):
    cfg=read(root/'config.json');parallel=read(root/'concurrency_benchmark.json')['chosen_concurrency']
    if not cfg['miniature']:
        mini=Path(cfg['miniature_root']);audit=read(mini/'completion_audit.json')
        if audit['status']!='PASS' or read(mini/'pipeline_state.json')['status']!='COMPLETED':raise ValueError('miniature not completed')
        from .orchestrator import verify_all_stages
        verify_all_stages(mini)
        if read(mini/'miniature_restart_verification.json')['status']!='PASS':raise ValueError('miniature restart fixture missing')
        if read(mini/'full_horizon_smoke_verification.json')['status']!='PASS':raise ValueError('full-length runtime fixture missing')
        if read(root/'source_identity.json')['hashes']!=read(mini/'source_identity.json')['hashes']:
            raise ValueError('primary source differs from tested miniature')
    retries={c:0 for c in 'ABC'};attempt=0;history=[]
    while True:
        pending=[]
        for c in 'ABC':
            p=root/('train_'+c)/'result.json'
            if p.exists() and read(p)['steps']==cfg['training']['updates'] and read(p)['status']=='COMPLETE':continue
            pending.append(c)
        if not pending:break
        attempt+=1;batch=pending[:parallel]
        r=launch(root,batch,root/'training_attempts'/str(attempt));history.append(r)
        for c,code in r['codes'].items():
            if code==0:continue
            logs=list((root/'training_attempts'/str(attempt)).glob(c+'_*.log'))
            text=logs[-1].read_text() if logs else ''
            oom='out of memory' in text.lower()
            transient=oom or 'Resource temporarily unavailable' in text
            retries[c]+=1
            if not transient or retries[c]>3:raise RuntimeError(f'{c} worker failure; attempt logs retained, no contract bypass')
            if oom:parallel=max(1,parallel-1)
        write(root/'training_recovery.json',dict(attempts=history,retries=retries,current_concurrency=parallel,
            batch_size_unchanged=True,checkpoint_only_resume=True))
    return dict(status='PASS',files=[f'train_{c}/result.json' for c in 'ABC']+
        [f'train_{c}/last.pt' for c in 'ABC']+[f'train_{c}.csv' for c in 'ABC'])
