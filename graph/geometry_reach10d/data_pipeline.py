"""Two spawn/exec CPU collectors overlap one CUDA causal-vision cache worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from graph.plan_a_bc.common import read,write,sha,append_csv,now
from graph.geometry_reach10d.prepare import prepare


def run(root):
    assert (root/'renderer_gpu_gate.json').exists()
    prepare(root)
    candidates=read(root/'collection_candidates.json')
    sources={str(p.name):sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(root/'collection_execution_source.json',dict(hashes=sources,created_at=now()))
    children=[];files=[];start=time.perf_counter()
    workers=read(root/'worker_benchmark.json')['selected_collectors']
    try:
        # Separate executables avoid forking any initialized CUDA state. Each
        # collector owns its env and episode partition; only cache owns vision.
        for worker in range(workers):
            manifest=root/f'collection_worker{worker}_manifest.json'
            write(manifest,[e['spec'] for e in candidates[worker::workers]])
            f=(root/f'collection_worker{worker}.log').open('a');files.append(f)
            cmd=[sys.executable,'-u','-B','-m','graph.geometry_reach10d.collect','--root',str(root),
                '--manifest',str(manifest),'--output-manifest',f'collection_worker{worker}_result.json']
            children.append((f'collection_worker{worker}',subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)))
        f=(root/'cache_full.log').open('a');files.append(f)
        children.append(('cache_full',subprocess.Popen([sys.executable,'-u','-B','-m','graph.geometry_reach10d.cache',
            '--root',str(root),'--stream'],stdout=f,stderr=subprocess.STDOUT)))
        write(root/'data_processes.json',dict(started_at=now(),children={k:p.pid for k,p in children},
            collectors=workers,cuda_vision_workers=1,tracker_isolation='per episode and per camera',start_method='subprocess exec'))
        done=set()
        while len(done)<len(children):
            for name,process in children:
                if name in done:continue
                code=process.poll()
                if code is None:continue
                if code:
                    write(root/(name+'_failure.json'),dict(returncode=code,at=now()))
                    raise RuntimeError(name+' failed; other processes preserved until orderly teardown')
                done.add(name);print('DATA SUBSTAGE COMPLETE',name,flush=True)
            time.sleep(2)
        receipts=[read(root/'raw'/str(e['spec']['episode_id'])/'receipt.json') for e in candidates]
        write(root/'capture_manifest_full.json',dict(episodes=receipts,attempted=len(candidates),
            accepted=sum(r['status']=='ACCEPTED' for r in receipts),wall_seconds=time.perf_counter()-start))
        write(root/'data_pipeline_complete.json',dict(status='COMPLETE',wall_seconds=time.perf_counter()-start,
            collection_workers=workers,cache_workers=1,source_hashes=sources),immutable=True)
    finally:
        # This only concerns child processes created by this task invocation.
        for name,process in children:
            if process.poll() is None:process.terminate()
        for _,process in children:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill()
        for f in files:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as e:
        write(a.root/'data_pipeline_failure.json',dict(status='FAILED',exception=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise
