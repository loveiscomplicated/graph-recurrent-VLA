"""Independent supervised lifecycle. No policy entrypoint is imported/executed."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

from graph.plan_a_bc.common import read,write,now


def run(root):
    children=[];streams=[]
    try:
        for worker in range(4):
            stream=(root/f'collect_{worker}.log').open('a');streams.append(stream)
            children.append((f'collect_{worker}',subprocess.Popen([sys.executable,'-u','-B','-m',
                'graph.gt_pretrain10f1.collect','--root',str(root),'--worker',str(worker),'--workers','4'],
                stdout=stream,stderr=subprocess.STDOUT)))
        assert read(root/'concurrent_worker_benchmark.json')['status']=='PASS'
        for worker in range(2):
            stream=(root/f'cache_{worker}.log').open('a');streams.append(stream)
            children.append((f'cache_{worker}',subprocess.Popen([sys.executable,'-u','-B','-m','graph.gt_pretrain10f1.cache',
                '--root',str(root),'--worker',str(worker),'--workers','2'],stdout=stream,stderr=subprocess.STDOUT)))
        write(root/'pipeline_state.json',dict(stage='COLLECT_AND_CACHE',children={n:p.pid for n,p in children},at=now()))
        done=set()
        while len(done)<len(children):
            for name,process in children:
                if name in done:continue
                rc=process.poll()
                if rc is None:continue
                if rc:
                    write(root/'collection_failure.json',dict(process=name,returncode=rc,at=now()))
                    raise RuntimeError(f'{name} exited {rc}')
                done.add(name);print('FINISHED',name,flush=True)
            time.sleep(2)
        manifests=[read(root/f'graph_cache_manifest_worker{w}.json') for w in range(2)]
        write(root/'graph_cache_manifest.json',dict(
            episodes=sorted([e for m in manifests for e in m['episodes']],key=lambda e:e['episode_id']),
            paired_episodes=sorted([e for m in manifests for e in m['paired_episodes']],key=lambda e:e['spec']['episode_id']),
            source_before_GNN=True,CUDA_episode_workers=2,geometry_workers_per_CUDA_worker=4),immutable=True)
        # The training source is finalized before this independent entrypoint runs.
        write(root/'pipeline_state.json',dict(stage='DATA_COMPLETE',at=now()))
        print('DATA COMPLETE; primary training requires explicit frozen trainer launch',flush=True)
    except BaseException as exc:
        write(root/'pipeline_failure.json',dict(status='FAIL',type=type(exc).__name__,traceback=traceback.format_exc(),at=now()))
        raise
    finally:
        for _,p in children:
            if p.poll() is None:p.terminate()
        for _,p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill()
        for f in streams:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
