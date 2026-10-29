"""Bounded two-collector benchmark, excluded from training and policy results."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
from graph.plan_a_bc.common import read,write


def run(root):
    gate=read(root/'renderer_gpu_gate.json');pilot=read(root/'capture_manifest.json')['episodes']
    selected=[r for r in pilot if r['spec']['episode_id'] in [1000001,1000014]]
    folder=root/'worker_benchmark';folder.mkdir(exist_ok=True)
    for name in ['normalizer.json','inputs','renderer_gpu_gate.json']:
        if not (folder/name).exists():(folder/name).symlink_to(root/name,target_is_directory=name=='inputs')
    children=[];files=[];start=time.perf_counter()
    try:
        for i,r in enumerate(selected):
            manifest=folder/f'worker{i}.json';write(manifest,[r['spec']])
            f=(folder/f'worker{i}.log').open('a');files.append(f)
            children.append(subprocess.Popen([sys.executable,'-B','-m','graph.geometry_reach10d.collect','--root',str(folder),
                '--manifest',str(manifest),'--output-manifest',f'result{i}.json'],stdout=f,stderr=subprocess.STDOUT))
        for p in children:p.wait(timeout=180);assert p.returncode==0
        rows=[read(folder/'raw'/str(r['spec']['episode_id'])/'receipt.json') for r in selected]
        ends=[datetime.fromisoformat(r['created_at']).timestamp() for r in rows]
        begins=[t-r['wall_seconds'] for t,r in zip(ends,rows)]
        simultaneous_wall=max(ends)-min(begins)
        serial=sum(r['gpu_renderer_seconds' if gate['adopted'] else 'cpu_seconds'] for r in gate['results'])
        speedup=serial/simultaneous_wall
        write(root/'worker_benchmark.json',dict(status='PASS',serial_capture_seconds=serial,
            parallel_capture_window_seconds=simultaneous_wall,parallel_total_seconds=time.perf_counter()-start,
            speedup=speedup,selected_collectors=2 if speedup>1.05 else 1,frames=sum(r['actions'] for r in rows),
            note='same complete two offset episodes, benchmark-only copies excluded from training; CUDA initialization via exec not fork'),immutable=True)
    finally:
        for f in files:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
