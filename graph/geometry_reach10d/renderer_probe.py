"""Complete identical CPU-physics Reach with CPU vs CUDA sensor readback."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from graph.plan_a_bc.common import read,write
from graph.geometry_reach10d.runtime import gpu_render_environment
from graph.geometry_reach10d.collect import collect_episode
from graph.geometry_reach10d.profile import Resources


def run(root):
    folder=root/'renderer_probe';folder.mkdir(exist_ok=True)
    for name in ['normalizer.json','inputs']:
        if not (folder/name).exists():(folder/name).symlink_to(root/name,target_is_directory=name=='inputs')
    rows=read(root/'capture_manifest.json')['episodes']
    selected=[r for r in rows if r['spec']['episode_id'] in [1000001,1000014]]
    torch.set_num_threads(1);env=gpu_render_environment(root);result=[];start=time.perf_counter()
    try:
        with Resources(root,'CUDA_renderer_complete_capture'):
            for nominal in selected:
                actual=collect_episode(folder,nominal['spec'],env)
                assert actual['status']=='ACCEPTED'
                checks=[]
                with h5py.File(nominal['raw_path']) as a,h5py.File(actual['raw_path']) as b:
                    keys=[];a.visititems(lambda k,v:keys.append(k) if isinstance(v,h5py.Dataset) else None)
                    for k in keys:
                        aa,bb=a[k][:],b[k][:]
                        checks.append(dict(key=k,exact=bool(np.array_equal(aa,bb,equal_nan=True)),
                            max_abs=float(np.nanmax(abs(aa.astype(float)-bb.astype(float)))) if aa.shape==bb.shape else None,
                            same_shape=aa.shape==bb.shape))
                result.append(dict(episode_id=nominal['spec']['episode_id'],checks=checks,
                    cpu_seconds=nominal['wall_seconds'],gpu_renderer_seconds=actual['wall_seconds']))
        exact=all(c['exact'] for r in result for c in r['checks'])
        speedup=sum(r['cpu_seconds'] for r in result)/sum(r['gpu_renderer_seconds'] for r in result)
        write(root/'renderer_gpu_gate.json',dict(status='PASS' if exact else 'FAIL',adopted=exact and speedup>1,
            complete_raw_capture_exact=exact,measured_speedup=speedup,results=result,
            physics_unchanged='physx_cpu; only sensor renderer/readback backend changed',wall_seconds=time.perf_counter()-start),immutable=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
