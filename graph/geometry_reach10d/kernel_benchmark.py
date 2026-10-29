"""End-to-end transfer-inclusive CPU/CUDA geometry candidates; never adopted here."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from graph.plan_a_bc.common import read,write
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_geometry.core import components
from graph.geometry_reach10d.optimizations import components as optimized


def canonical(labels):
    ids={};return np.array([ids.setdefault(int(x),len(ids)) for x in labels])


def timed(fn,repeats=10):
    fn();start=time.perf_counter()
    for _ in range(repeats):out=fn()
    torch.cuda.synchronize();return out,(time.perf_counter()-start)/repeats


def run(root):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    schema=read(root/'normalizer.json')['schema'];geom=CurrentGeometry(root/'inputs',schema)
    with h5py.File(root/'raw/1000001/episode.h5') as f:cameras,_,_=geom.from_h5(f,0,'kernel_probe')
    camera=cameras[1];depth=camera.depth;K=camera.intrinsic;mat=camera.world_from_camera.astype(float)
    def gpu_world():
        d=torch.from_numpy(depth).cuda();k=torch.from_numpy(K.astype(float)).cuda();m=torch.from_numpy(mat).cuda()
        v,u=torch.meshgrid(torch.arange(d.shape[0],device='cuda'),torch.arange(d.shape[1],device='cuda'),indexing='ij')
        xyz=torch.stack([(u+.5-k[0,2])*d/k[0,0],(v+.5-k[1,2])*d/k[1,1],d],-1)
        return (xyz@m[:3,:3].T+m[:3,3]).cpu().numpy()
    cpu,ct=timed(camera.world_points,100);gpu,gt=timed(gpu_world,100)
    rows=[dict(stage='backprojection_world_128x128',cpu_seconds=ct,gpu_transfer_inclusive_seconds=gt,
        exact=bool(np.array_equal(cpu,gpu,equal_nan=True)),max_abs=float(np.nanmax(abs(cpu-gpu))),adopted=False)]
    with np.load(root/'pilot_cache/original/cache/ep1000001/graphs.npz') as z:
        mask=np.unpackbits(z['mask_bits'][0]).reshape(2,2,128,128)[1,0].astype(bool)
    xyz=camera.world_points()[mask&np.isfinite(depth)]
    radius=read(Path(__file__).resolve().parents[1]/'plan_a_geometry/protocol.json')['gate']['component_radius_m']
    # Dense CUDA adjacency is bounded explicitly, with no downsampling.
    if len(xyz)<=6000:
        def gpu_cc():
            a=torch.from_numpy(xyz).cuda()
            adj=((a[:,None]-a[None,:]).square().sum(-1)<=radius**2).cpu().numpy()
            return connected_components(csr_matrix(adj),directed=False)[1]
        baseline,bt=timed(lambda:components(xyz,radius),3)
        opt,ot=timed(lambda:optimized(xyz,radius),3)
        gpu,gt=timed(gpu_cc,3)
        rows.append(dict(stage='wrist_raw_support_components',points=len(xyz),cpu_seconds=bt,
            optimized_cpu_seconds=ot,gpu_transfer_inclusive_seconds=gt,
            same_partition=bool(np.array_equal(canonical(baseline),canonical(gpu))),
            optimized_same_partition=bool(np.array_equal(canonical(baseline),canonical(opt))),adopted=False,
            reason='Dense GPU adjacency scales quadratically; frozen sparse radius semantics retained'))
    else:rows.append(dict(stage='wrist_components',points=len(xyz),status='NOT_TESTED',
        reason='Dense all-pairs candidate exceeds conservative memory limit; no pre-gate subsampling'))
    # Exact representative pairwise-distance primitive at cap128, including transfers.
    points=cpu.reshape(-1,3);points=points[np.isfinite(points).all(1)][:128]
    a,bt=timed(lambda:((points[:,None]-points[None,:])**2).sum(-1),100)
    def gpu_dist():
        p=torch.from_numpy(points).cuda();return ((p[:,None]-p[None,:])**2).sum(-1).cpu().numpy()
    b,gt=timed(gpu_dist,100)
    rows.append(dict(stage='cap128_pair_distance_primitive',cpu_seconds=bt,gpu_transfer_inclusive_seconds=gt,
        exact=bool(np.array_equal(a,b)),adopted=False,reason='Full graph/provenance assembly stays in audited CPU implementation'))
    write(root/'geometry_kernel_benchmark.json',dict(rows=rows,scope='primitive and connected-component candidates, not full GPU geometry pipeline speedup',
        TF32=False,precision='float64 to match geometry arithmetic',output_changes_not_adopted=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
