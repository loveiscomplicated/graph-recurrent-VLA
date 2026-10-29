"""Actual nonempty pre-gate Wrist support, without downsampling."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from graph.plan_a_bc.common import read,write
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_geometry.core import components
from graph.persistent_geometry.self_filter import classify_robot
from graph.persistent_geometry.fusion import filtered_masks
from graph.geometry_reach10d.optimizations import components as optimized
from graph.geometry_reach10d.kernel_benchmark import timed,canonical


def run(root):
    output=root/'nonempty_geometry_kernel_benchmark_v2.json'
    if output.exists():return
    torch.set_num_threads(1);choices=[]
    for eid in [1000001,1000014]:
        with np.load(root/'pilot_cache/original/cache'/f'ep{eid}'/'graphs.npz') as z:
            all_masks=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            counts=all_masks[:,1,0].sum((1,2));order=np.argsort(-counts)
            for t in order[:5]:choices.append((int(counts[t]),eid,int(t),all_masks[t,1]))
    geometry=CurrentGeometry(root/'inputs',read(root/'normalizer.json')['schema']);selected=None
    for count,eid,t,mask in sorted(choices,key=lambda x:-x[0]):
        if count==0:continue
        with h5py.File(root/'raw'/str(eid)/'episode.h5') as f:cs,_,surfaces=geometry.from_h5(f,t,str(eid))
        c=next(x for x in cs if x.camera=='hand_camera')
        clean,_=filtered_masks(mask,classify_robot(c,surfaces))
        xyz=c.world_points()[clean[0]&np.isfinite(c.depth)]
        if len(xyz)>0:selected=(eid,t,xyz);break
    if selected is None:
        write(output,dict(status='NOT_TESTED',reason='No nonempty clean Wrist Peg support in the two representative episodes'),immutable=True);return
    eid,t,xyz=selected;radius=read(Path(__file__).resolve().parents[1]/'plan_a_geometry/protocol.json')['gate']['component_radius_m']
    available=torch.cuda.mem_get_info()[0];estimated=96*len(xyz)**2+256*2**20
    if estimated>.4*available:
        write(output,dict(status='NOT_TESTED',points=len(xyz),available_VRAM_bytes=available,
            conservative_estimated_bytes=estimated,reason='Dense GPU adjacency above40% free VRAM bound; no cap before gate'),immutable=True);return
    def gpu_cc():
        x=torch.from_numpy(xyz).cuda();adj=((x[:,None]-x[None,:]).square().sum(-1)<=radius**2).cpu().numpy()
        return connected_components(csr_matrix(adj),directed=False)[1]
    base,bt=timed(lambda:components(xyz,radius),5);opt,ot=timed(lambda:optimized(xyz,radius),5);gpu,gt=timed(gpu_cc,5)
    write(output,dict(status='PASS',episode_id=eid,timestep=t,points=len(xyz),
        available_VRAM_bytes=available,conservative_estimated_bytes=estimated,
        original_CPU_seconds=bt,optimized_CPU_seconds=ot,CUDA_transfer_inclusive_seconds=gt,
        optimized_partition_exact=bool(np.array_equal(canonical(base),canonical(opt))),
        CUDA_partition_exact=bool(np.array_equal(canonical(base),canonical(gpu))),
        GPU_adopted=False,scope='actual clean Peg support before original Wrist Gate, no point cap/downsampling',
        reason='GPU dense adjacency has quadratic memory; end-to-end geometry overlap retained; this primitive candidate is not used'),immutable=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
