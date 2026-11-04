"""Staged depth/pose/hulls vs existing filter, every evaluated pixel/field."""
import argparse,json
from pathlib import Path
import torch
from .benchmark import geometry,load,ROOT
from .ingress import TensorIngress
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from utils.experiment_logger import ExperimentLogger


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    geometry_cpu=geometry('triton_fused');filter=CudaRobotFilter();ingress=TensorIngress(filter);rows=[]
    scenes=json.loads((ROOT/'protocol.json').read_text())['scenes']
    with ExperimentLogger(team='integration',model='task10i_staged_filter_equivalence',job_type='evaluate',seed=42,
            root=output/'experiments',config=dict(scenes='all saved actual repeat0',sizes=[128,512,768,1024],diagnostics=True)) as log:
        try:
            for scene in scenes:
                for size in (128,512,768,1024):
                    cameras,data,surfaces,masks=load(scene['id'],size,geometry_cpu)
                    ref=filter.filter_cameras(cameras,surfaces,masks,diagnostics=True)
                    frame=ingress.stage(cameras,data,surfaces)
                    current={name:dict(depth_cuda=t.depth64,world_cuda=t.world) for name,t in frame.tensors.items()}
                    actual=filter.filter_cameras(frame.cameras,frame.surfaces,masks,diagnostics=True,
                        prepared=frame.prepared,camera_tensors=current)
                    maximum=0.
                    for name in masks:
                        a,b=actual[name],ref[name];evaluated=b['evaluated_pixels']
                        for k in ('clean_masks','evaluated_pixels'):
                            if not torch.equal(a[k],b[k]):raise AssertionError((scene['id'],size,name,k))
                        for k in ('confirmed_robot','uncertain','candidate','link_owner','inside_known_link_volume'):
                            if not torch.equal(a[k][evaluated],b[k][evaluated]):raise AssertionError((scene['id'],size,name,k))
                        aa,bb=a['predicted_robot_depth'][evaluated],b['predicted_robot_depth'][evaluated]
                        torch.testing.assert_close(aa,bb,rtol=0,atol=1e-12)
                        finite=torch.isfinite(aa)&torch.isfinite(bb)
                        if bool(finite.any()):maximum=max(maximum,float((aa[finite]-bb[finite]).abs().max()))
                    rows.append(dict(scene=scene['id'],size=size,all_discrete_exact=True,depth_max_abs=maximum))
                    print(json.dumps(rows[-1]),flush=True)
            path=output/'results.json';path.write_text(json.dumps(dict(status='PASS',rows=rows),indent=2));log.link_output(path)
        finally:ingress.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
