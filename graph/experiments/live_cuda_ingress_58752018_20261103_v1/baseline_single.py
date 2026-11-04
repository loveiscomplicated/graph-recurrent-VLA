"""Frozen512 CPU image adapter/gate -> current GPU graph, excluding robot filter."""
import os,json,argparse,hashlib
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.persistent_geometry.tensor_graph import numpy_graph,FIELDS


def load(root):
    def read(name):
        with np.load(root/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    s=read('hand_camera.npz');data=read('robot.npz');mask=read('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,s['rgb'],s['depth_raw_mm'],s['intrinsic_cv'],s['extrinsic_cv'],s['cam2world_gl'],data['tcp_pose']).validate()
    cameras=(replace(hand,camera='base_camera'),hand);masks=dict(base_camera=np.zeros_like(mask),hand_camera=mask)
    data.update(episode=np.array(hand.episode),timestep=np.int64(80),timestamp=np.float64(4.))
    return cameras,data,masks


def stats(v):return dict(median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),min_ms=float(np.min(v)),max_ms=float(np.max(v)))


def main():
    p=argparse.ArgumentParser()
    for k in ('inputs','assets','schema','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=0);p.add_argument('--repeats',type=int,default=1)
    a=p.parse_args();assert os.environ.get('CONTAINER_ID')=='58752018';torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False);cameras,data,raw=load(a.inputs)
    geo=CurrentGeometry(a.assets,json.loads(a.schema.read_text())['schema'],robot_filter_backend='cpu')
    surfaces=geo.fk.surfaces(data['robot76']);gpu={k:torch.tensor(v,device='cuda') for k,v in raw.items()}
    # Fixed clean masks isolate the non-robot path. No RF is executed here.
    def run():
        host_raw={k:v.cpu().numpy() for k,v in gpu.items()};clean={k:v.cpu().numpy() for k,v in gpu.items()}
        return geo._build_prepared(cameras,data,surfaces,host_raw,clean)
    config=dict(task='10i_live_cuda_ingress_before',warmups=a.warmups,repeats=a.repeats,robot_filter_excluded=True,
        boundary='CPU camera RGB-D + raw/clean CUDA masks ready -> CPU mask readback/gate/shared CUDA views/optimized graph -> CUDA Graph ready',
        fixed_clean_equals_raw=True,actual_sensor_views=1,empty_cloned_base=True,terra_SAM2_render_FK_and_IO_excluded=True,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.inputs.glob('*.npz')})
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=a.output/'experiments',config=config)
    try:
        for _ in range(a.warmups):out=run();torch.cuda.synchronize();del out
        times=[]
        for i in range(a.repeats):
            torch.cuda.synchronize();start=perf_counter();out=run();torch.cuda.synchronize();times.append((perf_counter()-start)*1000);last=out
            if (i+1)%20==0:print('samples',i+1,flush=True)
        g,r,gate=last;cpu=numpy_graph(g)
        np.savez(a.output/'timings.npz',total=np.asarray(times));np.savez(a.output/'graph.npz',**{k:getattr(cpu,k).numpy() for k in FIELDS})
        (a.output/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
        (a.output/'packet.json').write_text(json.dumps(cpu.packet,indent=2)+'\n')
        summary=dict(run_id=logger.run_id,config=config,timings=stats(times),nodes=len(g.node_type),edges=g.edge_index.shape[1])
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');logger.record(summary=summary)
        for p in a.output.iterdir():
            if p.is_file():logger.link_output(p)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:logger.finish(failed=True);raise

if __name__=='__main__':main()
