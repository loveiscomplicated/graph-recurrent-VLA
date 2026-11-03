"""Remeasure identical CPU Graph endpoint after sparse raw-depth lookup."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.tensor_records import budget_tensor,numpy_records
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.experiments.graph_build_timing_58752018_20261103_v1.benchmark import FIELDS,Trace,same,stats


class SparseTrace(Trace):
    def reset(self):
        super().reset()
        self.ms['depth_decode']=0.;self.calls['depth_decode']=0


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema','baseline'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--diagnostic-warmups',type=int,default=5)
    p.add_argument('--diagnostic-repeats',type=int,default=20)
    a=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('another CUDA process active; do not stop it')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    def load(name):
        with np.load(a.inputs/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    s=load('hand_camera.npz');data=load('robot.npz');mask=load('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,s['rgb'],s['depth_raw_mm'],
        s['intrinsic_cv'],s['extrinsic_cv'],s['cam2world_gl'],data['tcp_pose']).validate()
    cameras=(replace(hand,camera='base_camera'),hand)
    masks=dict(base_camera=np.zeros_like(mask),hand_camera=mask)
    data.update(episode=np.array(hand.episode),timestep=np.int64(hand.timestep),timestamp=np.float64(hand.timestamp))
    schema=json.loads(a.schema.read_text())['schema'];frames=('ee','world');gate=dict(accepted=True)
    baseline=json.loads((a.baseline/'summary.json').read_text())
    with np.load(a.baseline/'graph_reference.npz',allow_pickle=False) as z:
        reference={k:z[k].copy() for k in z.files}
    packets={f:json.loads((a.baseline/f'{f}_graph_packet.json').read_text()) for f in frames}
    def check(graph,frame):
        for name in FIELDS:
            value=getattr(graph,name)
            assert value.device.type=='cpu'
            array=value.numpy();expected=reference[frame+'__'+name]
            assert array.dtype==expected.dtype
            np.testing.assert_array_equal(array,expected)
        same(graph.packet,packets[frame])
    geometries={};records={};verification={}
    for frame in frames:
        g=CurrentGeometry(a.assets,schema,coordinate_frame=frame,robot_filter_backend='cpu')
        g.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
        surfaces=g.fk.surfaces(data['robot76'])
        g.system.prepare=lambda *args:dict(conditions={'C3':masks},gate=gate)
        pools,_,_,_=g.system.represent(cameras,{'conditions':{'C3':masks}},'C3',('SF',),tensor_records=True)
        records[frame]=numpy_records(budget_tensor(pools['SF'],128))
        graph,public_records,_=g.build(cameras,data,surfaces,masks)
        same(public_records,records[frame]);check(graph,frame)
        geometries[frame]=g
        verification[frame]=dict(input_points=len(records[frame]),nodes=len(graph.node_type),
            edges=int(graph.edge_index.shape[1]),surface_edge_audit=graph.packet['surface_edge_audit'],
            all_graph_tensors_packet_and_audit_exact_vs_saved_before=True,
            public_current_geometry_records_exact=True)
    saved_records=deepcopy(records);saved_data=deepcopy(data)
    torch.cuda.synchronize()
    def run(frame):
        g=geometries[frame];graph=g.system.graph(records[frame],data,cameras)
        return g._finish(graph,records[frame],gate,data)
    def forbidden(_):raise AssertionError('full-image depth conversion forbidden')
    with patch.object(CameraObservation,'depth',property(forbidden)):
        for frame in frames:check(run(frame)[0],frame)
    config=dict(task='10i_graph_depth_lookup_20261103',warmups=a.warmups,repeats=a.repeats,
        diagnostic_warmups=a.diagnostic_warmups,diagnostic_repeats=a.diagnostic_repeats,
        endpoint=baseline['config']['endpoint'],baseline_run_id=baseline['run_id'],
        baseline_is_prior_separate_run=True,actual_sensor_views=1,empty_cloned_base=True,
        gpu_readback_and_upstream_and_robot_filter_gate_excluded=True,
        input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.inputs.glob('*.npz')},
        baseline_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.baseline.iterdir() if f.is_file()},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert config['input_sha256']==baseline['config']['input_sha256']
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=a.output/'experiments',config=config)
    try:
        for _ in range(a.warmups):
            for frame in frames:run(frame)
        times={frame:[] for frame in frames}
        for i in range(a.repeats):
            for frame in (frames if i%2==0 else frames[::-1]):
                start=perf_counter();out=run(frame)
                times[frame].append((perf_counter()-start)*1000);del out
            if (i+1)%20==0:print(f'primary samples {i+1}/{a.repeats}',flush=True)
        diagnostics={f:{} for f in frames};calls={f:{} for f in frames}
        with SparseTrace() as trace:
            for _ in range(a.diagnostic_warmups):
                for frame in frames:trace.run(geometries[frame],records[frame],data,cameras,gate)
            for i in range(a.diagnostic_repeats):
                for frame in (frames[::-1] if i%2==0 else frames):
                    out=trace.run(geometries[frame],records[frame],data,cameras,gate)
                    assert trace.calls['depth_decode']==0
                    for k,v in trace.ms.items():diagnostics[frame].setdefault(k,[]).append(v)
                    for k,v in trace.calls.items():calls[frame].setdefault(k,[]).append(v)
                    check(out[0],frame);del out
        same(records,saved_records);same(data,saved_data)
        arrays={}
        for frame in frames:
            graph=run(frame)[0];check(graph,frame)
            for name in FIELDS:arrays[frame+'__'+name]=getattr(graph,name).numpy().copy()
            (a.output/f'{frame}_graph_packet.json').write_text(json.dumps(graph.packet,indent=2)+'\n')
            verification[frame].update(inputs_unchanged=True,no_full_depth_getter=True)
        after={f:stats(v) for f,v in times.items()}
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],config=config,
            before_timings=baseline['timings'],timings=after,
            median_speedup={f:baseline['timings'][f]['median_ms']/after[f]['median_ms'] for f in frames},
            diagnostics={f:{k:stats(v) for k,v in values.items()} for f,values in diagnostics.items()},
            diagnostic_calls={f:{k:dict(min=min(v),max=max(v),median=float(np.median(v))) for k,v in values.items()} for f,values in calls.items()},
            verification=verification,timed_device='cpu',numpy=np.__version__,torch=torch.__version__,
            threads=1,cpu_affinity_fixed=False,garbage_collection_default=True)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(a.output/'timings.npz',**times)
        np.savez(a.output/'diagnostics.npz',**{f+'__'+k:np.asarray(v) for f,d in diagnostics.items() for k,v in d.items()})
        np.savez(a.output/'diagnostic_calls.npz',**{f+'__'+k:np.asarray(v) for f,d in calls.items() for k,v in d.items()})
        np.savez(a.output/'graph_reference.npz',**arrays)
        logger.record(summary=summary)
        for path in a.output.glob('*'):
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
