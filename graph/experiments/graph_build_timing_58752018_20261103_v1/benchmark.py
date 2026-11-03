"""Unchanged CPU edges/Graph baseline after final-FPS CPU records are ready."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.tensor_records import budget_tensor,numpy_records
from graph.plan_a_bc.geometry import CurrentGeometry
import graph.plan_a_geometry.core as core
import graph.persistent_geometry.graph as graph_module


FIELDS=('node_input','node_type','edge_index','edge_attr','edge_type')


def same(a,b):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)


def same_graph(a,b):
    for name in FIELDS:
        x=getattr(a,name);y=getattr(b,name)
        assert x.device.type==y.device.type=='cpu' and x.dtype==y.dtype and torch.equal(x,y),name
    same(a.packet,b.packet)


def stats(x):
    return dict(median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),
                min_ms=float(np.min(x)),max_ms=float(np.max(x)))


class Trace(ExitStack):
    """Separate diagnostic invocation; never active during primary timings."""
    def __enter__(self):
        super().__enter__();self.reset()
        original=core.build_graph
        def build(*args,**kwargs):
            self.build_start=perf_counter();result=original(*args,**kwargs)
            self.ms['build_total']=(perf_counter()-self.build_start)*1000
            return result
        self.enter_context(patch.object(core,'build_graph',build))
        for name in ('surface_edges','nearest_robot_pairs','global_peg_box_pairs','edge_attributes','free_space_violation'):
            original_function=getattr(graph_module,name)
            def measured(*args,_name=name,_original=original_function,**kwargs):
                start=perf_counter()
                if _name=='surface_edges':self.ms['node_preparation']=(start-self.build_start)*1000
                result=_original(*args,**kwargs)
                self.ms[_name]=self.ms.get(_name,0.)+(perf_counter()-start)*1000
                self.calls[_name]=self.calls.get(_name,0)+1
                return result
            self.enter_context(patch.object(graph_module,name,measured))
        getter=CameraObservation.depth.fget
        def depth(camera):
            start=perf_counter();result=getter(camera)
            self.ms['depth_decode']=self.ms.get('depth_decode',0.)+(perf_counter()-start)*1000
            self.calls['depth_decode']=self.calls.get('depth_decode',0)+1
            return result
        self.enter_context(patch.object(CameraObservation,'depth',property(depth)))
        return self

    def reset(self):
        self.ms={};self.calls={};self.build_start=None

    def run(self,geometry,records,data,cameras,gate):
        self.reset();start=perf_counter()
        graph=geometry.system.graph(records,data,cameras);built=perf_counter()
        result=geometry._finish(graph,records,gate,data);end=perf_counter()
        self.ms['plan_a_schema_checks']=(built-start)*1000-self.ms['build_total']
        self.ms['finish_checks']=(end-built)*1000
        self.ms['diagnostic_total']=(end-start)*1000
        direct=('node_preparation','surface_edges','nearest_robot_pairs','global_peg_box_pairs','edge_attributes')
        self.ms['other_graph_assembly']=self.ms['build_total']-sum(self.ms.get(k,0.) for k in direct)
        return result


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--diagnostic-warmups',type=int,default=5)
    p.add_argument('--diagnostic-repeats',type=int,default=20)
    a=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('another CUDA process active; do not stop it')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    assert core.build_graph is graph_module.build_graph
    assert graph_module.surface_edges.__module__=='graph.persistent_geometry.graph'
    def load(name):
        with np.load(a.inputs/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    s=load('hand_camera.npz');data=load('robot.npz');mask=load('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,s['rgb'],s['depth_raw_mm'],
        s['intrinsic_cv'],s['extrinsic_cv'],s['cam2world_gl'],data['tcp_pose']).validate()
    assert hand.depth_raw_mm.shape==(512,512)
    cameras=(replace(hand,camera='base_camera'),hand)
    masks=dict(base_camera=np.zeros_like(mask),hand_camera=mask)
    data.update(episode=np.array(hand.episode),timestep=np.int64(hand.timestep),timestamp=np.float64(hand.timestamp))
    schema=json.loads(a.schema.read_text())['schema'];frames=('ee','world')
    geometries={};records={};expected={};checks={};gate=dict(accepted=True)
    for frame in frames:
        g=CurrentGeometry(a.assets,schema,coordinate_frame=frame,robot_filter_backend='cpu')
        g.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
        surfaces=g.fk.surfaces(data['robot76'])
        g.system.prepare=lambda *args:dict(conditions={'C3':masks},gate=gate)
        pools,_,_,_=g.system.represent(cameras,{'conditions':{'C3':masks}},'C3',('SF',),tensor_records=True)
        records[frame]=numpy_records(budget_tensor(pools['SF'],128))  # GPU/CPU conversion OUTSIDE timer.
        graph,public_records,public_gate=g.build(cameras,data,surfaces,masks)
        same(public_records,records[frame]);same(public_gate,gate)
        assert len(records[frame])==128
        expected[frame]=graph;geometries[frame]=g
        checks[frame]=dict(input_peg=128,input_box=0,nodes=len(graph.node_type),
            edges=int(graph.edge_index.shape[1]),edge_counts_by_type={str(i):int((graph.edge_type==i).sum()) for i in range(3)},
            surface_edge_audit=graph.packet['surface_edge_audit'],
            public_current_geometry_records_exact=True)
    before_records=deepcopy(records);before_data=deepcopy(data)
    torch.cuda.synchronize()
    def run(frame):
        g=geometries[frame]
        graph=g.system.graph(records[frame],data,cameras)
        return g._finish(graph,records[frame],gate,data)
    for frame in frames:same_graph(run(frame)[0],expected[frame])
    config=dict(task='10i_graph_build_timing_20261103',warmups=a.warmups,repeats=a.repeats,
        diagnostic_warmups=a.diagnostic_warmups,diagnostic_repeats=a.diagnostic_repeats,
        endpoint='CPU post-FPS records + CPU robot/calibration/raw depth ready -> PlanA.graph + CurrentGeometry._finish',
        excludes=['GPU-to-CPU return','upstream projection/features/sampling/view/records/dedup/FPS',
                  'robot filter/gate','FK','model packing/device upload/inference'],
        actual_sensor_views=1,empty_cloned_base=True,input_peg=128,input_box=0,
        primary_has_no_diagnostic_hooks=True,
        input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=a.output/'experiments',config=config)
    try:
        for i in range(a.warmups):
            for frame in frames:run(frame)
            if (i+1)%10==0:print(f'primary warmup {i+1}/{a.warmups}',flush=True)
        times={frame:[] for frame in frames}
        for i in range(a.repeats):
            for frame in (frames if i%2==0 else frames[::-1]):
                start=perf_counter();out=run(frame)
                times[frame].append((perf_counter()-start)*1000)
                del out
            if (i+1)%20==0:print(f'primary samples {i+1}/{a.repeats}',flush=True)
        diagnostics={frame:{} for frame in frames};calls={frame:{} for frame in frames}
        with Trace() as trace:
            for _ in range(a.diagnostic_warmups):
                for frame in frames:trace.run(geometries[frame],records[frame],data,cameras,gate)
            for i in range(a.diagnostic_repeats):
                for frame in (frames[::-1] if i%2==0 else frames):
                    out=trace.run(geometries[frame],records[frame],data,cameras,gate)
                    for k,v in trace.ms.items():diagnostics[frame].setdefault(k,[]).append(v)
                    for k,v in trace.calls.items():calls[frame].setdefault(k,[]).append(v)
                    same_graph(out[0],expected[frame]);del out
                if (i+1)%10==0:print(f'diagnostic samples {i+1}/{a.diagnostic_repeats}',flush=True)
        same(records,before_records);same(data,before_data)
        graph_arrays={}
        for frame in frames:
            final=run(frame)[0];same_graph(final,expected[frame])
            checks[frame].update(all_graph_tensors_and_packet_exact=True,inputs_unchanged=True,
                                 diagnostic_graph_exact=True,output_device='cpu')
            for field in FIELDS:graph_arrays[frame+'__'+field]=getattr(final,field).numpy().copy()
            (a.output/f'{frame}_graph_packet.json').write_text(json.dumps(final.packet,indent=2)+'\n')
        cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                 if line.startswith('model name'))
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],config=config,
            timings={f:stats(v) for f,v in times.items()},
            diagnostics={f:{k:stats(v) for k,v in values.items()} for f,values in diagnostics.items()},
            diagnostic_calls={f:{k:dict(min=min(v),max=max(v),median=float(np.median(v))) for k,v in values.items()} for f,values in calls.items()},
            verification=checks,cpu=cpu,gpu_preparation=torch.cuda.get_device_name(),
            numpy=np.__version__,torch=torch.__version__,python=platform.python_version(),
            threads=1,timed_device='cpu',cpu_affinity_fixed=False,garbage_collection_default=True)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(a.output/'timings.npz',**times)
        np.savez(a.output/'diagnostics.npz',**{f+'__'+k:np.asarray(v) for f,values in diagnostics.items() for k,v in values.items()})
        np.savez(a.output/'diagnostic_calls.npz',**{f+'__'+k:np.asarray(v) for f,values in calls.items() for k,v in values.items()})
        np.savez(a.output/'graph_reference.npz',**graph_arrays)
        logger.record(summary=summary)
        for path in a.output.glob('*'):
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
