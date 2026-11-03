"""Unmodified per-object final128 NumPy FPS; post-dedup CPU records are ready."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.tensor_records import numpy_records
from graph.persistent_geometry.coordinates import record_xyz
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_geometry.core import budget


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


def stats(x):
    return dict(median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),
                min_ms=float(np.min(x)),max_ms=float(np.max(x)))


def main():
    parser=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=100)
    a=parser.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('another CUDA process is active; do not stop it')
    torch.set_num_threads(1)
    # Detect opt-in native/other patched implementations: measure only the public NumPy path.
    import graph.plan_a_geometry.core as core
    import graph.plan_a_bc.geometry as adapter
    import graph.persistent_geometry.graph as graph_module
    assert budget is core.budget is adapter.budget
    assert core.farthest_rows is graph_module.farthest_rows
    a.output.mkdir(parents=True,exist_ok=False)
    def load(name):
        with np.load(a.inputs/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    sensor=load('hand_camera.npz');data=load('robot.npz');mask=load('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,sensor['rgb'],sensor['depth_raw_mm'],
        sensor['intrinsic_cv'],sensor['extrinsic_cv'],sensor['cam2world_gl'],data['tcp_pose']).validate()
    assert hand.depth.shape==(512,512)
    cameras=(replace(hand,camera='base_camera'),hand)
    masks=dict(base_camera=np.zeros_like(mask),hand_camera=mask)
    data.update(episode=np.array(hand.episode),timestep=np.int64(hand.timestep),timestamp=np.float64(hand.timestamp))
    schema=json.loads(a.schema.read_text())['schema']
    frames=('ee','world');inputs={};before={};expected={};checks={};arrays={}
    for frame in frames:
        options=dict(coordinate_frame=frame)
        if 'robot_filter_backend' in inspect.signature(CurrentGeometry).parameters:
            options['robot_filter_backend']='cpu'
        geometry=CurrentGeometry(a.assets,schema,**options)
        geometry.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
        surfaces=geometry.fk.surfaces(data['robot76'])
        prepared=dict(conditions={'C3':masks},gate=dict(accepted=True))
        pools,audits,_,_=geometry.system.represent(cameras,prepared,'C3',('SF',),tensor_records=True)
        assert pools['SF'].xyz.is_cuda
        inputs[frame]=numpy_records(pools['SF'])  # Actual post-dedup readback OUTSIDE timing.
        before[frame]=deepcopy(inputs[frame])
        expected[frame]=budget(inputs[frame],128)
        geometry.system.prepare=lambda *args:prepared  # Robot filter/gate are excluded.
        public_graph,public_records,_=geometry.build(cameras,data,surfaces,masks)
        same(public_records,expected[frame]);same(inputs[frame],before[frame])
        counts={str(oid):dict(input=sum(r['object_id']==oid for r in inputs[frame]),
            selected=sum(r['object_id']==oid for r in expected[frame])) for oid in (0,1)}
        assert counts=={'0':{'input':212,'selected':128},'1':{'input':0,'selected':0}}
        index={r['persistent_node_id']:i for i,r in enumerate(inputs[frame])}
        assert len(index)==len(inputs[frame])
        arrays[frame+'_xyz']=np.asarray([record_xyz(r) for r in inputs[frame]],np.float64)
        arrays[frame+'_ids']=np.asarray([r['persistent_node_id'] for r in inputs[frame]])
        arrays[frame+'_object_id']=np.asarray([r['object_id'] for r in inputs[frame]],np.int64)
        arrays[frame+'_structural']=np.asarray([r['structural'] for r in inputs[frame]],bool)
        arrays[frame+'_selected_indices']=np.asarray([index[r['persistent_node_id']] for r in expected[frame]],np.int64)
        checks[frame]=dict(public_current_geometry_records_all_fields_and_sources_exact=True,
            counts=counts,upstream_dedup_audit=audits['SF'],graph_nodes=len(public_graph.node_type),
            graph_edges=int(public_graph.edge_index.shape[1]),input_records_unchanged=True)
    torch.cuda.synchronize()
    config=dict(task='10i_final_fps_timing_20261103',warmups=a.warmups,repeats=a.repeats,
        frames=list(frames),count_per_object=128,hybrid=False,
        endpoint='CPU post-dedup records ready -> budget(records,128) CPU selected records',
        includes=['per-object row filtering','frame checks','XYZ/ID preparation',
                  'stable-ID seed and tie order','127 farthest-point iterations for Peg',
                  'selected-record list gathering'],
        excludes=['GPU dedup','GPU-to-CPU conversion','packet/observation/view preparation',
                  'graph construction and its record reordering','robot filtering/gate'],
        actual_sensor_views=1,empty_cloned_base=True,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,
        root=a.output/'experiments',config=config)
    try:
        for _ in range(a.warmups):
            for frame in frames:budget(inputs[frame],128)
        times={frame:[] for frame in frames}
        # No GPU calls, internal clocks, logging, or file IO in this CPU endpoint.
        for i in range(a.repeats):
            for frame in (frames if i%2==0 else frames[::-1]):
                start=perf_counter();selected=budget(inputs[frame],128)
                elapsed=(perf_counter()-start)*1000
                times[frame].append(elapsed)
                del selected
        for frame in frames:
            same(inputs[frame],before[frame]);same(budget(inputs[frame],128),expected[frame])
            (a.output/f'{frame}_selected_provenance.json').write_text(json.dumps([
                dict(id=r['persistent_node_id'],object_id=r['object_id'],structural=r['structural'],
                    sources=r['source_observation_ids']) for r in expected[frame]],indent=2)+'\n')
        cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                 if line.startswith('model name'))
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],
            config=config,timings={k:stats(v) for k,v in times.items()},verification=checks,
            cpu=cpu,gpu_preparation=torch.cuda.get_device_name(),timed_device='cpu',
            numpy=np.__version__,torch=torch.__version__,python=platform.python_version(),
            torch_threads=torch.get_num_threads(),environment_threads={k:os.environ.get(k) for k in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            cpu_affinity_fixed=False,garbage_collection_default=True)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(a.output/'timings.npz',**{k:np.asarray(v) for k,v in times.items()})
        np.savez(a.output/'fps_inputs.npz',**arrays)
        logger.record(summary=summary)
        for path in a.output.glob('*'):
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
