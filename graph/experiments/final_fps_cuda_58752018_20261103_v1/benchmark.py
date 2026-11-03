"""Paired unchanged NumPy final FPS and GPU-resident final FPS endpoints."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.tensor_records import numpy_records,budget_tensor
from graph.plan_a_geometry.core import budget
from graph.plan_a_bc.geometry import CurrentGeometry


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


def make_geometry(cls,assets,schema,frame):
    # The separate robot-overlap work may add this option. Both interfaces can
    # use prepared masks/gate without executing robot filtering in this benchmark.
    options=dict(coordinate_frame=frame)
    if 'robot_filter_backend' in inspect.signature(cls).parameters:
        options['robot_filter_backend']='cpu'
    return cls(assets,schema,**options)


def verify_graph(cameras,masks,data,assets,schema,reference):
    spec=importlib.util.spec_from_file_location('graph.plan_a_bc._fps_reference',reference/'graph/plan_a_bc/geometry.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    result={}
    for frame in ('ee','world'):
        old=make_geometry(module.CurrentGeometry,assets,schema,frame)
        new=make_geometry(CurrentGeometry,assets,schema,frame)
        new.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
        surfaces=new.fk.surfaces(data['robot76'])
        for g in (old,new):g.system.prepare=lambda *args:dict(conditions={'C3':masks},gate=dict(accepted=True))
        expected,er,eg=old.build(cameras,data,surfaces,masks)
        actual,ar,ag=new.build(cameras,data,surfaces,masks)
        same(ar,er);same(ag,eg)
        assert actual.packet['final_fps_backend']=='torch_cuda'
        assert actual.packet['node_ids']==expected.packet['node_ids']
        errors={}
        for name in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            a=getattr(actual,name);b=getattr(expected,name)
            assert torch.equal(a,b),name
            errors[name]=float((a-b).abs().max()) if a.numel() else 0.
        result[frame]=dict(all_graph_tensors_and_records_exact=True,nodes=len(actual.node_type),
            edges=int(actual.edge_index.shape[1]),max_abs_errors=errors)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema','reference-source','baseline-inputs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
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
    schema=json.loads(a.schema.read_text())['schema'];frames=('ee','world')
    gpu={};cpu={};saved={};checks={}
    with np.load(a.baseline_inputs,allow_pickle=False) as baseline:
        for frame in frames:
            g=make_geometry(CurrentGeometry,a.assets,schema,frame)
            pools,_,_,_=g.system.represent(cameras,{'conditions':{'C3':masks}},'C3',('SF',),tensor_records=True)
            gpu[frame]=pools['SF'];cpu[frame]=numpy_records(gpu[frame]);saved[frame]=deepcopy(cpu[frame])
            assert len(gpu[frame])==212 and gpu[frame].xyz.is_cuda
            np.testing.assert_array_equal(np.array([r['xyz'] for r in cpu[frame]]),baseline[frame+'_xyz'])
            np.testing.assert_array_equal(np.array([r['persistent_node_id'] for r in cpu[frame]]),baseline[frame+'_ids'])
            expected=budget(cpu[frame],128)
            assert [r['persistent_node_id'] for r in expected]==baseline[frame+'_ids'][baseline[frame+'_selected_indices']].tolist()
    def run(frame,mode):
        if mode=='cpu':return budget(cpu[frame],128)
        if mode=='legacy_from_gpu':return budget(numpy_records(gpu[frame]),128)
        selected=budget_tensor(gpu[frame],128)
        return numpy_records(selected) if mode=='gpu_to_cpu' else selected
    for frame in frames:
        expected=run(frame,'cpu')
        for mode in ('legacy_from_gpu','gpu_to_cpu'):same(run(frame,mode),expected)
        with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('array readback forbidden')), \
             patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy forbidden')), \
             patch.object(torch.Tensor,'tolist',side_effect=AssertionError('host point lists forbidden')):
            selected=run(frame,'gpu')
            assert selected.xyz.is_cuda and selected.source_pixels.is_cuda
        same(numpy_records(selected),expected)
        checks[frame]=dict(input_points=212,selected_points=len(selected),box_input_points=0,
            all_record_fields_order_provenance_exact=True,no_cpu_numpy_tolist=True,
            prior_baseline_xyz_ids_and_selection_exact=True,selected_sources=len(selected.source_pixels))
    modes=('cpu','gpu','gpu_to_cpu','legacy_from_gpu')
    config=dict(task='10i_final_fps_cuda_20261103',warmups=a.warmups,repeats=a.repeats,
        modes=list(modes),frames=list(frames),actual_sensor_views=1,empty_cloned_base=True,
        includes_gpu_lex_string_order_and_ragged_source_gather=True,
        upstream_preparation_filter_gate_and_graph_excluded=True,
        input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.inputs.glob('*.npz')},
        baseline_inputs_sha256=hashlib.sha256(a.baseline_inputs.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=a.output/'experiments',config=config)
    try:
        combinations=[(f,m) for f in frames for m in modes]
        for _ in range(a.warmups):
            for f,m in combinations:run(f,m)
        torch.cuda.synchronize();times={f+'__'+m:[] for f,m in combinations}
        for i in range(a.repeats):
            order=combinations[i%len(combinations):]+combinations[:i%len(combinations)]
            if i%2:order=order[::-1]
            for frame,mode in order:
                torch.cuda.synchronize();start=perf_counter();out=run(frame,mode)
                if mode!='cpu':torch.cuda.synchronize()
                end=perf_counter();times[frame+'__'+mode].append((end-start)*1000)
                del out
        for frame in frames:
            same(cpu[frame],saved[frame]);same(numpy_records(gpu[frame]),saved[frame])
            rows=numpy_records(run(frame,'gpu'))
            (a.output/f'{frame}_selected_provenance.json').write_text(json.dumps([
                dict(id=r['persistent_node_id'],object_id=r['object_id'],structural=r['structural'],
                    sources=r['source_observation_ids']) for r in rows],indent=2)+'\n')
        data.update(episode=np.array(hand.episode),timestep=np.int64(hand.timestep),timestamp=np.float64(hand.timestamp))
        graph=verify_graph(cameras,masks,data,a.assets,schema,a.reference_source)
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],config=config,
            timings={k:stats(v) for k,v in times.items()},verification=checks,public_graph=graph,
            inputs_unchanged=True,gpu=torch.cuda.get_device_name(),numpy=np.__version__,torch=torch.__version__,
            threads=1,cpu_affinity_fixed=False,garbage_collection_default=True,
            gpu_endpoints_synchronized=True,static_identity_cache_and_jit_warmed=True)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(a.output/'timings.npz',**{k:np.asarray(v) for k,v in times.items()})
        logger.record(summary=summary)
        for path in a.output.glob('*'):
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
