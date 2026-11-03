"""Paired old CPU SF records/dedup and resident CUDA tensor records/dedup."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.map import current_records
from graph.persistent_geometry.tensor_fusion import (prepare_tensor_views,
    observations_from_tensor_packet, consistent_views_tensor, numpy_observations)
from graph.persistent_geometry.tensor_records import current_records_tensor, deduplicate_tensor, numpy_records
from graph.plan_a_geometry.core import deduplicate, key
from graph.plan_a_bc.geometry import CurrentGeometry
from visual.runtime.graph_perception import make_packet_cuda


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


def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def verify_public_graph(cameras,cm,data,assets,schema,reference):
    old_core=module_from_file('_records_reference_core',reference/'graph/plan_a_geometry/core.py')
    old_geo=module_from_file('graph.plan_a_bc._records_reference_geometry',reference/'graph/plan_a_bc/geometry.py')
    old_geo.PlanA=old_core.PlanA
    results={}
    for frame in ('ee','world'):
        old=old_geo.CurrentGeometry(assets,schema,coordinate_frame=frame)
        new=CurrentGeometry(assets,schema,coordinate_frame=frame)
        new.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
        surfaces=new.fk.surfaces(data['robot76'])
        for g in (old,new):
            g.system.prepare=lambda *args:dict(conditions={'C3':cm},gate=dict(accepted=True))
        expected,er,eg=old.build(cameras,data,surfaces,cm)
        actual,ar,ag=new.build(cameras,data,surfaces,cm)
        same(ar,er);same(ag,eg)
        assert actual.packet['records_dedup_backend']=='torch_cuda'
        assert actual.packet['node_ids']==expected.packet['node_ids']
        errors={}
        for field in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            a=getattr(actual,field);b=getattr(expected,field)
            assert torch.equal(a,b),field
            errors[field]=float((a-b).abs().max()) if a.numel() else 0.
        results[frame]=dict(all_graph_tensors_exact=True,all_record_fields_and_sources_exact=True,
            nodes=len(actual.node_type),edges=actual.edge_index.shape[1],max_abs_errors=errors)
    return results


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','reference-source','assets','schema'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('another CUDA process is active; do not stop it')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    def load(name):
        with np.load(a.inputs/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    s=load('hand_camera.npz');data=load('robot.npz');mask=load('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,s['rgb'],s['depth_raw_mm'],
        s['intrinsic_cv'],s['extrinsic_cv'],s['cam2world_gl'],data['tcp_pose']).validate()
    assert hand.depth.shape==(512,512)
    cameras=(replace(hand,camera='base_camera'),hand)
    cm=dict(base_camera=np.zeros_like(mask),hand_camera=mask)
    frames=('ee','world');gpu={};cpu={};before={}
    for frame in frames:
        views=prepare_tensor_views(cameras,cm,coordinate_frame=frame)
        gpu[frame]={};cpu[frame]={};before[frame]={}
        for c in cameras:
            v=views.by_name[c.camera]
            packet=make_packet_cuda(c,v.mask,depth_cuda=v.depth.to(torch.float32),
                intrinsic_cuda=v.intrinsic,transform_cuda=v.frame_from_camera,coordinate_frame=frame)
            obs=observations_from_tensor_packet(packet,c)
            obs,_=consistent_views_tensor(obs,views)
            gpu[frame][c.camera]=obs
            cpu[frame][c.camera]=numpy_observations(obs,include_structural=True)
            before[frame][c.camera]=[asdict(o) for o in cpu[frame][c.camera][0]]
    torch.cuda.synchronize()
    def cpu_rows(inputs,frame):
        sf=[]
        for c in cameras:
            obs,flags=inputs[c.camera]
            structural_pixels={int(p) for o,f in zip(obs,flags) for p in o.pixels[f]}
            rows=current_records(obs,fuse_views=False,coordinate_frame=frame)
            for r in rows:r['structural']=key(r)[3] in structural_pixels
            sf.extend(rows)
        return deduplicate(sf,.003)
    def run(frame,mode):
        if mode=='cpu':return cpu_rows(cpu[frame],frame)
        if mode=='legacy_from_gpu':
            host={c.camera:numpy_observations(gpu[frame][c.camera],include_structural=True) for c in cameras}
            return cpu_rows(host,frame)
        obs=[o for c in cameras for o in gpu[frame][c.camera]]
        result,audit=deduplicate_tensor(current_records_tensor(obs,coordinate_frame=frame),.003)
        return (numpy_records(result) if mode=='gpu_to_cpu' else result),audit
    checks={}
    for frame in frames:
        expected,trace=run(frame,'cpu')
        for mode in ('legacy_from_gpu','gpu_to_cpu'):
            rows,audit=run(frame,mode);same(rows,expected);same(audit,trace)
        with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('array readback forbidden')), \
             patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy conversion forbidden')), \
             patch.object(torch.Tensor,'tolist',side_effect=AssertionError('host point lists forbidden')):
            result,audit=run(frame,'gpu')
            assert result.xyz.is_cuda and result.source_pixels.is_cuda
        same(numpy_records(result),expected);same(audit,trace)
        checks[frame]=dict(all_record_fields_dtypes_values_order_provenance_and_audit_exact=True,
            resident_no_cpu_numpy_tolist=True,audit=trace,
            provenance_sources=int(result.source_pixels.numel()))
    modes=('cpu','gpu','gpu_to_cpu','legacy_from_gpu')
    config=dict(task='10i_records_dedup_cuda_20261103',warmups=a.warmups,repeats=a.repeats,
        modes=list(modes),frames=list(frames),actual_sensor_views=1,empty_cloned_base=True,
        robot_filter_and_gate_excluded=True,final_fps_and_edges_excluded=True,
        input_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=a.output/'experiments',config=config)
    try:
        combinations=[(f,m) for f in frames for m in modes]
        for _ in range(a.warmups):
            for f,m in combinations:run(f,m)
        torch.cuda.synchronize()
        times={f+'__'+m:[] for f,m in combinations}
        for i in range(a.repeats):
            order=combinations[i%len(combinations):]+combinations[:i%len(combinations)]
            if i%2:order=order[::-1]
            for f,m in order:
                torch.cuda.synchronize();start=perf_counter();output=run(f,m)
                if m!='cpu':torch.cuda.synchronize()
                end=perf_counter();times[f+'__'+m].append((end-start)*1000)
                del output
        for frame in frames:
            for c in cameras:same([asdict(o) for o in cpu[frame][c.camera][0]],before[frame][c.camera])
            result,_=run(frame,'gpu')
            host=numpy_records(result)
            (a.output/f'{frame}_provenance.json').write_text(json.dumps([
                dict(id=r['persistent_node_id'],sources=r['source_observation_ids']) for r in host],indent=2)+'\n')
        data.update(episode=np.array(hand.episode),timestep=np.int64(hand.timestep),timestamp=np.float64(hand.timestamp))
        graph=verify_public_graph(cameras,cm,data,a.assets,json.loads(a.schema.read_text())['schema'],a.reference_source)
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],config=config,
            timings={k:stats(v) for k,v in times.items()},verification=checks,public_graph=graph,
            cpu_observations_unchanged=True,hand_points=1105,base_points=0,box_points=0,
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
            gpu_endpoints_synchronized=True,garbage_collection_default=True,cpu_affinity_fixed=False)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(a.output/'timings.npz',**{k:np.asarray(v) for k,v in times.items()})
        logger.record(summary=summary)
        for path in a.output.glob('*'):
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
