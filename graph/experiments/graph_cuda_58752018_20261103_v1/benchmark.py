"""Same 512 input, CPU sparse-depth reference versus resident CUDA graph."""
import argparse
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
from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
from graph.persistent_geometry.tensor_graph import FIELDS,numpy_graph
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.experiments.graph_build_timing_58752018_20261103_v1.benchmark import stats,same


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','output','assets','schema','baseline'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--warmups',type=int,default=20);p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('another CUDA process active')
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=False)
    def load(name):
        with np.load(a.inputs/name,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    s=load('hand_camera.npz');data=load('robot.npz');mask=load('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,s['rgb'],s['depth_raw_mm'],s['intrinsic_cv'],s['extrinsic_cv'],s['cam2world_gl'],data['tcp_pose']).validate()
    cameras=(replace(hand,camera='base_camera'),hand);masks=dict(base_camera=np.zeros_like(mask),hand_camera=mask)
    data.update(episode=np.array(hand.episode),timestep=np.int64(80),timestamp=np.float64(4.))
    schema=json.loads(a.schema.read_text())['schema'];frames=('ee','world');gate={'accepted':True}
    baseline=json.loads((a.baseline/'summary.json').read_text())
    with np.load(a.baseline/'graph_reference.npz',allow_pickle=False) as z:reference={k:z[k].copy() for k in z.files}
    geometries={};records={};cpu={};views={};verification={};outputs={}
    def check(graph,frame):
        out=numpy_graph(graph);errors={};exact={}
        for name in FIELDS:
            actual=getattr(out,name).numpy();expected=reference[frame+'__'+name]
            if name in ('node_input','edge_attr'):
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                errors[name]=float(np.max(np.abs(actual-expected),initial=0))
            else:np.testing.assert_array_equal(actual,expected)
            exact[name]=bool(np.array_equal(actual,expected))
        packet=json.loads((a.baseline/f'{frame}_graph_packet.json').read_text())
        assert out.packet['node_ids']==packet['node_ids'] and out.packet['surface_edge_audit']==packet['surface_edge_audit']
        return dict(max_absolute_error=errors,bit_exact=exact,nodes=len(out.node_type),edges=out.edge_index.shape[1],audit=out.packet['surface_edge_audit'])
    for frame in frames:
        g=CurrentGeometry(a.assets,schema,coordinate_frame=frame,robot_filter_backend='cpu');geometries[frame]=g
        g.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose']);surfaces=g.fk.surfaces(data['robot76'])
        prepared={'conditions':{'C3':masks},'gate':gate};g.system.prepare=lambda *args:prepared
        views[frame]=prepare_tensor_views(cameras,masks,coordinate_frame=frame)
        pools,_,_,_=g.system.represent(cameras,prepared,'C3',('SF',),tensor_records=True,views=views[frame])
        records[frame]=budget_tensor(pools['SF'],128);cpu[frame]=numpy_records(records[frame])
        public,returned,_=g.build(cameras,data,surfaces,masks)
        assert public.node_input.is_cuda and returned.xyz.is_cuda
        same(numpy_records(returned),cpu[frame]);verification[frame]=check(public,frame)
        verification[frame]['public_build_cuda_graph_and_records']=True
    def run(frame,mode):
        g=geometries[frame]
        if mode=='cpu':r=cpu[frame];graph=g.system.graph(r,data,cameras)
        else:
            r=records[frame];v=views[frame] if mode=='cuda' else prepare_tensor_views(cameras,masks,coordinate_frame=frame)
            graph=g.system.graph(r,data,cameras,views=v)
        return g._finish(graph,r,gate,data)
    for frame in frames:
        def forbidden(*args,**kwargs):raise AssertionError('host point-array readback forbidden')
        with patch.object(torch.Tensor,'cpu',forbidden),patch.object(torch.Tensor,'numpy',forbidden),patch.object(torch.Tensor,'tolist',forbidden),patch('graph.plan_a_geometry.core.build_graph',side_effect=forbidden):
            result=run(frame,'cuda');torch.cuda.synchronize()
        verification[frame]['no_host_array_readback_or_cpu_graph']=True
        verification[frame]['endpoint']=check(result[0],frame)
    modes=('cpu','cuda','cuda_with_views')
    config=dict(task='10i_graph_cuda_20261103',warmups=a.warmups,repeats=a.repeats,
        boundary='final 128/object records -> PlanA.graph -> CurrentGeometry._finish including validation',
        cuda='resident records and existing TensorViews -> CUDA Graph and records; completion synchronized inside timer',
        cuda_with_views='also includes fresh raw depth/mask/calibration upload and TensorViews preparation',
        cpu='ready NumPy records and raw camera data -> CPU Graph; previous sparse-depth implementation',
        actual_sensor_views=1,empty_cloned_base=True,box_points=0,upstream_perception_filter_gate_and_FPS_excluded=True,
        baseline_run_id=baseline['run_id'],prior_baseline_timings=baseline['timings'],
        input_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.inputs.glob('*.npz')},
        source_manifest_sha256=hashlib.sha256((Path(__file__).resolve().parents[3]/'source_manifest.json').read_bytes()).hexdigest() if (Path(__file__).resolve().parents[3]/'source_manifest.json').exists() else None,
        snapshot_identity=json.loads((Path(__file__).resolve().parents[3]/'execution_identity.json').read_text()) if (Path(__file__).resolve().parents[3]/'execution_identity.json').exists() else None,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert config['input_sha256']==baseline['config']['input_sha256']
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=a.output/'experiments',config=config)
    try:
        for _ in range(a.warmups):
            for frame in frames:
                for mode in modes:out=run(frame,mode);torch.cuda.synchronize();del out
        times={f+'__'+m:[] for f in frames for m in modes}
        for i in range(a.repeats):
            schedule=[(f,m) for f in (frames if i%2==0 else frames[::-1]) for m in (modes[i%3:]+modes[:i%3])]
            for frame,mode in schedule:
                torch.cuda.synchronize();start=perf_counter();out=run(frame,mode);torch.cuda.synchronize()
                times[frame+'__'+mode].append((perf_counter()-start)*1000);del out
            if (i+1)%20==0:print('samples',i+1,flush=True)
        np.savez(a.output/'timings.npz',**times)  # Preserve raw measurements before metadata/export.
        for frame in frames:
            graph=run(frame,'cuda')[0];check(graph,frame);saved=numpy_graph(graph)
            for name in FIELDS:outputs[frame+'__'+name]=getattr(saved,name).numpy()
            (a.output/f'{frame}_graph_packet.json').write_text(json.dumps(saved.packet,indent=2)+'\n')
        measured={key:stats(value) for key,value in times.items()}
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],config=config,timings=measured,
            median_speedup={f:measured[f+'__cpu']['median_ms']/measured[f+'__cuda']['median_ms'] for f in frames},
            verification=verification,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),threads=1,robot_filter_or_gate_modified=False)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez(a.output/'timings.npz',**times);np.savez(a.output/'graphs.npz',**outputs)
        logger.record(summary=summary)
        for path in a.output.iterdir():
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
