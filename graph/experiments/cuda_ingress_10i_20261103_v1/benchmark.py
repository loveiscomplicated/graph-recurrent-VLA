"""Saved-input paired validation and synchronized wall timing; no inference."""
import argparse,hashlib,json,os,platform,sys,time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.persistent_geometry.contracts import CameraObservation
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.graph_boundary import GpuMaskBoundary
from visual.runtime.tensor_packet import numpy_packet
from utils.experiment_logger import ExperimentLogger

FIELDS=('node_input','node_type','edge_index','edge_attr','edge_type')
ROOT=Path(os.environ.get('TASK10I_ROOT','/workspace/task10i_resolution_perception_graph_20261103_v1'))
_BASE_FILTER=None


def geometry(backend):
    return CurrentGeometry('/workspace/task10g_f1_abc_20261031/full/inputs',
        json.loads(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json').read_text())['schema'],perception_backend=backend)


def load(scene,size,geo):
    spec=next(s for s in json.loads((ROOT/'protocol.json').read_text())['scenes'] if s['id']==scene)
    folder=ROOT/'sensors'/scene/str(size)
    with np.load(folder/'robot.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    cameras=[]
    for name in ('base_camera','hand_camera'):
        with np.load(folder/(name+'.npz'),allow_pickle=False) as z:
            cameras.append(CameraObservation(scene,int(spec['timestep']),name,float(spec['timestep']/20),
                *[z[k].copy() for k in ('rgb','depth_raw_mm','intrinsic_cv','extrinsic_cv','cam2world_gl')],data['tcp_pose'].copy()).validate())
    surfaces=geo.fk.surfaces(data['robot76']);geo.fk.verify_robot(data['robot76'],data['robot_world_geometry'],data['tcp_pose'])
    data.update(episode=np.array(scene),timestep=np.int64(spec['timestep']),timestamp=np.float64(spec['timestep']/20),
        qpos=data['robot76'][:9],robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    helper=Path(__file__).resolve().parents[1]/'resolution_perception_graph_20261103_v1'
    sys.path.insert(0,str(helper))
    from graph_worker import masks_for
    cpu_masks=masks_for(spec,size,0,'actual');masks={k:torch.as_tensor(v,device='cuda') for k,v in cpu_masks.items()}
    load.identity=dict(sensor_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npz')},
        mask_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in cpu_masks.items()},
        mask_pixels={k:[int(x.sum()) for x in v] for k,v in cpu_masks.items()})
    return cameras,data,surfaces,masks


def baseline_context():
    global _BASE_FILTER
    if _BASE_FILTER is None:
        from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
        _BASE_FILTER=CudaRobotFilter()
    helper=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(helper/'graph_latency_10i_20261103_v2'),str(helper/'resolution_perception_graph_20261103_v1')]
    from candidates import Candidate
    s=ExitStack();s.enter_context(Candidate(9));s.boundary=s.enter_context(GpuMaskBoundary(_BASE_FILTER));return s


def compare(actual,expected,path='root'):
    if isinstance(expected,torch.Tensor):
        return compare(actual.detach().cpu().numpy(),expected.detach().cpu().numpy(),path)
    if isinstance(expected,np.ndarray):
        if expected.dtype.kind=='f':
            if path=='node_input':
                np.testing.assert_allclose(actual[:,:3],expected[:,:3],rtol=0,atol=1e-6,err_msg=path)
                np.testing.assert_allclose(actual[:,3:8],expected[:,3:8],rtol=0,atol=2e-5,err_msg=path)
                np.testing.assert_array_equal(actual[:,8:],expected[:,8:],err_msg=path)
            elif any(s in path for s in ('geometric_features','last_f5')):
                from visual.runtime.vision_packet import float32_tolerance
                a=np.asarray(actual);b=expected
                np.testing.assert_array_equal(np.isnan(a),np.isnan(b),err_msg=path)
                if not np.all(np.isnan(b[...,:3])| (np.abs(a[...,:3]-b[...,:3])<=float32_tolerance(b[...,:3],normal=True))):raise AssertionError(path)
                np.testing.assert_array_equal(a[...,3:],b[...,3:],err_msg=path)
            elif 'normal' in path:
                from visual.runtime.vision_packet import float32_tolerance
                a=np.asarray(actual);np.testing.assert_array_equal(np.isnan(a),np.isnan(expected),err_msg=path)
                if not np.all(np.isnan(expected)|(np.abs(a-expected)<=float32_tolerance(expected,normal=True))):raise AssertionError(path)
            elif path.endswith('/points') or path=='edge_attr':np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6,equal_nan=True,err_msg=path)
            elif path.endswith(('world_xyz','object_local_xyz')):np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12,equal_nan=True,err_msg=path)
            else:np.testing.assert_array_equal(actual,expected,err_msg=path)
        else:np.testing.assert_array_equal(actual,expected,err_msg=path)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise AssertionError((path,actual.keys(),expected.keys()))
        for k,v in expected.items():compare(actual[k],v,path+'/'+k)
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise AssertionError((path,len(actual),len(expected)))
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i))
    elif isinstance(expected,(float,np.floating)):
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12,equal_nan=True,err_msg=path)
    elif actual!=expected:raise AssertionError((path,actual,expected))


def validate(ga,gb,inputs):
    with baseline_context() as context:
        reference=ga.build(*inputs)
        clean_reference={k:v.clone() for k,v in context.boundary.clean_cuda.items()}
    ga.reset();actual=gb.build(*inputs);gb.reset()
    for name in FIELDS:compare(getattr(actual[0],name),getattr(reference[0],name),name)
    compare(actual[1],reference[1],'records');compare(actual[2],reference[2],'gate')
    compare({k:v for k,v in actual[0].packet.items() if k!='perception_backend'},
        {k:v for k,v in reference[0].packet.items() if k!='perception_backend'},'Graph.packet')
    from visual.runtime.graph_perception import make_packet
    from graph.vision_input import validate_packet
    resident=gb.tensor_geometry.last
    compare(resident['clean'],clean_reference,'clean_masks')
    serial=gb.tensor_geometry.ingress.robot_filter.prepare_surfaces(resident['frame'].surfaces,resident['frame'].cameras)
    for (_,a),(_,b) in zip(serial.hulls,resident['frame'].prepared.hulls):np.testing.assert_array_equal(a,b)
    for c in inputs[0]:
        raw=numpy_packet(resident['packets'][c.camera]);validate_packet(raw)
        expected,_=make_packet(c,resident['conditions'][c.camera].cpu().numpy())
        compare(raw,expected,'packet/'+c.camera)
    return dict(nodes=len(actual[0].node_type),edges=actual[0].edge_index.shape[1],gate=actual[2]['accepted'],
        readback_bytes=resident['selected_readback_bytes'],max_node_abs=float((actual[0].node_input.cpu()-reference[0].node_input).abs().max()))


def timed(geo,inputs):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();start=time.perf_counter();out=geo.build(*inputs)
    for name in FIELDS:setattr(out[0],name,getattr(out[0],name).to('cuda'))
    torch.cuda.synchronize()
    ms=(time.perf_counter()-start)*1000;geo.reset()
    geo.measured_memory=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    return ms


def main(a):
    torch.set_num_threads(a.threads);torch.manual_seed(42);np.random.seed(42)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ga=geometry('triton_fused');gb=geometry('cuda_resident');rows=[];times=[]
    scenes=([s['id'] for s in json.loads((ROOT/'protocol.json').read_text())['scenes']]
        if a.scenes=='all' else a.scenes.split(','))
    config=dict(args=vars(a),python=sys.version,host=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        gpu=torch.cuda.get_device_name(),boundary='CPU RGB/raw signed-int16 depth/current calibration/FK surfaces + raw CUDA masks ready -> final CUDA Graph ready; all internal staging/readback included; excludes SAM2/grounding/policy/FK/simulator/I/O',
        baseline='current triton_fused + existing GPU mask boundary + Stage 9 including ownership-aware SF-only records and exact native small CPU operations',
        affinity=sorted(os.sched_getaffinity(0)),cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        cpu=next(x for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        threads=torch.get_num_threads(),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        frozen_protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[3])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parent.glob('*.py')})
    with ExperimentLogger(team='integration',model='task10i_cuda_ingress',job_type='evaluate',seed=42,root=out/'experiments',config=config) as log:
        for scene in scenes:
            for size in a.sizes:
                inputs=load(scene,size,ga)
                row=dict(scene=scene,size=size,input_identity=load.identity,**validate(ga,gb,inputs));rows.append(row);print(json.dumps(row),flush=True)
                if a.repeats and (size==512 or a.time_all_sizes) and scene==scenes[0]:
                    for _ in range(3):
                        with baseline_context():timed(ga,inputs)
                        timed(gb,inputs)
                    for i in range(a.repeats):
                        pair={}
                        for variant in (('baseline','resident') if i%2==0 else ('resident','baseline')):
                            if variant=='baseline':
                                with baseline_context():pair[variant]=timed(ga,inputs)
                            else:pair[variant]=timed(gb,inputs)
                        times.append(dict(scene=scene,size=size,repeat=i,**pair,baseline_memory=ga.measured_memory,resident_memory=gb.measured_memory));print(json.dumps(times[-1]),flush=True)
        repo=Path(__file__).resolve().parents[3]
        config['loaded_sources_sha256']={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
            for mod in list(sys.modules.values()) if (name:=getattr(mod,'__file__',None)) and
            (p:=Path(name)).is_file() and p.suffix=='.py' and p.is_relative_to(repo)}
        config['peak_rss_kib']=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))
        path=out/'results.json';path.write_text(json.dumps(dict(status='PASS',config=config,validation=rows,times=times),indent=2))
        log.link_output(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scenes',default='10i_s1_0924_t080')
    p.add_argument('--sizes',type=int,nargs='+',default=[512]);p.add_argument('--repeats',type=int,default=15)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--time-all-sizes',action='store_true');main(p.parse_args())
