"""Real frozen SAM2, serial vs overlapped robot prep, saved RGB-D only."""
import argparse,gc,hashlib,importlib.metadata,json,os,platform,statistics,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from graph.plan_a_bc.geometry import RobotFK,CurrentGeometry
from graph.persistent_geometry.contracts import CameraObservation
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.preparation import RobotFilterPreparation
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.sam2_bridge import image_masks_cuda
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from visual.runtime.tracking import SAM2Tracker
from utils.experiment_logger import ExperimentLogger

ROOT=Path('/workspace/task10i_resolution_perception_graph_20261103_v1')
CHECKPOINT=Path('/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt')
CONFIG='configs/sam2.1/sam2.1_hiera_s.yaml'
NAMES=('base_camera','hand_camera')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2))


class Inputs:
    def __init__(self):
        self.protocol=json.loads((ROOT/'protocol.json').read_text())
        self.spec={s['id']:s for s in self.protocol['scenes']}
        self.predictions={r['id']:r for r in json.loads((ROOT/'api_predictions.json').read_text())['rows']}
        schema=json.loads(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json').read_text())['schema']
        self.fk=RobotFK('/workspace/task10g_f1_abc_20261031/full/inputs',schema);self.hashes={}

    def load(self,scene):
        s=self.spec[scene];folder=ROOT/'sensors'/scene/'512';path=folder/'robot.npz'
        self.hashes[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            robot=z['robot76'].copy();tcp=z['tcp_pose'].copy();pads=z['robot_world_geometry'].copy()
        self.fk.verify_robot(robot,pads,tcp);surfaces=self.fk.surfaces(robot);cameras=[]
        for name in NAMES:
            path=folder/(name+'.npz');self.hashes[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as z:
                cameras.append(CameraObservation(scene,int(s['timestep']),name,float(s['timestep']/20),
                    *[z[k].copy() for k in ('rgb','depth_raw_mm','intrinsic_cv','extrinsic_cv','cam2world_gl')],tcp.copy()).validate())
        return tuple(cameras),surfaces

    def boxes(self,scene):
        return {name:[self.predictions[f'r0__{scene}__512__{name}']['parsed'][o]['pixels'] for o in ('peg','box')] for name in NAMES}


def snapshot(raw,clean):
    result={}
    for name in NAMES:
        a=raw[name].detach().cpu().numpy();b=clean[name].detach().cpu().numpy()
        result[name]=dict(raw=a,clean=b,raw_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            clean_sha256=hashlib.sha256(b.tobytes()).hexdigest(),raw_pixels=a.sum((1,2)).tolist(),clean_pixels=b.sum((1,2)).tolist())
    return result


def check(a,b):
    for name in NAMES:
        for key in ('raw','clean'):np.testing.assert_array_equal(a[name][key],b[name][key],err_msg=name+'/'+key)


def default_smoke(inputs,scene,cameras,surfaces,produce,reference):
    """Real default entrypoint to its clean-mask handoff; Graph excluded."""
    with np.load(ROOT/'sensors'/scene/'512'/'robot.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    c=cameras[0]
    data.update(episode=np.array(c.episode),timestep=np.int64(c.timestep),timestamp=np.float64(c.timestamp),
        qpos=data['robot76'][:9],robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    runtime=CurrentGeometry('/workspace/task10g_f1_abc_20261031/full/inputs',inputs.fk.schema)
    def handoff(cameras,data,surfaces,raw,clean_masks=None):
        assert clean_masks is not None
        for name in NAMES:
            np.testing.assert_array_equal(raw[name],reference[name]['raw'])
            np.testing.assert_array_equal(clean_masks[name],reference[name]['clean'])
        return dict(raw_clean_bitwise_exact=True)
    runtime._build_prepared=handoff
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=runtime.build(cameras,data,surfaces,produce)
        assert runtime.robot_filter_backend=='cuda' and runtime.last_robot_schedule=='before_vision'
        return dict(**result,backend=runtime.robot_filter_backend,schedule=runtime.last_robot_schedule,
            real_sam2=True,default_constructor=True,stopped_before_gate_graph=True)
    finally:runtime.close()


@torch.inference_mode()
def execute(prep,variant,cameras,surfaces,produce):
    """Only schedule differs. No synchronize between SAM2 and CPU preparation."""
    job=None
    try:
        if variant=='parallel':job=prep.submit(cameras,surfaces)
        raw=produce()
        if variant=='legacy_serial':return raw,prep.robot_filter.filter_cameras(cameras,surfaces,raw)
        if variant=='serial':job=prep.submit(cameras,surfaces)
        clean=prep.filter(job,raw)
        return raw,clean
    finally:
        if job is not None:prep.discard(job)


def measure(prep,variant,cameras,surfaces,produce):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):raw,clean=execute(prep,variant,cameras,surfaces,produce)
    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
    memory=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
    return ms,snapshot(raw,clean),memory


class TracedFilter(CudaRobotFilter):
    def prepare_surfaces(self,surfaces,cameras=(),**kwargs):
        start=time.perf_counter_ns();result=super().prepare_surfaces(surfaces,cameras,**kwargs)
        self.trace=dict(start=start,end=time.perf_counter_ns());return result


def trace_call(cameras,surfaces,produce):
    filt=TracedFilter()
    with RobotFilterPreparation(filt) as prep:
        torch.cuda.synchronize();begin=time.perf_counter_ns();job=prep.submit(cameras,surfaces)
        submitted=time.perf_counter_ns();sam_start=time.perf_counter_ns()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=produce()
        sam_end=time.perf_counter_ns();wait_start=time.perf_counter_ns();job.future.result();wait_end=time.perf_counter_ns()
        prep.filter(job,raw);torch.cuda.synchronize();end=time.perf_counter_ns();p=filt.trace
        return dict(submit_snapshot_ms=(submitted-begin)/1e6,prepare_cpu_ms=(p['end']-p['start'])/1e6,
            sam2_host_call_ms=(sam_end-sam_start)/1e6,prepare_wait_after_sam2_ms=(wait_end-wait_start)/1e6,
            cpu_prepare_overlap_with_sam2_call_ms=max(0,min(p['end'],sam_end)-max(p['start'],sam_start))/1e6,
            prepare_done_before_sam2_return=p['end']<=sam_end,total_ms=(end-begin)/1e6,
            timing_domain='CPU wall; detailed separate run, not added to primary medians')


def static_run(inputs,out,repeats):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    scene='10i_s1_0921_t080';cameras,surfaces=inputs.load(scene);boxes=inputs.boxes(scene)
    torch.cuda.synchronize();start=time.perf_counter();model=build_sam2(CONFIG,str(CHECKPOINT),device='cuda');model.eval()
    predictor=SAM2ImagePredictor(model);torch.cuda.synchronize();load_ms=(time.perf_counter()-start)*1000
    def produce():return {c.camera:image_masks_cuda(predictor,c.rgb,boxes[c.camera])[0] for c in cameras}
    preps={v:RobotFilterPreparation(CudaRobotFilter()) for v in ('legacy_serial','serial','parallel')};rows=[]
    try:
        reference=None
        for _ in range(3):
            for v in preps:
                _,value,_=measure(preps[v],v,cameras,surfaces,produce)
                if reference is None:reference=value
                else:check(value,reference)
        for i in range(repeats):
            row=dict(repeat=i,ms={},memory={});values={}
            order=list(preps);order=order[i%3:]+order[:i%3]
            if i%2:order=order[::-1]
            for v in order:
                row['ms'][v],values[v],row['memory'][v]=measure(preps[v],v,cameras,surfaces,produce)
                check(values[v],reference)
            check(values['serial'],values['parallel']);check(values['legacy_serial'],values['parallel']);row['exact']=True
            row['masks']={n:{k:x for k,x in values['parallel'][n].items() if k not in ('raw','clean')} for n in NAMES}
            rows.append(row);print('STATIC',json.dumps(row['ms']),flush=True)
        traces=[trace_call(cameras,surfaces,produce) for _ in range(3)]
        default_result=default_smoke(inputs,scene,cameras,surfaces,produce,reference)
        write(out/'static.json',dict(status='PASS',scene=scene,boxes=boxes,load_ms=load_ms,internal_image_size=model.image_size,
            rows=rows,traces=traces,default_runtime_smoke=default_result))
    finally:
        for p in preps.values():p.close()
    del predictor,model;gc.collect();torch.cuda.empty_cache()


def initializer(boxes):
    return {obj:dict(box=box,frame=0 if box is not None else None,score=1. if box is not None else None)
        for obj,box in zip(('peg','box'),boxes)}


def video_run(inputs,out):
    from sam2.build_sam import build_sam2_video_predictor
    parent='10i_s1_0531';scenes=sorted([s for s in inputs.protocol['scenes'] if s['cohort']=='tracking' and s['parent']==parent],key=lambda s:s['timestep'])
    assert [s['timestep'] for s in scenes]==list(range(21))
    loaded=[inputs.load(s['id']) for s in scenes];boxes=inputs.boxes(scenes[0]['id'])
    torch.cuda.synchronize();start=time.perf_counter();model=build_sam2_video_predictor(CONFIG,str(CHECKPOINT),device='cuda');model.eval()
    torch.cuda.synchronize();load_ms=(time.perf_counter()-start)*1000;tracker=SAM2Tracker(predictor=model)
    preps={v:RobotFilterPreparation(CudaRobotFilter()) for v in ('legacy_serial','serial','parallel')}
    adapters={v:{n:OnlineVisionAdapter(None,tracker) for n in NAMES} for v in preps}
    for v in adapters:
        for a in adapters[v].values():a.reset(v)
    rows=[]
    try:
        for t,(cameras,surfaces) in enumerate(loaded):
            row=dict(timestep=t,ms={},memory={});values={}
            order=list(preps);order=order[t%3:]+order[:t%3]
            if t%2:order=order[::-1]
            for v in order:
                def produce():
                    result={}
                    for c in cameras:
                        # Recorded RGB-derived prompts only. No new DINO/API,
                        # no GT; default adapter's real JPEG/causal SAM2 remains.
                        with patch('graph.integration_baseline.online_vision_adapter.detect_initial',return_value=initializer(boxes[c.camera])):
                            result[c.camera]=adapters[v][c.camera].process(c.rgb,v,t,mask_device='cuda')[0]
                    return result
                row['ms'][v],values[v],row['memory'][v]=measure(preps[v],v,cameras,surfaces,produce)
            check(values['serial'],values['parallel']);check(values['legacy_serial'],values['parallel']);row['exact']=True
            row['masks']={n:{k:x for k,x in values['parallel'][n].items() if k not in ('raw','clean')} for n in NAMES}
            rows.append(row);print('VIDEO',t,json.dumps(row['ms']),flush=True)
        write(out/'video.json',dict(status='PASS',parent=parent,boxes=boxes,load_ms=load_ms,rows=rows,
            internal_image_size=model.image_size,
            aggregation='t0 initialization separate, t1..5 warm progression, t6..20 fifteen paired steady frames; not fifteen repetitions of an identical state',
            adapter='real default OnlineVisionAdapter; recorded prompts replace detection only; no future frames exposed'))
    finally:
        for p in preps.values():p.close()
        for group in adapters.values():
            for a in group.values():a.close()
    del model,tracker;gc.collect();torch.cuda.empty_cache()


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if sha(CHECKPOINT)!='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38':raise ValueError('checkpoint changed')
    inputs=Inputs();repo=Path(__file__).resolve().parents[3]
    config=dict(args=vars(args),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        python=sys.version,host=platform.platform(),threads=torch.get_num_threads(),affinity=sorted(os.sched_getaffinity(0)),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),model_config=CONFIG,precision='FP32 weights, bfloat16 autocast, TF32 enabled; eager unchanged model',
        boundary='two CPU 512 RGB-D observations/current calibration/precomputed FK + fixed recorded prompts -> two CUDA clean masks; CPU snapshots/submission, real SAM2 preprocessing/encoding/prompt/propagation and filtering included; CUDA synchronized at endpoints only',
        exclusions='separate FK, frozen input loading, model construction, DINO/API, gate/Shape-First/Graph/action/simulator; video includes default JPEG preprocessing/I/O',
        source_sha256={str(p.relative_to(repo)):sha(p) for p in [Path(__file__),repo/'graph/experiments/robot_filter_gpu_io_10i_20261103_v1/gpu_filter.py',repo/'graph/experiments/robot_filter_gpu_io_10i_20261103_v1/preparation.py',repo/'graph/experiments/robot_filter_gpu_io_10i_20261103_v1/sam2_bridge.py',repo/'graph/integration_baseline/online_vision_adapter.py',repo/'visual/runtime/tracking.py']})
    with ExperimentLogger(team='integration',model='task10i_real_sam2_overlap',job_type='evaluate',seed=42,root=out/'experiments',config=config) as log:
        write(out/'config.json',config)
        if args.mode in ('static','both'):static_run(inputs,out,args.repeats);log.link_output(out/'static.json')
        if args.mode in ('video','both'):video_run(inputs,out);log.link_output(out/'video.json')
        write(out/'input_hashes.json',inputs.hashes);log.link_output(out/'input_hashes.json')
        write(out/'complete.json',dict(status='PASS',modes=args.mode));log.link_output(out/'complete.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['static','video','both'],default='both');p.add_argument('--repeats',type=int,default=15);main(p.parse_args())
