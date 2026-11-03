"""Saved-frame equivalence and CPU preparation timing; no SAM2/Graph runs."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from graph.plan_a_bc.geometry import RobotFK
from graph.persistent_geometry.contracts import CameraObservation
from utils.experiment_logger import ExperimentLogger
from .gpu_filter import CudaRobotFilter
from .preparation import RobotFilterPreparation

ROOT=Path('/workspace/task10i_resolution_perception_graph_20261103_v1')


def main(output):
    torch.set_num_threads(4);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scene='10i_s1_0921_t080';spec=next(x for x in json.loads((ROOT/'protocol.json').read_text())['scenes'] if x['id']==scene)
    if spec['cohort']=='tracking':raise ValueError('this diagnostic expects the saved static fixture')
    schema=json.loads(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json').read_text())['schema']
    fk=RobotFK('/workspace/task10g_f1_abc_20261031/full/inputs',schema);filt=CudaRobotFilter();checks=[];timings=[]
    config=dict(scene=scene,sizes=[128,512,768,1024],repeat=0,gpu=torch.cuda.get_device_name(),host=platform.platform(),
        torch=torch.__version__,cuda=torch.version.cuda,threads=4,warmup=3,repeats=15,
        boundary='CPU prepare_surfaces only: world vertices, visible Qhull, both-camera projection bounds and packed metadata; excludes snapshots/submit, GPU filter and SAM2',
        no_sam2_inference=True,no_graph_execution=True,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    with ExperimentLogger(team='integration',model='task10i_sam2_cpu_prepare_overlap',job_type='evaluate',seed=42,
            root=output/'experiments',config=config) as log,RobotFilterPreparation(filt) as prep:
        for size in (128,512,768,1024):
            folder=ROOT/'sensors'/scene/str(size);files=[folder/'robot.npz']
            with np.load(files[0],allow_pickle=False) as z:robot=z['robot76'].copy();tcp=z['tcp_pose'].copy()
            surfaces=fk.surfaces(robot);cameras=[];raw={}
            for name in ('base_camera','hand_camera'):
                path=folder/(name+'.npz');files.append(path)
                with np.load(path,allow_pickle=False) as z:
                    cameras.append(CameraObservation(scene,int(spec['timestep']),name,float(spec['timestep']/20),
                        *[z[k].copy() for k in ('rgb','depth_raw_mm','intrinsic_cv','extrinsic_cv','cam2world_gl')],tcp.copy()))
                path=ROOT/'masks'/'actual'/'static'/f'r0__{scene}__{size}__{name}'/'masks.npz';files.append(path)
                with np.load(path,allow_pickle=False) as z:raw[name]=torch.as_tensor(z['masks'].copy(),device='cuda')
            reference=filt.filter_cameras(cameras,surfaces,raw,diagnostics=True)
            job=prep.submit(cameras,surfaces);prepared=job.future.result()
            serial=filt.prepare_surfaces(surfaces,cameras)
            for (_,a),(_,b) in zip(prepared.hulls,serial.hulls):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(prepared.packed,serial.packed)
            actual=prep.filter(job,raw,diagnostics=True)
            for name in raw:
                for key in reference[name]:
                    if not torch.equal(actual[name][key],reference[name][key]):raise AssertionError((size,name,key))
            checks.append(dict(size=size,cameras=2,all_filter_fields_bitwise_exact=True,hulls_metadata_bitwise_exact=True,
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
            if size==512:
                for _ in range(3):filt.prepare_surfaces(surfaces,cameras)
                for i in range(15):
                    start=time.perf_counter();filt.prepare_surfaces(surfaces,cameras)
                    timings.append(dict(repeat=i,prepare_ms=(time.perf_counter()-start)*1000))
        path=output/'results.json';path.write_text(json.dumps(dict(status='PASS',checks=checks,preparation_timings=timings,config=config),indent=2));log.link_output(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
