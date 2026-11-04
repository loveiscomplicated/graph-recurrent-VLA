"""Ready CUDA sampling output -> existing CPU EE packet/receiver return."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.vision_input import receive_vision_packet
from visual.runtime.cuda_sampling import sample_cuda,numpy_selection,_map_tensors
from visual.runtime.graph_perception import process_cuda,_selection_packet,make_packet
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.vision_packet import complete_selected_vision_packet


def load_case(inputs):
    with np.load(inputs/'hand_camera.npz',allow_pickle=False) as a:s={k:a[k].copy() for k in a.files}
    with np.load(inputs/'robot.npz',allow_pickle=False) as a:tcp=a['tcp_pose'].copy()
    with np.load(inputs/'masks.npz',allow_pickle=False) as a:masks=a['masks'].copy()
    c=CameraObservation('10i_s1_0844',80,'hand_camera',4.,s['rgb'],s['depth_raw_mm'],
        s['intrinsic_cv'],s['extrinsic_cv'],s['cam2world_gl'],tcp)
    c.validate()
    f=TrackedRGBDFrame(c.timestep,c.camera,c.rgb,c.depth.astype(np.float32),c.intrinsic,
        c.world_from_camera,c.tcp_pose,dict(enumerate(masks)),c.timestep,c.timestamp)
    return c,f,masks


def assert_same(a,b):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b,(a,b)


def flat_tensors(tree):
    if isinstance(tree,dict):return [t for value in tree.values() for t in flat_tensors(value)]
    return [tree]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=20);parser.add_argument('--repeats',type=int,default=100)
    args=parser.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('another CUDA compute process is active; do not stop it')
    args.output.mkdir(parents=True,exist_ok=False)
    camera,frame,masks=load_case(args.inputs)
    d=torch.as_tensor(frame.depth,device='cuda');m=torch.as_tensor(masks,device='cuda');k=torch.as_tensor(frame.intrinsic,device='cuda')
    projected,features=process_cuda(d,m,k)
    gpu=sample_cuda(projected,features,d,m,k);torch.cuda.synchronize()
    leaves=flat_tensors(gpu);assert all(t.device.type=='cuda' for t in leaves)
    def transfer():return _map_tensors(gpu,lambda t:t.to(device='cpu'))
    def packet_return():
        cpu=numpy_selection(transfer())
        selected=_selection_packet(frame,camera.episode,cpu)
        return complete_selected_vision_packet(selected,frame,cpu['metadata'],cpu['valid'])
    def consumer_return():
        packet=packet_return();receive_vision_packet(packet);return packet
    functions=dict(ee_packet_return=packet_return,existing_make_packet_return=consumer_return)
    config=dict(task='10i_packet_cpu_timing_58752018_20261103_v1',warmups=args.warmups,repeats=args.repeats,
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='ready and synchronized CUDA sampling output -> existing CPU EE packet / receiver return',
        sampling_excluded=True,robot_filter_excluded=True)
    logger=ExperimentLogger(team='graph',job_type='benchmark',config=config,seed=42,root=args.output/'experiments')
    try:
        for _ in range(args.warmups):
            for fn in functions.values():result=fn()
        times={name:[] for name in functions};names=list(functions)
        for i in range(args.repeats):
            for name in (names if i%2==0 else names[::-1]):
                torch.cuda.synchronize();start=perf_counter();result=functions[name]()
                times[name].append((perf_counter()-start)*1000)
        phases={name:[] for name in ('gpu_to_cpu','numpy_and_selection_packet','checks_transform_and_packet','receiver_validation_copy')}
        for _ in range(args.repeats):
            torch.cuda.synchronize();t0=perf_counter();cpu_tensors=transfer();t1=perf_counter()
            cpu=numpy_selection(cpu_tensors);selected=_selection_packet(frame,camera.episode,cpu);t2=perf_counter()
            packet=complete_selected_vision_packet(selected,frame,cpu['metadata'],cpu['valid']);t3=perf_counter()
            receive_vision_packet(packet);t4=perf_counter()
            for name,elapsed in zip(phases,(t1-t0,t2-t1,t3-t2,t4-t3)):phases[name].append(elapsed*1000)
        # Actual public path parity and output export are outside all timings.
        expected,_=make_packet(camera,masks)
        assert_same(packet_return(),expected);assert_same(consumer_return(),expected)
        assert expected['coordinate_frame']=='EE_metres'
        stats=lambda t:dict(median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),min_ms=min(t),max_ms=max(t))
        summary=dict(run_id=logger.run_id,resolution=list(frame.depth.shape),camera=camera.camera,
            selected_points=len(expected['points']),gpu_tensor_fields=len(leaves),
            logical_transfer_bytes=sum(t.numel()*t.element_size() for t in leaves),
            timings={name:stats(t) for name,t in times.items()},diagnostic_phases={name:stats(t) for name,t in phases.items()},
            coordinate_frame=expected['coordinate_frame'],all_packet_fields_exact=True)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(args.output/'timings.npz',**{name:np.array(t) for name,t in times.items()})
        np.savez(args.output/'phase_timings.npz',**{name:np.array(t) for name,t in phases.items()})
        np.savez_compressed(args.output/'coordinates.npz',points_ee=expected['points'],points_camera=expected['camera_points'],
            source=expected['provenance']['source_index'],T_EE_from_camera=expected['T_EE_from_camera'],features=expected['geometric_features'])
        logger.record(summary=summary)
        for p in args.output.glob('*.*'):logger.link_output(p)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:logger.finish(failed=True);raise


if __name__=='__main__':main()
