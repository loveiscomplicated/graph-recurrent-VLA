"""Ready selected packet -> GPU observations -> resident GPU view checking.

Current World/EE behavior is compared with the matching same-frame CPU path.
The historical0.541ms EE->World adapter is a different measurement boundary.
"""
import argparse
from dataclasses import asdict, replace
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
from graph.persistent_geometry.fusion import observations_from_packet, consistent_views
from graph.persistent_geometry.tensor_fusion import (prepare_tensor_views,
    observations_from_tensor_packet, consistent_views_tensor, numpy_observations)
from visual.runtime.graph_perception import make_packet_cuda
from visual.runtime.tensor_packet import numpy_packet


def same(a, b):
    if isinstance(b, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b, (a, b)


def stats(values):
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values,95)),
        min_ms=float(np.min(values)), max_ms=float(np.max(values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmups', type=int, default=20); parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018' and torch.cuda.is_available()
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active: raise RuntimeError('another CUDA process is active; do not stop it')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    def load(name):
        with np.load(args.inputs/name, allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    sensor=load('hand_camera.npz'); tcp=load('robot.npz')['tcp_pose']; masks=load('masks.npz')['masks']
    hand=CameraObservation('10i_s1_0844',80,'hand_camera',4.,sensor['rgb'],sensor['depth_raw_mm'],
        sensor['intrinsic_cv'],sensor['extrinsic_cv'],sensor['cam2world_gl'],tcp).validate()
    cameras=(replace(hand,camera='base_camera'),hand)
    cm=dict(base_camera=np.zeros_like(masks),hand_camera=masks)
    contexts={}; gpu_packets={}; cpu_packets={}
    for frame in ('world','ee'):
        ctx=prepare_tensor_views(cameras,cm,coordinate_frame=frame); contexts[frame]=ctx
        gpu_packets[frame]={};cpu_packets[frame]={}
        for c in cameras:
            v=ctx.by_name[c.camera]
            packet=make_packet_cuda(c,v.mask,depth_cuda=v.depth.to(torch.float32),intrinsic_cuda=v.intrinsic,
                transform_cuda=v.frame_from_camera,coordinate_frame=frame)
            gpu_packets[frame][c.camera]=packet
            cpu_packets[frame][c.camera]=numpy_packet(packet)
    torch.cuda.synchronize()
    modes=('cpu_observation_hand','gpu_observation_hand','cpu_observation_and_check',
        'gpu_observation_and_check_resident','gpu_observation_and_check_with_sensor_upload')
    def run(frame,mode):
        if mode=='cpu_observation_hand':
            return observations_from_packet(cpu_packets[frame][hand.camera],hand,coordinate_frame=frame)
        if mode=='gpu_observation_hand':
            return observations_from_tensor_packet(gpu_packets[frame][hand.camera],hand)
        ctx=(prepare_tensor_views(cameras,cm,coordinate_frame=frame)
            if mode.endswith('_with_sensor_upload') else contexts[frame])
        result=[]
        for c in cameras:
            if mode.startswith('cpu_'):
                obs=observations_from_packet(cpu_packets[frame][c.camera],c,coordinate_frame=frame)
                result.append(consistent_views(obs,cameras,cm))
            else:
                obs=observations_from_tensor_packet(gpu_packets[frame][c.camera],c)
                result.append(consistent_views_tensor(obs,ctx))
        return result
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,root=args.output/'experiments',
        config=dict(task='10i_observation_cuda_chain_20261103',warmups=args.warmups,repeats=args.repeats,
        modes=list(modes),actual_sensor_views=1,empty_cloned_base=True,robot_filter_excluded=True,
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_boundary='validated selected packets already on CPU / GPU respectively; sampling/packet production excluded'))
    try:
        for _ in range(args.warmups):
            for frame in contexts:
                for mode in modes: run(frame,mode)
        torch.cuda.synchronize()
        times={frame+'__'+mode:[] for frame in contexts for mode in modes}
        keys=list(times)
        for i in range(args.repeats):
            order=keys[i%len(keys):]+keys[:i%len(keys)]
            if i%2:order=order[::-1]
            for key in order:
                frame,mode=key.split('__')
                torch.cuda.synchronize();start=perf_counter();result=run(frame,mode)
                if mode.startswith('gpu_'):torch.cuda.synchronize()
                times[key].append((perf_counter()-start)*1000)
        counts={}
        for frame in contexts:
            reference=run(frame,'cpu_observation_hand')
            actual=run(frame,'gpu_observation_hand')
            same([asdict(o) for o in numpy_observations(actual)],[asdict(o) for o in reference])
            counts[frame]=[len(o.xyz) for o in reference]
            expected=run(frame,'cpu_observation_and_check')
            for mode in modes[3:]:
                output=run(frame,mode)
                for (selected,trace),(ref,audit) in zip(output,expected):
                    assert all(t.is_cuda for o in selected for t in
                        (o.xyz,o.normals,o.normal_valid,o.f5,o.feature_valid,o.pixels,o.camera_origin,o.structural))
                    same([asdict(o) for o in numpy_observations(selected)],[asdict(o) for o in ref])
                    assert [int(t['unknown_conflict'].item()) for t in trace]==[t['unknown_conflict'] for t in audit]
            for c in cameras:
                same(numpy_packet(gpu_packets[frame][c.camera]),cpu_packets[frame][c.camera])
        summary=dict(run_id=logger.run_id,instance='58752018',resolution=[512,512],
            actual_sensor_views=1,empty_cloned_base=True,hand_points_by_object=counts,
            timings={key:stats(value) for key,value in times.items()},
            all_observation_fields_and_audits_exact=True,input_packets_unchanged=True,
            observation_and_check_outputs_all_cuda=True,warmups=args.warmups,repeats=args.repeats,
            hand_observation_modes_source_cameras=1,chain_modes_source_cameras=2,
            torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),
            sampling_packet_and_observation_validation_at_producer_excluded=True)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(args.output/'timings.npz',**{key:np.asarray(v) for key,v in times.items()})
        logger.record(summary=summary)
        logger.link_output(args.output/'summary.json');logger.link_output(args.output/'timings.npz')
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
