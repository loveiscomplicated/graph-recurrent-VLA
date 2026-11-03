"""Correctness only: compare original and grouped packets on real512 input."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess

import numpy as np
import torch

from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.fusion import observations_from_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.graph_perception import make_packet


def same(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    args = p.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018' and torch.cuda.is_available()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('another CUDA compute process is active; do not stop it')
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.inputs/'hand_camera.npz', allow_pickle=False) as a:
        sensor = {k: a[k].copy() for k in a.files}
    with np.load(args.inputs/'robot.npz', allow_pickle=False) as a:
        tcp = a['tcp_pose'].copy()
    with np.load(args.inputs/'masks.npz', allow_pickle=False) as a:
        masks = a['masks'].copy()
    c = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4., sensor['rgb'],
        sensor['depth_raw_mm'], sensor['intrinsic_cv'], sensor['extrinsic_cv'],
        sensor['cam2world_gl'], tcp).validate()
    logger = ExperimentLogger(team='graph', job_type='diagnostic', seed=42,
        root=args.output/'experiments', config=dict(task='10i_object_groups_20261103',
        speed_measured=False, reference=str(args.reference) if args.reference else None,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        packet, _ = make_packet(c, masks)
        observations = [asdict(o) for o in observations_from_packet(packet, c)]
        result = dict(packet=packet, observations=observations)
        if args.reference:
            # This is our own capture from the frozen source, never user input.
            with args.reference.open('rb') as f:
                expected = pickle.load(f)
            same({k: v for k, v in packet.items() if k != 'object_slices'}, expected['packet'])
            same(observations, expected['observations'])
            n0 = len(observations[0]['xyz'])
            assert packet['object_slices'] == ((0, n0), (n0, len(packet['points'])))
        with (args.output/'capture.pkl').open('wb') as f:
            pickle.dump(result, f, protocol=5)
        summary = dict(run_id=logger.run_id, passed=True, instance=os.environ['CONTAINER_ID'],
            resolution=list(c.depth.shape), mask_pixels=masks.sum((1, 2)).tolist(),
            selected_points=len(packet['points']), object_slices=packet.get('object_slices'),
            reference_compared=bool(args.reference), speed_measured=False,
            all_existing_packet_fields_exact=bool(args.reference),
            all_observation_fields_exact=bool(args.reference), robot_filter_executed=False)
        path = args.output/'verification.json'
        path.write_text(json.dumps(summary, indent=2)+'\n')
        logger.record(verification=summary)
        logger.link_output(path)
        logger.link_output(args.output/'capture.pkl')
        logger.finish()
        print(json.dumps(summary, indent=2))
    except BaseException:
        logger.finish(failed=True)
        raise


if __name__ == '__main__':
    main()
