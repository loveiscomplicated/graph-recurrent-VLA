"""Execution/output validation only; deliberately no performance measurement."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.fusion import make_packet as legacy_packet
from graph.plan_a_bc.geometry import CurrentGeometry
from utils.experiment_logger import ExperimentLogger
from visual.runtime.graph_perception import process_cuda, to_cpu_evidence, make_packet
from visual.runtime.pca import build_geometry
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.test.pca_torch import TorchPCA5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(inputs, assets, schema):
    assert os.environ.get('CONTAINER_ID') == '58752018'
    assert torch.cuda.is_available()
    with np.load(inputs/'hand_camera.npz', allow_pickle=False) as a:
        sensor = {k: a[k].copy() for k in a.files}
    with np.load(inputs/'robot.npz', allow_pickle=False) as a:
        data = {k: a[k].copy() for k in a.files}
    with np.load(inputs/'masks.npz', allow_pickle=False) as a:
        masks = a['masks'].copy()
    camera = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4.,
        sensor['rgb'], sensor['depth_raw_mm'], sensor['intrinsic_cv'],
        sensor['extrinsic_cv'], sensor['cam2world_gl'], data['tcp_pose'])
    camera.validate()
    assert camera.depth.shape == (512, 512)
    frame = TrackedRGBDFrame(80, camera.camera, camera.rgb, camera.depth.astype(np.float32),
        camera.intrinsic, camera.world_from_camera, camera.tcp_pose,
        dict(enumerate(masks)), 80, 4.)
    depth = torch.as_tensor(frame.depth, device='cuda')
    tm = torch.as_tensor(masks, device='cuda')
    old = TorchPCA5(frame.intrinsic, frame.depth.shape, backend='triton_fused')(
        depth, dict(enumerate(tm)))
    prepared, values = process_cuda(depth, tm, torch.as_tensor(frame.intrinsic, device='cuda'))
    evidence = to_cpu_evidence(frame, prepared, values)
    assert evidence.frame.geometry.xyz is None
    pixels = evidence.frame.geometry.source_pixels
    np.testing.assert_array_equal(evidence.frame.geometry.points_at(pixels), build_geometry(frame).points_at(pixels))
    errors = {}
    for oid in (0, 1):
        for name in evidence.frame.objects[oid].__dataclass_fields__:
            a = getattr(evidence.frame.objects[oid], name)
            b = old['objects'][oid][name].cpu().numpy()
            if a.dtype.kind == 'f':
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=0, equal_nan=True, err_msg=f'{oid}:{name}')
                finite = np.isfinite(a) & np.isfinite(b)
                errors[f'{oid}:{name}'] = float(np.max(np.abs(a[finite]-b[finite]), initial=0))
            else:
                np.testing.assert_array_equal(a, b, err_msg=f'{oid}:{name}')
    expected, _ = legacy_packet(camera, masks)
    with patch('visual.runtime.pca.build_geometry', side_effect=AssertionError('dense XYZ forbidden')):
        actual, _ = make_packet(camera, masks)
    for name in ('points', 'object_id', 'feature_field_valid'):
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    np.testing.assert_array_equal(actual['provenance']['source_index'], expected['provenance']['source_index'])
    np.testing.assert_allclose(actual['geometric_features'], expected['geometric_features'], atol=2e-5, rtol=0, equal_nan=True)

    gpu = CurrentGeometry(assets, schema)
    cpu = CurrentGeometry(assets, schema, perception_backend='cpu')
    # Original CPU packet builder, including its unused Raw EDE work.
    cpu.system.packet_builder = None
    gpu.fk.verify_robot(data['robot76'], data['robot_world_geometry'], data['tcp_pose'])
    surfaces = gpu.fk.surfaces(data['robot76'])
    # Only the recorded hand view is available. Empty base support isolates it;
    # this is not a two-real-camera rollout or robot-filter validation.
    cameras = [replace(camera, camera='base_camera'), camera]
    cm = dict(base_camera=np.zeros_like(masks), hand_camera=masks)
    ready = dict(conditions={'C3': cm}, gate=dict(accepted=True))
    gpu.system.prepare = lambda *args: ready
    cpu.system.prepare = lambda *args: ready
    data.update(episode=np.array(camera.episode), timestep=np.int64(80), timestamp=np.float64(4.))
    with patch('visual.runtime.pca.build_geometry', side_effect=AssertionError('dense XYZ forbidden')):
        a, ar, ag = gpu.build(cameras, data, surfaces, cm)
    b, br, bg = cpu.build(cameras, data, surfaces, cm)
    for name in ('node_type', 'edge_index', 'edge_type', 'edge_attr'):
        np.testing.assert_array_equal(getattr(a, name).numpy(), getattr(b, name).numpy(), err_msg=name)
    np.testing.assert_allclose(a.node_input.numpy(), b.node_input.numpy(), atol=2e-5, rtol=0)
    assert [r['source_observation_ids'] for r in ar] == [r['source_observation_ids'] for r in br]
    assert ag == bg and a.packet['perception_backend'] == 'triton_fused'
    return dict(passed=True, instance='58752018', gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, resolution=[512,512], mask_pixels=masks.sum((1,2)).tolist(),
        projected_points=len(pixels), dense_xyz_allocated=False, feature_max_abs_errors=errors,
        packet_points=len(actual['points']), graph_nodes=len(a.node_type), graph_edges=a.edge_index.shape[1],
        graph_node_input_max_abs_error=float((a.node_input-b.node_input).abs().max()),
        discrete_graph_and_provenance_exact=True, robot_filter_executed=False,
        scope='recorded 512 hand view; empty base support; identical prepared masks/gate',
        speed_measured=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--schema', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(task='10i_graph_perception_default_20261103_v1', speed_measurement=False,
        input_hashes={f.name:sha(f) for f in sorted(args.inputs.glob('*.npz'))},
        verifier_sha256=sha(__file__))
    logger = ExperimentLogger(team='graph', job_type='diagnostic', config=config, seed=42,
        root=args.output/'experiments')
    try:
        result = verify(args.inputs, args.assets, json.loads(args.schema.read_text())['schema'])
        result['run_id'] = logger.run_id
        output = args.output/'verification.json'
        output.write_text(json.dumps(result, indent=2)+'\n')
        logger.record(verification=result)
        logger.link_output(output, kind='metrics')
        logger.finish()
        print(json.dumps(result, indent=2))
    except BaseException:
        logger.finish(failed=True)
        raise


if __name__ == '__main__':
    main()
