"""Real512 World/EE correctness, with actual FK and prepared-mask boundaries."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation, transform
from graph.persistent_geometry.fusion import observations_from_packet
from graph.plan_a_bc.geometry import CurrentGeometry
from visual.runtime.coordinate_frames import FRAME_NAMES
from visual.runtime.graph_perception import make_packet, make_packet_cuda
from visual.runtime.tensor_packet import numpy_packet
from visual.runtime.vision_packet import pose_matrix
from graph.vision_input import validate_packet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--schema', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018' and torch.cuda.is_available()
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('another CUDA compute process is active; do not stop it')
    args.output.mkdir(parents=True, exist_ok=False)
    def arrays(name):
        with np.load(args.inputs/name, allow_pickle=False) as a:
            return {k: a[k].copy() for k in a.files}
    s = arrays('hand_camera.npz'); data = arrays('robot.npz'); masks = arrays('masks.npz')['masks']
    camera = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4., s['rgb'], s['depth_raw_mm'],
        s['intrinsic_cv'], s['extrinsic_cv'], s['cam2world_gl'], data['tcp_pose']).validate()
    schema = json.loads(args.schema.read_text())['schema']
    cameras = (replace(camera, camera='base_camera'), camera)
    cm = dict(base_camera=np.zeros_like(masks), hand_camera=masks)
    data.update(episode=np.array(camera.episode), timestep=np.int64(80), timestamp=np.float64(4.))
    logger = ExperimentLogger(team='graph', job_type='diagnostic', seed=42,
        root=args.output/'experiments', config=dict(task='10i_coordinate_frames_20261103',
        coordinate_frames=['world', 'ee'], speed_measured=False, robot_filter_executed=False,
        scope='real hand512 + empty cloned base support; current recorded FK; prepared masks/gate',
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        packets = {}; graphs = {}; rows = {}; details = {}; saved = {}
        for frame in ('world', 'ee'):
            packet, _ = make_packet(camera, masks, coordinate_frame=frame)
            tensor = make_packet_cuda(camera, torch.as_tensor(masks, device='cuda'), coordinate_frame=frame)
            validate_packet(tensor); host = numpy_packet(tensor); validate_packet(host)
            np.testing.assert_array_equal(host['provenance']['source_index'], packet['provenance']['source_index'])
            np.testing.assert_array_equal(host['object_id'], packet['object_id'])
            np.testing.assert_array_equal(host['feature_field_valid'], packet['feature_field_valid'])
            np.testing.assert_allclose(host['points'], packet['points'], rtol=0, atol=1e-6)
            np.testing.assert_allclose(host['geometric_features'], packet['geometric_features'], rtol=0, atol=2e-5)
            matrix = camera.world_from_camera if frame == 'world' else np.linalg.inv(pose_matrix(camera.tcp_pose)) @ camera.world_from_camera
            reference = transform(packet['camera_points'], matrix)
            np.testing.assert_allclose(packet['points'], reference, rtol=0, atol=1e-6)
            observations = observations_from_packet(packet, camera, coordinate_frame=frame)
            for o in observations:
                take = packet['object_id'] == o.object_id
                np.testing.assert_array_equal(o.xyz, packet['points'][take])
                np.testing.assert_array_equal(o.f5, packet['geometric_features'][take])
            geometry = CurrentGeometry(args.assets, schema, coordinate_frame=frame)
            geometry.fk.verify_robot(data['robot76'], data['robot_world_geometry'], data['tcp_pose'])
            surfaces = geometry.fk.surfaces(data['robot76'])
            geometry.system.prepare = lambda *args: dict(conditions={'C3': cm}, gate=dict(accepted=True))
            graph, records, _ = geometry.build(cameras, data, surfaces, cm)
            assert graph.packet['coordinate_frame'] == FRAME_NAMES[frame]
            by_id = {r['persistent_node_id']: r for r in records}
            for i, name in enumerate(graph.packet['node_ids']):
                r = by_id[name]
                assert r['coordinate_frame'] == frame and 'world_xyz' not in r
                np.testing.assert_array_equal(graph.node_input[i+3, :3], np.asarray(r['xyz'], np.float32))
                expected_normal = r['last_f5'][:3] if r['last_feature_valid'][0] else np.zeros(3)
                np.testing.assert_array_equal(graph.node_input[i+3, 3:6], np.asarray(expected_normal, np.float32))
            expected_robot = data['robot_world_geometry'] if frame == 'world' else transform(data['robot_world_geometry'], np.linalg.inv(pose_matrix(camera.tcp_pose)))
            np.testing.assert_allclose(graph.node_input[:3, :3], expected_robot, rtol=0, atol=1e-7)
            packets[frame] = packet; graphs[frame] = graph; rows[frame] = by_id
            details[frame] = dict(selected_points=len(packet['points']), final_records=len(records),
                graph_nodes=len(graph.node_type), graph_edges=graph.edge_index.shape[1],
                coordinate_frame=graph.packet['coordinate_frame'], packet_to_observation_exact=True,
                records_to_graph_positions_normals_exact=True, tensor_points_max_abs_m=float(np.max(np.abs(host['points']-packet['points']), initial=0)),
                direct_camera_transform_max_abs_m=float(np.max(np.abs(packet['points']-reference), initial=0)))
            saved.update({frame+'_packet_points': packet['points'], frame+'_source': packet['provenance']['source_index'],
                frame+'_node_input': graph.node_input.numpy(), frame+'_edge_index': graph.edge_index.numpy(),
                frame+'_edge_attr': graph.edge_attr.numpy()})
        np.testing.assert_array_equal(packets['world']['provenance']['source_index'], packets['ee']['provenance']['source_index'])
        np.testing.assert_allclose(packets['world']['points'], transform(packets['ee']['points'], pose_matrix(camera.tcp_pose)), rtol=0, atol=1e-6)
        shared = sorted(rows['world'].keys() & rows['ee'].keys())
        if shared:
            a = np.array([rows['world'][k]['xyz'] for k in shared])
            b = transform(np.array([rows['ee'][k]['xyz'] for k in shared]), pose_matrix(camera.tcp_pose))
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        summary = dict(run_id=logger.run_id, passed=True, resolution=[512, 512], mask_pixels=masks.sum((1,2)).tolist(),
            frames=details, pre_budget_sources_exact=True, shared_final_source_ids=len(shared),
            final_source_ids_equal=graphs['world'].packet['node_ids'] == graphs['ee'].packet['node_ids'],
            speed_measured=False, robot_filter_executed=False,
            scope='real hand view and empty cloned base support; no two-real-camera rollout')
        (args.output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
        np.savez_compressed(args.output/'geometry.npz', **saved)
        logger.record(verification=summary)
        logger.link_output(args.output/'verification.json'); logger.link_output(args.output/'geometry.npz')
        logger.finish(); print(json.dumps(summary, indent=2))
    except BaseException:
        logger.finish(failed=True); raise


if __name__ == '__main__':
    main()
