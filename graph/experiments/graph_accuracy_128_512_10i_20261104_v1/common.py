"""Sensor loading and paired identities; no experiment-specific model code."""
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ('base_camera', 'hand_camera')
PARENTS = ('10i_s1_0197', '10i_s1_0275', '10i_s1_0305', '10i_s1_0531')
SIZES = (128, 512)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def serial(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=serial) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world_points(records, tcp_pose):
    """Honor the current record frame instead of assuming legacy world_xyz."""
    from graph.persistent_geometry.contracts import transform
    from visual.runtime.vision_packet import pose_matrix
    xyz = np.asarray([r['xyz'] for r in records], dtype=np.float64).reshape(-1, 3)
    if not records:
        return xyz
    frame = records[0]['coordinate_frame']
    if any(r['coordinate_frame'] != frame for r in records):
        raise ValueError('mixed record coordinates')
    if frame == 'world':
        return xyz
    if frame != 'ee' or any(tuple(r['reference_ee_pose']) != tuple(tcp_pose) for r in records):
        raise ValueError('record EE reference does not match current frame')
    return transform(xyz, pose_matrix(tcp_pose))


def entries(capture):
    rows = read(Path(capture) / 'capture_manifest.json')['entries']
    rows = [e for e in rows if e['cohort'] == 'tracking' and e['parent'] in PARENTS]
    result = {(e['parent'], e['timestep'], e['size'], e['camera']): e for e in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate captured identity')
    for parent in PARENTS:
        for t in range(21):
            hashes = set()
            for size in (*SIZES, 1024):
                for camera in CAMERAS:
                    e = result[parent, t, size, camera]
                    hashes.add(e['physical_sha256'])
            if len(hashes) != 1:
                raise ValueError('resolution pairs differ in physical state')
    return result


def sensor_frame(capture, index, parent, t, size, identity):
    """Load only public sensor/robot arrays; no actor pose or label arrays."""
    from graph.persistent_geometry.contracts import CameraObservation
    capture = Path(capture)
    cameras = []
    for name in CAMERAS:
        e = index[parent, t, size, name]
        path = capture / e['sensor_file']
        if sha(path) != e['sensor_sha256']:
            raise ValueError('sensor file changed')
        with np.load(path, allow_pickle=False) as z:
            v = {k: z[k].copy() for k in ('rgb', 'depth_raw_mm', 'intrinsic_cv', 'extrinsic_cv', 'cam2world_gl')}
        if v['rgb'].shape != (size, size, 3) or v['depth_raw_mm'].shape != (size, size):
            raise ValueError('native sensor shape')
        robot_path = path.parent / 'robot.npz'
        with np.load(robot_path, allow_pickle=False) as z:
            data = {k: z[k].copy() for k in z.files}
        cameras.append(CameraObservation(identity, t, name, t / 20,
            v['rgb'], v['depth_raw_mm'], v['intrinsic_cv'], v['extrinsic_cv'],
            v['cam2world_gl'], data['tcp_pose'].copy()).validate())
    return cameras, data
