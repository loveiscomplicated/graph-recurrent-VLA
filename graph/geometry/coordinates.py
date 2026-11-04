"""Frame-aware current records; legacy persistent records remain World-only."""
from dataclasses import replace
import numpy as np
from graph.perception.coordinate_frames import check_frame, world_from_frame, camera_to_frame
from graph.perception.vision_packet import pose_matrix


def record_xyz(record):
    return record['xyz'] if 'coordinate_frame' in record else record['world_xyz']


def record_normal(record):
    return record['normal'] if 'coordinate_frame' in record else record['world_normal']


def validate_record_frames(records, frame=None, tcp_pose=None):
    frames = {r.get('coordinate_frame', 'world') for r in records}
    if len(frames) > 1:
        raise ValueError('mixed coordinate frames in records')
    actual = next(iter(frames), frame or 'world')
    check_frame(actual)
    if frame is not None and actual != check_frame(frame):
        raise ValueError('record/config coordinate frame mismatch')
    if actual == 'ee' and records:
        poses = {tuple(r['reference_ee_pose']) for r in records}
        if len(poses) != 1:
            raise ValueError('mixed EE reference poses in records')
        pose = next(iter(poses))
        pose_matrix(pose)
        if tcp_pose is not None and not np.array_equal(pose, tcp_pose):
            raise ValueError('stale EE reference pose in records')
    return actual


def check_observation_frame(observation, camera):
    check_frame(observation.coordinate_frame)
    if observation.coordinate_frame == 'ee' and not np.array_equal(observation.reference_ee_pose, camera.tcp_pose):
        raise ValueError('stale EE reference pose in observation')


def observation_in_frame(observation, camera, frame):
    """Adapter for legacy World observations; new packets already use frame."""
    from graph.geometry.contracts import transform
    check_observation_frame(observation, camera)
    if observation.coordinate_frame == check_frame(frame):
        return observation
    matrix = np.linalg.inv(world_from_frame(frame, camera.tcp_pose)) @ world_from_frame(observation.coordinate_frame, camera.tcp_pose)
    normals = observation.normals @ matrix[:3, :3].T
    f5 = observation.f5.copy()
    f5[:, :3] = np.where(observation.normal_valid[:, None], normals, np.nan)
    return replace(observation, xyz=transform(observation.xyz, matrix), normals=normals,
        f5=f5, camera_origin=transform(observation.camera_origin[None], matrix)[0],
        coordinate_frame=frame, reference_ee_pose=tuple(camera.tcp_pose) if frame == 'ee' else None).validate()


def camera_origin(camera, frame):
    from types import SimpleNamespace
    return camera_to_frame(SimpleNamespace(tcp_pose=camera.tcp_pose, cam2world=camera.world_from_camera), frame)[:3, 3].copy()
