"""Explicit common frame after camera-local perception: World or current TCP."""
from copy import deepcopy
import numpy as np

FRAME_NAMES = {'world': 'World_metres', 'ee': 'EE_metres'}


def check_frame(frame):
    if not isinstance(frame, str) or frame not in FRAME_NAMES:
        raise ValueError("coordinate_frame must be 'world' or 'ee'")
    return frame


def packet_frame(packet):
    for frame, name in FRAME_NAMES.items():
        if packet.get('coordinate_frame') == name:
            return frame
    raise ValueError('World_metres or EE_metres packet required')


def feature_fields(frame):
    from graph.perception.vision_packet import FEATURE_FIELDS
    fields = deepcopy(FEATURE_FIELDS)
    fields[0]['frame'] = 'World' if check_frame(frame) == 'world' else 'EE'
    return fields


def world_from_frame(frame, tcp_pose):
    from graph.perception.vision_packet import pose_matrix
    return np.eye(4) if check_frame(frame) == 'world' else pose_matrix(tcp_pose)


def camera_to_frame(observation, frame):
    from graph.perception.vision_packet import validate_transform, pose_matrix
    world = validate_transform(observation.cam2world)
    if check_frame(frame) == 'world':
        return world
    return validate_transform(np.linalg.inv(pose_matrix(observation.tcp_pose)) @ world)


def camera_from_frame(camera, frame):
    return np.linalg.inv(camera.world_from_camera) @ world_from_frame(frame, camera.tcp_pose)
