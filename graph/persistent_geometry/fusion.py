"""Same-time view consistency and independent predicted-mask packet adapters."""
import numpy as np

from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.vision_packet import pose_matrix
from visual.runtime.object_groups import object_rows, object_slices_match
from visual.runtime.coordinate_frames import check_frame, packet_frame, camera_from_frame
from graph.integration_baseline.vision import packet_from_frame

from .contracts import SurfaceObservation, synchronized, transform
from .coordinates import camera_origin, check_observation_frame


def observations_from_packet(packet, camera, *, coordinate_frame=None):
    if packet['camera'] != camera.camera or packet['source_frame'] != camera.timestep:
        raise ValueError('packet/sensor identity mismatch')
    if not object_slices_match(packet):
        raise ValueError('object_slices disagrees with object_id')
    source_frame = packet_frame(packet)
    target = 'world' if coordinate_frame is None else check_frame(coordinate_frame)
    if coordinate_frame is not None:
        if source_frame != target:
            raise ValueError('packet/config coordinate frame mismatch')
        if packet['episode'] != camera.episode or not np.array_equal(packet['observation_ee_pose'], camera.tcp_pose):
            raise ValueError('packet/sensor reference mismatch')
    world_from_ee = pose_matrix(camera.tcp_pose)
    result = []
    for oid, rows in enumerate(object_rows(packet)):
        f5 = np.asarray(packet['geometric_features'][rows]).copy()
        valid = packet['feature_field_valid'][rows].copy()
        normals = np.zeros((len(f5), 3))
        if source_frame == target:
            normals[valid[:, 0]] = f5[valid[:, 0], :3]
            xyz = np.asarray(packet['points'][rows], dtype=float).copy()
        else:  # Explicitly retained legacy EE packet -> World diagnostic adapter.
            normals[valid[:, 0]] = f5[valid[:, 0], :3] @ world_from_ee[:3, :3].T
            xyz = transform(packet['points'][rows], world_from_ee)
        f5[:, :3] = np.where(valid[:, :1], normals, np.nan)
        observation = SurfaceObservation(camera.episode, camera.timestep, camera.camera, oid,
            xyz, normals, valid[:, 0].copy(),
            f5, valid, packet['provenance']['source_index'][rows].copy(),
            camera_origin(camera, target), target, tuple(camera.tcp_pose) if target == 'ee' else None)
        result.append(observation.validate())
    return result


def make_packet(camera, masks):
    masks = np.asarray(masks, bool)
    frame = TrackedRGBDFrame(camera.timestep, camera.camera, camera.rgb,
        camera.depth.astype(np.float32), camera.intrinsic, camera.world_from_camera,
        camera.tcp_pose, {oid: masks[oid] for oid in (0, 1)}, camera.timestep, camera.timestamp)
    packet, timing = packet_from_frame(frame, camera.episode, keep_ratio=.5)
    return packet, timing


def filtered_masks(masks, result):
    masks = np.asarray(masks, bool).copy()
    overlap = masks[0] & masks[1]
    rejected = overlap | result['confirmed_robot'] | result['uncertain']
    masks[:, rejected] = False
    return masks, rejected


def consistent_views(observations, cameras, masks):
    """A depth-supported semantic disagreement is Unknown, never an override.

    Missing masks in another view are not negative evidence. Complementary
    surfaces need not overlap. Only another *different* identity at the same
    measured surface contradicts an assignment.
    """
    synchronized(cameras)
    if len({o.coordinate_frame for o in observations}) > 1:
        raise ValueError('mixed coordinate frames in observations')
    by_name = {c.camera: c for c in cameras}
    result, audit = [], []
    for observation in observations:
        check_observation_frame(observation, by_name[observation.camera])
        keep = np.ones(len(observation.xyz), bool)
        for name, camera in by_name.items():
            if name == observation.camera or not len(observation.xyz):
                continue
            xyz = transform(observation.xyz, camera_from_frame(camera, observation.coordinate_frame))
            k = camera.intrinsic
            uv = xyz[:, :2]/np.maximum(xyz[:, 2, None], 1e-12)
            uv = np.floor(uv*np.array([k[0, 0], k[1, 1]])+k[:2, 2]).astype(int)
            h, w = camera.depth.shape
            inside = (xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            rows = np.flatnonzero(inside)
            u, v = uv[rows].T
            depth_agrees = np.abs(camera.depth[v, u]-xyz[rows, 2]) < .004
            other_only = masks[name][1-observation.object_id, v, u] & ~masks[name][observation.object_id, v, u]
            keep[rows[depth_agrees & other_only]] = False
        result.append(observation.subset(keep))
        audit.append(dict(camera=observation.camera, object_id=observation.object_id,
                          count=len(keep), unknown_conflict=int((~keep).sum())))
    return result, audit
