"""Same-time view consistency and independent predicted-mask packet adapters."""
import numpy as np

from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.vision_packet import pose_matrix
from graph.integration_baseline.vision import packet_from_frame

from .contracts import SurfaceObservation, synchronized, transform


def observations_from_packet(packet, camera):
    if packet['camera'] != camera.camera or packet['source_frame'] != camera.timestep:
        raise ValueError('packet/sensor identity mismatch')
    world_from_ee = pose_matrix(camera.tcp_pose)
    result = []
    for oid in (0, 1):
        rows = packet['object_id'] == oid
        f5 = np.asarray(packet['geometric_features'][rows]).copy()
        valid = packet['feature_field_valid'][rows].copy()
        normals = np.zeros((int(rows.sum()), 3))
        normals[valid[:, 0]] = f5[valid[:, 0], :3] @ world_from_ee[:3, :3].T
        f5[:, :3] = np.where(valid[:, :1], normals, np.nan)
        observation = SurfaceObservation(camera.episode, camera.timestep, camera.camera, oid,
            transform(packet['points'][rows], world_from_ee), normals, valid[:, 0].copy(),
            f5, valid, packet['provenance']['source_index'][rows].copy(),
            camera.world_from_camera[:3, 3].copy())
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
    by_name = {c.camera: c for c in cameras}
    result, audit = [], []
    for observation in observations:
        keep = np.ones(len(observation.xyz), bool)
        for name, camera in by_name.items():
            if name == observation.camera or not len(observation.xyz):
                continue
            xyz = transform(observation.xyz, np.linalg.inv(camera.world_from_camera))
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
