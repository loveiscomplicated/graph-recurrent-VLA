"""Opt-in Task07-F current-mask intervention. Never modifies the RGB tracker.

Only boolean, visible pixel masks cross this oracle boundary. Poses, meshes,
contacts, goals and expert stages are deliberately absent from this interface.
"""
from dataclasses import dataclass

import numpy as np

from graph.integration_baseline.grasp_intervention import digest
from graph.integration_baseline.graph import build_graph
from visual.runtime.rgbd import TrackedRGBDFrame

POLICIES = dict(A='PREDICTED', B='GT_FP_FILTERED', C='GT_MASK_ORACLE')
SCOPE = 'VISION_ORACLE_CAUSAL_ABLATION_NOT_STANDARD_POLICY_EVALUATION'


@dataclass(frozen=True)
class CurrentSegmentation:
    episode: str
    timestep: int
    rgbd_hash: str
    masks: np.ndarray


def rgbd_hash(row):
    return digest(row['rgb'], row['depth_raw_mm'], row['intrinsic'], row['cam2world'])


def apply_masks(predicted, gt, *, condition, episode, timestep, row):
    if condition not in POLICIES:
        raise ValueError('unknown oracle condition')
    p = np.asarray(predicted)
    if p.dtype != np.bool_ or p.shape != (2, *row['depth'].shape):
        raise ValueError('pixel-aligned boolean predicted masks required')
    if (gt.episode, gt.timestep, gt.rgbd_hash) != (episode, timestep, rgbd_hash(row)):
        raise ValueError('foreign, future, stale or misaligned GT segmentation')
    if gt.masks.dtype != np.bool_ or gt.masks.shape != p.shape:
        raise ValueError('pixel-aligned boolean visible GT masks required')
    if np.any(gt.masks[0] & gt.masks[1]):
        raise ValueError('ambiguous GT actor identity')
    return (p if condition == 'A' else p & gt.masks if condition == 'B' else gt.masks).copy()


def build_input(row, predicted, gt, *, condition, episode, timestep, graph_config):
    # There is intentionally no previous packet/graph argument or cache.
    from graph.integration_baseline.vision import packet_from_frame
    masks = apply_masks(predicted, gt, condition=condition, episode=episode,
                        timestep=timestep, row=row)
    frame = TrackedRGBDFrame(timestep, 'base_camera', row['rgb'], row['depth'],
        row['intrinsic'], row['cam2world'], row['tcp_pose'],
        {0: masks[0], 1: masks[1]}, timestep, timestep/20)
    packet, timing = packet_from_frame(frame, episode, keep_ratio=.5)
    packet['oracle_audit'] = dict(input_policy=POLICIES[condition], episode=episode,
        timestep=timestep, rgbd_hash=gt.rgbd_hash, applied_mask_hash=digest(masks),
        scope=SCOPE)
    graph = build_graph(packet, row['robot_ee_geometry'], **graph_config)
    validate_input(packet, graph, masks, row, episode, timestep, POLICIES[condition])
    return masks, packet, graph, timing


def validate_input(packet, graph, masks, row, episode, timestep, policy):
    """Reject stale A tensors and check every point against its original pixel."""
    audit = packet['oracle_audit']
    expected = dict(input_policy=policy, episode=episode, timestep=timestep,
                    rgbd_hash=rgbd_hash(row), applied_mask_hash=digest(masks), scope=SCOPE)
    if audit != expected or graph.packet['oracle_audit'] != expected:
        raise ValueError('packet/graph mask provenance differs')
    source = packet['provenance']['source_index']
    obj = packet['object_id']
    if not np.all(masks.reshape(2, -1)[obj, source]):
        raise ValueError('point source is outside applied mask')
    v, u = np.unravel_index(source, row['depth'].shape)
    z = row['depth'].ravel()[source].astype(np.float64)
    k = row['intrinsic']
    camera = np.column_stack(((u+.5-k[0, 2])*z/k[0, 0],
                              (v+.5-k[1, 2])*z/k[1, 1], z))
    np.testing.assert_allclose(packet['camera_points'], camera, atol=1e-6, rtol=1e-6)
    # This is the unchanged receiver/network view, independently rebuilt.
    from graph.integration_baseline.rollout_stop_clock import numpy
    np.testing.assert_array_equal(numpy(graph.node_input)[3:, :3], packet['points'])


class OwnedHistory:
    """Verify that a live stream carries only its own last returned hidden."""
    def __init__(self, stream, episode):
        stream.reset()
        self.stream, self.episode, self.next_t, self.last = stream, episode, 0, None

    def before(self, timestep):
        s = self.stream
        if timestep != self.next_t:
            raise ValueError('noncontiguous live history')
        if timestep == 0:
            if s.hidden is not None or s.identity is not None or s.timestep is not None:
                raise ValueError('hidden must start fresh')
        elif (s.identity != self.episode or s.timestep != timestep-1 or
              digest(s.hidden.detach().cpu().numpy()) != self.last):
            raise ValueError('foreign or modified hidden history')

    def after(self, timestep):
        s = self.stream
        if s.identity != self.episode or s.timestep != timestep:
            raise ValueError('stream did not consume current observation')
        self.last = digest(s.hidden.detach().cpu().numpy())
        self.next_t += 1
