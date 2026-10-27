"""Explicit evaluation-only upper bounds. Never imported by input modules."""
import numpy as np
from visual.runtime.vision_packet import pose_matrix
from .map import PersistentMap
from .registration import Registration


class OracleMotionMap(PersistentMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gt_reference = {}
        self.gt_current = {}
        self.map_reference = {}

    def reset(self, episode):
        super().reset(episode)
        self.gt_reference, self.gt_current, self.map_reference = {}, {}, {}

    def update_with_gt_motion(self, episode, timestep, observations, actor_poses, *, cameras=()):
        if episode != self.episode:
            self.reset(episode)
        self.gt_current = {oid:pose_matrix(actor_poses[oid]) for oid in (0, 1)}
        super().update(episode, timestep, observations, cameras=cameras)
        for oid, obj in self.objects.items():
            if obj.active() and oid not in self.gt_reference:
                self.gt_reference[oid] = self.gt_current[oid].copy()
                self.map_reference[oid] = obj.world_from_object.copy()
            # During complete occlusion the GT-motion upper bound deliberately
            # moves *only previously observed* nodes. This is not practical
            # sensing and is always labelled O2/O3.
            if oid in self.gt_reference:
                obj.world_from_object = (self.gt_current[oid] @ np.linalg.inv(self.gt_reference[oid])
                                         @ self.map_reference[oid])
                obj.pose_confidence = 1.
                obj.pose_uncertainty = 0.
        return self

    def _register(self, obj, current):
        oid = obj.object_id
        pose = self.gt_current[oid] @ np.linalg.inv(self.gt_reference[oid]) @ self.map_reference[oid]
        return Registration(pose, True, 'ORACLE_GT_MOTION', 0., 1., 1.,
            float(np.linalg.norm(pose[:3, 3]-obj.world_from_object[:3, 3])), 0., 3)
