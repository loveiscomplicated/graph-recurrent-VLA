"""Evaluation-only motion oracle; excluded from practical input module."""
import numpy as np
from graph.persistent_geometry.registration import Registration
from .core import BoxMemory


class OracleBoxMemory(BoxMemory):
    def __init__(self,cfg,main='base_camera'):
        super().__init__('B2',cfg,main)
        self.initial_gt=None

    def update_oracle(self,episode,timestep,records,camera,gt_matrix):
        if episode!=self.episode:
            if timestep!=0:raise ValueError('oracle episode boundary')
            self.reset(episode);self.initial_gt=None
        if self.initial_gt is None and records:self.initial_gt=gt_matrix.copy()
        self.oracle_pose=gt_matrix@np.linalg.inv(self.initial_gt) if self.initial_gt is not None else np.eye(4)
        super().update_records(episode,timestep,records,camera)
        if self.objects[1].active():self.objects[1].world_from_object=self.oracle_pose.copy()
        self.trace[-1]['oracle_GT_motion']=True
        return self

    def _register(self,obj,current):
        return Registration(self.oracle_pose.copy(),True,'ORACLE_GT_MOTION',0.,1.,1.,
            float(np.linalg.norm(self.oracle_pose[:3,3]-obj.world_from_object[:3,3])),0.,6)
