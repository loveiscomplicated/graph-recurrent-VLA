"""Extra common inputs. No contact, phase, reward, or future-state features."""
from dataclasses import dataclass
import torch
from .goals import goal_geometry, rotation6

STATE_DIMS={'none':0,'proprio':9,'goal':31}


def shared_features(g,qvel,mode='goal'):
    if qvel.shape!=(len(g.qpos),9):raise ValueError('qvel must be [B,9], arm rad/s and fingers m/s')
    if mode not in STATE_DIMS:raise ValueError(mode)
    if mode=='none':return qvel[:,:0]
    result=qvel
    if mode=='goal':
        goal=goal_geometry(g)
        extra=torch.cat((goal['ee_to_grasp'][:,0],rotation6(goal['grasp_relative_rotation']),
                         goal['head_in_hole'],rotation6(goal['peg_relative_rotation']),
                         g.hole_radius-g.peg_half_size[:,1:2],goal['insertion_axis']),-1)
        result=torch.cat((qvel,extra),-1)
    if not torch.isfinite(result).all():raise ValueError('nonfinite shared state')
    if result.shape[1]!=STATE_DIMS[mode]:raise AssertionError('feature dimensions')
    return result


@dataclass
class StateNormalizer:
    mean:torch.Tensor
    std:torch.Tensor

    @classmethod
    def fit(cls,g,qvel,mode):
        features=shared_features(g,qvel,mode)
        return cls(features.mean(0),features.std(0,correction=0).clamp_min(.001))

    def apply(self,g,qvel,mode):return (shared_features(g,qvel,mode)-self.mean)/self.std
    def as_dict(self):return dict(mean=self.mean,std=self.std)
