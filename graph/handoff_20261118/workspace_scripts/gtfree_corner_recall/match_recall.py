"""Retain legitimate observations while ranking Hole candidates using front-plane evidence."""
import numpy as np

from match import CornerMatcher, HoleMatcher, XY_RIM


class EvidenceHoleMatcher(HoleMatcher):
    def __init__(self, hint_penalty_m, normal_weight, strict_normal_m=None):
        super().__init__()
        self.hint_penalty_m=hint_penalty_m
        self.normal_weight=normal_weight
        self.strict_normal_m=strict_normal_m

    def step(self, pose, points, hints):
        if pose is None or not len(points):
            return np.full(4,-1,dtype=int)
        centered=points[:,0]+1j*points[:,1]-pose.center_xy
        plane=centered*np.exp(-1j*pose.yaw)
        local=np.stack((plane.imag,points[:,2]-pose.side_m/2),axis=1)
        axial=np.abs(plane.real+pose.side_m/2)
        on_face=axial<.012
        if self.shape is None:
            self.shape=self.fit(local[on_face&hints],pose.side_m)
        result=np.full(4,-1,dtype=int)
        if self.shape is None:
            return result
        center,radius=self.shape
        lateral=np.linalg.norm((center+radius*XY_RIM)[:,None]-local[None],axis=-1)
        # Keep the previous lateral acceptance envelope. Rank with corroborating evidence.
        allowed=on_face[None]&(lateral<=self.gate)
        if self.strict_normal_m is not None:
            allowed &= (hints | (axial<=self.strict_normal_m))[None]
        cost=np.hypot(lateral,self.normal_weight*axial[None])+self.hint_penalty_m*(~hints)[None]
        cost=np.where(allowed,cost,np.inf)
        chosen=set()
        for pair in np.argsort(cost,axis=None):
            slot,i=divmod(int(pair),len(points))
            if not np.isfinite(cost[slot,i]):
                break
            if result[slot]<0 and i not in chosen:
                result[slot]=i
                chosen.add(i)
        return result


def build(params):
    model=CornerMatcher()
    if params is not None:
        model.hole=EvidenceHoleMatcher(*params)
    return model
