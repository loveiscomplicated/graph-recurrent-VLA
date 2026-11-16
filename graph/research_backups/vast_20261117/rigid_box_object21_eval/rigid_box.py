"""Causal rigid translation of Box and Hole points using exterior observations."""
import numpy as np


class RigidBoxTranslation:
    """Use a shared Box shift while the Box orientation is fixed.

    Slot coordinates are anchored at their first measured pose. An exterior
    observation earns a vote only when the detector and the previous track
    agree that it belongs to the Box exterior. Measurements stay authoritative.
    """
    def __init__(self, *, use_velocity, inlier_m=.008, min_support=3,
                 max_speed_m_per_frame=.012, max_prediction_frames=5):
        self.use_velocity=use_velocity
        self.inlier_m=inlier_m
        self.min_support=min_support
        self.max_speed=max_speed_m_per_frame
        self.max_prediction=max_prediction_frames
        self.anchors={}
        self.exterior=set()
        self.translation=np.zeros(3)
        self.velocity=np.zeros(3)
        self.last_frame=None
        self.frame=-1

    def update(self, xyz, observed, kinds, exterior_ids):
        """Return corrected positions and frame-level fit information."""
        self.frame+=1
        candidates=sorted(self.exterior.intersection(exterior_ids))
        support=0
        if self.last_frame is None:
            if exterior_ids:
                self.last_frame=self.frame
        elif len(candidates)>=self.min_support:
            deltas=np.stack([xyz[tid]-self.anchors[tid] for tid in candidates])
            support_sets=np.linalg.norm(deltas[:,None]-deltas[None],axis=2)<=self.inlier_m
            center=support_sets.sum(1).argmax()
            agreed=support_sets[center]
            if agreed.sum()>=self.min_support:
                step=self.frame-self.last_frame
                update=np.median(deltas[agreed],axis=0)
                speed=(update-self.translation)/step
                if np.linalg.norm(speed)<=self.max_speed:
                    support=int(agreed.sum())
                    self.velocity=.35*self.velocity+.65*speed
                    self.translation=update
                    self.last_frame=self.frame
        since_update=0 if self.last_frame is None else self.frame-self.last_frame
        lag=min(since_update,self.max_prediction) if self.use_velocity else 0
        current=self.translation+self.velocity*lag
        # Keep fixed slot offsets after registration. Do not move the RGB-D
        # measurements themselves, including observations rejected for fitting.
        for tid in np.flatnonzero(observed&(kinds==1)):
            if tid not in self.anchors:
                self.anchors[tid]=np.asarray(xyz[tid],float)-current
                if tid in exterior_ids:
                    self.exterior.add(tid)
        result=xyz.copy()
        for tid in np.flatnonzero(~observed&(kinds==1)):
            if tid in self.anchors:
                result[tid]=self.anchors[tid]+current
        return result,dict(support=support,age=since_update,pose_m=current.tolist())
