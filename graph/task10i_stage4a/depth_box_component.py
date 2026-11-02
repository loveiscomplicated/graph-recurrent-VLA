"""Depth-component scope ablation: Main Peg and Box, same rule and same inputs."""
import numpy as np
from .depth_component import dominant_component_keep


class MainObjectDepthComponents:
    def __init__(self,geometry):
        self.geometry=geometry;self.identity=None;self.last_t=-1;self.events=[]
        self.proposal_masks=[];self.validated_proposal_masks=[];self.active_raw_masks=[]
    def reset(self,identity):self.identity=identity;self.last_t=-1;self.events=[]
    def close(self):pass
    def process(self,cameras,data,surfaces,baseline_masks):
        t=int(data['timestep'])
        if str(data['episode'])!=self.identity or t!=self.last_t+1:raise ValueError('episode/branch/timestamp discontinuity')
        self.last_t=t;out={c:m.copy() for c,m in baseline_masks.items()}
        prepared=self.geometry.system.prepare(cameras,data,out,surfaces)
        camera=next(c for c in cameras if c.camera=='base_camera');details={};removed=0
        world=camera.world_points().reshape(-1,3)
        for oid in (0,1):
            valid=prepared['clean']['base_camera'][oid]&np.isfinite(camera.depth)
            pixels=np.flatnonzero(valid);keep,info=dominant_component_keep(world[pixels])
            out['base_camera'][oid].ravel()[pixels[~keep]]=False
            details[str(oid)]=info;removed+=int((~keep).sum())
        event=dict(episode=self.identity,timestep=t,correction_used=removed>0,attempted=True,details=details,GT_used=False)
        self.events.append(event);return out,event
