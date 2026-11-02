"""Opt-in depth-connected Main Peg candidate. No object size/pose or GT inputs."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def dominant_component_keep(xyz, radius=.012, minimum_fraction=.8, minimum_points=6):
    xyz=np.asarray(xyz).reshape(-1,3)
    keep=np.ones(len(xyz),bool)
    if len(xyz)<minimum_points:return keep,dict(reason='insufficient_support',removed=0)
    pairs=cKDTree(xyz).query_pairs(radius,output_type='ndarray')
    adjacency=coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(xyz),len(xyz)))
    _,labels=connected_components(adjacency,directed=False)
    counts=np.bincount(labels);largest=int(np.argmax(counts));fraction=float(counts[largest]/len(xyz))
    if counts[largest]>=minimum_points and fraction>=minimum_fraction:keep=labels==largest
    return keep,dict(radius_m=radius,minimum_dominant_fraction=minimum_fraction,
        dominant_fraction=fraction,components=len(counts),removed=int((~keep).sum()),
        reason='dominant_component' if not keep.all() else 'unchanged_or_ambiguous')


class MainPegDepthComponent:
    def __init__(self,geometry):
        self.geometry=geometry;self.identity=None;self.last_t=-1
        self.proposal_masks=[];self.validated_proposal_masks=[];self.active_raw_masks=[];self.events=[]
    def reset(self,identity):self.identity=identity;self.last_t=-1;self.events=[]
    def close(self):pass
    def process(self,cameras,data,surfaces,baseline_masks):
        t=int(data['timestep'])
        if str(data['episode'])!=self.identity or t!=self.last_t+1:raise ValueError('episode/branch/timestamp discontinuity')
        self.last_t=t
        out={c:m.copy() for c,m in baseline_masks.items()}
        prepared=self.geometry.system.prepare(cameras,data,out,surfaces)
        camera=next(c for c in cameras if c.camera=='base_camera')
        valid=prepared['clean']['base_camera'][0]&np.isfinite(camera.depth)
        pixels=np.flatnonzero(valid);xyz=camera.world_points().reshape(-1,3)[pixels]
        keep,details=dominant_component_keep(xyz)
        out['base_camera'][0].ravel()[pixels[~keep]]=False
        event=dict(episode=self.identity,timestep=t,correction_used=not bool(keep.all()),attempted=True,
                   clean_support_before=len(pixels),details=details,GT_used=False)
        self.events.append(event)
        return out,event
