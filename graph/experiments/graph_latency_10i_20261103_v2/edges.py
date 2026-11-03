"""Batch the unchanged small final-graph edge gates on one CPU thread."""
import numpy as np
from scipy.spatial import cKDTree


def surface_edges(xyz,normals,normal_valid,object_id,reference,cameras=(),k=8,radius=.02):
    from graph.integration_baseline.graph import bidirectional
    from graph.persistent_geometry.graph import free_space_violation
    candidates=set()
    for oid in (0,1):
        rows=np.flatnonzero(object_id==oid)
        if len(rows)<2:continue
        _,near=cKDTree(xyz[rows]).query(xyz[rows],k=min(k+1,len(rows)))
        a=np.repeat(rows,near.shape[1]);b=rows[near.ravel()]
        candidates.update(zip(np.minimum(a,b)[a!=b].tolist(),np.maximum(a,b)[a!=b].tolist()))
    pairs=np.asarray(sorted(candidates),np.int64).reshape(-1,2)
    audit=dict(candidates=0,rejected_normal=0,rejected_witness=0,rejected_free_space=0)
    if not len(pairs):return bidirectional(pairs),audit
    a,b=pairs.T;delta=xyz[b]-xyz[a];length=np.linalg.norm(delta,axis=1)
    eligible=(length<=radius)&(length>=1e-9)
    dot=np.sum(normals[a]*normals[b],axis=1)
    tangent=np.maximum(np.abs(np.sum(normals[a]*delta,axis=1)),np.abs(np.sum(normals[b]*delta,axis=1)))
    bad_normal=normal_valid[a]&normal_valid[b]&((dot<.5)|(tangent>.002+length*.25))
    samples=xyz[a,None,:]+np.array([.25,.5,.75])[None,:,None]*delta[:,None,:]
    witness=np.zeros(len(pairs),bool)
    for oid in (0,1):
        rows=np.flatnonzero(object_id[a]==oid)
        if len(rows):witness[rows]=(cKDTree(reference[oid]).query(samples[rows].reshape(-1,3))[0].reshape(-1,3)>.004).any(1)
    free=free_space_violation(samples.reshape(-1,3),cameras).reshape(-1,3).any(1)
    audit.update(candidates=int(eligible.sum()),rejected_normal=int((eligible&bad_normal).sum()),
        rejected_witness=int((eligible&~bad_normal&witness).sum()),
        rejected_free_space=int((eligible&~bad_normal&~witness&free).sum()))
    return bidirectional(pairs[eligible&~bad_normal&~witness&~free]),audit
