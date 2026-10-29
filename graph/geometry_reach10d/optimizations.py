"""Independent candidates. Never injected into a running or frozen BC source.

Component cells are cliques, with cross-cell edges admitted only by the same
SciPy exact radius predicate as P0. Labels are canonicalized by first source row.
Detection reuse caches only the pure detector call, never prefix selection/state.
"""
import copy
import hashlib
import json
import time
import numpy as np
from scipy.spatial import cKDTree


def components(xyz, radius):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    xyz=np.asarray(xyz)
    def original():
        if not len(xyz):return np.empty(0,int)
        pairs=cKDTree(xyz).query_pairs(radius,output_type='ndarray')
        return connected_components(coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(xyz),len(xyz))),directed=False)[1]
    if len(xyz)<128 or radius<=0 or not np.isfinite(xyz).all():return original()
    width=radius/2
    scaled=xyz/width
    if np.max(abs(scaled))>2**40:return original()
    cells,inverse=np.unique(np.floor(scaled).astype(np.int64),axis=0,return_inverse=True)
    if len(xyz)/len(cells)<8:return original()
    rows=[[] for _ in cells]
    for i,c in enumerate(inverse):rows[c].append(i)
    points=[xyz[r] for r in rows]
    # A strict clique bound avoids any threshold rounding at the cell diagonal.
    if any(np.linalg.norm(np.ptp(p,axis=0))>=radius for p in points):return original()
    parent=np.arange(len(cells))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    centers=(cells.astype(float)+.5)*width
    # Conservative superset of every pair of cells that could touch. The extra
    # margin protects cell-center rounding; it cannot create a point edge.
    pairs=cKDTree(centers).query_pairs(radius+np.sqrt(3)*width+width*1e-8,output_type='ndarray')
    trees={}
    for a,b in pairs:
        ra,rb=find(a),find(b)
        if ra==rb:continue
        if a not in trees:trees[a]=cKDTree(points[a])
        # Stop at the first real witness; do not enumerate the dense pair set.
        touching=any(trees[a].query_ball_point(p,radius,return_length=True)>0 for p in points[b])
        if touching:parent[rb]=ra
    labels=np.array([find(c) for c in inverse]);mapping={}
    return np.array([mapping.setdefault(int(c),len(mapping)) for c in labels],dtype=np.int32)


class DetectionMemo:
    """One explicit episode/camera namespace and immutable model identity.

    Caller must hash actual weight/preprocessor files and assert model eval.
    Full boxes/scores/order (including negative results) are deep copied.
    Selection context is deliberately recomputed by original detect_initial.
    """
    def __init__(self,detector,namespace,model_identity):
        self.detector=detector;self.namespace=namespace
        self.identity=json.dumps(model_identity,sort_keys=True,separators=(',',':'))
        self.cache={};self.trace=[];self.last_ms={}
    def __call__(self,rgb,prompt):
        rgb=np.asarray(rgb);tick=time.perf_counter()
        key=(self.namespace,self.identity,rgb.dtype.str,rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest(),prompt)
        hit=key in self.cache
        if not hit:self.cache[key]=copy.deepcopy(self.detector(rgb,prompt))
        result=copy.deepcopy(self.cache[key]);self.last_ms=getattr(self.detector,'last_ms',{}).copy() if not hit else {}
        self.trace.append(dict(rgb_sha256=key[-2],prompt=prompt,hit=hit,seconds=time.perf_counter()-tick,boxes=result[0],scores=result[1],model_identity_sha256=hashlib.sha256(self.identity.encode()).hexdigest()))
        return result
