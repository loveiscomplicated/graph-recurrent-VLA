"""Causal physical IDs from RGB-D detector candidates and workspace geometry."""
from dataclasses import dataclass

import numpy as np

from box_matcher import BoxCornerMatcher, SIGNS

XY_RIM = SIGNS[:4,1:]
MIN_RADIUS,MAX_RADIUS=.016,.030


def assign(expected, points, gate):
    """Match closest available pairs within a distance gate."""
    if not len(points):
        return np.full(len(expected),-1,dtype=int)
    distance=np.linalg.norm(expected[:,None]-points[None],axis=-1)
    result=np.full(len(expected),-1,dtype=int)
    used=set()
    for pair in np.argsort(distance,axis=None):
        i,j=divmod(int(pair),len(points))
        if distance[i,j]>gate:
            break
        if result[i]<0 and j not in used:
            result[i]=j
            used.add(j)
    return result


class HoleMatcher:
    def __init__(self, gate=.008):
        self.gate=gate
        self.shape=None

    def step(self, pose, points, hints):
        if pose is None or not len(points):
            return np.full(4,-1,dtype=int)
        proper, height=points[:,:2],points[:,2]
        centered=proper[:,0]+1j*proper[:,1]-pose.center_xy
        plane=centered*np.exp(-1j*pose.yaw)
        local=np.stack((plane.imag,height-pose.side_m/2),axis=1)
        on_face=np.abs(plane.real+pose.side_m/2)<.012
        if self.shape is None:
            self.shape=self.fit(local[on_face & hints],pose.side_m)
        if self.shape is None:
            return np.full(4,-1,dtype=int)
        center,radius=self.shape
        candidates=np.flatnonzero(on_face)
        ref=center+radius*XY_RIM
        selected=assign(ref,local[candidates],self.gate)
        if len(candidates):
            selected=np.where(selected>=0,candidates[selected.clip(min=0)],-1)
        return selected

    @staticmethod
    def fit(points,side):
        if len(points)<3:
            return None
        i,j=np.triu_indices(len(points),1)
        span=np.abs(points[i]-points[j]).max(axis=1)*.5
        radii=span[(span>=MIN_RADIUS)&(span<=MAX_RADIUS)]
        if not len(radii):
            return None
        centers=points[None,:,None]-radii[:,None,None,None]*XY_RIM[None,None,:,:]
        centers=centers.reshape(-1,2)
        radii=np.repeat(radii,len(points)*4)
        world=centers[:,None]+radii[:,None,None]*XY_RIM
        valid=(np.abs(world).max(axis=(1,2))<side/2-.005)
        if not valid.any():
            return None
        radii,centers,world=radii[valid],centers[valid],world[valid]
        dist=np.linalg.norm(world[:,:,None]-points[None,None],axis=-1).min(axis=-1)
        support=(dist<.006).sum(axis=1)
        best=np.lexsort((np.minimum(dist,.006).sum(axis=1),-support))[0]
        if support[best]<3:
            return None
        match=assign(world[best],points,.006)
        keep=match>=0
        offsets=points[match[keep]]-radii[best]*XY_RIM[keep]
        return offsets.mean(axis=0),radii[best]


@dataclass
class PegPose:
    shape: np.ndarray
    center: np.ndarray
    rotation: np.ndarray

    def corners(self):
        return (SIGNS*self.shape)@self.rotation.T+self.center


def rotation_z(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    out=np.zeros(yaw.shape+(3,3))
    out[...,0,0]=out[...,1,1]=c
    out[...,1,0]=s
    out[...,0,1]=-s
    out[...,2,2]=1
    return out


def fit_flat_peg(points):
    # The initial Peg lies flat on the table and its positive long axis points generally +Y.
    if len(points)<3:
        return None
    i,j=np.triu_indices(len(points),1)
    delta=points[j,:2]-points[i,:2]
    long=np.linalg.norm(delta,axis=1)
    plausible=(long>.163)&(long<.259)
    if not plausible.any():
        return None
    delta,long=delta[plausible],long[plausible]
    edge=np.arctan2(delta[:,1],delta[:,0])%np.pi
    width=np.median(points[points[:,2]>.023,2])*.5 if (points[:,2]>.023).any() else .02
    width=np.clip(width,.015,.025)
    skew=np.arctan2(2*width,long)
    yaw=(edge[:,None]+np.stack((np.zeros_like(skew),-skew,skew),axis=1)).ravel()
    half=np.stack((long,np.sqrt(np.maximum(long**2-(2*width)**2,0)),
                   np.sqrt(np.maximum(long**2-(2*width)**2,0))),axis=1).ravel()*.5
    valid=(yaw>.47)&(yaw<2.67)&(half>=.081)&(half<=.129)
    yaw,half=yaw[valid],half[valid]
    if not len(yaw):
        return None
    rot=rotation_z(yaw)
    shape=np.stack((half,np.full_like(half,width),np.full_like(half,width)),axis=1)
    offsets=(SIGNS[None]*shape[:,None])@rot.transpose(0,2,1)
    centers=points[None,:,None,:]-offsets[:,None,:,:]
    centers[...,2]=width
    expected=centers[:,:,:,None]+offsets[:,None,None]
    dif=np.linalg.norm(expected[...,None,:]-points,axis=-1).min(axis=-1)
    n=(dif<.009).sum(axis=-1).ravel()
    cost=np.minimum(dif,.009).sum(axis=-1).ravel()
    best=np.lexsort((cost,-n))[0]
    if n[best]<3:
        return None
    i=best//(len(points)*8)
    return PegPose(shape[i],centers.reshape(-1,3)[best],rot[i])


def fit_rigid(local, observed, prior):
    """Fit a rigid pose with three non-collinear anchors; preserve rotation otherwise."""
    centered=local-local.mean(axis=0)
    rotation=prior
    if len(local)>=3 and np.linalg.matrix_rank(centered,tol=.0003)>=2:
        u,_,vt=np.linalg.svd(centered.T@(observed-observed.mean(axis=0)))
        sign=np.linalg.det(vt.T@u.T)
        rotation=vt.T@np.diag((1.,1.,sign))@u.T
    return observed.mean(axis=0)-local.mean(axis=0)@rotation.T,rotation


class PegMatcher:
    def __init__(self,gate=.008):
        self.gate=gate
        self.pose=None
        self.velocity=np.zeros(3)
        self.tcp=None

    def step(self, points, tcp):
        delta=np.zeros(3) if self.tcp is None else tcp-self.tcp
        self.tcp=np.asarray(tcp).copy()
        if self.pose is None:
            self.pose=fit_flat_peg(points)
            if self.pose is None:
                return np.full(8,-1,dtype=int)
        previous=self.pose
        corners=previous.corners()
        offsets=points[None]-corners[:,None]
        offsets=offsets[np.linalg.norm(offsets,axis=-1)<.035]
        candidates=np.vstack((np.zeros(3),self.velocity,delta,offsets))
        if not len(points):
            self.pose=PegPose(previous.shape,previous.center+self.velocity,previous.rotation)
            self.velocity*=.6
            return np.full(8,-1,dtype=int)
        distance=np.linalg.norm(corners[None,:,None]+candidates[:,None,None]-points[None,None],axis=-1).min(axis=-1)
        matches=(distance<.012).sum(axis=1)
        cost=np.minimum(distance,.012).sum(axis=1)+.14*np.linalg.norm(candidates-self.velocity,axis=1)
        ranking=np.lexsort((cost,-matches))
        for idx in ranking[:8]:
            picked=assign(corners+candidates[idx],points,.013)
            observed=picked>=0
            if observed.sum()<2:
                continue
            local=(SIGNS*previous.shape)[observed]
            center,rot=fit_rigid(local,points[picked[observed]],previous.rotation)
            if np.linalg.norm(center-previous.center)>.045 or np.trace(previous.rotation.T@rot)<2.65:
                continue
            new=PegPose(previous.shape,center,rot)
            self.pose=new
            self.velocity=.4*self.velocity+.6*(center-previous.center)
            return assign(new.corners(),points,self.gate)
        self.pose=PegPose(previous.shape,previous.center+self.velocity,previous.rotation)
        self.velocity*=.6
        return assign(self.pose.corners(),points,self.gate)


class CornerMatcher:
    def __init__(self, peg_gate=.008,hole_gate=.008):
        self.peg=PegMatcher(peg_gate)
        self.box=BoxCornerMatcher(assignment_m=.006)
        self.hole=HoleMatcher(hole_gate)

    def step(self, points, family, tcp):
        peg=np.flatnonzero(family==0)
        box=np.flatnonzero(family!=0)
        assigned=np.full(20,-1,dtype=int)
        outer=self.box.step(points[box])
        front=np.ones(len(box),bool)
        front[outer[outer>=0]]=False
        unclaimed=box[front]
        mappings=((slice(8),peg,self.peg.step(points[peg],tcp)),
                  (slice(8,16),box,outer),
                  (slice(16,20),unclaimed,self.hole.step(self.box.pose,points[unclaimed],family[unclaimed]==2)))
        for span,indices,values in mappings:
            sel=values>=0
            assigned[span][sel]=indices[values[sel]]
        chosen=assigned[assigned>=0]
        assert len(chosen)==len(np.unique(chosen))
        return assigned
