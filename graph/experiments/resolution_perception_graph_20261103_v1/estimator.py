"""Exact function-source export of the frozen geometric precision probe; no new estimator."""
import numpy as np
from scipy.spatial import cKDTree, Delaunay, ConvexHull
from visual.runtime.vision_packet import pose_matrix
from graph.persistent_geometry.contracts import transform

def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),-1,1))))

def canon(a):return a*(-1 if a[np.abs(a).argmax()]<0 else 1)

def peg_estimate(x,n,nv):
    result=dict(status='NOT_ESTIMABLE',reason='INSUFFICIENT_OBSERVED_SUPPORT',input_count=len(x))
    if len(x)<12:return result
    nn=cKDTree(x).query(x,k=2)[0][:,1]; med=np.median(nn);mad=np.median(abs(nn-med))
    keep=nn<=med+4*max(mad,.001); x=x[keep];n=n[keep];nv=nv[keep]
    result['retained_count']=len(x);result['isolated_removed']=int((~keep).sum())
    if len(x)<12:return result
    c=x.mean(0);v,u=np.linalg.eigh(np.cov((x-c).T)); order=np.argsort(v)[::-1];v=v[order];u=u[:,order]
    a=canon(u[:,0]);u[:,0]=a
    p=(x-c)@u;lo,hi=np.quantile(p[:,0],[.01,.99]);ratio=v[0]/max(v[1],1e-15)
    result.update(axis_eigenratio=float(ratio),observed_span_mm=float((hi-lo)*1000))
    if ratio<4:result['reason']='AMBIGUOUS_GEOMETRY';return result
    if hi-lo<.04:return result
    q=np.quantile(p,[.02,.98],axis=0);mid=(q[0]+q[1])/2;mid[0]=np.median(p[:,0]);anchor=c+u@mid
    ends=[];support=[];faces=[]
    for e in (lo,hi):
        ep=mid.copy();ep[0]=e;ends.append(c+u@ep)
        band=abs(p[:,0]-e)<=.005
        face=band&nv&(abs(n@a)>=.85)
        s=dict(envelope_band_count=int(band.sum()),normal_face_count=int(face.sum()),estimable=False)
        fc=None
        if face.sum()>=3:
            z=p[face];span=np.ptp(z[:,1:],axis=0);std=float(np.std(z[:,0]));s.update(tangential_span_mm=(span*1000).tolist(),axial_std_mm=std*1000)
            if np.all(span>=.006) and std<=.002:
                f=(z.min(0)+z.max(0))/2;f[0]=np.median(z[:,0]);fc=c+u@f;s['estimable']=True
        support.append(s);faces.append(fc)
    result.update(status='ESTIMABLE',reason='GEOMETRY_ESTIMATED',axis=a,anchor=anchor,observed_ends=np.array(ends),end_support=support,physical_ends=faces,
        end_pair_status='ESTIMABLE' if all(f is not None for f in faces) else 'NOT_ESTIMABLE',end_pair_reason='GEOMETRY_ESTIMATED' if all(f is not None for f in faces) else 'INSUFFICIENT_OBSERVED_SUPPORT',tip_identity='TIP_IDENTITY_UNRESOLVED')
    return result

def hole_estimate(x,n,nv,jump,jv):
    out=dict(status='NOT_ESTIMABLE',reason='INSUFFICIENT_OBSERVED_SUPPORT',input_count=len(x),plane_supports=[],candidate_count=0)
    if len(x)<12:return out
    planes=[]
    for i in np.flatnonzero(nv):
        normal=canon(n[i]/np.linalg.norm(n[i])); off=float(x[i]@normal)
        mask=(abs(x@normal-off)<=.002)&((~nv)|(abs(n@normal)>=.95))
        if mask.sum()<12:continue
        if any(angle(normal,a)<5 and abs(off-b)<.003 for _,a,b,_ in planes):continue
        planes.append((int(mask.sum()),normal,off,int(i)))
    planes.sort(key=lambda p:(-p[0],p[3])); candidates=[]
    for count,normal,off,i in planes[:3]:
        mask=(abs(x@normal-off)<=.002)&((~nv)|(abs(n@normal)>=.95));xx=x[mask];cc=xx.mean(0)
        _,_,u=np.linalg.svd(xx-cc,full_matrices=False);nn=canon(u[-1]);uv=u[:2].T
        xy=(xx-cc)@uv
        out['plane_supports'].append(dict(nodes=count,seed=i))
        try: tri=Delaunay(xy);hull=ConvexHull(xy)
        except Exception as e:out['numerical_warning']=type(e).__name__;continue
        gaps=[]
        for ix in tri.simplices:
            a,b,c=xy[ix]
            try:center=np.linalg.solve(2*np.stack([b-a,c-a]),np.array([b@b-a@a,c@c-a@a]))
            except np.linalg.LinAlgError:continue
            dist=np.linalg.norm(xy-center,axis=1);r=float(dist.min())
            if not .008<=r<=.05:continue
            clearance=float(np.min(-(hull.equations[:,:2]@center+hull.equations[:,2])))
            if clearance<1.8*r:continue
            ring=(dist>=r-1e-8)&(dist<=1.55*r)
            theta=(np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0])+np.pi)*8/(2*np.pi)
            sectors=np.floor(theta).astype(int)%8
            rim=ring&jv[mask]&(jump[mask]>=.008)
            if ring.sum()<8 or len(set(sectors[ring]))<6 or rim.sum()<8 or len(set(sectors[rim]))<6:continue
            if (dist<.8*r).any():continue
            gaps.append(dict(center=cc+uv@center,normal=nn,radius_mm=r*1000,rim_support=int(rim.sum()),ring_support=int(ring.sum()),rim_sectors=len(set(sectors[rim])),plane_nodes=count))
        gaps.sort(key=lambda g:-g['radius_mm'])
        for g in gaps[:10]:
            if not any(np.linalg.norm(g['center']-h['center'])<.005 for h in candidates):candidates.append(g)
    out['candidate_count']=len(candidates);out['candidates']=candidates
    if len(candidates)>1:out['reason']='AMBIGUOUS_GEOMETRY'
    elif len(candidates)==1:out.update(status='ESTIMABLE',reason='GEOMETRY_ESTIMATED',**candidates[0])
    return out

def estimate_graph(g,tcp):
    # Hard boundary: GT, dense cameras, masks, reference, pool and provenance unavailable here.
    inp=g.node_input.numpy();typ=g.node_type.numpy();m=pose_matrix(tcp)
    results={}
    for oid,name in ((0,'peg'),(1,'hole')):
        f=inp[typ==oid+3];x=transform(f[:,:3],m);norm=f[:,3:6]@m[:3,:3].T;nv=f[:,11].astype(bool)
        assert np.isfinite(x).all()
        try:results[name]=peg_estimate(x,norm,nv) if oid==0 else hole_estimate(x,norm,nv,f[:,7],f[:,13].astype(bool))
        except Exception as exc:results[name]=dict(status='NOT_ESTIMABLE',reason='ESTIMATOR_FAILURE',exception=type(exc).__name__+':'+str(exc))
    return results
