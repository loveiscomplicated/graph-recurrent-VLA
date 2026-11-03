"""GPU dense features and endpoint evidence, reusing the installed PCA5 kernels.

This private adapter exposes only candidates(), which is all the SF Graph path
consumes. It is deliberately not a replacement for the general PairEvidence API.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from visual.runtime.geometry import GeometryFrame,GeometryParameters,GeometricFeatures
from visual.runtime.sampling import SamplingEvidence
from visual.runtime.pca import build_geometry
from visual.runtime.test.pca_triton import fused_normals,fused_features


@triton.jit
def _endpoints(XYZ,VD,SUPPORT,VAR,DISC,VB,MB,OUT,AMB,
               H:tl.constexpr,W:tl.constexpr,ABS:tl.constexpr,REL:tl.constexpr,
               THRESHOLD:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);inside=i<H*W;v=i//W;u=i%W
    support=tl.load(SUPPORT+i,inside,0);vd=tl.load(VD+i,inside,0)
    variation=tl.load(VAR+i,inside,float('nan'))
    degrees=variation*tl.full((),57.29577951308232,tl.float32)
    orientation=support&(degrees>THRESHOLD)
    depth=tl.load(DISC+i,inside,0)
    invalid=tl.load(VB+i,inside,0)&support;mask=tl.load(MB+i,inside,0)&support
    near=tl.full((B,),False,tl.int1);far=near
    z=tl.load(XYZ+3*i+2,inside,0)
    for direction in tl.static_range(4):
        offset=tl.where(direction==0,-1,tl.where(direction==1,1,tl.where(direction==2,-W,W)))
        available=inside&tl.where(direction==0,u>0,tl.where(direction==1,u+1<W,tl.where(direction==2,v>0,v+1<H)))
        q=i+offset;qvd=tl.load(VD+q,available,0);qz=tl.load(XYZ+3*q+2,available,0)
        jump=available&support&vd&qvd&(tl.abs(z-qz)>tl.full((),ABS,tl.float32)+tl.full((),REL,tl.float32)*tl.minimum(z,qz))
        near=near|(jump&(z<qz));far=far|(jump&(z>qz))
    for j in tl.static_range(7):
        value=tl.where(j==0,orientation,tl.where(j==1,depth,tl.where(j==2,orientation|depth,
              tl.where(j==3,near,tl.where(j==4,far,tl.where(j==5,invalid,mask))))))
        tl.store(OUT+j*H*W+i,value,inside)
    tl.store(AMB+i,support&(tl.abs(degrees-THRESHOLD)<.001),inside)


class Endpoints:
    def __init__(self,geometry,oid,normals,valid,parameters,maps):
        self.args=(geometry,oid,normals,valid,parameters);self.maps=maps

    def candidates(self,threshold):
        if not np.isfinite(threshold) or threshold<0:raise ValueError('angle threshold must be finite and nonnegative')
        if threshold!=20 or self.maps is None:
            from visual.runtime.edge_evidence import pair_evidence
            return pair_evidence(*self.args).candidates(threshold)
        return self.maps


def extract(frame,estimator='pca5',parameters=GeometryParameters()):
    if estimator!='pca5':raise ValueError('this private adapter requires PCA5')
    geometry=build_geometry(frame)  # Retain original float32 XYZ exactly.
    xyz=torch.as_tensor(np.ascontiguousarray(geometry.xyz),device='cuda')
    vd=torch.as_tensor(np.ascontiguousarray(geometry.valid_depth),device='cuda')
    objects={};pairs={}
    for oid in (0,1):
        support=torch.as_tensor(np.ascontiguousarray(geometry.observed_mask(oid)),device='cuda')
        mask=torch.as_tensor(np.ascontiguousarray(geometry.object_masks[oid]),device='cuda')
        normals,nv=fused_normals(xyz,support,parameters)
        values=fused_features(xyz,vd,mask,support,normals,nv,parameters)
        maps=torch.empty((7,*mask.shape),device='cuda',dtype=torch.bool)
        ambiguous=torch.empty_like(mask)
        _endpoints[(triton.cdiv(mask.numel(),256),)](xyz,vd,support,values['normal_variation_rad'],
            values['depth_discontinuity'],values['depth_validity_boundary'],values['mask_boundary'],maps,ambiguous,
            *mask.shape,parameters.depth_jump_abs_m,parameters.depth_jump_rel,20.,enable_fp_fusion=False)
        # One packed transfer for bool fields and one for float fields.
        bool_names=[k for k,v in values.items() if v.dtype==torch.bool]
        booleans=torch.stack([values[k] for k in bool_names]+list(maps)+[ambiguous]).cpu().numpy()
        floating=torch.cat([normals.permute(2,0,1),values['depth_jump_m'][None],values['normal_variation_rad'][None]]).cpu().numpy()
        arrays={k:booleans[i] for i,k in enumerate(bool_names)}
        arrays.update(normals=np.ascontiguousarray(floating[:3].transpose(1,2,0)),
            depth_jump_m=floating[3],normal_variation_rad=floating[4],
            normal_neighbor_count=values['normal_neighbor_count'].cpu().numpy())
        objects[oid]=GeometricFeatures(**arrays)
        candidate_maps=dict(zip(('orientation','depth','union','occluding','occluded','invalid','mask'),booleans[len(bool_names):-1]))
        if booleans[-1].any():candidate_maps=None  # Preserve delicate discrete threshold decisions.
        pairs[oid]=Endpoints(geometry,oid,arrays['normals'],arrays['normal_valid'],parameters,candidate_maps)
    gf=GeometryFrame(frame.source_frame,frame.camera,frame.frame_idx,frame.timestamp,geometry,objects,{})
    return SamplingEvidence(gf,pairs,'pca5',{})
