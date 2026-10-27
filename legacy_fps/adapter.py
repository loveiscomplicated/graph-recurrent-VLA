"""Exact legacy point policy with source row joins and explicit F5 augmentation.

The original numerical helpers are imported from the frozen dependency copy.
No GT or future evidence is accepted. RGB-D depth is float32 metres, matching
the current G0 input; the old source's depth dtype is a caller responsibility.
"""
from dataclasses import dataclass
import hashlib
import cv2
import numpy as np
from visual.scripts import pointcloud as original
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence
from visual.runtime.intra_object_edges import observed_components
from visual.runtime.test.raw_ede_comparison_adapter import build_raw_ede
from graph.persistent_geometry.contracts import SurfaceObservation

PARAMETERS=dict(erode_px=2,min_pts=50,sor_k=16,sor_std=2.,fps_pre=2048,max_pts=1024,cluster=False)

def frame_seed(episode,timestep,camera,object_id,seed=42):
    import json
    # Same-pose and translated causal branches share the parent's RNG stream.
    # Their complete source identity remains in SurfaceObservation, not this key.
    rng_episode=str(episode).split(':branch_',1)[0]
    key=json.dumps([seed,rng_episode,int(timestep),str(camera),int(object_id)],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')

def row_join(before,after):
    """Exact float32 XYZ byte identity, never nearest-neighbour provenance."""
    before=np.ascontiguousarray(before,dtype=np.float32)
    after=np.ascontiguousarray(after,dtype=np.float32)
    keys=[p.tobytes() for p in before]
    if len(set(keys))!=len(keys):raise ValueError('ambiguous duplicate XYZ cannot identify source pixels')
    index={k:i for i,k in enumerate(keys)}
    try:return np.array([index[p.tobytes()] for p in after],np.int64)
    except KeyError as e:raise ValueError('point stage invented or changed XYZ') from e

@dataclass
class Points:
    camera_xyz:np.ndarray
    world_xyz:np.ndarray
    source_pixel:np.ndarray
    audit:dict
    stages:dict

def extract(mask,depth,K,c2w,rng,*,max_pts=1024,erode_px=2,cluster=False):
    mask=np.asarray(mask)
    if mask.dtype!=bool or mask.shape!=depth.shape or depth.ndim!=2:raise ValueError('mask/depth alignment')
    if not isinstance(max_pts,int) or max_pts<1:raise ValueError('positive final budget required')
    if not np.isfinite(depth[depth>0]).all():raise ValueError('positive infinite depth is invalid')
    info=dict(n_mask_px=int(mask.sum()));m=mask
    eroded=mask.copy();fallback=False
    if erode_px>0 and info['n_mask_px']:
        eroded=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=erode_px)>0
        if (eroded&(depth>0)).sum()>=original.MIN_PTS:m=eroded
        else:fallback=True
    ids=np.flatnonzero(m&(depth>0)).astype(np.int64)
    pts=original.backproject(depth,K,m)
    stages=dict(eroded_mask=eroded,selected_mask=m.copy(),backproject_source=ids.copy())
    info['n_raw_pts']=len(pts)
    if len(pts)>original.FPS_PRE:
        choose=rng.choice(len(pts),original.FPS_PRE,replace=False);pts=pts[choose];ids=ids[choose]
    stages['presample_source']=ids.copy()
    filtered=original.sor_filter(pts);ids=ids[row_join(pts,filtered)];pts=filtered
    info['n_sor_pts']=len(pts);stages['sor_source']=ids.copy()
    if cluster:
        n=len(pts);filtered=original.drop_small_clusters(pts)
        ids=ids[row_join(pts,filtered)];pts=filtered
        info['cluster_removed_frac']=round(1-len(pts)/n,4) if n else 0.
    stages['cluster_source']=ids.copy()
    final=original.fps(pts,max_pts,rng);ids=ids[row_join(pts,final)]
    info.update(erosion_fallback=fallback,eroded_valid_count=int((eroded&(depth>0)).sum()),
        final_count=len(final),provenance='exact XYZ byte row join through original SOR/cluster/FPS helpers',
        label='LEGACY_EXACT_GIVEN_INPUT_AND_RNG',depth_dtype=str(depth.dtype))
    return Points(final,original.to_world(final,c2w),ids,info,stages)

def frame_points(camera,masks,*,seed=42,f5=False):
    """Returns native XYZ plus optional original PCA5/F5 at actual FPS pixels.

    The original script consumed one sequential RNG across episodes/objects.
    Here frame seeds are independent. Exactness is against the same RNG state,
    not a claim to reproduce the historical sequential run's selected pixels.
    """
    camera.validate();masks=np.asarray(masks,bool)
    if masks.shape!=(2,*camera.depth.shape):raise ValueError('peg0/box1 mask contract')
    depth=camera.depth.astype(np.float32)
    evidence=None;components=None
    if f5:
        frame=TrackedRGBDFrame(camera.timestep,camera.camera,camera.rgb,depth,camera.intrinsic,
            camera.world_from_camera,camera.tcp_pose,{i:masks[i] for i in (0,1)},camera.timestep,camera.timestamp)
        evidence=extract_sampling_evidence(frame,estimator='pca5')
        components,_=observed_components(evidence)
    points=[];observations=[];metadata=[]
    for oid in (0,1):
        s=frame_seed(camera.episode,camera.timestep,camera.camera,oid,seed)
        p=extract(masks[oid],depth,camera.intrinsic,camera.world_from_camera,np.random.default_rng(s))
        n=len(p.source_pixel);valid=np.zeros((n,3),bool);features=np.full((n,5),np.nan)
        normals=np.zeros((n,3));component=np.full(n,-1,np.int64)
        if f5:
            dense=evidence.frame.objects[oid];idx=p.source_pixel
            valid=np.column_stack([getattr(dense,k).ravel()[idx] for k in ('normal_valid','normal_variation_valid','depth_pair_valid')])
            features=np.column_stack([dense.normals.reshape(-1,3)[idx],dense.normal_variation_rad.ravel()[idx],dense.depth_jump_m.ravel()[idx]])
            normals[valid[:,0]]=features[valid[:,0],:3]@camera.world_from_camera[:3,:3].T
            features[:,:3]=np.where(valid[:,:1],normals,np.nan)
            component=components[oid].ravel()[idx]
        obs=SurfaceObservation(camera.episode,camera.timestep,camera.camera,oid,p.world_xyz,
            normals,valid[:,0],features,valid,p.source_pixel,camera.world_from_camera[:3,3].copy()).validate()
        p.audit.update(rng_seed=s,seed_policy='sha256 canonical JSON first8 little endian; per episode/frame/camera/object',
            f5_added=f5,assignment_confidence_available=False,measurement_validity='depth>0; predicted mask; finite XYZ')
        points.append(p);observations.append(obs);metadata.append(component)
    return observations,points,metadata

def raw_ede(camera,observations,components):
    """Only one current camera's actual pixels/components; rejects maps/mixed views."""
    for o in observations:
        if (o.episode,o.timestep,o.camera)!=(camera.episode,camera.timestep,camera.camera):
            raise ValueError('Raw EDE requires one current raster')
    oid=np.concatenate([np.full(len(o.xyz),o.object_id,np.int64) for o in observations])
    source=np.concatenate([o.pixels for o in observations]);comp=np.concatenate(components)
    if (comp<0).any():raise ValueError('BLOCKED_INPUT: original observed component missing/ambiguous')
    return build_raw_ede(dict(object_id=oid,provenance=dict(source_index=source),
        observed_component_id=comp,image_shape=np.array(camera.depth.shape,np.int64)))
