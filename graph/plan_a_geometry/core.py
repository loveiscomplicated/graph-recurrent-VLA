"""GT-free, stateless Plan A camera gate, point selection and 3-D graphs."""
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from graph.persistent_geometry.contracts import synchronized, transform
from graph.persistent_geometry.self_filter import classify_robot
from graph.persistent_geometry.fusion import make_packet, observations_from_packet, filtered_masks, consistent_views
from graph.persistent_geometry.map import current_records
from graph.persistent_geometry.graph import build_graph, farthest_rows
from legacy_fps.adapter import frame_points


def components(xyz, radius):
    if not len(xyz): return np.empty(0, int)
    pairs = cKDTree(xyz).query_pairs(radius, output_type='ndarray')
    matrix = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    return connected_components(matrix, directed=False)[1]


def wrist_gate(camera, raw_masks, clean_masks, cameras, clean_all, cfg):
    """Uses current calibrated depth/masks only; missing Main is not rejection."""
    raw = raw_masks[0]
    support = clean_masks[0] & np.isfinite(camera.depth)
    pix = np.flatnonzero(support)
    xyz = camera.world_points().reshape(-1, 3)[pix]
    labels = components(xyz, cfg['component_radius_m'])
    counts = np.bincount(labels) if len(labels) else np.array([], int)
    largest = labels == counts.argmax() if len(counts) else np.zeros(0, bool)
    local = cKDTree(xyz).query_ball_point(xyz, cfg['local_neighbor_radius_m'], return_length=True) >= 3 if len(xyz) else np.zeros(0, bool)
    raw_n = int(raw.sum()); valid_n = int((raw & np.isfinite(camera.depth)).sum())
    fraction = lambda a,b: float(a/b) if b else 0.
    trace = dict(camera=camera.camera, episode=camera.episode, timestep=camera.timestep,
        raw_pixels=raw_n, valid_depth=valid_n, clean_points=len(pix),
        valid_depth_fraction=fraction(valid_n,raw_n), clean_fraction=fraction(len(pix),valid_n),
        largest_component_fraction=fraction(largest.sum(),len(pix)), local_support_fraction=float(local.mean()) if len(local) else 0.,
        component_diameter_m=float(np.linalg.norm(np.ptp(xyz[largest],axis=0))) if largest.any() else 0.,
        tracker_confidence_available=False, main_required=False)
    # Reuse 09-C's depth-supported cross-camera semantic contradiction contract.
    from graph.persistent_geometry.contracts import SurfaceObservation
    n=len(pix)
    obs=SurfaceObservation(camera.episode,camera.timestep,camera.camera,0,xyz,np.zeros((n,3)),np.zeros(n,bool),
        np.full((n,5),np.nan),np.zeros((n,3),bool),pix,camera.world_from_camera[:3,3]).validate()
    checked, association = consistent_views([obs], cameras, clean_all)
    allowed = np.isin(pix,checked[0].pixels)
    trace['identity_conflict_fraction']=fraction((~allowed).sum(),n)
    reasons=[]
    for reject,name in (
        (n<cfg['minimum_valid_points'],'insufficient_clean_depth'),
        (trace['valid_depth_fraction']<cfg['minimum_valid_depth_fraction'],'invalid_depth_support'),
        (trace['clean_fraction']<cfg['minimum_clean_fraction'],'robot_or_overlap_dominated'),
        (trace['largest_component_fraction']<cfg['minimum_component_fraction'],'fragmented_3d_support'),
        (trace['local_support_fraction']<cfg['minimum_local_support_fraction'],'isolated_surface'),
        (trace['component_diameter_m']>cfg['maximum_component_diameter_m'],'implausible_extent'),
        (trace['identity_conflict_fraction']>cfg['maximum_identity_conflict_fraction'],'cross_camera_identity_conflict')):
        if reject: reasons.append(name)
    accepted=np.zeros_like(support)
    if not reasons: accepted.ravel()[pix[largest & allowed]]=True
    trace.update(accepted=not reasons, reasons=reasons, accepted_points=int(accepted.sum()),
        association=association, GT_used=False, future_used=False)
    return accepted,trace


def sor_surface_observations(camera,masks,points):
    """Actual post-SOR pixels, BEFORE legacy final FPS; current-camera F5."""
    from graph.persistent_geometry.contracts import SurfaceObservation
    from visual.runtime.rgbd import TrackedRGBDFrame
    from visual.runtime.sampling import extract_sampling_evidence
    from visual.scripts.pointcloud import backproject,to_world
    frame=TrackedRGBDFrame(camera.timestep,camera.camera,camera.rgb,camera.depth.astype(np.float32),
        camera.intrinsic,camera.world_from_camera,camera.tcp_pose,{i:masks[i] for i in (0,1)},camera.timestep,camera.timestamp)
    evidence=extract_sampling_evidence(frame,estimator='pca5');observations=[]
    for oid,p in enumerate(points):
        pixels=p.stages['sor_source'];dense=evidence.frame.objects[oid]
        support=np.zeros_like(masks[oid]);support.ravel()[pixels]=True
        # SOR source order can reflect random pre2048 order. Preserve that exact order.
        sorted_pixels=np.flatnonzero(support);lookup={int(x):i for i,x in enumerate(sorted_pixels)}
        xyz=backproject(frame.depth,frame.intrinsic,support)[[lookup[int(x)] for x in pixels]]
        world=to_world(xyz,camera.world_from_camera)
        valid=np.column_stack([getattr(dense,k).ravel()[pixels] for k in ('normal_valid','normal_variation_valid','depth_pair_valid')])
        f5=np.column_stack([dense.normals.reshape(-1,3)[pixels],dense.normal_variation_rad.ravel()[pixels],dense.depth_jump_m.ravel()[pixels]])
        normals=np.zeros((len(pixels),3));normals[valid[:,0]]=f5[valid[:,0],:3]@camera.world_from_camera[:3,:3].T
        f5[:,:3]=np.where(valid[:,:1],normals,np.nan)
        observations.append(SurfaceObservation(camera.episode,camera.timestep,camera.camera,oid,world,normals,
            valid[:,0],f5,valid,pixels,camera.world_from_camera[:3,3].copy()).validate())
    return observations


def key(r):
    return tuple(r['source_observation_ids'][0])


def deduplicate(records, distance=.003):
    """Keep real representative measurements; never average unrelated F5/UV."""
    out=[]; buckets={}; duplicates=0
    for r in sorted(records,key=lambda x:(x['object_id'],not x.get('structural',False),key(x))):
        cell=tuple(np.floor(r['world_xyz']/distance).astype(int)); oid=r['object_id']
        candidates=[]
        for x in (-1,0,1):
            for y in (-1,0,1):
                for z in (-1,0,1): candidates.extend(buckets.get((oid,cell[0]+x,cell[1]+y,cell[2]+z),[]))
        hits=[i for i in candidates if np.linalg.norm(out[i]['world_xyz']-r['world_xyz'])<distance]
        if hits:
            representative=out[min(hits)]
            representative['source_observation_ids'].extend(r['source_observation_ids'])
            representative['structural']=representative['structural'] or r.get('structural',False)
            duplicates+=1
        else:
            r=deepcopy(r); r.setdefault('structural',False)
            buckets.setdefault((oid,*cell),[]).append(len(out));out.append(r)
    return out,dict(input_points=len(records),unique_points=len(out),duplicates=duplicates,
        duplicate_fraction=duplicates/len(records) if records else 0.,representative='stable real measurement, no XYZ/F5 averaging')


def select(records, count, *, anchors=False):
    if count is None or len(records)<=count: return list(records)
    if count==0:return []
    xyz=np.array([r['world_xyz'] for r in records]); ids=np.array([r['persistent_node_id'] for r in records])
    structural=np.flatnonzero([r.get('structural',False) for r in records]) if anchors else np.empty(0,int)
    if len(structural)>count:
        chosen=structural[farthest_rows(xyz[structural],ids[structural],count)]
    elif not len(structural): chosen=farthest_rows(xyz,ids,count)
    else:
        chosen=list(structural[np.argsort(ids[structural])]);dist=np.full(len(records),np.inf)
        for i in chosen: dist=np.minimum(dist,np.sum((xyz-xyz[i])**2,axis=1))
        order=np.argsort(ids)
        while len(chosen)<count:
            dist[chosen]=-1;best=dist.max();i=int(next(i for i in order if dist[i]==best))
            chosen.append(i);dist=np.minimum(dist,np.sum((xyz-xyz[i])**2,axis=1))
    return [records[i] for i in chosen]


def budget(records,counts,hybrid=False):
    return sum((select([r for r in records if r['object_id']==oid], counts[oid] if isinstance(counts,dict) else counts,anchors=hybrid) for oid in (0,1)),[])


class PlanA:
    def __init__(self,main,wrist,protocol): self.main,self.wrist,self.protocol=main,wrist,protocol

    def prepare(self,cameras,data,masks,surfaces):
        synchronized(cameras)
        allowed={'robot76','robot_world_geometry','tcp_pose','qpos','robot_link_world','timestep','timestamp','episode'}
        allowed.update(c.camera+'/'+k for c in cameras for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'))
        if set(data)-allowed:raise ValueError('forbidden non-sensor input')
        if len(cameras)!=2 or {c.camera for c in cameras}!={self.main,self.wrist} or set(masks)!={self.main,self.wrist}:raise ValueError('camera identity')
        if str(data['episode'])!=cameras[0].episode or int(data['timestep'])!=cameras[0].timestep:raise ValueError('sensor identity join')
        by={c.camera:c for c in cameras}; filters={};clean={}
        for c in cameras:
            if np.shape(masks[c.camera])!=(2,*c.depth.shape):raise ValueError('mask alignment')
            filters[c.camera]=classify_robot(c,surfaces)
            clean[c.camera],_=filtered_masks(masks[c.camera],filters[c.camera])
        accepted,gate=wrist_gate(by[self.wrist],masks[self.wrist],clean[self.wrist],cameras,clean,self.protocol['gate'])
        conditions={}
        for condition in ('C0','C1','C2','C3'):
            cm={name:m.copy() for name,m in clean.items()}
            if condition=='C0':cm[self.wrist][:]=False
            if condition in ('C2','C3'):cm[self.wrist][1]=False
            if condition=='C3':cm[self.wrist][0]=accepted
            conditions[condition]=cm
        return dict(filters=filters,clean=clean,conditions=conditions,gate=gate)

    def represent(self,cameras,prepared,condition,representations=('SF','L','H')):
        cm=prepared['conditions'][condition];observed=[];sf=[];legacy=[];surface_candidates=[];stages={};packets={}
        for c in sorted(cameras,key=lambda c: c.camera!=self.main):
            packet,_=make_packet(c,cm[c.camera]);packets[c.camera]=packet
            os=observations_from_packet(packet,c)
            os,_=consistent_views(os,cameras,cm)
            rows=current_records(os,fuse_views=False)
            structural_pixels=set(packet['provenance']['source_index'][~packet['provenance']['surface_selected']].tolist())
            for r in rows:r['structural']=key(r)[3] in structural_pixels
            sf.extend(rows)
            if any(v in representations for v in ('L','H')):
                los,points,_=frame_points(c,cm[c.camera],f5=True)
                los,_=consistent_views(los,cameras,cm)
                lr=current_records(los,fuse_views=False)
                for r in lr:r['structural']=False
                legacy.extend(lr)
                if 'H' in representations:
                    candidates=sor_surface_observations(c,cm[c.camera],points)
                    candidates,_=consistent_views(candidates,cameras,cm)
                    for r in current_records(candidates,fuse_views=False):
                        r['structural']=False;surface_candidates.append(r)
                stages[c.camera]=points
        sf,sa=deduplicate(sf,self.protocol['merge_distance_m'])
        legacy,la=deduplicate(legacy,self.protocol['merge_distance_m'])
        legacy=budget(legacy,1024)
        # Preserve structural candidates before any erosion/SOR of surface pool.
        structural=[deepcopy(r) for r in sf if r['structural']]
        hybrid,ha=deduplicate(structural+surface_candidates,self.protocol['merge_distance_m'])
        hybrid=budget(hybrid,self.protocol['native_hybrid_budget'],True)
        pools=dict(SF=sf,L=legacy,H=hybrid)
        return {name:pools[name] for name in representations},dict(SF=sa,L=la,H=ha),stages,packets

    def graph(self,records,data,cameras):
        graph=build_graph(records,data['robot_world_geometry'],data['tcp_pose'],int(data['timestep']),
            cameras=cameras,point_cap=None,include_observation_status=False)
        # 09-C tensor schema retained; temporal/status extras disabled explicitly.
        graph.packet.update(schema_id='task09d_current_surface21_v1',persistent_geometry=False,
            current_only=True,temporal_features_disabled=True,point_budget_policy='task09d_variant_selection_before_graph')
        if (graph.node_input[:,14:]!=0).any():raise ValueError('temporal feature active')
        return graph
