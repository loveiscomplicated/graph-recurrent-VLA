"""GT scoring of already frozen predictions, with a common native1024 denominator."""
from common import *
from functools import lru_cache
import argparse
import time
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from graph.persistent_geometry.contracts import CameraObservation,transform
from visual.runtime.vision_packet import pose_matrix
from graph_worker import masks_for,resolve_folder


def bbox(mask):
    yy,xx=np.where(mask)
    return [float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)] if len(xx) else None


def box_iou(a,b):
    if a is None or b is None:return 0.
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def boundary_f1(pred,gt,tolerance):
    a=np.argwhere(pred&~binary_erosion(pred));b=np.argwhere(gt&~binary_erosion(gt))
    if not len(a) or not len(b):return 0. if len(a) or len(b) else None
    precision=float((cKDTree(b).query(a)[0]<=tolerance).mean())
    recall=float((cKDTree(a).query(b)[0]<=tolerance).mean())
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def score_mask(pred,truth,oid,tolerance):
    gt=truth[oid];n=int(pred.sum());g=int(gt.sum());inter=int((pred&gt).sum());union=n+g-inter
    counts=[int((pred&m).sum()) for m in truth]
    return dict(mask_pixels=n,native_visible_pixels=g,intersection=inter,union=union,
        mask_iou=inter/union if union else None,precision=inter/n if n else None,recall=inter/g if g else None,
        boundary_f1=boundary_f1(pred,gt,tolerance),
        robot_fraction=counts[2]/n if n else None,other_object_fraction=counts[1-oid]/n if n else None,
        background_fraction=(n-sum(counts))/n if n else None,
        wrong_object=bool(n and (counts[1-oid]/n>=.5 or counts[2]/n>=.5)))


def camera(scene,size,name):
    folder=ROOT/'sensors'/scene/str(size)
    with np.load(folder/'robot.npz',allow_pickle=False) as r:tcp=r['tcp_pose'].copy()
    with np.load(folder/(name+'.npz'),allow_pickle=False) as z:
        return CameraObservation(scene,0,name,0.,z['rgb'].copy(),z['depth_raw_mm'].copy(),z['intrinsic_cv'].copy(),
                                 z['extrinsic_cv'].copy(),z['cam2world_gl'].copy(),tcp).validate()


@lru_cache(maxsize=8)
def gt_masks(scene,size,name):
    manifest=read(ROOT/'capture_manifest.json')['entries']
    e=next(e for e in manifest if e['scene']==scene and e['size']==size and e['camera']==name)
    with np.load(ROOT/e['gt_file'],allow_pickle=False) as z:seg=z['actor_segmentation']
    return [np.isin(seg,e['object_ids'][o]) for o in ('peg','box','robot')]


def region_flags(xyz,oid,state):
    m=pose_matrix(np.asarray(state['peg_pose'] if oid==0 else state['hole_pose']))
    local=transform(xyz,np.linalg.inv(m));L=state['peg_half'][0]
    if oid==0:
        return dict(end_minus=abs(local[:,0]+L)<.004,end_plus=abs(local[:,0]-L)<.004)
    yz=abs(local[:,1:]);r=state['hole_radius'];outside=np.linalg.norm(np.maximum(yz-r,0),axis=1)
    inside=(yz<=r).all(1);outside[inside]=r-yz[inside].max(1)
    return dict(rim=(abs(local[:,0]+L)<.004)&(outside<.004))


def canonical(scene):
    state=read(ROOT/'GT'/scene/'state.json');refs={};all_refs={};boundaries={};visibility={}
    for oid in (0,1):
        rr=[];bb=[];aa=[]
        for c in CAMERAS:
            obs=camera(scene,1024,c);truth=gt_masks(scene,1024,c)[oid]
            valid=truth&np.isfinite(obs.depth);xyz=obs.world_points()
            visibility[oid,c]=dict(visible_pixels=int(truth.sum()),visibility='VISIBLE' if truth.sum()>=1024 else 'TINY_VISIBLE' if truth.any() else 'ABSENT')
            aa.append(xyz[valid].astype(np.float32))
            if oid==0 or c=='base_camera':
                rr.append(xyz[valid].astype(np.float32));bb.append(xyz[valid&~binary_erosion(truth)].astype(np.float32))
        refs[oid]=np.concatenate(rr).reshape(-1,3);boundaries[oid]=np.concatenate(bb).reshape(-1,3);all_refs[oid]=np.concatenate(aa).reshape(-1,3)
    return dict(state=state,refs=refs,all_refs=all_refs,boundaries=boundaries,visibility=visibility,
                regions={oid:region_flags(refs[oid],oid,state) for oid in (0,1)})


def meshes(scene,state):
    import trimesh
    result={}
    with np.load(ROOT/'GT'/scene/'meshes.npz',allow_pickle=False) as z:
        for oid in (0,1):
            mm=[];i=0
            while f'object{oid}_vertices{i}' in z:
                mm.append(trimesh.Trimesh(z[f'object{oid}_vertices{i}'],z[f'object{oid}_faces{i}'],process=False));i+=1
            result[oid]=trimesh.util.concatenate(mm)
    return result


def mesh_distance(xyz,oid,state,mesh):
    import trimesh
    if not len(xyz):return np.empty(0),np.empty((0,3))
    matrix=pose_matrix(np.asarray(state['peg_pose'] if oid==0 else state['box_pose']))
    local=transform(xyz,np.linalg.inv(matrix));dist=[];norm=[]
    for start in range(0,len(xyz),512):
        _,d,tri=trimesh.proximity.closest_point_naive(mesh[oid],local[start:start+512])
        dist.append(d);norm.append(mesh[oid].face_normals[tri]@matrix[:3,:3].T)
    return np.concatenate(dist),np.concatenate(norm)


def coverage(points,reference):
    if not len(reference):return None
    if not len(points):return 0.
    return float((cKDTree(points).query(reference)[0]<.004).mean())


def score_points(arrays,scene,size,oid,ref):
    take=arrays['object_id']==oid;xyz=arrays['xyz'][take];ci=arrays['camera'][take];pix=arrays['pixel'][take]
    labels=np.full(len(xyz),3,np.int8)
    for k,c in enumerate(CAMERAS):
        inds=np.flatnonzero(ci==k)
        for label,m in enumerate(gt_masks(scene,size,c)):
            labels[inds[m.ravel()[pix[inds]]]]=label
    correct=labels==oid;good=xyz[correct];reference=ref['refs'][oid]
    distance=cKDTree(reference).query(xyz)[0] if len(reference) and len(xyz) else np.empty(0)
    result=dict(point_count=len(xyz),correct_points=int(correct.sum()),reference_points=len(reference),
        contamination_fraction=float((~correct).mean()) if len(xyz) else None,
        robot_fraction=float((labels==2).mean()) if len(xyz) else None,
        other_object_fraction=float((labels==1-oid).mean()) if len(xyz) else None,
        background_fraction=float((labels==3).mean()) if len(xyz) else None,
        surface_coverage_4mm=coverage(good,reference),boundary_coverage_4mm=coverage(good,ref['boundaries'][oid]),
        all_camera_coverage_4mm=coverage(good,ref['all_refs'][oid]),
        visible_surface_distance_mean_mm=float(distance.mean()*1000) if len(distance) else None,
        visible_surface_distance_p95_mm=float(np.quantile(distance,.95)*1000) if len(distance) else None,
        merged_source_count=int(arrays['merged_sources'][take].sum()))
    flags=region_flags(good,oid,ref['state'])
    for name,rr in ref['regions'][oid].items():
        result[name+'_correct_points']=int(flags[name].sum())
        result[name+'_reference_points']=int(rr.sum())
        result[name+'_coverage_4mm']=coverage(good,reference[rr])
    return result


def angle(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


def score_estimate(est,state):
    pm=pose_matrix(np.asarray(state['peg_pose']));hm=pose_matrix(np.asarray(state['hole_pose']));L=state['peg_half'][0]
    peg=est['peg'];hole=est['hole']
    out=dict(peg_estimable=peg['status']=='ESTIMABLE',peg_reason=peg['reason'],peg_position_error_mm=None,peg_axis_error_deg=None,
        peg_end_pair_estimable=peg.get('end_pair_status')=='ESTIMABLE',peg_max_end_error_mm=None,
        hole_estimable=hole['status']=='ESTIMABLE',hole_reason=hole['reason'],hole_center_error_mm=None,hole_axis_error_deg=None,
        relative_status='TIP_IDENTITY_UNRESOLVED',relative_error_mm=None)
    if out['peg_estimable']:
        a=pm[:3,0];aa=np.asarray(peg['axis']);anchor=np.asarray(peg['anchor']);denom=float(aa@a)
        if abs(denom)>1e-10:
            q=anchor+aa*((pm[:3,3]-anchor)@a/denom);out['peg_position_error_mm']=float(np.linalg.norm(q-pm[:3,3])*1000)
        out['peg_axis_error_deg']=angle(aa,a)
        if out['peg_end_pair_estimable']:
            ends=pm[:3,3]+np.array([-L,L])[:,None]*a;pred=np.asarray(peg['physical_ends'])
            distances=min((np.linalg.norm(pred-ends[v],axis=1) for v in ([0,1],[1,0])),key=lambda d:float(d@d))
            out['peg_max_end_error_mm']=float(distances.max()*1000)
    if out['hole_estimable']:
        entrance=hm[:3,3]-L*hm[:3,0]
        out['hole_center_error_mm']=float(np.linalg.norm(np.asarray(hole['center'])-entrance)*1000)
        out['hole_axis_error_deg']=angle(hole['normal'],hm[:3,0])
    out['peg_position_below3mm']=bool(out['peg_position_error_mm'] is not None and out['peg_position_error_mm']<3)
    return out


def score_graph(folder,ref,mesh,scene,size):
    result=read(folder/'result.json')
    with np.load(folder/'graph.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    scores=[]
    with np.load(ROOT/'sensors'/scene/str(size)/'robot.npz',allow_pickle=False) as z:matrix=pose_matrix(z['tcp_pose'])
    world=transform(a['node_input'][:,:3],matrix)
    for oid in (0,1):
        met=score_points(a,scene,size,oid,ref);take=a['node_type']==oid+3
        dist,normals=mesh_distance(world[take],oid,ref['state'],mesh)
        met['mesh_distance_mean_mm']=float(dist.mean()*1000) if len(dist) else None
        met['mesh_distance_p95_mm']=float(np.quantile(dist,.95)*1000) if len(dist) else None
        features=a['node_input'][take];valid=features[:,11].astype(bool)
        if valid.any():
            nn=features[valid,3:6]@matrix[:3,:3].T
            deg=np.degrees(np.arccos(np.clip(np.abs((nn*normals[valid]).sum(1)),-1,1)))
            met['normal_error_mean_deg']=float(deg.mean())
        else:met['normal_error_mean_deg']=None
        edge=a['edge_index'];keep=(a['edge_type']==0)&(edge[0]<edge[1])
        if edge.shape[1]:keep&=(a['node_type'][edge[0]]==oid+3)&(a['node_type'][edge[1]]==oid+3)
        edges=edge[:,keep];invalid=[]
        if edges.shape[1]:
            points=world[edges[0],None,:]+np.array([.25,.5,.75])[None,:,None]*(world[edges[1]]-world[edges[0]])[:,None,:]
            d,_=mesh_distance(points.reshape(-1,3),oid,ref['state'],mesh);invalid=(d.reshape(-1,3)>.004).any(1)
        met['intra_edges']=edges.shape[1];met['surface_bridge_violation_fraction']=float(np.mean(invalid)) if len(invalid) else None
        rows=np.flatnonzero(take);lookup={v:i for i,v in enumerate(rows)}
        ij=np.array([(lookup[x],lookup[y]) for x,y in edges.T],int).reshape(-1,2)
        adjacency=coo_matrix((np.ones(len(ij)),(ij[:,0],ij[:,1])),shape=(len(rows),len(rows)))
        met['connected_components']=int(connected_components(adjacency,directed=False)[0]) if len(rows) else 0
        met['object_id']=oid;scores.append(met)
    return scores,score_estimate(result['estimate'],ref['state'])


def main(shard=0,shards=1):
    assert_source()
    for arm in ('actual','gt_box','gt_mask'):
        assert sha(ROOT/('graph_index_'+arm+'.json'))==read(ROOT/(arm+'_graph_freeze.json'))['index_sha256']
    actual=read(ROOT/'actual_graph_freeze.json')
    for path,h in actual['receipts'].items():assert sha(ROOT/path)==h
    predictions={p['id']:p for p in read(ROOT/'api_predictions.json')['rows']}
    oracles={p['id']:p for p in read(ROOT/'oracle_jobs.json')['rows']}
    indices={arm:read(ROOT/('graph_index_'+arm+'.json'))['rows'] for arm in ('actual','gt_box','gt_mask')}
    mask_rows=[];scene_rows=[];point_rows=[];pose_rows=[];stage_rows=[]
    specs=[s for i,s in enumerate(read(ROOT/'protocol.json')['scenes']) if i%shards==shard]
    suffix='' if shards==1 else f'_shard{shard}'
    with logger('GT_evaluate') as log:
        for si,s in enumerate(specs):
            ref=canonical(s['id']);mesh=meshes(s['id'],ref['state']);graph_cache={};mask_cache={}
            for size in SIZES:
                for arm in ('actual','gt_box','gt_mask'):
                    repeats=range(3) if arm=='actual' else range(1)
                    for repeat in repeats:
                        meta=dict(scene=s['id'],parent=s['parent'],cohort=s['cohort'],timestep=s['timestep'],size=size,arm=arm,repeat=repeat)
                        if arm!='gt_mask':
                            mm=masks_for(s,size,repeat,arm)
                            for c in CAMERAS:
                                view=[]
                                iid=image_id(s['parent']+'_track_t000' if s['cohort']=='tracking' else s['id'],size,c)
                                p=(predictions if arm=='actual' else oracles)[job_id(iid,repeat)]
                                for oid,obj in enumerate(('peg','box')):
                                    key=(size,c,oid,array_sha(mm[c][oid]))
                                    if key not in mask_cache:mask_cache[key]=score_mask(mm[c][oid],gt_masks(s['id'],size,c),oid,size/128)
                                    met=dict(mask_cache[key]);vis=ref['visibility'][oid,c];pb=p['parsed'][obj]['pixels'];gb=bbox(gt_masks(s['id'],size,c)[oid])
                                    biou=box_iou(pb,gb)
                                    other_biou=box_iou(pb,bbox(gt_masks(s['id'],size,c)[1-oid]))
                                    center=float(np.linalg.norm((np.array(pb[:2])+pb[2:]-np.array(gb[:2])-gb[2:])/2)/(size*np.sqrt(2))) if pb is not None and gb is not None else None
                                    row=dict(meta,camera=c,object_id=oid,**met,**vis,box_iou=biou,box_detected=bool(pb and biou>=.5),
                                        center_error_image_diagonal=center,api_status=p.get('status','GT_ORACLE'),box_status=p['parsed'][obj]['status'],
                                        other_object_box_iou=other_biou,box_wrong_object=bool(pb is not None and biou<.5 and other_biou>=.5),
                                        mask08=bool(met['mask_iou'] is not None and met['mask_iou']>=.8),
                                        box_metric_applicable=s['cohort']!='tracking' or s['timestep']==0)
                                    mask_rows.append(row);view.append(row)
                                scene_rows.append(dict(meta,camera=c,both_visible=all(v['visibility']=='VISIBLE' for v in view),
                                                       both_mask08=all(v['mask08'] for v in view),wrong_object_any=any(v['wrong_object'] for v in view)))
                        row=next(r for r in indices[arm] if r['scene']==s['id'] and r['size']==size and r['repeat']==repeat)
                        if row['status']!='COMPLETE':
                            pose_rows.append(dict(meta,graph_status=row['status'],peg_estimable=False,hole_estimable=False,peg_end_pair_estimable=False,peg_position_below3mm=False,
                                                  peg_position_error_mm=None,peg_axis_error_deg=None,hole_center_error_mm=None));continue
                        folder=resolve_folder(row);key=str(folder)
                        if key not in graph_cache:graph_cache[key]=score_graph(folder,ref,mesh,s['id'],size)
                        scores,pose=graph_cache[key]
                        for score in scores:point_rows.append(dict(meta,**score,graph_status='COMPLETE'))
                        pose_rows.append(dict(meta,**pose,graph_status='COMPLETE'))
                        if repeat==0 and (folder/'stages.npz').exists():
                            with np.load(folder/'stages.npz',allow_pickle=False) as z:
                                for stage in ('raw','filtered','shape_first','cross_view','pre_budget','final'):
                                    arrays={k:z[stage+'__'+k] for k in ('xyz','object_id','camera','pixel','merged_sources')}
                                    for oid in (0,1):stage_rows.append(dict(meta,stage=stage,object_id=oid,**score_points(arrays,s['id'],size,oid,ref)))
            write(ROOT/('evaluation_progress'+suffix+'.json'),dict(states_done=si+1,states_total=len(specs),mask_rows=len(mask_rows),graph_point_rows=len(point_rows)))
            print('SCORE',si+1,'/',len(specs),s['id'],flush=True)
        for name,rows in [('mask_metrics',mask_rows),('scene_metrics',scene_rows),('graph_point_metrics',point_rows),
                          ('graph_pose_metrics',pose_rows),('stagewise_metrics',stage_rows)]:
            write(ROOT/(name+suffix+'.json'),dict(rows=rows));csvwrite(ROOT/(name+suffix+'.csv'),rows)
        for path,h in actual['receipts'].items():assert sha(ROOT/path)==h
        write(ROOT/('evaluation_freeze'+suffix+'.json'),dict(files={name+suffix+'.json':sha(ROOT/(name+suffix+'.json')) for name in
            ('mask_metrics','scene_metrics','graph_point_metrics','graph_pose_metrics','stagewise_metrics')},
            actual_graph_receipts_unchanged=True,GT_input_to_actual_pipeline=False,frozen_utc=now()))
        log.log(dict(mask_rows=len(mask_rows),pose_rows=len(pose_rows),stage_rows=len(stage_rows)))
        log.link_output(ROOT/('evaluation_freeze'+suffix+'.json'))


def merge(shards):
    for name in ('mask_metrics','scene_metrics','graph_point_metrics','graph_pose_metrics','stagewise_metrics'):
        rows=[]
        for shard in range(shards):rows.extend(read(ROOT/(f'{name}_shard{shard}.json'))['rows'])
        write(ROOT/(name+'.json'),dict(rows=rows));csvwrite(ROOT/(name+'.csv'),rows)
    write(ROOT/'evaluation_freeze.json',dict(files={name+ext:sha(ROOT/(name+ext)) for name in
        ('mask_metrics','scene_metrics','graph_point_metrics','graph_pose_metrics','stagewise_metrics') for ext in ('.json','.csv')},
        actual_graph_receipts_unchanged=True,GT_input_to_actual_pipeline=False,frozen_utc=now(),workers=shards))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--merge',action='store_true');a=ap.parse_args()
    if a.merge:merge(a.shards)
    else:main(a.shard,a.shards)
