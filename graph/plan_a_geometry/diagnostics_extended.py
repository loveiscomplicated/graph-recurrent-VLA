"""Evaluation-only condition counts, Main-visible coverage and near-finger audit."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.metrics import visible_reference,sample_meshes
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix
from .summarize import macro


def arrays(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def occluded_by_robot(camera,mesh,oid,gt):
    actor=pose_matrix(gt['actor_pose'][oid]);xyz=transform(mesh[oid]['points'][::15],actor)
    normals=mesh[oid]['normals'][::15]@actor[:3,:3].T
    front=((xyz-camera.world_from_camera[:3,3])*normals).sum(1)<0
    camera_xyz=transform(xyz,np.linalg.inv(camera.world_from_camera));K=camera.intrinsic
    uv=np.floor(camera_xyz[:,:2]/np.maximum(camera_xyz[:,2,None],1e-12)*np.array([K[0,0],K[1,1]])+K[:2,2]).astype(int)
    h,w=camera.depth.shape;inside=front&(camera_xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    rows=np.flatnonzero(inside);u,v=uv[rows].T;depth=camera.depth[v,u]
    robot=np.isin(gt[camera.camera+'/segmentation'][v,u],gt['robot_ids'])
    return int((robot&np.isfinite(depth)&(camera_xyz[rows,2]>depth+.006)).sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text());conditions=[];coverage=[];near=[];budget_rows=[];legacy_stages=[];camera_counts=[]
    for stage in ('development_geometry','final_geometry'):
        root=Path(state['completed'][stage]);manifest=json.loads((root/'geometry_manifest.json').read_text());capture=Path(manifest['config']['capture'])
        captured=json.loads((capture/'capture_manifest.json').read_text());gate=json.loads((root/'gate_trace.json').read_text());filterrows=json.loads((root/'filters.json').read_text())
        for ep in captured['episodes']:
            source=capture/'collection'/f'seed{ep["seed"]}';mesh=sample_meshes(source/'evaluation_meshes.npz');count=Counter();initial=None
            for frame in ep['frames']:
                t=frame['timestep'];cameras,data=load_cameras(source/(frame['stem']+'_input.npz'));gt=arrays(source/(frame['stem']+'_gt.npz'))
                if initial is None:initial=gt['actor_pose'][0,:3].copy()
                main=next(c for c in cameras if c.camera=='base_camera');wrist=next(c for c in cameras if c.camera=='hand_camera')
                gr=next(r for r in gate if r['seed']==ep['seed'] and r['timestep']==t and r['condition']=='baseline')
                count['frames']+=1;count['expert_phase_'+str(frame['phase'])]+=1
                count['main_predicted_peg_missing']+=gr['main_mask_points']==0;count['main_GT_peg_missing']+=gr['main_gt_visible']==0
                count['wrist_GT_peg_visible']+=gr['gt_wrist_visible']>0;count['wrist_only_candidate_observed']+=gr['wrist_only_recovery'];count['wrist_only_peg_recovered']+=bool(gr['wrist_only_recovery'] and gr['correct_wrist_after']>0)
                count['robot_occludes_projected_main_peg']+=occluded_by_robot(main,mesh,0,gt)>0
                count['robot_occludes_projected_wrist_peg']+=occluded_by_robot(wrist,mesh,0,gt)>0
                count['box_main_visible_and_robot_partly_occludes']+=bool((gt[main.camera+'/segmentation']==gt['actor_ids'][1]).any() and occluded_by_robot(main,mesh,1,gt)>0)
                count['gripper_opening_le_30mm']+=float(data['qpos'][-2:].sum())<=.03
                count['peg_translation_gt_gt_5mm']+=np.linalg.norm(gt['actor_pose'][0,:3]-initial)>.005
                count['peg_lift_gt_gt_5mm']+=gt['actor_pose'][0,2]-initial[2]>.005
                fr=[r for r in filterrows if r['seed']==ep['seed'] and r['timestep']==t and r['condition']=='baseline' and r['object_id']==0]
                count['finger_near_GT_peg_within20mm']+=any(r['contact_near_pixels']>0 for r in fr)
                count['cross_camera_identity_conflict']+=gr['identity_conflict_fraction']>0
            conditions.append(dict(stage=stage,seed=ep['seed'],split=ep['split'],**dict(count)))
        for f in manifest['frames']:
            if f['condition']!='baseline':continue
            folder=root/f['folder'];cameras,data=load_cameras(f['input_path']);gt=arrays(f['gt_path']);refs=visible_reference(cameras,gt,['base_camera'])
            filters=arrays(folder/'robot_filter.npz')
            for c in cameras:
                world=c.world_points();seg=gt[c.camera+'/segmentation']
                for oid in (0,1):
                    stage_data=arrays(folder/f'legacy_{c.camera}_object{oid}_stages.npz')
                    valid_eroded=int((stage_data['eroded_mask']&np.isfinite(c.depth)).sum())
                    clean_source=arrays(folder/'accepted_masks.npz')['C3/'+c.camera][oid]
                    legacy_stages.append(dict(seed=f['seed'],split=f['split'],timestep=f['timestep'],camera=c.camera,object_id=oid,
                        clean_valid=int((clean_source&np.isfinite(c.depth)).sum()),eroded_valid=valid_eroded,
                        erosion_fallback=bool(clean_source.any() and valid_eroded<50),
                        backproject_count=len(stage_data['backproject_source']),presample_count=len(stage_data['presample_source']),
                        SOR_count=len(stage_data['sor_source']),FPS_count=len(stage_data['source_pixel']),
                        FPS_reduced=bool(len(stage_data['sor_source'])>len(stage_data['source_pixel']))))
                    mask=(seg==gt['actor_ids'][oid])&np.isfinite(c.depth)
                    close=np.zeros_like(mask)
                    if mask.any():close[mask]=np.linalg.norm(world[mask,None]-data['robot_world_geometry'][None,1:],axis=-1).min(1)<.02
                    n=int(close.sum());confirmed=filters[c.camera+'/confirmed_robot'];uncertain=filters[c.camera+'/uncertain']
                    near.append(dict(seed=f['seed'],split=f['split'],timestep=f['timestep'],camera=c.camera,object_id=oid,near_pixels=n,
                        near_confirmed_removed=int((close&confirmed).sum()),near_uncertain_removed=int((close&uncertain).sum()),
                        near_total_preservation=float((close&~confirmed&~uncertain).sum()/n) if n else None))
            for path in folder.glob('C*_cap128_points.npz'):
                pp=arrays(path);variant=path.name.removesuffix('_cap128_points.npz')
                native=arrays(folder/(variant+'_native.npz'));nativesrc=arrays(folder/(variant+'_native_sources.npz'));capsrc=arrays(path.with_name(path.stem+'_sources.npz'))
                for mode,point_set,sources in [('native',native,nativesrc),('cap128',pp,capsrc)]+([('matched_N',arrays(folder/(variant+'_matched_N_points.npz')),arrays(folder/(variant+'_matched_N_points_sources.npz')))] if variant.startswith('C3_') else []):
                    first={i:np.flatnonzero(sources['node_row']==i)[0] for i in range(len(point_set['object_id']))}
                    for oid in (0,1):
                        rows=np.flatnonzero(point_set['object_id']==oid);names=[str(sources['camera'][first[i]]) for i in rows]
                        camera_counts.append(dict(seed=f['seed'],split=f['split'],timestep=f['timestep'],variant=variant,budget=mode,object_id=oid,points=len(rows),main_representative_nodes=names.count('base_camera'),wrist_representative_nodes=names.count('hand_camera'),wrist_node_fraction=names.count('hand_camera')/len(rows) if len(rows) else None))
                for oid in (0,1):
                    xyz=pp['world_xyz'][pp['object_id']==oid];reference=refs[oid]
                    cov=float((cKDTree(xyz).query(reference)[0]<.004).mean()) if len(xyz) and len(reference) else 0. if len(reference) else None
                    coverage.append(dict(seed=f['seed'],split=f['split'],timestep=f['timestep'],variant=variant,object_id=oid,main_visible_reference=len(reference),main_visible_coverage_4mm=cov))
                    budget_rows.append(dict(seed=f['seed'],split=f['split'],timestep=f['timestep'],variant=variant,object_id=oid,
                        native_points=int((native['object_id']==oid).sum()),cap_points=int((pp['object_id']==oid).sum()),
                        native_structural=int((nativesrc['structural']&(native['object_id']==oid)).sum()),
                        cap_structural=int((capsrc['structural']&(pp['object_id']==oid)).sum())))
    for name,rows in [('condition_counts',conditions),('main_visible_coverage',coverage),('finger_near_removal',near),('structural_budget',budget_rows),('legacy_stage_counts',legacy_stages),('representative_camera_counts',camera_counts)]:
        write_json(a.output/(name+'.json'),rows);csv_write(a.output/(name+'.csv'),rows)
    summaries=dict(camera_representation=macro(camera_counts,('split','variant','budget','object_id'),('main_representative_nodes','wrist_representative_nodes','wrist_node_fraction')),main_visible=macro(coverage,('split','variant','object_id'),('main_visible_coverage_4mm',)),
        finger_near=macro(near,('split','camera','object_id'),('near_total_preservation','near_confirmed_removed','near_uncertain_removed')),
        structural=macro(budget_rows,('split','variant','object_id'),('native_points','cap_points','native_structural','cap_structural')))
    write_json(a.output/'summary.json',summaries)
if __name__=='__main__':main()
