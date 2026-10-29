"""Same-prefix motion/occlusion/reappearance evaluation; no GT correction."""
import argparse,json,copy
from pathlib import Path
from dataclasses import replace
import numpy as np,torch
from graph.integration_baseline.io import write_json,save_npz,sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.metrics import sample_meshes
from graph.persistent_geometry.contracts import transform
from graph.plan_a_geometry.core import PlanA
from graph.plan_a_geometry.evaluate import save_records
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger
from .core import BoxMemory
from .evaluate import coverage


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def rebased(memory,episode,t):
    m=copy.deepcopy(memory)
    if t==0:m.reset(episode)
    else:m.episode=episode;m.clock.episode=episode
    return m


def measured(memory,gt,meshes,main):
    rr=memory.active_records();xyz=np.array([r['world_xyz'] for r in rr]).reshape(-1,3)
    distance=meshes[1]['tree'].query(transform(xyz,np.linalg.inv(pose_matrix(gt['actor_pose'][1]))))[0]
    ref=main.world_points()[(gt[main.camera+'/segmentation']==gt['actor_ids'][1])&np.isfinite(main.depth)]
    estimated=np.array([r['estimated_only'] for r in rr],bool)
    return dict(nodes=len(rr),estimated_nodes=int(estimated.sum()),current_nodes=sum(r['currently_observed'] for r in rr),
        surface_mean_mm=float(distance.mean()*1000) if len(distance) else None,
        surface_p95_mm=float(np.percentile(distance,95)*1000) if len(distance) else None,
        ghost_nodes=int(((distance>.008)&estimated).sum()),current_visible_coverage=coverage(xyz,ref),
        gt_visible_pixels=len(ref),uncertainty_mean_mm=float(np.mean([r['position_uncertainty'] for r in rr])*1000) if rr else None,
        low_confidence_nodes=sum(r['status']==2 for r in rr))


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initialize-t0-baseline',action='store_true');p.add_argument('--only-initial',action='store_true');a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());acfg=json.loads(Path(__file__).parents[1].joinpath('plan_a_geometry/protocol.json').read_text())
    manifest=json.loads((a.capture/'capture_manifest.json').read_text());torch.set_num_threads(1)
    with ExperimentLogger(team='integration',config=dict(task='09-E',stage='motion_evaluation',capture=str(a.capture),
        reappearance='same physical timestep visible branch re-presented one diagnostic tick after full occlusion; not independent temporal physics',
        rules_frozen=True,policy_training=False,initialize_t0_baseline=a.initialize_t0_baseline,only_initial=a.only_initial,
        support_amendment='optional separate diagnostic: baseline t0 RGB-D initializes map at tick0, actual same-robot t0 intervention at tick1; original no-history run preserved'),seed=42,model='task09e_motion_evaluation',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True)
        rows=[];audits=[]
        for ep in manifest['episodes']:
            source=a.capture/'collection'/f'seed{ep["seed"]}';geom=load(source/'robot_meshes.npz');meshes=sample_meshes(source/'evaluation_meshes.npz')
            system=PlanA(ep['roles']['main'],ep['roles']['wrist'],acfg);maps={n:BoxMemory(n,cfg['memory']) for n in ('B1','B2')}
            def prepare(input_path,mask_path):
                cameras,data=load_cameras(input_path);masks=load(mask_path)
                surfaces=[LinkSurface(n,geom[f'vertices_{i}'],data['robot_link_world'][i]) for i,n in enumerate(geom['links'])]
                prepared=system.prepare(cameras,data,masks,surfaces);pools,_,_,_=system.represent(cameras,prepared,'C3',('SF',))
                return cameras,data,[r for r in pools['SF'] if r['object_id']==1]
            for f in ep['frames']:
                if a.only_initial and f['timestep']>0:break
                t=f['timestep'];cameras,data,box=prepare(source/(f['stem']+'_input.npz'),source/(f['stem']+'_masks.npz'))
                main_camera=next(c for c in cameras if c.camera=='base_camera');prefix=copy.deepcopy(maps)
                for n,m in maps.items():m.update_records(main_camera.episode,t,box,main_camera)
                if t==0 and a.initialize_t0_baseline:prefix=copy.deepcopy(maps)
                diagnostic_t=1 if t==0 and a.initialize_t0_baseline else t
                branches=[b for b in ep['branches'] if b['timestep']==t and b['object']=='box']
                for b in branches:
                    folder=source/b['source']
                    for suffix,key in (('input.npz','input_sha256'),('masks.npz','masks_sha256'),('gt.npz','gt_sha256')):
                        if sha256(folder/suffix)!=b[key]:raise ValueError('branch identity changed')
                    cs,bd,br=prepare(folder/'input.npz',folder/'masks.npz');bc=next(c for c in cs if c.camera=='base_camera')
                    if diagnostic_t!=t:
                        br=copy.deepcopy(br)
                        for r in br:r['source_observation_ids']=[(s[0],diagnostic_t,s[2],s[3]) for s in r['source_observation_ids']]
                        bc=replace(bc,timestep=diagnostic_t,timestamp=diagnostic_t/20)
                    # Actual hidden RGB-D is generated by the visual occluder, not array masking.
                    gt=load(folder/'gt.npz');delta=np.array(b['translation_world_m']);condition=b['condition']
                    for n in maps:
                        branch=rebased(prefix[n],bc.episode,diagnostic_t)
                        branch.update_records(bc.episode,diagnostic_t,br,bc)
                        out=root/f't{t:04d}'/condition/n;out.mkdir(parents=True)
                        save_records(out/'points.npz',branch.active_records());write_json(out/'trace.json',branch.trace[-1])
                        pose=branch.objects[1].world_from_object
                        baseline_pose=maps[n].objects[1].world_from_object
                        actor=gt['actor_pose'][1,:3]-delta
                        predicted=transform(actor,pose@np.linalg.inv(baseline_pose))-actor
                        row=dict(seed=ep['seed'],parent_split=ep['split'],physical_timestep=t,diagnostic_tick=diagnostic_t,
                            condition=condition,mode=n,phase='intervention',physical_valid=b['physical_valid'],
                            prior_nodes=len(prefix[n].active_records()),
                            imposed_delta_mm=float(np.linalg.norm(delta)*1000),estimated_delta_mm=float(np.linalg.norm(predicted)*1000),
                            pose_delta_error_mm=float(np.linalg.norm(predicted-delta)*1000),
                            detected=bool(branch.trace[-1]['static_violation'] or branch.trace[-1]['state']=='TRACKING_FAILURE_OR_MOTION_UNKNOWN'),
                            state=branch.trace[-1]['state'],static_violation=branch.trace[-1]['static_violation'],
                            **measured(branch,gt,meshes,bc))
                        rows.append(row)
                        if b.get('occlusion')=='full' and np.linalg.norm(delta)>0:
                            visible_name=condition.replace('box_full_','box_')
                            visible=next(x for x in branches if x['condition']==visible_name)
                            vf=source/visible['source'];vc,vd,vr=prepare(vf/'input.npz',vf/'masks.npz');vmain=next(c for c in vc if c.camera=='base_camera')
                            # Re-present actual saved visible RGB-D at a diagnostic next tick.
                            vr=copy.deepcopy(vr)
                            for r in vr:
                                r['source_observation_ids']=[(bc.episode,diagnostic_t+1,s[2],s[3]) for s in r['source_observation_ids']]
                            vmain=replace(vmain,episode=bc.episode,timestep=diagnostic_t+1,timestamp=(diagnostic_t+1)/20)
                            branch.update_records(bc.episode,diagnostic_t+1,vr,vmain);vgt=load(vf/'gt.npz')
                            save_records(out/'reappearance_points.npz',branch.active_records());write_json(out/'reappearance_trace.json',branch.trace[-1])
                            rows.append(dict(seed=ep['seed'],parent_split=ep['split'],physical_timestep=t,diagnostic_tick=diagnostic_t+1,
                                condition=condition,mode=n,phase='reappearance_represented_snapshot',physical_valid=visible['physical_valid'],
                                imposed_delta_mm=float(np.linalg.norm(delta)*1000),state=branch.trace[-1]['state'],
                                static_violation=branch.trace[-1]['static_violation'],
                                detected=bool(branch.trace[-1]['static_violation'] or branch.trace[-1]['state']=='TRACKING_FAILURE_OR_MOTION_UNKNOWN'),
                                **measured(branch,vgt,meshes,vmain)))
                    if b.get('occlusion')=='full':
                        valid_hidden=b['main_gt_visible_pixels']==0
                        if np.linalg.norm(delta)>0:
                            valid_hidden &= b['main_RGB_same_as_occluded_baseline'] and b['main_depth_same_as_occluded_baseline'] and b['main_mask_same_as_occluded_baseline']
                        audits.append(dict(timestep=t,condition=condition,hidden_sensor_indistinguishable=bool(valid_hidden),**{k:v for k,v in b.items() if k.startswith('main_')}))
                if t%10==0:print(f'motion evaluation t={t} branches={len(branches)}',flush=True)
        csv_write(root/'motion_metrics.csv',rows);write_json(root/'hidden_controls.json',audits)
        write_json(root/'motion_manifest.json',dict(capture=str(a.capture),capture_sha256=sha256(a.capture/'capture_manifest.json'),
            rows=len(rows),hidden_negative_control_pass=bool(audits) and all(x['hidden_sensor_indistinguishable'] for x in audits),
            reappearance_scope='actual visible actor intervention snapshot re-presented after actual occluded snapshot; no new physics/contact-force claim'))
        log.link_output(root/'motion_metrics.csv',kind='metrics');log.link_output(root/'motion_manifest.json',kind='metrics')
if __name__=='__main__':main()
