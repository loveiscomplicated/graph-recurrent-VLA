"""Observed negatives and labelled corruption tests on recorded real inputs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from graph.integration_baseline.io import save_npz,write_json
from graph.integration_baseline.geometry09b_probe import csv_write
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger
from .evaluate import load_cameras
from .render import read_npz,table
from .contracts import transform
from .metrics import sample_meshes
from .history import restore_history
from .audit import protection_snapshot,verify_protection
from .collect import inventory_sources


def actual_reset(capture,parent,episodes):
    """Reuse a real previous map, then reset on a different recorded episode."""
    if len(episodes)<2:return dict(status='NOT_TESTED',reason='one physical episode')
    from .pipeline import GeometrySystem
    from .self_filter import LinkSurface
    old,new=episodes[:2]
    memories={name:restore_history(parent,old,name,len(old['frames'])-1,old['episode']) for name in ('G2','G4')}
    old_ids={r['persistent_node_id'] for memory in memories.values() for r in memory.records()}
    folder=capture/'collection'/f'seed{new["seed"]}'
    cameras,data=load_cameras(folder/'f0000_input.npz');masks=read_npz(folder/'f0000_masks.npz')
    mesh=read_npz(folder/'robot_meshes.npz')
    surfaces=[LinkSurface(str(link),mesh[f'vertices_{i}'],data['robot_link_world'][i]) for i,link in enumerate(mesh['links'])]
    contract=json.loads((folder/'sensor_contract.json').read_text())['roles']
    reused=GeometrySystem(contract['main'],contract['wrist'],memories['G2'],memories['G4']).process(cameras,data,masks,surfaces)
    fresh=GeometrySystem(contract['main'],contract['wrist']).process(cameras,data,masks,surfaces)
    counts={}
    for variant in ('G2','G4'):
        a,b=reused['records'][variant],fresh['records'][variant]
        if [r['persistent_node_id'] for r in a]!=[r['persistent_node_id'] for r in b]:raise ValueError('real reset fresh ID mismatch')
        if old_ids&{r['persistent_node_id'] for r in a}:raise ValueError('old real map IDs leaked after reset')
        np.testing.assert_array_equal(np.array([r['world_xyz'] for r in a]),np.array([r['world_xyz'] for r in b]))
        counts[variant]=len(a)
    return dict(status='PASS',old_seed=old['seed'],new_seed=new['seed'],
        scope='actual sensor-derived previous map reused at a new recorded episode boundary; no GT inputs',
        old_ids_leaked=0,fresh_state_exact=True,new_counts=counts)


def analyze(capture,geometry,destination):
    manifest=json.loads((geometry/'geometry_manifest.json').read_text())
    collected=json.loads((capture/'capture_manifest.json').read_text())
    parent=Path(manifest['config'].get('geometry',geometry))
    by_frame={(f['seed'],f['timestep'],f.get('condition','baseline')):f for f in manifest['frames']}
    hidden=[];moves=[];purity=[];resets=[];failures=[];discoveries=[]
    meshes={}
    for episode in collected['episodes']:
        seed=episode['seed'];source=capture/'collection'/f'seed{seed}'
        mesh=meshes.setdefault(seed,sample_meshes(source/'evaluation_meshes.npz'))
        for frame in episode['frames']:
            t=frame['timestep'];gt=read_npz(source/f'f{t:04d}_gt.npz')
            masks=read_npz(source/f'f{t:04d}_masks.npz')
            geom=geometry/by_frame[(seed,t,'baseline')]['folder']
            for oid,actor in enumerate(gt['actor_ids']):
                visible=sum(int((gt[name+'/segmentation']==actor).sum()) for name in masks)
                for variant in ('G2','G4','O1','O2','O3'):
                    a=read_npz(geom/(variant+'_map.npz'));active=(a['object_id']==oid)&(a['status']!=3)
                    if visible==0:
                        state=json.loads((geom/(variant+'_tracker.json')).read_text())['objects'][str(oid)]
                        historical=a['estimated_only']&active
                        local=transform(a['world_xyz'][active],np.linalg.inv(pose_matrix(gt['actor_pose'][oid])))
                        distance=mesh[oid]['tree'].query(local)[0]
                        hidden.append(dict(seed=seed,timestep=t,object_id=oid,variant=variant,
                            GT_visible_both=0,active=int(active.sum()),
                            incorrectly_marked_current=int((a['currently_observed']&active).sum()),
                            historical=int(historical.sum()),tracking_state=state['tracking_state'],
                            pose_confidence=state['pose_confidence'],pose_uncertainty_mm=state['pose_uncertainty']*1000,
                            correct_4mm=int((distance<.004).sum()),incorrect_8mm=int((distance>.008).sum())))
                    if t==0 and variant in ('G2','G4'):
                        resets.append(dict(seed=seed,variant=variant,object_id=oid,initial_ids=a['persistent_node_id'][active].tolist()))
                for name,pred in masks.items():
                    selected=pred[oid];seg=gt[name+'/segmentation'];den=int(selected.sum())
                    robot=int((selected&np.isin(seg,gt['robot_ids'])).sum())
                    if robot:purity.append(dict(seed=seed,timestep=t,camera=name,object_id=oid,
                        predicted_pixels=den,robot_pixels=robot,robot_contamination=float(robot/den)))
        for variant in ('G2','G4','O1','O2','O3'):
            events=json.loads((parent/'geometry'/f'seed{seed}'/(variant+'_lifecycle_trace.json')).read_text())
            failures.extend(dict(seed=seed,variant=variant,**event) for event in events if any(word in event['event'] for word in ('FAILED','REJECT','JUMP','INSUFFICIENT')))
        for branch in episode.get('branches',[]):
            if not branch['physical_valid'] or branch['condition']=='baseline_repeat':continue
            t=branch['timestep'];oid=0 if branch['object']=='peg' else 1
            gt=read_npz(source/branch['source']/'gt.npz');old=read_npz(source/f'f{t:04d}_gt.npz')
            count=sum(int((gt[n+'/segmentation']==gt['actor_ids'][oid]).sum()) for n in masks)
            oldcount=sum(int((old[n+'/segmentation']==old['actor_ids'][oid]).sum()) for n in masks)
            if count or oldcount:continue
            a=read_npz(source/branch['source']/'input.npz');b=read_npz(source/f'f{t:04d}_input.npz')
            same=all(np.array_equal(a[name+'/'+key],b[name+'/'+key]) for name in masks for key in ('rgb','depth_raw_mm'))
            dependency=[];oracle_shift=None
            if same:
                baseline_folder=geometry/by_frame[(seed,t,'baseline')]['folder']
                branch_folder=geometry/by_frame[(seed,t,branch['condition'])]['folder']
                for variant in ('G0','G1','G2','G3','G4'):
                    old_graph=read_npz(baseline_folder/(variant+'_graph.npz'))
                    new_graph=read_npz(branch_folder/(variant+'_graph.npz'))
                    maximum=0.
                    for field,values in old_graph.items():
                        if values.dtype.kind in 'fc':
                            np.testing.assert_allclose(values,new_graph[field],atol=1e-6,rtol=1e-5)
                            maximum=max(maximum,float(np.max(abs(values-new_graph[field]))) if values.size else 0.)
                        else:np.testing.assert_array_equal(values,new_graph[field])
                    dependency.append(dict(variant=variant,practical_graph_unchanged=True,max_absolute_error=maximum))
                old_map=read_npz(baseline_folder/'O3_map.npz');new_map=read_npz(branch_folder/'O3_map.npz')
                old_index={str(identity):i for i,identity in enumerate(old_map['persistent_node_id']) if old_map['object_id'][i]==oid and old_map['status'][i]!=3}
                shift=[np.linalg.norm(new_map['world_xyz'][i]-old_map['world_xyz'][old_index[str(identity)]])*1000
                    for i,identity in enumerate(new_map['persistent_node_id']) if str(identity) in old_index]
                oracle_shift=float(np.mean(shift)) if shift else None
            moves.append(dict(seed=seed,timestep=t,condition=branch['condition'],object_id=oid,
                actual_translation_mm=float(np.linalg.norm(gt['actor_pose'][oid,:3]-old['actor_pose'][oid,:3])*1000),
                same_robot=branch['robot_state_exact'],same_calibration=branch['calibration_exact'],
                RGBD_identical=bool(same),both_baseline_branch_fully_hidden=True,
                practical_graphs_identical_to_tolerance=all(r['practical_graph_unchanged'] for r in dependency) if same else None,
                practical_dependency_checks=json.dumps(dependency),O3_historical_node_displacement_mm=oracle_shift,
                physical_valid=True,no_physics_step=branch['no_physics_step'],
                scope='actual actor pose intervention and render; no dynamic PhysX contact-force claim'))
    # Real calibrated observations with coherent CV+GL corruption: checking the
    # convention alone must not be mistaken for checking the moving mount.
    episode=collected['episodes'][0];source=capture/'collection'/f'seed{episode["seed"]}'
    first,data=load_cameras(source/'f0000_input.npz')
    contract=json.loads((source/'sensor_contract.json').read_text());wrist=contract['roles']['wrist']
    original=next(c for c in first if c.camera==wrist)
    with np.load(source/'robot_meshes.npz',allow_pickle=False) as a:links=a['links'].tolist()
    mount=contract['installed'][wrist]['mount_link_name']
    mount_inverse=np.linalg.inv(data['robot_link_world'][links.index(mount)])
    anchor=mount_inverse@original.world_from_camera
    world=original.world_from_camera.astype(float);world[0,3]+=.05
    changed=replace(original,extrinsic_cv=np.linalg.inv(world)[:3],cam2world_gl=world@np.diag([1.,-1,-1,1]))
    changed.validate()
    mount_error=float(np.max(abs(mount_inverse@changed.world_from_camera-anchor)))
    if mount_error<=2e-5:raise ValueError('calibration corruption was not detected')
    wrong_alignment=dict(status='PASS',scope='controlled fault on recorded real calibration',
        CV_GL_still_consistent=True,mount_anchor_detects_error=True,injected_translation_mm=50.,
        maximum_matrix_element_difference=mount_error)
    # Copy a real past map into a fresh fixture and inject a future source event.
    fixture=destination/'wrong_history_fixture';folder=fixture/'geometry'/f'seed{episode["seed"]}'/'f0000'
    folder.mkdir(parents=True,exist_ok=False)
    actual=parent/'geometry'/f'seed{episode["seed"]}'/'f0000'
    for suffix in ('_map.npz','_tracker.json'):
        shutil.copyfile(actual/('G4'+suffix),folder/('G4'+suffix))
    events=read_npz(actual/'G4_source_observations.npz');events['source_timestep']+=1
    save_npz(folder/'G4_source_observations.npz',**events)
    try:restore_history(fixture,episode,'G4',0,episode['episode']+':bad_future')
    except ValueError as error:
        if 'future' not in str(error):raise
    else:raise ValueError('actual-history future injection accepted')
    # Actual reset IDs must be disjoint even when different seeds share shapes.
    for i,row in enumerate(resets):
        for other in resets[:i]:
            if row['seed']!=other['seed'] and row['variant']==other['variant'] and set(row['initial_ids'])&set(other['initial_ids']):raise ValueError('actual reset identity leak')
    metric_rows=table(parent/'geometry_metrics.csv');reappearances=[]
    for row in metric_rows:
        if row['budget_mode']=='native_variable_N' and float(row.get('new_correct_previously_unseen_nodes') or 0)>0:
            discoveries.append(row)
    for episode in collected['episodes']:
        for variant in ('G2','G4','O1','O3'):
            for oid in ('0','1'):
                rr=sorted([r for r in metric_rows if int(r['seed'])==episode['seed'] and r['variant']==variant
                    and r['object_id']==oid and r['budget_mode']=='native_variable_N'],key=lambda r:int(r['timestep']))
                for index in range(1,len(rr)):
                    if int(rr[index-1]['current_visible_reference'])!=0 or int(rr[index]['current_visible_reference'])==0:continue
                    start=int(rr[index]['timestep']);recovery=None
                    for later in rr[index:]:
                        if int(later['current_visible_reference'])==0:break
                        if later['surface_distance_mean_mm'] and float(later['surface_distance_mean_mm'])<=4 and float(later['current_visible_coverage_4mm'])>=.5:
                            recovery=int(later['timestep'])-start;break
                    reappearances.append(dict(seed=episode['seed'],variant=variant,object_id=oid,reappeared_timestep=start,
                        recovery_frames=recovery,status='RECOVERED' if recovery is not None else 'NOT_RECOVERED_WITHIN_VISIBLE_INTERVAL',
                        criterion='native-map mean mesh distance<=4mm and own-current-visible coverage>=0.5',
                        input_sensor_scope=rr[index]['input_sensor_scope']))
    csv_write(destination/'full_occlusion_cases.csv',hidden);csv_write(destination/'hidden_motion_cases.csv',moves)
    csv_write(destination/'actual_wrong_mask_cases.csv',purity);csv_write(destination/'actual_registration_failures.csv',failures)
    csv_write(destination/'actual_new_surface_cases.csv',discoveries)
    csv_write(destination/'reappearance_recovery.csv',reappearances)
    reset_result=actual_reset(capture,parent,collected['episodes'])
    write_json(destination/'actual_reset_inference.json',reset_result)
    summary=dict(negative1=dict(scope='actual RGB-D plus same-time GT visibility, not mask absence',
        hidden_object_frames=len({(r['seed'],r['timestep'],r['object_id']) for r in hidden}),
        practical_false_observed_cases=sum(r['incorrectly_marked_current']>0 and r['variant'] in ('G2','G4') for r in hidden)),
        negative2=dict(scope='actual same-robot actor intervention and rerender',pairs=moves,
            note='identical RGB-D cannot identify hidden translation; increasing uncertainty is the expected practical behavior'),
        negative3=dict(scope='actual predicted masks evaluated against GT robot IDs',contaminated_masks=len(purity)),
        negative4=dict(scope='actual estimator rejection/reappearance metrics; large-disagreement synthetic fixture additionally tested',events=len(failures),reappearance_cases=len(reappearances)),
        negative5=dict(scope='actual RGB-D node additions evaluated against past GT-visible support',new_surface_rows=len(discoveries),
            persistent_new_surface_rows=sum(r['variant'] in ('G2','G4','O1','O2','O3') for r in discoveries)),
        negative6=dict(scope='actual episode reset identity disjointness',
            status=reset_result['status'],episodes=len(collected['episodes']),actual_reset_inference=reset_result),
        negative7=wrong_alignment,negative8=dict(status='PASS',scope='controlled future event injected into copied real snapshot',original_untouched=True))
    write_json(destination/'negative_experiments.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','geometry','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    protected=protection_snapshot(p for parent in (args.capture,args.geometry) for p in parent.rglob('*') if p.is_file())
    with ExperimentLogger(team='integration',config=dict(task='09-C',stage='negative_experiments',capture=str(args.capture),geometry=str(args.geometry)),
            seed=42,model='task09c_negative_cases',job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_NEGATIVES_DIRECTORY='+str(root),flush=True)
        inventory_sources(root);write_json(root/'protection_before.json',protected)
        analyze(args.capture,args.geometry,root)
        result=verify_protection(protected);write_json(root/'protection_audit.json',result)
        if result['status']!='PASS':raise ValueError('negative tests changed source')
        logger.link_output(root/'negative_experiments.json',kind='metrics')


if __name__=='__main__':main()
