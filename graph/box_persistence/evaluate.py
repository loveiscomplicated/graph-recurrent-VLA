"""Paired all-frame maps, interval geometry/graphs; GT strictly evaluation-only."""
import argparse,hashlib,json,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from graph.integration_baseline.io import write_json,save_npz,sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.evaluate import load_cameras,save_graph
from graph.persistent_geometry.metrics import sample_meshes,visible_reference,provenance_metrics,filter_metrics
from graph.persistent_geometry.geometry_audit import aperture_reference,graph_audit
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.collect import inventory_sources
from graph.plan_a_geometry.core import PlanA,budget
from graph.plan_a_geometry.evaluate import save_records
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger
from .core import BoxMemory,paired_records
from .oracle import OracleBoxMemory


def arrays_hash(records):
    h=hashlib.sha256()
    for r in sorted(records,key=lambda r:r['persistent_node_id']):
        for k in ('world_xyz','world_normal','last_f5','last_feature_valid'):
            h.update(np.asarray(r[k]).tobytes())
        h.update(json.dumps(r['source_observation_ids']).encode())
    return h.hexdigest()


def coverage(xyz,reference):
    return float((cKDTree(xyz).query(reference)[0]<.004).mean()) if len(xyz) and len(reference) else 0. if len(reference) else None


def metric(records,gt,meshes,refs,historical,boundary,historical_boundary,aperture,t,gt_history,cache):
    rr=[r for r in records if r['object_id']==1]
    xyz=np.array([r['world_xyz'] for r in rr]).reshape(-1,3)
    distance=meshes[1]['tree'].query(transform(xyz,np.linalg.inv(pose_matrix(gt['actor_pose'][1]))))[0]
    estimated=np.array([r['estimated_only'] for r in rr],bool)
    purity=provenance_metrics('B',rr,gt_history,cache,t)[1]
    hole,radius,xlo,xhi,rim=aperture
    local=transform(xyz,np.linalg.inv(hole))
    intrusion=(abs(local[:,1])<radius-.001)&(abs(local[:,2])<radius-.001)&(local[:,0]>xlo+.001)&(local[:,0]<xhi-.001)
    row=dict(count=len(rr),surface_distance_mean_mm=float(distance.mean()*1000) if len(rr) else None,
        surface_distance_median_mm=float(np.median(distance)*1000) if len(rr) else None,
        surface_distance_p90_mm=float(np.percentile(distance,90)*1000) if len(rr) else None,
        surface_distance_p95_mm=float(np.percentile(distance,95)*1000) if len(rr) else None,
        surface_support_4mm=float((distance<.004).mean()) if len(rr) else None,
        ghost_nodes=int(((distance>.008)&estimated).sum()),
        estimated_nodes=int(estimated.sum()),estimated_ghost_fraction=float((distance[estimated]>.008).mean()) if estimated.any() else 0.,
        geometry_outlier_fraction_8mm=float((distance>.008).mean()) if len(rr) else None,
        current_visible_coverage=coverage(xyz,refs),historical_coverage=coverage(xyz,historical),
        current_boundary_coverage=coverage(xyz,boundary),historical_boundary_coverage=coverage(xyz,historical_boundary),
        current_rim_coverage=coverage(xyz,rim[cKDTree(refs).query(rim)[0]<.004]) if len(refs) else None,
        historical_rim_coverage=coverage(xyz,rim[cKDTree(historical).query(rim)[0]<.004]) if len(historical) else None,
        aperture_intrusion_nodes=int(intrusion.sum()),low_confidence_fraction=float(np.mean([r['status']==2 for r in rr])) if rr else None,
        current_nodes=sum(r['currently_observed'] for r in rr),
        current_reference_points=len(refs),historical_reference_points=len(historical),
        box_wrist_sources=sum(s[2]!='base_camera' for r in rr for s in r['source_observation_ids']),**purity)
    if rr:
        u=np.array([r['position_uncertainty'] for r in rr]);row['uncertainty_covers_error_fraction']=float((distance<=u).mean())
        row['uncertainty_error_correlation']=float(np.corrcoef(u,distance)[0,1]) if np.std(u)>1e-9 and np.std(distance)>1e-9 else None
    return row


def unique_history(parts):
    xyz=np.concatenate(parts) if parts else np.empty((0,3))
    if not len(xyz):return xyz
    # Observed-surface voxels are a set, not repeated-frame weights; 1mm canonical bins.
    _,index=np.unique(np.floor(xyz/.001).astype(int),axis=0,return_index=True)
    return xyz[np.sort(index)]


def run_episode(args):
    capture,root,ep,cfg,baseline=args
    torch.set_num_threads(1)
    source=capture/'collection'/f'seed{ep["seed"]}'
    output=root/'geometry'/f'seed{ep["seed"]}';output.mkdir(parents=True,exist_ok=False)
    roles=json.loads((source/'sensor_contract.json').read_text())['roles']
    system=PlanA(roles['main'],roles['wrist'],baseline)
    with np.load(source/'robot_meshes.npz',allow_pickle=False) as a:
        links=a['links'].tolist();vertices=[a[f'vertices_{i}'].copy() for i in range(len(links))]
    meshes=sample_meshes(source/'evaluation_meshes.npz')
    maps={n:BoxMemory(n if n in ('B1','B2') else 'B1',cfg['memory'],roles['main']) for n in ('B1','B2','O1')}
    maps.update({n:OracleBoxMemory(cfg['memory'],roles['main']) for n in ('O2','O3')})
    rows=[];edge_rows=[];motion=[];frames=[];audits=[];filters=[];history=[];boundary_history=[];gt_history={};purity_cache={}
    initial_gt=None;initial_map_gt={};initial_map_pose={}
    for f in ep['frames']:
        t=f['timestep'];stem=f['stem']
        for suffix,key in (('_input.npz','input_sha256'),('_masks.npz','mask_sha256'),('_gt.npz','gt_sha256')):
            if sha256(source/(stem+suffix))!=f[key]:raise ValueError('capture hash changed')
        cameras,data=load_cameras(source/(stem+'_input.npz'))
        with np.load(source/(stem+'_masks.npz'),allow_pickle=False) as a:masks={k:a[k].copy() for k in a.files}
        surfaces=[LinkSurface(n,v,p) for n,v,p in zip(links,vertices,data['robot_link_world'])]
        prepared=system.prepare(cameras,data,masks,surfaces)
        pool,_,_,_=system.represent(cameras,prepared,'C3',('SF',));current=pool['SF']
        box=[r for r in current if r['object_id']==1];peg_hash=arrays_hash([r for r in current if r['object_id']==0])
        main=next(c for c in cameras if c.camera==roles['main'])
        for n in ('B1','B2'):maps[n].update_records(main.episode,t,box,main)
        # The practical input and maps above have completed before opening GT.
        with np.load(source/(stem+'_gt.npz'),allow_pickle=False) as a:gt={k:a[k].copy() for k in a.files}
        gt_history[t]=gt;gt_pose=pose_matrix(gt['actor_pose'][1])
        if initial_gt is None:initial_gt=gt_pose.copy()
        relative=gt_pose@np.linalg.inv(initial_gt)
        motion.append(dict(seed=ep['seed'],timestep=t,translation_from_initial_mm=float(np.linalg.norm(gt_pose[:3,3]-initial_gt[:3,3])*1000),
            rotation_from_initial_rad=float(Rotation.from_matrix(relative[:3,:3]).magnitude()),
            contact_force='NOT_TESTED; source sidecar has no contact force'))
        oracle_masks={k:v.copy() for k,v in masks.items()}
        oracle_masks[roles['main']][1]=gt[roles['main']+'/segmentation']==gt['actor_ids'][1]
        op=system.prepare(cameras,data,oracle_masks,surfaces)
        oracle_pool,_,_,_=system.represent(cameras,op,'C3',('SF',))
        obox=[r for r in oracle_pool['SF'] if r['object_id']==1]
        maps['O1'].update_records(main.episode,t,obox,main)
        maps['O2'].update_oracle(main.episode,t,box,main,gt_pose)
        maps['O3'].update_oracle(main.episode,t,obox,main,gt_pose)
        for n,m in maps.items():
            if m.objects[1].active() and n not in initial_map_gt:
                initial_map_gt[n]=gt_pose.copy();initial_map_pose[n]=m.objects[1].world_from_object.copy()
            if n in initial_map_gt:
                expected=gt_pose@np.linalg.inv(initial_map_gt[n])@initial_map_pose[n]
                obj=m.objects[1];center=obj.world_points().mean(0)
                local_center=transform(center,np.linalg.inv(obj.world_from_object))
                m.trace[-1].update(evaluation_pose_center_error_mm=float(np.linalg.norm(center-transform(local_center,expected))*1000),
                    evaluation_pose_rotation_error_rad=float(Rotation.from_matrix(obj.world_from_object[:3,:3]@expected[:3,:3].T).magnitude()))
        refs=visible_reference([main],gt)[1]
        actor_inv=np.linalg.inv(gt_pose)
        history.append(transform(refs,actor_inv))
        correct=(gt[main.camera+'/segmentation']==gt['actor_ids'][1])&np.isfinite(main.depth)
        edge=correct & ~(cv2.erode(correct.astype('uint8'),np.ones((3,3),'uint8'))>0)
        boundary=main.world_points()[edge];boundary_history.append(transform(boundary,actor_inv))
        if t%cfg['geometry_interval']!=0 and f!=ep['frames'][-1]:continue
        folder=output/stem;folder.mkdir()
        filters.extend(dict(seed=ep['seed'],timestep=t,**r) for r in filter_metrics(main,masks[main.camera],prepared['filters'][main.camera],gt))
        save_npz(folder/'robot_filter.npz',**{k:v for k,v in prepared['filters'][main.camera].items() if isinstance(v,np.ndarray)})
        save_npz(folder/'accepted_masks.npz',main_box=prepared['conditions']['C3'][main.camera][1])
        native={'B0':current,**{n:paired_records(current,m.active_records()) for n,m in maps.items()}}
        matched=min(128,*[sum(r['object_id']==1 for r in native[n]) for n in ('B0','B1','B2')])
        historical=transform(unique_history(history),gt_pose);historical_boundary=transform(unique_history(boundary_history),gt_pose)
        aperture=aperture_reference(meshes,gt)
        save_npz(folder/'evaluation_reference.npz',main_visible=refs,historical_observed=historical,
            current_boundary=boundary,historical_boundary=historical_boundary,rim=aperture[-1],actor_pose=gt['actor_pose'][1])
        for n,rr in native.items():
            if arrays_hash([r for r in rr if r['object_id']==0])!=peg_hash:raise ValueError('Peg changed')
            save_records(folder/(n+'_native.npz'),rr)
            for mode,counts in (('native',None),('cap128',128),('matched_N',{0:128,1:matched})):
                selected=budget(rr,counts);box_selected=[r for r in selected if r['object_id']==1]
                common=dict(seed=ep['seed'],split=ep['split'],timestep=t,variant=n,budget=mode)
                rows.append({**common,**metric(selected,gt,meshes,refs,historical,boundary,historical_boundary,aperture,t,gt_history,purity_cache.setdefault(n,{}))})
                if mode=='native':continue
                graph=system.graph(selected,data,cameras)
                # Historical visibility uses Main only; graph edge policy remains exactly PlanA.
                graph.packet.update(task='09-E',variant=n,current_only=n=='B0',persistent_geometry=n!='B0')
                save_graph(folder/(n+'_'+mode+'_graph.npz'),graph)
                arrays={k:getattr(graph,k).numpy() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
                er=graph_audit(arrays,cameras,gt,meshes,aperture,{0:np.empty((0,3)),1:refs})[1]
                edge_rows.append({**common,**er})
                if n=='B0' and mode=='cap128':
                    old=Path('/dev/shm/task09d_plan_a/v2/development_geometry/d8af1d51f8b54d0a927627bf05938e9c/geometry')/f'seed{ep["seed"]}'/stem/'C3_SF_cap_graph.npz'
                    # Original exact source filename discovered by preflight, never guessed silently.
                    reference=list(old.parent.glob('C3_SF*graph.npz'))
                    reference=[p for p in reference if 'matched' not in p.name and 'native' not in p.name]
                    if len(reference)!=1:raise ValueError('missing/ambiguous PlanA cap graph')
                    with np.load(reference[0],allow_pickle=False) as a:
                        for k in arrays:np.testing.assert_array_equal(arrays[k],a[k])
                peg_rows=graph.node_input[graph.node_type==3].numpy()
                audits.append(dict(**common,peg_hash=hashlib.sha256(peg_rows.tobytes()).hexdigest(),
                    robot76_hash=hashlib.sha256(data['robot76'].tobytes()).hexdigest(),baseline_exact=n=='B0' and mode=='cap128',
                    matched_box_N=matched,GT_in_input=False,temporal_extras_zero=not bool(graph.node_input[:,14:].any())))
        frames.append(dict(seed=ep['seed'],split=ep['split'],timestep=t,folder=str(folder.relative_to(root))))
        if t%25==0:print(f'geometry seed={ep["seed"]} t={t} B0={len(box)} B1={len(maps["B1"].active_records())} B2={len(maps["B2"].active_records())}',flush=True)
    for n,m in maps.items():
        write_json(output/(n+'_tracking.json'),m.trace);write_json(output/(n+'_lifecycle.json'),m.events)
    csv_write(output/'metrics.csv',rows);csv_write(output/'edges.csv',edge_rows);csv_write(output/'motion.csv',motion)
    csv_write(output/'filter.csv',filters);write_json(output/'paired_identity.json',audits)
    return dict(seed=ep['seed'],frames=frames,rows=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
    baseline=json.loads(Path(__file__).parents[1].joinpath('plan_a_geometry/protocol.json').read_text())
    manifest=json.loads((a.capture/'capture_manifest.json').read_text())
    if any(e['seed'] in cfg['final_seeds']+cfg['excluded_final_seeds'] for e in manifest['episodes']):raise ValueError('development/final separation')
    with ExperimentLogger(team='integration',config={**cfg,'capture':str(a.capture),'stage':'development_geometry','policy_training':False},seed=42,model='task09e_box_geometry',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True)
        inventory_sources(root);write_json(root/'protocol.json',cfg)
        episodes=[]
        with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[pool.submit(run_episode,(a.capture,root,e,cfg,baseline)) for e in manifest['episodes']]
            for future in as_completed(pending):
                episodes.append(future.result());write_json(root/'progress.json',episodes)
        write_json(root/'geometry_manifest.json',dict(config=cfg,capture=str(a.capture),capture_hash=sha256(a.capture/'capture_manifest.json'),episodes=episodes,frames=sum([e['frames'] for e in episodes],[])))
        log.link_output(root/'geometry_manifest.json',kind='metrics')
if __name__=='__main__':main()
