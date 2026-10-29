"""Same-input geometry evaluation before any diagnostic encoder training."""
import argparse,json,time
from pathlib import Path
from datetime import datetime,timezone
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from graph.integration_baseline.io import write_json,save_npz,sha256,save_packet
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.integration_baseline.geometry09b import target as position_target
from graph.persistent_geometry.evaluate import load_cameras,save_graph,snapshot_records
from graph.persistent_geometry.metrics import sample_meshes,visible_reference,geometry_summary_numpy,filter_metrics,provenance_metrics
from graph.persistent_geometry.geometry_audit import aperture_reference,graph_audit
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.collect import inventory_sources
from visual.runtime.vision_packet import pose_matrix
from visual.scripts.pointcloud import backproject,to_world,fps
from graph.persistent_geometry.fusion import observations_from_packet,make_packet
from graph.persistent_geometry.map import current_records
from utils.experiment_logger import ExperimentLogger
from .core import PlanA,budget,deduplicate

VARIANTS=('C0_SF','C1_SF','C2_SF','C3_SF','C3_L','C3_H')

def save_records(path,records):
    snapshot_records(path,records)
    sources=[(i,s) for i,r in enumerate(records) for s in r['source_observation_ids']]
    save_npz(path.with_name(path.stem+'_sources.npz'),node_row=np.array([i for i,s in sources],int),
        episode=np.array([s[0] for i,s in sources],str),timestep=np.array([s[1] for i,s in sources],int),
        camera=np.array([s[2] for i,s in sources],str),pixel=np.array([s[3] for i,s in sources],int),
        structural=np.array([r.get('structural',False) for r in records],bool))


def measure(records,variant,mode,cameras,gt,meshes,refs,t):
    purity=provenance_metrics(variant,records,{t:gt},{},t);rows=[]
    for oid in (0,1):
        rr=[r for r in records if r['object_id']==oid];xyz=np.array([r['world_xyz'] for r in rr]).reshape(-1,3)
        distance=meshes[oid]['tree'].query(transform(xyz,np.linalg.inv(pose_matrix(gt['actor_pose'][oid]))))[0]
        tree=cKDTree(xyz) if len(xyz) else None
        def coverage(reference):return float((tree.query(reference)[0]<.004).mean()) if tree is not None and len(reference) else 0. if len(reference) else None
        boundary=[]
        for c in cameras:
            correct=(gt[c.camera+'/segmentation']==gt['actor_ids'][oid])&np.isfinite(c.depth)
            edge=correct & ~(cv2.erode(correct.astype('uint8'),np.ones((3,3),'uint8'))>0)
            boundary.append(c.world_points()[edge])
        boundary=np.concatenate(boundary)
        normals=np.array([r['normal_valid'] for r in rr],bool)
        src=[s for r in rr for s in r['source_observation_ids']]
        row=dict(variant=variant,budget=mode,object_id=oid,timestep=t,count=len(rr),
            surface_distance_mean_mm=float(distance.mean()*1000) if len(distance) else None,
            surface_distance_median_mm=float(np.median(distance)*1000) if len(distance) else None,
            surface_distance_p95_mm=float(np.percentile(distance,95)*1000) if len(distance) else None,
            surface_precision_4mm=float((distance<.004).mean()) if len(distance) else None,
            geometry_outlier_fraction_8mm=float((distance>.008).mean()) if len(distance) else None,
            current_both_visible_coverage_4mm=coverage(refs[oid]),
            boundary_coverage_4mm=coverage(boundary),boundary_reference_points=len(boundary),
            gt_visible_points=len(refs[oid]),missing_when_visible=bool(len(refs[oid]) and not len(rr)),
            normal_valid_fraction=float(normals.mean()) if len(rr) else None,
            camera_main_sources=sum(s[2]=='base_camera' for s in src),camera_wrist_sources=sum(s[2]=='hand_camera' for s in src),
            source_count=len(src),unique_sources=len(set(src)),
            **purity[oid])
        if variant.startswith(('C2_','C3_')) and oid==1 and row['camera_wrist_sources']:raise ValueError('Wrist Box leakage')
        if any(s[1]!=t for s in src):raise ValueError('noncurrent geometry')
        rows.append(row)
    return rows


def run_episode(args):
    capture,output,episode,protocol=args
    torch.set_num_threads(1)
    source=capture/'collection'/f'seed{episode["seed"]}'
    sensor_contract=json.loads((source/'sensor_contract.json').read_text());roles=sensor_contract['roles'];system=PlanA(roles['main'],roles['wrist'],protocol)
    with np.load(source/'robot_meshes.npz',allow_pickle=False) as a:
        names=a['links'].tolist();vertices=[a[f'vertices_{i}'].copy() for i in range(len(names))]
    meshes=sample_meshes(source/'evaluation_meshes.npz')
    metrics=[];filter_rows=[];gates=[];graphs=[];frames=[];ablation=[];mount_reference=None
    tasks=episode['frames']+[{**b,'is_branch':True,'stem':f"f{b['timestep']:04d}_{b['condition']}",'mask_sha256':b['masks_sha256']} for b in episode.get('branches',[])]
    for frame in tasks:
        t=frame['timestep'];stem=frame['stem'];start=time.perf_counter()
        branch=frame.get('is_branch',False)
        paths={suffix:source/frame['source']/suffix.removeprefix('_') if branch else source/(stem+suffix) for suffix in ('_input.npz','_masks.npz','_gt.npz')}
        for suffix,key in (('_input.npz','input_sha256'),('_masks.npz','mask_sha256'),('_gt.npz','gt_sha256')):
            if sha256(paths[suffix])!=frame[key]:raise ValueError('capture identity changed')
        cameras,data=load_cameras(paths['_input.npz'])
        with np.load(paths['_masks.npz'],allow_pickle=False) as a:masks={k:a[k].copy() for k in a.files}
        surfaces=[LinkSurface(n,v,p) for n,v,p in zip(names,vertices,data['robot_link_world'])]
        prepared=system.prepare(cameras,data,masks,surfaces)
        mount_name=sensor_contract['installed'][roles['wrist']]['mount_link_name']
        wrist_camera=next(c for c in cameras if c.camera==roles['wrist'])
        mounted=np.linalg.inv(data['robot_link_world'][names.index(mount_name)])@wrist_camera.world_from_camera
        if mount_reference is None:mount_reference=mounted.copy()
        np.testing.assert_allclose(mounted,mount_reference,atol=2e-5,rtol=0)
        if branch:
            _,parent=load_cameras(source/f'f{t:04d}_input.npz')
            np.testing.assert_array_equal(data['robot76'],parent['robot76'])
        # Ground truth is opened only after the sensor-only gate has produced its decision.
        with np.load(paths['_gt.npz'],allow_pickle=False) as a:gt={k:a[k].copy() for k in a.files}
        common=dict(seed=episode['seed'],split=episode['split'],timestep=t,condition=frame.get('condition','baseline'),physical_valid=frame.get('physical_valid',True))
        gate={**common,**prepared['gate']};wrist=next(c for c in cameras if c.camera==roles['wrist'])
        correct=gt[wrist.camera+'/segmentation']==gt['actor_ids'][0]
        before=prepared['clean'][wrist.camera][0]&np.isfinite(wrist.depth)
        after=prepared['conditions']['C3'][wrist.camera][0]
        gate.update(correct_wrist_before=int((before&correct).sum()),correct_wrist_after=int((after&correct).sum()),
            wrong_wrist_before=int((before&~correct).sum()),wrong_wrist_after=int((after&~correct).sum()),
            main_mask_points=int(prepared['clean'][roles['main']][0].sum()),
            wrist_only_recovery=bool(not prepared['clean'][roles['main']][0].any() and after.any()),
            gt_wrist_visible=int(correct.sum()),
            main_gt_visible=int((gt[roles['main']+'/segmentation']==gt['actor_ids'][0]).sum()))
        gates.append(gate)
        for c in cameras:
            filter_rows.extend({**common,**r} for r in filter_metrics(c,masks[c.camera],prepared['filters'][c.camera],gt,finger_world=data['robot_world_geometry'][1:]))
        if not branch and t%protocol['geometry_interval']!=0 and frame!=episode['frames'][-1]:continue
        folder=output/'geometry'/f'seed{episode["seed"]}'/stem;folder.mkdir(parents=True,exist_ok=False)
        save_npz(folder/'robot_filter.npz',**{name+'/'+key:value for name,f in prepared['filters'].items() for key,value in f.items() if isinstance(value,np.ndarray)})
        save_npz(folder/'accepted_masks.npz',**{cond+'/'+name:m for cond,cm in prepared['conditions'].items() for name,m in cm.items()})
        write_json(folder/'gate.json',gate)
        refs=visible_reference(cameras,gt);inverse=np.linalg.inv(pose_matrix(data['tcp_pose']))
        target=np.concatenate([position_target(gt['actor_pose'][oid],data['tcp_pose']) for oid in (0,1)]+[geometry_summary_numpy(transform(refs[oid],inverse)) for oid in (0,1)])
        save_npz(folder/'evaluation_targets.npz',target36=target,target_valid=np.r_[np.ones(6,bool),np.repeat([bool(len(refs[o])) for o in (0,1)],15)],
            robot76=data['robot76'],episode=data['episode'],timestep=np.array(t))
        native={};dedup={}
        for cond in ('C0','C1','C2','C3'):
            pools,audit,stages,packets=system.represent(cameras,prepared,cond,('SF','L','H') if cond=='C3' else ('SF',))
            for rep,records in pools.items():native[cond+'_'+rep]=records;dedup[cond+'_'+rep]=audit[rep]
            if cond=='C3':
                for name,points in stages.items():
                    for oid,p in enumerate(points):
                        save_npz(folder/f'legacy_{name}_object{oid}_stages.npz',**p.stages,source_pixel=p.source_pixel,camera_xyz=p.camera_xyz,world_xyz=p.world_xyz)
                for name,packet in packets.items():save_packet(folder/f'SF_{name}_original_packet.npz',packet)
        write_json(folder/'deduplication.json',dedup)
        matched={oid:min(protocol['point_cap'],*[sum(r['object_id']==oid for r in native[v]) for v in ('C3_SF','C3_L','C3_H')]) for oid in (0,1)}
        write_json(folder/'matched_counts.json',dict(counts=matched,GT_used=False,variants=['C3_SF','C3_L','C3_H']))
        aperture=aperture_reference(meshes,gt)
        for variant,records in native.items():
            save_records(folder/(variant+'_native.npz'),records)
            modes=[('native',records),('cap128',budget(records,protocol['point_cap'],variant.endswith('_H')))]
            if variant.startswith('C3_'):modes.append(('matched_N',budget(records,matched,variant.endswith('_H'))))
            for mode,rr in modes:
                metrics.extend({**common,**r} for r in measure(rr,variant,mode,cameras,gt,meshes,refs,t))
                if mode=='native':continue
                graph=system.graph(rr,data,cameras)
                save_graph(folder/(variant+('_graph.npz' if mode=='cap128' else '_matched_N_graph.npz')),graph)
                save_records(folder/(variant+'_'+mode+'_points.npz'),rr)
                arrays={k:getattr(graph,k).numpy() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
                graphs.extend({**common,'variant':variant,'budget':mode,**r} for r in graph_audit(arrays,cameras,gt,meshes,aperture,refs))
        # Distinct preprocessing-vs-selection comparison, excluded from GNN ranking.
        for c in cameras:
            cm=prepared['conditions']['C3'][c.camera]
            for oid in (0,1):
                raw=np.flatnonzero(cm[oid]&np.isfinite(c.depth));point=stages[c.camera][oid]
                clean_support=point.stages['sor_source']
                for label,pixels in (('common_clean_support',raw),('erosion_SOR_support',clean_support)):
                    xyz=c.world_points().reshape(-1,3)[pixels]
                    dist=meshes[oid]['tree'].query(transform(xyz,np.linalg.inv(pose_matrix(gt['actor_pose'][oid]))))[0]
                    ablation.append({**common,'camera':c.camera,'object_id':oid,'comparison':label,'count':len(pixels),'surface_distance_mean_mm':float(dist.mean()*1000) if len(dist) else None})
                support_mask=np.zeros_like(cm);support_mask[oid].ravel()[clean_support]=True
                common_packet,_=make_packet(c,support_mask)
                sf=observations_from_packet(common_packet,c)[oid]
                candidates=to_world(backproject(c.depth.astype('float32'),c.intrinsic,support_mask[oid]),c.world_from_camera)
                n=min(128,len(sf.xyz),len(candidates))
                from graph.persistent_geometry.graph import farthest_rows
                choices={'SF_same_processed_support':sf.xyz[farthest_rows(sf.xyz,sf.pixels,n)],
                    'FPS_same_processed_support':candidates[farthest_rows(candidates,clean_support,n)]}
                for label,xyz in choices.items():
                    dist=meshes[oid]['tree'].query(transform(xyz,np.linalg.inv(pose_matrix(gt['actor_pose'][oid]))))[0]
                    ablation.append({**common,'camera':c.camera,'object_id':oid,'comparison':label,'count':n,'surface_distance_mean_mm':float(dist.mean()*1000) if n else None})
        frames.append(dict(**common,folder=str(folder.relative_to(output)),source_folder=str(source),input_sha256=frame['input_sha256'],masks_sha256=frame['mask_sha256'],gt_sha256=frame['gt_sha256'],input_path=str(paths['_input.npz']),masks_path=str(paths['_masks.npz']),gt_path=str(paths['_gt.npz']),elapsed_seconds=time.perf_counter()-start))
        if t%25==0:print(f'geometry seed={episode["seed"]} t={t}',flush=True)
    return dict(metrics=metrics,filters=filter_rows,gates=gates,graphs=graphs,frames=frames,ablation=ablation)


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--max-frames',type=int);a=p.parse_args()
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());manifest=json.loads((a.capture/'capture_manifest.json').read_text())
    config=dict(task='09-D',capture=str(a.capture),protocol=protocol,GT_scope='evaluation-only after gate',variants=VARIANTS,geometry_interval=protocol['geometry_interval'])
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09d_plan_a_geometry',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True)
        episodes=manifest['episodes']
        if a.max_frames:episodes=[{**episodes[0],'frames':episodes[0]['frames'][:a.max_frames],'branches':[]}]
        allrows={k:[] for k in ('metrics','filters','gates','graphs','frames','ablation')}
        if a.workers>1:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                results=pool.map(run_episode,[(a.capture,root,e,protocol) for e in episodes])
                for result in results:
                    for k,v in result.items():allrows[k].extend(v)
        else:
            for e in episodes:
                result=run_episode((a.capture,root,e,protocol))
                for k,v in result.items():allrows[k].extend(v)
        for key in ('metrics','filters','graphs','ablation'):csv_write(root/(key+'.csv'),allrows[key]);write_json(root/(key+'.json'),allrows[key])
        write_json(root/'gate_trace.json',allrows['gates'])
        write_json(root/'geometry_manifest.json',dict(config=config,frames=allrows['frames'],episodes=[dict(seed=e['seed'],split=e['split'],actual_reset_group=e['actual_reset_group']) for e in episodes],completed=True))
        inventory_sources(root)
        for path in root.glob('*.json'):log.link_output(path,kind='metrics')
if __name__=='__main__':main()
