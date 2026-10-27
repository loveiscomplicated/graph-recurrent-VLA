"""Replay finalized same-observation captures into independent G/L comparisons.

Pilot evaluates fixed prefixes plus t=0 in every development episode; no final
test is selected here. --full evaluates all stored frames. Existing outputs
are never opened for writing. Branches are causal forks at t=0 in this pilot.
"""
import argparse,copy,csv,hashlib,json,os,resource,shutil,subprocess,time
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import write_json,save_npz,sha256,save_packet
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.integration_baseline.geometry09b import target as position_target
from graph.integration_baseline.graph import edge_attributes
from graph.persistent_geometry.evaluate import load_cameras,snapshot_records,save_graph
from graph.persistent_geometry.pipeline import GeometrySystem
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.metrics import (GeometryEvaluator,sample_meshes,filter_metrics,
    provenance_metrics,visible_reference,geometry_summary_numpy)
from graph.persistent_geometry.geometry_audit import graph_audit,aperture_reference
from graph.persistent_geometry.graph import build_graph,common_baseline,farthest_rows
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.audit import validate_split
from visual.runtime.vision_packet import pose_matrix
from visual.runtime.test.raw_ede_comparison_adapter import build_raw_ede
from .pipeline import LegacySystem
from .adapter import extract,original,frame_seed

from .runtime import BASE, CAPTURE
VARIANTS=[f'{a}{i}' for a in 'GL' for i in range(5)]

def arrays(graph):return {k:getattr(graph,k).numpy() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
def selected_records(graph,records,original=False):
    if original:return [records[i-3] for i in graph.packet['selected_original_rows'] if i>=3]
    lookup={r['persistent_node_id']:r for r in records}
    return [lookup[s] for s in graph.packet['node_ids']]
def no_edges(graph):
    g=copy.deepcopy(graph);g.edge_index=torch.empty((2,0),dtype=torch.int64)
    g.edge_attr=torch.empty((0,4),dtype=graph.edge_attr.dtype);g.edge_type=torch.empty(0,dtype=torch.int64)
    g.packet.update(edge_policy='legacy_no_edges_v1',edge_scope='no intra/object/robot messages')
    return g
def raw_graph(graph,records,components,base_points,camera):
    g=copy.deepcopy(graph);by_id={r['persistent_node_id']:r for r in records}
    selected=[by_id[s] for s in g.packet['node_ids']]
    sources=np.array([r['source_observation_ids'][0][3] for r in selected],np.int64)
    oid=np.array([r['object_id'] for r in selected],np.int64)
    lookup={(i,int(p)):int(c) for i,(ps,cs) in enumerate(zip(base_points,components)) for p,c in zip(ps.source_pixel,cs)}
    comp=np.array([lookup[(int(o),int(p))] for o,p in zip(oid,sources)],np.int64)
    if (comp<0).any():raise ValueError('BLOCKED_INPUT: ambiguous observed source cannot use Raw EDE')
    packet=dict(object_id=oid,provenance=dict(source_index=sources),observed_component_id=comp,image_shape=np.array(camera.depth.shape,np.int64))
    edge,metadata,_=build_raw_ede(packet)
    # Keep the same robot/object interactions; replace only intra-object edges.
    extra=g.edge_type!=0;g.edge_index=torch.cat((torch.from_numpy(edge+3),g.edge_index[:,extra]),1)
    g.edge_type=torch.cat((torch.zeros(edge.shape[1],dtype=torch.int64),g.edge_type[extra]))
    g.edge_attr=torch.from_numpy(edge_attributes(g.node_input[:,:3].numpy(),g.edge_index.numpy()))
    g.packet.update(metadata,single_current_camera=camera.camera,source_pixels=sources,components=comp)
    return g

def extra_geometry(records,gt,camera,mesh):
    result={}
    for oid in (0,1):
        selected=[r for r in records if r['object_id']==oid and not r.get('removed',False)]
        xyz=np.array([r['world_xyz'] for r in selected]).reshape(-1,3)
        local=transform(xyz,np.linalg.inv(pose_matrix(gt['actor_pose'][oid])))
        dist=mesh[oid]['tree'].query(local)[0]
        seg=gt[camera.camera+'/segmentation'];mask=seg==gt['actor_ids'][oid]
        boundary=mask & ~(cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8'))>0)
        boundary &= np.isfinite(camera.depth)
        support=camera.world_points()[boundary]
        tree=cKDTree(xyz) if len(xyz) else None
        result[oid]=dict(surface_distance_median_mm=float(np.median(dist)*1000) if len(dist) else None,
            boundary_visible_reference=len(support),boundary_coverage_4mm=float((tree.query(support)[0]<.004).mean()) if tree and len(support) else (0. if len(support) else None),
            occupied_2mm_voxels=len(np.unique(np.floor(local/.002).astype(int),axis=0)),
            geometry_outlier_fraction_8mm=float((dist>.008).mean()) if len(dist) else None,
            source_count=len(xyz),synthetic_or_interpolated_points=0,
            object_missing=bool(mask.any() and not len(xyz)))
        if len(xyz)>1:
            spacing=cKDTree(xyz).query(xyz,k=2)[0][:,1]*1000
            result[oid].update(neighbour_spacing_median_mm=float(np.median(spacing)),neighbour_spacing_p95_mm=float(np.percentile(spacing,95)))
    return result

def record_metrics(evaluator,name,records,system,gt,refs,t,history,cache,budget,camera,mesh):
    scope='G'+name[1] if name.startswith('L') else name
    memory=(system.main_map if name.endswith('2') else system.fused_map) if name.endswith(('2','4')) else None
    poses={oid:o.world_from_object for oid,o in memory.objects.items()} if memory else None
    measured=evaluator.measure(scope,records,poses,gt,refs,t,budget_mode=budget)
    purity=provenance_metrics(name,records,history,cache,t)
    extra=extra_geometry(records,gt,camera,mesh)
    for r in measured:
        r.update(variant=name,**purity[r['object_id']],**extra[r['object_id']])
        if memory:
            obj=memory.objects[r['object_id']]
            r.update(tracking_state=obj.tracking_state,registration_rank=obj.registration.get('rank'),
                registration_residual_m=obj.registration.get('residual_m'))
    return measured

def verify_input(path,expected):
    if sha256(path)!=expected:raise ValueError('frozen observation hash changed: '+str(path))

def run_episode(root,capture,episode,prefix,config):
    seed=episode['seed'];source=capture/'collection'/f'seed{seed}'
    contract=json.loads((source/'sensor_contract.json').read_text());main,wrist=contract['roles']['main'],contract['roles']['wrist']
    with np.load(source/'robot_meshes.npz',allow_pickle=False) as a:
        names=a['links'].tolist();vertices=[a[f'vertices_{i}'] for i in range(len(names))]
    gs=GeometrySystem(main,wrist);ls=LegacySystem(main,wrist,f5=True)
    mesh=sample_meshes(source/'evaluation_meshes.npz');evaluators={a:GeometryEvaluator(mesh) for a in 'GL'}
    history={};caches={a:{} for a in 'GL'};rows=[];edges=[];filters=[];frames=[];costs=[];source_ids=[];camera_audit=[]
    frames_in=episode['frames'] if config['full'] else episode['frames'][:prefix]
    for item in frames_in:
        t=item['timestep'];stem=item['stem'];folder=root/'geometry'/f'seed{seed}'/stem;folder.mkdir(parents=True)
        for suffix,key in (('_input.npz','input_sha256'),('_masks.npz','mask_sha256'),('_gt.npz','gt_sha256')):verify_input(source/(stem+suffix),item[key])
        cameras,data=load_cameras(source/(stem+'_input.npz'));by={c.camera:c for c in cameras}
        with np.load(source/(stem+'_masks.npz'),allow_pickle=False) as a:masks={k:a[k] for k in a.files}
        with np.load(source/(stem+'_gt.npz'),allow_pickle=False) as a:gt={k:a[k] for k in a.files}
        history[t]=gt
        # Pose/mask sidecars stay out of process() calls; only evaluated below.
        links=[LinkSurface(n,v,p) for n,v,p in zip(names,vertices,data['robot_link_world'])]
        tick=time.perf_counter();g=gs.process(cameras,data,masks,links);gtime=time.perf_counter()-tick
        tick=time.perf_counter();l=ls.process(cameras,data,masks,links);ltime=time.perf_counter()-tick
        for c in cameras:
            for key in ('confirmed_robot','uncertain'):
                np.testing.assert_array_equal(g['filter_outputs'][c.camera][key],l['filter_outputs'][c.camera][key])
            filters.extend(dict(seed=seed,timestep=t,split=episode['split'],**r) for r in filter_metrics(c,masks[c.camera],l['filter_outputs'][c.camera],gt,finger_world=data['robot_world_geometry'][1:]))
            detail=contract['installed'][c.camera];mount=detail.get('mount_link_name')
            camera_audit.append(dict(seed=seed,timestep=t,camera=c.camera,role='main' if c.camera==main else 'wrist',timestamp=c.timestamp,
                world_from_camera=c.world_from_camera,cv_gl_agreement=True,mount_link_name=mount,
                link_from_camera=np.linalg.inv(data['robot_link_world'][names.index(mount)])@c.world_from_camera if mount in names else None))
        save_npz(folder/'robot_filter.npz',**{cn+'/'+k:v for cn,out in l['filter_outputs'].items() for k,v in out.items() if isinstance(v,np.ndarray)})
        save_packet(folder/'G0_original_packet.npz',g['baseline_packet'])
        save_npz(folder/'G0_original_graph.npz',**arrays(g['baseline_graph']))
        # Exactness checked at actual current inputs, same RNG state per object.
        point_times=[]
        for oid,p in enumerate(l['base_points']):
            s=frame_seed(by[main].episode,t,main,oid)
            tick=time.perf_counter();ref=original.extract(masks[main][oid],by[main].depth.astype(np.float32),by[main].intrinsic,by[main].world_from_camera,np.random.default_rng(s));point_times.append((time.perf_counter()-tick)*1000)
            np.testing.assert_array_equal(p.camera_xyz,ref[0]);np.testing.assert_array_equal(p.world_xyz,ref[1])
        for camera,points in [(main+'_unfiltered',l['base_points']),*l['filtered_points'].items()]:
            for oid,p in enumerate(points):
                save_npz(folder/f'fps_{camera}_object{oid}.npz',point_xyz_camera=p.camera_xyz,point_xyz_world=p.world_xyz,
                    source_pixel_id=p.source_pixel,**p.stages)
                source_ids.append(dict(seed=seed,timestep=t,camera=camera,object_id=oid,**p.audit))
        inverse=np.linalg.inv(pose_matrix(data['tcp_pose']));target=np.concatenate([position_target(gt['actor_pose'][oid],data['tcp_pose']) for oid in (0,1)])
        visible=visible_reference(cameras,gt)
        target36=np.r_[target,np.concatenate([geometry_summary_numpy(transform(visible[oid],inverse)) for oid in (0,1)])]
        save_npz(folder/'evaluation_targets.npz',target36=target36,target_valid=np.r_[np.ones(6,bool),np.repeat([bool(len(visible[o])) for o in (0,1)],15)],robot76=data['robot76'],timestep=np.array(t),episode=np.array(episode['episode']))
        refs={a:evaluators[a].references(cameras,gt,main) for a in 'GL'}
        aperture=aperture_reference(mesh,gt)
        for i in range(5):
            counts={oid:min(sum(r['object_id']==oid and not r.get('removed',False) for r in product['records'][a+str(i)]) for a,product in (('G',g),('L',l))) for oid in (0,1)}
            for a,product,system in (('G',g,gs),('L',l,ls)):
                name=a+str(i);records=product['records'][name];is_g0=name=='G0'
                tick=time.perf_counter()
                graph=common_baseline(g['baseline_graph'],point_cap=128) if is_g0 else build_graph(records,data['robot_world_geometry'],data['tcp_pose'],t,cameras=[by[main]] if i<3 else cameras,point_cap=128)
                # L0 is strictly no-edge. Keep same-point surface-edge control separate.
                surface=copy.deepcopy(graph)
                if name=='L0':graph=no_edges(graph)
                selected=selected_records(graph,records,is_g0)
                save_graph(folder/(name+'_graph.npz'),graph)
                snapshot_records(folder/(name+'_map.npz'),records)
                write_json(folder/(name+'_provenance.json'),[{k:r[k] for k in ('persistent_node_id','object_id','source_observation_ids')} for r in records])
                for budget,rr in (('native_variable_N',records),('cap128_per_object',selected)):
                    measured=record_metrics(evaluators[a],name,rr,system,gt,refs[a],t,history,caches[a],budget,by[main],mesh)
                    rows.extend(dict(seed=seed,split=episode['split'],**r) for r in measured)
                # Matched-N is a declared post-selection diagnostic: neither native rule is changed.
                matched=[]
                for oid in (0,1):
                    eligible=[r for r in records if r['object_id']==oid and not r.get('removed',False)]
                    idx=farthest_rows(np.array([r['world_xyz'] for r in eligible]).reshape(-1,3),[r['persistent_node_id'] for r in eligible],counts[oid])
                    matched.extend(eligible[int(j)] for j in idx)
                snapshot_records(folder/(name+'_matched_map.npz'),matched)
                matched_metrics=record_metrics(evaluators[a],name,matched,system,gt,refs[a],t,history,caches[a],'matched_native_minN_posthoc',by[main],mesh)
                rows.extend(dict(seed=seed,split=episode['split'],matched_target_count=counts[r['object_id']],**r) for r in matched_metrics)
                edge_metric=graph_audit(arrays(graph),cameras,gt,mesh,aperture,visible)
                edges.extend(dict(seed=seed,timestep=t,split=episode['split'],variant=name,budget='cap128',**r) for r in edge_metric)
                if name=='L0':
                    save_graph(folder/'L0_surface_graph.npz',surface)
                    try:
                        raw=raw_graph(surface,records,l['base_components'],l['base_points'],by[main]);save_graph(folder/'L0_rawede_graph.npz',raw)
                        controls=[('L0_surface',surface),('L0_rawede',raw)]
                    except ValueError as e:
                        write_json(folder/'rawede_blocked.json',dict(status='BLOCKED_INPUT',reason=str(e)));controls=[('L0_surface',surface)]
                    for label,gg in controls:
                        edges.extend(dict(seed=seed,timestep=t,split=episode['split'],variant=label,budget='cap128_same_points',**r) for r in graph_audit(arrays(gg),cameras,gt,mesh,aperture,visible))
                costs.append(dict(seed=seed,timestep=t,variant=name,graph_construction_and_metric_ms=(time.perf_counter()-tick)*1000))
        for name,memory in (('G2',gs.main_map),('G4',gs.fused_map),('L2',ls.main_map),('L4',ls.fused_map)):
            write_json(folder/(name+'_tracker.json'),dict(objects={oid:dict(pose=obj.world_from_object,tracking_state=obj.tracking_state,registration=obj.registration,confidence=obj.pose_confidence,uncertainty=obj.pose_uncertainty) for oid,obj in memory.objects.items()}))
        write_json(folder/'fusion_audit.json',dict(G=g['fusion_audit'],L=l['fusion_audit']))
        frames.append(dict(seed=seed,timestep=t,split=episode['split'],phase=item['phase'],condition='baseline',folder=str(folder.relative_to(root)),input_sha256=item['input_sha256'],mask_sha256=item['mask_sha256']))
        costs.append(dict(seed=seed,timestep=t,variant='pipeline',G_seconds=gtime,L_seconds=ltime,legacy_native_extract_ms=point_times,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        write_json(root/'progress.json',dict(seed=seed,timestep=t,frames_done=len(frames),elapsed_at=time.time()))
        print(f'legacy compare seed={seed} t={t} G={gtime:.3f}s L={ltime:.3f}s',flush=True)
        if shutil.disk_usage(BASE).free<3*1024**3:raise RuntimeError('BLOCKED_RESOURCE: protect original pipeline disk reserve3GiB')
    out=root/'geometry'/f'seed{seed}'
    for name,memory in (('G2',gs.main_map),('G4',gs.fused_map),('L2',ls.main_map),('L4',ls.fused_map)):
        write_json(out/(name+'_lifecycle.json'),memory.events)
    csv_write(out/'geometry_metrics.csv',rows);csv_write(out/'edge_metrics.csv',edges);csv_write(out/'robot_filter_metrics.csv',filters)
    write_json(out/'source_pixel_audit.json',source_ids);write_json(out/'camera_audit.json',camera_audit);write_json(out/'costs.json',costs)
    return frames,rows,edges,filters

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,default=CAPTURE);parser.add_argument('--output',type=Path,default=BASE/'experiments');parser.add_argument('--full',action='store_true');parser.add_argument('--prefix',type=int,default=8);parser.add_argument('--seeds',type=int,nargs='*');args=parser.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((args.capture/'capture_manifest.json').read_text())
    if json.loads((args.capture/'metadata.json').read_text())['status']!='finished':raise ValueError('finalized inputs only')
    validate_split(manifest['config']['split'],{e['seed']:e['actual_reset_group'] for e in manifest['episodes']})
    config=dict(task='09-C-Legacy-FPS',full=args.full,prefix=args.prefix,variants=VARIANTS,seed=42,
        capture=str(args.capture),split=manifest['config']['split'],point_cap=128,
        temporal_pilot_seeds=[30002,30125],static_all_development_t0=True,
        pilot_scope='fixed prefixes plus all episode t0; not independent final test',
        fps_feature_contract='Native raw XYZ; all common-schema graph/map comparisons explicitly FPS+original PCA5/F5 augmentation',
        matched_N='per-object native min(Gi,Li), posthoc identity-seeded farthest subset; not native algorithm effect',
        depth='same current signed-int16-mm source converted float32 metres; no resize',
        no_policy_no_gru=True,no_publish=True,resource_policy='single CPU thread nice19; disk reserve3GiB')
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_legacy_fps_extension',job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('LEGACY_RUN='+str(root),flush=True)
        (BASE/'latest_run.txt').write_text(str(root)+'\n')
        for name in ('protection_before.json','runtime_before.json','dependency_source_hashes.json'):shutil.copy2(BASE/name,root/name)
        shutil.copy2(BASE/'contract_tests.log',root/'contract_tests.log')
        write_json(root/'capture_identity.json',dict(capture_run_id=args.capture.name,capture_manifest_sha256=sha256(args.capture/'capture_manifest.json'),
            capture_metadata_sha256=sha256(args.capture/'metadata.json'),episodes=[{k:e[k] for k in ('seed','split','actual_reset_group','original','original_reset_match')} for e in manifest['episodes']],
            screenshot_identity='NOT_PROVEN: no original screenshot RGB-D/runtime identity supplied; current capture original_reset_match false'))
        frames=[];rows=[];edges=[];filters=[]
        for episode in manifest['episodes']:
            if args.seeds and episode['seed'] not in args.seeds:continue
            n=args.prefix if episode['seed'] in config['temporal_pilot_seeds'] else 1
            f,r,e,ff=run_episode(root,args.capture,episode,n,config);frames+=f;rows+=r;edges+=e;filters+=ff
            logger.log(dict(completed_episodes=len(set(x['seed'] for x in frames)),frames=len(frames)))
            write_json(root/'geometry_manifest.json',dict(config=config,frames=frames))
        csv_write(root/'geometry_metrics.csv',rows);csv_write(root/'edge_metrics.csv',edges);csv_write(root/'robot_filter_metrics.csv',filters)
        write_json(root/'geometry_metrics.json',rows)
        write_json(root/'stage_status.json',dict(Stage0='PASS',Stage1='PASS',Stage2='PARTIAL',Stage3='PARTIAL',Stage4='PARTIAL',Stage5='NOT_EXECUTED',Stage6='PARTIAL'))
        for name in ('geometry_metrics.csv','geometry_metrics.json','edge_metrics.csv','robot_filter_metrics.csv','geometry_manifest.json'):logger.link_output(root/name,kind='metrics')
if __name__=='__main__':main()
