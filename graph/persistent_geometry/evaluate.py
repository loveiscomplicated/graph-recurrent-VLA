"""G0--G4 and separate O1--O3 causal geometry evaluation on saved diagnostics."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing

import numpy as np
import torch

from graph.integration_baseline.graph import build_graph as original_graph
from graph.integration_baseline.io import write_json, save_npz, save_packet, sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.integration_baseline.geometry09b import target as position_target
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger

from .contracts import CameraObservation, synchronized, transform
from .self_filter import LinkSurface, classify_robot
from .fusion import make_packet, observations_from_packet, filtered_masks, consistent_views
from .map import PersistentMap, current_records
from .oracle import OracleMotionMap
from .graph import build_graph, common_baseline
from .metrics import GeometryEvaluator, sample_meshes, filter_metrics, visible_reference, geometry_summary_numpy,provenance_metrics
from .audit import protection_snapshot, verify_protection, validate_split
from .pipeline import GeometrySystem

VARIANTS = ('G0','G1','G2','G3','G4','O1','O2','O3')


def load_cameras(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {k:source[k] for k in source.files}
    names = [k[:-4] for k in arrays if k.endswith('/rgb')]
    cameras = [CameraObservation(str(arrays['episode']), int(arrays['timestep']), name,
        float(arrays['timestamp']), **{k:arrays[name+'/'+k] for k in
        ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl')}, tcp_pose=arrays['tcp_pose']) for name in names]
    synchronized(cameras)
    return cameras, arrays


def snapshot_records(path, records):
    arrays = {}
    specs = dict(object_local_xyz=(float,3), object_local_normal=(float,3),
                 world_xyz=(float,3), world_normal=(float,3),
                 object_id=(int,None), normal_valid=(bool,None),
                 confidence=(float,None), first_seen_timestep=(int,None),
                 last_seen_timestep=(int,None), observation_count=(int,None),
                 currently_observed=(bool,None), estimated_only=(bool,None),
                 position_uncertainty=(float,None), pose_confidence=(float,None),
                 status=(int,None), last_f5=(float,5), last_feature_valid=(bool,3))
    specs['contradiction_count']=(int,None)
    for name, (dtype, dimension) in specs.items():
        a = np.asarray([r.get(name,0) for r in records], dtype=dtype)
        arrays[name] = a.reshape(-1, dimension) if dimension else a
    arrays['persistent_node_id'] = np.array([r['persistent_node_id'] for r in records], dtype=str)
    arrays['last_observed_camera'] = np.array([r['last_observed_camera'] for r in records], dtype=str)
    save_npz(path, **arrays)


def save_graph(path, graph):
    save_npz(path, **{name:getattr(graph,name).detach().cpu().numpy() for name in
        ('node_input','node_type','edge_index','edge_attr','edge_type')})
    write_json(path.with_suffix('.json'), graph.packet)


def run_episode(root, source_root, episode, *, point_cap=128):
    torch.set_num_threads(1)
    seed = episode['seed']
    folder = source_root/'collection'/f'seed{seed}'
    output = root/'geometry'/f'seed{seed}'; output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((folder/'sensor_contract.json').read_text())
    roles = contract['roles']; main, wrist = roles['main'], roles['wrist']
    with np.load(folder/'robot_meshes.npz', allow_pickle=False) as a:
        names = a['links'].tolist()
        vertices = [a[f'vertices_{i}'] for i in range(len(names))]
    maps = {name:PersistentMap() for name in ('G2','G4','O1')}
    maps.update({name:OracleMotionMap() for name in ('O2','O3')})
    practical=GeometrySystem(main,wrist,maps['G2'],maps['G4'])
    evaluator = GeometryEvaluator(sample_meshes(folder/'evaluation_meshes.npz'))
    rows, filters, frames, costs = [], [], [], []
    gt_history = {}
    source_label_cache={}
    source_provenance = {name:{} for name in VARIANTS}
    camera_relative = {}
    camera_motion=[]
    for frame in episode['frames']:
        t, stem = frame['timestep'], frame['stem']
        start = time.perf_counter()
        for suffix, digest in (('_input.npz', frame['input_sha256']), ('_masks.npz',frame['mask_sha256']), ('_gt.npz',frame['gt_sha256'])):
            if sha256(folder/(stem+suffix)) != digest:
                raise ValueError('capture identity changed')
        cameras, data = load_cameras(folder/(stem+'_input.npz'))
        with np.load(folder/(stem+'_masks.npz'), allow_pickle=False) as a:
            masks = {k:a[k] for k in a.files}
        with np.load(folder/(stem+'_gt.npz'), allow_pickle=False) as a:
            gt = {k:a[k] for k in a.files}
        gt_history[t] = gt
        references = evaluator.references(cameras, gt, main)
        same_time = {c.camera:c for c in cameras}
        # A moving wrist calibration is anchored to the installed robot mount
        # transform, rather than inferred from a static camera assumption.
        for camera in cameras:
            if camera.camera == wrist:
                detail=contract['installed'][wrist]
                link_name=detail.get('mount_link_name')
                if link_name is None:
                    # Explicit adapter for the pilot's actual serialized Link
                    # representation; no camera/link name is guessed.
                    candidates=[n for n in set(names) if detail['mount'].startswith('<'+n+': ') or detail['entity_uid']==n]
                    if len(candidates)!=1:
                        raise ValueError('BLOCKED_SENSOR_MOUNT_CONTRACT')
                    link_name=candidates[0]
                if link_name not in names:
                    raise ValueError('mounted robot link geometry not recorded')
                link_world = data['robot_link_world'][names.index(link_name)]
                relative = np.linalg.inv(link_world) @ camera.world_from_camera
                if wrist not in camera_relative:
                    camera_relative[wrist] = relative
                np.testing.assert_allclose(relative, camera_relative[wrist], atol=2e-5, rtol=0)
                camera_motion.append(dict(timestep=t,mount_link=link_name,world_from_camera=camera.world_from_camera,
                    link_from_camera=relative))
        surfaces = [LinkSurface(name, v, pose) for name,v,pose in zip(names,vertices,data['robot_link_world'])]
        produced=practical.process(cameras,data,masks,surfaces)
        filter_outputs=produced['filter_outputs']
        g0_packet,g0_graph=produced['baseline_packet'],produced['baseline_graph']
        oracle={}
        for camera in cameras:
            filtered = filter_outputs[camera.camera]
            filters.extend({**r, 'seed':seed, 'split':episode['split'], 'timestep':t}
                           for r in filter_metrics(camera,masks[camera.camera],filtered,gt,
                                                   finger_world=data['robot_world_geometry'][1:]))
            # Deliberately separate oracle construction, after estimated inputs.
            gt_masks = np.stack([gt[camera.camera+'/segmentation']==i for i in gt['actor_ids']])
            clean_gt, _ = filtered_masks(gt_masks,filtered)
            oracle[camera.camera], _ = make_packet(camera,clean_gt)
        visible_oracle = {name:observations_from_packet(packet,same_time[name]) for name,packet in oracle.items()}
        dual,fusion_audit=produced['dual_observations'],produced['fusion_audit']
        # Oracle masks agree on actor identity by construction; both still use
        # actual measured depths and only actually visible source pixels.
        dual_oracle = visible_oracle[main]+visible_oracle[wrist]
        records=produced['records']
        maps['O1'].update(episode['episode'],t,dual_oracle,cameras=cameras)
        records['O1']=maps['O1'].records()
        for name, observations in (('O2',dual),('O3',dual_oracle)):
            maps[name].update_with_gt_motion(episode['episode'],t,observations,gt['actor_pose'],cameras=cameras)
            records[name] = maps[name].records()
        frame_out = output/stem; frame_out.mkdir()
        save_packet(frame_out/'G0_original_packet.npz',g0_packet)
        for camera,packet in produced['filtered_packets'].items():
            save_packet(frame_out/('observed_'+camera+'_packet.npz'),packet)
        for camera,packet in oracle.items():
            save_packet(frame_out/('oracle_visible_'+camera+'_packet.npz'),packet)
        # Preserve full G0 tensors separately from common-schema budget views.
        save_npz(frame_out/'G0_original_graph.npz',**{name:getattr(g0_graph,name).numpy() for name in
            ('node_input','node_type','edge_index','edge_attr','edge_type')})
        save_npz(frame_out/'robot_filter.npz',**{camera+'/'+key:value for camera,output_filter in filter_outputs.items()
            for key,value in output_filter.items() if isinstance(value,np.ndarray)})
        frame_metrics = []
        inverse_ee = np.linalg.inv(pose_matrix(data['tcp_pose']))
        target = np.concatenate([position_target(gt['actor_pose'][oid],data['tcp_pose']) for oid in (0,1)])
        visible = visible_reference(cameras,gt)
        summary = np.concatenate([geometry_summary_numpy(transform(visible[oid],inverse_ee)) for oid in (0,1)])
        target_valid = np.r_[np.ones(6,bool), np.repeat([bool(len(visible[oid])) for oid in (0,1)],15)]
        save_npz(frame_out/'evaluation_targets.npz',target36=np.r_[target,summary],target_valid=target_valid,
                 robot76=data['robot76'],timestep=np.array(t),episode=np.array(episode['episode']))
        matched_counts={oid:min(point_cap,*[sum(r['object_id']==oid and not r.get('removed',False)
            for r in records[name]) for name in ('G0','G1','G2','G3','G4')]) for oid in (0,1)}
        write_json(frame_out/'matched_count_contract.json',dict(counts=matched_counts,
            eligible_for_nonempty_geometry_ranking=all(n>=6 for n in matched_counts.values()),
            scope='geometry-only posthoc matched counts; not used by the primary GNN training/selection',
            GT_used_to_choose_counts=False))
        for name in VARIANTS:
            poses = {oid:o.world_from_object for oid,o in maps[name].objects.items()} if name in maps else None
            measured = evaluator.measure(name,records[name],poses,gt,references,t)
            purity=provenance_metrics(name,records[name],gt_history,source_label_cache,t)
            for r in measured:
                r.update(seed=seed, split=episode['split'], condition='estimated' if name.startswith('G') else 'oracle')
                r.update(purity[r['object_id']])
                if name in maps:
                    obj = maps[name].objects[r['object_id']]
                    r.update(tracking_state=obj.tracking_state, **{k:v for k,v in obj.registration.items()
                             if k in ('residual_m','inlier_ratio','rank')})
            frame_metrics.extend(measured)
            graph = (common_baseline(g0_graph,point_cap=point_cap) if name=='G0' else
                     build_graph(records[name],data['robot_world_geometry'],data['tcp_pose'],t,
                         cameras=[same_time[main]] if name in ('G1','G2') else cameras,point_cap=point_cap))
            if name=='G0':
                selected_records=[records[name][i-3] for i in graph.packet['selected_original_rows'] if i>=3]
            else:
                by_id={r['persistent_node_id']:r for r in records[name]}
                selected_records=[by_id[i] for i in graph.packet['node_ids']]
            fixed=evaluator.measure(name,selected_records,poses,gt,references,t,budget_mode=f'cap{point_cap}_per_object')
            fixed_purity=provenance_metrics(name,selected_records,gt_history,source_label_cache,t)
            for r in fixed:
                r.update(seed=seed,split=episode['split'],condition='estimated' if name.startswith('G') else 'oracle')
                r.update(fixed_purity[r['object_id']])
            frame_metrics.extend(fixed)
            matched_graph=(common_baseline(g0_graph,point_cap=matched_counts) if name=='G0' else
                build_graph(records[name],data['robot_world_geometry'],data['tcp_pose'],t,
                    cameras=[same_time[main]] if name in ('G1','G2') else cameras,point_cap=matched_counts))
            if name=='G0':
                matched_records=[records[name][i-3] for i in matched_graph.packet['selected_original_rows'] if i>=3]
            else:
                matched_records=[by_id[i] for i in matched_graph.packet['node_ids']]
            matched=evaluator.measure(name,matched_records,poses,gt,references,t,budget_mode='matched_predicted_N')
            matched_purity=provenance_metrics(name,matched_records,gt_history,source_label_cache,t)
            for r in matched:
                r.update(seed=seed,split=episode['split'],condition='estimated' if name.startswith('G') else 'oracle',
                    matched_target_count=matched_counts[r['object_id']],
                    matched_count_satisfied=r['count']==matched_counts[r['object_id']])
                r.update(matched_purity[r['object_id']])
            frame_metrics.extend(matched)
            save_graph(frame_out/(name+'_matched_N_graph.npz'),matched_graph)
            save_graph(frame_out/(name+'_graph.npz'),graph)
            snapshot_records(frame_out/(name+'_map.npz'),records[name])
            sources=[(point['persistent_node_id'],source) for point in records[name]
                     for source in point['source_observation_ids'] if source[1]==t]
            save_npz(frame_out/(name+'_source_observations.npz'),
                persistent_node_id=np.array([r[0] for r in sources],dtype=str),
                source_episode=np.array([r[1][0] for r in sources],dtype=str),
                source_timestep=np.array([r[1][1] for r in sources],np.int64),
                source_camera=np.array([r[1][2] for r in sources],dtype=str),
                source_pixel=np.array([r[1][3] for r in sources],np.int64))
            if name in maps:
                memory=maps[name]
                state=dict(episode=memory.episode,timestep=memory.clock.timestep,next_id=memory.next_id,
                    geometry_variant=name,condition_type='estimated' if name.startswith('G') else 'oracle',
                    objects={oid:dict(world_from_object=obj.world_from_object,pose_uncertainty=obj.pose_uncertainty,
                        pose_confidence=obj.pose_confidence,tracking_state=obj.tracking_state,
                        last_tracking_timestep=obj.last_tracking_timestep,registration=obj.registration)
                        for oid,obj in memory.objects.items()})
                if isinstance(memory,OracleMotionMap):
                    state['oracle_only']=dict(gt_reference=memory.gt_reference,map_reference=memory.map_reference)
                write_json(frame_out/(name+'_tracker.json'),state)
            for point in records[name]:
                source_provenance[name][point['persistent_node_id']] = dict(object_id=point['object_id'],
                    first_seen_timestep=point['first_seen_timestep'], last_seen_timestep=point['last_seen_timestep'],
                    source_observation_ids=point['source_observation_ids'])
            for r in measured+fixed:
                selected_kind = graph.node_type.numpy()==r['object_id']+3
                r.update(graph_nodes=int(selected_kind.sum()),graph_edges=int(graph.edge_type.numel()),
                         feature_dimension=int(graph.node_input.shape[1]))
            # Explicit status-feature ablation on identical nodes/edges.
            if name=='G4':
                no_status = deepcopy(graph); no_status.node_input[:,14:] = 0
                no_status.packet['status_features'] = False
                save_graph(frame_out/'G4_status_off_graph.npz',no_status)
        rows.extend(frame_metrics)
        write_json(frame_out/'geometry_metrics.json',frame_metrics)
        write_json(frame_out/'fusion_audit.json',fusion_audit)
        costs.append(time.perf_counter()-start)
        frames.append(dict(seed=seed, timestep=t, split=episode['split'],phase=frame['phase'],
            folder=str(frame_out.relative_to(root)),capture_input_sha256=frame['input_sha256'],
            capture_gt_sha256=frame['gt_sha256'],elapsed_seconds=costs[-1]))
        if t%10==0:
            print(f'geometry seed={seed} t={t} time={costs[-1]:.3f}s maps='+str({k:len(v) for k,v in records.items()}),flush=True)
    for name,memory in maps.items():
        write_json(output/(name+'_lifecycle_trace.json'),memory.events)
        write_json(output/(name+'_tracker_final.json'),{oid:dict(pose=obj.world_from_object,
            state=obj.tracking_state,confidence=obj.pose_confidence,uncertainty=obj.pose_uncertainty)
            for oid,obj in memory.objects.items()})
    for name,provenance in source_provenance.items():
        write_json(output/(name+'_point_provenance.json'),provenance)
    write_json(output/'frame_manifest.json',frames)
    write_json(output/'camera_motion_contract.json',dict(status='PASS',mounted_extrinsic_updates=camera_motion))
    csv_write(output/'geometry_metrics.csv',rows); csv_write(output/'robot_filter_metrics.csv',filters)
    return rows,filters,frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--point-cap',type=int,default=128)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    capture=json.loads((args.capture/'capture_manifest.json').read_text())
    metadata=json.loads((args.capture/'metadata.json').read_text())
    if metadata['status']!='finished' or not all(e['complete'] for e in capture['episodes']):
        raise ValueError('only finalized diagnostic captures may be evaluated')
    validate_split(capture['config']['split'],{e['seed']:e['actual_reset_group'] for e in capture['episodes']})
    protected=protection_snapshot([p for p in args.capture.rglob('*') if p.is_file()])
    config=dict(task='09-C',capture=str(args.capture),capture_run_id=args.capture.name,
                variants=list(VARIANTS),point_cap_per_object=args.point_cap,
                geometry_revision='task09c_final_candidate_v1',registration='observable point-to-plane subspace',
                workers=args.workers,
                all_oracles_separate=True,stage='geometry_pilot' if capture['config']['pilot'] else 'geometry_expanded')
    torch.set_num_threads(1)
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_geometry_variants',
                          job_type='evaluate',root=args.output) as logger:
        root=logger.directory
        from .collect import inventory_sources
        inventory_sources(root)
        logger.record(source_run_id=args.capture.name)
        print('TASK09C_GEOMETRY_DIRECTORY='+str(root),flush=True)
        write_json(root/'protection_before.json',protected)
        rows,filters,frames=[],[],[]
        if args.workers<1:raise ValueError('positive worker count required')
        if args.workers==1:
            results=[run_episode(root,args.capture,e,point_cap=args.point_cap) for e in capture['episodes']]
        else:
            # Each worker owns an independent episode/map and unique output
            # folder. Only the parent writes the common ExperimentLogger.
            results_by_seed={}
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                future_to_seed={pool.submit(run_episode,root,args.capture,e,point_cap=args.point_cap):e['seed'] for e in capture['episodes']}
                for future in as_completed(future_to_seed):
                    seed=future_to_seed[future]
                    results_by_seed[seed]=future.result()
                    logger.log(dict(completed_episodes=len(results_by_seed)))
            results=[results_by_seed[e['seed']] for e in capture['episodes']]
        for rr,ff,fr in results:
            rows.extend(rr);filters.extend(ff);frames.extend(fr)
        write_json(root/'geometry_manifest.json',dict(config=config,frames=frames))
        logger.log(dict(geometry_frames=len(frames)))
        csv_write(root/'geometry_metrics.csv',rows);csv_write(root/'robot_filter_metrics.csv',filters)
        audit=verify_protection(protected);write_json(root/'protection_audit.json',audit)
        if audit['status']!='PASS':
            raise ValueError('original diagnostic capture changed')
        for name,kind in (('geometry_metrics.csv','reports'),('robot_filter_metrics.csv','reports'),
                          ('geometry_manifest.json','metrics'),('protection_audit.json','metrics')):
            logger.link_output(root/name,kind=kind)


if __name__=='__main__':
    main()
