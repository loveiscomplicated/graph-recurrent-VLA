"""Full causal Legacy replay with stable copies of existing same-capture G rows.

Only finalized per-episode artifacts are accepted: frame manifest and final
tracker/lifecycle must exist; each copied file is hashed before/after copying.
The active parent's metadata may still be running on other episodes. No active
output is read as a completed episode. This stage never writes to the 09-C run.
"""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import write_json,save_npz,sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.evaluate import load_cameras,snapshot_records,save_graph
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.metrics import GeometryEvaluator,sample_meshes,provenance_metrics
from graph.persistent_geometry.graph import build_graph
from .pipeline import LegacySystem
from .benchmark import BASE,record_metrics,verify_input

def stable_copy(source,dest,identity):
    before=sha256(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if sha256(source)!=before or sha256(dest)!=before:raise ValueError('source artifact not stable')
    identity[str(source)]=before

def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--seed',type=int,default=30002);args=p.parse_args()
    original=json.loads((args.geometry/'geometry_manifest.json').read_text());capture=Path(original['config']['capture'])
    manifest=json.loads((capture/'capture_manifest.json').read_text());ep=next(e for e in manifest['episodes'] if e['seed']==args.seed)
    ref=args.reference/'geometry'/f'seed{args.seed}'
    for n in ('frame_manifest.json','G2_tracker_final.json','G4_tracker_final.json','G2_lifecycle_trace.json','G4_lifecycle_trace.json'):
        if not (ref/n).exists():raise RuntimeError('BLOCKED_INPUT: reference episode not finalized')
    rf=json.loads((ref/'frame_manifest.json').read_text())
    if len(rf)!=len(ep['frames']):raise ValueError('reference episode length mismatch')
    src=capture/'collection'/f'seed{args.seed}';contract=json.loads((src/'sensor_contract.json').read_text())
    main,wrist=contract['roles']['main'],contract['roles']['wrist']
    with np.load(src/'robot_meshes.npz',allow_pickle=False) as a:names=a['links'].tolist();vertices=[a[f'vertices_{i}'] for i in range(len(names))]
    cfg=dict(stage='full_temporal_replay',seed=args.seed,capture=str(capture),reference=str(args.reference),
        reference_scope='completed episode files only; parent may run other episodes',metric_interval=10,
        no_future_backfill=True,FPS_plus_F5=True,same_map_algorithm=True)
    torch.set_num_threads(1)
    with ExperimentLogger(team='integration',config=cfg,seed=42,model='legacy_fps_full_temporal',job_type='evaluate',root=BASE/'temporal_runs') as logger:
        root=logger.directory;(BASE/'latest_temporal.txt').write_text(str(root)+'\n');print('LEGACY_TEMPORAL_RUN='+str(root),flush=True)
        system=LegacySystem(main,wrist);mesh=sample_meshes(src/'evaluation_meshes.npz');evaluator=GeometryEvaluator(mesh)
        history={};cache={};rows=[];frames=[];identities={};costs=[]
        stable_copy(ref/'geometry_metrics.csv',root/'G_reference_geometry_metrics.csv',identities)
        for frame in ep['frames']:
            t,stem=frame['timestep'],frame['stem'];folder=root/'geometry'/f'seed{args.seed}'/stem;folder.mkdir(parents=True)
            matched_reference=next(f for f in rf if int(f['timestep'])==t)
            if matched_reference['capture_input_sha256']!=frame['input_sha256']:raise ValueError('G/L observation identity mismatch')
            for suffix,key in (('_input.npz','input_sha256'),('_masks.npz','mask_sha256'),('_gt.npz','gt_sha256')):verify_input(src/(stem+suffix),frame[key])
            cameras,data=load_cameras(src/(stem+'_input.npz'));by={c.camera:c for c in cameras}
            with np.load(src/(stem+'_masks.npz'),allow_pickle=False) as a:masks={k:a[k] for k in a.files}
            with np.load(src/(stem+'_gt.npz'),allow_pickle=False) as a:gt={k:a[k] for k in a.files}
            history[t]=gt;links=[LinkSurface(n,v,p) for n,v,p in zip(names,vertices,data['robot_link_world'])]
            start=time.perf_counter();l=system.process(cameras,data,masks,links);elapsed=time.perf_counter()-start
            refs=evaluator.references(cameras,gt,main)
            # G0/G1/G3 reference frames and persistent G2/G4 are from the same
            # stable source snapshot and identical original capture input.
            for name in ('G0','G1','G2','G3','G4'):
                for suffix in ('_map.npz','_graph.npz'):
                    stable_copy(ref/stem/(name+suffix),folder/(name+suffix),identities)
            for name in ('L0','L1','L2','L3','L4'):
                rr=l['records'][name];snapshot_records(folder/(name+'_map.npz'),rr)
                write_json(folder/(name+'_provenance.json'),[{k:r[k] for k in ('persistent_node_id','object_id','source_observation_ids')} for r in rr])
                if t%10==0 or t in (25,65,len(ep['frames'])-1):
                    measured=record_metrics(evaluator,name,rr,system,gt,refs,t,history,cache,'native_variable_N',by[main],mesh)
                    rows.extend(dict(seed=args.seed,phase=frame['phase'],**r) for r in measured)
                if t%5==0:
                    graph=build_graph(rr,data['robot_world_geometry'],data['tcp_pose'],t,cameras=[by[main]] if name in ('L0','L1','L2') else cameras,point_cap=128)
                    save_graph(folder/(name+'_surface_graph.npz'),graph)
            for name,memory in (('L2',system.main_map),('L4',system.fused_map)):
                write_json(folder/(name+'_tracker.json'),{oid:dict(pose=o.world_from_object,state=o.tracking_state,registration=o.registration,confidence=o.pose_confidence,uncertainty=o.pose_uncertainty) for oid,o in memory.objects.items()})
            costs.append(dict(timestep=t,legacy_pipeline_seconds=elapsed))
            frames.append(dict(seed=args.seed,timestep=t,split=ep['split'],phase=frame['phase'],folder=str(folder.relative_to(root)),input_sha256=frame['input_sha256'],mask_sha256=frame['mask_sha256']))
            if t%5==0:print('legacy temporal t='+str(t)+' seconds='+str(round(elapsed,3)),flush=True)
            if shutil.disk_usage(BASE).free<3*1024**3:raise RuntimeError('BLOCKED_RESOURCE: disk reserve3GiB')
        write_json(root/'geometry_manifest.json',dict(config=cfg,frames=frames));csv_write(root/'geometry_metrics.csv',rows)
        write_json(root/'reference_source_identity.json',identities);write_json(root/'costs.json',costs)
        for name,memory in (('L2',system.main_map),('L4',system.fused_map)):write_json(root/(name+'_lifecycle.json'),memory.events)
        for name in ('geometry_manifest.json','geometry_metrics.csv','reference_source_identity.json'):logger.link_output(root/name,kind='metrics')
if __name__=='__main__':main()
