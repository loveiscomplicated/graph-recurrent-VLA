"""Evaluate same-robot interventions from exact causal parent-map snapshots."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

from graph.integration_baseline.geometry09b import target as position_target
from graph.integration_baseline.io import save_npz,save_packet,write_json,sha256
from utils.experiment_logger import ExperimentLogger
from visual.runtime.vision_packet import pose_matrix
from .contracts import transform
from .self_filter import LinkSurface
from .fusion import make_packet,filtered_masks,observations_from_packet
from .history import restore_history
from .pipeline import GeometrySystem
from .graph import build_graph,common_baseline
from .evaluate import load_cameras,save_graph,snapshot_records,VARIANTS
from .metrics import visible_reference,geometry_summary_numpy
from .audit import protection_snapshot,verify_protection
from .collect import inventory_sources


def evaluate_episode(root,capture,geometry,episode,point_cap):
    source=capture/'collection'/f'seed{episode["seed"]}'
    contract=json.loads((source/'sensor_contract.json').read_text())
    roles=contract['roles']
    with np.load(source/'robot_meshes.npz',allow_pickle=False) as a:
        links=a['links'].tolist();vertices=[a[f'vertices_{i}'] for i in range(len(links))]
    frames=[];repeat_checks=[];parent_snapshots={}
    for condition in episode.get('branches',[]):
        t=condition['timestep'];branch=source/condition['source']
        for filename,key in (('input.npz','input_sha256'),('masks.npz','masks_sha256'),('gt.npz','gt_sha256')):
            if sha256(branch/filename)!=condition[key]:raise ValueError('branch capture changed')
        cameras,data=load_cameras(branch/'input.npz')
        with np.load(branch/'masks.npz',allow_pickle=False) as a:masks={k:a[k] for k in a.files}
        with np.load(branch/'gt.npz',allow_pickle=False) as a:gt={k:a[k] for k in a.files}
        baseline_cameras,baseline=load_cameras(source/f'f{t:04d}_input.npz')
        np.testing.assert_array_equal(data['robot76'],baseline['robot76'])
        for current,old in zip(cameras,baseline_cameras):
            for key in ('intrinsic','extrinsic_cv','cam2world_gl','tcp_pose'):
                np.testing.assert_array_equal(getattr(current,key),getattr(old,key))
        namespace=condition['episode']
        if t not in parent_snapshots:
            # Each immutable parent prefix is read once. Every intervention
            # receives an independent deep copy; no branch can affect another.
            parent_snapshots[t]={name:restore_history(geometry,episode,name,t-1,
                episode['episode']+f':cached_prefix_t{t}') for name in ('G2','G4','O1','O2','O3')}
        maps=deepcopy(parent_snapshots[t])
        for memory in maps.values():
            memory.episode=namespace;memory.clock.episode=namespace
        system=GeometrySystem(roles['main'],roles['wrist'],maps['G2'],maps['G4'])
        surfaces=[LinkSurface(link,v,p) for link,v,p in zip(links,vertices,data['robot_link_world'])]
        result=system.process(cameras,data,masks,surfaces)
        records=result['records'];oracle_observations=[]
        for camera in cameras:
            oracle_masks=np.stack([gt[camera.camera+'/segmentation']==i for i in gt['actor_ids']])
            oracle_masks,_=filtered_masks(oracle_masks,result['filter_outputs'][camera.camera])
            packet,_=make_packet(camera,oracle_masks)
            oracle_observations.extend(observations_from_packet(packet,camera))
        maps['O1'].update(namespace,t,oracle_observations,cameras=cameras);records['O1']=maps['O1'].records()
        for name,observation in (('O2',result['dual_observations']),('O3',oracle_observations)):
            maps[name].update_with_gt_motion(namespace,t,observation,gt['actor_pose'],cameras=cameras)
            records[name]=maps[name].records()
        output=root/'branches'/f'seed{episode["seed"]}'/f't{t:04d}'/condition['condition']
        output.mkdir(parents=True,exist_ok=False)
        save_packet(output/'G0_original_packet.npz',result['baseline_packet'])
        relative=np.linalg.inv(pose_matrix(data['tcp_pose']))
        target=np.concatenate([position_target(gt['actor_pose'][oid],data['tcp_pose']) for oid in (0,1)])
        visible=visible_reference(cameras,gt)
        shape=np.concatenate([geometry_summary_numpy(transform(visible[oid],relative)) for oid in (0,1)])
        valid=np.r_[np.ones(6,bool),np.repeat([bool(len(visible[oid])) for oid in (0,1)],15)]
        save_npz(output/'evaluation_targets.npz',target36=np.r_[target,shape],target_valid=valid,
            robot76=data['robot76'],timestep=np.array(t),episode=np.array(namespace))
        for name in VARIANTS:
            selected=cameras if name not in ('G1','G2') else [c for c in cameras if c.camera==roles['main']]
            graph=(common_baseline(result['baseline_graph'],point_cap=point_cap) if name=='G0' else
                build_graph(records[name],data['robot_world_geometry'],data['tcp_pose'],t,cameras=selected,point_cap=point_cap))
            save_graph(output/(name+'_graph.npz'),graph)
            snapshot_records(output/(name+'_map.npz'),records[name])
            write_json(output/(name+'_point_provenance.json'),{r['persistent_node_id']:r['source_observation_ids'] for r in records[name]})
            if name=='G4':
                off=deepcopy(graph);off.node_input[:,14:]=0;off.packet['status_features']=False
                save_graph(output/'G4_status_off_graph.npz',off)
            if condition['condition']=='baseline_repeat':
                path=geometry/'geometry'/f'seed{episode["seed"]}'/f'f{t:04d}'/(name+'_graph.npz')
                with np.load(path,allow_pickle=False) as a:
                    for field in ('node_input','node_type','edge_index','edge_attr','edge_type'):
                        actual=getattr(graph,field).numpy()
                        if actual.dtype.kind in 'fc':np.testing.assert_allclose(actual,a[field],atol=1e-6,rtol=1e-5)
                        else:np.testing.assert_array_equal(actual,a[field])
                repeat_checks.append(dict(seed=episode['seed'],timestep=t,variant=name,exact_history_repeat=True))
        frame=dict(seed=episode['seed'],timestep=t,split=episode['split'],condition=condition['condition'],
            physical_valid=condition['physical_valid'],folder=str(output.resolve()),
            parent_history_through=t-1,parent_episode=episode['episode'],branch_episode=namespace,
            source_input_sha256=condition['input_sha256'],same_robot_state=True)
        frames.append(frame)
        print(f'branch geometry seed={episode["seed"]} t={t} {condition["condition"]} physical_valid={condition["physical_valid"]}',flush=True)
    return frames,repeat_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    capture=json.loads((args.capture/'capture_manifest.json').read_text())
    geometry=json.loads((args.geometry/'geometry_manifest.json').read_text())
    for root in (args.capture,args.geometry):
        if json.loads((root/'metadata.json').read_text())['status']!='finished':raise ValueError('unfinished parent artifact')
    if Path(geometry['config']['capture']).resolve()!=args.capture.resolve():
        raise ValueError('parent map and branch capture identity mismatch')
    protected=protection_snapshot(p for parent in (args.capture,args.geometry) for p in parent.rglob('*') if p.is_file())
    config=dict(task='09-C',stage='causal_branch_geometry',capture=str(args.capture),geometry=str(args.geometry),
        point_cap_per_object=geometry['config']['point_cap_per_object'],future_history_used=False,
        original_episode_reset_semantics_unchanged=True)
    torch.set_num_threads(1)
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_causal_branch_geometry',
                          job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_BRANCH_GEOMETRY_DIRECTORY='+str(root),flush=True)
        write_json(root/'protection_before.json',protected);inventory_sources(root)
        branch_frames=[];repeats=[]
        for episode in capture['episodes']:
            rows,checks=evaluate_episode(root,args.capture,args.geometry,episode,config['point_cap_per_object'])
            branch_frames.extend(rows);repeats.extend(checks)
        frames=[{**f,'folder':str((args.geometry/f['folder']).resolve()),'condition':'baseline','physical_valid':True}
                for f in geometry['frames']]
        frames.extend(branch_frames)
        write_json(root/'geometry_manifest.json',dict(config=config,frames=frames))
        write_json(root/'branch_manifest.json',dict(frames=branch_frames,repeat_checks=repeats))
        audit=verify_protection(protected);write_json(root/'protection_audit.json',audit)
        if audit['status']!='PASS':raise ValueError('causal parent artifacts changed')
        logger.link_output(root/'geometry_manifest.json',kind='metrics')
        logger.link_output(root/'branch_manifest.json',kind='metrics')
        logger.link_output(root/'protection_audit.json',kind='metrics')


if __name__=='__main__':main()
