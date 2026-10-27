"""Installed-runtime, original G0 readout and protected artifact audit."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from envs.maniskill import make_peg_insertion_env
from graph.integration_baseline.io import write_json, save_npz, sha256, read_packet
from graph.integration_baseline.geometry09b import packet_graph
from graph.integration_baseline.model import GraphEncoder
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.robot import robot_observation
from graph.integration_baseline.rollout_stop_clock import numpy
from recurrent.expert import ScriptedExpert
from utils.experiment_logger import ExperimentLogger

from .collect import FULL_PLAN, TASK09B, sensor_inputs, inventory_sources
from .kinematics import urdf_fk
from .audit import protection_snapshot, verify_protection, validate_split


def candidate_g0_identity(root,capture,plan):
    from .evaluate import load_cameras
    from .fusion import make_packet
    from .pipeline import original_g0
    from graph.integration_baseline.graph import build_graph
    from graph.integration_baseline.robot import geometry_in_ee
    manifest=json.loads((capture/'capture_manifest.json').read_text())
    checks=[]
    for episode in manifest['episodes'][:2]:
        folder=capture/'collection'/f'seed{episode["seed"]}'
        roles=json.loads((folder/'sensor_contract.json').read_text())['roles']
        for t in (0,10,min(90,len(episode['frames'])-1)):
            cameras,data=load_cameras(folder/f'f{t:04d}_input.npz')
            camera=next(c for c in cameras if c.camera==roles['main'])
            with np.load(folder/f'f{t:04d}_masks.npz',allow_pickle=False) as a:masks=a[roles['main']]
            row=dict(rgb=camera.rgb,depth=camera.depth_raw_mm.astype(np.float32)/1000,
                intrinsic=camera.intrinsic,cam2world=camera.world_from_camera,tcp_pose=camera.tcp_pose,
                robot_ee_geometry=geometry_in_ee(data['robot_world_geometry'],camera.tcp_pose))
            old,oldgraph=packet_graph(row,masks,t,episode['seed'],plan)
            new,graph=original_g0(camera,data,masks)
            for key in ('points','camera_points','geometric_features','object_id','intra_object_edge_index',
                        'feature_field_available','feature_field_valid'):
                if not np.array_equal(old[key],new[key],equal_nan=True):raise ValueError('candidate G0 changed '+key)
            for key in ('node_input','node_type','edge_index','edge_attr','edge_type'):
                np.testing.assert_array_equal(getattr(graph,key).numpy(),getattr(oldgraph,key).numpy())
            checks.append(dict(seed=episode['seed'],timestep=t,packet_graph_exact=True))
    write_json(root/'candidate_G0_original_path_identity.json',dict(status='PASS',samples=checks,
        calibration_precision='canonical float32 inverse CV extrinsic, matching existing G0'))


def baseline(root, plan):
    status=json.loads((FULL_PLAN.parent/'status.json').read_text())
    if status['status']!='ERROR':
        raise ValueError('original user-stopped pipeline status changed')
    receipt=json.loads((Path(status['stages']['full_C']['directory'])/'best.json').read_text())
    checkpoint=Path(receipt['path'])
    expected='e615ec3ded96b0242004e08ee13273b782ba4af1032c8a84f205144f4ab8d177'
    if sha256(checkpoint)!=expected:
        raise ValueError('Full_C frozen checkpoint identity')
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
    encoder=GraphEncoder(d=64,L=2).cuda().eval().requires_grad_(False)
    state={k[len('graph_encoder.'):]:v for k,v in cp['model'].items() if k.startswith('graph_encoder.')}
    encoder.load_state_dict(state,strict=True)
    del cp
    before={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    samples=json.loads((TASK09B/'samples.json').read_text())
    selected=[s for s in samples if s['condition']=='baseline' and s['seed'] in (30002,30047,30125)]
    result=[]
    for sample in selected:
        folder=TASK09B/sample['sample_id']
        with np.load(folder/'observation.npz',allow_pickle=False) as a:
            row={k:a[k] for k in a.files}
        row['depth']=row['depth_raw_mm'].astype(np.float32)/1000
        packet,graph=packet_graph(row,row['masks'],sample['t'],sample['seed'],plan)
        saved=read_packet(folder/'packet.npz')
        checks={key:bool(np.array_equal(packet[key],saved[key],equal_nan=True)) for key in
            ('points','camera_points','object_id','geometric_features','feature_field_available',
             'feature_field_valid','intra_object_edge_index')}
        with np.load(folder/'graph.npz',allow_pickle=False) as a:
            checks.update({'graph_'+k:np.array_equal(getattr(graph,k).numpy(),a[k]) for k in a.files})
        with torch.inference_mode():
            z,presence=encoder(pack_graphs([graph]).to('cuda'))
        with np.load(folder/'features.npz',allow_pickle=False) as a:
            error=float(np.max(np.abs(numpy(z)[0]-a['Full_C_readout320'])))
            np.testing.assert_allclose(numpy(z)[0],a['Full_C_readout320'],atol=1e-6,rtol=1e-5)
        if not all(checks.values()):
            raise ValueError('original G0 reconstruction mismatch '+sample['sample_id'])
        result.append(dict(sample=sample['sample_id'],checks=checks,readout_max_error=error))
    unchanged=all(torch.equal(before[k],v) for k,v in encoder.state_dict().items())
    if not unchanged or sha256(checkpoint)!=expected:
        raise ValueError('frozen encoder/checkpoint changed')
    write_json(root/'G0_baseline_reproduction.json',dict(status='PASS',samples=result,
        checkpoint_sha256=expected,encoder_only=True,gru_executed=False,weights_unchanged=unchanged,
        dtype='FP32',device=torch.cuda.get_device_name(0),readout_atol=1e-6,readout_rtol=1e-5))


def runtime(root,capture):
    env=make_peg_insertion_env(device='cpu',obs_mode='rgb+depth+segmentation',control_mode='pd_ee_delta_pose')
    obs,_=env.reset(seed=30002)
    try:
        names=[joint.name for joint in env.unwrapped.agent.robot.get_active_joints()]
        expert=ScriptedExpert(env)
        rows=[]
        for t in range(6):
            fields,pads=robot_observation(env)
            from visual.runtime.vision_packet import pose_matrix
            expected=urdf_fk(env.unwrapped.agent.urdf_path,names,fields['qpos'],pose_matrix(fields['robot_base_pose']))
            links={link.name:link._objs[0].entity.pose.to_transformation_matrix() for link in env.unwrapped.agent.robot.get_links()}
            error={name:float(np.max(np.abs(expected[name]-observed))) for name,observed in links.items()}
            if max(error.values())>3e-6:
                raise ValueError('independent URDF FK mismatch')
            cameras,state,pads,schema=sensor_inputs(env,obs,'runtime_audit',t)
            fresh=env.unwrapped.get_obs()
            for camera in cameras:
                for kind in ('rgb','depth','segmentation'):
                    np.testing.assert_array_equal(numpy(obs['sensor_data'][camera.camera][kind]),numpy(fresh['sensor_data'][camera.camera][kind]))
            rows.append(dict(timestep=t,link_max_error=error,camera_names=[c.camera for c in cameras],
                             camera_rgbd_repeat_exact=True))
            if t<5:
                obs,*_=env.step(torch.from_numpy(expert.act()[None]))
        write_json(root/'installed_runtime_contract.json',dict(status='PASS',rows=rows,
            active_joint_order=names,urdf_sha256=sha256(env.unwrapped.agent.urdf_path),
            backend=str(env.unwrapped.backend),cuda_physx='NOT_EXECUTED',
            independent_fk='XML URDF joint origins/axes and current observed qpos; robot root pose'))
    finally:
        env.close()
    capture_manifest=json.loads((capture/'capture_manifest.json').read_text())
    identity=json.loads((capture/'dataset_identity.json').read_text())
    dataset=Path(identity['directory'])
    stored_manifest=json.loads((dataset/'dataset_manifest.json').read_text())
    episodes={e['episode_id']:e for e in stored_manifest['episodes']}
    source=[]
    for episode in capture_manifest['episodes']:
        seed=episode['seed']; e=episodes[seed]; path=dataset/e['path']
        with h5py.File(path,'r') as f:
            keys={}
            f.visititems(lambda k,v:keys.update({k:dict(shape=list(v.shape),dtype=str(v.dtype))}) if isinstance(v,h5py.Dataset) else None)
        if sha256(path)!=e['sha256'] or not all(f'{name}/rgb' in keys and f'{name}/depth_raw_mm' in keys for name in ('base_camera','hand_camera')):
            raise ValueError('original stored sensor contract')
        source.append(dict(seed=seed,path=str(path),sha256=e['sha256'],keys=keys,
            continuous_both_cameras=True,per_timestep_object_pose=any('peg_pose' in k or 'box_pose' in k for k in keys)))
    validate_split(capture_manifest['config']['split'],{e['seed']:e['actual_reset_group'] for e in capture_manifest['episodes']})
    write_json(root/'original_dataset_sensor_audit.json',dict(status='PASS',episodes=source,
        original_dataset_identity=identity,live_capture_reason='missing time-resolved object-pose evaluation sidecar'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    plan=json.loads(FULL_PLAN.read_text())
    paths=[FULL_PLAN,FULL_PLAN.parent/'status.json']
    historical=json.loads((TASK09B/'protected_before.json').read_text())
    paths.extend(p for p in historical if Path(p).is_file())
    for base in (TASK09B,Path('/workspace/task09a/experiments/073de7b8e59d4f23b650c87c377c1829')):
        paths.extend(p for p in base.rglob('*') if p.is_file())
    protected=protection_snapshot(paths)
    with ExperimentLogger(team='integration',config=dict(task='09-C',stage='0',capture=str(args.capture)),
                          seed=42,model='task09c_preflight',job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_PREFLIGHT_DIRECTORY='+str(root),flush=True)
        write_json(root/'protection_before.json',protected)
        write_json(root/'historical_frozen_identity.json',verify_protection(historical))
        baseline(root,plan);runtime(root,args.capture);candidate_g0_identity(root,args.capture,plan)
        inventory_sources(root)
        audit=verify_protection(protected);write_json(root/'protection_audit.json',audit)
        if audit['status']!='PASS':
            raise ValueError('source/09-A/B artifact protection')
        for name in ('G0_baseline_reproduction.json','installed_runtime_contract.json',
                     'original_dataset_sensor_audit.json','candidate_G0_original_path_identity.json',
                     'protection_audit.json','source_hashes.json'):
            logger.link_output(root/name,kind='metrics')


if __name__=='__main__':
    main()
