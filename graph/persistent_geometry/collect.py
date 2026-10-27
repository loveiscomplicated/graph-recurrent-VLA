"""Independent live expert geometry diagnostics, never a BC data extension.

This collector writes a sensor/robot input archive and a separate GT sidecar.
No policy, GRU or action head is constructed or executed. Expert commands only
advance the physical diagnostic sequence; they never become probe targets.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

from envs.maniskill import make_peg_insertion_env, get_gt_state
from graph.integration_baseline.robot import robot_observation, state_schema, flatten_state
from graph.integration_baseline.rollout_stop_clock import reset_diagnostics, numpy
from graph.integration_baseline.rollout_stop_clock import PhysicsAudit
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.io import write_json, save_npz, sha256
from recurrent.expert import ScriptedExpert
from utils.experiment_logger import ExperimentLogger
from visual.runtime.grounding import GDino
from visual.runtime.tracking import SAM2Tracker, DEFAULT_CHECKPOINT

from .contracts import CameraObservation, synchronized
from .self_filter import robot_surfaces

FULL_PLAN = Path('/workspace/task08/experiments/3ed77d8b64884f879b9a628e34a9426a/plan.json')
TASK09B = Path('/workspace/task09b/experiments/64414e993c6e42e3999894e8940668d4')


def configuration(plan, pilot=False):
    train = [s for s in plan['splits']['train'] if s != 30003]
    return dict(task='09-C', seed=42, pilot=pilot, geometry_revision='task09c_final_candidate_v1',
        split=dict(train=train[:2 if pilot else 8],
                   validation=plan['splits']['validation'][:1 if pilot else 4],
                   evaluation=[] if pilot else [62091, 62092]),
        split_selection='first frozen split order, exclude previously selected30003; final fresh seeds fixed before results',
        data_kind='LIVE_DIAGNOSTIC; current ScriptedExpert; original HDF5 read-only identity/replay comparison',
        horizon=250, control_mode='pd_ee_delta_pose', simulation='physx_cpu', perception='cuda',
        branch_timesteps=[] if pilot else [0,25,65],
        branch_seeds=[] if pilot else train[:2]+plan['splits']['validation'][:2]+[62091,62092],
        branch_conditions='same-pose baseline repeat; Peg X+10/Y-10mm, Box X-10/Y+10mm; identical robot and causal prefix; collision audited',
        sensor_contract='installed sensor names/roles discovered by mount; original signed-int16-mm depths retained',
        gt_scope='separate same-time evaluation/oracle sidecar; not a sensor input',
        sampler=dict(keep_ratio=.5, original_G0='Shape-First/PCA5/F5/Raw EDE unmodified'),
        geometry=dict(merge_distance=.003, registration_gate_m=.012, robot_surface_tolerance_m=.0025,
                      observability_singular_ratio=.05,
                      robot_filter='per-shape front depth or strict measured collision-volume interior v2',
                      tracking='point-to-plane observable subspace; explicit partial observability; P2P fallback without normals',
                      robot_uncertain_band_m=.006, surface_edge_radius_m=.02, point_cap_per_object=128),
        encoders=dict(E0='edge-aware L2 d64', E1='same node encoder, no messages',
                      E2='E0 + explicit observed geometry summary', input_dim=21,
                      seeds=[42], optimizer='AdamW', lr=.001, updates=2000, batch=16,
                      selection='minimum validation geometry loss; same fixed optimization budget',
                      target='TCP-relative actor position6 and visible surface summary30, no goals/actions'),
        probes=dict(ridge_alphas=[0., .01, .1, 1., 10., 100.], mlp_hidden=64,
                    mlp_epochs=2000, mlp_patience=150, mlp_weight_decay=[.0001, .01]),
        no_future_backfill=True, policy_training=False, gru_execution=False, no_publish=True)


def sensor_inputs(env, obs, episode, timestep):
    task = env.unwrapped
    fields, pads = robot_observation(env)
    schema = state_schema(fields)
    if int(task.elapsed_steps.item()) != timestep:
        raise ValueError('simulator/observation timestep mismatch')
    cameras = []
    for name, data in obs['sensor_data'].items():
        p = obs['sensor_param'][name]
        cameras.append(CameraObservation(episode, timestep, name, timestep/task.control_freq,
            numpy(data['rgb'])[0].copy(), numpy(data['depth'])[0, ..., 0].copy(),
            numpy(p['intrinsic_cv'])[0].copy(), numpy(p['extrinsic_cv'])[0].copy(),
            numpy(p['cam2world_gl'])[0].copy(), fields['tcp_pose'].copy()))
    synchronized(cameras)
    np.testing.assert_array_equal(numpy(obs['agent']['qpos'])[0], fields['qpos'])
    return cameras, flatten_state(fields, schema), pads, schema


def evaluation_sidecar(env, obs):
    """The only collection function that reads object GT/segmentation."""
    state = get_gt_state(env)
    task = env.unwrapped
    actor_ids = np.array([task.peg._objs[0].per_scene_id, task.box._objs[0].per_scene_id], np.int64)
    robot_ids = np.array([link._objs[0].entity.per_scene_id for link in task.agent.robot.get_links()], np.int64)
    result = dict(actor_pose=np.stack((numpy(state.peg_pose)[0], numpy(state.box_pose)[0])),
                  actor_ids=actor_ids, robot_ids=robot_ids,
                  hole_pose=numpy(state.hole_pose)[0], hole_radius=numpy(state.hole_radius)[0],
                  peg_half_sizes=numpy(state.peg_half_sizes)[0])
    for name, data in obs['sensor_data'].items():
        result[name+'/segmentation'] = numpy(data['segmentation'])[0, ..., 0].copy()
    return result


def snapshot_meshes(env, folder):
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    surfaces = robot_surfaces(env)
    save_npz(folder/'robot_meshes.npz', **{f'vertices_{i}':s.local_vertices for i,s in enumerate(surfaces)},
             links=np.array([s.link for s in surfaces]))
    arrays = {}
    for oid, actor in enumerate((env.unwrapped.peg, env.unwrapped.box)):
        meshes = [m for c in actor._objs[0].components if isinstance(c, physx.PhysxRigidBaseComponent)
                  for m in get_component_meshes(c) if m is not None]
        for i, mesh in enumerate(meshes):
            arrays[f'object{oid}_vertices{i}'] = np.asarray(mesh.vertices).copy()
            arrays[f'object{oid}_faces{i}'] = np.asarray(mesh.faces).copy()
    save_npz(folder/'evaluation_meshes.npz', **arrays)
    urdf = Path(env.unwrapped.agent.urdf_path)
    shutil.copyfile(urdf, folder/'robot.urdf')
    write_json(folder/'geometry_identity.json', dict(robot_urdf_path=str(urdf), robot_urdf_sha256=sha256(urdf),
        link_shapes=[dict(link=s.link, vertices=len(s.local_vertices)) for s in surfaces],
        object_mesh_scope='evaluation/oracle only, excluded from input path'))


def sensor_roles(env):
    roles = {}
    detail = {}
    for name, sensor in env.unwrapped._sensors.items():
        mounted = sensor.entity is not None
        role = 'wrist' if mounted else 'main'
        if role in roles:
            raise ValueError('ambiguous sensor role; explicit calibrated selection needed')
        roles[role] = name
        detail[name] = dict(role=role, mount=str(sensor.config.mount),
            mount_link_name=sensor.entity.name if sensor.entity is not None else None,
            entity_uid=str(sensor.config.entity_uid), config_pose=sensor.config.pose.to_transformation_matrix().tolist())
    if set(roles) != {'main', 'wrist'}:
        raise ValueError('actual main+wrist camera support not available')
    return roles, detail


def collect_episode(root, seed, split, detector, tracker, original, config):
    from mani_skill.utils.registration import TimeLimitWrapper
    folder = root/'collection'/f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    env = TimeLimitWrapper(make_peg_insertion_env(device='cpu', obs_mode='rgb+depth+segmentation',
        control_mode=config['control_mode']).unwrapped, config['horizon'])
    obs, _ = env.reset(seed=seed)
    episode = root.name+f':seed{seed}'
    actual_reset = reset_diagnostics(env)
    roles, detail = sensor_roles(env)
    adapters = {name:OnlineVisionAdapter(detector, tracker) for name in roles.values()}
    for adapter in adapters.values():
        adapter.reset(episode)
    expert = ScriptedExpert(env)
    physics=PhysicsAudit(env)
    snapshot_meshes(env, folder)
    source = original.get(seed)
    source_h5 = h5py.File(source['absolute_path'], 'r') if source else None
    if source and sha256(source['absolute_path']) != source['sha256']:
        raise ValueError('frozen HDF5 changed')
    frames, costs, branches = [], [], []
    start = time.perf_counter()
    try:
        for timestep in range(config['horizon']+1):
            tick = time.perf_counter()
            cameras, robot76, pads, schema = sensor_inputs(env, obs, episode, timestep)
            # All sidecars address precisely the observation above, before act.
            gt = evaluation_sidecar(env, obs)
            surfaces = robot_surfaces(env)
            sensor = dict(robot76=robot76, robot_world_geometry=pads, tcp_pose=cameras[0].tcp_pose,
                          qpos=numpy(env.unwrapped.agent.robot.get_qpos())[0],
                          robot_link_world=np.stack([s.world_from_link for s in surfaces]),
                          timestep=np.int64(timestep), timestamp=np.float64(cameras[0].timestamp),
                          episode=np.array(episode))
            masks, vision_record = {}, {}
            branch_now=seed in config['branch_seeds'] and timestep in config['branch_timesteps']
            if branch_now:
                from graph.integration_baseline.geometry09b import memory
                pre={name:memory(adapter) for name,adapter in adapters.items()}
            for camera in cameras:
                values = {k:getattr(camera,k) for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl')}
                sensor.update({camera.camera+'/'+k:v for k,v in values.items()})
                masks[camera.camera], vision_record[camera.camera] = adapters[camera.camera].process(camera.rgb, episode, timestep)
            stem = f'f{timestep:04d}'
            save_npz(folder/(stem+'_input.npz'), **sensor)
            save_npz(folder/(stem+'_masks.npz'), **masks)
            save_npz(folder/(stem+'_gt.npz'), **gt)
            if branch_now:
                from .branches import collect_branches
                branch_rows,obs=collect_branches(env,adapters,pre,sensor,masks,episode,timestep,folder,physics)
                branches.extend(branch_rows)
                del pre
            compare = None
            if source_h5 is not None and timestep < len(source_h5['robot_state']):
                compare = dict(robot76_max=float(np.max(np.abs(source_h5['robot_state'][timestep]-robot76))),
                    tcp_max=float(np.max(np.abs(source_h5['tcp_pose'][timestep]-cameras[0].tcp_pose))))
            record = dict(timestep=timestep, stem=stem, phase=str(expert.phase), split=split,
                          original_comparison=compare, vision=vision_record,
                          input_sha256=sha256(folder/(stem+'_input.npz')),
                          mask_sha256=sha256(folder/(stem+'_masks.npz')),
                          gt_sha256=sha256(folder/(stem+'_gt.npz')))
            frames.append(record)
            costs.append(time.perf_counter()-tick)
            # A collection-time independent camera calibration anchor: the
            # mounted pose must remain fixed in the actual robot mount frame.
            if timestep == 0:
                write_json(folder/'sensor_contract.json', dict(roles=roles, installed=detail,
                    camera_shapes={c.camera:list(c.rgb.shape) for c in cameras},
                    same_physics_timestep=True, depth_unit='signed int16 millimetres; >0 valid; <=0 invalid',
                    coordinate_convention='CV z; pixel centres u+.5/v+.5; extrinsic world-to-camera',
                    backend=str(env.unwrapped.backend), schema=schema))
            if timestep%10 == 0:
                print(f'capture seed={seed} split={split} t={timestep} phase={expert.phase} wall={sum(costs):.1f}s', flush=True)
                write_json(folder/'progress.json', dict(episode=episode, frames=frames, complete=False))
            if timestep >= config['horizon']:
                break
            # Expert is collection-only. No action appears in sensor archives.
            action = expert.act()
            obs, reward, terminated, truncated, info = env.step(torch.from_numpy(action[None]))
            if bool(terminated.item()) or bool(truncated.item()):
                # Save the actual final observation by letting the next loop
                # iteration run, then terminate before another expert step.
                final_timestep = timestep+1
                config_terminal = True
            else:
                config_terminal = False
            if config_terminal:
                # The final frame is important for temporal coverage. A local
                # stop bound carries no hidden future information into Vision.
                config = {**config, 'horizon':final_timestep}
        manifest = dict(episode=episode, seed=seed, split=split, actual_reset_group=actual_reset,
            original=source, original_reset_match=bool(source and source['reset_group'] == actual_reset),
            branches=branches,
            source_kind='LIVE_DIAGNOSTIC', frames=frames, complete=True, roles=roles,
            elapsed_seconds=time.perf_counter()-start,
            frame_wall_mean_s=float(np.mean(costs)), frame_wall_p95_s=float(np.percentile(costs,95)))
        write_json(folder/'manifest.json', manifest)
        return manifest
    finally:
        if source_h5 is not None:
            source_h5.close()
        for adapter in adapters.values():
            adapter.close()
        env.close()


def inventory_sources(root):
    base = Path(__file__).resolve().parents[2]
    sources = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(base) and path.suffix == '.py':
                relative = str(path.relative_to(base))
                sources[relative] = sha256(path)
                destination = root/'source_snapshot'/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    write_json(root/'source_hashes.json', sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--branch-smoke',action='store_true')
    parser.add_argument('--development',action='store_true',help='defer independent evaluation until weights/selection rules are frozen')
    parser.add_argument('--evaluation-only',action='store_true')
    args = parser.parse_args()
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('WANDB_MODE') != 'disabled':
        raise ValueError('offline model cache and disabled external logging required')
    torch.set_num_threads(1); np.random.seed(42); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads(FULL_PLAN.read_text())
    config = configuration(plan, args.pilot)
    config['reserved_independent_evaluation']=[62091,62092]
    if sum((args.pilot,args.branch_smoke,args.development,args.evaluation_only))>1:
        raise ValueError('choose one collection stage')
    if args.branch_smoke:
        config['split']={'train':config['split']['train'][:1],'validation':[],'evaluation':[]}
        config['stage']='branch_contract_smoke'
    elif args.development:
        config['split']['evaluation']=[]
        config['stage']='expanded_development'
    elif args.evaluation_only:
        config['split']={'train':[],'validation':[],'evaluation':[62091,62092]}
        config['stage']='frozen_independent_evaluation'
    data_directory = Path(plan['dataset_directory'])
    if (data_directory/'data_source.json').exists():
        data_directory = Path(json.loads((data_directory/'data_source.json').read_text())['directory'])
    manifest = json.loads((data_directory/'dataset_manifest.json').read_text())
    original = {e['episode_id']:{**{k:e[k] for k in ('sha256','reset_group','path')},
                              'absolute_path':str(data_directory/e['path'])} for e in manifest['episodes']}
    original_protection = {str(data_directory/p):sha256(data_directory/p) for p in
                           ('dataset_manifest.json','split_manifest.json','robot_state_schema.json')}
    original_protection[str(FULL_PLAN.parent/'status.json')] = sha256(FULL_PLAN.parent/'status.json')
    for split in ('train','validation'):
        for seed in config['split'][split]:
            original_protection[original[seed]['absolute_path']] = original[seed]['sha256']
    original_protection[str(DEFAULT_CHECKPOINT)] = sha256(DEFAULT_CHECKPOINT)
    if shutil.disk_usage(args.output.parent).free < 3*1024**3:
        raise RuntimeError('BLOCKED_RESOURCE: less than3GiB free')
    with ExperimentLogger(team='integration', config=config, seed=42,
                          model='task09c_live_geometry_capture', job_type='evaluate', root=args.output) as logger:
        root = Path(logger.directory)
        print('TASK09C_RUN_DIRECTORY='+str(root), flush=True)
        write_json(root/'protection_before.json', original_protection)
        write_json(root/'split_manifest.json', config['split'])
        write_json(root/'dataset_identity.json', dict(directory=str(data_directory),
            dataset_sha256=sha256(data_directory/'dataset_manifest.json'),
            split_sha256=sha256(data_directory/'split_manifest.json'),
            original_dataset_id=manifest['dataset_id'], original_wrist_present=True,
            reason_for_live_capture='stored original HDF5 has both cameras but no per-timestep actor pose evaluation sidecar',
            original_data_not_modified=True))
        detector = GDino(device='cuda'); tracker = SAM2Tracker(device='cuda')
        episodes = []
        for split, seeds in config['split'].items():
            for seed in seeds:
                episode = collect_episode(root, seed, split, detector, tracker, original, config)
                episodes.append(episode)
                write_json(root/'capture_manifest.json', dict(config=config, episodes=episodes))
                logger.log(dict(captured_episodes=len(episodes), frames=sum(len(e['frames']) for e in episodes)))
        groups = [e['actual_reset_group'] for e in episodes]
        if len(groups) != len(set(groups)):
            raise ValueError('actual reset group leakage')
        inventory_sources(root)
        after = {p:sha256(p) for p in original_protection}
        write_json(root/'protection_audit.json', dict(status='PASS' if after == original_protection else 'FAIL',
                                                     hashes=after))
        if after != original_protection:
            raise ValueError('protected source/artifact changed')
        for name in ('capture_manifest.json','split_manifest.json','dataset_identity.json','protection_audit.json','source_hashes.json'):
            logger.link_output(root/name, kind='metrics')


if __name__ == '__main__':
    main()
