"""Opt-in Task06-B direct collection; unchanged ScriptedExpert, at most five resets.

No success filtering, action adapter, replay, training or publishing is performed.
HDF5 is an independent append-time recorder; NPZ is a final in-memory export.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import h5py
import numpy as np
import torch

from envs.maniskill import make_peg_insertion_env
from graph.integration_baseline.io import save_npz, sha256, write_json
from graph.integration_baseline.robot import robot_observation, state_schema, flatten_state, geometry_in_ee
from recurrent.common import to_action_tensor, versions
from recurrent.expert import ScriptedExpert
from utils.experiment_logger import ExperimentLogger


def array(x):
    return x.detach().cpu().numpy().copy()


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def append_h5(stream, name, value):
    value = np.asarray(value)
    if name not in stream:
        stream.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                              dtype=value.dtype, compression='gzip')
    ds = stream[name]
    ds.resize(len(ds)+1, axis=0)
    ds[-1] = value


def capture(env, obs, schema):
    task = env.unwrapped
    before = int(task.elapsed_steps.item())
    fields, pads = robot_observation(env)
    assert state_schema(fields) == schema
    row = dict(robot_state=flatten_state(fields, schema), robot_world_geometry=pads,
               tcp_pose=fields['tcp_pose'], timestep=np.int64(before),
               timestamp=np.float64(before/task.control_freq))
    # Observation returned by reset/step and independently read live accessors.
    for key in ('qpos', 'qvel'):
        np.testing.assert_array_equal(array(obs['agent'][key])[0], fields[key])
    fresh = task.get_obs()
    camera_exact = True
    for camera, data in obs['sensor_data'].items():
        params = obs['sensor_param'][camera]
        for kind in params:
            np.testing.assert_array_equal(array(params[kind]), array(fresh['sensor_param'][camera][kind]))
        for kind in ('rgb', 'depth'):
            np.testing.assert_array_equal(array(data[kind]), array(fresh['sensor_data'][camera][kind]))
        entries = dict(rgb=array(data['rgb'])[0], depth_raw_mm=array(data['depth'])[0, ..., 0],
                       intrinsic=array(params['intrinsic_cv'])[0],
                       extrinsic_cv=array(params['extrinsic_cv'])[0],
                       cam2world_gl=array(params['cam2world_gl'])[0])
        entries['depth'] = entries['depth_raw_mm'].astype(np.float32)/1000
        entries['cam2world'] = np.linalg.inv(np.vstack((entries['extrinsic_cv'], [0, 0, 0, 1]))).astype(np.float32)
        row.update({camera+'/'+k: v for k, v in entries.items()})
    geometry_in_ee(pads, fields['tcp_pose'])
    assert before == int(task.elapsed_steps.item())
    return row, camera_exact


def install_audit(env, schema):
    """Observe actual controller entry and preprocessing; delegate byte-identically."""
    controller = env.unwrapped.agent.controller
    audit = {}
    original = controller.set_action
    def set_action(command):
        fields, pads = robot_observation(env)
        audit.update(controller_input=array(command)[0], controller_pre_state=flatten_state(fields, schema),
                     controller_pre_geometry=pads, controller_timestep=np.int64(env.unwrapped.elapsed_steps.item()))
        return original(command)
    controller.set_action = set_action
    for name, child in controller.controllers.items():
        preprocess = child._preprocess_action
        def wrapped(action, _name=name, _preprocess=preprocess):
            result = _preprocess(action)
            audit[_name+'_scaled_input'] = array(result)[0]
            return result
        child._preprocess_action = wrapped
    return audit


def reset_for_recording(env, seed, schema):
    """reset may replace the controller, so auditing must be installed after it."""
    obs, info = env.reset(seed=seed)
    return obs, info, install_audit(env, schema)


def validate_episode(a, schema, seed, h5_path=None):
    n = len(a['actions'])
    assert a['actions'].shape == (n, 7) and a['actions'].dtype == np.float32
    np.testing.assert_array_equal(a['actions'], a['env_step_input'])
    np.testing.assert_array_equal(a['actions'], a['controller_input'])
    expected_arm = a['actions'][:, :6].copy()
    expected_arm[:, :3] = np.clip(expected_arm[:, :3], -1, 1)*.1
    expected_arm[:, 3:] /= np.maximum(np.linalg.norm(expected_arm[:, 3:], axis=1, keepdims=True), 1)
    expected_arm[:, 3:] *= -.1
    np.testing.assert_allclose(a['arm_scaled_input'], expected_arm, atol=2e-8, rtol=1e-6)
    expected_gripper = (np.clip(a['actions'][:, 6:], -1, 1)+1)*.025-.01
    np.testing.assert_allclose(a['gripper_scaled_input'], expected_gripper, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(a['robot_state'][:-1], a['controller_pre_state'])
    np.testing.assert_array_equal(a['robot_world_geometry'][:-1], a['controller_pre_geometry'])
    np.testing.assert_array_equal(a['controller_timestep'], np.arange(n))
    np.testing.assert_array_equal(a['timestep'], np.arange(n+1))
    np.testing.assert_array_equal(a['timestamp'], np.arange(n+1)/20)
    assert np.all(a['episode_id'] == seed) and np.all(a['reset_seed'] == seed)
    assert a['robot_state'].shape == (n+1, schema['dimension'])
    assert np.isfinite(a['robot_state']).all()
    assert not a['terminated'][:-1].any() and not a['truncated'][:-1].any()
    assert not a['success'][:-1].any()  # stop at first success, no post-success tail
    fields = {r['name']: r for r in schema['fields']}
    for name, index, offset in [('left_finger_pose', 1, [0, .00758, .04525]),
                                ('right_finger_pose', 2, [0, -.00758, .04525])]:
        from visual.runtime.vision_packet import pose_matrix
        spec = fields[name]
        for t in range(n+1):
            mat = pose_matrix(a['robot_state'][t, spec['offset']:spec['offset']+7])
            np.testing.assert_allclose(mat[:3, 3]+mat[:3, :3]@offset,
                                       a['robot_world_geometry'][t, index], rtol=0, atol=1e-6)
    for t in range(n+1):
        geometry_in_ee(a['robot_world_geometry'][t], a['tcp_pose'][t])
    cameras = sorted(k[:-4] for k in a if k.endswith('/rgb'))
    depth = {}
    for camera in cameras:
        raw = a[camera+'/depth_raw_mm']
        assert raw.dtype == np.int16 and len(raw) == n+1
        assert a[camera+'/rgb'].shape == (n+1, 128, 128, 3)
        np.testing.assert_array_equal(a[camera+'/depth'], raw.astype(np.float32)/1000)
        ext = np.zeros((n+1, 4, 4)); ext[:, :3] = a[camera+'/extrinsic_cv']; ext[:, 3, 3] = 1
        np.testing.assert_allclose(ext@a[camera+'/cam2world'], np.broadcast_to(np.eye(4), (n+1,4,4)), atol=2e-6, rtol=0)
        depth[camera] = dict(unit='mm', dtype=str(raw.dtype), min=int(raw.min()), max=int(raw.max()),
                             nonpositive=int((raw <= 0).sum()), int16_min=int((raw == -32768).sum()),
                             pixels=int(raw.size), valid_positive=int((raw > 0).sum()))
    # Independently recorded datasets have the same shape, dtype and exact bytes.
    if h5_path is not None:
        with h5py.File(h5_path, 'r') as stream:
            keys = []
            stream.visititems(lambda k, v: keys.append(k) if isinstance(v, h5py.Dataset) else None)
            assert set(keys) == set(a)
            for k in keys:
                actual = stream[k][()]
                assert actual.dtype == a[k].dtype and actual.shape == a[k].shape
                assert actual.tobytes() == a[k].tobytes(), k
    return dict(status='PASS', actions=n, observations=n+1, cameras=cameras,
                raw_action_to_step_to_controller_exact=True, pre_action_state_exact=True,
                independent_h5_npz_exact=h5_path is not None, depth=depth,
                terminal_observation_labelled=False, gt_task_student_inputs=False)


def snapshot(run, preflight):
    shutil.copytree(preflight, run/'before_snapshot')
    source = run/'source_snapshot'; source.mkdir()
    paths = [*Path('recurrent').rglob('*.py'), *Path('recurrent/configs').glob('*.yaml'),
             *Path('graph/integration_baseline').glob('*.py'), *Path('envs/maniskill').glob('*.py'),
             Path('graph/vision_input.py'), Path('visual/runtime/vision_packet.py'),
             Path('visual/runtime/tracking.py'), Path('visual/runtime/initialization.py')]
    hashes = {}
    for p in paths:
        target = source/p; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
        hashes[str(p)] = sha256(p)
    refs = ['docs/reports/recurrent_main_20261023/README.md',
            'docs/reports/recurrent_main_20261023/episodes.csv',
            'docs/reports/recurrent_main_20261023/summary.csv',
            'docs/reports/recurrent_main_20261023/provenance.json',
            'recurrent/evaluate.py', 'recurrent/initial_states.py', 'recurrent/collect.py',
            'recurrent/expert.py', 'recurrent/common.py', 'recurrent/configs/experiment.yaml']
    for name in refs:
        data = subprocess.check_output(['git', 'show', 'origin/recurrent:'+name])
        p = source/'origin_recurrent'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    write_json(run/'source_hashes.json', hashes)
    (run/'implementation.diff').write_bytes(subprocess.check_output(['git', 'diff', '--no-index', '--', '/dev/null', __file__]))


def run(args):
    assert 1 <= len(args.seeds) <= 5 and len(set(args.seeds)) == len(args.seeds)
    assert args.horizon == 250, 'this pilot fixes horizon250 explicitly'
    torch.set_num_threads(1)
    config = dict(task='Task06-B', seeds=args.seeds, horizon=args.horizon, noise=0, perturb_frac=0,
                  expert='recurrent.expert.ScriptedExpert', expert_kwargs={},
                  condition='normal', control_mode='pd_ee_delta_pose', obs_mode='rgb+depth',
                  sim_backend='physx_cpu', device='cpu', threads=1,
                  data_schema_version='direct_scripted_expert_v1', max_episodes=5)
    logger = ExperimentLogger(team='integration', model='recurrent_scripted_expert_collection_pilot',
                              seed=0, config=config, job_type='evaluate')
    run_dir = logger.directory
    print('RUN_DIR='+str(run_dir), flush=True)
    # git diff --no-index exits 1 when the new file differs, intentionally.
    try:
        snapshot(run_dir, args.preflight)
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 1: raise
        (run_dir/'implementation.diff').write_bytes(exc.output)
    manifest = dict(run_id=logger.run_id, schema_version=config['data_schema_version'], config=config,
                    recurrent_ref=subprocess.check_output(['git','rev-parse','origin/recurrent'], text=True).strip(),
                    expert_source_sha256=sha256('recurrent/expert.py'), episodes=[], failures=[],
                    split_status='PILOT_ONLY_NO_FINAL_SPLIT', source='live direct Expert, no trajectory restore')
    (run_dir/'episodes').mkdir()
    env = None
    try:
        from mani_skill.utils.registration import TimeLimitWrapper
        env = make_peg_insertion_env(obs_mode='rgb+depth', control_mode='pd_ee_delta_pose', device='cpu')
        env = TimeLimitWrapper(env.unwrapped, args.horizon)
        task = env.unwrapped
        obs, _ = env.reset(seed=args.seeds[0])
        schema = state_schema(robot_observation(env)[0])
        previous = json.loads(Path('experiments/c184b85dadf14db5953f11cdcacb6674/robot_state_schema.json').read_text())
        assert schema == previous and schema['dimension'] == 76
        write_json(run_dir/'robot_state_schema.json', schema)
        controller = task.agent.controller
        action_schema = dict(id='pd_ee_delta_pose_regression7', dimension=7, dtype='float32',
            fields=['dx','dy','dz','rotation_X','rotation_Y','rotation_Z','gripper'],
            target='exact native expert output = env.step input = CombinedController input', adapter=None,
            translation='normalized root translation: clip [-1,1], scale to +/-0.1 m',
            rotation='native normalized XYZ components; clip vector norm to1 then multiply rot_lower=-0.1; XYZ Euler delta, left composition in root frame',
            expert_rotation='servo computes world relative quaternion axis-angle, negates/divides by0.1; intention is approximate, controller interprets XYZ Euler; unchanged algorithm',
            gripper='normalized mimic target: -1 -> -0.01m, +1 -> 0.04m, both fingers',
            nominal_bounds=[-1,1], out_of_space_policy='preserve native values; audit actual installed controller preprocessing; never clip saved targets',
            controller_config={k:asdict(v.config) for k,v in controller.controllers.items()})
        write_json(run_dir/'expert_action_schema.json', action_schema)
        conditions = dict(versions=versions(), backend=str(task.backend), robot=task.agent.uid,
                          control_freq=int(task.control_freq), physics_freq=int(task.sim_freq),
                          sim_steps_per_control=int(task._sim_steps_per_control), horizon=env._max_episode_steps,
                          reset_source='reset(seed), original CPU fixture unavailable locally',
                          cuda='NOT_EXECUTED', mps='NOT_EXECUTED')
        write_json(run_dir/'executed_conditions.json', conditions); logger.record(environment=conditions)
        installed = run_dir/'installed_source'; installed.mkdir()
        from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
        from mani_skill.envs.sapien_env import BaseEnv
        from mani_skill.agents.controllers.base_controller import CombinedController
        for cls in (PDEEPoseController, BaseEnv, CombinedController):
            p = Path(inspect.getfile(cls)); shutil.copy2(p, installed/p.name)
        for seed in args.seeds:
            tick = perf_counter(); rows = {}; stages = []; checks = []
            path = run_dir/'episodes'/f'ep{seed}.npz'; h5_path = path.with_suffix('.h5')
            ep = dict(episode_id=seed, seed=seed, control_mode='pd_ee_delta_pose', action_schema=action_schema['id'],
                      expert_source_sha256=manifest['expert_source_sha256'], success=False, first_success_step=None,
                      failure_reason=None, exceptions=[], actions=0, observations=0)
            manifest['episodes'].append(ep)
            def record(stream, values):
                for k,v in values.items():
                    rows.setdefault(k, []).append(np.asarray(v).copy()); append_h5(stream, k, v)
            with h5py.File(h5_path, 'x') as stream:
                try:
                    obs, reset_info, audit = reset_for_recording(env, seed, schema)
                    expert = ScriptedExpert(env)
                    row, exact = capture(env, obs, schema)
                    row.update(episode_id=np.int64(seed), reset_seed=np.int64(seed))
                    record(stream, row); checks.append(exact)
                    for t in range(args.horizon):
                        phase = expert.phase; command = expert.act(); tensor = to_action_tensor(command.copy(), env)
                        assert command.shape == (7,) and command.dtype == np.float32 and np.isfinite(command).all()
                        audit.clear()
                        obs, reward, terminated, truncated, info = env.step(tensor)
                        success = bool(info['success'].item())
                        ep.update(actions_applied=t+1, observations_saved=len(rows['robot_state']),
                                  actions_saved=len(rows.get('actions', [])))
                        values = dict(actions=command.copy(), env_step_input=array(tensor)[0],
                                      terminated=np.bool_(terminated.item()), truncated=np.bool_(truncated.item()),
                                      success=np.bool_(success), **audit)
                        np.testing.assert_array_equal(command, values['controller_input'])
                        np.testing.assert_array_equal(rows['robot_state'][-1], values['controller_pre_state'])
                        record(stream, values)
                        stages.append(dict(action_t=t, before=phase, after=expert.phase, counter=expert.counter))
                        row, exact = capture(env, obs, schema)
                        row.update(episode_id=np.int64(seed), reset_seed=np.int64(seed))
                        record(stream, row); checks.append(exact)
                        ep.update(actions=t+1, observations=t+2, actions_saved=t+1, observations_saved=t+2,
                                  terminated=bool(terminated.item()), truncated=bool(truncated.item()))
                        if success:
                            ep.update(success=True, first_success_step=t+1); break
                        if ep['terminated'] or ep['truncated']: break
                    if not ep['success']: ep['failure_reason'] = 'horizon' if ep.get('truncated') else 'termination_without_success'
                except Exception as exc:
                    ep['exceptions'].append(dict(type=type(exc).__name__, reason=str(exc)))
                    ep['failure_reason'] = 'collection_exception'
                    manifest['failures'].append(dict(seed=seed, **ep['exceptions'][-1]))
            arrays = {k:np.stack(v) for k,v in rows.items()}
            save_npz(path, **arrays)
            write_json(path.with_suffix('.expert_diagnostics.json'), dict(student_input=False, phases=stages))
            if not ep['exceptions']:
                validation = validate_episode(arrays, schema, seed, h5_path)
                assert all(checks)
                validation['returned_observation_vs_independent_live_camera_exact'] = True
                write_json(path.with_suffix('.alignment.json'), validation)
                ep['alignment'] = validation
            ep.update(path=str(path.relative_to(run_dir)), sha256=sha256(path), h5_sha256=sha256(h5_path),
                      elapsed_seconds=perf_counter()-tick, compressed_npz_bytes=path.stat().st_size,
                      uncompressed_array_bytes=sum(x.nbytes for x in arrays.values()),
                      action_min=arrays['actions'].min(0).tolist() if 'actions' in arrays else None,
                      action_max=arrays['actions'].max(0).tolist() if 'actions' in arrays else None,
                      nominal_box_exceeding_rows=int((np.abs(arrays['actions'])>1).any(1).sum()) if 'actions' in arrays else None)
            write_json(run_dir/'pilot_manifest.json', manifest)
            logger.log(dict(pilot_seed=seed, success=int(ep['success']), actions=ep['actions']), step=seed)
            print(json.dumps({k:ep[k] for k in ('seed','success','first_success_step','actions','observations','failure_reason')}), flush=True)
        manifest.update(status='COLLECTED', successes=sum(e['success'] for e in manifest['episodes']),
                        success_within100=sum(e['success'] and e['first_success_step']<=100 for e in manifest['episodes']))
        write_json(run_dir/'pilot_manifest.json', manifest)
        for p in run_dir.rglob('*'):
            if p.is_file() and p.suffix in {'.npz','.h5'}: logger.link_output(p, kind='trajectory')
        for name in ('pilot_manifest.json','expert_action_schema.json','robot_state_schema.json','executed_conditions.json'):
            logger.link_output(run_dir/name, kind='metrics')
    except Exception as exc:
        write_json(run_dir/'execution_failure.json', dict(type=type(exc).__name__, reason=str(exc)))
        logger.finish(failed=True); raise
    finally:
        if env is not None: env.close()
        logger.finish()
    return run_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(2000, 2005)))
    parser.add_argument('--horizon', type=int, default=250)
    parser.add_argument('--preflight', type=Path, required=True)
    run(parser.parse_args())
