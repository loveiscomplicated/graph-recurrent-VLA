"""Freeze Task08-B single-episode plan and audit actual Episode 30003 inputs."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from graph.integration_baseline.bc08_codec import ActionCodec, controller_commands
from graph.integration_baseline.bc08_eval import environment
from graph.integration_baseline.bc08_train import (load_plan, normalizer_hash, restricted_load,
                                                     tensor_hash)
from graph.integration_baseline.direct_dataset import DirectDataset, load_episode
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.direct_train import batch, load_normalizer, seed_all
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.robot import robot_observation, flatten_state, state_schema
from graph.integration_baseline.rollout_stop_clock import reset_diagnostics, numpy
from utils.experiment_logger import ExperimentLogger


MILESTONES = [2000, 5000, 10000, 15000, 20000]
EPISODE = 30003


def prepare(source_plan, output_root):
    source_plan = Path(source_plan).resolve()
    source = load_plan(source_plan)
    ds = DirectDataset(source['dataset_directory'])
    if EPISODE not in ds.splits['train']:
        raise ValueError('Episode 30003 is not in frozen train split')
    original = ds.episodes[EPISODE]
    if not (original['success'] and original['actions'] == 137 and original['observations'] == 138):
        raise ValueError('unexpected expert episode identity')
    plan = copy.deepcopy(source)
    plan['overfit_ids'] = [EPISODE]
    plan['overfit_selection'] = [e for e in source['overfit_selection'] if e['episode_id'] == EPISODE]
    plan['overfit_selection_rule'] = 'Task08-B user-fixed frozen train Expert Episode 30003'
    plan['overfit_updates'] = MILESTONES[-1]
    plan['milestones'] = MILESTONES
    plan['codecs']['overfit']['A'] = copy.deepcopy(source['codecs']['full']['A'])
    plan['codecs']['overfit']['C'] = copy.deepcopy(source['codecs']['full']['C'])
    plan['probe_batch'] = dict(episode_ids=[EPISODE], start=0)
    plan['experiments_root'] = str(Path(output_root).resolve())
    plan['selection'] = ('Train exactly 20000 updates; among fixed milestones passing all four Task08 '
                         'offline thresholds, choose lowest decoded native Action7 MSE; otherwise '
                         'the final 20000-update checkpoint. Closed-loop results never select a checkpoint.')
    plan['task08b_scheduler_note'] = ('Same Task08 CosineAnnealingLR and eta_min=0; '
                                      'T_max extended from 2000 to the fixed 20000-update budget.')
    plan['task08b_initialization'] = 'same frozen Task08 initial.pt independently loaded for A and C; fresh AdamW state'
    root = Path(__file__).resolve().parents[2]
    for name in ('bc08b_prepare.py', 'bc08b_train.py', 'bc08b_cached_train.py'):
        path = root/'graph/integration_baseline'/name
        plan['source_hashes'][str(path)] = sha(path)
    raw_path = ds.run/original['path']
    plan['input_hashes'][str(raw_path)] = original['sha256']
    for t in range(original['actions']):
        row = ds.packet_rows[(EPISODE, t)]
        plan['input_hashes'][str(ds.output/row['path'])] = row['sha256']

    with ExperimentLogger(team='integration', model='task08b_single_episode_ac', seed=42,
                          config=dict(source_plan=str(source_plan), episode_id=EPISODE,
                                      milestones=MILESTONES, selection=plan['selection']),
                          root=output_root, job_type='prepare') as logger:
        run = logger.directory
        plan_path = run/'plan.json'
        atomic(plan_path, plan, immutable=True)
        # Immutable source/run files are protected; active Full jobs have mutable progress.
        protection = dict(source_plan={str(source_plan): sha(source_plan)},
                          input_hashes=plan['input_hashes'], source_hashes=plan['source_hashes'])
        for method, run_id in [('A', '2bc5701a80494a72910af0044219ce43'),
                               ('C', '9a8cec829aaa4c2aaccc10ed91770c30')]:
            small = Path(source['experiments_root'])/run_id
            for relative in ('completion.json', 'last_valid.json', 'action_codec.json'):
                path = small/relative
                if path.exists(): protection[f'small_{method}_{relative}'] = {str(path): sha(path)}
            if (small/'last_valid.json').exists():
                receipt = read(small/'last_valid.json')
                protection[f'small_{method}_checkpoint'] = {receipt['path']: sha(receipt['path'])}
        atomic(run/'protected_before.json', protection, immutable=True)
        logger.link_output(plan_path, kind='metrics')

        raw = load_episode(raw_path)
        for t in range(original['actions']):
            frame = ds.frame(EPISODE, t)
            np.testing.assert_array_equal(frame.action, raw['actions'][t])
        try:
            ds.frame(EPISODE, original['actions'])
        except ValueError:
            final_unlabelled = True
        else:
            raise ValueError('final observation accidentally labelled')
        np.testing.assert_array_equal(raw['actions'], raw['controller_input'])
        np.testing.assert_array_equal(raw['actions'], raw['env_step_input'])
        np.testing.assert_allclose(controller_commands(raw['actions'])[:, :6], raw['arm_scaled_input'], atol=2e-8, rtol=1e-6)
        np.testing.assert_allclose(controller_commands(raw['actions'])[:, 6:], raw['gripper_scaled_input'], atol=1e-8, rtol=0)
        if np.flatnonzero(raw['success']).tolist() != [136]:
            raise ValueError('expert success or action alignment changed')
        episode_audit = dict(status='PASS', episode_id=EPISODE, seed=EPISODE,
                             raw_sha256=original['sha256'], reset_group=original['reset_group'],
                             actions=137, observations=138, first_success_step=137,
                             final_observation_unlabelled=final_unlabelled,
                             graph_action_join='137 / 137 matched by DirectDataset.frame',
                             robot_schema=ds.schema, action_schema=read(ds.run/'expert_action_schema.json'),
                             packet_sequence_sha256=plan['overfit_selection'][0]['packet_sequence_sha256'],
                             student_input_policy='current RGB-D predicted mask stored Graph and Robot76 only; GT/phase/future action excluded')
        atomic(run/'episode_identity.json', episode_audit, immutable=True)

        # Independent numeric reset audit. Collection does not store GT peg/box arrays;
        # its reset hash is compared with the live GT aggregate instead.
        env, conditions = environment(plan)
        try:
            obs, _ = env.reset(seed=EPISODE)
            fields, _ = robot_observation(env)
            if state_schema(fields) != ds.schema:
                raise ValueError('live Robot76 schema mismatch')
            live_robot = flatten_state(fields, ds.schema)
            numeric = {}
            comparisons = [
                ('robot76', live_robot, raw['robot_state'][0]),
                ('qpos', live_robot[:9], raw['robot_state'][0, :9]),
                ('qvel', live_robot[9:18], raw['robot_state'][0, 9:18]),
                ('tcp_pose', fields['tcp_pose'], raw['tcp_pose'][0]),
            ]
            for camera in ('base_camera', 'hand_camera'):
                comparisons.extend([
                    (camera+'_rgb', numpy(obs['sensor_data'][camera]['rgb'])[0], raw[camera+'/rgb'][0]),
                    (camera+'_depth_raw_mm', numpy(obs['sensor_data'][camera]['depth'])[0, :, :, 0], raw[camera+'/depth_raw_mm'][0]),
                    (camera+'_intrinsic', numpy(obs['sensor_param'][camera]['intrinsic_cv'])[0], raw[camera+'/intrinsic'][0]),
                    (camera+'_extrinsic', numpy(obs['sensor_param'][camera]['extrinsic_cv'])[0], raw[camera+'/extrinsic_cv'][0]),
                ])
            for name, live, saved in comparisons:
                difference = np.asarray(live, np.float64)-np.asarray(saved, np.float64)
                numeric[name] = dict(exact=bool(np.array_equal(live, saved)),
                                     max_abs=float(np.max(np.abs(difference))),
                                     different_fraction=float(np.mean(difference != 0)))
            group = reset_diagnostics(env)
            physical_match = (group == original['reset_group'] and numeric['qpos']['exact'] and
                              numeric['qvel']['exact'] and numeric['robot76']['max_abs'] <= 1e-6 and
                              numeric['tcp_pose']['max_abs'] <= 1e-6)
            sensor_match = all(numeric[camera+'_'+kind]['max_abs'] <= (2 if kind == 'rgb' else 1) and
                               numeric[camera+'_'+kind]['different_fraction'] <= (.10 if kind == 'rgb' else .01)
                               for camera in ('base_camera', 'hand_camera') for kind in ('rgb', 'depth_raw_mm'))
            calibration_match = all(numeric[camera+'_intrinsic']['exact'] and
                                    numeric[camera+'_extrinsic']['max_abs'] <= 2e-6
                                    for camera in ('base_camera', 'hand_camera'))
            reset = dict(status='PASS_PHYSICAL_WITH_RENDER_QUANTIZATION' if physical_match and sensor_match and calibration_match else 'MISMATCH',
                         reset_group_live=group, reset_group_collection=original['reset_group'],
                         numeric=numeric, conditions=conditions,
                         acceptance_rule='exact GT reset SHA/qpos/qvel/camera intrinsic; Robot76/TCP max 1e-6; camera extrinsic max 2e-6; RGB max 2 integer units and <=10% differing values; depth max 1 mm unit and <=1% differing pixels',
                         peg_box_pose_identity='included in reset group SHA256, not individually stored in source HDF5')
            atomic(run/'reset_identity.json', reset, immutable=True)
        finally:
            env.close()

        # Check the actual Task08 A/C math and full CUDA gradient path before training.
        seed_all(42, plan['config']['threads'])
        init = restricted_load(plan['initial_path'])
        norm = load_normalizer(init['normalizer'])
        if tensor_hash(init['model']) != plan['initial_weight_identity'] or normalizer_hash(init['normalizer']) != plan['robot_normalizer_identity']:
            raise ValueError('frozen initial weights/Robot76 normalizer changed')
        if norm.fitted_episode_ids != sorted(ds.splits['train']):
            raise ValueError('Robot76 normalizer not fitted on Train96')
        device = plan['config']['device']
        g, s, y, valid = batch(ds, [EPISODE], 0, 32, device, plan['config'])
        if valid.shape != (1, 32) or not bool(valid.all()):
            raise ValueError('unexpected sequence padding')
        gradient = {}
        for method in ('A', 'C'):
            codec = ActionCodec(plan['codecs']['overfit'][method])
            model = IntegrationPolicy(ds.schema, **plan['config']['model']).to(device)
            model.load_state_dict(init['model'], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['config']['training']['learning_rate'],
                                         weight_decay=plan['config']['training']['weight_decay'],
                                         betas=tuple(plan['config']['training']['betas']),
                                         eps=plan['config']['training']['eps'])
            if optimizer.state:
                raise ValueError('optimizer not fresh')
            model.eval()
            with torch.no_grad():
                seq, h = model(g, s, norm)
                gh = None
                steps = []
                for t in range(32):
                    one_g, one_s, _, _ = batch(ds, [EPISODE], t, 1, device, plan['config'])
                    one, gh = model.step(one_g, one_s, norm, gh)
                    steps.append(one)
                torch.testing.assert_close(seq, torch.stack(steps, 1), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(h, gh, atol=1e-6, rtol=1e-5)
                decoded = codec.decode(codec.tag(seq))
                if method == 'A':
                    torch.testing.assert_close(codec.loss(seq, y, valid), F.mse_loss(seq, y), atol=0, rtol=0)
                    torch.testing.assert_close(decoded, seq, atol=0, rtol=0)
                else:
                    mean, std = codec._stats(y)
                    expected_loss = (6*F.mse_loss(seq[..., :6], (y[..., :6]-mean)/std) +
                                     F.binary_cross_entropy_with_logits(seq[..., 6], (y[..., 6]>0).float()))/7
                    torch.testing.assert_close(codec.loss(seq, y, valid), expected_loss, atol=1e-7, rtol=1e-6)
                    torch.testing.assert_close(decoded[..., :6], seq[..., :6]*std+mean, atol=0, rtol=0)
                    assert decoded[..., 6].unique().tolist() == [-1.0] or set(decoded[..., 6].unique().tolist()) <= {-1.0, 1.0}
                    threshold = codec.decode(codec.tag(torch.zeros_like(seq)))
                    assert bool((threshold[..., 6] == -1).all())
            model.train()
            output, _ = model(g, s, norm)
            loss = codec.loss(output, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward()
            groups = {'gnn': 0., 'gru': 0., 'head': 0.}
            for name, param in model.named_parameters():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise FloatingPointError('missing/nonfinite gradient: '+name)
                key = 'gnn' if name.startswith('graph_encoder.') else 'gru' if '.rnn.' in name else 'head'
                groups[key] += float(param.grad.abs().sum())
            if any(v <= 0 for v in groups.values()):
                raise ValueError('missing gradient through GNN/GRU/action head')
            gradient[method] = dict(loss=float(loss), gradient_abs_sum=groups,
                                    codec_identity=codec.identity, initial_weight_sha256=tensor_hash(model.state_dict()),
                                    fresh_optimizer=True, sequence_step_equivalence=True,
                                    valid_targets=32, final_observation_excluded=True)
            del model, optimizer
        if gradient['A']['initial_weight_sha256'] != gradient['C']['initial_weight_sha256']:
            raise ValueError('A/C initial weights differ')
        if gradient['A']['initial_weight_sha256'] != plan['initial_weight_identity']:
            raise ValueError('loaded model differs from frozen initial weights')
        atomic(run/'contract_tests.json', dict(status='PASS', methods=gradient,
             hidden='zero on episode start; sequence/step equality with carried hidden',
             padding='32 valid targets; final observation is never labelled'), immutable=True)
        logger.link_output(run/'episode_identity.json', kind='metrics')
        logger.link_output(run/'reset_identity.json', kind='metrics')
        logger.link_output(run/'contract_tests.json', kind='metrics')
        atomic(run/'prepare_complete.json', dict(status='PASS', plan=str(plan_path),
             reset_identity=reset['status'], run_id=logger.run_id), immutable=True)
        return dict(run_id=logger.run_id, plan=str(plan_path), reset_identity=reset['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_plan, args.output_root)), flush=True)
