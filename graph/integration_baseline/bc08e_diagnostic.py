"""Read-only Task08-E best/final offline fit and one live seed-30003 rollout per method."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from graph.integration_baseline.bc08_codec import DecodedPolicy, controller_commands, native_metrics
from graph.integration_baseline.bc08_eval import environment, validate_checkpoint_plan
from graph.integration_baseline.bc08_train import evaluate, load_checkpoint, load_normalizer, load_plan
from graph.integration_baseline.direct_dataset import DirectDataset
from graph.integration_baseline.direct_train import batch
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.rollout_checkpoint import vision_runtime_identity
from graph.integration_baseline.rollout_stop_clock import run_episode, reset_diagnostics, numpy
from utils.experiment_logger import ExperimentLogger

SEED = 30003
EXPECTED_H5 = 'ba578d3e17e804c48c5490dc9970406a2b560d9ea214519094965fa815a09122'
STAGES = ('reach', 'descend', 'close', 'lift', 'align', 'insert')


def dump(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic(path, value, immutable=True)


def model_for(plan, plan_path, receipt, method):
    cp, codec = load_checkpoint(receipt)
    validate_checkpoint_plan(cp, codec, read(receipt), plan, plan_path, 'full', method)
    model = IntegrationPolicy(cp['normalizer']['schema'], **plan['config']['model']).to(plan['config']['device'])
    model.load_state_dict(cp['model'], strict=True)
    model.eval()
    return model, load_normalizer(cp['normalizer']), codec, cp


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {k: float(v) for k, v in zip(('mean', 'p50', 'p95', 'p99', 'max'),
        (x.mean(), *np.quantile(x, [.5, .95, .99]), x.max()))}


def extra_metrics(pred, target, stages):
    e = pred.astype(np.float64) - target.astype(np.float64)
    cp, cy = controller_commands(pred), controller_commands(target)
    trans = np.linalg.norm(cp[:, :3] - cy[:, :3], axis=1) * 1000
    rot = (Rotation.from_euler('XYZ', cp[:, 3:6]).inv() *
           Rotation.from_euler('XYZ', cy[:, 3:6])).magnitude()
    grip = np.abs(cp[:, 6] - cy[:, 6]) * 1000
    return dict(native=native_metrics(pred, target), translation3_mae=float(np.abs(e[:, :3]).mean()),
        rotation3_mae=float(np.abs(e[:, 3:6]).mean()), per_dimension_mae=np.abs(e).mean(0).tolist(),
        action7_frame_l2=summary(np.linalg.norm(e, axis=1)),
        controller_translation_mm=summary(trans), controller_rotation_rad=summary(rot),
        controller_rotation_deg=summary(np.rad2deg(rot)), controller_gripper_target_mm=summary(grip),
        clipping=dict(translation=int(np.any(np.abs(pred[:, :3]) > 1, axis=1).sum()),
                      rotation=int((np.linalg.norm(pred[:, 3:6], axis=1) > 1).sum()),
                      gripper=int((np.abs(pred[:, 6]) > 1).sum())),
        first_close=next((int(t) for t in np.flatnonzero(pred[:, 6] < 0)), None),
        expert_first_close=next((int(t) for t in np.flatnonzero(target[:, 6] < 0)), None),
        transitions=(np.flatnonzero(np.diff(pred[:, 6] > 0)) + 1).tolist(),
        stage={s: dict(count=int(np.sum(stages == s)),
                       action7_mse=float(np.mean(e[stages == s] ** 2)),
                       controller_translation_mm=summary(trans[stages == s]),
                       controller_rotation_rad=summary(rot[stages == s]),
                       gripper_sign_accuracy=float(np.mean((pred[stages == s, 6] > 0) ==
                                                            (target[stages == s, 6] > 0))))
               for s in STAGES if np.any(stages == s)},
        largest_error_frames=np.argsort(np.linalg.norm(e, axis=1))[-10:][::-1].tolist())


def offline(plan, plan_path, run, dataset, stages, receipts):
    results = {}
    for method in 'ABC':
        results[method] = {}
        for label in ('best', 'final'):
            receipt = receipts[method][label]
            model, norm, codec, cp = model_for(plan, plan_path, receipt, method)
            if label == 'best':
                graph, robot, _, _ = batch(dataset, [SEED], 0, 16,
                                            plan['config']['device'], plan['config'])
                with torch.inference_mode():
                    sequence, sequence_hidden = model(graph, robot, norm)
                    hidden = None
                    steps = []
                    for t in range(16):
                        g, s, _, _ = batch(dataset, [SEED], t, 1,
                                            plan['config']['device'], plan['config'])
                        value, hidden = model.step(g, s, norm, hidden)
                        steps.append(value)
                    torch.testing.assert_close(sequence[:, :16], torch.stack(steps, 1), atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(sequence_hidden, hidden, atol=1e-6, rtol=1e-5)
            out = run / 'offline' / method / label
            metrics = evaluate(dataset, model, norm, codec, [SEED], plan['config'], output=out)
            with np.load(out / f'ep{SEED}.npz', allow_pickle=False) as f:
                pred, target = f['native_prediction'].copy(), f['native_target'].copy()
            if pred.shape != (137, 7) or target.shape != (137, 7):
                raise ValueError('full expert sequence not evaluated')
            np.testing.assert_array_equal(target, dataset.raw(SEED)['actions'])
            details = extra_metrics(pred, target, stages)
            details.update(epoch=cp['progress']['epoch'], global_step=cp['progress']['step'],
                           checkpoint_sha256=read(receipt)['sha256'],
                           normalizer_sha256=cp['identity']['normalizer_sha256'],
                           codec_sha256=codec.identity, optimization_loss=metrics['optimization_loss'])
            dump(out / 'metrics.json', details)
            with (out / 'timeline.csv').open('x', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['t', 'stage', *[f'expert_{i}' for i in range(7)],
                                 *[f'pred_{i}' for i in range(7)], 'action7_error_l2',
                                 'controller_translation_mm'])
                c1, c2 = controller_commands(pred), controller_commands(target)
                for t in range(137):
                    writer.writerow([t, stages[t], *target[t], *pred[t],
                                     np.linalg.norm(pred[t] - target[t]),
                                     1000 * np.linalg.norm(c1[t, :3] - c2[t, :3])])
            results[method][label] = details
            del model
            torch.cuda.empty_cache()
    dump(run / 'best_final_comparison.json', results)
    return results


def live(plan, plan_path, run, method, receipt, reset_group):
    model, norm, codec, _ = model_for(plan, plan_path, receipt, method)
    if vision_runtime_identity() != plan['reference_provenance']['vision_runtime']:
        raise ValueError('Vision runtime mismatch')
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from envs.maniskill import get_gt_state
    policy = DecodedPolicy(model, codec)
    vision = OnlineVisionAdapter(GDino(device=plan['config']['device']),
                                 SAM2Tracker(device=plan['config']['device']))
    env, conditions = environment(plan)
    diag = []
    original_step = env.step
    baseline = [None]
    def step(action):
        if baseline[0] is None:
            baseline[0] = float(get_gt_state(env).peg_pose[0, 2])
        pre = get_gt_state(env)
        result = original_step(action)
        post = get_gt_state(env)
        task = env.unwrapped
        diag.append(dict(timestep=len(diag), grasp=bool(task.agent.is_grasping(task.peg).item()),
            peg_lift_m=float(post.peg_pose[0, 2]) - baseline[0],
            tcp_pose=numpy(post.ee_pose)[0].tolist(), peg_pose=numpy(post.peg_pose)[0].tolist(),
            box_pose=numpy(post.box_pose)[0].tolist(),
            pre_tcp_pose=numpy(pre.ee_pose)[0].tolist(), pre_peg_pose=numpy(pre.peg_pose)[0].tolist(),
            finger_forces=np.stack([numpy(task.scene.get_pairwise_contact_forces(link, task.peg))[0]
                for link in (task.agent.finger1_link, task.agent.finger2_link)]).tolist()))
        return result
    env.step = step
    folder = run / 'rollout' / method
    try:
        ep = run_episode(env, policy, norm, vision,
            dict(seed=SEED, group='train_seed', reset_group=reset_group, status='READY'),
            plan['config'], folder, smoke=False)
    finally:
        env.close()
    if ep['status'] != 'COMPLETE' or len(diag) != ep['episode_length']:
        raise ValueError('rollout incomplete or invalid: ' + str(ep.get('error')))
    np.savez_compressed(folder / 'encoded_action_output.npz',
                        raw_output=np.asarray(policy.raw_outputs), codec_identity=codec.identity)
    dump(folder / 'grasp_contact_lift.json', diag)
    with np.load(folder / 'observations.npz', allow_pickle=False) as f:
        rgb = f['rgb'].copy()
    writer = cv2.VideoWriter(str(folder / 'after_action_rgb.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             20, (rgb.shape[2], rgb.shape[1]))
    if not writer.isOpened():
        raise RuntimeError('after-action MP4 encoder unavailable')
    for frame in rgb[1:]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    actions = np.load(folder / 'actions.npz', allow_pickle=False)['actions']
    grasp = np.array([d['grasp'] for d in diag], bool)
    lift = np.array([d['peg_lift_m'] for d in diag])
    bilateral = np.array([all(np.linalg.norm(f) >= .5 for f in d['finger_forces']) for d in diag])
    result = dict(method=method, checkpoint_sha256=read(receipt)['sha256'],
        reset_group=ep['reset_group'], reset_identity='PASS_PHYSICAL_WITH_RENDER_QUANTIZATION',
        success=ep['success'], episode_length=ep['episode_length'],
        first_success_step=ep['first_success_step'], first_close=next((int(t) for t in np.flatnonzero(actions[:, 6] < 0)), None),
        first_grasp=next((int(t) for t in np.flatnonzero(grasp)), None),
        bilateral_contact=bool(bilateral.any()), grasp=bool(grasp.any()), lift_4cm=bool((lift >= .04).any()),
        first_grasp_loss=next((int(t) for t in range(1, len(grasp)) if grasp[t-1] and not grasp[t]), None),
        max_peg_lift_m=float(lift.max()), conditions=conditions)
    dump(folder / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('offline', 'rollout', 'report'), required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--method', choices=tuple('ABC'))
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if not set(plan['splits']) == {'train', 'validation', 'test'} or [len(plan['splits'][s]) for s in ('train', 'validation', 'test')] != [96, 12, 12]:
        raise ValueError('frozen split changed')
    if SEED not in plan['splits']['train'] or SEED in plan['splits']['validation'] + plan['splits']['test']:
        raise ValueError('train seed membership mismatch')
    dataset = DirectDataset(plan['dataset_directory'])
    if dataset.episodes[SEED]['sha256'] != EXPECTED_H5 or dataset.episodes[SEED]['actions'] != 137 or not dataset.episodes[SEED]['success']:
        raise ValueError('expert HDF5 identity mismatch')
    root = args.plan.parent
    state = read(root / 'status.json')
    if state['status'] == 'RUNNING' or any(k.startswith('final_') for k in state['stages']):
        raise ValueError('original Full A/B/C still active or final state changed')
    receipts = {m: {label: Path(state['stages']['full_' + m]['directory']) / name
                    for label, name in (('best', 'best.json'), ('final', 'last_valid.json'))}
                for m in 'ABC'}
    if any(state['stages']['full_' + m]['status'] != 'COMPLETE' for m in 'ABC'):
        raise ValueError('Full training incomplete')
    for m in 'ABC':
        if read(receipts[m]['final'])['epoch'] != 50:
            raise ValueError('final epoch missing')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.phase == 'offline':
        if args.run is not None:
            raise ValueError('offline creates its own ExperimentLogger run')
        episode = dataset.episodes[SEED]
        stages = np.asarray([p['before'] for p in episode['phases']])
        if len(stages) != 137 or set(stages) - set(STAGES):
            raise ValueError('expert stage sidecar mismatch')
        with ExperimentLogger(team='integration', model='task08e_full_trainseed_diagnostic',
            seed=SEED, config=dict(task='08-E', plan=str(args.plan), checkpoint_selection='validation best before train-seed evaluation'),
            root=plan['experiments_root'], job_type='evaluate') as logger:
            run = logger.directory
            shutil.copy2('/workspace/task08e/pre_stop_audit.json', run / 'pre_stop_audit.json')
            shutil.copy2('/workspace/task08e/post_stop_audit.json', run / 'post_stop_audit.json')
            dump(run / 'original_references.json', dict(plan=str(args.plan), plan_sha256=sha(args.plan),
                dataset_directory=plan['dataset_directory'], hdf5_sha256=EXPECTED_H5,
                split_membership='train', original_status=state['status'],
                checkpoints={m: {label: read(p) for label, p in receipts[m].items()} for m in 'ABC'},
                uncompleted_final={m: 40 for m in 'ABC'},
                interruption='INTERRUPTED_BY_USER_REQUEST'))
            print('RUN_ID', logger.run_id, flush=True)
            result = offline(plan, args.plan, run, dataset, stages, receipts)
            for m in 'ABC':
                logger.log({'offline_best_action7_mse_' + m: result[m]['best']['native']['native_action7_mse']})
    elif args.phase == 'rollout':
        if args.run is None or args.method is None:
            raise ValueError('rollout needs run and method')
        run = args.run.resolve()
        ref = read(run / 'original_references.json')
        if ref['plan_sha256'] != sha(args.plan) or not (run / 'best_final_comparison.json').is_file():
            raise ValueError('offline prerequisite absent')
        expected = dataset.episodes[SEED]['reset_group']
        result = live(plan, args.plan, run, args.method, receipts[args.method]['best'], expected)
        print(json.dumps(result, default=str), flush=True)
    else:
        raise NotImplementedError('report generated after all three rollouts')


if __name__ == '__main__':
    main()
