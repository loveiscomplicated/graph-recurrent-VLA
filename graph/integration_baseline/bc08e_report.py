"""Render Task08-E measured comparison and verify saved diagnostics."""
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from graph.integration_baseline.direct_runtime import atomic, read, sha
from utils.experiment_artifacts import register_existing_output

STAGES = ('reach', 'descend', 'close', 'lift', 'align', 'insert')
SINGLE = Path('/workspace/task08d/experiments/090a4bd46e75455db893543e35c2da2b/comparison.json')


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=145)
    plt.close()


def first_above(x, threshold):
    rows = np.flatnonzero(np.asarray(x) > threshold)
    return int(rows[0]) if len(rows) else None


def make_report(run, plan_path):
    run, plan_path = Path(run).resolve(), Path(plan_path).resolve()
    plan = read(plan_path)
    refs = read(run / 'original_references.json')
    full = read(run / 'best_final_comparison.json')
    single = read(SINGLE)
    summary = {m: read(run / 'rollout' / m / 'summary.json') for m in 'ABC'}
    fig = run / 'figures'
    fig.mkdir(exist_ok=False)
    dataset = Path(read(Path(plan['dataset_directory']) / 'data_source.json')['directory'])
    ep = next(e for e in plan['reference_provenance']['dataset']['episodes'] if e['seed'] == 30003)
    with h5py.File(dataset / ep['path'], 'r') as f:
        expert_rgb = f['base_camera/rgb'][()]
        expert_tcp = f['tcp_pose'][()]
        expert_robot = f['robot_state'][0]
        expert_intrinsic = f['base_camera/intrinsic'][0]
        expert_cam2world = f['base_camera/cam2world'][0]
    if sha(dataset / ep['path']) != ep['sha256']:
        raise ValueError('expert HDF5 changed')
    replay_path = Path('/workspace/task08b/experiments/60468642a39545318ebf0256727d7a08/expert_replay_gt.json')
    replay = read(replay_path)
    if len(replay['trace']) != 137 or replay.get('status') != 'STATE_DIVERGENCE':
        raise ValueError('expert replay GT sidecar unavailable')
    expert_rel = np.asarray([np.asarray(replay['trace'][0]['pre']['ee_pose'][:3]) -
                             np.asarray(replay['trace'][0]['pre']['peg_pose'][:3])] +
                            [np.asarray(row['post']['ee_pose'][:3]) - np.asarray(row['post']['peg_pose'][:3])
                             for row in replay['trace']])
    colors = dict(A='#1f77b4', B='#2ca02c', C='#d62728')
    paths = []

    plt.figure(figsize=(9, 4))
    x = np.arange(3)
    plt.bar(x - .18, [single['offline'][m]['native_action7_mse'] for m in 'ABC'], .36, label='Single 20k')
    plt.bar(x + .18, [full[m]['best']['native']['native_action7_mse'] for m in 'ABC'], .36, label='Full best')
    plt.yscale('log'); plt.xticks(x, list('ABC')); plt.ylabel('Native Action7 MSE'); plt.legend()
    savefig(fig / 'single_vs_full_action7.png'); paths.append('figures/single_vs_full_action7.png')

    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, unit in zip(axes, ('controller_translation_mm', 'controller_rotation_rad', 'controller_gripper_target_mm'), ('mm', 'rad', 'mm')):
        vals = [full[m]['best'][key]['mean'] for m in 'ABC']
        ax.bar(list('ABC'), vals, color=[colors[m] for m in 'ABC']); ax.set_title(key.replace('_', ' ')); ax.set_ylabel(unit)
    savefig(fig / 'controller_errors.png'); paths.append('figures/controller_errors.png')

    timeline = {m: np.load(run / 'offline' / m / 'best' / 'ep30003.npz', allow_pickle=False)['native_prediction'] for m in 'ABC'}
    target = np.load(run / 'offline' / 'A' / 'best' / 'ep30003.npz', allow_pickle=False)['native_target']
    fig3, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for d, ax in zip((0, 3, 6), axes):
        ax.plot(target[:, d], color='black', lw=2, label='Expert')
        for m in 'ABC': ax.plot(timeline[m][:, d], color=colors[m], alpha=.8, label='Full ' + m)
        ax.set_ylabel(('Translation x', 'Rotation x', 'Gripper')[d // 3 if d < 6 else 2])
    axes[0].legend(ncol=4); axes[-1].set_xlabel('Expert action frame')
    savefig(fig / 'action_timeline.png'); paths.append('figures/action_timeline.png')

    heat = np.array([[full[m]['best']['stage'][s]['action7_mse'] for s in STAGES] for m in 'ABC'])
    plt.figure(figsize=(10, 3.5)); plt.imshow(np.log10(heat), cmap='magma', aspect='auto')
    plt.colorbar(label='log10 Action7 MSE'); plt.yticks(range(3), list('ABC')); plt.xticks(range(6), STAGES)
    savefig(fig / 'stage_heatmap.png'); paths.append('figures/stage_heatmap.png')

    traces, diags, actions, rgbs = {}, {}, {}, {}
    reset_audit = {}
    for m in 'ABC':
        folder = run / 'rollout' / m
        traces[m] = read(folder / 'trace.json')
        diags[m] = read(folder / 'grasp_contact_lift.json')
        with np.load(folder / 'actions.npz', allow_pickle=False) as f: actions[m] = f['actions'].copy()
        with np.load(folder / 'observations.npz', allow_pickle=False) as f:
            rgbs[m] = f['rgb'].copy()
            robot_error = np.abs(f['robot_state'][0] - expert_robot)
            tcp_error = np.abs(f['tcp_pose'][0] - expert_tcp[0])
            intrinsic_error = np.abs(f['intrinsic'][0] - expert_intrinsic)
            cam_error = np.abs(f['cam2world'][0] - expert_cam2world)
            rgb_error = np.abs(f['rgb'][0].astype(np.int16) - expert_rgb[0].astype(np.int16))
            reset_audit[m] = dict(reset_hash_match=summary[m]['reset_group'] == ep['reset_group'],
                robot76_max_abs=float(robot_error.max()), tcp_pose_max_abs=float(tcp_error.max()),
                intrinsic_max_abs=float(intrinsic_error.max()), cam2world_max_abs=float(cam_error.max()),
                rgb_max_abs=int(rgb_error.max()), rgb_different_pixels=int(np.count_nonzero(rgb_error)),
                verdict='PASS_PHYSICAL_WITH_RENDER_QUANTIZATION' if
                    summary[m]['reset_group'] == ep['reset_group'] and robot_error.max() <= 1e-5 and
                    tcp_error.max() <= 1e-5 and intrinsic_error.max() <= 1e-5 and cam_error.max() <= 1e-5
                    else 'BLOCKED_RESET_IDENTITY')
        if len(actions[m]) != len(diags[m]) or len(rgbs[m]) != len(actions[m]) + 1:
            raise ValueError('closed-loop trace mismatch')
    if any(reset_audit[m]['verdict'] != 'PASS_PHYSICAL_WITH_RENDER_QUANTIZATION' for m in 'ABC'):
        raise ValueError('initial physical reset identity failed')
    atomic(run / 'reset_identity_audit.json', reset_audit, immutable=True)

    fig4 = plt.figure(figsize=(10, 7)); ax = fig4.add_subplot(111, projection='3d')
    ax.plot(*expert_tcp[:, :3].T, color='black', label='Expert')
    for m in 'ABC':
        xyz = np.array([d['tcp_pose'][:3] for d in diags[m]])
        ax.plot(*xyz.T, color=colors[m], label='Full ' + m)
    ax.set_xlabel('X m'); ax.set_ylabel('Y m'); ax.set_zlabel('Z m'); ax.legend()
    savefig(fig / 'tcp_3d.png'); paths.append('figures/tcp_3d.png')

    fig5, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for m in 'ABC':
        d = diags[m]; t = np.arange(len(d)); a = actions[m]
        axes[0].step(t, a[:, 6], where='post', label=m, color=colors[m])
        axes[1].step(t, [int(v['grasp']) for v in d], where='post', label=m, color=colors[m])
        axes[2].plot(t, [v['peg_lift_m'] * 1000 for v in d], label=m, color=colors[m])
    for ax, label in zip(axes, ('Gripper native', 'is_grasping', 'Peg lift mm')): ax.set_ylabel(label)
    axes[2].axhline(40, color='gray', ls='--'); axes[0].legend(ncol=3); axes[-1].set_xlabel('Autonomous action frame')
    savefig(fig / 'grasp_release_lift.png'); paths.append('figures/grasp_release_lift.png')

    plt.figure(figsize=(12, 4))
    plt.plot(np.linalg.norm(expert_rel, axis=1), color='black', label='Expert TCP–Peg')
    divergence = {}
    for m in 'ABC':
        rel = np.array([np.array(d['tcp_pose'][:3]) - np.array(d['peg_pose'][:3]) for d in diags[m]])
        plt.plot(np.arange(1, len(rel) + 1), np.linalg.norm(rel, axis=1), label='Full ' + m, color=colors[m])
        upto = min(len(rel), len(expert_rel) - 1)
        divergence[m] = dict(first_tcp_peg_vector_difference_gt_1cm=first_above(
            np.linalg.norm(rel[:upto] - expert_rel[1:upto+1], axis=1), .01),
            maximum_same_timestep_difference_m=float(np.linalg.norm(rel[:upto] - expert_rel[1:upto+1], axis=1).max()))
    plt.xlabel('Action frame'); plt.ylabel('TCP–Peg distance m'); plt.legend()
    savefig(fig / 'relative_geometry.png'); paths.append('figures/relative_geometry.png')

    fig7, axes = plt.subplots(1, 3, figsize=(14, 4))
    for m, ax in zip('ABC', axes):
        history = [json.loads(line) for line in (Path(refs['checkpoints'][m]['best']['path']).parent.parent / 'history.jsonl').read_text().splitlines()]
        ax.plot([r['epoch'] for r in history], [r['train']['native_action7_mse'] for r in history], label='Train96')
        ax.plot([r['epoch'] for r in history], [r['validation']['native_action7_mse'] for r in history], label='Validation12')
        ax.axvline(full[m]['best']['epoch'], color='gray', ls='--'); ax.set_title(m); ax.set_yscale('log'); ax.set_xlabel('Epoch')
    axes[0].set_ylabel('Native Action7 MSE'); axes[0].legend()
    savefig(fig / 'epoch_trend.png'); paths.append('figures/epoch_trend.png')

    video = run / 'full_abc_vs_expert_after_action.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20, (512, 128))
    if not writer.isOpened(): raise RuntimeError('comparison MP4 encoder unavailable')
    for t in range(250):
        frames = [expert_rgb[min(t + 1, 137)]]
        frames += [rgbs[m][min(t + 1, len(rgbs[m]) - 1)] for m in 'ABC']
        blocks = []
        for name, frame in zip(('Expert', 'Full A', 'Full B', 'Full C'), frames):
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(image, f'{name} t={t}', (3, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
            blocks.append(image)
        writer.write(np.concatenate(blocks, axis=1))
    writer.release()

    thresholds = {}
    for m in 'ABC':
        rows = list(csv.DictReader((run / 'offline' / m / 'best' / 'timeline.csv').open()))
        thresholds[m] = dict(first_offline_translation_gt_1mm=first_above(
            [float(r['controller_translation_mm']) for r in rows], 1),
            first_offline_action7_l2_gt_0p1=first_above([float(r['action7_error_l2']) for r in rows], .1),
            first_gripper_sign_error=next((i for i, r in enumerate(rows)
                if (float(r['expert_6']) > 0) != (float(r['pred_6']) > 0)), None),
            equal_timestep_state_difference=divergence[m])
    atomic(run / 'first_error_diagnostics.json', thresholds, immutable=True)

    table = []
    for m in 'ABC':
        s, b, f, r = single['offline'][m], full[m]['best'], full[m]['final'], summary[m]
        table.append(dict(method=m, single_mse=s['native_action7_mse'], full_best_mse=b['native']['native_action7_mse'],
            gap=b['native']['native_action7_mse'] / s['native_action7_mse'], full_epoch50_mse=f['native']['native_action7_mse'],
            best_epoch=b['epoch'], best_update=b['global_step'], final_update=f['global_step'],
            single_translation_mm=1000*s['controller_translation_error_m']['mean'],
            full_translation_mm=b['controller_translation_mm']['mean'],
            single_rotation_rad=s['controller_rotation_geodesic_rad']['mean'],
            full_rotation_rad=b['controller_rotation_rad']['mean'],
            full_gripper_target_mm=b['controller_gripper_target_mm']['mean'],
            single_insert=single['closed_loop'][m]['episode']['success'],
            full_grasp=r['grasp'], full_lift=r['lift_4cm'], full_insert=r['success'],
            full_steps=r['episode_length']))
    atomic(run / 'comparison.json', dict(table=table, thresholds=thresholds,
        interpretation='One same-training-state rollout per Full method; equal-time state comparison is descriptive.'), immutable=True)

    original = read(plan_path.parent / 'status.json')
    source = Path(__file__).resolve()
    (run / 'source_snapshot_report.py').write_bytes(source.read_bytes())
    diagnostic = source.with_name('bc08e_diagnostic.py')
    (run / 'source_snapshot_diagnostic.py').write_bytes(diagnostic.read_bytes())
    diffs = []
    for path in (source, diagnostic):
        diffs.append(subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', str(path)],
                                    capture_output=True, text=True).stdout)
    (run / 'implementation.diff').write_text('\n'.join(diffs))
    links = ['figures/' + Path(p).name for p in paths]
    links += ['full_abc_vs_expert_after_action.mp4']
    links += [f'rollout/{m}/after_action_rgb.mp4' for m in 'ABC']
    links += ['comparison.json', 'best_final_comparison.json', 'first_error_diagnostics.json', 'reset_identity_audit.json']
    docs = run / '08e_full_trainseed_diagnostic.md'
    lines = ['# Task 08-E — Full A/B/C train-seed diagnostic', '',
      f'ExperimentLogger UUID `{run.name}`. Vast.ai 57640710 RTX 3090 inference; CPU PhysX 20 Hz control / 100 Hz physics. No training.', '',
      '## Full 작업 중지와 보존', '',
      f'원본 pipeline `{plan_path.parent.name}`은 user-request interrupt 후 `{original["status"]}` 상태다. Full A/B/C 50 Epoch와 Development10/Method는 완료됐다. Historical7 A/B는 완료, C는 1/7 완료 뒤 interrupt. Final40/Method는 0/40 완료, 미실행은 실패로 계산하지 않는다. 중지 사유 `INTERRUPTED_BY_USER_REQUEST`.',
      '원본 Checkpoint, Metrics, Plan, Split의 23개 고정 파일은 중지 전후 SHA256 일치. 원본 status/log/job 파일 3개는 중지 기록으로 변경됐다. 세부 내역은 pre/post stop audit JSON.', '',
      '## Train seed 30003: Offline 및 실제 Closed-loop', '',
      '| Method | Single MSE | Full Best MSE | 배율 | Full Epoch50 MSE | Best Epoch/Update | Controller translation mm | Rotation rad | Grasp | Lift4cm | Insert | Steps |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|']
    for row in table:
        lines.append(f'| {row["method"]} | {row["single_mse"]:.7g} | {row["full_best_mse"]:.7g} | {row["gap"]:.0f}× | {row["full_epoch50_mse"]:.7g} | {row["best_epoch"]}/{row["best_update"]} | {row["full_translation_mm"]:.3f} | {row["full_rotation_rad"]:.5f} | {row["full_grasp"]} | {row["full_lift"]} | {row["full_insert"]} | {row["full_steps"]} |')
    lines += ['', 'Single A/B/C는 각각 20,000 update와 약 4,000 Episode 노출로 학습했다. Full Best는 A35/B48/C38 Epoch 및 A8036/B11026/C8725 update다. Full은 96개 서로 다른 학습 Episode를 동시에 Fit해야 한다. 20,000 대 11,487 update 차이만으로 원인을 단정할 수 없다.', '',
      'Epoch50은 세 방법 모두 Train Seed Action7 MSE가 Best보다 낮다. 이는 특정 학습 Episode에서 추가 노출이 효과를 낼 수 있음을 지지하지만 Validation 및 실제 성공 개선을 보여주지 않는다. Epoch50 Closed-loop는 실행하지 않았다.', '',
      '## 최초 오류와 상태 이탈', '',
      'Threshold: 정상 Expert observation의 controller translation 오차 >1 mm, Native Action7 L2 >0.1, Gripper sign 오류; 실제 rollout의 같은 시점 TCP–Peg 상대 벡터와 Expert 차이 >1 cm. 마지막 지표는 서로 다른 자율 history의 기술적 비교이며 인과 위치를 확정하지 않는다.', '']
    for m in 'ABC':
        z = thresholds[m]
        lines.append(f'- {m}: offline translation >1 mm t={z["first_offline_translation_gt_1mm"]}; Action7 L2 >0.1 t={z["first_offline_action7_l2_gt_0p1"]}; gripper sign t={z["first_gripper_sign_error"]}; same-time relative state >1 cm t={z["equal_timestep_state_difference"]["first_tcp_peg_vector_difference_gt_1cm"]}.')
    lines += ['', '## 연구 판단', '',
      'Full Best의 정상 Expert sequence 오차는 Single보다 수천~수만 배 크므로 Training Fit 한계가 확인된다. 이는 학습 예산, 96개 상태의 다양성, capacity 또는 optimizer 문제 중 어느 하나를 단독으로 확정하지 않는다.',
      '실제 rollout은 새로운 RGB-D를 매 step 처리하고 이전 Student 행동이 만든 상태에서 다음 action을 예측했다. 상대 geometry의 Expert 기준은 기존 저장 Action 재생 GT sidecar이며 원본 Robot76과 byte-identical하지 않다. Offline 오차와 실제 상태 이탈은 함께 관측되므로 Distribution Shift만의 독립 기여를 분리할 수 없다.',
      '다음 우선순위는 같은 학습 seed의 stage별 action/controller 오차와 rollout 이탈 시점을 연결하고, 별도 사전등록 실험에서 Epoch 연장 시 Train/Validation/Closed-loop를 함께 평가하는 것이다. 필요하면 실패 state에 대한 복구 자료와 representation 진단을 후속 실험으로 분리한다.', '',
      '## Q1–Q10', '']
    q = [
      ('Q1','CONFIRMED','대상 Full pipeline/worker 종료, 선택 checkpoint와 학습 지표 hash 일치. Historical7 C 부분 결과 보존; Final 0/40.'),
      ('Q2','CONFIRMED','137-frame Expert observation offline MSE와 controller 오차를 위 표와 JSON에 기록.'),
      ('Q3','CONFIRMED','Single 대비 Full MSE 배율과 controller 오차를 comparison.json에 기록.'),
      ('Q4','CONFIRMED','A/B/C 각 1회 실제 CPU PhysX closed-loop의 grasp/lift/insert 결과를 기록.'),
      ('Q5','SUPPORTED','분석용 수치 threshold의 최초 frame과 상대 geometry 이탈을 기록. 인과적 첫 오류 시점은 단일 rollout으로 확정 불가.'),
      ('Q6','CONFIRMED','Epoch50의 동일 seed offline MSE를 Best와 읽기 전용 비교.'),
      ('Q7','SUPPORTED','Single보다 큰 학습 seed 오차는 현재 Full 설정의 부족한 fit을 지지. 원인 분리는 미완료.'),
      ('Q8','SUPPORTED','Epoch50 train-seed 개선과 원래 validation 선택 시점은 구분 가능. Epoch50 closed-loop 효과는 NOT_TESTED.'),
      ('Q9','INCONCLUSIVE','Development 0/10에는 train fit과 새로운 상태 일반화가 모두 기여할 수 있음.'),
      ('Q10','SUPPORTED','즉시 학습 연장 단정 대신 검증 악화와 폐루프 이탈을 분리할 후속 실험 근거 확보.')]
    lines += [f'- **{number} {status}:** {answer}' for number, status, answer in q]
    lines += ['', '## 시각화 및 Artifact', '']
    lines += [f'- [{Path(p).stem}]({p})' for p in paths]
    lines += [f'- [Expert vs Full A/B/C after-action MP4]({video.name})']
    lines += [f'- [Full {m} after-action MP4](rollout/{m}/after_action_rgb.mp4)' for m in 'ABC']
    lines += ['', '원본 Checkpoint와 학습 산출물은 수정하지 않았다. 원본 운영 상태·로그·작업 기록 3개는 중지에 따라 갱신됐다. CUDA PhysX, real-time deadline, Epoch50 closed-loop 및 새로운 학습은 실행하지 않았다. 외부 업로드·dependency 설치·Git commit 없음.', '']
    docs.write_text('\n'.join(lines))
    body = '<h1>Task 08-E Full A/B/C train-seed diagnostic</h1><pre>' + html.escape('\n'.join(lines[:45])) + '</pre>'
    body += ''.join(f'<h2>{html.escape(Path(p).stem)}</h2><img src="{p}" width="900">' for p in paths)
    body += f'<h2>After-action RGB</h2><video controls width="1024" src="{video.name}"></video>'
    body += ''.join(f'<p><a href="{p}">{html.escape(p)}</a></p>' for p in links)
    webpage = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Task 08-E</title><style>body{font:16px system-ui;max-width:1200px;margin:2em}img{max-width:100%}pre{white-space:pre-wrap}</style>' + body + '</html>'
    (run / 'full_trainseed_diagnostic.html').write_text(webpage)
    checks = {'json': 0, 'npz': 0, 'csv': 0, 'png': 0, 'mp4': 0, 'html_links': 0}
    for p in run.rglob('*'):
        if not p.is_file(): continue
        if p.suffix == '.json': json.load(p.open()); checks['json'] += 1
        elif p.suffix == '.npz':
            with np.load(p, allow_pickle=False) as f:
                for name in f.files: f[name]
            checks['npz'] += 1
        elif p.suffix == '.csv':
            if not list(csv.reader(p.open())): raise ValueError('empty CSV')
            checks['csv'] += 1
        elif p.suffix == '.png':
            if cv2.imread(str(p)) is None: raise ValueError('bad PNG')
            checks['png'] += 1
        elif p.suffix == '.mp4':
            cap = cv2.VideoCapture(str(p))
            ok, _ = cap.read(); cap.release()
            if not ok: raise ValueError('bad MP4: ' + str(p))
            checks['mp4'] += 1
    for p in links:
        if not (run / p).is_file(): raise ValueError('broken relative HTML link: ' + p)
        checks['html_links'] += 1
    atomic(run / 'verification.json', dict(status='PASS', checks=checks,
        questions={number: status for number, status, _ in q},
        source_snapshot_sha256={p:sha(run / p) for p in ('source_snapshot_report.py','source_snapshot_diagnostic.py')},
        process_exit='Full coordinator and historical7_C worker not alive before diagnostic'), immutable=True)
    for path, kind in [(run / 'comparison.json', 'metrics'),
                       (run / 'verification.json', 'metrics'),
                       (run / 'full_trainseed_diagnostic.html', 'reports'),
                       (docs, 'reports'), (video, 'rollouts')]:
        register_existing_output(run, path, kind=kind)
    return docs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    print(make_report(args.run, args.plan))
