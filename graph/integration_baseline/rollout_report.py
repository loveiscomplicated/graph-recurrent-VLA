"""Local static HTML, source snapshot and artifact decode/link verification."""
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np

from graph.integration_baseline.direct_runtime import atomic, read, sha


def snapshot(run):
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
    selected = [Path(p) for p in paths if Path(p).is_file() and
                (p.startswith(('graph/', 'visual/runtime/', 'recurrent/', 'envs/', 'utils/'))
                 or p in ('AGENTS.md', 'README.md'))]
    hashes = {}
    for path in selected:
        dest = run/'source_snapshot'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(path)] = sha(path)
    atomic(run/'source_hashes.json', hashes, immutable=True)
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'])
    for path in subprocess.check_output(['git','ls-files','--others','--exclude-standard'], text=True).splitlines():
        if path.startswith('graph/integration_baseline/'):
            part = subprocess.run(['git','diff','--no-index','--binary','/dev/null',path], capture_output=True)
            if part.returncode not in (0, 1):
                raise RuntimeError('source diff failed')
            diff += part.stdout
    (run/'implementation.diff').write_bytes(diff)


def render(run, report):
    atomic(run/'rollout_results.json', report)
    links = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and not any(x in path.parts for x in ('source_snapshot', 'installed_source')) and path.name not in ('stop_the_clock_rollout.html',):
            rel = path.relative_to(run).as_posix()
            links.append(f'<li><a href="{html.escape(rel, quote=True)}">{html.escape(rel)}</a></li>')
    text = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Task07 Stop-the-clock</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}'
            'pre{white-space:pre-wrap;background:#f3f4f6;padding:18px}a{color:#185bc4}</style>'
            '<h1>Task07 Stop-the-clock rollout</h1><p>'+html.escape(report['official_name'])+'</p>'
            '<p>Simulation control: 20Hz / 50ms; physics: 100Hz / 5 substeps. '
            'Inference wall time has no deadline. Offline BC (A), this protocol (B), '
            'and real-time performance (C) are separate.</p>'
            '<h2>Architecture and time contract</h2><pre>Live base_camera RGB-D + robot-only kinematics\n'
            '→ current RGB → existing Grounding DINO Tiny / SAM2.1 Small\n'
            '→ unchanged Shape-First / PCA5 / F5 / Raw EDE in EE metres\n'
            '→ Robot3 + object graph (k4 / M16) → edge-aware GNN L2 d64\n'
            '→ structured readout320 + presence2 + normalized Robot76\n'
            '→ GRU144 → native Action7 → pd_ee_delta_pose → next observation</pre>'
            '<p>Single synchronous environment. Observation, masks, geometry, graph and recurrent inference finish '
            'before exactly one env.step. Actual scene.step calls and elapsed control steps are audited. '
            'A run-local TimeLimitWrapper sets horizon250; the shared 100-step default remains. '
            'Wall throughput and 20FPS simulation videos are distinct.</p>'
            '<h2>Online state</h2><p>Only observed JPEG frames are registered into a bounds-checked SAM2 image sequence. '
            'Existing feature caches, temporal/object memories and forward propagation are preserved; '
            'num_frames=t+1. Original prefix DINO initialization and separate late-object states are reused. '
            'No backward refinement, future suffix, GT mask or object pose enters the policy. '
            'EpisodeStream carries hidden across consecutive frames and empty objects, rejects gaps, '
            'and resets on every episode boundary. Hidden NPZs identify episode and consumed timestep.</p>'
            '<h2>Checkpoint and protocol</h2><p>Evaluation requires an immutable SHA-verified transfer bundle '
            'with direct_bc_v1 weights, Best receipt, original run UUID, config, robot/action schemas, '
            'dataset/split/Vision manifests, train-only normalizer, source inventory, simulator conditions '
            'and Vision weight/SAM2 source identity. Restricted weights_only loading and strict model loading '
            'are mandatory. No optimizer or training RNG is restored. Missing metadata blocks evaluation. '
            'Frozen test12 and held-out40 seeds are fixed before performance execution and reported separately '
            'with Wilson 95% intervals. Every planned seed retains a status. Collision and task-failure '
            'metrics remain null when uninstrumented.</p>'
            '<h2>Readiness and limits</h2><p>Use rollout_stop_clock --mode smoke --config rollout_config.json '
            'for contract validation. After receiving the completed immutable checkpoint bundle, build its '
            'manifest, use --mode prepare-seeds, then --mode evaluate with both manifests. '
            'See the repository README and copied Task07 report for complete commands. '
            'Random-weight episodes validate the execution path only; trained policy success remains '
            'ROLLOUT_NOT_EXECUTED until an explicit checkpoint evaluation. CUDA/MPS and real-time control '
            'are not established by this CPU smoke.</p><h2>Results</h2><pre>'+html.escape(json.dumps(report, indent=2, ensure_ascii=False))+
            '</pre><h2>Artifacts</h2><ul>'+''.join(links)+'</ul></html>')
    (run/'stop_the_clock_rollout.html').write_text(text)


def verify(run):
    counts = dict(json=0, npz=0, png=0, mp4=0, links=0)
    for path in run.rglob('*'):
        if not path.is_file() or 'source_snapshot' in path.parts or 'installed_source' in path.parts:
            continue
        suffix = path.suffix
        if suffix == '.json':
            read(path); counts['json'] += 1
        elif suffix == '.npz':
            with np.load(path, allow_pickle=False) as arrays:
                for key in arrays.files:
                    arrays[key]  # force decode every compressed member
            counts['npz'] += 1
        elif suffix == '.png':
            if cv2.imread(str(path)) is None:
                raise ValueError('PNG decode failed')
            counts['png'] += 1
        elif suffix == '.mp4':
            video = cv2.VideoCapture(str(path))
            try:
                n = 0
                if abs(video.get(cv2.CAP_PROP_FPS)-20) > .01:
                    raise ValueError('video must use simulation 20FPS')
                while True:
                    ok, _ = video.read()
                    if not ok:
                        break
                    n += 1
                if n == 0 or n != int(video.get(cv2.CAP_PROP_FRAME_COUNT)):
                    raise ValueError('MP4 decode/frame count failed')
            finally:
                video.release()
            counts['mp4'] += 1
    for target in re.findall(r'href="([^"]+)"', (run/'stop_the_clock_rollout.html').read_text()):
        path = (run/html.unescape(target)).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            raise ValueError('invalid relative report link: '+target)
        counts['links'] += 1
    return dict(status='PASS', decoded=counts)
