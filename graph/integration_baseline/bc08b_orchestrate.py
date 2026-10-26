"""Complete Task08-B evaluation after both independent fixed-budget workers finish."""
import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from graph.integration_baseline.bc08_train import load_plan, load_checkpoint
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.bc08_report import snapshot


def protected_end(before):
    failures = []
    checked = 0
    def audit(path, expected):
        nonlocal checked
        actual = sha(path)
        checked += 1
        if actual != expected:
            failures.append(dict(path=path, expected=expected, actual=actual))
    for path, expected in before.items():
        audit(path, expected)
    return dict(status='PASS' if not failures else 'FAIL', checked_files=checked,
                failures=failures, scope='frozen source/dataset/initial/checkpoint and completed Small A/C; active Full outputs excluded')


def finalize(plan_path):
    root = Path(plan_path).resolve().parent
    status = read(root/'task08b_status.json')
    if status['stage'] != 'EVALUATING' or set(status['evaluations']) != {'A', 'C'}:
        raise ValueError('both independent evaluations must complete before finalizing')
    for method, directory in status['evaluations'].items():
        report = read(Path(directory)/'evaluation.json')
        if report['status'] != 'COMPLETE' or len(report['episodes']) != 1:
            raise ValueError('incomplete same-seed evaluation: '+method)
    protection = protected_end(read(root/'protected_before.json'))
    atomic(root/'protected_after.json', protection, immutable=True)
    status['stage'] = 'COMPLETE' if protection['status'] == 'PASS' else 'PROTECTION_FAILED'
    status['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    atomic(root/'task08b_status.json', status)
    return 0 if status['stage'] == 'COMPLETE' else 3


def main(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path)
    root = plan_path.parent
    snapshot(root, plan)
    original = Path(__file__).resolve().with_name('bc08_train.py')
    worker = Path(__file__).resolve().with_name('bc08b_cached_train.py')
    diff = ''.join(difflib.unified_diff(original.read_text().splitlines(keepends=True),
                                        worker.read_text().splitlines(keepends=True),
                                        fromfile='Task08/bc08_train.py', tofile='Task08-B/bc08b_cached_train.py'))
    (root/'task08b_worker.diff').write_text(diff)
    prior = read(Path(plan['source_comparison_plan']).parent/'seeds_manifest.json') if 'source_comparison_plan' in plan else None
    if prior is None:
        source = Path('/workspace/task08/experiments/3ed77d8b64884f879b9a628e34a9426a/seeds_manifest.json')
        prior = read(source)
    prior['groups']['train_seed'] = [e for e in prior['groups']['train_seed'] if e['seed'] == 30003]
    if len(prior['groups']['train_seed']) != 1 or prior['groups']['train_seed'][0]['status'] != 'READY':
        raise ValueError('frozen Seed 30003 reset identity is not READY')
    atomic(root/'seeds_manifest.json', prior, immutable=True)
    reset = read(root/'reset_identity.json')
    if reset['status'] != 'PASS_PHYSICAL_WITH_RENDER_QUANTIZATION':
        raise ValueError('independent current reset identity did not pass')
    status = dict(stage='WAIT_TRAINING', plan=str(plan_path), selected={}, evaluations={},
                  start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    atomic(root/'task08b_status.json', status)
    while True:
        complete = {}
        for method in ('A', 'C'):
            marker = root/f'overfit-{method}-train.json'
            if marker.exists():
                run = Path(read(marker)['directory'])
                if (run/'completion.json').exists():
                    complete[method] = read(run/'completion.json')
                elif (run/'error_bundle.json').exists() or read(run/'metadata.json').get('status') == 'failed':
                    status.update(stage='TRAINING_FAILED', failed_method=method,
                                  run=str(run), reason='worker error; no automatic resume')
                    atomic(root/'task08b_status.json', status)
                    return 1
        if len(complete) == 2:
            break
        time.sleep(60)
    for method, result in complete.items():
        if result['status'] != 'COMPLETE' or result['updates'] != 20000 or not result['reload_output_exact']:
            raise ValueError('training incomplete or checkpoint reload failed: '+method)
        load_checkpoint(result['selected'])
        status['selected'][method] = result['selected']
    status['stage'] = 'EVALUATING'
    atomic(root/'task08b_status.json', status)
    environment = os.environ.copy()
    for method in ('A', 'C'):
        cmd = [sys.executable, '-m', 'graph.integration_baseline.bc08b_eval',
               '--plan', str(plan_path), '--phase', 'overfit', '--method', method,
               '--group', 'train_seed', '--receipt', str(Path(complete[method]['directory']) /
                                                      ('best.json' if Path(complete[method]['directory'], 'best.json').exists() else 'last_valid.json'))]
        log_path = root/f'rollout_{method}.log'
        before = set(p.name for p in Path(plan['experiments_root']).iterdir() if p.is_dir())
        with log_path.open('w') as log:
            completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2],
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
        after = set(p.name for p in Path(plan['experiments_root']).iterdir() if p.is_dir())
        new = [Path(plan['experiments_root'])/name for name in after-before]
        found = [p for p in new if (p/'evaluation.json').exists()]
        if completed.returncode or len(found) != 1:
            status.update(stage='EVALUATION_FAILED', failed_method=method,
                          log=str(log_path), returncode=completed.returncode,
                          candidate_runs=[str(p) for p in new])
            atomic(root/'task08b_status.json', status)
            return 2
        report = read(found[0]/'evaluation.json')
        if report['status'] != 'COMPLETE' or len(report['episodes']) != 1:
            raise ValueError('incomplete same-seed evaluation: '+method)
        status['evaluations'][method] = str(found[0])
        atomic(root/'task08b_status.json', status)
    return finalize(plan_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    raise SystemExit(finalize(args.plan) if args.finalize_only else main(args.plan))
