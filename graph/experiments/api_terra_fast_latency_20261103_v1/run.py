"""Isolated user-requested Terra Fast HTTP timing; no SAM2/GT/GPU execution."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
BASE_SOURCE = REPO / 'graph/experiments/api_terra_luna_grounding_20261103_v1'
sys.path.insert(0, str(BASE_SOURCE))
import common as base
from adapters import payload, call

ROOT = REPO / 'outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1'
REPORT = REPO / 'docs/visual-graph-integration/10i_stage4a2_api_terra_fast_latency_20261103.md'
PRIOR = [REPO / 'outputs/task10i/stage4a2' / n for n in (
    'api_full_image_grounding_20261103_v1', 'api_sol_kimi_grounding_20261103_v1',
    'api_terra_luna_grounding_20261103_v1')]
FAST_TIERS = {'fast', 'priority'}
write, read, sha, now = base.write, base.read, base.sha, base.now


def request_for(prompt, png):
    return dict(payload('terra', prompt, png), service_tier='fast')


def prior_cost():
    paths = [p / 'request_ledger.json' for p in PRIOR]
    paths.append(PRIOR[0] / 'first_request_repeat/ledger.json')
    return sum(base.ledger_total(read(p)['jobs']) for p in paths)


def reservation(prompt):
    return 2 * base.budget_reservation('terra', prompt)


def response_cost(result):
    cost = base.used_cost('terra', result.get('usage'))
    tier = (result.get('raw') or {}).get('service_tier')
    return None if cost is None else cost * (1 if tier == 'default' else 2)


def may_send(state, image_id):
    # RESERVED is never retried: a previous request may already have been billed.
    return not state.get('stopped') and image_id not in state['jobs']


def snapshot():
    files = set()
    for directory in PRIOR:
        files.update(p for p in directory.rglob('*') if p.is_file())
    for directory in [BASE_SOURCE, SOURCE,
                      REPO / 'graph/experiments/api_full_image_grounding_20261103_v1',
                      REPO / 'graph/experiments/api_sol_kimi_grounding_20261103_v1',
                      REPO / 'graph/task10i_stage4a2']:
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(REPO / 'docs/visual-graph-integration' / n for n in (
        '10i_stage4a2_api_openai_family_grounding_comparison.md',
        '10i_stage4a2_api_sol_kimi_grounding_comparison.md',
        '10i_stage4a2_api_full_image_grounding_comparison.md',
        '10i_stage4a2_paused_scope_review.md'))
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(files)}


def prepare():
    assert not ROOT.exists(), 'Existing experiment: refusing to overwrite preparation'
    prompt = (base.ROOT / 'grounding_prompt.txt').read_text()
    assert sha(base.ROOT / 'grounding_prompt.txt') == 'df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709'
    manifest = read(base.ROOT / 'input_manifest.json')
    assert len(manifest['images']) == 60
    for image in manifest['images']:
        assert sha(image['image_path']) == image['image_sha256']
    prior = prior_cost()
    bound = reservation(prompt) * 60
    assert bound <= 4 and prior + bound <= 20
    protocol = dict(created_utc=now(), user_request='terra fast의 소요 시간을 측정하고 싶어.',
        model='gpt-6.1-terra', requested_tier='fast', accepted_fast_tiers=sorted(FAST_TIERS),
        reasoning='high', image_detail='original', max_output_tokens=5120, store=False,
        requests_max=60, concurrency=1, retries=0, warmup_calls=0,
        prompt_sha256=sha(base.ROOT / 'grounding_prompt.txt'), prior_accounted_usd=prior,
        additional_reserved_max_usd=bound, aggregate_cap_usd=20,
        prices_per_million=dict(fast_input=2, fast_output=10, default_input=1, default_output=5),
        price_note='Uncached usage estimate; no cache discount. Unknown tier uses Fast premium conservatively. Not an invoice.',
        docs_checked_date_kst='2026-11-03', official_sources=[
            'https://developers.openai.com/api/docs/guides/fast-mode',
            'https://developers.openai.com/api/docs/models/gpt-6.1-terra'],
        historical_baseline=str(base.ROOT / 'request_ledger.json'),
        timing='Existing adapters.call: perf_counter around one non-streamed requests.post including response JSON parsing. No persistent HTTP session.',
        interpretation='Historical comparison, not randomized contemporaneous tier ablation. Network, output/reasoning length, caching, and service load can differ.',
        stop_rules=['terminal API error', 'three consecutive transport/service errors',
                    'OK response not served at fast/priority', 'budget or reservation breach'],
        excluded=['SAM2', 'GT scoring', 'simulator', 'GPU', 'tracking', 'graph', 'training', 'production', 'commit', 'publishing'])
    ROOT.mkdir(parents=True)
    (ROOT / 'grounding_prompt.txt').write_text(prompt)
    write(ROOT / 'input_manifest.json', manifest)
    write(ROOT / 'protocol.json', protocol)
    write(ROOT / 'protected_before.json', snapshot())
    write(ROOT / 'preparation_freeze.json', {n: sha(ROOT / n) for n in (
        'grounding_prompt.txt', 'input_manifest.json', 'protocol.json', 'protected_before.json')})
    print(json.dumps(dict(stage='PREPARED', requests_max=60, prior_usd=prior, reserved_max_usd=bound)))


def infer():
    for name, expected in read(ROOT / 'preparation_freeze.json').items():
        assert sha(ROOT / name) == expected, name
    protected = read(ROOT / 'protected_before.json')
    assert all(sha(REPO / p) == h for p, h in protected.items()), 'Protected source/artifact changed'
    assert not (ROOT / 'api_freeze.json').exists(), 'API already frozen; no additional calls permitted'
    from dotenv import load_dotenv
    load_dotenv(REPO / '.env', override=False)
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    assert key, 'OPENAI_API_KEY missing (value not logged)'

    def scrub(x):
        if isinstance(x, dict): return {k: scrub(v) for k, v in x.items()}
        if isinstance(x, list): return [scrub(v) for v in x]
        return x.replace(key, '[REDACTED]') if isinstance(x, str) else x

    state_path = ROOT / 'request_ledger.json'
    state = read(state_path) if state_path.exists() else dict(jobs={}, started_utc=now(), stopped=None, consecutive_errors=0)
    prompt = (ROOT / 'grounding_prompt.txt').read_text()
    images = read(ROOT / 'input_manifest.json')['images']
    protocol = read(ROOT / 'protocol.json')
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(REPO))
    from utils.experiment_logger import ExperimentLogger
    with ExperimentLogger(team='integration', model='gpt-6.1-terra_fast_latency', seed=42,
            job_type='evaluate', root=ROOT / '.execution/experiments', config=protocol) as log:
        for i, image in enumerate(images):
            if not may_send(state, image['id']): continue
            bound = reservation(prompt)
            if protocol['prior_accounted_usd'] + base.ledger_total(state['jobs']) + bound > 20:
                state['stopped'] = 'BUDGET_LIMIT'
                break
            p = Path(image['image_path'])
            assert sha(p) == image['image_sha256']
            request = request_for(prompt, p.read_bytes())
            job = dict(id=image['id'], status='RESERVED', started_utc=now(),
                reserved_usd=bound, accounted_usd=bound, request_sha256=base.digest(request),
                image_sha256=image['image_sha256'])
            state['jobs'][image['id']] = job
            write(state_path, scrub(state))
            wall_start = time.time()
            result = scrub(call('terra', request, key))
            wall_elapsed = time.time() - wall_start
            raw_path = ROOT / 'raw' / (image['id'] + '.json')
            write(raw_path, result)
            cost = response_cost(result)
            tier = (result.get('raw') or {}).get('service_tier')
            job.update({k: result.get(k) for k in ('status', 'latency_ms', 'usage', 'returned_model', 'request_id', 'http_status', 'finish_reason')})
            job.update(served_tier=tier, finished_utc=now(), wall_elapsed_s=wall_elapsed,
                wall_minus_perf_s=wall_elapsed - result['latency_ms'] / 1000,
                raw_file=str(raw_path.relative_to(ROOT)), raw_sha256=sha(raw_path),
                usage_estimated_usd=cost, accounted_usd=cost if cost is not None else bound,
                parsed=base.strict_parse(result['final_text']) if result['status'] == 'OK' else base.strict_parse(''))
            if cost is not None and cost > bound: state['stopped'] = 'RESERVATION_BOUND_BREACHED'
            if result['status'] in {'AUTH_ERROR', 'MODEL_UNAVAILABLE', 'REQUEST_ERROR', 'MODEL_MISMATCH'}:
                state['stopped'] = result['status']
            if result['status'] == 'OK' and tier not in FAST_TIERS:
                state['stopped'] = 'FAST_TIER_NOT_CONFIRMED'
            transient = result['status'] in {'SERVICE_ERROR', 'TIMEOUT', 'CONNECTION_ERROR', 'RESPONSE_ERROR'}
            state['consecutive_errors'] = state['consecutive_errors'] + 1 if transient else 0
            if state['consecutive_errors'] >= 3: state['stopped'] = 'THREE_CONSECUTIVE_SERVICE_ERRORS'
            write(state_path, scrub(state))
            log.log({'requests': len(state['jobs']), 'latency_s': result['latency_ms'] / 1000,
                     'accounted_usd': base.ledger_total(state['jobs'])})
            print(json.dumps(dict(image=i + 1, status=job['status'], served_tier=tier,
                latency_s=round(result['latency_ms']/1000, 3),
                accounted_usd=round(base.ledger_total(state['jobs']), 6))), flush=True)
        state['finished_utc'] = now()
        write(state_path, scrub(state))
        write(ROOT / 'api_freeze.json', dict(frozen_utc=now(), ledger_sha256=sha(state_path),
            raw_files={j['raw_file']:j['raw_sha256'] for j in state['jobs'].values() if 'raw_file' in j}))
        log.link_output(state_path)
    print('API_FROZEN', len(state['jobs']), state['stopped'], flush=True)


def statistics(jobs):
    import numpy as np
    values = [j['latency_ms']/1000 for j in jobs if j.get('latency_ms') is not None]
    return dict(n=len(values), mean_s=float(np.mean(values)), median_s=float(np.median(values)),
        p95_s=float(np.percentile(values, 95)), min_s=min(values), max_s=max(values)) if values else dict(n=0)


def summarize():
    state = read(ROOT / 'request_ledger.json')
    freeze = read(ROOT / 'api_freeze.json')
    assert sha(ROOT / 'request_ledger.json') == freeze['ledger_sha256']
    for p, h in freeze['raw_files'].items(): assert sha(ROOT / p) == h
    jobs = list(state['jobs'].values())
    fast = [j for j in jobs if j['status'] == 'OK' and j['served_tier'] in FAST_TIERS]
    old = [j for j in read(base.ROOT / 'request_ledger.json')['jobs'].values() if j['provider'] == 'terra']
    old_ok = [j for j in old if j['status'] == 'OK']
    old_by_id = {j['image_id']:j for j in old_ok}
    paired = [j for j in fast if j['id'] in old_by_id]
    import numpy as np
    paired_rows = [dict(image_id=j['id'], standard_s=old_by_id[j['id']]['latency_ms']/1000,
        fast_s=j['latency_ms']/1000, fast_minus_standard_s=(j['latency_ms']-old_by_id[j['id']]['latency_ms'])/1000,
        standard_over_fast=old_by_id[j['id']]['latency_ms']/j['latency_ms'],
        standard_output_tokens=(old_by_id[j['id']].get('usage') or {}).get('output_tokens'),
        fast_output_tokens=(j.get('usage') or {}).get('output_tokens')) for j in paired]
    base.csvwrite(ROOT / 'paired_latency.csv', paired_rows)
    protocol = read(ROOT / 'protocol.json')
    summary = dict(status='COMPLETE' if len(fast) == 60 else 'PARTIAL', stopped=state['stopped'],
        requests=len(jobs), statuses=dict(Counter(j['status'] for j in jobs)),
        served_tiers=dict(Counter(j.get('served_tier') or 'unreported' for j in jobs)),
        fast_ok=statistics(fast), all_attempts_with_timing=statistics(jobs), standard_historical_ok=statistics(old_ok),
        fast_first=statistics(fast[:1]), fast_after_first=statistics(fast[1:]),
        paired_n=len(paired_rows),
        paired_median_standard_over_fast=float(np.median([r['standard_over_fast'] for r in paired_rows])) if paired_rows else None,
        paired_faster_count=sum(r['fast_s'] < r['standard_s'] for r in paired_rows),
        strict_json_valid=sum(j.get('parsed',{}).get('json_valid',False) for j in jobs),
        coordinate_contract_including_null=sum(j.get('parsed',{}).get('json_valid',False) and all(j['parsed'][o]['status'] in {'VALID','NULL'} for o in ['peg','box']) for j in jobs),
        usage_estimated_usd=sum(j.get('usage_estimated_usd') or 0 for j in jobs),
        unknown_usage_reservation_usd=sum(j['accounted_usd'] for j in jobs if j.get('usage_estimated_usd') is None),
        conservative_usd=base.ledger_total(state['jobs']),
        cumulative_conservative_usd=protocol['prior_accounted_usd'] + base.ledger_total(state['jobs']),
        elapsed_s=(datetime.fromisoformat(state['finished_utc'])-datetime.fromisoformat(state['started_utc'])).total_seconds(),
        timing_clock_discrepancies=sum(abs(j.get('wall_minus_perf_s',0)) > 2 for j in jobs),
        raw_and_ledger_sha_verified=True, accuracy_measured=False)
    write(ROOT / 'latency_summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'infer', 'summarize'])
    args = ap.parse_args()
    if args.stage == 'prepare': return prepare()
    if args.stage == 'summarize': return summarize()
    guard = None
    with (ROOT / '.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sys.platform == 'darwin':
            guard = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: infer()
        finally:
            if guard is not None:
                guard.terminate()
                guard.wait(timeout=5)


if __name__ == '__main__': main()
