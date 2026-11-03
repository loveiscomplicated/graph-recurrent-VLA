"""Only this stage reads API credentials. No evaluation manifest/GT is loaded."""
from common import *
import argparse
from dotenv import load_dotenv
from adapters import payload, call

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    verify_frozen()
    images = read(ROOT / 'input_manifest.json')['images']
    prompt = (ROOT / 'grounding_prompt.txt').read_text()
    if args.dry_run:
        print(json.dumps({'network_calls': 0, 'requests_max': len(images)*len(MODELS),
                          'reserved_max_usd': sum(budget_reservation(p, prompt)*len(images) for p in MODELS)}))
        return
    load_dotenv(REPO / '.env', override=False)
    keys = {p: os.environ.get(c['key'], '').strip() for p, c in MODELS.items()}
    assert all(keys.values()), 'One or more required API keys are missing (values not logged)'
    def scrub(value):
        if isinstance(value, dict): return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list): return [scrub(v) for v in value]
        if isinstance(value, str):
            for key in keys.values(): value = value.replace(key, '[REDACTED]')
        return value
    statepath = ROOT / 'request_ledger.json'
    state = read(statepath) if statepath.exists() else dict(jobs={}, stopped={}, consecutive_errors={p: 0 for p in MODELS})
    v = read(ROOT / 'verification.json')
    if v['stage'] in ['API_PREDICTIONS_FROZEN', 'MASKS_FROZEN', 'COMPLETE']:
        print('API already frozen; no requests sent')
        return
    # A RESERVED job after interruption is never sent again: the server may have processed it.
    def persist(): write(statepath, scrub(state))
    with logger('infer') as log:
        for i, image in enumerate(images):
            providers = list(MODELS)
            providers = providers[i % len(providers):] + providers[:i % len(providers)]
            for provider in providers:
                jid = provider + '__' + image['id']
                if jid in state['jobs'] or provider in state['stopped']:
                    continue
                reserve = budget_reservation(provider, prompt)
                if PRIOR_ACCOUNTED_USD + ledger_total(state['jobs']) + reserve > 20.:
                    state['stopped'][provider] = 'BUDGET_LIMIT'
                    persist()
                    continue
                p = Path(image['image_path'])
                assert sha(p) == image['image_sha256']
                request = payload(provider, prompt, p.read_bytes())
                assert len(request.get('input', request.get('messages'))) == 1
                job = dict(id=jid, provider=provider, image_id=image['id'], status='RESERVED',
                           reserved_usd=reserve, accounted_usd=reserve, started_utc=now(),
                           request_sha256=digest(request), image_sha256=image['image_sha256'])
                state['jobs'][jid] = job
                persist()  # Durable intent exists before any network operation.
                result = scrub(call(provider, request, keys[provider]))
                served_tier = (result.get('raw') or {}).get('service_tier')
                if result['status'] == 'OK' and served_tier != 'default':
                    result['status'] = 'TIER_MISMATCH'
                rawpath = ROOT / 'raw' / (jid + '.json')
                write(rawpath, result)
                # Non-OK responses cannot become accepted boxes through partial text.
                parsed = strict_parse(result['final_text']) if result['status'] == 'OK' else strict_parse('')
                fenced = fence_parse(result['final_text']) if result['status'] == 'OK' else strict_parse('')
                diagnostic = diagnostic_parse(result['final_text'])
                mechanical = any(diagnostic[o]['status'] == 'VALID' for o in ['peg', 'box'])
                cost = used_cost(provider, result.get('usage'))
                if cost is not None and served_tier in ('fast', 'priority'):
                    cost *= 2  # Preserve actual-tier accounting even when an unexpected tier stops this effort.
                job.update({k: result.get(k) for k in ['status', 'latency_ms', 'usage', 'returned_model',
                                                      'request_id', 'finish_reason', 'http_status']})
                job['served_tier'] = served_tier
                job.update(raw_path=str(rawpath), raw_sha256=sha(rawpath), parsed=parsed,
                           diagnostic=diagnostic, fenced=fenced, mechanical_coordinates=mechanical,
                           usage_estimated_usd=cost, accounted_usd=cost if cost is not None else reserve,
                           finished_utc=now())
                if cost is not None and cost > reserve:
                    state['stopped'].update({p: 'RESERVATION_BOUND_BREACHED' for p in MODELS})
                if result['status'] in ['AUTH_ERROR', 'MODEL_UNAVAILABLE', 'REQUEST_ERROR', 'MODEL_MISMATCH', 'TIER_MISMATCH']:
                    state['stopped'][provider] = result['status']
                transient = result['status'] in ['SERVICE_ERROR', 'TIMEOUT', 'CONNECTION_ERROR', 'RESPONSE_ERROR']
                state['consecutive_errors'][provider] = state['consecutive_errors'][provider] + 1 if transient else 0
                if state['consecutive_errors'][provider] >= 3:
                    state['stopped'][provider] = 'THREE_CONSECUTIVE_SERVICE_ERRORS'
                if i == 2:
                    first = [state['jobs'].get(provider + '__' + x['id'], {}) for x in images[:3]]
                    if all(j.get('status') != 'RESERVED' for j in first) and not any(j.get('mechanical_coordinates') for j in first):
                        state['stopped'][provider] = 'GROUNDING_FORMAT_BLOCKED'
                persist()
                log.log({'requests': len(state['jobs']), 'accounted_usd': ledger_total(state['jobs'])})
                print(json.dumps({'image': i + 1, 'provider': provider, 'status': job['status'],
                                  'valid_boxes': sum(parsed[o]['status'] == 'VALID' for o in ['peg', 'box']),
                                  'fence_valid_boxes': sum(fenced[o]['status'] == 'VALID' for o in ['peg','box']),
                                  'latency_s': round(result['latency_ms']/1000,3), 'tier': served_tier,
                                  'accounted_usd': round(ledger_total(state['jobs']), 6)}), flush=True)
        # Every planned request gets an explicit status; skipped work is not silently dropped.
        rows = []
        for image in images:
            for provider in MODELS:
                j = state['jobs'].get(provider + '__' + image['id'])
                rows.append(dict(provider=provider, image_id=image['id'],
                    status=j['status'] if j else 'NOT_ATTEMPTED_' + state['stopped'].get(provider, 'UNKNOWN'),
                    attempted=j is not None, parsed=j.get('parsed', strict_parse('')) if j else strict_parse(''),
                    raw_sha256=j.get('raw_sha256') if j else None))
        fenced_rows = [dict(r, parsed=state['jobs'].get(r['provider']+'__'+r['image_id'],{}).get('fenced',strict_parse(''))) for r in rows]
        write(ROOT / 'fence_grounding_predictions.json', {'rows':fenced_rows, 'rule':'whole-answer optional json code fence only; no coordinate repair'})
        write(ROOT / 'full_image_grounding_predictions.json', {'rows': rows})
        csvwrite(ROOT / 'full_image_grounding_predictions.csv', [dict(provider=r['provider'], image_id=r['image_id'],
            status=r['status'], attempted=r['attempted'], json_valid=r['parsed']['json_valid'],
            **{o+'_status': r['parsed'][o]['status'] for o in ['peg','box']},
            **{o+'_pixels': r['parsed'][o]['pixels'] for o in ['peg','box']}) for r in rows])
        write(ROOT / 'full_image_grounding_raw_responses.json', dict(jobs=list(state['jobs'].values()), stopped=state['stopped']))
        v.update(stage='API_PREDICTIONS_FROZEN', request_count=len(state['jobs']), api_frozen_utc=now(),
                 api_files={n: sha(ROOT / n) for n in ['fence_grounding_predictions.json', 'request_ledger.json', 'full_image_grounding_predictions.json',
                       'full_image_grounding_predictions.csv', 'full_image_grounding_raw_responses.json']},
                 raw_files={str(Path(j['raw_path']).relative_to(ROOT)): j['raw_sha256']
                            for j in state['jobs'].values() if 'raw_path' in j})
        write(ROOT / 'verification.json', v)
        log.link_output(ROOT / 'full_image_grounding_predictions.json')
        print('API_PREDICTIONS_FROZEN', len(state['jobs']), state['stopped'], flush=True)

if __name__ == '__main__': main()
