"""Single-attempt HTTP adapters. No history, GT, tools, or SDK retries."""
import base64
import time
import requests
from common import MODELS

def payload(provider, prompt, png):
    encoded = base64.b64encode(png).decode('ascii')
    c = MODELS[provider]
    if MODELS[provider].get('kind') == 'openai':
        return dict(model=c['model'], reasoning={'effort': c['effort']}, service_tier='default', max_output_tokens=c['max_output'],
                    store=False, input=[{'role': 'user', 'content': [
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,' + encoded, 'detail': 'original'},
                        {'type': 'input_text', 'text': prompt}]}])
    if provider == 'zai':
        return dict(model=c['model'], messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}},
            {'type': 'text', 'text': prompt}]}], thinking={'type': 'enabled'},
            reasoning_effort='max', temperature=1., top_p=.95, max_tokens=c['max_output'], stream=False)
    return dict(model=c['model'], messages=[{'role': 'user', 'content': prompt, 'images': [encoded]}],
                think=True, stream=False,
                options={'temperature': 1., 'top_p': .95, 'top_k': 64, 'num_predict': c['max_output']})

def normalize(provider, body, http_status):
    c = MODELS[provider]
    result = dict(status='OK', final_text='', returned_model=body.get('model'), usage=None,
                  request_id=body.get('id') or body.get('request_id'), finish_reason=None)
    if http_status != 200:
        err = body.get('error', body)
        msg = str(err).lower()
        result['status'] = ('AUTH_ERROR' if http_status in (401, 403) else
                            'MODEL_UNAVAILABLE' if http_status == 404 or any(t in msg for t in
                              ['model_not_found', 'model not found', 'does not exist', 'model is not available']) else
                            'REQUEST_ERROR' if http_status in (400, 405, 422) else 'SERVICE_ERROR')
        return result
    if MODELS[provider].get('kind') == 'openai':
        messages = [x for x in body.get('output', []) if x.get('type') == 'message']
        result['final_text'] = ''.join(x.get('text', '') for m in messages for x in m.get('content', [])
                                       if x.get('type') == 'output_text')
        refused = any(x.get('type') == 'refusal' for m in messages for x in m.get('content', []))
        result['finish_reason'] = (body.get('incomplete_details') or {}).get('reason') or body.get('status')
        result['status'] = ('REFUSAL' if refused else 'TOKEN_LIMIT' if result['finish_reason'] == 'max_output_tokens'
                            else 'RESPONSE_ERROR' if body.get('status') != 'completed' else 'OK')
        u = body.get('usage')
        if u:
            result['usage'] = dict(input_tokens=u.get('input_tokens'), output_tokens=u.get('output_tokens'),
                                   reasoning_tokens=(u.get('output_tokens_details') or {}).get('reasoning_tokens'))
    elif provider == 'zai':
        choices = body.get('choices', [])
        choice = choices[0] if choices else {}
        m = choice.get('message', {})
        result['final_text'] = m.get('content') or ''
        result['finish_reason'] = choice.get('finish_reason')
        result['status'] = ('TOKEN_LIMIT' if result['finish_reason'] == 'length' else
                            'RESPONSE_ERROR' if not choices or m.get('tool_calls') else 'OK')
        u = body.get('usage')
        if u:
            result['usage'] = dict(input_tokens=u.get('prompt_tokens'), output_tokens=u.get('completion_tokens'),
                reasoning_tokens=(u.get('completion_tokens_details') or {}).get('reasoning_tokens'))
    else:
        result['final_text'] = (body.get('message') or {}).get('content') or ''
        result['finish_reason'] = body.get('done_reason')
        result['status'] = ('TOKEN_LIMIT' if result['finish_reason'] == 'length' else
                            'RESPONSE_ERROR' if not body.get('done') or body.get('error') else 'OK')
        if 'prompt_eval_count' in body and 'eval_count' in body:
            result['usage'] = dict(input_tokens=body['prompt_eval_count'], output_tokens=body['eval_count'],
                                   reasoning_tokens=None)
    if not isinstance(result['final_text'], str):
        result['final_text'] = ''
        result['status'] = 'RESPONSE_ERROR'
    if result['returned_model'] and result['returned_model'] != c['model']:
        # A server revision suffix is recorded; only actual different model IDs stop acceptance.
        model = result['returned_model']
        if not (MODELS[provider].get('kind') == 'openai' and isinstance(model,str) and model.startswith(c['model'] + '-')):
            result['status'] = 'MODEL_MISMATCH'
    return result

def call(provider, data, key):
    start = time.perf_counter()
    try:
        # requests' default HTTPAdapter retries=0. Redirects are disabled to avoid credential forwarding.
        response = requests.post(MODELS[provider]['endpoint'], json=data,
                                 headers={'Authorization': 'Bearer ' + key},
                                 timeout=(15, 240), allow_redirects=False)
        try:
            body = response.json()
            if not isinstance(body, dict):
                body = {'error': 'Non-object HTTP response'}
        except ValueError:
            body = {'error': response.text[:20000]}
        result = normalize(provider, body, response.status_code)
        result.update(http_status=response.status_code, raw=body,
                      header_request_id=response.headers.get('x-request-id'),
                      latency_ms=(time.perf_counter() - start) * 1000)
        return result
    except requests.RequestException as e:
        # Do not serialize requests exceptions: they may retain the authenticated request.
        return dict(status='TIMEOUT' if isinstance(e, requests.Timeout) else 'CONNECTION_ERROR',
                    final_text='', returned_model=None, usage=None, request_id=None, finish_reason=None,
                    http_status=None, raw={'error_type': type(e).__name__},
                    latency_ms=(time.perf_counter() - start) * 1000)
