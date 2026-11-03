"""Single-attempt local Terra calls with a locked, durable reservation ledger."""
from common import *
import argparse
import base64
import fcntl
import time
import requests


def payload(prompt,png):
    return dict(model=MODEL,reasoning={'effort':'low'},service_tier='default',max_output_tokens=MAX_OUTPUT,store=False,
        input=[dict(role='user',content=[dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(png).decode('ascii'),detail='original'),
                                      dict(type='input_text',text=prompt)])])


def request_once(data,key):
    start=time.perf_counter()
    try:
        response=requests.post('https://api.openai.com/v1/responses',json=data,
            headers={'Authorization':'Bearer '+key},timeout=(15,240),allow_redirects=False)
        http_ms=(time.perf_counter()-start)*1000
        begin=time.perf_counter()
        try:body=response.json()
        except ValueError:body={'error_type':'NON_JSON_RESPONSE'}
        if not isinstance(body,dict):body={'error_type':'NON_OBJECT_RESPONSE'}
        text=''.join(x.get('text','') for m in body.get('output',[]) if m.get('type')=='message'
                     for x in m.get('content',[]) if x.get('type')=='output_text')
        refused=any(x.get('type')=='refusal' for m in body.get('output',[]) if m.get('type')=='message' for x in m.get('content',[]))
        model=body.get('model','');tier=body.get('service_tier')
        status=('AUTH_ERROR' if response.status_code in (401,403) else 'SERVICE_ERROR' if response.status_code>=429 else
                'REQUEST_ERROR' if response.status_code!=200 else 'MODEL_MISMATCH' if not(model==MODEL or model.startswith(MODEL+'-')) else
                'TIER_MISMATCH' if tier!='default' else 'REFUSAL' if refused else
                'TOKEN_LIMIT' if (body.get('incomplete_details') or {}).get('reason')=='max_output_tokens' else
                'RESPONSE_ERROR' if body.get('status')!='completed' else 'OK')
        return dict(status=status,final_text=text,returned_model=model,served_tier=tier,
                    usage=body.get('usage'),raw=body,http_status=response.status_code,
                    request_id=response.headers.get('x-request-id'),http_ms=http_ms,
                    decode_ms=(time.perf_counter()-begin)*1000)
    except requests.RequestException as e:
        return dict(status='TIMEOUT' if isinstance(e,requests.Timeout) else 'CONNECTION_ERROR',
                    final_text='',usage=None,raw={'error_type':type(e).__name__},http_ms=(time.perf_counter()-start)*1000,
                    decode_ms=0.,returned_model=None,served_tier=None)


def main(dry_run=False):
    p=assert_source();manifest=read(ROOT/'capture_manifest.json')
    if (ROOT/'api_freeze.json').exists():
        freeze=read(ROOT/'api_freeze.json')
        assert sha(ROOT/'api_predictions.json')==freeze['predictions_sha256']
        print('API_ALREADY_FROZEN; zero requests; artifacts unchanged',flush=True)
        return
    all_images=manifest['entries']
    images=[e for e in all_images if e['cohort']!='tracking' or e['timestep']==0]
    assert len(images)==280 and sum(e['cohort']!='tracking' for e in images)==248
    # All API inputs must already exist and freeze independently of later tracking capture.
    api_inputs=ROOT/'api_input_manifest.json'
    sanitized=[{k:e[k] for k in ('id','scene','size','camera','cohort','timestep','parent','image_file','image_sha256')} for e in images]
    if api_inputs.exists():assert read(api_inputs)['images']==sanitized
    else:write(api_inputs,dict(images=sanitized,frozen_utc=now()))
    # Image dimensions are interleaved within physical state/camera and repeated sessions.
    by={(e['scene'],e['size'],e['camera']):e for e in images}
    pairs=list(dict.fromkeys((e['scene'],e['camera']) for e in images))
    jobs=[]
    for repeat in range(3):
        for i,pair in enumerate(pairs):
            for size in balanced_order(i+repeat):
                e=by[pair[0],size,pair[1]]
                jobs.append(dict(id=job_id(e['id'],repeat),image=e['id'],scene=e['scene'],parent=e['parent'],
                                 cohort=e['cohort'],camera=e['camera'],size=size,repeat=repeat,image_file=e['image_file'],image_sha256=e['image_sha256']))
    if dry_run:
        print(json.dumps(dict(requests_max=len(jobs),budget_usd=BUDGET,network_calls=0)));return
    from dotenv import load_dotenv
    load_dotenv(REPO/'.env',override=False)
    key=os.environ.get('OPENAI_API_KEY','').strip()
    assert key,'OPENAI_API_KEY missing; value not logged'
    def scrub(v):
        if isinstance(v,dict):return {k:scrub(x) for k,x in v.items()}
        if isinstance(v,list):return [scrub(x) for x in v]
        return v.replace(key,'[REDACTED]') if isinstance(v,str) else v
    lock=(ROOT/'api.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statefile=ROOT/'request_ledger.json'
    state=read(statefile) if statefile.exists() else dict(jobs={},stop_reason=None,consecutive_errors=0)
    prompt=(ROOT/'grounding_prompt.txt').read_text();assert sha(ROOT/'grounding_prompt.txt')==p['prompt_sha256']
    with logger('api') as log:
        for job in jobs:
            if state.get('stop_reason'):break
            if job['id'] in state['jobs']:continue
            path=ROOT/job['image_file'];assert sha(path)==job['image_sha256']
            begin=time.perf_counter();data=payload(prompt,path.read_bytes());prepare_ms=(time.perf_counter()-begin)*1000
            if not reserve_job(state,job,p['request_reservation_usd']):write(statefile,state);break
            state['jobs'][job['id']]['request_sha256']=digest(data)
            write(statefile,state)
            result=scrub(request_once(data,key))
            begin=time.perf_counter();parsed=strict_parse(result['final_text'],job['size'],job['size']) if result['status']=='OK' else strict_parse('',job['size'],job['size'])
            parse_ms=(time.perf_counter()-begin)*1000
            usage=result.get('usage') or {};cost=None
            if all(type(usage.get(k)) is int and usage[k]>=0 for k in ('input_tokens','output_tokens')):
                cost=(usage['input_tokens']+5*usage['output_tokens'])/1e6
                if result.get('served_tier') in ('fast','priority'):cost*=2
            j=state['jobs'][job['id']]
            j.update(result,parsed=parsed,prepare_ms=prepare_ms,parse_ms=parse_ms,
                     accounted_usd=cost if cost is not None else j['reserved_usd'],finished_utc=now())
            rawpath=ROOT/'api_raw'/(job['id']+'.json');write(rawpath,result)
            j.pop('raw',None);j['raw_file']=str(rawpath.relative_to(ROOT));j['raw_sha256']=sha(rawpath)
            transient=result['status'] in ('SERVICE_ERROR','TIMEOUT','CONNECTION_ERROR','RESPONSE_ERROR')
            state['consecutive_errors']=state['consecutive_errors']+1 if transient else 0
            if result['status'] in ('AUTH_ERROR','REQUEST_ERROR','MODEL_MISMATCH','TIER_MISMATCH') or state['consecutive_errors']>=3:
                state['stop_reason']=result['status']
            if cost is not None and cost>j['reserved_usd']:state['stop_reason']='RESERVATION_BOUND_BREACHED'
            write(statefile,state)
            if len(state['jobs'])%10==0:
                log.log(dict(requests=len(state['jobs']),accounted_usd=accounted(state)))
                print('API',len(state['jobs']),'/840',result['status'],round(accounted(state),5),flush=True)
        rows=[]
        for job in jobs:
            existing=state['jobs'].get(job['id'])
            row=dict(existing) if existing is not None else dict(job,status='NOT_ATTEMPTED')
            row.setdefault('parsed',strict_parse('',job['size'],job['size']))
            row['unknown_server_outcome']=row['status']=='RESERVED'
            rows.append(row)
        write(ROOT/'api_predictions.json',dict(rows=rows,stop_reason=state['stop_reason'],frozen_utc=now()))
        write(ROOT/'api_freeze.json',dict(predictions_sha256=sha(ROOT/'api_predictions.json'),ledger_sha256=sha(statefile),
             raw_files={j['raw_file']:j['raw_sha256'] for j in state['jobs'].values() if 'raw_file' in j},
             requested=len(state['jobs']),planned=840,accounted_usd=accounted(state),frozen_utc=now()))
        log.link_output(ROOT/'api_freeze.json')
        print('API_FROZEN',len(state['jobs']),accounted(state),state['stop_reason'],flush=True)
    lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');main(ap.parse_args().dry_run)
