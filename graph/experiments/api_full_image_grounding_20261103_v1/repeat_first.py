"""User-authorized one repeat/provider of first image; excluded from primary accuracy."""
from common import *
from adapters import payload, call
from dotenv import load_dotenv

def main():
    verify_frozen()
    prompt=(ROOT/'grounding_prompt.txt').read_text()
    image=read(ROOT/'input_manifest.json')['images'][0]
    r=ROOT/'first_request_repeat';r.mkdir(exist_ok=True)
    primary_bound=sum(budget_reservation(p,prompt)*60 for p in MODELS)
    repeat_bound=sum(budget_reservation(p,prompt) for p in MODELS)
    assert primary_bound+repeat_bound<=20
    identity=r/'authorization.json'
    if not identity.exists():
        write(identity,dict(user_request='Repeat first request once for each of three models; inspect latency',
            primary_unchanged=True,excluded_from_primary_accuracy=True,requests_max=3,
            total_requests_including_primary_max=183,combined_worst_case_usd=primary_bound+repeat_bound,
            image_id=image['id'],source_sha256=sha(__file__),created_utc=now(),
            isolation_note='Primary evaluation remains active; not a pure warm-up or isolated latency benchmark.'))
    load_dotenv(REPO/'.env',override=False)
    keys={p:os.environ.get(c['key'],'').strip() for p,c in MODELS.items()}
    assert all(keys.values())
    def scrub(x):
        if isinstance(x,dict):return {k:scrub(v) for k,v in x.items()}
        if isinstance(x,list):return [scrub(v) for v in x]
        if isinstance(x,str):
            for key in keys.values():x=x.replace(key,'[REDACTED]')
        return x
    lp=r/'ledger.json';ledger=read(lp) if lp.exists() else {'jobs':{}}
    assert sha(image['image_path'])==image['image_sha256']
    for p in MODELS:
        if p in ledger['jobs']:continue
        reserve=budget_reservation(p,prompt)
        job=dict(provider=p,status='RESERVED',reserved_usd=reserve,accounted_usd=reserve,started_utc=now())
        ledger['jobs'][p]=job;write(lp,ledger)
        data=payload(p,prompt,Path(image['image_path']).read_bytes())
        result=scrub(call(p,data,keys[p]));write(r/(p+'_raw.json'),result)
        cost=used_cost(p,result.get('usage'))
        job.update({k:result.get(k) for k in ['status','latency_ms','usage','returned_model','finish_reason','http_status']})
        job.update(usage_estimated_usd=cost,accounted_usd=cost if cost is not None else reserve,
                   parsed=strict_parse(result['final_text']) if result['status']=='OK' else strict_parse(''),
                   diagnostic=diagnostic_parse(result['final_text']),request_sha256=digest(data),
                   raw_sha256=sha(r/(p+'_raw.json')),finished_utc=now())
        write(lp,ledger)
        print(json.dumps({'repeat_provider':p,'status':job['status'],'latency_s':job['latency_ms']/1000,
                          'usage':job['usage'],'estimated_usd':cost}),flush=True)
    primary=read(ROOT/'request_ledger.json')['jobs']
    summary={}
    for p in MODELS:
        first=primary[p+'__'+image['id']];second=ledger['jobs'][p]
        summary[p]=dict(first_seconds=first.get('latency_ms',0)/1000,
                       repeat_seconds=second.get('latency_ms',0)/1000,
                       first_usage=first.get('usage'),repeat_usage=second.get('usage'),
                       first_status=first['status'],repeat_status=second['status'],
                       repeat_raw_sha256=second.get('raw_sha256'))
    write(r/'latency_comparison.json',dict(models=summary,samples_per_condition=1,
        additional_requests=len(ledger['jobs']),extra_accounted_usd=ledger_total(ledger['jobs']),
        paired_image_id=image['id'],GT_used=False,concurrent_primary=True,
        note='One repeat per model; changes may reflect reasoning token count, caching, service load and network.'))
    print('FIRST_REQUEST_REPEATS_COMPLETE',flush=True)

if __name__=='__main__':main()
