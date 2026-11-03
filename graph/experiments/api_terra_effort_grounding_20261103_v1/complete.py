"""Final no-network verification, key-value scan and reproducible artifact index."""
from common import *
from datetime import datetime
import re
import subprocess
from dotenv import dotenv_values


def main():
    v=verify_frozen();assert v['stage']=='COMPLETE'
    for group in ['protocol_files','api_files','raw_files','sam2_files']:
        for n,h in v[group].items():assert sha(ROOT/n)==h,n
    for p,h in v['protected_before'].items():assert sha(p)==h,p
    for name in ['independent_verification.json','fence_independent_verification.json','remote_preservation_verification.json']:
        assert read(ROOT/name)['status']=='PASS',name
    state=read(ROOT/'request_ledger.json');jobs=list(state['jobs'].values())
    assert len(jobs)<=180 and PRIOR_ACCOUNTED_USD+ledger_total(state['jobs'])<=20
    assert len({j['id'] for j in jobs})==len(jobs)
    for j in jobs:
        if j['status']=='OK':assert j['served_tier']=='default'
    times=[j for j in jobs if j.get('finished_utc') and j.get('latency_ms') is not None]
    gaps=[dict(id=j['id'],wall_s=(datetime.fromisoformat(j['finished_utc'])-datetime.fromisoformat(j['started_utc'])).total_seconds(),
               monotonic_s=j['latency_ms']/1000) for j in times]
    if times:
        start=min(datetime.fromisoformat(j['started_utc']) for j in times)
        end=max(datetime.fromisoformat(j['finished_utc']) for j in times)
        data=subprocess.run(['pmset','-g','log'],capture_output=True,timeout=30).stdout.decode(errors='replace')
        events=[]
        for line in data.splitlines():
            m=re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+([+-]\d{4})\s+(Sleep|Wake|DarkWake)\s',line)
            if m:
                dt=datetime.strptime(m[1]+m[2],'%Y-%m-%d %H:%M:%S%z')
                if start<=dt<=end:events.append(dict(utc=dt.isoformat(),event=m[3]))
    else:events=[]
    sleep=[e for e in events if e['event']=='Sleep']
    write(ROOT/'host_runtime_verification.json',dict(events=events,sleep_events_during_API=len(sleep),jobs=gaps,
        max_abs_clock_gap_s=max((abs(x['wall_s']-x['monotonic_s']) for x in gaps),default=None),
        status='HOST_SLEEP_CONTAMINATED' if sleep else 'NO_SLEEP_EVENT_OBSERVED'))
    secrets=[value.encode() for key,value in dotenv_values(REPO/'.env').items()
             if value and ('KEY' in key or 'TOKEN' in key or 'SECRET' in key) and len(value)>8]
    paths=[p for base in [ROOT,SOURCE] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REPORT]
    for p in paths:
        data=p.read_bytes();assert not any(secret in data for secret in secrets),'Credential value detected (path suppressed)'
    links=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    local=[x for x in links if not x.startswith(('https:','http:','#'))]
    for link in local:assert (REPORT.parent/link).exists(),link
    masks=read(ROOT/'sam2/sam2_predictions.json')
    checks=dict(status='PASS',verified_utc=now(),new_requests=len(jobs),planned_max=180,automatic_retries=0,
        high_reused=True,SAM2_loads=read(ROOT/'sam2/sam2_latency_vram.json')['SAM2_loads'],SAM2_masks=len(masks['rows']),
        protected_files_unchanged=len(v['protected_before']),source_files_unchanged=len(v['source_before']),
        credential_scan_files=len(paths),credential_matches=0,report_local_links=len(local),report_sha256=sha(REPORT),
        sleep_events=len(sleep),production_changed=False,commit=False,published=False)
    write(ROOT/'completion_verification.json',checks)
    indexed=[p for p in ROOT.rglob('*') if p.is_file() and '.execution' not in p.relative_to(ROOT).parts
             and p.name not in ['artifact_index.json'] and p.suffix!='.log']
    index={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in indexed}
    write(ROOT/'artifact_index.json',dict(files=index,report=dict(path=str(REPORT),sha256=sha(REPORT))))
    for n,c in index.items():assert sha(ROOT/n)==c['sha256']
    print(json.dumps(dict(checks=checks,indexed_files=len(index)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
