"""Read-only validation, secret-value scan, runtime audit and local artifact index."""
from common import *
from datetime import datetime,timedelta
import re
import subprocess
from dotenv import dotenv_values

def main():
    v=verify_frozen();assert v['stage']=='COMPLETE'
    for group in ['protocol_files','api_files','raw_files','sam2_files','segmentation_inputs']:
        for n,h in v[group].items():assert sha(ROOT/n)==h,n
    for p,h in v['protected_before'].items():assert sha(p)==h,p
    assert read(ROOT/'independent_verification.json')['status']=='PASS'
    assert read(ROOT/'fence_independent_verification.json')['status']=='PASS'
    assert read(ROOT/'remote_preservation_verification.json')['status']=='PASS'
    jobs=list(read(ROOT/'request_ledger.json')['jobs'].values())
    assert len(jobs)<=120 and PRIOR_ACCOUNTED_USD+sum(j['accounted_usd'] for j in jobs)<=20
    assert read(ROOT/'sam2/sam2_latency_vram.json')['SAM2_loads']==1
    assert len(read(ROOT/'sam2/sam2_predictions.json')['rows'])<=358
    start=min(datetime.fromisoformat(j['started_utc']) for j in jobs)
    end=max(datetime.fromisoformat(j['finished_utc']) for j in jobs)
    pm=subprocess.run(['pmset','-g','log'],capture_output=True,timeout=30).stdout.decode(errors='replace')
    events=[]
    for line in pm.splitlines():
        m=re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+([+-]\d{4})\s+(Sleep|Wake|DarkWake)\s',line)
        if not m:continue
        dt=datetime.strptime(m[1]+m[2],'%Y-%m-%d %H:%M:%S%z')
        if start-timedelta(minutes=10)<=dt<=end+timedelta(minutes=10):events.append(dict(utc=dt.astimezone(start.tzinfo).isoformat(),event=m[3]))
    sleeps=[e for e in events if e['event']=='Sleep' and start<=datetime.fromisoformat(e['utc'])<=end]
    gaps=[dict(id=j['id'],UTC_elapsed_s=(datetime.fromisoformat(j['finished_utc'])-datetime.fromisoformat(j['started_utc'])).total_seconds(),HTTP_monotonic_s=j['latency_ms']/1000) for j in jobs]
    runtime=dict(events=events,sleep_events_during_API=len(sleeps),jobs=gaps,
        max_abs_clock_gap_s=max(abs(g['UTC_elapsed_s']-g['HTTP_monotonic_s']) for g in gaps),
        status='NO_SLEEP_EVENT_OBSERVED' if not sleeps else 'HOST_SLEEP_CONTAMINATED')
    write(ROOT/'host_runtime_verification.json',runtime)
    values=dotenv_values(REPO/'.env')
    secrets=[v.encode() for k,v in values.items() if v and ('KEY' in k or 'TOKEN' in k) and len(v)>8]
    paths=[p for base in [ROOT,SOURCE] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REPORT]
    for p in paths:
        data=p.read_bytes()
        assert not any(s in data for s in secrets),'Credential value found in an artifact (path suppressed)'
    links=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    local=[x for x in links if not x.startswith(('https:','http:','#'))]
    for x in local:assert (REPORT.parent/x).exists(),x
    checks=dict(status='PASS',verified_utc=now(),new_requests=len(jobs),automatic_retries=0,
        SAM2_loads=1,SAM2_masks=len(read(ROOT/'sam2/sam2_predictions.json')['rows']),
        original_files_unchanged=len(v['protected_before']),initial_source_unchanged=len(v['source_before']),
        credential_scan_files=len(paths),credential_matches=0,report_local_links=len(local),report_sha256=sha(REPORT),
        host_runtime_status=runtime['status'],test_count=12,production_changed=False,commit=False,published=False)
    write(ROOT/'completion_verification.json',checks)
    # Execution logs may still receive process-exit messages; exclude mutable logs/bundles/cache.
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '.execution' not in p.relative_to(ROOT).parts
           and p.name not in ['artifact_index.json','pipeline_state.json'] and p.suffix!='.log']
    index={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    write(ROOT/'artifact_index.json',dict(files=index,report=dict(path=str(REPORT),sha256=sha(REPORT))))
    for n,c in index.items():assert sha(ROOT/n)==c['sha256']
    print(json.dumps(dict(checks,indexed_files=len(index))))

if __name__=='__main__':main()
