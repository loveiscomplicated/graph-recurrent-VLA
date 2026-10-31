"""Local tmux delivery observer; never starts/restarts remote simulator work."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import traceback

from .core import read,write,sha

REPO=Path(__file__).resolve().parents[2]
REMOTE='/workspace/task10i_stage3_20261031_v5_finalization'
COLLECTION_REMOTE='/workspace/task10i_stage3_20261031_v3'
LOCAL=REPO/'outputs/task10i/stage3'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
    '-o','IPQoS=none','-o','Compression=yes',
    '-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/task10i-stage3-delivery-%C','-p','13159','root@104.233.209.157']

def call(args,timeout=180):
    last=None
    for host,port in [('104.233.209.157','13159'),('ssh4.vast.ai','18984')]:
        actual=list(args)
        if actual[0]=='ssh':
            pi=actual.index('-p');actual[pi+1]=port;actual[pi+2]='root@'+host
        elif actual[0]=='rsync':
            ei=actual.index('-e')+1;shell=shlex.split(actual[ei]);pi=shell.index('-p');shell[pi+1]=port
            actual[ei]=shlex.join(shell)
            for i,arg in enumerate(actual):
                if arg.startswith(SSH[-1]+':'):actual[i]='root@'+host+':'+arg.split(':',1)[1]
        try:return subprocess.run(actual,capture_output=True,text=True,timeout=timeout,check=True).stdout
        except subprocess.CalledProcessError as e:
            last=e
            if e.returncode not in (10,12,23,30,35,255):raise
        except subprocess.TimeoutExpired as e:last=e
    raise last

def status():
    script='''import json,pathlib
r=pathlib.Path("REMOTE_ROOT/run")
s=json.load(open(r/"pipeline_state.json"));p=pathlib.Path("/proc")/str(s["pid"])/"cmdline"
s["coordinator_live"]=p.exists() and any(x in p.read_bytes() for x in (b"graph.task10i_stage3.run_pipeline",b"graph.task10i_stage3.finalize_existing")) and str(r).encode() in p.read_bytes()
s["run_identity"]=json.load(open(r/"run_identity.json"))
parents=[]
for f in (r/"parents").glob("*/parent_result.json"):parents.append(json.load(open(f)))
s["parent_results"]=parents
s["committed_receipts"]=len(list((r/"parents").glob("*/*/receipt.json")))
if (r/"engineering_gates.json").exists():s["engineering_gates"]=json.load(open(r/"engineering_gates.json"))
print(json.dumps(s))
'''.replace('REMOTE_ROOT',REMOTE)
    return json.loads(call(SSH+['/venv/main/bin/python -c '+shlex.quote(script)],timeout=60))

def sync(terminal=False):
    excludes=[] if terminal else ['--exclude=*.partial/','--exclude=interrupted/','--exclude=engineering_*/']
    target=LOCAL/'run';target.mkdir(parents=True,exist_ok=True)
    previous=LOCAL/'collection_run_v3'
    reuse=['--link-dest='+str(previous)] if previous.exists() else []
    call(['rsync','-azL','--bwlimit=8192','--timeout=60','-e',shlex.join(SSH[:-1]),*reuse,
        '--exclude=*.tmp','--exclude=owner.lock','--exclude=run.lock',*excludes,
        SSH[-1]+':'+REMOTE+'/run/',str(target)+'/'],timeout=1200 if terminal else 240)

def progress_report(s):
    dest=REPO/'docs/visual-graph-integration/10i_stage3_expert_dataset_collection.md'
    n=len(s['parent_results']);passed=sum(p['status']=='PASS' for p in s['parent_results'])
    text=f'''# Task10-I Stage3 Expert Dataset Collection

상태: **RUNNING / 최종 완료 아님**. ExperimentLogger UUID `{s['run_identity']['run_id']}`.

현재 자동 Stage: `{s['stage']}`. 원격 Orchestrator PID `{s['pid']}`, 마지막 확인 시 실제 생존 여부 `{s['coordinator_live']}`. 실행 root `{REMOTE}/run`, tmux `task10i-stage3`.

본 수집 Parent 결과 {n}개 중 전체 요구 Branch PASS {passed}개, commit된 Raw Episode {s['committed_receipts']}개다. Engineering 자료는 별도이며 최종 연구 데이터셋에 자동 포함하지 않는다. R96/272, D96/272 및 Shared Validation16/48의 최종 Count·물리·Action/GT 전수 감사와 Stage4 Receipt 완료는 아직 입증되지 않았다.

독립 Python Orchestrator가 입력 검증, Contract/Smoke/실제 Crash-resume/Miniature/1·2·3 Worker 검증, 본 수집, 동결 Reserve 처리, Dataset 조립, 전수 감사, Media 및 최종 보고서까지 담당한다. 원본 Stage1/2 및 기존 Raw/GT/Checkpoint는 읽기 전용이다. Stage4/5, 정책 평가, Git Commit과 외부 게시는 실행하지 않는다.

이 문서는 실행 상태를 기록한 임시 보고서다. 서버가 최종 보고서를 생성하면 **전체 본문과 Static HTML**을 자동으로 이 문서 위치에 전달한다. [최근 실제 상태](../../outputs/task10i/stage3/delivery/latest_state.json), [전송 기록](../../outputs/task10i/stage3/delivery/status.json)을 확인한다.
'''
    dest.write_text(text)

def final_delivery(s):
    if s['status']=='COMPLETED':
        # Audit runs outside the immutable execution source/output tree.
        auditor=Path(__file__).with_name('audit_handoff.py')
        replay=Path(__file__).with_name('native_initial_replay.py')
        remote_audit=REMOTE+'_independent_audit'
        call(SSH+['mkdir -p '+shlex.quote(remote_audit)])
        for source in (auditor,replay):
            call(['rsync','-az','--timeout=60','-e',shlex.join(SSH[:-1]),str(source),SSH[-1]+':'+remote_audit+'/'+source.name])
        command=shlex.join(['/venv/main/bin/python','-B',remote_audit+'/native_initial_replay.py',REMOTE+'/run',remote_audit+'/native_initial_replay_audit.json'])
        call(SSH+[command],timeout=1200)
        replay_result=json.loads(call(SSH+['cat '+shlex.quote(remote_audit+'/native_initial_replay_audit.json')]))
        assert replay_result['status']=='PASS' and replay_result['source_sha256']==sha(replay)
        write(LOCAL/'native_initial_replay_audit.json',replay_result)
        command=shlex.join(['/venv/main/bin/python','-B',remote_audit+'/audit_handoff.py',REMOTE+'/run',remote_audit+'/independent_handoff_audit.json'])
        call(SSH+[command],timeout=1200)
        audit_text=call(SSH+['cat '+shlex.quote(remote_audit+'/independent_handoff_audit.json')])
        audit=json.loads(audit_text);assert audit['status']=='PASS' and audit['source_sha256']==sha(auditor)
        write(LOCAL/'independent_handoff_audit.json',audit)
    sync(terminal=True)
    source=LOCAL/'execution_source';source.mkdir(exist_ok=True)
    call(['rsync','-az','--timeout=60','-e',shlex.join(SSH[:-1]),'--exclude=__pycache__',SSH[-1]+':'+REMOTE+'/source/',str(source)+'/'],timeout=300)
    collection_source=LOCAL/'collection_source';collection_source.mkdir(exist_ok=True)
    call(['rsync','-az','--timeout=60','-e',shlex.join(SSH[:-1]),'--exclude=__pycache__',SSH[-1]+':'+COLLECTION_REMOTE+'/source/',str(collection_source)+'/'],timeout=300)
    root=LOCAL/'run'
    index=read(root/'artifact_index.json') if (root/'artifact_index.json').exists() else {'files':{}}
    index_mismatches=[]
    for name,entry in index['files'].items():
        p=root/name
        if not p.exists() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:index_mismatches.append(name)
    if s['status']=='COMPLETED':assert not index_mismatches,index_mismatches
    for p in root.glob('parents/*/*/receipt.json'):
        receipt=read(p)
        for name,h in receipt['hashes'].items():assert sha(p.parent/name)==h,str(p.parent/name)
    if s['status']=='COMPLETED':
        audit=read(LOCAL/'independent_handoff_audit.json')
        for name,entry in {**audit['auxiliary_parent_files'],**audit['failed_attempt_files']}.items():
            p=root/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
    for name,h in read(root/'collector_source_identity.json').items():assert sha(collection_source/name)==h,name
    if (root/'finalizer_source_identity.json').exists():
        for name,h in read(root/'finalizer_source_identity.json').items():assert sha(source/name)==h,name
    if s['status']=='COMPLETED':
        assert read(root/'completion_audit.json')['status']=='PASS'
        assert read(root/'stage4_raw_dataset_receipt.json')['STAGE4_READY']
        assert read(root/'verification.json')['status']=='PASS'
    docs=REPO/'docs/visual-graph-integration';name='10i_stage3_expert_dataset_collection'
    text=(root/(name+'.md')).read_text()
    if s['status']=='COMPLETED':
        # Independently derive role-level attrition and expose all frame counts
        # in the delivered full report, while preserving the remote report bytes.
        stage2=REPO/'outputs/task10i/stage2/run'
        receipt=read(root/'stage3_input_identity.json')['receipt']
        assert sha(stage2/'selection_identity.json')==receipt['files']['selection_identity.json']
        selection=read(stage2/'selection_identity.json')
        assert sha(stage2/'sampling_trace.json')==selection['files']['sampling_trace.json']
        common=set(read(stage2/'sampling_trace.json')['common'][:44])
        plan=read(root/'parent_branch_collection_plan.json')
        with (root/'expert_attempts.csv').open() as f:attempts={r['parent_id']:r for r in csv.DictReader(f)}
        groups={k:[] for k in ('Common44','R-exclusive','D-exclusive','Shared tail','Validation')}
        for cid,p in plan.items():
            ds={a['dataset'] for a in p['allocations']}
            group='Common44' if cid in common else 'Validation' if ds=={'validation'} else 'R-exclusive' if ds=={'R'} else 'D-exclusive' if ds=={'D'} else 'Shared tail'
            groups[group].append(attempts[cid])
        text+='\n## 전달 시 독립 집계 — 원래90 Parent의 실제 역할별 결과\n\n'
        text+='| 원래 역할 | Parent | Full-task 성공 | 전체 요구 Branch PASS |\n|---|---:|---:|---:|\n'
        for group,rows in groups.items():text+=f"| {group} | {len(rows)} | {sum(r['full_task_success']=='True' for r in rows)} | {sum(r['parent_status']=='PASS' for r in rows)} |\n"
        text+='\n이 집계는 동결 Stage2 Selection Hash와 실제 `expert_attempts.csv`에서 독립 계산했다. Full-task 성공과 모든 필수 Corrective Branch 승인 여부를 구분한다. 원래 원격 보고서는 출력 root에 그대로 보존했다.\n\n'
        summary=read(root/'dataset_completion_summary.json')['datasets']
        text+='| Dataset | Full-task frame | Prefix | Disturbance | Recovery | BC | Geometry eligible |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for ds,row in summary.items():
            text+=f"| {ds} | {row['full_task_action_frames']} | {row['corrective_prefix_frames']} | {row['disturbance_frames']} | {row['recovery_frames']} | {row['BC_supervised_frames']} | {row['geometry_eligible_frames']} |\n"
        text+='\n실제 Frame 수를 맞추기 위한 절단이나 재선정은 수행하지 않았다. Geometry eligible은 정합된 원자료의 이용 가능성을 뜻하며, 이번 단계에서 Geometry Loss를 계산한 결과가 아니다.\n'
        with (root/'replacement_coverage_comparison.csv').open() as f:coverage=list(csv.DictReader(f))
        text+='\n| 평가 모집단 | Dataset / 선정 상태 | Mean nearest distance | p90 | p95 | Maximum |\n|---|---|---:|---:|---:|---:|\n'
        for row in coverage:
            text+=f"| {row['cohort']} | {row['dataset']} | {float(row['mean']):.6f} | {float(row['p90']):.6f} | {float(row['p95']):.6f} | {float(row['maximum']):.6f} |\n"
        text+='\n위 수치는 동결 거리·Scale을 그대로 사용한다. 최초 선정, Full-task 성공 조건부 선정, Reserve 적용 이후 최종 선정을 모두 보존했다. Selection/Weight/Greedy는 다시 계산하지 않았다.\n'
        failures=read(LOCAL/'independent_handoff_audit.json')['failed_attempt_diagnostics']
        if failures:
            text+='\n실패 원본의 실제 Trace에서 확인한 내용:\n\n'
            for failure in failures:
                transition=failure.get('final_transition') or {}
                text+=f"- `{failure['parent_id']}`: `{failure['reason']}`; 마지막 Phase `{failure.get('last_phase','not recorded')}`, 기록된 Action {failure.get('actions_recorded','not recorded')}개, truncated={transition.get('truncated','not recorded')}, official success={transition.get('success','not recorded')}. Phase별 기록 `{failure.get('phase_counts',{})}`. 실패 원본은 보존하고 최종 학습 Manifest에서 제외했다.\n"
        text+='\n[별도 구현의 독립 인계 감사](../independent_handoff_audit.json)는 원래 Template·Common44·Split·모든 원자료와 GT38 재조립·Controller Trace·실제 교란 변위·안정화·Frame 수 및 SHA를 다시 확인했다. 이 감사는 Collector/Assembler 구현을 import하지 않는다.\n'
        text+='\n[원래 Stage1 환경 생성 코드의 독립 초기 재생](../native_initial_replay_audit.json)은 최종 신규 Parent 전부에서 Main/Wrist RGB·Depth·Calibration을 저장된 Stage3 T0와 exact 비교했다. Stage1의 역사적 NPZ에는 Main만 저장되어 있으므로, Wrist는 역사적 파일 비교가 아니라 동일 Native Runtime의 독립 재생 검증이다. 이 검사에서는 Action·Perception·Graph·Policy를 실행하지 않았다.\n'
    prefix='../../outputs/task10i/stage3/run/'
    text=re.sub(r'\]\((?!https?://|#|/)([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',text)
    (docs/(name+'.md')).write_text(text)
    body=(root/(name+'.html')).read_text()
    body=re.sub(r'(href|src)="(?!https?://|#|/)([^"]+)"',lambda m:m.group(1)+'="'+prefix+m.group(2)+'"',body)
    if s['status']=='COMPLETED':
        import markdown
        body='<!doctype html><html><meta charset="utf-8"><title>Task10-I Stage3</title><style>body{max-width:1100px;margin:36px auto;font:16px/1.6 system-ui;padding:16px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}img{max-width:100%}</style>'+markdown.markdown(text,extensions=['tables','fenced_code'])+'</html>'
    (docs/(name+'.html')).write_text(body)
    for match in re.finditer(r'\]\(([^)]+)\)',text):
        target=match.group(1)
        if not target.startswith(('http:','https:','#','/')):assert (docs/target).exists(),target
    write(LOCAL/'delivery/delivery_verification.json',dict(status='PASS',remote_status=s['status'],
        indexed_artifacts=len(index['files']),committed_episodes=len(list(root.glob('parents/*/*/receipt.json'))),
        finalization_index_mismatches=index_mismatches,
        raw_GT_new_store_hash_verified=True,execution_source_SHA_verified=True,
        historical_sources='Original remote paths remain authoritative; remote full readback and preservation audit retained.',
        report=str(docs/(name+'.md')),report_sha256=sha(docs/(name+'.md')),html_sha256=sha(docs/(name+'.html')),
        delivery_source_sha256=sha(Path(__file__)),external_publication=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    delivery=LOCAL/'delivery';delivery.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+24*3600;last_sync=0;errors=0
    while time.monotonic()<deadline:
        try:
            s=status();write(delivery/'latest_state.json',s)
            if s['status'] in ('COMPLETED','INCOMPLETE','BLOCKED'):
                final_delivery(s);write(delivery/'status.json',dict(status='DELIVERED',remote_status=s['status'],at=time.time()));return
            progress_report(s)
            if time.monotonic()-last_sync>120:
                sync();last_sync=time.monotonic()
            write(delivery/'status.json',dict(status='OBSERVING',remote_stage=s['stage'],coordinator_live=s['coordinator_live'],at=time.time()))
            errors=0
        except Exception as e:
            errors+=1
            write(delivery/'last_error.json',dict(error=str(e),traceback=traceback.format_exc(),consecutive=errors,at=time.time()))
            if args.once:raise
            # Observation errors never restart, stop or reclassify the remote run.
        if args.once:return
        time.sleep(30)
    write(delivery/'status.json',dict(status='OBSERVATION_DEADLINE',remote_run_not_modified=True))

if __name__=='__main__':main()
