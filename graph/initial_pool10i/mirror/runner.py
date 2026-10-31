"""Local tmux transfer coordinator. Source is opened read-only throughout.

prepare AUDIT; sync/resume AUDIT; status AUDIT; verify AUDIT; report AUDIT.
Only network/transport errors receive bounded retries. Invariants fail closed.
"""
import argparse
import datetime
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

from graph.initial_pool10i.mirror.verify import sha, read, write, inventory, compare, consumer

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'outputs/task10i/stage1/verified_pool_v2'
REMOTE_BASE = '/workspace/task10i_stage1_20261031'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(audit):
    assert not audit.exists(), 'New preparation requires a new audit directory'
    assert SOURCE.is_dir()
    actual = inventory(SOURCE)
    verification = read(SOURCE / 'candidate_pool_verification.json')
    assert verification['status'] == 'PASS' and len(verification['files']) == 8506
    for path, info in verification['files'].items():
        assert actual[path] == info, ('Canonical source differs from Stage1 index', path)
    extras = sorted(set(actual) - set(verification['files']))
    expected_extras = ['run.lock','progress.json','candidate_pool_verification.json',
        'integration_fixture/candidate_pool_verification.json'] + [
        'experiments/098bd09a011b4c9d99ca881d6a83b760/' + f
        for f in ['metrics.json','metadata.json','config.yaml','manifest.json']]
    assert extras == sorted(expected_extras), extras
    assert len(actual) == 8514 and not any('unreadable' in f for f in actual.values())
    result = consumer(SOURCE, str(SOURCE))
    audit.mkdir(parents=True)
    os.environ['WANDB_MODE'] = 'disabled'
    from utils.experiment_logger import ExperimentLogger
    with ExperimentLogger(team='graph', model='task10i_stage1_5_mirror_recovery', seed=0,
        job_type='data_delivery', root=audit / 'experiments', config=dict(source=str(SOURCE),
        original_stage1_run_id='098bd09a011b4c9d99ca881d6a83b760', instance=57987772,
        simulator_executed=False, stage2_selection=False)) as logger:
        run_id = logger.run_id
        logger.log(dict(source_files=8514, eligible=300))
    config = dict(run_id=run_id, source_root=str(SOURCE), remote_base=REMOTE_BASE,
        remote_root=REMOTE_BASE+'/verified_pool_v2',
        staging_root=REMOTE_BASE+'/verified_pool_v2_staging_'+run_id,
        remote_audit='/workspace/task10i_stage1_5_mirror_'+run_id,
        host='104.233.209.157', port=13159, instance=57987772,
        routes=[dict(host='104.233.209.157',port=13159),dict(host='ssh4.vast.ai',port=18984)],
        max_retries=4, bandwidth_kib_s=4096,
        source_code={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),Path(__file__).with_name('verify.py')]})
    write(audit/'config.json', config)
    write(audit/'source_inventory.json', dict(status='PASS', source_root=str(SOURCE),
        stage1_index_count=8506, complete_tree_count=len(actual), total_bytes=sum(f['bytes'] for f in actual.values()),
        additional_files=extras, scope_explanation='8506 historical indexed scientific/report files plus 8 existing bookkeeping/self-index files; mirror covers all 8514 existing files byte-for-byte.',
        stage1_index_sha256=sha(SOURCE/'candidate_pool_verification.json'), files=actual))
    write(audit/'source_consumer.json', result)
    write(audit/'transfer_attempts.json', [])
    write(audit/'transfer_plan.json', dict(direction='LOCAL_CANONICAL_TO_REMOTE_STAGING',
        source=str(SOURCE), destination=config['staging_root'], canonical_destination=config['remote_root'],
        isolated_audit=config['remote_audit'], use_delete=False, source_read_only=True,
        partials_outside_candidate_tree=True, reuse='Only original partial files matching path,size,SHA256; copy, no hardlinks',
        retry_count=4, retry_delays_seconds=[3,6,9], bwlimit_kib_s=4096,
        stages=['SOURCE_AUDIT','REMOTE_PARTIAL_AUDIT','TRANSFER_RECOVERY','FULL_STAGING_HASH','ATOMIC_PROMOTION','FULL_FINAL_HASH','CONSUMER_READBACK','SOURCE_PRESERVATION','REPORT']))
    write(audit/'progress.json',dict(status='PREPARED',stage='SOURCE_AUDIT',run_id=run_id,stage2_ready=False))
    print(json.dumps(dict(status='PREPARED',audit=str(audit),run_id=run_id,files=len(actual))))


class Runner:
    def __init__(self, audit):
        self.audit = audit.resolve(); self.cfg = read(audit/'config.json')
        for p,h in self.cfg['source_code'].items():
            assert sha(REPO/p) == h, 'Source changed; cannot resume this audit'
        self.ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=10',
                    '-o','Compression=yes','-o','IPQoS=none','-o','ControlMaster=auto','-o','ControlPersist=60',
                    '-o','ControlPath=/tmp/10i-mirror-'+self.cfg['run_id'][:8]+'-%C',
                    '-o','ServerAliveCountMax=3','-p',str(self.cfg['port']), 'root@'+self.cfg['host']]
        self.log = self.audit/'transfer.log'

    def step(self, name):
        write(self.audit/'progress.json',dict(status='RUNNING',stage=name,run_id=self.cfg['run_id'],stage2_ready=False,updated=now()))
        with self.log.open('a') as f:
            f.write(now()+' '+name+'\n')
        print(now(),name,flush=True)

    def call(self, cmd, label, timeout=600):
        for attempt in range(self.cfg['max_retries']):
            route=self.cfg['routes'][attempt % len(self.cfg['routes'])]
            routed_ssh=list(self.ssh)
            routed_ssh[-2]=str(route['port']);routed_ssh[-1]='root@'+route['host']
            actual_cmd=list(cmd)
            if cmd[0]=='ssh':actual_cmd=routed_ssh+cmd[len(self.ssh):]
            elif cmd[0]=='rsync':
                ei=actual_cmd.index('-e')+1;actual_cmd[ei]=shlex.join(routed_ssh[:-1])
                actual_cmd[-1]='root@'+route['host']+':'+actual_cmd[-1].split(':',1)[1]
            start=now()
            try:
                p=subprocess.run(actual_cmd,capture_output=True,text=True,timeout=timeout)
                code,stdout,stderr=p.returncode,p.stdout,p.stderr
            except subprocess.TimeoutExpired as e:
                code,stdout,stderr=124,'','transport timeout'
            history=read(self.audit/'transfer_attempts.json')
            row=dict(index=len(history),label=label,retry=attempt,start=start,end=now(),exit_code=code,
                     route=route,
                     stdout_bytes=len(stdout.encode()),stderr_bytes=len(stderr.encode()))
            sent=re.search(r'Total transferred file size:\s*([\d,]+)',stdout)
            if sent: row['transferred_file_bytes']=int(sent[1].replace(',',''))
            history.append(row);write(self.audit/'transfer_attempts.json',history)
            with self.log.open('a') as f:
                f.write(json.dumps(row)+'\n'+stdout+stderr+'\n')
            if code == 0: return stdout
            # SSH/rsync transport failures only; do not retry invariant failures.
            if code not in [10,12,23,30,35,124,255] or attempt+1 == self.cfg['max_retries']:
                raise RuntimeError(f'{label} failed, exit={code}; see transfer.log')
            time.sleep(10*(attempt+1))

    def remote(self, command, label):
        return self.call(self.ssh+[command],label)

    def fetch(self, name):
        text=self.remote('cat '+shlex.quote(self.cfg['remote_audit']+'/'+name),'readback:'+name)
        obj=json.loads(text);write(self.audit/name,obj);return obj

    def action(self, name):
        if (self.audit/(name+'.done.json')).exists() and read(self.audit/(name+'.done.json'))['status']=='PASS':
            return
        session='10i-mirror-'+self.cfg['run_id'][:8]+'-'+name
        remote_audit=self.cfg['remote_audit']
        command=shlex.join(['/venv/main/bin/python','-u',remote_audit+'/verify.py',name,remote_audit])+' >> '+shlex.quote(remote_audit+'/'+name+'.log')+' 2>&1'
        # Completed proof survives a disconnected SSH; never run action twice.
        done=shlex.quote(remote_audit+'/'+name+'.done.json')
        launch='if test -f '+done+'; then true; elif tmux has-session -t '+shlex.quote(session)+' 2>/dev/null; then true; else tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote(command)+'; fi'
        self.remote(launch,'launch:'+name)
        for _ in range(100):
            value=self.remote('if test -f '+done+'; then cat '+done+'; else printf RUNNING; fi','poll:'+name)
            if value.strip()!='RUNNING':
                result=json.loads(value);write(self.audit/(name+'.done.json'),result)
                assert result['status']=='PASS', result
                return
            time.sleep(3)
        raise RuntimeError('Remote action deadline: '+name)

    def source_preservation(self):
        expected=read(self.audit/'source_inventory.json')['files']
        actual=inventory(Path(self.cfg['source_root']))
        assert actual == expected, 'Canonical source changed'
        write(self.audit/'source_preservation_audit.json',dict(status='PASS',files=len(actual),
            total_bytes=sum(f['bytes'] for f in actual.values()),all_paths_sizes_sha256_unchanged=True,
            source_inventory_sha256=sha(self.audit/'source_inventory.json'),checked=now()))

    def sync(self):
        self.step('SOURCE_RECHECK'); self.source_preservation()
        if (self.audit/'promote.done.json').exists():
            self.verify();return
        self.step('REMOTE_CONTROL_BOOTSTRAP')
        self.remote('test -d '+shlex.quote(self.cfg['remote_base'])+' && mkdir -p '+shlex.quote(self.cfg['remote_audit']), 'bootstrap')
        control=self.audit/'control';control.mkdir(exist_ok=True)
        for name in ['config.json','source_inventory.json','source_consumer.json']:
            shutil.copy2(self.audit/name,control/name)
        shutil.copy2(Path(__file__).with_name('verify.py'),control/'verify.py')
        self.rsync(control, self.cfg['remote_audit'], 'control_upload')
        self.step('REMOTE_PARTIAL_AUDIT');self.action('inspect');self.fetch('remote_partial_inventory.json')
        self.fetch('protected_run_v3_inventory.json')
        self.step('REUSE_VERIFIED_PARTIAL');self.action('seed');self.fetch('reuse_audit.json')
        self.step('TRANSFER_RECOVERY')
        self.rsync(Path(self.cfg['source_root']),self.cfg['staging_root'],'canonical_transfer')
        self.verify()

    def rsync(self, source, destination, label):
        cmd=['rsync','-rtz','--checksum','--partial','--partial-dir='+self.cfg['remote_audit']+'/rsync-partials',
             '--timeout=45','--stats','--itemize-changes','--bwlimit='+str(self.cfg['bandwidth_kib_s']),
             '-e',shlex.join(self.ssh[:-1]),str(source)+'/',self.ssh[-1]+':'+destination+'/']
        self.call(cmd,label,timeout=1800)

    def verify(self):
        self.step('FULL_STAGING_HASH');self.action('hash_staging');self.fetch('hash_staging.json')
        self.step('CANONICAL_PROMOTION');self.action('promote');self.fetch('promotion.json')
        self.step('FULL_REMOTE_HASH');self.action('hash_final');self.fetch('hash_final.json')
        self.step('CONSUMER_READBACK');self.action('consumer')
        for name in ['remote_manifest_verification.json','remote_candidate_readback.json','remote_raw_rgbd_readback.json','frozen_source_readback.json']:
            self.fetch(name)
        csvtext=self.remote('cat '+shlex.quote(self.cfg['remote_audit']+'/remote_file_hash_audit.csv'),'file_hash_csv_readback')
        (self.audit/'remote_file_hash_audit.csv').write_text(csvtext)
        self.step('SOURCE_POSTCHECK');self.source_preservation()
        self.report()

    def report(self):
        inv=read(self.audit/'source_inventory.json');cfg=self.cfg
        final=read(self.audit/'hash_final.json');promotion=read(self.audit/'promotion.json')
        assert final['status']=='PASS' and promotion['status']=='PASS'
        assert read(self.audit/'source_preservation_audit.json')['status']=='PASS'
        assert read(self.audit/'remote_candidate_readback.json')['count']==300
        assert read(self.audit/'remote_raw_rgbd_readback.json')['count']==300
        mapping=dict(status='PASS',source_root=cfg['source_root'],remote_root=cfg['remote_root'],
            candidate_paths='All export data paths are relative to pool root; no JSON changes required',
            provenance_absolute_paths='Local verification_root/generation_root are historical provenance, not data loader paths',
            adapter_required=False,consumer_candidates_resolved=300)
        write(self.audit/'remote_path_mapping.json',mapping)
        verification=dict(status='PASS',run_id=cfg['run_id'],files=len(inv['files']),total_bytes=inv['total_bytes'],
            source_inventory_sha256=sha(self.audit/'source_inventory.json'),**final['counts'],
            consumer_readback='PASS',source_preserved=True,partial_preserved=True,run_v3_preserved=True,
            stage2_selection_executed=False,simulator_executed=False,STAGE2_READY=True)
        write(self.audit/'remote_mirror_verification.json',verification)
        receipt=dict(status='COMPLETED',run_id=cfg['run_id'],canonical_source_root=cfg['source_root'],
            verified_remote_root=cfg['remote_root'],instance_id=57987772,total_files=len(inv['files']),
            total_bytes=inv['total_bytes'],full_hash_verification_status='PASS',eligible_candidate_count=300,
            candidate_pool_manifest_sha256=sha(SOURCE/'candidate_pool_manifest.json'),
            original_stage1_run_id='098bd09a011b4c9d99ca881d6a83b760',
            visibility_protocol_identity=read(SOURCE/'visibility_protocol_identity.json'),
            path_mapping_sha256=sha(self.audit/'remote_path_mapping.json'),consumer_readback_status='PASS',
            STAGE2_READY=True,stage2_selection='NOT_EXECUTED',expert_collection='NOT_EXECUTED',
            remote_audit_root=cfg['remote_audit'])
        write(self.audit/'stage2_input_receipt.json',receipt)
        (self.audit/'verified_remote_pool_path.txt').write_text(cfg['remote_root']+'\n')
        attempts=read(self.audit/'transfer_attempts.json')
        write(self.audit/'transfer_summary.json',dict(status='PASS',direction='LOCAL_TO_REMOTE',
            source_files=len(inv['files']),source_bytes=inv['total_bytes'],reused=read(self.audit/'reuse_audit.json')['reused'],
            commands=len(attempts),failed_transport_attempts=sum(a['exit_code']!=0 for a in attempts),
            rsync_transferred_file_bytes=sum(a.get('transferred_file_bytes',0) for a in attempts),
            transfer_bytes_note='rsync transferred file sizes; transport protocol overhead is recorded in transfer.log',exit_code=0))
        partial=read(self.audit/'remote_partial_inventory.json')
        text=f'''# Task10-I Stage1.5 — Canonical Candidate Pool Remote Mirror Recovery

상태: **COMPLETED**, **STAGE2_READY=true**. 별도 실행 ID `{cfg['run_id']}`. Stage1 원본 UUID `098bd09a011b4c9d99ca881d6a83b760`는 변경하지 않았다.

로컬 Canonical: `{cfg['source_root']}`

검증된 원격 후보 풀: **`{cfg['remote_root']}`**, Vast.ai **57987772**.

원격 감사 Root: `{cfg['remote_audit']}`.

## 원본 및 전체 파일 범위

기존 Stage1 인덱스의 8,506개 파일을 원본 크기·SHA256과 전수 대조했다. 실제 디렉터리에는 인덱스 자체, progress, lock, ExperimentLogger 기록 및 fixture 자체 검증 파일 8개가 추가로 존재한다. 추가 목록은 [Source Inventory](source_inventory.json)에 있다. 집계 범위 차이를 명시하고 **8,514개 전체**, **{inv['total_bytes']:,} bytes**를 바이트 그대로 전달했다. Canonical JSON/CSV/NPZ를 수정·재생성·재압축하지 않았다.

전송 전후 로컬 전체 경로·크기·SHA256이 동일하다. [원본 보존 감사](source_preservation_audit.json).

## 원격 복구

전송 전 Partial 분류: `{json.dumps(partial['counts'])}`. 정상 파일은 SHA 확인 후 별도 Staging에 복사했으며 hardlink를 사용하지 않았다. 나머지는 로컬 Canonical에서 checksum rsync로 보냈다. Retry는 전송 오류에만 최대 4회이며 Partial은 Canonical 밖 감사 Root에 둔다. 로컬 tmux runner가 전 단계를 관리하고 원격 Hash/Readback은 독립 tmux에서 실행했다.

Staging 전수 Hash PASS 후 기존 Partial을 `{promotion['partial_archive']}`에 보존하고 같은 파일시스템의 rename으로 승격했다. 승격 후 최종 경로를 다시 전수 Hash 검사했다. 기존 `run_v3` 전체 byte inventory가 동일하며 BLOCKED/GL 진단을 보존했다.

## 독립 원격 검증

경로·크기·SHA256 **8,514/8,514 MATCHED**, MISSING=0, MISMATCHED=0, EXTRA=0, UNREADABLE=0. [전체 파일 감사](remote_file_hash_audit.csv), [최종 검증](remote_mirror_verification.json).

926행 CSV와 aggregate NPZ의 모든 배열을 실제로 읽고 shape/dtype/finite를 확인했다. Eligible 300개 전부 ID·seed·UNASSIGNED·Physical VALID·Replay PASS·기존 Visibility PASS를 연결했다. 모든 초기 NPZ 배열이 aggregate의 동일 후보와 exact 일치하고, 로컬 Readback fingerprint와 원격 결과가 exact 일치했다. 각 후보의 Descriptor JSON/CSV/Export, Pose/Quaternion/Shape, Peg–TCP/Peg–Hole 관계, 투영·Boundary Margin을 대조했다. [후보 Readback](remote_candidate_readback.json).

300개 원본 RGB uint8 128×128×3 및 Depth int16 128×128 mm를 initial.npz의 rgb/depth_raw_mm key에서 읽었다. 비양수 Depth는 원본 그대로 보존했다. Gallery의 annotated_rgb_preview를 원본으로 사용하지 않았다. GT Evidence NPZ도 읽고 로컬 배열 fingerprint와 비교했다. Snapshot은 바이트 Hash만 검증하며 pickle 실행이나 Simulator Restore는 수행하지 않았다. [RGB-D Readback](remote_raw_rgbd_readback.json).

Export 데이터 경로는 모두 Pool Root 기준 상대 경로이므로 원격에서 직접 해결된다. JSON 내부의 macOS verification_root/generation_root는 과거 실행 provenance이며 데이터 로더 입력 경로로 사용하지 않는다. [Path Mapping](remote_path_mapping.json). 원본 Source/Runtime identity를 보존했고 기존 원격 frozen source 파일도 전수 대조했다.

## 기존 실험 보호 및 범위

Task10-H 상태는 전송 전 읽기 전용으로 조사했다. 프로세스 종료·세션 종료·Source/Checkpoint/Dataset 변경을 수행하지 않았다. 전송은 4MiB/s로 제한했다. Simulator Reset/Render, 가시성 Gate 재실행, Expert 수집, Stage2 Diversity Selection, 모델 적재·학습·평가는 실행하지 않았다. GPU/PhysX backend 실행 검증은 이번 단계 범위에 없다. 기존 Stage1 실험 결과를 재생산했다고 주장하지 않는다.

## 다음 단계 인계

300개 후보 모두 UNASSIGNED이며 Expert NOT_TESTED다. **[stage2_input_receipt.json](stage2_input_receipt.json)** 및 **[검증된 원격 경로](verified_remote_pool_path.txt)**를 사용한다. Stage2를 시작하지 않았으며 이번 작업은 전달·무결성·읽기 검증에서 종료한다. 실행/재개 CLI와 코드 identity는 transfer_plan/config에 보존했다.
'''
        self.write_report(text)
        write(self.audit/'completion_audit.json',dict(status='COMPLETED',STAGE2_READY=True,
            requirements=dict(local_source_hash=True,staging_hash=True,atomic_promotion=True,final_remote_hash=True,
                all_300_consumer=True,raw_rgbd=True,path_mapping=True,source_preservation=True,protected_runs=True,
                stage2_not_executed=True),completed=now()))
        # Audit manifests live outside the byte-identical candidate tree.
        files={str(p.relative_to(self.audit)):dict(bytes=p.stat().st_size,sha256=sha(p))
               for p in self.audit.rglob('*') if p.is_file() and p.name not in ['artifact_index.json','runner.lock','progress.json','transfer.log','transfer_attempts.json'] and 'control' not in p.parts}
        write(self.audit/'artifact_index.json',dict(files=files,scope='Immutable audit outputs; excludes mutable transport log, cursor, lock, self-index'))
        handoff=self.audit/'handoff';handoff.mkdir(exist_ok=True)
        for p in self.audit.iterdir():
            if p.is_file() and p.suffix in ['.json','.csv','.md','.html','.txt'] and p.name!='runner.lock':
                shutil.copy2(p,handoff/p.name)
        self.rsync(handoff,cfg['remote_audit']+'/handoff','handoff_upload')
        write(self.audit/'progress.json',dict(status='COMPLETED',stage='STOP_BEFORE_STAGE2',run_id=cfg['run_id'],STAGE2_READY=True))
        print('COMPLETED',cfg['remote_root'],flush=True)

    def write_report(self, text):
        import markdown
        report=self.audit/'task10i_stage1_mirror_recovery.md'
        report.write_text(text)
        style='<style>body{max-width:1100px;margin:40px auto;font:16px/1.7 sans-serif}pre{white-space:pre-wrap}code{overflow-wrap:anywhere}</style>'
        (self.audit/'task10i_stage1_mirror_recovery.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I Stage1.5</title>'+style+markdown.markdown(text,extensions=['tables','fenced_code']))
        docs=REPO/'docs/visual-graph-integration'
        prefix=os.path.relpath(self.audit,docs)+'/'
        doc=re.sub(r'\]\(([^)]+)\)',lambda m:']('+prefix+m[1]+')',text)
        (docs/'task10i_stage1_mirror_recovery.md').write_text(doc)
        (docs/'task10i_stage1_mirror_recovery.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I Stage1.5</title>'+style+markdown.markdown(doc,extensions=['tables','fenced_code']))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','status','sync','resume','verify','report']);p.add_argument('audit',type=Path);a=p.parse_args();audit=a.audit.resolve()
    if a.action=='prepare': prepare(audit);return
    if a.action=='status': print(json.dumps(read(audit/'progress.json'),indent=2));return
    lock=(audit/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        runner=Runner(audit)
        if a.action in ['sync','resume']:
            if (audit/'progress.json').exists() and read(audit/'progress.json')['status']=='COMPLETED':
                print('Already completed; use verify for explicit re-audit');return
            runner.sync()
        elif a.action=='verify': runner.verify()
        else: runner.report()
    except BaseException:
        write(audit/'progress.json',dict(status='BLOCKED',STAGE2_READY=False,traceback=traceback.format_exc()))
        write(audit/'completion_audit.json',dict(status='BLOCKED',STAGE2_READY=False,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
