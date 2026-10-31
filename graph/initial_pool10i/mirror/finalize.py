"""Final independent handoff-index check outside the immutable pool.

Runs after coordinator completion, with no simulation or selection. Mutable
transport logs/cursors are explicitly outside the immutable audit index.
"""
import csv
import json
import shutil
import shlex
import sys
import time
from pathlib import Path

from .runner import Runner, REPO
from .verify import read, write, sha, inventory


def main(audit):
    audit=Path(audit).resolve();r=Runner(audit)
    for _ in range(60):
        if read(audit/'progress.json')['status']!='RUNNING':break
        time.sleep(3)
    assert read(audit/'progress.json')['status']=='COMPLETED'
    expected=read(audit/'source_inventory.json')
    assert inventory(Path(r.cfg['source_root']))==expected['files']
    final=read(audit/'hash_final.json')
    assert final['source_inventory_sha256']==sha(audit/'source_inventory.json')
    assert final['counts']==dict(MATCHED=8514,MISSING=0,MISMATCHED=0,EXTRA=0,UNREADABLE=0)
    with (audit/'remote_file_hash_audit.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==8514
    for row in rows:
        info=expected['files'][row['path']]
        assert row['classification']=='MATCHED'
        assert row['remote_sha256']==row['expected_sha256']==info['sha256']
        assert int(row['remote_bytes'])==int(row['expected_bytes'])==info['bytes']
    assert read(audit/'remote_manifest_verification.json')==read(audit/'source_consumer.json')['manifest']
    assert read(audit/'remote_candidate_readback.json')==read(audit/'source_consumer.json')['candidates']
    assert read(audit/'remote_raw_rgbd_readback.json')==read(audit/'source_consumer.json')['rgbd']
    r.source_preservation()
    tests=REPO/'outputs/task10i/stage1_5/mirror_recovery/contract_tests.log'
    assert '\nOK\n' in tests.read_text()
    shutil.copy2(tests,audit/'contract_tests.log')
    code=audit/'code';code.mkdir(exist_ok=True)
    for name in ['runner.py','verify.py','finalize.py','test_verify.py','README.md']:
        shutil.copy2(Path(__file__).with_name(name),code/name)
    receipt=read(audit/'stage2_input_receipt.json')
    assert receipt['total_bytes']==expected['total_bytes'] and receipt['STAGE2_READY'] is True
    text=(audit/'task10i_stage1_mirror_recovery.md').read_text()
    text+='''
## 연결 복구 이력 및 최종 감사

첫 실행 `6e9dd459c0f34d958e60d46c59934bbd`는 제어 파일 전송 중 SSH 단절과 재시도 한도 도달로 BLOCKED 기록을 남겼다. 별도 실행 `7934732624c148ad86a8f4d518bf7c18`에서 동일 Canonical을 다시 사전 검증하고 Direct/Proxy 경로, SSH 연결 재사용, 압축 전송을 적용했다. 네트워크 오류와 중단된 전송 기록은 이전 감사 디렉터리와 transfer_attempts/transfer.log에 보존했다. 원본 Stage1과 실패한 감사 이력을 정상 결과로 바꾸지 않았다.

새 경로/Hash/부분 복구/원자적 감사 기록 계약 테스트 4개가 통과했다. 테스트 초기 macOS /var→/private/var 경로 정규화 기대값을 수정한 뒤 전체 4개를 다시 실행했다. 최종 후보 내용에는 영향이 없다. 원격 최종 파일 감사 CSV의 8,514행을 Source Inventory와 다시 대조했고, Source Inventory 자체의 SHA도 원격 proof와 동일함을 확인했다. 후보/Raw RGB-D/Manifest Readback JSON은 로컬 baseline과 exact 일치한다.

Stage2 Input Receipt와 불변 감사 산출물은 원격 감사 Root의 handoff/에도 전달하고 SHA로 재검증한다. 전송 로그·Cursor·Lock 및 인덱스 자신의 순환 참조는 불변 Artifact Index 범위에서 명시적으로 제외한다. 후보 Canonical 트리에는 어떤 감사 파일도 추가하지 않는다.
'''
    r.write_report(text)
    completion=read(audit/'completion_audit.json')
    completion.update(independent_handoff_check='PASS',source_inventory_identity='PASS',
                      finalizer_source_sha256=sha(Path(__file__)),contract_tests_passed=4)
    write(audit/'completion_audit.json',completion)
    excluded={'artifact_index.json','runner.lock','progress.json','transfer.log','transfer_attempts.json',
              'runner.log','finalizer.log','handoff_delivery_verification.json'}
    files={str(p.relative_to(audit)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in audit.rglob('*') if p.is_file() and p.name not in excluded
           and not {'control','handoff'} & set(p.relative_to(audit).parts)}
    write(audit/'artifact_index.json',dict(files=files,
        scope='Immutable primary audit outputs and frozen code; control/handoff transport replicas excluded',
        excluded_mutable_or_self_referential=sorted(excluded)))
    handoff=audit/'handoff'
    for rel in list(files)+['artifact_index.json']:
        dst=handoff/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(audit/rel,dst)
    r.rsync(handoff,r.cfg['remote_audit']+'/handoff','final_handoff_upload')
    script='''import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]);idx=root/'artifact_index.json'
assert hashlib.sha256(idx.read_bytes()).hexdigest()==sys.argv[2]
d=json.loads(idx.read_text())
for rel,info in d['files'].items():
 p=root/rel
 assert p.stat().st_size==info['bytes'],rel
 assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'],rel
print(json.dumps(dict(status='PASS',files=len(d['files']),artifact_index_sha256=sys.argv[2])))
'''
    out=r.remote(shlex.join(['/venv/main/bin/python','-c',script,r.cfg['remote_audit']+'/handoff',sha(audit/'artifact_index.json')]),'handoff_independent_sha_readback')
    proof=json.loads(out);assert proof['status']=='PASS'
    write(audit/'handoff_delivery_verification.json',proof)
    # This receipt deliberately remains outside the self-referential index.
    assert inventory(Path(r.cfg['source_root']))==expected['files']
    print(json.dumps(dict(status='COMPLETED',STAGE2_READY=True,files=8514,
        bytes=expected['total_bytes'],remote_root=r.cfg['remote_root'],audit=str(audit),
        audit_handoff_verified=proof['files'])),flush=True)


if __name__=='__main__':main(sys.argv[1])
