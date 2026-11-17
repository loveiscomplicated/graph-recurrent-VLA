# Task10-F1 직접 감시·연속 실행 기록 — 2026-10-30

현재 실행 상태는 **RECACHE_VERIFIED / WAITING_PRODUCER_APPROVAL_EVIDENCE**다. 연구 목표인 본 GNN 사전학습과 독립 성능 평가는 완료되지 않았다. 기존 준비 보고서와 실패 기록은 보존한다.

## 실제 완료한 작업

- Vast.ai57987772의 기존 native 재캐시 PID116962를 중단·재시작하지 않고 감시했다. 2026-10-30 09:48 UTC에212/212 Episode가 완료됐으며 Supervisor가 정상 종료를 기록했다.
- 새로운 독립 검사에서848개 입력 파일과212개 Graph/Receipt/Gate 파일의 SHA를 확인했다. 전체 **11,375 pre-action frame**에서 Graph21/Edge4, cap128, temporal extras0, presence, sampling Episode ID 및 RGB-D/Robot/GT timestamp 정합성이 PASS했다. Full-task120 Graph와28 paired Episode cache의 SHA도 재확인했다.
- 새 Corrective Graph212개를 RTX3090에 packing했다. 원래 pack_graphs와 정확하게 일치했고, GNN/State Head 및 두 message layer의 유한한 nonzero gradient를 확인했다. 저장한 model/optimizer/RNG에서 다음 업데이트의 loss와 weight가 연속 실행과 정확히 같았다. Train9,148/Validation2,227 frame를 사용한 **2-update engineering fixture**이며 본 학습 가중치는 아니다.
- 기존17개 계약 테스트가 통과했다. 기존 E0,18D Head,12k budget, AdamW, loss와 validation selection 규칙은 유지한다. 본 학습 전 검증 기록을 보강해 물리 단위별 validation error, 완전한 final dataset의 checkpoint identity, 파생 Peg-tip/Hole error와 실제 물체 이동 집단을 기록하게 했다. Raw centroid의 미지원 orientation/shape는 NaN으로 남긴다.
- 기존 자동 호출 Watcher와 대기 Bridge만 종료하고 로그를 보존했다. 현재 세션이 직접 이어갈 책임을 갖는다. Codex callback invocation은0회이며 원본 Task10-F 작업이나 F1 재캐시는 종료하지 않았다.

## 현재 승인 Gate

Vast.ai 조회 결과57640710은 `actual_status=exited`, `intended_status=stopped`이며 기존 직접 SSH 연결은 거부됐다. 인스턴스를 임의로 시작하거나 수집 작업을 재시작하지 않았다.

F1에는212개 모두의 새 RGB-D/Robot/GT/physical-mask/receipt가 전송돼 있다. 그러나 원본 물리 감사 상세, 실행 provenance, snapshot/replay 및 원자료 보존의 최종 증거를 결합한 sealed export가 아직 완성되지 않았다. 로컬 백업도 조사했다. **119개 physical_audit와 trace는 capture receipt의 원본 SHA와 정확히 일치했지만93개는 백업에 없다.** `approved=True` 요약이나 bool mask로 누락된 상세 증거를 대신하지 않는다.

따라서 `gnn_dataset_ready.json`, `data_ready_release.json`, 최종 training manifest 및 primary GNN checkpoint를 생성하지 않았다. cache 검증 성공을 DATA_READY 또는 학습 완료로 취급하지 않는다. 제공 서버 재가동 또는 누락 자료의 정확한 백업 경로를 요청했다.

## 이후 같은 Goal에서 이어갈 작업

1. 원 서버 또는 검증된 백업에서 누락 physical audit/trace와 source/config/runtime/preservation 증거를 확보한다. 원본212개와 실행 Source는 read-only다.
2. F1에서 이미 완료된 native Graph만 producer의 독립 export에 복사하고 전후 SHA를 확인한다. 사전 cross-runtime 비교4/6 FAIL 기록은 그대로 보존한다. 모든 Graph가 F1 native runtime에서 생성됐다는 변환 계약을 적용한다.
3. 물리 승인·GT38/Graph 정합성·Parent split·hash·runtime 검증을 독립 두 과정에서 통과시킨 뒤에만 DATA_READY를 원자적으로 게시한다.
4. F1 import는 이미 전송된 SHA-identical raw/native cache의 독립 복사본을 활용하고 누락 metadata만 전송한다. release 검증으로 데이터와 기존12k budget을 동결한다.
5. 기존 fresh E0 GNN+Head 학습, Robot-only 기준, frozen held-out absolute/실제 object displacement 평가, GT38 oracle/predicted assembly 및 동일 checkpoint A/B extraction을 수행한다. GRU/Action/BC는 실행하지 않는다.
6. 최종 CSV·figures·checkpoint/optimizer/RNG·SHA/readback·전체 Markdown/HTML 연구 보고서를 남긴다.

## 근거

- [전체 cache 독립 검증](../../outputs/task10f1/remediation_20261030/closed_cache_verification.json)
- [새 Corrective CUDA engineering 검증](../../outputs/task10f1/remediation_20261030/corrective_native_engineering_verification.json)
- [로컬 물리 승인 백업 감사](../../outputs/task10f1/remediation_20261030/local_backup_approval_audit.json)
- [현재 Goal·Source identity](../../outputs/task10f1/remediation_20261030/direct_continuation_identity.json)
- [17개 계약 테스트](../../outputs/task10f1/remediation_20261030/continuation_contract_tests_v2.log)

현재 판정: 재캐시·정합성·CUDA engineering은 **CONFIRMED**, 공간 상태 복원·일반화 및 후속 GRU의 근거는 본 학습/독립 평가 전이므로 **NOT_TESTED**다. source 서버 접근 문제는 데이터나 연구 가설의 실패로 판정하지 않는다.

추가 보존: F1의 기존 완료 artifact455개 SHA가 그대로 유지됨을 재검사했다. 새 GT/mask/receipt636개와 native manifest4개를 별도 로컬 archive로 백업하고 SHA 및 member640개 readback을 통과했다. [백업 readback](../../outputs/task10f1/remediation_20261030/closed_native_gt_backup_readback.json). [정확한 누락186개 경로](../../outputs/task10f1/remediation_20261030/missing_approval_evidence.files).

Goal blocked 감사: 동일한 producer 접근 불가와93개 승인 증거 누락을3개 연속 Goal turn에서 재확인했다. source 접근 복구 또는 정확한 백업이 없으면 전체 데이터 승인·본 학습을 정당하게 진행할 수 없어 `blocked`로 기록한다. 재캐시/검증의 실패나 Goal 달성을 의미하지 않는다. 57987772와 기존 파일은 그대로 보존한다. [blocked 감사](../../outputs/task10f1/remediation_20261030/blocked_audit.json).
