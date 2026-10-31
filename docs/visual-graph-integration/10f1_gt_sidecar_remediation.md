# Task10-F1 GT Sidecar 보완 실행 기록

상태: **진행 중. DATA_READY 및 GNN 본 학습 완료가 아니다.**

2026-10-30. 원본212 Episode와 기존 Task10-F 프로세스를 보존하면서 누락을 실제 보완하는 독립 실행이다. 감사의 최초 결과는 [기존 감사 보고서](10f1_task10f_dataset_ready_audit.md)에 남아 있다.

## 실제 완료한 작업

- Vast57640710의 `/workspace/task10f_recovery_v2/f1_gt_supplement_20261030`에서 새212 Episode를 수집했다. Train nominal96/recovery80, validation nominal4/recovery32,100개 Parent이다. 기존 자료에 추정 GT를 붙이지 않았다.
- 원본212의 per-timestep GT 및 저장 snapshot 부재를 재확인했다. 원본 모델 checkpoint는 model/optimizer/scheduler/training RNG이며 simulator/controller/RNG snapshot이 아니다.
- 새 RGB-D/Robot76과 같은 pre-action timestep에 원본GT accessor8개 raw field, target18, 공식GT38, timestep/timestamp를 기록했다. 관측T+1과 actionT를 구분한다.
- 새 `gnn_supervision_mask`와 `physical_valid`, initial 및 매 physics-step contact를 포함한 physical audit를 저장했다. 원래 Safety/Recovery 기준을 유지했고 실패 Pilot을 승인하지 않았다.
- 새 parent의 anchor snapshot은 디스크에 저장·재읽기하고 동일 parent 초기화 조건에서5-step 반복 replay를 검사했다. 이것을 원본212의 snapshot이 복원됐다는 근거로 사용하지 않는다.
- 기존 data/config/audit1,187개와 Source620개의 SHA를 확인했다. 기존 config의 cache 시점 파일을 재구성해 manifest 기대SHA `095ddca684bfe9819862fa88f31bd8e0b8dc489f8502f30b4ab170ac78c0c45c`와 정확히 일치시켰다. 기존 파일은 변경하지 않았다. 이후 finetune이 바꾼 training 설정은 별도 lineage로 기록한다.
- 57987772는 재접속 가능하다. 원래 OOM 파이프라인을 재시작하지 않고, 완료104 Graph/13 Paired cache를 보존하면서 미완료분만 처리했다. 현재 Full-task Graph120개와 Paired28개 cache가 완료됐다. 미완료 partial pair2개는 별도 archive로 보존했다.

## Runtime 검사 실패 및 실제 해결 경로

57640710은 Torch2.8.0+cu128/NumPy2.1.2,57987772는 Torch2.14.1+cu130/NumPy2.5.4이다. 사전 지정한6 Episode의 전체 causal sequence328 frame를 native F1에서 재생성했다. Mask/Graph/Gate 및 provenance 배열의 정확한 일치를 요구했고 **2 PASS /4 FAIL**이었다. 실패 기록은 [runtime 비교 JSON](../../outputs/task10f1/remediation_20261030/runtime_equivalence_FAILED.json)에 보존한다. 허용오차를 늘리거나 이 실패를 PASS로 바꾸지 않았다.

따라서57640710에서 만들어진 Graph tensor를 본 학습용으로 혼합하지 않는다. 동일한 실제 관측589,109,174 bytes(848개 sealed input file)를57987772로 전송하고, **보완212개 전체를 F1 native runtime에서 재캐시**하는 별도 Supervisor 작업을 실행한다. 전송된 Episode부터 SHA와 Native GT38 oracle 계약을 확인하고 캐시한다. Label audit와 Graph writer의 입력 경로는 분리돼 있다.

실행 위치는 `/workspace/task10f1_gt_pretrain_20261030/operations/native_rebuild_20261030`이다. 단일 CUDA vision owner와 독립 CPU geometry worker8개를 사용한다. 단일 전송의 packet loss와 낮은 처리량을 확인해 완료 파일 목록을 나눈 전송을 사용한다. 전송을 위해 private key를 서버에 복사하거나 global agent/authorized_keys를 변경하지 않았다.

수집 runtime과 Graph 생성 runtime은 최종 계약에서 별도로 기록한다. 새 Graph가 원래576 cache와 같다고 주장하지 않는다. Native graph 생성이 끝난 뒤에만 전체GT/Mask/Frame/Hash/Runtime 검증과 READY 게시를 수행한다.

## Sampling identity 검증

기존 `legacy_fps.adapter.frame_seed`는 Episode ID도 RNG 입력으로 사용한다. 따라서 센서 배열만 같다고 새 ID로 cache를 relabel하면 충분하지 않다. Logical capture ID와 **graph sampling episode ID**를 분리해 기록하고, 원래 sampling namespace를 보존한다. Native 재캐시도 동일한 sampling identity를 사용한다. Readiness validator가 NPZ scalar와 manifest/receipt의 sampling identity를 확인한다.

## 아직 남은 단계

1. 보완212개의 F1 native cache 완료 및 모든 파일 SHA/readback.
2. 새 Graph를 producer의 독립 sealed export로 복사하고 수집/Graph runtime 및 실제 config 출처를 결합.
3. Physical initial/T-step evidence, GNN mask, Parent/Prefix/GT38 oracle 계약 전수 검증.
4. Candidate marker를 별도 경로에서 두 번 독립 검증한 후에만 `gnn_dataset_ready.json`을 atomic/no-overwrite 방식으로 게시.
5. Watcher transfer/readback 및 이후 별도 GNN 사전학습 재개. 현재 marker 부재, 실제 READY 자동 호출0회이다.

GNN 본 학습·새 GRU/Action 학습·Closed-loop BC·Git commit·외부 publishing은 이 보완 실행에서 시작하지 않았다. 현재의 검증17개는 PASS이며 [테스트 로그](../../outputs/task10f1/remediation_20261030/contract_tests_v9.log)에 있다. macOS에서는 CPU 계약 테스트만 실행했고, simulator/render/vision 실행은 위 Vast host에서 수행했다.

## 증거와 코드

- [Native 입력 직접 전송 기록](../../outputs/task10f1/remediation_20261030/native_parallel_transfer_v2.log)
- [Runtime 비교 사전 계획](../../outputs/task10f1/remediation_20261030/runtime_comparison_plan.json)
- [독립 수집 코드](../../graph/gt_pretrain10f1/remediation_collect.py)
- [Native 재캐시 코드](../../graph/gt_pretrain10f1/remediation_native.py)
- [게시 전 검증/Export 코드](../../graph/gt_pretrain10f1/remediation_export.py)

이 문서는 실행 중 기록이며, 최종 완료·checkpoint·성능을 선언하는 보고서가 아니다.
