# Task10-F → Task10-F1 Dataset Ready 감사

2026-10-30 06:18–06:20 UTC / 15:18–15:20 KST. 판정: **NOT_READY**. Task10-F의 수집·Graph Cache는 완료됐지만 GT-supervised F1 인계 계약은 충족하지 않는다. `gnn_dataset_ready.json`은 작성하지 않았다. 기존 작업의 중지·재시작·코드 수정, 데이터 전송 및 F1 본 학습을 수행하지 않았다.

## 조사 범위와 완료 상태

Vast **57640710**의 `/workspace/task10f_recovery_v2/full_cache/corrective_cache_manifest.json`이 선언한 **212개 Episode / 11,375개 Action Timestep / 100개 Parent**를 전수 조사했다. 수집 완료 기록은 04:24 UTC, 캐시 완료 기록은 05:01 UTC이다.

| Split | Nominal Reach | Corrective P Branch | 합계 |
| --- | ---: | ---: | ---: |
| Train | 96 | 80 | 176 |
| Validation | 4 | 32 | 36 |
| 합계 | 100 | 112 | 212 |

기존 실패 Pilot이나 선언 범위 밖 N/control branch를 승인 데이터에 추가하지 않았다. 동일 Parent의 nominal/prefix/branch가 Train과 Validation을 넘지 않는 것을 확인했다. 이 자료의 지원 범위는 Reach·Robot Disturbance·Expert Recovery이며 Full-task Grasp/Lift/Insertion으로 확장 해석하지 않는다.

## 인계 Gate 결과

| 항목 | 결과 | 실제 확인 범위 |
| --- | --- | --- |
| 수집 및 Graph Cache 완료 | PASS | collection status PASS, cache complete=true, 212/212 receipt |
| Raw / Graph / Action Mask SHA256·Readback | PASS | 원본 H5 212개, Graph NPZ 212개, Mask NPZ 212개; receipt SHA와 비교 |
| RGB-D / Robot76 / Graph Timestep | PASS | T actions, T+1 observations, Robot76 shape, 두 camera 배열 길이, 20Hz clock; Graph Robot76/timestep/timestamp 수치 일치 |
| Graph 저장 계약 | PASS | Node21 / Edge4, extras 7개=0, cap128, node types, graph-local edge bounds, Presence2, Box Main-only source |
| Parent / Branch provenance | PASS | producer candidate/receipt/parent/offset 연결, parent split 분리 |
| 생산자 물리 승인 근거 | PASS | recovery 112개 approved=true / physical_safety=true; nominal 100개 ACCEPTED; 저장된 physical check 9,415개 valid=true |
| F1용 per-frame GNN physical mask | FAIL | `gnn_supervision_mask` / `physical_valid`가 없음. 위 branch 승인과 별개로 pre-action/prefix까지 명시한 mask 필요 |
| GT38 Object Sidecar / GT 정합성 | FAIL | 212/212에서 필요한 GT 필드 누락; GT와 RGB-D의 실제 시점 정합성은 검증할 수 없음 |
| 완료 Manifest의 참조 SHA | FAIL | `full_cache/config.json`의 기록 SHA와 현재 SHA 불일치 |
| 봉인된 F1 runtime export | NOT_TESTED | 요구 schema의 runtime_contract/dataset_manifest/sealed file list가 제공되지 않음 |
| F1 runtime과의 동등성 | NOT_TESTED | 서로 다른 runtime을 자동 혼합하지 않음 |

이 감사는 저장된 승인 근거를 검사했다. Simulator를 새로 실행하거나 물리 안전성을 재시뮬레이션하지 않았다. RGB-D 내용 자체와 GT pose의 동시성을 PASS로 간주하지 않았다. Causal vision의 원본/최적화 동등성은 기존 생산자 검증 기록 범위에 의존하며 이번 감사에서 vision을 재실행하지 않았다.

## READY를 막는 GT 누락

실제 HDF5 전체 schema와 별도 NPZ inventory를 조사했다. 필요한 다음 필드는 선언된 212개 원자료 모두에 없다.

| 필드 | 필요한 Observation Shape | 용도 |
| --- | --- | --- |
| `raw_peg_pose` | `[T+1,7]`, world xyz m + unit wxyz | Peg 상대 위치·자세 |
| `raw_hole_pose` | `[T+1,7]`, world xyz m + unit wxyz | Hole 상대 위치·자세; Box origin과 구분 |
| `raw_peg_half` | `[T+1,3]`, m | Peg head endpoint 계산·형상 |
| `raw_hole_radius` | `[T+1,1]`, m | 원본 encoder의 Hole shape scalar |

212개 `supervision.npz`에는 `segment`, `supervision_mask`만 있다. 이는 Action BC Mask이며 GT sidecar나 물리 승인 mask가 아니다. Action Mask OFF인 관측은 **3,470개**이나, GT 및 GNN 물리 mask 검증 전에는 F1 학습 대상으로 승인하지 않는다.

루트 아래 NPZ inventory에서 GT sidecar 후보는 0개였고, 직렬화 snapshot 이름 후보도 없었다. `pilot.snapshot()`은 simulator/controller/RNG/expert state를 RAM에 deepcopy하는 구현이며 수집 코드에 그 snapshot을 디스크에 보존하는 경로가 없다. 모델 checkpoint는 정확한 episode simulator snapshot의 증거가 아니다. 현재 자료를 Graph centroid, 초기 pose 상수 또는 seed-only 근사 replay로 채우지 않는다. 정확한 보존 snapshot과 검증된 replay 없이 누락된 매 시점 GT가 복원 가능하다고 주장하지 않는다.

Task10-F 서버의 원본 `recurrent/features.py` SHA는 **`80de6fd742be4e7c9a3477cde0419e803b72253c09d1227fc0fec23b965b0761`**로 F1의 공식 GT38 계약과 일치한다. Robot76으로 재구성하는 로봇 15D, clean freshness/age 2D와 달리, 누락된 물체 GT는 센서 로봇 값으로 대체할 수 없다. 정확한 기록 요구사항은 [GT Sidecar 인계 계약](10f1_task10f_gt_sidecar_handoff.md)에 보존돼 있다.

## 해시 및 Runtime

SHA256을 읽은 파일은 **1,187개 / 881,858,173 bytes**이며 감사 중 size/mtime 변경은 없었다. Raw H5 및 Graph/Action Mask SHA는 각 receipt와 일치했다. 완료 manifest가 참조한 config만 다음과 같이 달랐다.

- Manifest 기록 SHA: `095ddca684bfe9819862fa88f31bd8e0b8dc489f8502f30b4ab170ac78c0c45c`
- 현재 config SHA: `bb168f4ca1832ed8087923dfb40778da2c256dd06a2cf8406dd005c19f466fe5`

`finetune.py`가 캐시 완료 후 training 설정을 갱신하며, `ab_training_config.json`에는 현재 SHA가 기록돼 있다. 따라서 이 차이는 저장된 Raw/Graph가 변했다는 증거가 아니다. 다만 F1 봉인 manifest로 그대로 승인할 수는 없다. 기존 manifest/config를 수정하지 않고, 실제 source identity와 training 변경 이력을 분리한 새 sealed export가 필요하다.

설치 runtime은 Torch **2.8.0+cu128**, NumPy **2.1.2**, ManiSkill **3.0.1**, SAPIEN **3.0.3**, Gymnasium **1.4.0**, h5py **3.16.0**이다. 생산자 runtime source와 채택 gate는 `physx_cpu` + GPU sensor renderer, control 20Hz / physics 100Hz / 5 substeps를 가리킨다. 상속된 `task08_reference_conditions.json`과 `sensor_contract.json`의 macOS/CPU runtime 문자열은 과거 reference이며 현재 실행 backend로 재해석하지 않았다. 카메라는 Main=`base_camera`, Wrist=`hand_camera`, 128×128 RGB와 int16 mm depth, Robot76 schema를 기록한다. 이 감사에서 새 backend/device 실행은 하지 않았다.

Graph receipt의 frozen geometry protocol SHA는 `422d1da293107b915963abbcd738e8f327b5ca82d44d00fd76a3e9cbe208eeb5`이다. Runtime/source/camera 증거는 아래 JSON에 저장했다. 최종 READY에는 별도로 실제 vision weight identity와 모든 dependency를 포함한 안전한 상대경로 export, SHA/크기 및 봉인 후 재검사가 필요하다.

## 운영 상태와 후속 인계 조건

06:19:59 UTC 기준 Task10-F `continuation` PID **41566**은 계속 RUNNING이었다. 수집·캐시 Supervisor는 완료된 EXITED 상태로 보존했다. F1 Watcher PID **21624**는 `WAITING_DATA`, marker absent, Invocation Journal의 `total_calls=0`, 실제 READY 자동 호출 **0회**이다. Watcher 설정이나 lock/journal을 변경하지 않았다.

참고로 F1 **57987772**의 기존 `task10f1_data`는 02:58 UTC CUDA OOM으로 EXITED였고 완료된 Episode cache receipt는 104개였다. 이 상태도 재시작하지 않았다. 유효한 DATA_READY가 이후 도착하면 기존 F1 상태를 확인한 뒤 미완료 단계만 재개해야 한다.

현재 재개 조건은 (1) 원본 시점의 정확한 GT sidecar 확보, (2) GT/Robot/RGB-D clock 정합성 및 oracle GT38 확인, (3) prefix를 포함한 명시적 GNN/physical mask, (4) runtime/vision/source identity와 변경 이력을 포함한 봉인 export, (5) 모든 참조 SHA·Readback 재검증이다. 이를 충족한 뒤에만 `gnn_dataset_ready.json`을 atomic하게 작성한다. Task10-F의 실행 중 수집·학습 코드를 이번 감사에서 수정하지 않았다.

## 증거 및 재현

- [212 Episode 전수 감사 JSON](../../outputs/task10f1/task10f_dataset_ready_audit_20261030.json)
- [Runtime / Camera / Source 증거](../../outputs/task10f1/task10f_runtime_ready_audit_20261030.json)
- [읽기 전용 감사 스크립트](../../outputs/task10f1/audit_task10f_ready_20261030.py)
- [감사 로그](../../outputs/task10f1/task10f_dataset_ready_audit_20261030.log)
- [기존 작업 상태](../../outputs/task10f1/task10f_ready_process_snapshot_20261030.txt)
- [Watcher 상태](../../outputs/task10f1/task10f_ready_watcher_snapshot_20261030.json) / [Invocation Journal](../../outputs/task10f1/task10f_ready_invocation_snapshot_20261030.json)
- [감사 산출물 SHA / Readback index](../../outputs/task10f1/task10f_ready_audit_artifact_index_20261030.json)

첫 감사는 config SHA 불일치를 발견하고 중단했다. 해당 v1 실패 로그·스크립트를 별도 보존한 뒤, 모든 불일치를 FAIL로 기록하면서 나머지 자료를 조사하는 방식으로 전수 감사를 완료했다. 승인 threshold는 변경하지 않았다. 완료 감사는 원격에서 25.35초가 걸렸다. 학습·최적화·정책 평가 및 publishing은 수행하지 않았다.
