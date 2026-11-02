# Task10-I Stage4A — 58576054 재개 가능성 검사

**최종 후속 상태:** 이후 Stage4A 전체 연구를 재개해 독립 검증·인계를 완료했다(판정
`NO_GO`, Production 변경 없음). [전체 최종 보고서](10i_stage4a_final_report.md).
아래의 “재개하지 않았다/미실행”은 이 개별 검사 시점의 기록이다.

후속 업데이트: 아래 내용은 설치 전 검사 기록이다. 사용자 요청으로 이후 독립 cu130
환경을 복원하고 import·CUDA 연산 smoke를 통과했다.
[환경 복원 전체 보고서](10i_stage4a_environment_restore_58576054.md).
추가로 사용자 요청에 따라 기존 검증 fixture101개 action frame의 Forward Equivalence도
PASS했다. [Forward Equivalence 보고서](10i_stage4a_forward_equivalence_58576054.md).
남은4개 Sequence와 이후 연구 runner는 아직 재개하지 않았다.

2026-11-01 KST. 판정: **데이터는 재사용 가능하며, 실행 환경 복원 후 조건부 재개 가능**.
현재 기본 환경에서 연구 runner를 바로 실행할 수 있는 상태는 아니다. 이번 작업은 검사만
진행했으며, 패키지 설치·모델 추론·시뮬레이터·Renderer·평가 runner·학습을 시작하지 않았다.
Stage1–3 원본과 기존 Checkpoint, 이전 Stage4A 산출물을 수정하거나 삭제하지 않았다.

## 데이터 및 기존 진행 보존 — PASS

| 검사 | 실제 확인 결과 |
|---|---:|
| Stage3 Raw/GT SHA256 | 712개 전체 일치 |
| 고정 실행 Source SHA256 | 672개 전체 일치 |
| 완료 Baseline Sequence | 60 / 64 |
| 완료 Sequence의 Mask/Graph/Trace 관측 | 4,026개 |
| 현재 Episode·Frame·Camera·Pixel source 대응 | 1,009,795건 검사 |
| 합친 EXACT_GT Reference | 64개 Sequence, 4,162개 관측 |
| 원시 DINO Query archive | 552개 모두 읽기 성공 |
| DINO/SAM2 가중치·Tokenizer·설정 SHA | 기존 model identity와 일치 |
| 기존 progress와 개별 Receipt | 60개 내용 일치 |

Mask NPZ는 Receipt SHA뿐 아니라 실제 배열을 읽어 dtype/shape와 stage별 SAM2 support 수를
대조했다. 완료된 모든 frame의 support NPZ를 열어 Graph21/Edge4 shape, finite 값,
edge endpoint, cap128/object, robot node, 비활성 temporal feature, 현재 frame source,
Main-only Box와 Wrist Gate 채택 계약도 확인했다. Trace의 source별 원시 mask/point 대응이
존재하는지 검사했다. 이 검사는 캐시의 무결성·입력 계약 점검이며, 모델 forward의 재실행이나
현재 host에서의 Geometry 성능 재평가는 아니다.

GT Reference는 이전 v5/v6의 **이전부터 로컬에 보존돼 있던** canonical segmentation SHA와
64개를 다시 대조했다. 복사된 Reference 내부의 True 표지만 보고 EXACT_GT라고 새로
선언하지 않았다. 기존 4,162개의 Reference가 이전 데이터와 동일함을 확인한 것이다.
Shared Validation의 영상이나 정책 Final/Test를 열어 후보를 선택하지 않았다.

근거:

- [상세 데이터 검사](../../outputs/task10i/stage4a/resumption_check_58576054/data_audit.json)
- [기존 canonical GT SHA와 비교](../../outputs/task10i/stage4a/resumption_check_58576054/canonical_reference_hash_comparison.json)
- [원래 progress/Receipt identity](../../outputs/task10i/stage4a/resumption_check_58576054/original_progress_identity.json)

## 남은 Baseline — 정확히 4개, 136개 관측

| Logical Episode | Parent | Action | 관측(T+1) | 기존 미완료 흔적 |
|---|---|---:|---:|---|
| 3000129 | existing_30069 | 32 | 33 | Receipt 없음, partial Trace19 frame |
| 3000139 | existing_30080 | 33 | 34 | 시작 전 |
| 3000152 | existing_30093 | 33 | 34 | 시작 전 |
| 3000155 | existing_30096 | 34 | 35 | 시작 전 |

네 개 모두 기존 Corrective source의 Nominal sequence다.3000129의19개 partial Trace를
완료된 Sequence로 세지 않는다. Causal SAM2 memory가 보존된 checkpoint는 없으므로
재개할 때 이 Sequence도 t0부터 처리해야 한다. 나머지60개를 다시 추론할 필요는 없다.

이4개가 Stage4A 전체의 남은 작업은 아니다. 이후 수정 Reference 대응, Detector/Temporal/
Robot Filter/Gate/Graph 감사, 실용 후보 비교와 동결, Internal Test 및 Shared Validation이 남아 있다.

## 새 Host와 실행 환경 — 복원 필요

새 인스턴스58576054는 RTX3090 24GB, driver595.84다. `/workspace`는200GiB overlay이며
확인 당시 약186GiB가 남았다. inode도 충분했다. 별도 persistent volume으로 확인한 것은
아니므로 overlay를 영구 백업이라고 부르지 않는다. 새 Host에서 실행 중인 Stage4A 연구
process나 tmux session은 발견되지 않았다. 복사된 RUNNING 상태 파일/PID를 새 실행으로
오인하지 않았다.

| 항목 | 기존 Baseline | 새 기본 `/venv/main` |
|---|---|---|
| Python | 3.12.15 | 3.12.15 |
| PyTorch | 2.14.1+cu130 | 2.14.1+cu132 |
| torchvision | 0.29.1+cu130 | 0.29.1+cu132 |
| NumPy | 2.5.4 | 2.5.4 |
| SciPy | 1.18.1 | 미설치 |
| h5py | 3.16.0 | 미설치 |
| OpenCV | 5.0.0.93 | 미설치 |
| ManiSkill | 3.0.1 | 미설치 |
| SAPIEN | 3.0.3 | 미설치 |
| Gymnasium | 1.4.0 | 미설치 |
| Transformers | 5.19.0 | 미설치 |
| Hydra | 1.3.7 | 미설치 |
| SAM2 package | `/workspace/full_pipeline_sam2` | Source는 있으나 미설치 |

CUDA device 인식과 `libvulkan/libEGL/libGL/libcuda` 탐색은 성공했다. CUDA 연산, DINO/SAM2
forward, SAPIEN 렌더링, CPU PhysX는 **이 새 Host에서 실행하지 않았다**. 따라서 같은
RTX3090이라는 이유로 이전 출력과의 exact equivalence를 PASS로 선언하지 않는다.

`/workspace/full_pipeline_environment.txt`에 이전 package pin이 보존돼 있고, SAM2 Source,
현재 사용하는 SAM2.1 YAML 및 모델 파일도 존재한다. 다만
`/workspace/.venv-backups/57987772/venv-main-latest.txt`는 requirements 텍스트 백업이다.
복원된 가상환경이나 Python/site-packages 전체 archive가 아니다. 새 기본 cu132 환경에
그대로 실행을 얹기보다 기존 cu130 기준의 독립 환경을 준비해야 한다.

[실제 환경 검사](../../outputs/task10i/stage4a/resumption_check_58576054/runtime_inspection.json).

## 재개 시 필요한 순서

1. 새 독립 환경에 이전 package pin과 SAM2 Source를 복원한다. 기존 기본 환경이나
   동결 Source를 조용히 바꾸지 않는다.
2. 이전 v5 Source를 보존하고 새 isolated source에 현재 로컬의 복구 helper를 반영한다.
   복사된 v5에는 `resume_baseline.py`가 없다. 경로만 바꿔 원래 coordinator를 실행하면
   완료60개를 재사용하는 올바른 복구 경로가 아니다.
3. 계약 시험 및 새 Host의 Baseline Forward Equivalence를 확인한다. Torch 설정은
   기존 `deterministic=False`, `matmul_tf32=False`, `cudnn_tf32=True`와 구분해 기록한다.
4. 새 ExperimentLogger UUID와 독립 output root에서 검증된60개를 재사용하고 미완료4개만
   독립 episode/camera reset으로 처리한다. 원래 실패 root와 partial 산출물은 보존한다.
5. 완료된 manifest를 전제로 후속 감사·후보 평가를 재개한다. 기존 대기 상태 파일을
   그대로 연구 완료 증거로 사용하지 않는다.

이번 검사는1–5의 연구 실행을 시작한 것이 아니다. 판정은
**DATA_READY / ENVIRONMENT_RESTORE_REQUIRED / FORWARD_EQUIVALENCE_NOT_EXECUTED**다.

## 검사 구현과 한계

새 task-owned read-only 검사 도구는 `graph/task10i_stage4a/resumption_audit.py`다.
모든 검사는 SSH stdin으로 실행하고 결과는 로컬 output에 보존했다. 원격 실험 root에는
새 Source나 Logger run을 만들지 않았다. 운영 점검이며 추론/evaluation run으로 세지 않는다.

완전한 NPZ, 디스크 부족으로 잘린 NPZ, pickle 객체 payload 및 불완전 JSON을 구별하는
로컬 계약4개가 PASS했다. [검사 도구 테스트](../../outputs/task10i/stage4a/resumption_check_58576054/audit_tests.log).
이 검증은 60개 완료 cache와 Stage3 handoff·현재 의존 파일을 대상으로 하며, 모든
Task의 checkpoint를 로드하거나 전체 simulator environment의 실행 호환성을 검증한 것은 아니다.

[요약 JSON](../../outputs/task10i/stage4a/resumption_check_58576054/summary.json) ·
[산출물 SHA256](../../outputs/task10i/stage4a/resumption_check_58576054/artifact_sha256.json).
