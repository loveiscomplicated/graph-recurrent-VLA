# Repository cleanup — 실행 경로·보존 대상 사전 조사

Task ID: `repo_cleanup_preflight_20261104` · 조사일: 2026-11-04 KST.

현재 상태: 후속 **D12에서 별도 모델 확정안 대기를 취소하고 현재 주 live를 기준으로 채택**했다.
[모듈화 진행 보고서](repo_cleanup_modular_live_20261104.md)에 backup commit과 구현·검증을 기록한다.
아래 사전 조사 당시에는 구조 이동·삭제·백업 커밋을 실행하지 않았다.
이 보고서는 현재 체크아웃의 근거와 논의안을 기록한다. 과거 실험의 재현 완료나
전체 아티팩트의 백업 완료 판정이 아니다.

진행 우선순위 재확인: **상위 목적은 repository cleanup이다.** Cache·해상도·데이터
선별에 관한 결정은 정리 후 기본 실행 경로가 충족할 요구사항이다. 이를 실제 데이터
수집·cache 생성 작업으로 확대하지 않는다. 이후 논의는 현재 runtime과 과거 실험의
경계, 진입점·문서, 보존·재현성, 백업 후 이동·정리 순서에 우선순위를 둔다.

## 1. 이번에 반영할 사용자 요구

- 초기 요구는 모델 확정안 대기였으나 D12로 변경됐다. 현재 주 live를 교체 가능한 기준으로 삼는다.
- Graph cache 생성, cache 기반 학습, 현재 관측 기반 live rollout을 기본 실행 흐름의
  **후보**로 삼는다. 지금 세 경로로 구조를 확정하지 않는다.
- Cache/live의 퍼셉션·Graph 생성 로직을 가능한 한 공유한다.
- GPU 처리와 World/EE 선택을 필요한 범위에서 유지한다.
- Peg가 완전히 보이는 데이터와 약 512 해상도를 고려한다.
- Grounding DINO 대신 GPT Terra를 쓰는 방향을 검토한다.
- 과거 코드·설정·결과는 재현성을 확인한 뒤 정리한다.
- **대화 중 범위 변경:** 로봇 필터와 손목 gate를 포함한 주 live 전체를 이번 정리
  대상에 포함한다. 다른 곳의 확정안을 기다리는 별도 제외 대상으로 취급하지 않는다.
- 실제 정리 직전에 미커밋 변경을 다시 확인하고 필요한 코드를 커밋해 보존한다.
  Git 밖 데이터·체크포인트·실험 결과는 별도로 보존 여부를 확인한다.

공통 simulator에는 모델·Graph·dataset 계약을 추가하지 않는다. 기존
`ExperimentLogger`와 artifact/publishing 인프라 및 run identity 규칙을 유지한다.

## 2. 조사 기준과 Git 상태

기준 HEAD: `d42f777bd8a2a23eef4f814f5922783383c17948`.
브랜치: `archive/visual-graph-wip-20261102`.
로컬 upstream ref와 비교해 **19 ahead / 0 behind**다. `fetch`하지 않았으므로
실제 원격 최신 상태까지 확인한 수치는 아니다.

이 보고서를 만들기 전 상태는 추적 파일 수정 **26개**, untracked 파일 **84개**,
staged 변경 **0개**였다. 추적 파일 diff는 513 insertions / 131 deletions다.
Untracked는 보고서 23개, `graph/experiments/` 59개 파일,
`graph/plan_a_bc/` 2개 파일이다. 폴더 단위 `git status` 표시와 파일 수를 구분했다.

수정에는 GPU robot filter/준비 작업, wrist gate·fusion, CurrentGeometry,
이전 live 평가 caller, frozen inference, README와 연구 색인이 함께 들어 있다.
현재 변경을 단순히 “새 실험 문서”로 묶어 커밋하면 실제 runtime 변경을 놓친다.

- [Git 상태와 변경 파일 SHA256](../../outputs/repo_cleanup_preflight_20261104/git_snapshot.json)
- [정적 import 의존성 목록](../../outputs/repo_cleanup_preflight_20261104/static_dependencies.json)

위 파일 hash는 관측 상태 식별용이다. 변경 파일의 실제 내용을 복구할 patch나
Git commit을 대신하지 않는다. 동시 작업이 가능한 비원자적 snapshot이므로 실제
정리를 시작할 때 다시 관측해야 한다.

별도 `graph-recurrent-VLA-legacy-fps-release` worktree가 존재하고 `/private/tmp/`
아래 과거 worktree 두 항목은 prunable로 표시된다. 이번에는 다른 worktree의
파일·변경·용량을 전수 조사하거나 worktree를 정리하지 않았다.

## 3. 현재 세 흐름에 대응하는 실제 코드

아래는 현재 구현 설명이며 최종 디렉터리·모델·CLI 제안이 아니다.

| 흐름 | 현재 대응 코드 | 현재의 제약 |
| --- | --- | --- |
| Graph cache 생성 | [cache.py](../../graph/plan_a_bc/cache.py)의 `make_vision`, `cache_episode`, `CachedDataset` | 특정 실험의 audit/config/normalizer/assets를 요구하며 Terra/SAM2를 다시 실행한다. 저장 mask를 입력받는 일반 CLI가 아니다. |
| 학습 | [train.py](../../graph/plan_a_bc/train.py) | GNN 이전 Graph tensor cache를 읽어 기존 정책을 학습한다. 현 E0/E1·split·일정·선택 규칙은 과거 실험 계약이다. |
| Live rollout | [rollout.py](../../graph/plan_a_bc/rollout.py)의 `evaluate → run_episode` | 현재 센서 관측으로 mask와 Graph를 새로 만든다. CUDA renderer·기존 controller/평가 assets를 요구한다. |
| 기존 전체 실험 조정 | [pipeline.py](../../graph/plan_a_bc/pipeline.py) | cache pilot/full, E0/E1 학습, 선택 freeze, Test cache/offline/live 평가와 report를 순서대로 실행한다. 범용 기본 launcher로 곧바로 채택할 수 없다. |

Cache와 주 live는 이미 다음 구현을 공유한다.

- [LiveGrounding](../../visual/runtime/live_grounding.py) 및
  [TensorOnlineVisionAdapter](../../graph/integration_baseline/tensor_online_vision.py), SAM2 tracker.
- [CurrentGeometry](../../graph/plan_a_bc/geometry.py)와 그 아래 PlanA·tensor geometry.
- `geometry.coordinate_frame`을 전달하는 World/EE 선택.
- 현재 mask producer를 `CurrentGeometry.build`에 전달해 robot 준비와 SAM2를 겹치는 호출 방식.

입력 취득은 다르다. Cache는 `CurrentGeometry.from_h5`, live는 `rollout.sensors`다.
Cache 저장 시 `numpy_graph`/`numpy_records`/`numpy_packet`으로 명시적으로 export하고,
live의 모델 입력은 CUDA Graph를 pack한다. 디스크 저장 경계와 계산 경계를 구분해
보존해야 한다. Cache는 GNN 출력 embedding cache가 아니므로 GNN 학습 가능성을
임의로 제거하는 방향으로 바꾸지 않는다.

`CurrentGeometry.build`는 이미 ready mask 사전도 받는다. 따라서 사용자가 말한
“저장 RGB·Depth·mask → Graph cache”는 이 구현을 재사용할 여지가 있다. 다만 현재
cache CLI에는 저장 mask loader/선택 기능이 없다. Mask가 raw SAM2인지,
robot filtering/gate가 적용된 결과인지 확인해야 중복 필터링과 의미 변경을 막는다.
새 인터페이스는 통합 모델과 데이터 결정 이후 논의한다.

이름만 보고 실행 경로를 선택하기 어려운 실제 사례도 있다.

- `integration/visual_graph/pipeline.py`와 `adapters.py`는 현재 0-byte placeholder다.
  이 디렉터리명이 현재 통합 모델의 실행 위치라는 근거는 없다.
- [visual/scripts/run_pipeline.py](../../visual/scripts/run_pipeline.py)는 과거 DINO/SAM2
  진입점이며 후보 부재 시 GT bbox fallback이 있다. 주 live의 Terra 경로와 다르다.
- [task10i_stage4a2/frozen_inference.py](../../graph/task10i_stage4a2/frozen_inference.py)는
  저장 HDF5를 재생하며 DINO 직접 생성·특정 `/workspace/` 경로·기록된 조건을 유지한다.
  Geometry 일부를 공유한다는 이유로 최신 주 live와 같은 설정이라고 부를 수 없다.
- 루트 README와 초기 project context의 독립 prototype 안내는 현재 모든 실행
  경로를 대표하는 안내가 아니다. 실제 정리 후 기본 경로 중심으로 문서 갱신이 필요하다.

## 4. 주 live 전체에서 함께 보존할 내용

현재 소스에서 확인한 연결이다. 이번 호스트에서 GPU 실행으로 새로 검증한 결과는 아니다.

| 구간 | 확인한 현재 연결·의미 |
| --- | --- |
| 관측 | `rollout.environment`: CPU PhysX + CUDA renderer, 두 카메라 native 512×512, `ObservationDevice` |
| Tensor 입력 | CUDA RGB/depth snapshot 유지. 작은 robot/calibration 값에는 기존 host 검증·FK 경계가 있음 |
| Grounding/추적 | Terra primary joint Peg/Box 초기화, SAM2 CUDA mask, episode/camera별 상태 reset |
| 로봇 필터 | `TensorRobotPreparation` + `CudaRobotFilter`, current-frame 준비를 mask producer 이전에 제출 |
| 보정 분기 | `RobotFilterFrame`이 full-screen classification을 correction과 최종 mask에 공유. 보정의 host 경계와 frame 수명 검사 보존 |
| 손목 gate | CUDA 우선 선택, raw-mask 분모·현재 C3 의미·GPU 오류 전파. CUDA 부재 시 기존 CPU 경로 |
| 기하·표본 | mask 픽셀 역투영, PCA/특징, Shape-first, 교차뷰 확인, dedup, 최종 object별 FPS |
| Graph | tensor edge/Graph 생성과 모델 입력 CUDA 유지, 현재 관측 source provenance 유지 |
| 좌표계 | World/EE 설정 전달, EE의 현재 TCP reference, packet/record/Graph frame 일치 검사 |
| 제어·평가 | 기존 action/controller·physics audit, GT 평가 경계와 cache를 쓰지 않는 live 관측 |
| 기록 | 명시적 Graph/mask export, timing·진단·영상·공통 실험 identity |

현재 기본 경로의 GPU filter는
`graph/experiments/robot_filter_gpu_io_10i_20261103_v1/`에서 직접 가져온다.
`experiments`라는 이름만으로 archive로 옮길 수 없는 구체적 의존성이다.

세 진입점의 정적 import closure에는 네 실험 package가 나타난다.
`robot_filter_gpu_io_10i_20261103_v1`, `cuda_ingress_10i_20261103_v1`,
`graph_latency_10i_20261103_v2`, `resolution_perception_graph_20261103_v1`다.
**조건부·선택 backend의 import까지 포함한 상한 목록**이며, 모두 기본 프레임마다
실행된다는 의미는 아니다. 특히 `cuda_ingress`는 opt-in `cuda_resident` 분기다.
Closure에는 untracked `cuda_ingress` helper 5개와 `plan_a_bc/robot_frame.py`도 있다.
보정·선택 backend까지 보존하려면 이 코드도 백업 검토에 포함해야 한다.

현재 `CurrentGeometry._build_prepared`는 object별 최종 128점과 기존 Graph 의미를
유지한다. 이는 현재 동작의 기록이며 향후 512 모델의 최종 설계를 확정하는 문장이 아니다.
[512 정보 손실 분석](10i_512_information_loss_localization_20261104.md)은 작은 초기
prefix에서 sampling/최종 압축의 영향을 보여준다. 512 관측 도입만으로 Graph 품질
향상이나 새 데이터 적합성을 보장한다고 해석하지 않는다.

[기존 live audit](10i_live_nonrobot_coverage_audit_20261104.md)의 옛 결함 목록도
그대로 적용하지 않았다. 현재 이전 평가 caller 일부는 tensor adapter와 export가
수정되어 있다. 한편 `graph_recurrent10g.eval_runner.vision`의 `DetectionMemo`는
여전히 callable detector를 가정하고 `LiveGrounding`은 joint initializer다.
`geometry_reach10d.runtime.gpu_render_environment`도 주 live와 별도 renderer 생성
분기를 유지한다. 이번에는 이 평가들을 실행하지 않았으므로 전체 동작 여부는 미판정이다.

영상 mask readback이 모델 호출 전에 있고, action 후 영상·주기적 Graph export가 있는
주 live 기록 경계도 현재 코드에서 확인했다. 정리 시 이 기능을 빠뜨리지 않되,
기능 분리나 기록 빈도 변경을 성능 개선으로 임의 적용하지 않는다.

## 5. Git 밖 보존 대상과 용량

[실측 요약](../../outputs/repo_cleanup_preflight_20261104/artifact_inventory_summary.json).
현재 checkout의 일반 파일 logical size이며 GB는 10⁹ bytes다.
File symlink와 `.venv`, `__pycache__`, `node_modules`는 제외했다. 중복·hardlink를
제거한 고유 용량이나 실제 회수 가능한 디스크 공간이 아니다.

| 위치 | 약 GB | 의미와 사전 보존 기준 |
| --- | ---: | --- |
| `outputs/` | 53.40 | 원자료·cache·체크포인트·진단·source archive·복원본·게시용 archive. 전체를 생성물로 삭제할 수 없음 |
| `experiments/` | 11.83 | run metadata/config/metrics, checkpoints, report, W&B offline, publishing/추적 자료 |
| `visual/` 전체 | 2.20 | 추적 소스와 ignored 자료 혼합. 아래 세부 분류 필요 |
| `audit_outputs/` | 0.32 | 과거 감사 근거·모델·검증 자료. 재생성 가능성 확인 전 보존 |

`outputs/task10i` 약 16.98GB, `outputs/hf_publication_remaining_20261031` 약 13.76GB가
큰 비중이다. 두 번째 경로에 게시 archive가 있다는 사실만으로 원격 백업 완료를
판정할 수 없다. `visual/results` 약 1.35GB, `visual/cache` 약 0.42GB,
`visual/third_party` 약 0.21GB, `visual/checkpoints` 약 0.18GB,
`visual/data` 약 0.036GB가 있다. Visual 전체가 ignored인 것은 아니다.

별도 보존 목록에는 다음 관계를 함께 묶어야 한다.

1. 원본 RGB-D HDF5/NPZ, calibration, robot/TCP state, action, mask 및 split/parent
   manifest. GT·pixel reference·시뮬레이터 state는 평가/재캡처 재현 용도로 연결한다.
2. 선택 best/last와 재개용 optimizer/scheduler/RNG/cursor, normalizer/action codec,
   모델 정의·외부 weight hash·설치 source/lock. `.pt`에는 simulator snapshot도 있다.
3. Graph cache·mask·point provenance·receipt와 생성 config/source identity.
   재생성 절차와 결과가 확인되기 전에는 중간물이라는 이유로 삭제하지 않는다.
4. 실험 당시 source archive/frozen source. Dirty 실험은 현재 Git SHA만으로 재현되지
   않으므로 `outputs` 안의 source tar도 독립 보존 대상이다.
5. `metadata.json`, 실제 적용 config·CLI override, metrics·local/offline 로그,
   `<artifact>.tracking.json`, artifact manifest·게시 receipt·고정 revision·경로 alias.
   Sidecar는 각각 outputs 5,527개, experiments 6,841개, visual 1,378개를 관측했다.
6. 결론의 근거가 된 진단 배열·CSV·figure·영상·원시 timing trace와 실패/partial 기록.
   실패 자료는 성공 결과와 구분하되 조사 전 삭제하지 않는다.

일반 다운로드 cache·설치 복사본·중복 archive는 추후 정리 후보일 수 있다.
같은 크기·이름으로 중복을 단정하지 않고 content hash, manifest 참조, 재다운로드
가능한 revision, 실제 복원 검증으로 결정한다. Symlink 대상은 현재 크기 집계 밖에
있을 수 있으므로 실제 백업에서는 별도로 확인한다.

`.env` 존재와 ignore 규칙만 확인했으며 내용은 읽거나 보고서로 복사하지 않았다.
인증 파일은 코드·실험 archive와 분리해 취급한다.

## 6. 재현 가능성의 현재 증거와 남은 확인

[기존 HF 게시 보고서](hf_remaining_publication_20261031.md)는 여러 snapshot의
고정 revision을 기록하지만 전체 상태는 `IN_PROGRESS`이며 recovery/legacy 입력의
미완료 기록도 있다. 후속
[Stage3 의존성 완료 보고서](10i_stage3_hf_dependency_completion_20261102.md)는 네
archive 20,402 member와 Stage3 raw/GT 712 참조의 복원·해시 검증을 기록한다.
그 완료를 11월 4일 새 GPU 진단까지 포함한 전체 백업 완료로 확대하지 않는다.
이번에는 HF/서버에 접속해 최신 원격 상태나 전체 payload를 재검증하지 않았다.

또한 Stage3의 기존 가시성 제한은 **초기 Main 카메라에서 Peg 양끝·파지 영역이
보이는 조건**이며 모든 timestep의 완전 가시성을 보장하지 않는다. 이를 앞으로의
“완전히 보이는 Peg” 선별 기준으로 자동 채택하지 않는다. 기존 128 원자료를
resize하는 것과 native 512 RGB-D를 재캡처하는 것도 구분해야 한다.

Archive·삭제 여부를 결정하기 전에는 실험 family별로 실제 적용 source/config,
data/split hash, weight/normalizer, 실행 환경, entrypoint와 결과 receipt를 연결한다.
대표 복원에서는 새 위치에 해제해 참조 해시를 확인하고 cache readback, checkpoint
load, 필요한 짧은 평가를 수행한다. 원래 경로의 절대경로는 alias로 복원하고 원본
manifest를 임의로 덮어쓰지 않는다. 과거 source의 재현과 새 runtime으로의 이식은
별도 판정으로 남긴다. CUDA 전용 검증은 실행 가능한 host에서 해야 한다.

## 7. 구조 확정 전에 논의할 항목과 결정

| 항목 | 논의안·미정 사항 |
| --- | --- |
| 세 기본 흐름 | 사용자 제안 세 흐름을 중심으로 검토. 폴더명·모델 클래스·config schema·CLI는 통합 모델 확정 후 결정 |
| 목표 디렉터리 구조 | **제안 P04 개정:** 사용자 제안에 따라 iteration 영역인 root `graph/gnn/controller`와 공통 안정 부분·통합을 맡는 `vla/`를 분리한다. 기본 명령 `tools/`, envs/운영 인프라/visualization은 유지한다. [구체 구조안](repo_cleanup_structure_proposal_20261104.md) 참조. 세부 경계·통합 모델·인터페이스 미확정이며 소스 이동 없음 |
| 캐시 경계 | **결정 D03:** 원본 관측·Grounding 기록・필터/gate 적용 전 SAM2 mask를 에피소드 시간 순서의 1차 cache로 보존한다. 로봇 필터·손목 gate와 Graph 생성을 2차 생성 경로에서 처리한다. 구체 형식·인터페이스는 미정. P01은 이전 검토안이며 P02의 기능 경계를 채택 |
| 데이터 준비 | **결정 D06·D08:** Main/Wrist native 512×512를 사용하며 향후 필요한 원자료는 expert 궤적을 새로 수집하는 B안으로 준비한다. 기존 궤적 재렌더를 기본 경로로 삼지 않는다. 실제 수집은 이번 cleanup에서 시작하지 않으며 규모·split·일정은 미정 |
| 과거 실험 호환 범위 | **결정 D09:** HF를 기본 보존소로 삼아 실험 산출물을 누락 없이 보존·검증하고, docs에 commit·고정 원격 위치·복원 방법을 남긴 뒤 과거 코드를 현재 tree에서 제거한다. Drive 기존 자료도 연결한다. 현재 runtime 의존성과 유지할 DINO fallback은 보존하며 Git 이력은 유지 |
| 오프라인 평가 | **결정 D01:** 학습 중 검증과 별도로, 저장된 cache·정답·checkpoint를 사용하는 독립 오프라인 평가를 지원한다. 세 기본 흐름 후보에 더하는 보조 기능이며 구체적인 CLI·지표·구조는 미정 |
| Terra 정책 | **결정 D07:** 새 기본 cache/live는 Terra 전용이며 키 누락·인증 실패 시 DINO로 자동 전환하지 않는다. 기존 Terra primary + DINO 인증 fallback 구현은 보존한다. 구체 선택 방법·인터페이스는 후속 결정. 현재 코드는 아직 변경하지 않음 |
| Peg 완전 가시성 | **결정 D04·D05:** 에피소드 초기 Main/base_camera 관측에 기존 `strict_100_pass`를 적용하는 방향으로 진행한다. 이후 조작 중 부분 가림은 허용한다. Native 512 검증은 별도이며 기존 데이터·split은 아직 변경하지 않음 |
| 기록·진단 | **결정 D10:** Live rollout의 모든 episode 영상을 반드시 저장한다. 영상 off를 선택 가능한 기본 기능으로 제안했던 내용은 채택하지 않는다. 상세 상태/Graph 진단 범위는 별도 논의이며 자동 artifact publishing은 추가하지 않음 |
| Graph 확인 도구 | **결정 D11:** Graph 3D 검사 도구를 현재 레포에서 실제 실행 가능한 기능으로 유지하며 삭제하지 않는다. 시간축·World/EE 표시·노드 입력 확인과 필요한 runtime 의존성을 보존한다. 통합 모델 확정 후 새 포맷 연결과 실제 실행을 검증 |

Cache/live의 동일성을 확인할 후속 검증 후보는 같은 raw RGB-D·mask·calibration·
robot/TCP snapshot을 두 입력 경로에 넣어 World/EE별 Graph와 provenance를 비교하는
것이다. Robot filter/gate 결과와 device 유지도 포함한다. Tracker까지 비교할 때는
같은 episode prefix와 reset 상태가 필요하다. 저장 mask 재사용과 online mask
재생성을 같은 검증으로 섞지 않는다. 새 테스트는 아직 작성하지 않았다.

## 8. 실제 정리를 시작할 때의 작업 순서안

1. 통합 모델 결정과 이번 주 live 전체 범위를 맞춘 뒤 현재 Git 상태·diff를 다시
   읽는다. 다른 채팅의 미완료 변경과 필요한 helper/보고서를 파일별로 확인한다.
2. 검토한 소스·설정·문서를 명시적으로 stage해 백업 commit을 만들고 SHA를 기록한다.
   현재 19개 로컬 commit을 포함한 이력 보존도 확인한다. 원격 push는 이번에 하지 않았다.
3. Git 밖 보존 목록에 경로·크기·hash·run/source 연결·보존 위치를 기록하고 필요한
   독립 사본을 만든다. 이 사본의 복원을 검증한다. 기존 HF 게시 증거를 재사용할 수
   있어도 새 업로드를 자동 실행하지 않는다.
4. 재현 검증 결과로 active runtime 의존성, 과거 실험 재현 경로, 재생성 가능한
   자료를 구분한다. 현 runtime이 참조하는 실험 helper를 먼저 떼어내거나 삭제하지 않는다.
5. 합의한 구조에 맞춰 작은 단위로 이동·연결하고, 기존 실행의 의미와 artifact
   identity를 유지하며 필요한 검증을 수행한다. 기본 진입점 문서도 함께 갱신한다.

위 순서는 제안이며 이번 조사에서 commit·backup copy·이동·삭제를 실행하지 않았다.

## 9. 이번 실행과 검증의 한계

레포 `.venv`의 Python **3.11.13**으로 관련 소스 **1,064개**를 AST parse했고
syntax error는 없었다. 정적 closure는 동적 import, 파일 asset, 외부 dataset,
shell launcher를 완전히 포괄하지 않는다. Runtime 회귀 테스트 통과를 의미하지 않는다.
기본 `python3`와 레포 Python 버전이 달라 최종 parse는 `.venv`로 확인했다.

설치 metadata는 Torch 2.14.1, ManiSkill 3.0.1, SAPIEN 3.0.3,
Gymnasium 1.3.0이다. `torch.cuda.is_available()`는 **False**였다.
이번에는 simulator/renderer, CUDA/MPS 추론, robot filter/gate의 실제 GPU 실행,
학습·rollout·유료 API·원격 복원·게시를 실행하지 않았다. GPU 동작·시간 수치는
기존 보고서의 증거와 이번 소스 확인을 구분해서 읽어야 한다.

생성물은 이 보고서와 ignored `outputs/repo_cleanup_preflight_20261104/`의 조사
근거뿐이다. 기존 소스·config·공통 logger/publisher API를 변경하지 않았다.
문서·읽기 전용 조사이므로 학습/공통 인프라 단위 테스트는 반복 실행하지 않았다.

## 10. 사용자와 확정한 결정 기록

### D01 — 독립 오프라인 평가 지원

2026-11-04 KST, 사용자가 A안을 선택했다.

- 학습 중 검증을 유지하면서 저장된 Graph cache·정답·checkpoint로 독립 평가할 수 있게 한다.
- 재학습이나 simulator rollout 없이 저장 모델 비교·과거 결과 검증을 지원하는 기능 범위다.
- Live 평가는 현재 관측에 대한 실제 행동과 작업 수행을 평가하는 별도 역할을 유지한다.
- 세 기본 실행 흐름은 여전히 후보이며, 독립 오프라인 평가는 보조 기능으로 포함한다.
- 평가 지표, 모델 의존 입력, 폴더·명령어·인터페이스는 통합 모델 결정 이후 정한다.
- 이번 결정은 기능 범위 합의다. 평가 코드 구현·이동·실행은 하지 않았다.

### D02 — Graph cache는 저장된 mask 재사용 (초기 합의, D03으로 구체화)

2026-11-04 KST, 사용자가 저장 mask 사용을 선택했다. 주된 이유는 Graph cache
재생성마다 Terra API 비용이 반복 발생하는 것을 피하기 위해서다.

- 저장된 RGB·Depth에 대응하는 mask를 재사용해 Graph cache를 생성한다.
- Mask 생성과 저장 mask를 소비하는 Graph cache 생성을 기능상 구분한다.
  저장 mask 사용 경로에서 Terra/SAM2를 매번 다시 실행하는 것을 기본으로 삼지 않는다.
- 이 결정은 cache 경로에 관한 것이다. Live는 현재 관측에서 퍼셉션을 실행하며,
  cache와 공통 Graph 생성 로직을 사용하는 방향을 유지한다.
- 저장할 mask가 로봇 필터·손목 gate 적용 전인지 후인지, mask 누락·불일치 처리,
  재생성 절차와 출처·버전 기록은 아직 결정하지 않았다.
- 폴더·저장 형식·명령어·인터페이스를 확정하거나 cache 코드를 수정하지 않았다.

후속 논의에서 사용자가 RGB 프레임·Grounding 결과를 1차 cache로 두고, Graph 생성
방식이 정해지면 2차 Graph cache를 만들어 학습·평가하는 대안을 제시했다.
이는 아직 검토안이다. 위 초기 mask 재사용 합의를 최종 cache 경계로 고정하지 않는다.
로봇 필터·손목 gate 적용 전 mask를 필수 저장하자는 직전 제안도 확정되지 않았다.

이후 SAM2까지 포함하는 P02와 filter/gate의 단계 경계를 논의했고, 아래 D03에서
기능 경계를 확정했다. 위 미확정 설명은 그 이전 논의 시점의 기록이다.

### D03 — 에피소드 순서의 두 단계 cache와 filter/gate 경계

2026-11-04 KST. 사용자가 SAM2까지 처리한 1차 cache와, filter/gate를 포함한
2차 Graph cache 생성 경로에 동의했다.

- 1차: 원본 RGB·Depth·camera calibration·robot/TCP 상태에 Grounding 기록과
  로봇 필터·손목 gate 적용 전 SAM2 mask를 연결해 보존한다. 원본 파일 참조를 사용할 수 있다.
- Episode와 camera·timestep 연결 및 시간 순서, BC 관측·행동의 시점 대응을 보존한다.
- 2차: 저장 mask에서 공통 로봇 필터·손목 gate와 Graph 생성을 실행해 Graph cache를 만든다.
- 같은 1차 입력에서 2차를 재생성할 때 Terra와 SAM2를 반복 실행하지 않는다.
- Live는 현재 관측에서 퍼셉션을 실행하고 공통 filter/gate·Graph 연산을 사용한다.
  기존 GPU 처리와 live의 robot 준비/SAM2 overlap을 보존하는 방향을 유지한다.
- 사용자 제공 성능 근거: RTX 3090에서 해당 후처리가 약 7ms/프레임으로 감당 가능하다.
  이번 채팅에서 새로 측정한 수치가 아니며, SAM2 포함 전체 지연으로 해석하지 않는다.
- 폴더·저장 포맷·config schema·모델 인터페이스·세부 실패 정책은 아직 확정하지 않았다.
  이번 변경은 결정 기록뿐이며 cache 구현·이동·실행은 하지 않았다.

### D04 — Peg 가시성 조건은 에피소드 초기 관측에 적용

2026-11-04 KST. 사용자가 초기 관측 기준인 A안을 선택했다.

- 에피소드 초기 관측에서 Peg가 완전히 보이는 데이터를 선택한다.
- 이후 파지·삽입 등 조작 과정에서 발생하는 부분 가림은 허용한다.
- 모든 학습 관측에서 완전 가시성을 요구하는 조건을 적용하지 않는다.
- 어느 camera에서 충족해야 하는지, 완전 가시성의 구체 기준과 판정 방법은 아직 정하지 않았다.
- 기존 Stage3의 초기 Main 가시성 조건과 동일하다고 자동 판정하거나 과거 split을
  그대로 채택한 것은 아니다. 실제 데이터 선별·삭제·cache 재생성은 하지 않았다.

이후 기존 구현과 비용 기록을 확인했고 D05에서 초기 Main의 strict 기준을 채택했다.
위 camera/판정 미정 설명은 D04 당시의 기록이다.

### D05 — 초기 Main 관측의 strict_100_pass 채택

2026-11-04 KST. 사용자가 추가 탈락량과 수집 시간 기록을 확인한 뒤
"일단 엄격한 기준으로 가자"고 결정했다.

- 현재 데이터 준비 계획은 기존 `strict_100_pass` 판정을 사용한다.
- 에피소드 초기 Main/base_camera 관측에서 Peg 전체가 화면 안에 있고 양끝·파지부 및
  카메라를 향한 예상 표면의 모든 pixel이 Depth로 확인되어야 한다.
- 기존 near-plane·frustum 수치 허용 및 1.5mm depth 일치 기준을 임의로 바꾸지 않는다.
  물체 뒷면 등 자기 가림 표면까지 요구하는 조건은 아니다.
- 적용 시점은 초기 관측이다. 이후 파지·삽입 과정의 부분 가림은 D04대로 허용한다.
- 기존 128 기록을 근거로 채택했으며 native 512에서의 동작·탈락률은 별도 확인한다.
  검증 결과로 기준 변경이 필요하면 다시 논의한다.
- 기존 데이터·manifest·split·collector 코드는 아직 변경하지 않았다. 특정 episode 삭제,
  새 데이터 수집, split 보충/재배정이나 cache 생성을 실행한 결정은 아니다.

### D06 — 1차 cache의 Main/Wrist 원본 관측은 native 512×512

2026-11-04 KST. 사용자가 두 카메라 원본 관측 해상도를 512로 선택했다.

- Main/base_camera와 Wrist/hand_camera의 RGB·Depth를 native 512×512로 수집·보존한다.
- 1차 cache의 SAM2 물체 mask와 camera calibration·frame 연결은 그 원본 관측에 대응해야 한다.
- 기존 128 관측을 확대하는 것을 native 512 원자료 확보로 대신하지 않는다.
- 현재 주 live의 두 카메라 native 512 설정과 맞추는 방향을 유지한다.
- D05의 초기 strict 가시성은 Main에 적용한다. 두 카메라 해상도를 같게 정했다고
  Wrist에서도 초기 완전 가시성을 필수로 요구하는 것은 아니다.
- 모델 내부 resize·전처리, SAM2 내부 해상도, Graph 점 수·feature·구조·학습 설정은
  이 결정으로 확정하지 않는다. 통합 모델 결정과 관련 검증 이후 별도로 다룬다.
- 기존 128 원자료·과거 재현 경로를 수정하거나 삭제하지 않았다. 실제 native 512
  데이터 확보 방식, 수집 규모·split과 실행 일정은 아직 결정하지 않았다.
- 이번 작업은 문서상의 결정 기록이며 collector 변경·수집·재캡처·API 호출은 하지 않았다.

### D07 — 기본은 Terra 전용, 기존 DINO fallback 구현은 보존

2026-11-04 KST. 사용자가 기본 경로 A를 선택하면서 B의 구현도 남기도록 요청했다.

- 앞으로 정리할 기본 cache/live의 grounding 정책은 Terra 전용이다.
- 기본 실행에서 키 누락·401/403 인증 실패 때문에 Grounding DINO로 자동 전환하지 않는다.
  Terra를 사용할 수 없는 오류는 명시적으로 처리한다.
- 현재 Terra primary + DINO 인증 fallback 구현과 관련 재현 경로는 삭제하지 않는다.
  기본 경로와 보존된 대안의 선택을 구분하며, 구체적인 config/CLI/폴더 구조는 미정이다.
- Terra가 유효한 null/미검출을 반환하는 것과 인증/통신 오류는 구분한다.
  Null의 후속 scan이나 재시도 세부 정책을 이번 결정으로 새로 확정한 것은 아니다.
- 보존한 대안을 실행한다면 그 정책과 실제 backend를 식별할 수 있어야 한다.
  DINO 산출물을 Terra 전용 산출물로 잘못 취급하지 않도록 cache 재사용 조건을 검토한다.
- 기존 ExperimentLogger identity·project/group/run ID 규칙은 유지한다.
- 이번에는 결정 문서만 변경했다. 현재 `LiveGrounding`의 자동 fallback 동작이나
  과거 source를 수정하지 않았고 API를 호출하지 않았다.

### D08 — 향후 512 원자료는 새 expert 궤적으로 수집

2026-11-04 KST. 사용자가 기존 궤적의 512 재렌더보다 새 expert 수집인 B안을
선택했다. 의도는 과거 상태 복원·재렌더가 추가 실패 원인이 되는 것을 피하는 것이다.

- 향후 필요한 native 512 원자료는 해당 환경에서 expert를 실행해 관측·행동을 함께 수집한다.
- 기존 128 궤적의 상태 복원·512 재렌더를 새 기본 데이터 준비 경로로 삼지 않는다.
- 과거 원자료와 재렌더 연구 코드·결과의 보존·재현 여부는 cleanup 기준으로 별도 다룬다.
  이번 선택은 기존 자료의 삭제 허가가 아니다.
- 수집 규모·split·세부 수집 프로토콜·실행 일정은 정하지 않았고 수집을 시작하지 않았다.
- 사용자가 상위 목적을 repository cleanup으로 재확인했다. 데이터/cache 관련 논의는
  정리 요구사항으로 보존하고 현재 작업을 새 데이터 구축 프로젝트로 확대하지 않는다.
- 통합 모델 결정 공유 전 구조 확정을 보류하고, 실제 정리 전 변경 확인·백업 commit 및
  Git 밖 자료 보존 확인을 선행한다는 기존 순서는 유지한다.

### D09 — HF 중심의 완전 보존·복원 문서화 후 과거 코드 제거

2026-11-04 KST. 사용자가 P03과 HF 기본 보존소 제안에 동의했다.

- HF를 기본 보존소로 사용하고 Google Drive에 이미 있는 자료의 복원 위치도 문서에서 연결한다.
- 과거 실험은 소스 commit/필요한 source archive, 입력과 실험 산출물 전체를 보존한다.
  최종 checkpoint만이 아니라 중간·실패·진단 결과와 metadata/sidecar도 목록에 포함한다.
- 기존 원격 사본은 hash로 검증해 재사용하고 누락분을 보완한다. 전체 원래 경로와
  run/source 관계가 보존되도록 inventory와 mapping을 남긴다.
- Docs에 실제 접근 가능한 code commit, 고정된 HF revision/경로 또는 Drive 보존 위치,
  archive/member hash와 복원·실행 방법 및 검증 결과를 기록한다.
- 원격 보존과 복원을 검증한 과거 코드를 현재 Git tree에서 제거한다. Git 이력은 유지하며,
  복원 자료가 누락됐거나 검증되지 않은 대상은 삭제를 보류한다.
- 현재 runtime이 쓰는 helper, ignored 가시성 source 및 유지할 DINO fallback은 필요한
  기능을 보존·이관한 뒤 과거 wrapper 제거를 검토한다. 이름만으로 제거하지 않는다.
- 실제 구조 정리 전 미커밋 변경 확인과 backup commit, Git 밖 자료 보존 확인을 선행한다.
  통합 모델 결정 공유 전 모델·최종 인터페이스를 확정하지 않는다.
- 이번에는 방침을 문서화했다. 아직 backup commit·upload·코드/산출물 삭제를 실행하지 않았다.
  구체적인 대상·게시 계획과 검증 상태를 만들면서 기존 공통 publisher를 재사용한다.

### D10 — Live rollout 영상은 항상 저장

2026-11-04 KST. 사용자가 "영상도 무조건 저장"으로 결정했다.

- 실제 simulator에서 수행하는 live rollout은 매 episode 영상을 반드시 저장한다.
  학습에 수반되는 live 평가와 독립 live 평가에 같은 방침을 적용한다.
- 영상 기록을 끄는 일반 실행 옵션을 두자는 직전 제안은 채택하지 않는다.
- 성공·작업 실패 등 episode 결과와 영상이 연결되도록 보존한다. 중단·인코더 오류의
  구체 처리 정책은 구현 시 검토하며, 이를 이유로 영상 없이 정상 완료한 것으로 처리하지 않는다.
- 영상 생성·저장 비용을 포함한 실제 실행 성능을 확인한다. 기록을 생략한 수치를
  영상 필수 rollout의 전체 성능으로 표시하지 않는다.
- 영상의 구체 layout/codec·저장 구조와 상세 상태·mask/Graph 진단의 필수 여부는
  이번 결정으로 확정하지 않는다. 기존 ExperimentLogger/산출물 연결을 유지한다.
- 이 결정은 live rollout 영상에 관한 것이다. 저장 Graph만 사용하는 오프라인 평가나
  cache 생성 전체에 새 영상 생성 의무를 확장하는 것으로 해석하지 않는다.
- 이번에는 문서만 갱신했다. Recorder·live 코드 변경, rollout·영상 인코딩은 실행하지 않았다.

### D11 — Graph 3D 검사 도구는 실행 가능한 현재 코드로 유지

2026-11-04 KST. 사용자가 "실제로 실행 가능한 코드로 유지되어야" 하며 "제거하면 안 된다"고 명시했다.

- 기존 `visualization/`의 Graph 3D 검사 기능을 현재 레포의 유지 대상에 포함한다.
- 과거 코드 복원 문서·원격 archive만 남기는 D09의 제거 대상으로 이 도구를 취급하지 않는다.
  예전 환경을 복원해야만 사용할 수 있는 형태로 옮기는 것은 이 요구를 충족하지 않는다.
- 저장 Graph를 읽는 서버/loader/프런트엔드와 필요한 데이터 연결·의존성·테스트·사용
  안내를 함께 보존한다. 소스 파일만 남기고 import나 실행 경로가 끊기는 정리는 하지 않는다.
- 저장된 실제 Graph를 로드해 3D 표시, 시간축 이동, World/EE 표시와 노드 입력 확인을
  수행하는 것을 cleanup의 실행 검증 범위에 포함한다.
- 통합 모델의 Graph 형식이 정해진 뒤 필요한 연결을 맞춘다. 현재 Task07/09-C 등 과거
  형식 지원을 근거로 새 통합 형식도 이미 지원한다고 주장하지 않는다.
- 기존 사전 조사 정적 closure는 cache/train/live 진입점 중심이므로, 실제 제거 목록을
  만들 때 viewer 의존성을 별도로 포함해야 한다. Viewer가 참조하는 과거 helper도
  검토 없이 삭제하지 않는다.
- 이번 결정은 Graph 검사 기능의 보존이다. Live Graph를 모든 frame에 저장하는지,
  과거 보고서 catalogue 전체의 최신 호환을 유지하는지까지 새로 확정하지 않는다.
- 이번에는 문서만 갱신했으며 viewer 삭제·이동·기능 변경이나 새 포맷 구현은 하지 않았다.

### D12 — 별도 모델 확정안 없이 현재 주 live를 기준으로 모듈화

2026-11-04 KST. 사용자가 별도 모델 확정안이 없으며 현재 주 live로 진행하되 각 부분을
갈아끼울 수 있게 해달라고 명시했다. 앞선 확정안 대기는 취소한다.

- `graph/`, `gnn/`, `controller/`를 각각 반복 개발할 component 영역으로 삼는다.
- `vla/`는 최소 공통 계약·통합·실행 조정을 맡고 component 구현을 중복 복사하지 않는다.
- 현재 E0/E1·feature·학습/평가 동작을 기준으로 추출하고 미래 설계의 최종안으로 고정하지 않는다.
- 통합부가 GRU 내부를 직접 가정하지 않도록 상태 관리도 분리한다.
- Robot filter·wrist gate·GPU/World·EE·영상·실행 가능한 뷰어 보존 요구를 유지한다.
- 공통 코드 안정화와 모델 weight freeze를 구분한다. Weight 일괄 동결은 요청하지 않았다.

실제 backup·변경·검증과 남은 정리는 [모듈화 보고서](repo_cleanup_modular_live_20261104.md)를 따른다.

## 11. 검토 중인 제안 — P01: Grounding cache와 Graph cache의 두 단계

2026-11-04 KST. 사용자의 제안에 대한 평가이며 **최종 합의·구현은 아니다**.
여기서 Grounding은 Terra의 물체 bbox 결과를 의미하는 것으로 잠정 해석했다.
이후 사용자가 SAM2 mask까지 포함하는 1차 cache를 제안했으므로 최신 검토는 P02를 따른다.

### 제안의 타당성

- 외부 API로 얻은 관측 해석과 변경 가능한 Graph 생성을 분리한다. 같은 관측과
  Grounding 결과로 Graph 후보를 비교할 수 있고, 이 범위에서는 Terra 재호출을 피한다.
- SAM2·mask 후처리·로봇 필터·gate·sampling·Graph 구성의 변경 여지를 남긴다.
  어떤 변경이 어떤 산출물을 무효화하는지는 별도로 기록해야 한다.
- 최종 학습에는 선택한 생성 방식으로 고정한 Graph cache를 사용해 반복적인
  퍼셉션·Graph 생성을 줄일 수 있다. 독립 오프라인 평가에도 같은 입력을 재사용한다.

### 한계와 보완 제안

1. **BBox cache는 SAM2 계산 cache가 아니다.** 현재 tensor vision adapter는 scan
   시점의 bbox로 SAM2를 초기화하고 prefix 순서대로 mask를 만든다. BBox만 저장하면
   Graph cache를 재생성할 때 SAM2도 재실행된다. 필요하다면 mask를 생성 조건별로
   재사용하는 선택적 중간 산출물로 둘 수 있다. 별도 필수 cache 단계로 확정하지 않는다.
2. **RGB·bbox만으로 현재 3D Graph를 재현할 수 없다.** 원본 Depth, camera calibration,
   robot/TCP 상태, camera·episode·timestep 연결도 보존해야 한다. 원본 파일을 가리키는
   검증된 참조를 사용할 수 있으므로 영상 파일을 단계마다 복사할 필요는 없다.
3. **Grounding 시점도 입력의 일부다.** 현재 Terra는 매 프레임 호출되지 않고 초기화·
   미발견 scan에 사용된다. BBox/null 결과와 해당 camera/frame, 실제 backend 및 공개
   model/prompt 조건을 연결하고, tracker reset·처리 prefix를 재생할 수 있어야 한다.
   연구 중 scan 정책을 바꿔 새로운 frame의 Grounding이 필요하면 기존 cache만으로
   충분하지 않을 수 있다. 모든 frame의 API 처리를 기본 전제로 삼지 않는다.
4. **Cache 불일치는 추가 API 호출의 자동 허가가 아니다.** 관측·해상도·prompt·backend
   변경 시 기존 Grounding의 재사용 가능성을 검사하고, 필요한 새 생성은 명시적인
   별도 작업으로 다루는 안을 제안한다. 이 오류 처리 정책은 아직 사용자와 미합의다.
5. **Graph가 최종 확정될 때까지 모든 cache 실험을 미룰 필요는 없다.** 후보별 버전의
   작은 cache로 검증하고, 본 학습·평가에 사용할 버전을 고정하는 접근을 제안한다.
   원본 1차 cache를 덮어쓰지 않고 각 2차 cache가 사용한 입력·생성 조건을 식별한다.
6. **Cache/live의 공통 계산을 유지한다.** Offline은 저장 Grounding을 공급하고,
   live는 현재 관측으로 Grounding을 얻되 이후 처리 로직을 가능한 한 공유한다.
   학습용 Graph cache는 현재 요구처럼 GNN 학습에 쓰일 입력을 보존해야 하며,
   GNN 출력 embedding을 고정 저장하는 방식으로 임의 변경하지 않는다.

현재 소스에서 `LiveGrounding`의 작은 bbox cache는 프로세스 메모리의 제한된 LRU이며,
이 제안의 영속적인 dataset cache와는 다르다. 새 두 단계 cache가 이미 구현된 것으로
문서화하지 않는다. 이번 후속 확인도 소스 검토이며 SAM2/GPU/API 실행·성능 측정은 하지 않았다.

## 12. P02: 에피소드 순서의 SAM2 cache와 Graph cache (기능 경계는 D03으로 채택)

2026-11-04 KST. 사용자는 SAM2까지 처리한 결과에 원본 관측·관련 상태를 연결하고,
BC episode의 시간 순서대로 1차 cache를 보존한 뒤, Graph 생성 방식별 2차 cache를
만드는 안을 제안했다. 후속 논의에서 **아래 filter/gate 경계를 포함한 기능 범위는
D03으로 채택했다. 코드·파일 형식·최종 인터페이스를 확정하거나 구현한 것은 아니다.**

### 평가

Graph 구성의 변경이 주요 반복 작업이라면 이 경계는 비용과 유연성을 함께 확보한다.
같은 1차 cache로 2차 cache를 다시 만들 때 Terra와 SAM2를 반복 실행하지 않아도 된다.
Sampling·edge·좌표계 등의 변경은 그 변경이 보존 관측과 호환되는 범위에서 별도
Graph cache로 비교할 수 있다. 대신 mask 저장 공간이 추가되고, SAM2·초기화·입력
전처리 조건이 바뀌면 해당 1차 산출물의 재사용 가능성을 확인해야 한다.
동일 Grounding 조건으로 SAM2만 바꿀 경우 기존 bbox 기록을 재사용할 여지도 남는다.

### 1차 cache의 내용에 대한 제안

- Episode와 각 camera의 시간 순서를 보존하고, 원본 RGB·Depth·camera calibration·
  robot/TCP 상태와 per-frame object mask를 연결한다. 원본 참조를 사용해도 된다.
- Terra bbox/null 결과와 scan·물체 초기화 frame, camera/object 식별, 실제 backend,
  공개 model/prompt 조건 및 SAM2 weight·설정·전처리 조건을 연결한다.
- BC 관측과 action의 시점 대응을 보존한다. 대응 action이 없는 terminal observation은
  label이 있는 학습 관측과 구분할 수 있어야 한다. 구체 저장 schema는 아직 정하지 않는다.
- Mask를 해석할 때 빈 mask, 미초기화·미검출, 유효성/누락을 혼동하지 않도록 상태를
  보존하는 안을 제안한다. 세부 실패 처리 정책은 별도 결정 사항이다.

### 시간 순서와 재현성

에피소드별 순차 저장은 적절하다. 다만 저장 순서만으로 Grounding event와 tracker
동작이 재현되지는 않으므로 실제 Grounding/초기화 시점 기록도 필요하다.
Mask 생성 역시 episode 시작에서 reset하고 그 시점까지의 관측으로 순차 처리하는
현재 live의 causal 계약과 맞추는 것이 필요하다. 현재 tensor adapter는 순차 timestep과
episode 일치를 검사하고 `num_frames=timestep+1`로 현재 frame을 처리한다.
이 계약은 이번에 소스로 재확인했으며 새 inference를 실행하지 않았다.

저장 순서와 학습 sampling은 별도 문제다. Cache에서는 episode/time 연결을 보존하고,
학습의 episode/chunk 순서·batching·sequence 정책은 통합 모델 확정 후 결정한다.
기존 split/parent 연결도 유지해 추후 분할·재현 검증에 사용할 수 있게 한다.

### Filter/gate 단계 경계 — D03에서 확정

1차에는 로봇 필터·손목 gate 적용 전 SAM2 mask를 두고, 공통 filter/gate와 Graph
생성을 2차 생성 경로에서 처리한다. 이 경우 그 후처리도 API/SAM2
재실행 없이 변경할 수 있고 현재 gate에 필요한 raw-mask 통계가 보존된다.
**후속 사용자 동의로 이 기능 경계는 D03에서 확정했다.**

Live에서는 현재 관측으로 같은 퍼셉션·후처리·Graph 연산을 수행한다. Offline에서
저장 mask를 읽는 것과 live에서 SAM2를 호출하는 입력 공급 방식은 다를 수 있다.
현재 GPU robot 준비와 SAM2 overlap 같은 live 실행 스케줄까지 디스크 cache 구조에
맞춰 바꾸는 것으로 해석하지 않는다. 공통 연산 재사용과 실행 스케줄을 구분한다.

## 13. 기존 초기 Peg 가시성 판정 로직 확인

2026-11-04 KST. 사용자가 새 기준을 논의하기 전에 기존 구현부터 확인하도록 요청했다.
**기존 로직을 찾았으며 저장 관측의 현재 재계산도 통과했다. 새로운 선별 기준을
추가하거나 기존 데이터·runtime 소스를 변경하지 않았다.**

### 구현과 실제 호출 경로

- 판정 함수: [기존 geometry.py](../../outputs/task10g/initial_visibility_audit/geometry.py)의
  `evaluate`(35행). 독립적인 초기 상태의 GT 기하·실제 Depth 판정이다.
- [기존 protocol.json](../../outputs/task10g/initial_visibility_audit/protocol.json)에
  `camera=base_camera`, `frame_index=0`, detector/tracker 미사용이 명시되어 있다.
- [Stage1 pipeline](../../graph/initial_pool10i/pipeline.py)은 audit의 `geometry.evaluate`를
  import해 같은 판정을 회귀 확인하고 새 candidate에 적용한다.
- [Stage2 load_and_verify](../../graph/task10i_stage2/load_and_verify.py)는 기존 audit의
  `primary_geometry_depth_pass` ID를 읽어 기존 episode를 보존 대상으로 선택한다.
- [Stage3 worker](../../graph/task10i_stage3/worker.py)의 `reset_gate`(77행)는 원래 Stage1
  구현을 불러와 reset 재현을 확인하고 `visibility['requested_pass']`를 요구한다.
  실패하면 `VISIBILITY_REGRESSION`이다.

가시성 gate 자체와 물리 유효성·expert 성공·diversity/split 선택은 별개다.
가시성 통과만으로 최종 BC 데이터의 모든 적격 조건을 충족하는 것은 아니다.

### 실제 기본 선별 기준: requested_pass

1. **초기 Main/base_camera 관측**을 사용한다. Wrist 가시성을 필수로 요구하지 않는다.
2. Peg의 GT OBB 양 끝의 모든 꼭짓점, 즉 전체 8개 꼭짓점이 near plane 앞이며 화면
   연속 범위 안에 있어야 한다. Near plane은 1cm, 경계 수치 허용은 0.0001px다.
3. 흰색 끝·빨간색 끝·기존 expert의 파지 목표 띠가 각각 화면 안에 있고 최소 1 raster
   pixel의 예상 표면을 가져야 한다. 끝 영역 폭은 각 끝에서 17.5mm, 파지 띠는 목표
   중심에서 ±8.75mm다. 실제 finger rubber-pad 폭과 기존 expert OBB helper를 따른다.
4. 이 세 영역의 모든 카메라 방향 예상 표면 픽셀에서 실제 Depth가 유효하며,
   GT 기하로 계산한 앞면 깊이와 **1.5mm 이내**로 일치해야 한다.

따라서 bbox가 영상 안에 있기만 하면 통과하는 로직도, Terra/SAM2 예측 품질로
데이터를 고르는 로직도 아니다. GT는 이 데이터 선별·검증 경계의 입력이다.
Graph/모델의 예측 입력에 GT를 추가하는 결정으로 해석하지 않는다.

### 더 엄격한 별도 기준: strict_100_pass

`requested_pass`에 더해 카메라를 향한 Peg 예상 표면의 **모든 raster pixel**이
Depth로 확인되어야 한다. 뒤쪽 면 등 자기 가림 표면을 요구하는 것은 아니다.
기본 판정은 양끝·파지부 밖의 표면 일부가 가려져도 통과할 수 있다. 이 차이는
기존 `test_optional_full_silhouette_gate_is_stricter`에서 명시적으로 검증한다.
Stage1/Stage3의 기본 선별은 이 strict flag가 아닌 `requested_pass`를 사용한다.

### 이번에 실행한 확인

- 기존 `test_geometry.py`의 **6개 테스트 통과**. 잘린 끝, 파지부 가림, invalid depth,
  카메라 뒤의 Peg, 기본/strict 차이를 포함한다.
- 현재 원본 audit와 Stage1 audit 사본의 `geometry.py` SHA256이 모두
  `50cc79370317b0c8994522b27d77a79267b0022ec69e7dc974fbe86a998eb6d7`로
  기존 결과의 source hash와 일치한다. Protocol hash도 기존 결과와 일치한다.
- 현재 설치된 `compute_grasp_info_by_obb`와 `Panda.build_grasp_pose` 소스가 기록된
  helper 소스와 각각 exact 일치함을 확인했다.
- 저장된 **320개 초기 관측 전부**를 현재 `.venv`에서 재계산했다. 기본·strict·
  full-in-frame flag 및 세 필수 영역의 통과/픽셀 수/지지 픽셀 수 불일치는 **0개**다.

| 기존 저장 관측 | Episode | requested_pass | strict_100_pass |
| --- | ---: | ---: | ---: |
| Train | 272 | 82 | 80 |
| Validation | 48 | 23 | 22 |

이는 과거 F1/G 관측에 대한 확인이며 보완 수집 이후 Stage3 최종 데이터의 개수를
다시 센 결과가 아니다. Stage1 신규 후보의 과거 독립 검증 자료에는 CV primary
926개 일치와 별도 GL 경계 1픽셀 차이도 보존돼 있다. 이를 이번 새 renderer 실행으로
표시하거나 GL 차이를 없었던 것으로 처리하지 않는다.

공통 `ExperimentLogger`를 W&B disabled로 재사용했다.
이번 재계산 run ID: `de183c999341484da28166d33e115422`.

- [현재 재계산 결과](../../outputs/repo_cleanup_preflight_20261104/existing_visibility_recheck/summary.json)
- [현재 재계산 스크립트](../../outputs/repo_cleanup_preflight_20261104/recheck_existing_visibility.py)
- [기존 합성 테스트](../../outputs/task10g/initial_visibility_audit/test_geometry.py)
- [과거 후보 풀 보고서](10i_stage1_main_visible_candidate_pool.md)

재실행 명령:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s outputs/task10g/initial_visibility_audit -p test_geometry.py -v
```

현재 재계산 스크립트는 기존 결과를 덮어쓰지 않도록 보호하므로 다른 재실행에는
별도 output root가 필요하다. 이번에는 macOS CPU 저장 배열·SAPIEN Pose 수학만
실행했다. Simulator reset/step, renderer, CUDA, SAM2와 API 호출은 실행하지 않았다.
SAPIEN import 시 system libvulkan 미검출 후 builtin fallback 경고가 있었지만
이번 계산에서 renderer를 생성하지 않았다.

### 정리 및 512 적용에서 주의할 점

핵심 gate 소스·protocol·테스트는 **ignored `outputs/`에 있고 Git 추적 대상이 아니다**.
현재 추적된 Stage1/Stage3 wrapper는 audit와 `/workspace/...` 실행 경로에 의존한다.
이 소스와 helper/입력/증거를 실제 정리 전 보존 목록에 추가해야 하며, 재사용 시
실험 경로 의존성을 검토해야 한다. 이번에는 이동·공통 모듈화·commit하지 않았다.

이번 전수 재계산 입력은 **128×128**이다. Gate의 핵심 projection은 입력 크기를
읽지만 Stage1의 physical/descriptor·검증 wrapper에는 128 가정이 남아 있다.
Native 512에서 재캡처·gate 비교·wrapper 대응을 검증하기 전에는 이 전체 수집 경로가
512에서 그대로 동작한다고 선언하지 않는다. 조사 당시 primary/strict 채택은 미정이었고,
후속 논의에서 D05로 strict 기준을 채택했다. Native 512 실행 검증은 여전히 남아 있다.

## 14. 엄격한 기준의 추가 탈락과 수집 시간 조사

사용자 요청에 따라 과거 926개 초기 상태의 strict flag, Stage3 최종 manifest 및
수집 batch의 wall time을 집계했다. [전체 분석 보고서](repo_cleanup_strict_visibility_cost_20261104.md)를 따른다.
무작위 후보는 기본 301/926, strict 299/926이었다. Stage3 R/D는 각각 272→270개,
공통 Validation은 48→46개로 계산된다. 집계 작업 자체에서는 기준·manifest를 변경하지 않았다.
후속 논의에서 D05로 strict 기준을 채택했지만 실제 manifest는 아직 변경하지 않았다.
시간 기록은 과거 128 raw 수집이며 native 512·Terra/SAM2·Graph cache 총 시간과 구분한다.

## 15. 512 원자료 확보 방식의 기존 구현 확인 — D08 이전 검토 근거

D07 이후 다음 데이터 준비 논의를 위해 기존 native 재캡처 코드를 읽었다.
[resolution capture](../../graph/experiments/resolution_perception_graph_20261103_v1/capture.py)는
seed로 geometry를 재구성하고 기록된 qpos/qvel·Peg/Box pose·clock을 복원한 뒤,
physics step 없이 각 해상도의 native RGB-D를 얻고 TCP·상태 불변 등을 검사한다.
다만 특정 서버·실험 manifest·경로에 묶인 연구 코드이며 일반 dataset 변환기로
즉시 사용할 수 있다고 판정하지 않는다.

[기존 128/512 보고서](10i_current_graph_128_512_accuracy_20261104.md)는 이 방식의
4 Train Parent, t0..20 대응 관측을 사용한 검증을 기록한다. 이는 전체 BC episode나
모든 기존 raw의 복원 필드 완비 검증이 아니다. 기존 궤적의 native 512 재렌더를
우선 검증·재사용할지, native 512에서 expert episode를 새로 수집할지는 이 조사 당시 미정이었다.
후속 D08에서 새 expert 수집을 선택했으며 재렌더를 기본 준비 경로로 채택하지 않았다.
이번에는 소스·기존 보고서를 읽었고 재캡처·수집·서버 실행은 하지 않았다.

## 16. P03 — 복원 문서와 원격 보존 후 과거 코드 제거 (D09로 채택)

2026-11-04 KST. 사용자는 docs에 이전 코드 복원용 commit hash, HF 또는 Google Drive
위치를 명시하고 과거 코드를 제거하되 실험 산출물을 누락 없이 올리는 방식을 제안했다.
후속 사용자 동의로 **D09의 cleanup 방침으로 채택했다. 업로드·코드 삭제는 아직 시작하지 않았다.**

### 제안의 해석과 권장 범위

- 최신 checkout은 기본 실행 경로와 현재 필요한 runtime·공통 운영 인프라·테스트를
  중심으로 정리한다. 과거 모든 CLI의 최신 코드 호환을 계속 유지하는 것을 목표로 삼지 않는다.
- 과거 코드 제거는 현재 Git tree에서의 제거다. Git history 재작성·원격 backup 삭제와
  구분하며, 실제 정리 전 필요한 미커밋 변경의 backup commit을 만드는 순서는 유지한다.
- 현재 주 live가 사용하는 과거 실험 폴더의 helper, 초기 가시성 판정의 ignored source,
  유지하기로 한 DINO fallback은 실제 의존성에 따라 보존·이관한 뒤 과거 wrapper 제거를
  검토한다. 폴더 이름만으로 삭제 대상을 결정하지 않는다.
- 실험 산출물의 완전 보존은 best/last만 골라 게시하는 작업과 다르다. 원자료·GT·
  split·mask/Graph cache·checkpoint/재개 상태·metrics·영상·진단 배열/CSV/figure·
  local/offline 로그·실패/중간 결과와 metadata/sidecar/source snapshot을 목록화한다.
- 같은 내용의 중복 파일은 기존 원격 사본의 hash를 검증해 재사용할 수 있다.
  다만 원래 경로와 artifact/run/source 연결이 누락되지 않도록 mapping을 보존한다.
  인증 파일·token·설치/download cache는 실험 산출물과 구분하며 공개 범위를 바꾸지 않는다.

### Docs에 남길 복원 기록안

| 기록 | 필요한 내용 |
| --- | --- |
| 실험 식별 | Task·원래 run ID·실험 목적·대표 결과·원본 보고서 |
| 코드 | 실제 접근 가능한 Git repository URL과 전체 commit SHA·원래 entrypoint/path. Dirty/untracked 실행 소스는 별도 source archive와 hash |
| Commit 의미 | 실험 당시 commit과 이번 보존용 commit을 구분. 보존용 commit을 과거 실험의 실행 commit으로 바꿔 적지 않음 |
| 산출물 | HF repository의 고정 revision과 파일 경로 또는 Drive file ID/보존 버전·archive hash. 필요한 의존 archive까지 연결 |
| 완전성 | 파일별 경로·크기·hash inventory, archive member 목록, source/run/원래 경로 mapping |
| 실행 환경 | Package/source/외부 weight 조건, device/backend, config·seed·normalizer 등 입력 의존성 |
| 복원 절차 | 새 위치에서 source/data를 받는 명령과 alias 적용, 실행 명령, 기대 결과 및 검증 상태 |
| 삭제 판정 | 원격 객체 검증·실제 해제/참조 확인·필요한 실행 검증 결과, 미검증 항목과 보류 대상 |

단순한 local commit hash나 수정될 수 있는 폴더 링크만으로 복원 완료를 선언하지 않는다.
이미 공개된 archive가 있어도 현재 inventory와 비교해 누락·불일치를 확인한다.
모든 payload 보존 검증과 해당 실험의 실행 재현 검증은 구분해 기록한다.
복원 불가능하거나 필요한 입력이 누락된 자료는 삭제 보류 대상으로 남긴다.

### 순서안과 현재 상태

범위/의존성 확인 → 미커밋 source backup commit → 산출물 inventory → 기존 원격 사본
대조 및 누락 게시 → remote hash/고정 위치·복원 검증 → docs 복원 안내 작성 → 현재
runtime 의존성 이관 → 합의한 과거 코드 제거·관련 링크/기본 실행 확인 순서를 제안한다.
통합 모델 구조는 공유받기 전 확정하지 않는다.

기존 HF 공통 publisher와 maintenance/manifest/receipt를 재사용한다. 일반 publisher가
metadata·tracking sidecar 전체와 raw trajectory를 자동 게시하지 않는 현재 제약이 있어,
완전 보존에는 inventory 기반 명시적 archive/의존성 게시가 필요하다. 별도 team uploader나
run ID 규칙을 만들지 않는다. 후속 D09에서 HF를 기본 보존소로 선택했고, 기존 Drive
자료도 연결하기로 했다. 구체 게시 대상과 실행 계획은 준비·검증해야 한다.

기존 [Stage3 복원 안내](10i_stage3_hf_dependency_completion_20261102.md)는 고정 revision,
의존 archive, 원래 경로 alias와 712개 raw/GT 참조 검증을 연결한 선례다. 다만
[과거 전체 게시 기록](hf_remaining_publication_20261031.md)은 여전히 일부 미완료를
기록하므로 전체 실험이 이미 누락 없이 올라갔다고 단정하지 않는다.
