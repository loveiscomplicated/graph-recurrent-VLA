# Task10-I: 손목 gate 인덱스 공유·단일 cell 정렬·kernel 융합

2026-11-04. 요청한 세 변경을 기존 live 손목 gate에 함께 적용했다.
**World gate 단독 중앙값은 같은 실행의 전후 비교에서 4.1138065 → 3.683621ms,
10.46% 감소(1.117배)**했다. 512×512, clean 유효점 29,775개인 입력을 사용했다.
accepted mask와 전체 trace는 모든 반복에서 기존 CPU 기준과 정확히 같았다.

## 변경 범위

실제 live 호출은 기존
`CurrentGeometry._build_prepared → prepare_tensor_gate → wrist_gate_tensor`를 따른다.
새 실험 전용 gate를 live와 따로 두지 않았다. 기본 CUDA gate가 아래 코드를 사용하며,
CUDA가 없을 때만 기존 CPU gate를 선택하는 정책은 유지한다. GPU 오류를 CPU로
재시도하지 않는다.

손목 gate는 **World에서만 판정**한다. 이번 변경에서 EE gate 구현이나 새로운
Graph frame 변환을 추가하지 않았다. 성능 측정에서도 EE/Graph 준비를 실행하지 않았다.

구현과 검증 파일:

- [기존 gate의 준비·연결성·World 복원](../../graph/persistent_geometry/tensor_gate.py)
- [추가한 Triton kernel](../../graph/persistent_geometry/tensor_gate_kernels.py)
- [API 설명](../../graph/persistent_geometry/README.md)
- [새 계약 테스트](../../graph/experiments/wrist_gate_compact_fused_10i_20261104_v1/test_contracts.py)
- [paired gate 단독 runner](../../graph/experiments/wrist_gate_compact_fused_10i_20261104_v1/benchmark.py)

### 1. 같은 bool mask의 인덱스 압축 공유

cell 이웃의 `valid`, AABB의 `possible`, radius 검사 결과 `keep`에서 각각 선택
인덱스를 한 번만 구한다. 두 endpoint 배열은 그 인덱스를 `index_select`로 공유한다.
기존 `a[mask]`, `b[mask]`의 두 번 동적 압축을 한 번으로 줄였다.

최초 `valid`는 평탄화한 선택 인덱스의 행 번호로 `a`를 복원하고, 같은 인덱스로
`j`에서 `b`를 가져온다. row-major 순서와 endpoint 대응을 유지한다.

### 2. cell key 한 번 정렬

기존 cell code의 `unique(sorted=True)`와 별도 stable `argsort` 중복을 제거했다.
stable `argsort` 결과에 `unique_consecutive`를 적용해 key·count·정렬 상태의 inverse를
구하고, inverse를 원래 점 순서로 scatter한다. 같은 정렬 순서로 point와 cell 구간을
준비한다. 정수 key와 overflow 검사는 유지한다.

최종 `unique(first[roots])`는 canonical component label과 첫 원본 점 기준 동률
정책을 위해 유지한다. 모든 정렬을 없앤 변경은 아니다.

### 3. 작은 GPU 연산의 kernel 융합

**World 3D 복원**을 하나의 Triton kernel로 묶었다. signed depth의 float64 값,
pixel center `+0.5`, intrinsic 및 World 행렬의 기존 연산 순서를 유지한다.
FP64 round-to-nearest division은 설치된 Triton CUDA libdevice의 `div_rn`을 사용한다.
kernel의 FMA 융합은 비활성화했다. 비연속 intrinsic·행렬·pixel stride도 처리한다.

**label 전파**는 기존 반복의 의미를 유지하며 다음 세 단계로 실행한다.

1. 입력 label을 별도 `merged` buffer에 복사한다.
2. 하나의 edge kernel이 **변하지 않는 입력 label**에서 양 endpoint의 최소 root를
   읽고 `merged`의 두 endpoint에 integer atomic minimum을 적용한다.
3. pointer-jump kernel이 `merged`를 읽어 별도의 output buffer에 기록한다.
   수렴 확인 단계에서는 동시에 입력과 다른 label 수를 block별로 기록한다.

현재 stream의 kernel 순서로 복사·edge 갱신·pointer jump 사이의 경계를 보장한다.
갱신 중인 label을 다른 edge의 입력으로 읽도록 바꾸지 않았다. 반복마다 두 label
buffer를 교대하고, 기존처럼 최대 4회 묶음의 마지막 한 단계가 수렴했는지 확인한다.
전체 반복 상한 `nc`와 마지막 짧은 묶음도 유지한다. 임의의 고정 횟수로 자르지 않는다.

CUDA를 쓰지 않는 명시적 CPU test helper는 기존 Torch 연산으로 남겼다.
171개 이웃 offset, AABB·radius predicate, 희소점의 전체 점 거리 검사, 카메라 간
projection, threshold 및 전체 trace의 필드는 이번 변경에서 유지했다.

## gate 단독 전후 측정

| 항목 | 변경 전 | 변경 후 |
|---|---:|---:|
| 중앙값 | 4.1138065ms | **3.683621ms** |
| 최소 | 3.794381ms | 3.389791ms |
| 최대 | 4.591401ms | 3.924361ms |

중앙값 차이는 **0.4301855ms**, 감소율 **10.4571%**다.
이는 세 변경을 합친 결과이며 각 변경의 기여를 따로 측정한 ablation은 아니다.
이전 별도 실행의 4.087ms를 분모로 쓰지 않고, 같은 실행에서 교대한 전후 값을 사용했다.

- 입력: `10i_s1_0531_track_t011`, 두 카메라 512×512, 손목 clean 유효점 29,775개.
- 출력: accepted 29,269개, reasons 빈 목록.
- 환경: Vast instance 58752018, RTX 3090, AMD EPYC 7543.
- Python 3.12.15, Torch 2.14.1+cu130, NumPy 2.5.4, Triton 3.8.0.
- 각 구현 warmup 5회 후 30쌍. 인접한 GPU before/after 호출의 선행 순서를 매 반복 교대.
- 매 호출 직전 CUDA를 동기화하고, `wrist_gate_tensor` 호출 직후 CUDA 완료까지
  `perf_counter_ns`로 측정. 검증용 mask 다운로드·hash 계산은 타이머 밖에서 수행.
- 변경 전 source와 kernel은 별도 package namespace에서 읽어, 변경 후 kernel을
  잘못 공유하지 않도록 했다. 두 구현에 같은 준비 완료 World view를 전달했다.

before가 먼저인 15쌍의 중앙값은 **4.105822→3.683041ms**,
after가 먼저인 15쌍은 **4.121791→3.689371ms**로 양쪽 순서 모두 개선됐다.

계측에는 World 3D 복원·연결성·주변 지지·카메라 간 일치·Python trace 및 accepted
CUDA mask 생성에 필요한 CPU 실행과 동기화를 포함한다. 입력 업로드·World view
사전 준비, `prepare_tensor_gate`의 raw/clean 검사·C3 갱신, EE/Graph 준비, SAM2,
robot filter, Graph, policy, simulator, profiler 및 파일 I/O는 제외한다.

## 별도 profiler 확인

30쌍 시간 측정이 끝난 뒤 각 구현을 별도 1회 프로파일했다. 이 시간은 위 표에
포함하지 않았다. profiler 호출 자체의 결과도 mask와 전체 trace를 검증했다.

| 관찰 | 변경 전 | 변경 후 |
|---|---:|---:|
| 실제 kernel event | 312 | **233** |
| `aten::nonzero` | 8 | **5** |
| 길이 29,775의 `aten::_unique2` | 2 | **1** |
| `aten::unique_consecutive` | 0 | **1** |
| stable `argsort` | 1 | 1 |
| `cudaStreamSynchronize` | 19 | **16** |

총 kernel event는 약 25.3% 줄었다. `cudaLaunchKernel` API 호출만 세면 310→214회지만,
새 Triton kernel의 driver 경로를 빠뜨리므로 전체 kernel 수와 구분한다.
이 개수 감소율이 wall time 감소율과 같지는 않다. 아직 동적 크기와 Python trace
반환에 필요한 동기화·GPU 임시 배열이 남아 있다.

## 정확성·회귀 검증

- 로컬 macOS: **86개 중 69개 통과, CUDA 17개 skip**.
- 원격 RTX 3090: **86개 중 85개 통과**, source-only snapshot에 실제 preregistration이
  없어 schedule 검사 1개 skip.
- 새 World 복원: random 회전·translation, FP64 intrinsic 및 비연속 calibration에서
  CPU의 기존 비융합 연산과 **float64 bit pattern이 일치**했다.
- label: 빈 edge, 긴 chain, 순서를 섞은 chain, 중복 edge, 분리 component,
  block 경계 및 마지막 짧은 batch를 기존 비융합 reference와 비교했다.
- cell 정렬: random cloud·같은 크기 component의 tie·긴 chain에서 SciPy의 canonical
  label과 일치했다. 기존 radius 경계·중복점·sparse·overlap·invalid depth·conflict
  테스트도 통과했다.
- 기존 live native CUDA, frozen NumPy, saved 29,775-point 입력, EE/World Graph와
  robot overlap·selected TensorViews 관련 회귀를 실행했다.
- 별도로 `torch.use_deterministic_algorithms(True)` 및
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`에서 새 테스트 3개와 live/frozen 동등성 1개를 실행했다.
- 변경 Python compile 및 `git diff --check`를 수행했다.

모든 측정에서 accepted mask SHA-256은
`93bf7132bbd324a5a6d994e5937bb1f72ddb2de0e0ba0b839a9c7172b3bd4540`이었다.
trace는 keys·판정·fraction·count·association 및 diameter까지 저장된 CPU 기준과
정확히 같았다. diameter의 허용오차로 차이를 숨기지 않았다.

실제 simulator+SAM2 전체 live rollout은 이번 작업에서 실행하지 않았다.
로컬 macOS에는 CUDA가 없고 GPU 검증은 원격에서 수행했다. 다른 GPU/backend와
모든 입력 분포의 성능·비트 동등성을 포괄하는 결과는 아니다. 드물게 많은 cell이나
rare 점이 나오는 입력은 따로 비교해야 한다. cold JIT compile 시간은 본 측정에서 제외했다.

## identity와 산출물

기존 `ExperimentLogger`를 `WANDB_MODE=disabled`로 재사용했다.
run ID는 **`cff789491ac548af9a6f54851906ae79`**이며 artifact 게시는 하지 않았다.
작업 시작 commit은 `d42f777bd8a2a23eef4f814f5922783383c17948`이다. 기존 미커밋 변경이
있는 작업 트리이므로 source snapshot과 SHA로 측정 코드를 구분한다. 측정 후 다운받은
config의 gate·kernel·fusion·runner SHA가 로컬 파일과 일치하는지 확인했다.

- [최종 결과·30쌍 개별 시간·trace](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/measurement/results.json)
- [config·환경·전후 source SHA](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/measurement/config.json)
- [accepted masks](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/measurement/accepted_masks.npz)
- [변경 전 profile](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/measurement/before_profile.json),
  [변경 후 profile](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/measurement/after_profile.json)
- [로컬 검사](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/local_tests.log),
  [GPU 검사](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/gpu_tests.log),
  [deterministic 검사](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/deterministic_tests.log)
- [측정 로그](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/benchmark.log)
- [실행 명령](../../graph/experiments/wrist_gate_compact_fused_10i_20261104_v1/README.md)
- [변경 전 identity](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/before_identity.json),
  [변경 후 identity](../../outputs/task10i/wrist_gate_compact_fused_20261104_v1/after_identity.json)
- [이전 gate 단독 기준](10i_wrist_gate_gpu_only_timing_20261104.md),
  [이번 후보의 검토 근거](10i_wrist_gate_remaining_optimization_review_20261104.md)

`baseline/`에 변경 전 source가 있다. `source.tar.gz`는 원격 검사에 사용한 전체 Python
source snapshot이고, `final_source.tar.gz`를 그 위에 적용하면 후속 runner·문서까지
포함된다. 이번 변경은 아직 커밋하지 않았다.
