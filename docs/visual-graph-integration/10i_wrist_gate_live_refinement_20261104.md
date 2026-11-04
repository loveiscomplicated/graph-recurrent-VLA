# Task10-I: live GPU 손목 gate 준비·CPU 동기화 최적화

2026-11-04. 요청한 후보 1번(중복 준비·업로드 제거)과 2번(CPU 동기화 감소)을
기존 live 파이프라인에 반영했다. **손목 gate 판정은 World 좌표계를 유지한다.**
Graph의 기본 EE 및 선택 World 좌표계는 gate 이후 관측·Graph 구성에 적용된다.

최종 코드로 비어 있지 않은 512px 고정 입력을 비교한 결과,
live gate 준비 구간의 중앙값은 EE Graph에서 **7.1844 → 5.4232ms (24.51% 감소)**,
World Graph에서 **6.8600 → 5.1374ms (25.11% 감소)**였다.
accepted mask와 전체 trace는 기존 GPU 구현 및 저장된 CPU 기준과 정확히 같았다.

## 적용 경로와 변경

실제 호출 경로는
`CurrentGeometry.build → _build_prepared → prepare_tensor_gate → wrist_gate_tensor`다.
기본 CUDA live 경로가 robot preparation의 float64 depth 및 raw/clean CUDA mask를
전달하는 기존 연결을 사용한다. 별도의 최적화 전용 live 진입점은 만들지 않았다.

관련 구현:

- [live 호출부](../../graph/plan_a_bc/geometry.py)
- [gate 준비·판정](../../graph/persistent_geometry/tensor_gate.py)
- [공유 view 준비](../../graph/persistent_geometry/tensor_fusion.py)
- [NumPy 호환 adapter](../../graph/persistent_geometry/wrist_gate_dispatch.py)
- [API 설명](../../graph/persistent_geometry/README.md)

### 1. 중복 준비·업로드 제거

기존 live gate 준비는 입구의 `synchronized`, Graph view 준비, gate 내부 World view
재준비에서 카메라 메타데이터를 세 번 검증했다. 이제 view 준비에서 한 번 검증하고,
같은 현재 프레임의 depth·mask·intrinsic·World 행렬을 재사용한다.

`prepare_tensor_views`는 카메라별 World 행렬도 보관한다.
`TensorViews.world_views()`는 depth·mask·intrinsic과 이미 준비한 World 행렬을 공유하며,
영상이나 행렬을 다시 업로드하지 않는다. Graph가 EE를 선택했어도 gate는 이 World
view를 사용한다. 특히 component diameter는 기존 **World 축 정렬 bounding box의
대각선 길이**로 유지한다. 현재 Graph frame의 bbox로 바꾸지 않는다.

카메라 World 행렬은 한 번 계산하고 원래 calibration dtype에서 역행렬을 구한 후
float64로 승격한다. 기존 좌표 변환 helper의 연산 순서를 유지했다. gate의 3D 복원도
준비된 World 행렬을 재사용한다. 반환되는 C3 view의 mask는 판정 후 mask로 교체하며,
World view를 다시 꺼내도 같은 최신 mask를 공유한다.

재사용 범위는 현재 snapshot이다. 두 카메라 각각의 episode·timestep·timestamp·TCP
pose와 카메라 이름을 확인하며 다른 snapshot의 view를 전달하면 실패한다.
프레임 간 센서 데이터 캐시는 추가하지 않았다.

NumPy adapter는 wrist override를 업로드 전에 선택한다. 정상적인 두 카메라 입력에서
clean mask 업로드가 기존 3회에서 2회로 줄었다. raw wrist mask 업로드를 포함하면
`torch.as_tensor` 호출은 총 4회에서 3회다. 이 NumPy adapter 개선은 아래 resident
live 측정의 시간에는 포함되지 않는다.

### 2. CPU 동기화 감소

trace의 카운트 묶음, component diameter, accepted count를 한 번의 packed readback으로
읽는다. 이전에는 이 셋을 별도로 읽었다. accepted count는 GPU에서 미리 구한
`largest & allowed`의 점 수로 결정하므로 accepted mask를 다시 합산해서 읽지 않는다.
이미지 크기 수준의 정수 카운트는 float64에서 정확히 표현된다.

World bbox extrema는 선택 mask를 적용한 GPU reduction으로 계산한다. accepted mask도
이미 알고 있는 모든 유효 pixel 위치에 candidate bool 값을 기록한다. 이 두 경로에서
boolean indexing으로 동적 크기 배열을 추가 생성할 필요가 없어졌다.

두 카메라의 clean subset/overlap 검사를 고정 크기의 bool 연산으로 수행하고,
검사 결과를 한 번에 host로 읽는다. 검사를 생략하지 않는다.

연결성 label propagation의 정수 연산은 기존과 같다. 최대 4회씩 실행한 뒤 마지막
반복 전후 label을 비교한다. fixed point는 추가 반복에서도 바뀌지 않는다. 마지막
묶음이 4회보다 짧을 때도 기존 총 반복 상한 `nc`를 지키며, 마지막 한 단계의
수렴을 확인한다. canonical component 순서와 최대 component 동률 규칙은 유지한다.

셀 key 정렬·이웃 후보 생성 및 희소점의 전체 점 거리 검사 알고리즘은 이번 범위에
포함하지 않았다. `nonzero`, 동적 compact 결과의 크기, cell dimension readback 등
나머지 동기화가 모두 제거된 것은 아니다.

## 최종 측정

| 항목 | 값 |
|---|---|
| 입력 | `10i_s1_0531_track_t011`, 두 카메라 512×512 |
| 손목 clean 유효점 | 29,775 |
| accepted points | 29,269 |
| 환경 | Vast instance 58752018, RTX 3090, AMD EPYC 7543 |
| 소프트웨어 | Python 3.12.15, Torch 2.14.1+cu130, NumPy 2.5.4, Triton 3.8.0 |
| 반복 | 각 구현·Graph frame별 warmup 5회 후 측정 30회 |
| 순서 | 인접 GPU before/after 쌍, 반복마다 실행 순서 교대 |
| 시간 | 시작 전·종료 후 CUDA synchronize를 포함한 wall time |
| 검증 | 타이머 밖에서 매 결과의 mask hash·전체 trace 확인 |
| 최종 run ID | `b5647945eca34ca499b0f6702b495ad5` |

측정 구간은 **live의 `prepare_tensor_gate` 전체**다. native CUDA 카메라 영상,
raw/clean mask, robot preparation에서 만든 float64 depth가 준비된 상태에서 시작한다.
카메라 검증·calibration 업로드, gate의 3D 복원·연결성·주변 지지·카메라 간 일치 검사,
Python trace, C3 CUDA mask 및 Graph-frame view 반환을 포함한다.

SAM2, robot filter/depth 준비, Graph 구성, policy, simulator, 검증용 mask 다운로드 및
파일 입출력은 제외한다. 이전 보고서의 gate 단독 5.79ms와는 준비 구간의 경계가
다르므로 직접 전후 수치로 비교하지 않는다. 이번 표는 같은 경계에서 실행한
기존 GPU 코드와 최종 GPU 코드의 비교다. 빈 프레임의 시간은 측정하지 않았다.

| Graph frame | 변경 전 중앙값 | 변경 후 중앙값 | 감소 | 배속 |
|---|---:|---:|---:|---:|
| EE (기본) | 7.184415ms | 5.423217ms | 24.51% | 1.325× |
| World | 6.859976ms | 5.137367ms | 25.11% | 1.335× |

| Graph frame | 변경 전 min–max | 변경 후 min–max |
|---|---:|---:|
| EE | 6.592936–8.096025ms | 5.136167–10.228844ms |
| World | 6.424625–8.008064ms | 4.838717–6.102836ms |

EE 측정에는 변경 후 10.23ms인 반복도 있었다. 이 결과는 중앙값 개선을 보여주며
최대 지연이나 실시간 보장을 뜻하지 않는다. before 먼저 실행한 15쌍과 after 먼저
실행한 15쌍을 나누어도 각각 개선됐다. EE의 각 순서별 중앙값은
7.147905→5.427296ms와 7.245995→5.411886ms,
World는 6.847096→5.134407ms와 6.872855→5.140326ms였다.

별도 untimed audit에서 두 Graph frame 모두 카메라 검증은 **3→1회**, 연결성의
host 수렴 확인은 **7→2회**였다. 이는 이 입력에서 관찰한 횟수이며 모든 분포에서
같은 반복 횟수를 보장하지 않는다. 최대 4회 묶음은 빠르게 수렴하는 입력에서 일부
GPU 정수 연산을 더 실행할 수 있다.

## 정확성·회귀 검증

최종 측정의 모든 결과는 accepted mask SHA-256
`93bf7132bbd324a5a6d994e5937bb1f72ddb2de0e0ba0b839a9c7172b3bd4540`과 일치했다.
trace의 keys, 판정, 이유, 모든 fraction·count·association 및 diameter가 저장된
CPU 기준과 정확히 같았다. diameter는 `0.21690492708412626m`이며 허용오차 비교로
대체하지 않았다. gate config·threshold 및 GT/future 입력 정책은 변경하지 않았다.

최종 테스트:

- 로컬 macOS: 72개 중 **69개 통과, CUDA 3개 skip**.
- 원격 RTX 3090: 83개 중 **82개 통과, source-only 배포에 실제 preregistration이
  없어 해당 schedule 검사 1개 skip**.
- 재사용 World/EE 행렬의 기존 helper 대비 정확한 일치, depth/mask/intrinsic 공유,
  다른 카메라 시점의 view 거부, NumPy clean mask 1회 업로드를 검사했다.
- 빈 edge·chain·permuted chain·분리 component와 짧은 마지막 batch에서 기존 매 반복
  수렴 검사와 label 결과를 비교했다. GPU radius 경계·동률·중복점·random cloud 및
  conflict·overlap·invalid depth·sparse gate 회귀 검사도 통과했다.
- 기본 live의 native CUDA 입력, frozen NumPy 입력, EE/World robot overlap 경로,
  saved 29,775-point 기준, selected TensorViews를 사용하는 CUDA Graph 회귀를 실행했다.
- CUDA 부재 시 CPU fallback 및 GPU 오류 전파 기존 검사도 통과했다.
- `git diff --check` 및 변경 Python 파일 compile 검사를 통과했다.

GPU gate와 synthetic live/Graph 연결을 원격에서 실행했다. 로컬에는 CUDA가 없어
로컬 GPU 실행으로 간주하지 않는다. 이번 작업에서 실제 simulator+SAM2 전체 live
rollout은 실행하지 않았다. 입력의 점 수·분포가 달라지면 시간과 batch 비용이 달라진다.

## 재현성과 산출물

기존 `ExperimentLogger`를 재사용했고 `WANDB_MODE=disabled`로 로컬 identity와 sidecar를
기록했다. W&B/HF artifact 게시를 수행하지 않았다.

작업 기준 commit은 `d42f777bd8a2a23eef4f814f5922783383c17948`이며,
기존 미커밋 작업이 있는 상태에서 진행했다. 이 commit만으로 최종 코드를 재현할 수
없다. 전후 source SHA와 snapshot을 별도 보관했으며, 최종 측정에 기록된 gate·fusion·
runner hash가 현재 로컬 파일과 같은지 다운로드 후 다시 확인했다.

- [실행 runner 및 명령](../../graph/experiments/wrist_gate_live_refinement_10i_20261104_v1/README.md)
- [추가 계약 테스트](../../graph/experiments/wrist_gate_live_refinement_10i_20261104_v1/test_contracts.py)
- [최종 결과: 60쌍 개별 시간·통계·trace](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/remote/results.json)
- [최종 config·환경·source SHA](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/remote/config.json)
- [accepted masks](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/remote/accepted_masks.npz)
- [변경 전 identity](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/before_identity.json),
  [변경 후 identity](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/after_identity.json)
- [로컬 테스트 로그](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/local_tests.log),
  [최종 GPU 테스트 로그](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/gpu_tests_final.log),
  [최종 측정 로그](../../outputs/task10i/wrist_gate_live_refinement_20261104_v1/benchmark_final.log)
- [입력 출처: 원래 CPU 측정](10i_wrist_gate_cpu_timing_20261104.md),
  [최적화 후보 검토](10i_wrist_gate_optimization_review_20261104.md)

원래 packed input SHA-256은
`a1d6a86482a4c9a3bc2dcf82bf35544e0340f031df65663d60fdd8c9168748a5`다.
`outputs/task10i/wrist_gate_live_refinement_20261104_v1/baseline/`에 변경 전 gate/fusion
파일이 있다. `source.tar.gz`는 초기 전체 source bundle이며 `final_source.tar.gz`를
그 위에 풀면 최종 수정 파일이 적용된다. 첫 구현의 측정은 `remote_initial/`에 별도로
보관했고, 위 표에는 identity 검사 보강 후 `remote/`의 최종 측정만 사용했다.

이번 변경은 아직 커밋하지 않았다.
