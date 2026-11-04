# Task10-I: live 로봇 필터 재측정 — 2026-11-04

필터 본체의 속도는 이전 최적화 결과와 비슷하다. 이전 **7.159ms**와 동일한
입력·측정 경계에서 현재 코드는 **7.028ms**였다. 같은 실행에서 과거 코드를
재측정한 값도 **6.997ms**여서, 새 성능 개선으로 해석하지 않는다.

다만 실제 기본 live의 로봇 필터 호출에는 검사·snapshot·worker 호출이 더해진다.
SAM2를 제외하고 이 경계를 모두 재면 **9.587ms**다. 따라서 live 전체 로봇 필터
호출도 그대로 7ms라고 설명할 수는 없다. 실제 SAM2와 병렬 실행한 별도 tracking
입력에서는 SAM2 이후 잔여 **6.205ms**, SAM2 호출 앞부분까지 포함한 추가 wall
span **10.632ms**였다. 이 둘은 필터 단독 시간과 서로 다른 지표다.

이번 작업은 측정용 runner와 문서만 추가했다. production 로봇 필터, 손목 gate,
Graph, 모델, 시뮬레이터 및 실행 설정은 변경하지 않았다.

## 1. 동일 저장 입력, 필터 단독 비교

RTX3090, 512×512, base/hand 두 카메라 합계, 원본 scene
`10i_s1_0921_t080`, r0 실제 SAM2 저장 mask를 사용했다. 과거 실험의 mask,
depth, calibration, 모든 현재 FK surface vertex 해시가 정확히 일치했다.

각 variant를 3회 준비 실행하고 15회 측정했다. 순서는 정방향/역방향을 교대로
적용했다. 각 호출 전후 CUDA synchronize와 CPU monotonic wall clock을 사용했다.
검증·출력·파일 로딩은 타이머 밖이다. static ray/depth 변환 캐시는 준비된 상태지만,
현재 pose/depth/hull의 결과를 다음 호출에 재사용하지 않는다.

| 경로 | 중앙값 ms | Q25–Q75 ms | 최소–최대 ms |
| --- | ---: | ---: | ---: |
| 이전 최적화의 원래 기록 | 7.159 | 7.147–7.188 | 7.118–12.187 |
| 당시 코드를 이번 서버 상태에서 재실행 | 6.997 | 6.967–7.159 | 6.935–7.779 |
| 현재 필터, 동일한 과거 측정 경계 | **7.028** | 6.963–7.290 | 6.934–8.248 |
| 현재 기본 live 필터 호출, 준비 포함 | **9.587** | 8.967–10.922 | 8.443–14.706 |
| 보정 경로용 전체 화면 로봇 판정과 clean 재사용 | **36.728** | 35.974–37.798 | 35.709–41.060 |

과거 측정 경계는 CPU 센서 depth/calibration 및 사전 계산된 현재 FK surface와
상주 CUDA raw mask에서 CUDA clean mask까지다. 호출 내 hull/bounds 준비,
필요한 업로드·depth 변환·clipping·GPU 완료가 포함된다. FK 계산 자체는 제외한다.

현재 동일 경계의 7.028ms는 과거 기록보다 1.83% 작고, 동시에 재실행한 과거 코드보다
0.45% 크다. 차이는 이번 분포 범위에 있으며 성능이 유지된 결과로 판단한다.

live 행은 native CUDA RGB/depth 입력이 준비된 뒤 실제 기본 생성자의
`CurrentGeometry.build`를 실행했다. `_check_mount`와 metadata 검사, data와
camera/surface snapshot, worker submit, fused depth 변환, 준비 join, GPU clipping,
CUDA clean 결과 완료를 포함한다. 이미 계산된 raw CUDA mask를 callable로 즉시
반환하여 SAM2가 준비 시간을 가려 주지 않도록 했다. `_build_prepared`의 진입에서
clean mask만 반환하므로 손목 gate/Shape-First/Graph는 실행되지 않는다.
초기 입력 업로드는 live 입력 계약에 따라 타이머 밖이다.

live 경계의 9.587ms는 과거 7.159ms보다 수치상 33.9% 크지만 입력·호출 경계가
다르므로 필터 알고리즘 자체의 33.9% 회귀로 해석하지 않는다. 동시에 live 호출의
추가 비용을 무시해서도 안 된다.

마지막 행은 `mask_transform`이 필요한 보정 경로다. 보정은 기존 mask 밖으로 픽셀을
추가할 수 있어서 mask union 대신 전체 화면을 판정하고 진단 결과를 유지한다.
실제 `CurrentGeometry.build(..., mask_transform=...)` 분기를 실행하되 보정 callback은
입력 mask를 그대로 반환한다. **CPU 보정 알고리즘·진단의 host export·gate·Graph는
제외**했다. 일반 경로 7ms와 같은 작업량이 아니며, 보정 경로 전체 소요 시간도 아니다.

## 2. 실제 SAM2와 병렬 실행한 live 로봇 필터 구간

기본 live가 사용하는 `TensorOnlineVisionAdapter`와 실제 SAM2 video predictor를
연결해 `10i_s1_0531`의 현재 관측 t0–20을 순서대로 실행했다. 두 카메라의 RGB/depth는
CUDA에 준비하고, 각 시점의 surface/data는 현재 관측으로 갱신했다. 기록된 RGB 기반
t0 box prompt를 사용해 탐지 API만 대체했다. 미래 frame은 전달하지 않았다.

모델은 기존 `sam2.1_hiera_small.pt`, 내부 image size 1024, FP32 weight,
autocast 비활성인 현재 native live 설정을 유지했다. 이전 BF16/JPEG adapter 실험의
전체 SAM2 시간과 직접 비교하지 않는다. 모델 precision이나 tracker는 수정하지 않았다.

t0은 초기화, t1–5는 causal warm progression, **t6–20의 15개 frame**을 집계했다.
동일 정적 입력을 15회 반복한 1절과 다른 표본이다. 각 live 호출의 시작/끝에서만
GPU를 동기화하고, SAM2와 필터 사이에 측정용 동기화를 추가하지 않았다.

| 동일 호출에서 잰 wall span | 중앙값 ms | Q25–Q75 ms |
| --- | ---: | ---: |
| build 시작 → SAM2 호출 시작 | 4.345 | 4.030–4.859 |
| SAM2 반환 → CUDA clean 완료 | **6.205** | 5.946–6.801 |
| SAM2 호출 외부의 추가 시간: 앞부분 + 뒷부분 | **10.632** | 10.222–11.673 |

10.632ms는 매 frame의 `(build 전체 wall span − SAM2 host-call wall span)`을 먼저
계산한 후 중앙값을 구했다. 표의 개별 중앙값을 더한 결과가 아니다. CPU 준비가
SAM2 내부와 겹친 시간은 이 외부 span에 포함되지 않는다. SAM2 adapter는 기존
mask pixel count scalar를 읽으므로 모델 결과가 CPU로 관측되는 경계가 있다.
이 값은 동일 SAM2-only 대조군을 뺀 인과적 overhead나 GPU kernel 전용 시간으로
해석하지 않는다. 추가 시간에는 live snapshot/검사/스케줄링도 포함된다.

모든 21 frame에서 준비 future는 SAM2 반환 시점 전에 완료되어 있었다. 병렬화가
실제 연결돼 동작한 것은 확인했다. 다만 앞부분 비용도 남아 있으므로 “SAM2 뒤에
6.21ms만 남는다”와 “로봇 필터 때문에 전체 경로에 6.21ms만 추가된다”는 같은
설명이 아니다. 필터 단독 비교의 대표 수치는 1절의 7.028/9.587ms다.

설치된 SAM2의 선택적 `_C` 확장이 없어 hole-fill postprocessing을 건너뛴다는 기존
warning이 발생했다. predictor 자체는 21 frame을 끝까지 실행했다. 해당 확장을
설치하거나 fallback을 변경하지 않았다. 확장 활성 환경의 결과는 이번에 검증하지 않았다.

## 3. 정확도와 실행 범위

- 과거 7.159ms source의 SHA256을 고정하고 원본 입력 해시를 assert했다.
- 네 경로 각각 warmup 3회와 timed 15회, 총 **72회**의 CUDA clean mask가 과거
  필터와 픽셀 단위로 정확히 일치했다. 각 경로는 두 카메라 × 두 object mask다.
- 실제 SAM2의 **21 frame 모두**에서 새 raw CUDA mask를 과거 필터에 그대로
  넣은 결과와 현재 live clean mask가 정확히 일치했다. 이 비교는 타이머 밖이다.
  SAM2 raw mask 자체와 과거 JPEG adapter raw mask의 동등성을 주장하지 않는다.
- artifact 수집 후 중앙값, overlap span의 산술 관계, source archive의 모든
  1,500개 파일 hash 및 현재 로봇 필터 핵심 source 일치를 다시 검증했다.
- 공통 `ExperimentLogger`를 재사용했다. run ID는
  `f7de399d6d0041debdf9313ed7c8b1c5`, 로컬 상태는 `finished`, W&B는 disabled다.
  학습·외부 publishing·새 grounding 호출은 하지 않았다.
- 로컬 macOS에는 CUDA가 없어서 syntax 및 artifact 검증만 했다. 실제 CUDA
  필터·모델은 기존 RTX3090 서버에서 실행했다. 새 시뮬레이터 reset/step,
  controller/action, 전체 live rollout, 손목 gate/Graph는 실행하지 않았다.
  저장 관측을 실제 live RF 호출 경로에 재생한 측정이다.

## 4. 환경과 재현성

RTX3090, driver 595.84, Linux x86_64, Python 3.12.15,
Torch 2.14.1+cu130, CUDA 13.0, NumPy 2.5.4, SciPy 1.18.1, Triton 3.8.0.
Torch CPU threads 4, OMP/OpenBLAS/MKL threads 각각 1, GC 활성, seed 42,
RLIMIT_DATA 80GiB, cgroup CPU quota `646736 100000`이었다.

기존 중단 프로세스 327662–327666, 357110–357111은 측정 전후 모두 T 상태였다.
재개·종료·signal 전달 없이 새 격리 디렉터리에서만 실행했다. 측정 전후 GPU는
utilization 0%, 기존 resident memory 636MiB였다.

로컬 Git HEAD는 `d42f777bd8a2a23eef4f814f5922783383c17948`, branch는
`archive/visual-graph-wip-20261102`, WIP 포함 상태다. 다른 작업의 수정을 유지하고
stage/commit/revert하지 않았다. 기존 원격 code copy에 현재 source archive를 덮은
격리 실행본을 사용했다. Git 없는 실행본의 logger `git_commit=null`/`git_dirty=false`는
clean checkout의 증거로 쓰지 않는다. 실제 실행 identity는 다음 archive/hash다.

- 실행 source archive SHA256:
  `32403c37b0aa4e79012f149c360b9684eaa2254540bde0b5e4f3b4116db3aa8d`
- 과거 filter SHA256:
  `4c7402e66752473b19e6412982e6b6e353f5c2f1877ea23b472f1b65ef63f8df`
- 현재 filter SHA256:
  `9aba621626afeca4067fc53b482f67b466b6f560f2c02370063af60084e84ea2`
- SAM2 checkpoint SHA256:
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`

## 5. 코드와 원시 결과

- [측정 runner](../../graph/experiments/robot_filter_live_timing_10i_20261104_v1/run.py)
- [실행 방법](../../graph/experiments/robot_filter_live_timing_10i_20261104_v1/README.md)
- [동일 입력 15회 원시 결과](../../outputs/task10i/robot_filter_live_timing_20261104_v1/measured/paired.json)
- [실제 SAM2 21 frame 결과](../../outputs/task10i/robot_filter_live_timing_20261104_v1/measured/sam2.json)
- [완료 기록](../../outputs/task10i/robot_filter_live_timing_20261104_v1/measured/complete.json)
- [측정 config](../../outputs/task10i/robot_filter_live_timing_20261104_v1/measured/config.json)
- [실행 로그](../../outputs/task10i/robot_filter_live_timing_20261104_v1/measured.log)
- [artifact 검증](../../outputs/task10i/robot_filter_live_timing_20261104_v1/verification.json)
- [source identity](../../outputs/task10i/robot_filter_live_timing_20261104_v1/source_identity.json)
- [실행 source archive](../../outputs/task10i/robot_filter_live_timing_20261104_v1/executed_source.tar.gz)
- [과거 7.159ms 원시 결과](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/final_512/results.json)
