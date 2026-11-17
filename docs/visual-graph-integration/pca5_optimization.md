# PCA 5×5 최적화와 20Hz 반복 처리 검증

초기 grounding은 실행·시간 측정에서 제외했다. 기존 ep000의 초기 검출 box를 그대로
사용하며, 반복 구간의 tracking·데이터 이동·geometry 준비 시간을 측정했다.
PCA 이웃·depth gate·minimum support·normal 방향·normal variation 정의는 유지했다.
기존 `geometry.py`, Task 02-B 비교 구현·결과, 확정 interface는 변경하지 않았다.

**CPU geometry p95는 78.39→12.97ms, normal p95는 62.69→8.80ms로 감소했다.**
CPU와 CUDA의 동일 입력 비교에서 validity와 5/10/20/30° 반응 변경은 0개였다.
GPU geometry p95는 **3.22ms**, 초기 준비 후의 tracking 연결 구간은 **입력 도착→geometry
준비 p95 47.31ms**, 약 **20.02Hz**였다. 최종 Graph 데이터 구성은 아직 측정 범위 밖이다.
[성능 HTML](../../visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/performance.html),
[시간·queue 그림](../../visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/performance.png)과
[검증 기록](../../visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/verification.json),
[이 작업의 코드·문서 diff](../../visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/implementation.patch)에 근거를 모았다.

## 1. 실행·산출물

| 실행 | Run / 결과 |
| --- | --- |
| 원래 PCA5 profiling | [`4236d33efb5b453895dbf9048467acca`](../../visual/results/pca_optimization/4236d33efb5b453895dbf9048467acca/profile.txt) |
| CPU 최종 비교: 121프레임 × 3회, synthetic 288조건 | [`03d9777a004349e3a794147e4e55a2f0`](../../visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/optimization.json) / [HTML](../../visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/optimization.html) |
| CUDA Torch 기준 구현 | [`aa996b8c135d40b09116f5389125580f`](../../visual/results/pca_optimization/aa996b8c135d40b09116f5389125580f/cuda.json) |
| CUDA normal kernel 결합 | [`f53cd6b2edcf414ba31bb41fe114f70c`](../../visual/results/pca_optimization/f53cd6b2edcf414ba31bb41fe114f70c/cuda.json) |
| CUDA normal + feature kernel 결합 | [`eaa2d61078b841a396e63f7868a6b90c`](../../visual/results/pca_optimization/eaa2d61078b841a396e63f7868a6b90c/cuda.json) |
| CPU tracking + 병렬 PCA, 초기 이후 11프레임 | [`9de26edb2e774ffcbed16d00eb89439e`](../../visual/results/pca_optimization/9de26edb2e774ffcbed16d00eb89439e/streaming.json) |
| CUDA tracking 연결, compile 없음 | [`d78e9574a02946108484e471385e5c03`](../../visual/results/pca_optimization/d78e9574a02946108484e471385e5c03/streaming.json) |
| CUDA 최종 연결 비교 | [`a34de8fa4976434e9be4356eb8db198a`](../../visual/results/pca_optimization/a34de8fa4976434e9be4356eb8db198a/streaming.json) |

CPU는 현재 Mac arm64/Python 3.11.13/Torch 2.14.1/NumPy 2.4.6/SciPy 1.17.1이다.
CUDA는 사용자가 지정한 Vast.ai **57410033**, **RTX 3090 24GB**, driver 615.71.09,
Linux/Python 3.12.3/Torch 2.8.0+cu128/Triton 3.4.0/NumPy 2.4.6/SciPy 1.17.1이다.
전용 디렉터리 `/workspace/pca-normal-opt-20261024`와 별도 venv를 사용했다.
CUDA를 Mac/MPS에서 실행했다고 주장하지 않는다. 서로 다른 host의 CPU/GPU 시간으로
순수한 하드웨어 speedup을 계산하지 않는다.

데이터·mask SHA-256은 Task 02-B와 동일하다.

- Episode: `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`
- Tracking mask: `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`
- 기존 `geometry.py`: `36d20b7873d5ef9e1381e3bbe319da89e16fa5b44455488786dadd250bd814a9`

121프레임은 같은 저장 simulator episode의 반복 관측이다. 독립 episode나 물리 센서
검증으로 일반화하지 않는다. Synthetic 288조건은 Task 02-B의 원래 240조건과 추가 방향
48조건을 그대로 재사용했다. GT는 평가 이후의 ROI·정답 normal에만 사용했다.

## 2. 무엇을 최적화했는가

기존 frame 0, 두 물체 × 10회 profiling에서 고유값 계산 약 23%, covariance의 `einsum`
약 16%가 관측됐다. 나머지도 patch 처리와 메모리 작업에 분산돼 있었다. 고유값 계산만
교체해서 해결할 문제로 가정하지 않았다.

CPU 구현은 다음을 결합한다.

1. 같은 연산 순서의 dense broadcasting으로 backprojection 수행. 기존 XYZ와 bitwise 비교했다.
2. 동일 물체의 완전한 5×5 support이며 내부 40개 인접 link가 모두 gate를 통과하는 곳은
   연결 탐색을 생략하고 box moment를 재사용한다. 전체 관측점 중 **83.724%**가 이 경로를 쓴다.
3. 나머지는 **patch 내부**의 4-connected component를 25-bit 정수로 계산한다.
   프레임 전체에서 연결된 다른 조각을 무조건 포함하지 않는다.
4. Float64 covariance를 유지한다. 완전한 patch는 고정 좌표 원점을 빼고 1차·2차 moment를
   계산해 cancellation을 제한한다. 경계 patch는 중심 상대 좌표를 사용한다.
5. 3×3 대칭 행렬의 닫힌 형태 고유값·고유벡터 계산을 비교했다. 잔차가 크거나 퇴화·rank
   판정에 가까운 곳은 원래 `numpy.linalg.eigh`/`torch.linalg.eigh`로 fallback한다.
   `z=f(x,y)` 회귀나 근사 평면 방향으로 estimator를 교체하지 않았다.
6. 각 물체 raw mask의 bounding box와 1px halo에서 기존 feature 연산을 실행하고,
   결과를 원래 `[H,W]` 배열에 복원한다. GT ROI 축소·erosion·sampling이 아니다.
7. Peg/Box의 독립 normal·feature 계산을 persistent worker 2개로 병렬 실행한다.

CUDA는 일반 Torch 구현과 kernel 결합을 비교했다. Depth·tracker mask·XYZ·normal·variation을
같은 device에 유지하며, normal kernel은 연결 탐색·float64 covariance·고유벡터 계산을
관측 중심당 한 warp에서 수행한다. Feature kernel은 원래의 네 이웃 jump·경계·variation을
합쳐 계산한다. Camera-facing 방향, minimum 6점, rank/gap gate, invalid NaN 정책은 같다.
선택한 backend를 명시해야 실행되며 기존 NumPy adapter의 `.cpu().numpy()` 기본 동작을
전역 변경하지 않았다.

## 3. CPU 시간

아래 값은 같은 최종 실행의 **121프레임 × 방법별 3회**, 총 363 sample이다.
두 물체 모두 포함한다. 방법별 warmup 후 실행 순서를 회전했다.
로딩·GT·지표·그림은 제외하고 build/normal/features의 실제 시간을 측정했다.

| CPU 방법 | Build p95 | Normal p50 / p95 | Feature p95 | 전체 p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| 원래 PCA5 | 7.219ms | 21.947 / 62.695ms | 9.596ms | 36.856 / 78.392ms |
| Moment 재사용 + library eigh | 2.260ms | 10.684 / 26.543ms | 3.620ms | 14.834 / 32.020ms |
| 위 + 닫힌 형태 고유벡터 | 2.255ms | 6.882 / 15.176ms | 3.607ms | 10.827 / 20.847ms |
| 위 + 두 물체 병렬 처리 | 2.234ms | 5.394 / 8.795ms | 2.306ms | 8.742 / 12.974ms |

Normal p95 약 **7.13배**, 전체 p95 약 **6.04배** 개선이다. 단계별 percentile의 합은
전체 percentile과 같지 않다. CPU normal ≤10ms, geometry ≤25ms 개발 목표를 충족했다.
이 결과는 현재 두 물체·해상도·관측량에서의 값이며 다른 부하와 물체 수에 대한 보장은 아니다.

## 4. CUDA 시간과 전송

각 방법은 실제 121프레임 × 3회다. Resident 입력과 CPU 입력 전송 포함을 번갈아 측정했다.
CUDA 동기화를 포함한 wall time이며 device event 경과 시간도 JSON에 함께 저장했다.
Device event 경과 시간은 kernel 실행 사이의 간격도 포함하므로 연산 kernel 시간의 합과
동일하게 해석하지 않는다. 아래 범위에는 tracking이 없다.

| RTX 3090 방법 | Resident p50 / p95 | CPU 입력·전송 포함 p95 |
| --- | ---: | ---: |
| 일반 Torch float64 PCA5 | 25.505 / 30.115ms | 30.456ms |
| Normal kernel 결합 | 8.590 / 9.797ms | 10.209ms |
| Normal + feature kernel 결합 | 2.189 / 3.222ms | 3.669ms |

GPU 성능은 실제 실행한 수치이며 CPU 결과로 추정하지 않았다. Float32/float16 covariance로
정확도 조건을 완화하지 않았다. MPS는 float64 PCA 계약을 유지할 수 없어 이 backend의
실행 대상으로 사용하지 않았다.

## 5. 정확도·구조 보존 검증

CPU 세 최적화 후보는 원래 PCA5와 동일한 121프레임·288조건에서 다음을 만족했다.

- Normal/variation validity 차이 **0개**.
- 나머지 모든 feature field 차이 **0개**. Raw mask, point support, depth jump, 모든 경계,
  normal neighbour count를 포함한다.
- Normal 최대 차이 약 **3.53×10⁻⁶° 이하**, variation 최대 차이 약 **3.53×10⁻⁶° 이하**.
- 5/10/20/30° 반응의 위치 변경 **0개**.

GPU 최종 후보도 실제 121프레임·synthetic 288조건에서 validity와 나머지 feature field
차이 및 네 threshold의 반응 위치 변경이 모두 0개다. 실제 normal 최대 차이는
약 1.17×10⁻¹⁰°, variation 최대 차이는 1.37×10⁻⁵°다. Synthetic 최대 차이는 각각
1.67×10⁻⁹° / 3.42×10⁻⁶°다.

같은 입력의 구조 지표도 유지됐다. CPU 원래 PCA5와 최적화 후보 모두 Peg 평면 error p95
**6.246°**, variation p95 **7.534°**, Box rim normal **2910개**,
`jump OR variation>20°` 반응 **2187/3425개(63.854%)**다. 실제 seam·rim ROI와 threshold를
변경하지 않았다. GPU도 같은 support·경계 feature와 threshold 위치를 비교했다.

따라서 속도 개선으로 Task 02-B의 기존 구조 보존 성능을 유지했다. PCA5 자체의 crease
혼합·rim 반응 손실·부분 지원 편향을 해결한 것은 아니다. 비교 대상은 원래 PCA5이며
Task 01 baseline의 정확도로 되돌린 결과가 아니다.

## 6. 초기 grounding 제외, tracking 연결과 20Hz

입력 RGB-D는 메모리에 준비하고 50ms 간격으로 도착하도록 replay했다. 입력 drop 없이
같은 순서로 모든 프레임을 처리했다. 모델 로딩·처음 box prompt·초기 kernel 준비 시간은
시작 비용으로 따로 저장했다. Frame 0 이후의 모든 tracking step은 시간에 포함했다.
예정 입력 시각부터 geometry가 준비될 때까지 queue 대기와 처리 시간을 함께 측정한다.

GPU tracking은 같은 SAM2.1 Small checkpoint, image_size=1024, BF16 autocast,
기존 causal frame buffer·memory pruning을 쓴다. CPU tracking은 float32다.
미래 RGB에 접근하지 않는다. Compile 시험은 기존 코드에 있던 image encoder compile
옵션만 사용한다. 추가 encoder warmup은 이미 관측한 frame 0만 사용한다. 최종 조건에서는
같은 RGB로 tracker 2 step을 준비한 뒤 state를 reset하고 실제 frame 0에서 새로 시작한다.
실제 측정 state의 latest=0을 assert하며 미래 RGB나 실제 frame 1을 미리 사용하지 않는다.
Compile 전후 mask도 저장해 별도로 비교한다.

| 연결 경로 | Tracking p95 | Geometry p95 | 처리 p95 | 관측 처리율 | 50ms 도착 deadline |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU tracking + 병렬 PCA, 11 step | 1191.177ms | 14.695ms | 1205.009ms | 0.961Hz | 11/11 초과 |
| CUDA tracking + normal kernel, 120 step | 56.424ms | 9.514ms | 65.142ms | 16.100Hz | 120/120 초과 |
| CUDA compile + 전체 kernel 결합 + 초기 warmup, 120 step | 44.635ms | 2.978ms | 47.194ms | 20.019Hz | 1/120 초과 |

최종 조건의 **입력 도착→geometry 준비 p95는 47.314ms**, p99는 48.814ms,
최대 50.570ms다. Queue 대기 p95는 0.131ms로 누적되지 않았고, frame drop은 0개다.
120 step 약 6초의 이 episode에서 20Hz 처리와 p95 50ms 목표를 충족했다.
**모든 프레임의 50ms hard deadline을 보장한 결과는 아니다**(1/120 초과).
또한 p95와 50ms의 수치 차이가 약 2.69ms뿐이므로 남은 Graph 준비 비용을 넉넉히
수용한다고 주장할 수 없다. Percentile 예산은 단순 합산으로 보장되지 않는다.

**이는 최종 Graph 입력 전달의 완료 검증은 아니다.** 현재 경로에는 Structural/Surface
Sampling, 분류, intra-object edge, 최종 Vision Output packaging이 아직 없다. 남은 기능을
새로 구현하거나 시간을 0으로 간주하지 않았다. 실제 camera/render 지연도 이 측정에 없다.
Graph가 사용할 최종 데이터가 준비되는 시점까지의 20Hz는 연결 후 다시 측정해야 한다.

## 7. 추가 tracking compile 비교와 시작 비용

단계별 결과를 보존했다. 아래 모든 CUDA 시험은 실제 frame 1–120을 포함하며, 느린 frame을
사후 삭제하지 않았다. 각 실행의 frame별 값은 원래 UUID JSON에 남아 있다.

| CUDA 조건 | 처리 p95 | 도착→geometry p95 | 도착 deadline 초과 |
| --- | ---: | ---: | ---: |
| Compile 없음, normal만 kernel 결합 | 65.142ms | 1434.746ms | 120/120 |
| Encoder compile, normal만 kernel 결합 | 53.716ms | 787.485ms | 120/120 |
| Encoder compile + 전체 kernel 결합 + encoder warmup 3회 | 47.938ms | 53.248ms | 9/120 |
| 위 + 같은 초기 RGB로 tracker warmup 2 step 후 새 state | 47.194ms | 47.314ms | 1/120 |

Encoder만 준비한 세 번째 조건에서는 첫 실제 반복 step이 78.165ms였다. 같은 초기 RGB로
연산을 준비한 최종 조건에서는 첫 step이 44.700ms로 줄었다. Warmup은 실제 측정 state에
가짜 관측을 남기지 않도록 reset 후 다시 frame 0에서 초기화한다. Cold encoder compile
실행의 시작 비용은 **153.41초**, 캐시가 준비된 최종 실행의 시작 비용은 **16.38초**였다.
이는 grounding 비용이 아니라 모델 로딩·초기 prompt·compile/warmup 비용이다.
첫 grounding 자체는 캐시된 box를 사용해 실행하지 않았다.

Compile/warmup 경로의 tracking mask는 bitwise 동일하지 않았다. 최종 경로와 eager의
120프레임 mask micro IoU는 Peg **99.8907%**, Box **99.8861%**, 최저 frame IoU는
각각 **98.0843% / 98.2143%**다. 차이를 compile 하나의 기여로 분해하지 않았다.
PCA의 동일 입력 동일성 검증과 이 tracking 입력 변경을 구분한다.

[추가 GT·구조 평가](../../visual/results/pca_optimization/70a194604bd04ef1b975df5d90ade61d/compile_quality.json)는
실제 frame 1–120, 같은 GT 영역에서 모든 예측을 먼저 계산한 다음 수행했다.

| 항목 | Eager tracking | 최종 compile/warmup tracking |
| --- | ---: | ---: |
| Peg mask GT micro IoU | 97.9935% | 98.0036% |
| Box mask GT micro IoU | 97.4898% | 97.5048% |
| 확인된 Peg 평면 normal error p95 | 6.2461° | 6.2461° |
| GT seam band의 jump OR >20° 수 | 46751 / 94044 | 46766 / 94044 |
| GT 가시 rim의 관측 normal 수 | 2684 / 3202 | 2686 / 3202 |
| GT 가시 rim의 jump OR >20° 수 | 2010 / 3202 | 2012 / 3202 |

이 episode에서 큰 구조 보존 저하는 관측되지 않았다. Frame 0을 제외한 별도 causal mask
평가이므로 원래 121프레임의 rim 3425개 집계와 raw count를 직접 비교하지 않는다.
작은 IoU 차이를 일반적인 tracking 개선으로 해석하지 않으며 독립 episode 검증은 남아 있다.

SAM2의 optional `_C` extension이 없어 hole postprocessing을 건너뛰었다는 경고가 CPU와
GPU 모두 있었다. 해당 설치 상태를 숨기지 않으며 extension을 강제로 설치하거나 tracking
알고리즘을 교체하지 않았다. Compile autotuning에서는 shared memory가 부족한 후보가
제외됐다. 최종 실행 성공 여부와 별개로 초기 준비 비용은 raw JSON의 `startup_excluded_ms`에
남겼다. 초기 비용을 제외한 지표와 첫 반복 프레임에서 생긴 지연을 구분한다.

## 8. 코드·테스트·보존

추가 파일:

| 파일 | 역할 |
| --- | --- |
| [`pca_optimized.py`](../../visual/runtime/test/pca_optimized.py) | CPU moment/연결성/고유벡터/feature/두 물체 병렬 비교 구현 |
| [`pca_torch.py`](../../visual/runtime/test/pca_torch.py) | 명시적인 device-resident Torch 입력·출력, float64 계약 |
| [`pca_triton.py`](../../visual/runtime/test/pca_triton.py) | CUDA 연결성·normal·feature kernel과 퇴화 fallback |
| [`run_pca_optimization.py`](../../visual/runtime/test/run_pca_optimization.py) | CPU paired 전체 평가·fidelity·시간·그림 |
| [`benchmark_pca_cuda.py`](../../visual/runtime/test/benchmark_pca_cuda.py) | CUDA resident/전송 포함 시간·실제/합성 동일성 |
| [`benchmark_pca_streaming.py`](../../visual/runtime/test/benchmark_pca_streaming.py) | 기존 causal tracker 재사용, grounding 제외·20Hz queue 측정 |
| [`test_pca_optimized.py`](../../visual/runtime/test/test_pca_optimized.py) | 경계·퇴화·두 물체·비연속 layout·CPU/CUDA 계약 테스트 |

최종 검증:

- Mac runtime suite **66개 실행: 63 PASS / CUDA 3 SKIP / FAIL 0**.
- RTX 3090 새 최적화 테스트 **13/13 PASS**, 세 CUDA backend 포함.
- 공통 logger/artifact **42/42 PASS**.
- CPU 후보 각각 실제 121프레임·synthetic 288조건 fidelity PASS.
- CUDA normal/feature 결합 후보 실제 121프레임·synthetic 288조건 fidelity PASS.
- 새 raw mask의 compile/warmup 비교 및 같은 GT 영역의 구조 평가 완료.
- `git diff --check` PASS. 기존 파일의 변경·삭제 0개.

실험은 공통 `ExperimentLogger`와 `experiment_artifacts.digests`를 사용했고
`WANDB_MODE=disabled`로 로컬 기록했다. 원격 run의 원래 UUID/config/metadata/metrics를
로컬에도 복사했다. HF/W&B 게시, git stage/commit, simulator 변경은 하지 않았다.
새 인스턴스를 만들거나 다른 인스턴스를 시작·중지하지 않았다.

기존 파일 790개의 시작 hash snapshot과 비교해 변경·삭제 **0개**를 확인했다.
기존 Task 02-B의 미추적 파일과 원래 결과도 그대로다. 모든 새 실행은 UUID 디렉터리에
분리했다. 코드와 실행 source hash는 각 JSON에 있다. CPU 결과에는 실행 당시 source를,
GPU 결과에는 실행 source snapshot을 함께 보관했다. Remote checkout의 Git 정보가 없는
한계와 local dirty tree를 commit hash만으로 재현할 수 없는 한계를 구분한다.

## 9. 재현

Repository root의 CPU 환경:

```bash
MPLCONFIGDIR=/tmp/pca-mpl WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_pca_optimization --repeats 3
TQDM_DISABLE=1 MPLCONFIGDIR=/tmp/pca-mpl WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.benchmark_pca_streaming --device cpu --cpu-parallel --frames 12
MPLCONFIGDIR=/tmp/pca-mpl WANDB_MODE=disabled .venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
```

실제 CUDA 환경:

```bash
MPLCONFIGDIR=/tmp/pca-mpl WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.benchmark_pca_cuda --backend triton_fused --synthetic --repeats 3
TQDM_DISABLE=1 MPLCONFIGDIR=/tmp/pca-mpl WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.benchmark_pca_streaming --device cuda --backend triton_fused --frames 121 --compile-image-encoder --encoder-warmup 3 --tracker-warmup-steps 2
MPLCONFIGDIR=/tmp/pca-mpl WANDB_MODE=disabled .venv/bin/python -m unittest visual.runtime.test.test_pca_optimized -v
```

권장 후속 검토 대상은 CPU의 명시적 `parallel=True, solver='analytic'` 경로와 CUDA의
`backend='triton_fused'` 경로다. 20Hz 후보에는 GPU resident 처리와 별도 검증한 tracking
compile/warmup 조건이 필요하다. 기본 estimator 채택은 변경하지 않았으며 PCL 비교,
최종 분류·sampling·Graph·Stale Geometry 계약 변경은 이 작업에 포함하지 않았다.
