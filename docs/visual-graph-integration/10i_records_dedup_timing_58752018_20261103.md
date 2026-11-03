# Task 10-I — 기존 점별 기록 구성·3mm 중복 제거 시간

2026-11-03. 최적화 전 기본 SF 경로의 Python/NumPy 구현을 그대로 측정했다.
**기본 EE 좌표계에서 합계 중앙값30.744ms, p95 35.307ms**다.
별도 구간 진단에서는 기록 구성5.923ms, 중복 제거24.427ms로, 이 입력에서는
중복 제거가 더 큰 비중을 차지했다. 런타임 구현은 수정하지 않았다.

## 측정 범위

시작은 `numpy_observations(..., include_structural=True)`가 끝나서 CPU
`SurfaceObservation`과 그에 정렬된 구조점 flag가 준비된 시점이다.
종료는 SF의 `deduplicate(..., 0.003)`가 기록 목록과 audit를 반환하는 시점이다.

1. 카메라별 구조점 source pixel 집합 구성.
2. 기존 `current_records(..., fuse_views=False, coordinate_frame=...)` 호출.
   점별 Python dict, source ID, XYZ/normal/F5/validity 복사와 frame 검사 포함.
3. 기존 source pixel 기준 structural 표시와 카메라별 기록 합치기.
4. 기존 structural 우선·source ID 순서의 greedy 3mm 중복 제거.
   3mm grid의27개 인접 cell 조회, 실제 Euclidean 거리 **`<0.003m`** 검사,
   representative의 `deepcopy`, 중복 source ID 병합과 frame 검사 포함.

GPU→CPU 전송, packet/관측 구성, view 검사, 최종128점 FPS, 엣지 생성은 제외한다.
로봇 필터·sampling 전 wrist gate도 실행하지 않는다. 준비된 mask를 사용한다.
`PlanA.represent`가 추가로 수행하는 Legacy/Hybrid 진단 분기의 비용도 제외했다.
따라서 전체 `represent` 또는 전체 Graph 생성 시간이라는 뜻이 아니다.

## 조건

- VastAI **58752018**, AMD EPYC7543, GPU 준비는 RTX3090에서 실제 실행.
- Python3.12.15, Torch2.14.1+cu130, NumPy2.5.4.
- OMP/OPENBLAS/MKL/Torch threads1. CPU affinity·공유 host 부하는 고정하지 않음.
- 실제 `10i_s1_0844_t080` hand_camera **512×512** 입력.
- Hand Peg **1,105점**: structural1,008점 + surface97점. Box0점.
- Base는 hand depth/calibration을 복제하고 mask/관측은 비운 보조 입력이다.
  실제 서로 다른 두 카메라에서 점이 들어오는 실험이 아니며, nonempty Box도 없다.
- GPU mask projection → triton_fused → GPU Shape-First → GPU packet/관측/view
  → 실제 `numpy_observations`를 모두 **계측 밖에서** 실행해 CPU 입력을 준비했다.
- EE·World 각각20회 warmup 후100회, frame 순서를 매번 교대했다.
  CPU `perf_counter` wall time이며 계측 구간에는 GPU 호출이 없다.
- 합계 표본은 중간 clock 없이 끝까지 한 번에 측정했다. 구간별 표본은 별도의
  20회 warmup·100회 실행으로 나눠 측정했다. 이전 표본의 결과 해제, logger와
  file IO는 다음 표본의 timer 밖이다. Python GC는 기본 설정을 유지했다.

## 결과

| 좌표계 | 측정 구간 | 중앙값 | p95 |
| --- | --- | ---: | ---: |
| EE, 기본 | **기록 구성 → 3mm 중복 제거 합계** | **30.743949ms** | **35.307216ms** |
| EE, 기본 | 기록 구성·structural 표시, 별도 진단 | 5.923316ms | 7.191199ms |
| EE, 기본 | 3mm 중복 제거, 별도 진단 | 24.427467ms | 27.578849ms |
| World | **기록 구성 → 3mm 중복 제거 합계** | **29.040553ms** | **35.004899ms** |
| World | 기록 구성·structural 표시, 별도 진단 | 5.764917ms | 6.311641ms |
| World | 3mm 중복 제거, 별도 진단 | 22.983452ms | 26.147327ms |

별도 표본들의 중앙값을 더한 값은 직접 측정한 합계 중앙값과 같지 않다.
합계에는 중간 입력 기록의 수명 종료도 포함되며, 구간 진단에서는 그 해제가
계측 밖이다. 이 표를 정확히 합산되는 시간 예산으로 해석하지 않는다.

Raw 표본을 제거하거나 보정하지 않았다. World 합계 최대160.669ms,
EE 기록 구성 진단 최대167.523ms인 드문 지연도 원본에 남겼다. GC 또는 공유 host
스케줄링 중 무엇이 원인인지 분리 계측하지 않았으므로 원인을 단정하지 않는다.
동일 입력의 반복 결과이며, 다양한 장면·점 수에 대한 분포를 측정한 것은 아니다.

## 출력·실행 검증

두 좌표계 모두 기존 공개 `PlanA.represent(..., representations=('SF',))`를
실제로 호출한 결과와, 측정 helper가 반환한 **모든 기록 field, 배열 dtype/value,
representative 순서, structural 표시, 병합 source ID와 audit가 정확히 일치**했다.
공개 호출에는 미리 준비한 실제 CUDA packet을 공급했으며, 기존 view/CPU 반환과
SF 기록·dedup 로직은 그대로 실행했다. 로봇 필터/gate와 FK/Graph 검증은 이 작업의
범위가 아니다.

- 입력1,105점 → 대표점 **212개**, 중복893개(80.8145%).
- 대표점 중 structural118개. 최종128점 제한 전의 개수다.
- 총1,105개의 source ID를 대표점들에 보존했다.
- 입력 observations/structural flags와 dedup 입력 기록의 불변성을 확인했다.
- EE·World가 같은 representative/source 병합 결과를 내는지도 원시 결과로 확인했다.
- 로컬에서 benchmark import/CLI와 구문을 확인하고, Vast에서 실제 CUDA 입력 준비,
  공개 경로 비교와 CPU 계측을 완료했다. macOS CUDA 실행은 하지 않았다.
- CPU의 기존 구현을 측정한 작업이며, 후속 GPU 이식이나 속도 향상 검증은 하지 않았다.

## 재현·보존

Runner와 설명:

- [benchmark.py](../../graph/experiments/records_dedup_timing_58752018_20261103_v1/benchmark.py)
- [실행 방법](../../graph/experiments/records_dedup_timing_58752018_20261103_v1/README.md)

Remote root: `/workspace/task10i_records_dedup_timing_20261103_v1`.
앞서 실행했던 view CUDA 작업의 source/input을 새 디렉터리에 복사했다.
현재 로컬의 관련 runtime15개와 복사본의 SHA256 일치를 실행 전에 확인했고,
전체 source/input/runner manifest는 실행 전후에 보존했다. 기존 실행은 덮어쓰지 않았다.
다른 작업의 robot filter/SAM2/CUDA ingress 구현은 수정하지 않았다.

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했다. W&B disabled,
local 기록만 남겼으며 게시하지 않았다. Run ID:
**`7bfb2aae35ff4b14852b514f4c8944d6`**.

로컬 evidence:

- [summary.json](../../outputs/task10i/records_dedup_timing_58752018_20261103_v1/remote/measured/summary.json)
- [timings.npz](../../outputs/task10i/records_dedup_timing_58752018_20261103_v1/remote/measured/timings.npz)
- [검증·통계 재계산](../../outputs/task10i/records_dedup_timing_58752018_20261103_v1/verification.json)
- [전체 frozen source·입력·로그·원시 결과 archive](../../outputs/task10i/records_dedup_timing_58752018_20261103_v1/evidence.tar.gz)

Archive SHA256:
`90c0b98d7f6410b360e6025f6ad12307297cc68e69fefad72829a138efdd68b8`.
Archive manifest756개 파일의 hash와 원시100개 표본의 통계를 로컬에서 재검증했다.
