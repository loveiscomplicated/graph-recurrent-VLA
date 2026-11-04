# Task 10-I — 기존 EE packet의 물체 분리·World 관측 반환 시간

2026-11-03. 최적화 전 기준 측정이다. Runtime 코드를 변경하지 않고
`graph.persistent_geometry.fusion.observations_from_packet(packet, camera)`를
직접 실행했다.

## 조건과 측정 경계

VastAI **58752018**, AMD EPYC7543, RTX3090, Python3.12.15,
NumPy2.5.4, Torch2.14.1+cu130. 이전 packet 측정과 같은 실제
`10i_s1_0844_t080` hand-camera **512×512** RGB-D/추론 mask를 사용했다.
Peg mask61,056 pixels, Box mask0 pixels, packet 선택점 **1,105개**다.
기록된 camera calibration과 TCP pose를 그대로 사용했다.

측정 전에 기존 `make_packet`을 실행하고 CUDA를 동기화해 **CPU EE packet을
준비했다**. 이후 CPU `perf_counter`로 기존 관측 변환 함수의 진입부터 반환까지
측정했다. Warmup20회 뒤100회, NumPy/BLAS/MKL 스레드1. CPU affinity와
공유 host 부하는 고정하지 않았다.

포함 범위:

1. Packet/sensor identity 확인, EE→World 행렬 구성.
2. `object_id == 0/1`로 Peg·Box 행을 찾고 특징·유효성을 분리/복사.
3. 법선과 점 좌표를 EE→World로 변환.
4. Source pixel·camera origin 등을 담은 `SurfaceObservation` 구성 및 검증.
5. Peg·Box 관측 객체 두 개 반환. 이 입력의 Box 객체는 빈 배열이다.

역투영/PCA/Shape-First, GPU→CPU 전송, EE packet 생성/수신 검증은 **시작 전에
완료되어 제외**했다. 이후의 `consistent_views`, Python point records 구성,
3mm 중복 제거, 최종128점 제한, World→EE 최종 Graph 변환, 엣지 생성도 제외한다.
로봇 필터와 손목 gate는 실행하지 않았다.

## 결과

| 기존 public 함수 전체 | 중앙값 | p95 | 최소–최대 |
| --- | ---: | ---: | ---: |
| **물체 분리 + EE→World + 관측 구성·검증·반환** | **0.541ms** | **0.577ms** | 0.523–0.681ms |

이 전체 시간은 내부 계측을 넣지 않은 기존 함수를 직접 호출한 결과다.
앞서 측정한 CPU packet 반환 시간에 이 구간이 포함되어 있지는 않다.
서로 다른 실행의 중앙값을 더해 전체 pipeline 시간을 주장하지 않는다.

별도의100회 diagnostic pass에서 원래 함수와 같은 연산식/순서를 재현하며
구간별 clock을 넣었다. 각 값은 Peg와 빈 Box의 처리 합계다.

| 진단 구간 | 중앙값 | p95 |
| --- | ---: | ---: |
| Identity 확인·EE→World 행렬 준비 | 0.022ms | 0.027ms |
| 물체 ID 비교·특징/유효성 배열 분리·복사 | **0.048ms** | 0.068ms |
| 법선 회전·유효성 처리·특징 배열 배치 | **0.080ms** | 0.105ms |
| 점 행 선택·EE→World 변환·변환 행렬 검증 | **0.104ms** | 0.130ms |
| Source/camera origin 준비·관측 객체 구성·검증 | **0.291ms** | 0.354ms |

분리 구간에는 특징/유효성 gather가 들어가고, 점과 source pixel gather는 각각
뒤의 해당 구간에 들어간다. 따라서0.048ms가 모든 행 선택·복사의 총비용은 아니다.
점/법선 구간도 allocation, validity 처리와 transform 검증을 포함하므로 순수
행렬 곱셈 시간이라고 부르지 않는다. 독립 표본의 중앙값을 더해 정확한 전체
중앙값을 만들지 않는다.

이 입력에서는 해당 단계 전체가 약0.54ms다. World 왕복을 정리하거나 물체별
묶음을 유지했을 때의 절감량은 아직 측정하지 않았다. 특히 이 표만으로0.54ms
전체를 제거할 수 있다고 해석하면 안 된다. Box가 실제로 보이는 입력이나
두 실측 카메라의 점 수가 늘어난 상황의 latency로 일반화하지 않는다.

## 실행·출력 검증

- 계측 재현 함수와 기존 public 함수가 반환한 두 dataclass의 **모든 필드**를
  재귀적으로 exact 비교했다. 좌표, 법선, F5, validity, pixel source, origin,
  identity가 일치했다. NaN 위치도 비교에 포함했다.
- 입력 packet을 deep copy와 비교해 변형되지 않았음을 확인했다.
- 별도로 camera 원본 좌표를 직접 World로 변환한 결과와 관측 XYZ를
  `atol=2e-6m, rtol=0`에서 비교했다. 기존 Camera→EE float32 반올림은 유지했다.
- 측정 대상 `fusion.py`, `contracts.py`는 현재 checkout과 hash가 일치한다.
  성공 runner hash,709개 source/input 파일 manifest와 archive hash를 확인했고,
  NPZ 원본100회 표본에서 모든 중앙값/p95를 로컬에서 재계산했다.

CUDA는 Vast에서 기존 packet 준비에만 사용했고, **측정 구간은 Vast CPU NumPy**다.
로컬 macOS에서는 archive/source hash, syntax, 통계와 문서 링크만 확인했다.
Runtime 변경이 없어 별도 기능 테스트를 추가하지 않았다. Simulator 생성,
rollout, 학습/평가, 성공률 검증은 실행하지 않았다.

## 재현성과 산출물

공통 `ExperimentLogger`를 재사용했다. W&B disabled, publishing 없음.
성공 UUID: **`e59f95b951424ab39ecc198214e752e1`**.
원격 root: `/workspace/task10i_observation_cpu_timing_20261103_v1`.
이전 packet CPU 측정의 frozen source를 새 root로 복사했다. 별도 작업의
CUDA ingress/로봇 필터 변경이 섞이지 않도록 유지했고 해당 작업 파일은 수정하지
않았다. Frozen source는 Git checkout 자체가 아니므로 source manifest와 로컬
checkout identity를 함께 보존했다. Commit만으로 재현된다고 주장하지 않는다.

첫 runner는 namespace package `graph.__file__`가 None인 상태에서 identity 경로를
계산하다 **측정 시작 전에 실패**했다. Runtime은 바꾸지 않고 runner의 경로
조회만 `fusion.__file__` 기준으로 수정해 `benchmark_v2.py`로 실행했다.
실패 runner/log와 성공 runner/log를 모두 보존했다. 패키지/driver/service 변경이나
다른 작업 중지는 하지 않았으며, 시작 시 다른 CUDA compute process가 없음을
확인했다.

- [측정 summary](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/remote/measured_v2/summary.json)
- [전체 원본100회](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/remote/measured_v2/timings.npz)
- [구간별 원본100회](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/remote/measured_v2/phase_timings.npz)
- [반환 World 관측 배열](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/remote/measured_v2/observations.npz)
- [성공 실행 로그](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/remote/measured_v2.log)
- [Source/input manifest](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/remote/source_manifest.json)
- [다운로드·통계 검증 receipt](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/download_receipt.json)
- [로컬 checkout identity](../../outputs/task10i/observation_cpu_timing_58752018_20261103_v1/checkout_identity.json)
- [Runner·실행 방법](../../graph/experiments/observation_cpu_timing_58752018_20261103_v1/README.md)

Archive SHA256:
`d446108cb88002620af66567a48d36aafffa9cc4f2ee1a7339e665698faacd54`.
