# Task 10-I — 기존 카메라 간 관측 일관성 검사 시간

2026-11-03. 좌표계 통일이 적용된 현재 기존 NumPy
`graph.persistent_geometry.fusion.consistent_views`의 소요시간을 측정했다.
검사 알고리즘/임계값/출력 device는 바꾸지 않았다.

## 입력 조건과 한계

VastAI **58752018**, AMD EPYC7543, Python3.12.15, NumPy2.5.4,
Torch2.14.1+cu130의 기존 환경을 사용했다. 측정 구간은 **Vast CPU NumPy**다.
CUDA는 관측 준비에만 사용했고 그 완료를 동기화한 뒤 측정을 시작했다.

이전 단계와 같은 실제 `10i_s1_0844_t080` hand-camera512×512,
Peg mask61,056 pixels, Box0 pixels다. Sampling 후 입력 관측은 다음과 같다.

| 관측 camera | Peg 점 | Box 점 |
| --- | ---: | ---: |
| base_camera | 0 | 0 |
| hand_camera | **1,105** | 0 |

**실제 sensor는 hand 하나다.** 이전 검증처럼 같은 depth/calibration을 복제한
base camera에 빈 mask/관측을 붙였다. 원래 해상도 실험의 두 실제 camera 원본은
이번58752018 작업 root에 없었고, 로컬 해상도 자료에는 RGB PNG만 있었다.
따라서 이 수치를 서로 다른 두 실제 camera가 모두 유효 관측을 내는 경우의
latency라고 해석하면 안 된다. 이번 baseline은 앞서 사용한1,105점 경계를
유지한 진단 조건이다. 마스크 추론/필터/gate를 새로 실행하거나 GT로 채우지 않았다.

## 측정 경계

`make_packet`과 `observations_from_packet`으로 World/EE 관측을 **미리 준비**했다.
기존 `PlanA.represent`의 호출 구조대로 다음을 차례로 실행했다:

```python
consistent_views(base_observations, cameras, masks)
consistent_views(hand_observations, cameras, masks)
```

**위 두 호출의 합계가 한 표본의 총시간**이다. Base 쪽 점이 비어 있어도 기존
카메라 동기화/검증과 빈 관측 반환은 실행한다. Hand Peg에서는 상대 camera로의
투영과 깊이/mask 비교를 실행한다. 양쪽에 비어 있는 Box 관측도 그대로 전달한다.

포함하는 일:

- 기존 `synchronized(cameras)` 및 camera calibration/TCP 정합 검사.
- 선택한 World/EE frame에서 상대 camera로 투영.
- 기존 `CameraObservation.depth` property의 **전체512×512 깊이 변환**.
  원래 코드의 shape 조회와 해당 픽셀 깊이 조회를 그대로 실행하며 memoize하지 않는다.
- Depth 차이 `<4mm`와 반대 object-only mask의 동시 충돌 검사.
- Keep mask, audit 구성, `SurfaceObservation.subset`의 결과 배열 복사/반환.

File IO, 역투영/PCA/Shape-First, GPU→CPU 전송, packet/관측 생성, Python point
records,3mm 중복 제거,128점 제한, Graph, robot filter/gate는 제외했다.
World/EE 각각20회 warmup 뒤100회 측정했고, 매 반복의 frame 실행 순서를 번갈았다.
Python GC는 기본 상태, BLAS/OMP/MKL 스레드는1이다. CPU affinity와 공유 host
부하는 고정하지 않았다. 계측용 내부 clock이나 별도 대체 구현은 넣지 않았다.

## 결과

| 기존 검사 단계 전체: base1회 + hand1회 | 중앙값 | p95 | 최소–최대 |
| --- | ---: | ---: | ---: |
| **World** | **5.345ms** | **5.419ms** | 5.300–5.908ms |
| **EE** | **5.382ms** | **5.564ms** | 5.340–9.440ms |

현재 기본 설정은 EE이므로 이 표본에서의 기준 중앙값은 **약5.38ms**다.
이는 단순3D 행렬 곱셈이나 순수2D 투영 시간만이 아니다. 전체 깊이 image 변환,
검증과 반환 복사도 포함된다. 내부 구간별 시간은 이번에 분리하지 않았으므로
표만 보고 어느 연산이 가장 큰지 단정하지 않는다. World/EE 차이를 일반적인
성능 우열로 주장하지 않는다. 앞선 packet/관측 단계의 다른 실행 중앙값과 더해
전체 Graph latency를 만들지도 않는다.

## 출력·실행 검증

측정 밖에서 반환 관측의 **모든 dataclass 필드**를 기존 입력과 exact 비교했다.
좌표, 법선, F5/NaN 위치, validity, source pixel 순서, origin 및 frame/EE 기준
metadata가 일치했다. 이번 빈 상대 mask 조건에서는 모든 관측이 유지되어야 하며,
audit의 identity conflict는0이었다. 원본 관측 deep copy와도 비교해 입력이
변형되지 않았음을 확인했다.

기존 coordinate-frame 작업의 frozen source를 새 root로 복사해 실행했다.
측정 대상 `fusion.py`, 계약/좌표 helper들과 runner가 현재 checkout과 일치하는지
hash 확인했다. 원본100회 표본으로 중앙값/p95를 로컬에서 재계산했다.
Runtime 변경이 없으므로 별도 기능 테스트를 추가하지 않았다. 로컬 macOS에서는
source syntax, archive/hash, 문서 링크와 원본 통계를 검증했으며, timing은 Vast에서만
실행했다. 선택적 resident CUDA consistency 함수는 이번에 측정하지 않았다.
Simulator/rollout·학습/평가·성공률은 실행하지 않았다.

## Identity·산출물

공통 `ExperimentLogger` 재사용, W&B disabled, publishing 없음.
UUID: **`901869b8356249bf880d7f1ce4831a57`**.
원격 root: `/workspace/task10i_view_consistency_timing_20261103_v1`.
기존 실험을 덮어쓰지 않았고 package/driver/service 변경이나 다른 작업 중지는
하지 않았다. 시작 시 다른 CUDA compute process가 없음을 확인했다.

- [측정 summary](../../outputs/task10i/view_consistency_timing_58752018_20261103_v1/remote/measured/summary.json)
- [World/EE 원본100회씩](../../outputs/task10i/view_consistency_timing_58752018_20261103_v1/remote/measured/timings.npz)
- [실행 로그](../../outputs/task10i/view_consistency_timing_58752018_20261103_v1/remote/measured.log)
- [Source/input manifest](../../outputs/task10i/view_consistency_timing_58752018_20261103_v1/remote/source_manifest.json)
- [현재 checkout identity](../../outputs/task10i/view_consistency_timing_58752018_20261103_v1/checkout_identity.json)
- [다운로드·통계 검증 receipt](../../outputs/task10i/view_consistency_timing_58752018_20261103_v1/download_receipt.json)
- [Runner·실행 방법](../../graph/experiments/view_consistency_timing_58752018_20261103_v1/README.md)

Archive SHA256:
`6945ae75a898966c343c2b32bfde386ca22ca4c530013d9de9a568d6da4ee6de`.
