# Task 10-I — 관측 일관성 검사를 Torch CUDA로 전환

2026-11-03. 기본 Graph의 sampling 후 일관성 검사를 PyTorch tensor 연산으로
전환했다. World/EE와 기존 depth-supported opposite-object 규칙을 유지한다.

## 구현·실제 연결

공통 구현은 `graph.persistent_geometry.tensor_fusion`에 있다.

- `prepare_tensor_views`: 같은 시점 camera identity/calibration을 확인하고,
  depth/mask/행렬을 프레임당 한 번 준비한다. 선택점에 대한 연산은 GPU에서 수행한다.
  작은 기존 pose/calibration metadata의 host 검증은 유지한다.
- Signed int16 mm 깊이는 Torch lookup으로 float64 metre로 변환한다. IEEE division으로
  만든 정적 값 표만 device별로 재사용하며, 실제 센서/마스크/포즈는 매번 갱신한다.
  음수/0/overflow는 기존처럼 invalid다. NumPy 전체 깊이 image를 반복 생성하지 않는다.
- `consistency_keep`: float64 Torch projection, floor pixel, positive-z/image-bound
  조건, depth 차이 **`<0.004m`**, 반대 물체-only mask 조건을 그대로 적용한다.
  Invalid/outside 점의 안전한 gather는 clamp하되, 반드시 원래 inside 조건으로
  배제한다. Mask overlap이나 missing mask만으로 충돌을 만들지 않는다.
- `consistent_views_tensor`: keep과 반환 관측의 배열을 같은 device에 보존한다.
  Compact source row를 한 번 계산해 좌표/법선/F5/validity/source/structural flags에
  공유한다. Audit의 conflict 수 역시 GPU int64 tensor다.

**기본 `CurrentGeometry(perception_backend='triton_fused')`에 연결했다.** 이제
`make_packet_cuda` → 같은 frame의 tensor 관측 → Torch view check 순서로 처리한다.
View 검사 전에 selected packet을 CPU로 내려 다시 올리지 않는다. 기존 NumPy
view 함수를 호출하면 실패하도록 막은 default-path 테스트도 통과했다.

현재 다음 단계인 Python records/3mm dedup/최종 Graph 소비자는 여전히 CPU다.
그 경계에서만 `numpy_observations`로 compact selected 결과를 명시적으로 packed
readback한다. 이 작업은 그 후속 단계 전체를 GPU로 이식한 것이 아니다.
선택적 `cuda_resident` prototype도 같은 keep 함수를 사용하고 이미 staged된 깊이를
재사용한다. 기존 로봇 필터와 sampling 전 wrist gate는 변경하지 않았다.
Explicit CPU perception 및 historical standalone PlanA는 NumPy reference 경로를 유지한다.

Tensor 관측은 XYZ/법선/F5/origin float64, source int64, validity/structural bool이다.
현재 packet 기반 NumPy 관측과 같은 규격이며 NaN과 source 순서를 보존한다.
World/EE mismatch와 잘못된 시점/EE reference/layout은 거부한다. Dynamic compaction과
기존 packet validation에는 GPU 동기화가 생길 수 있으며 zero-sync 경로라고 주장하지 않는다.

## 측정 조건과 endpoint

VastAI **58752018**, RTX3090, AMD EPYC7543, Python3.12.15,
Torch2.14.1+cu130, NumPy2.5.4의 기존 환경에서 실행했다. OMP/BLAS/MKL/Torch
스레드는1, CPU affinity와 공유 host 부하는 고정하지 않았다.

이전과 같은 실제 `10i_s1_0844_t080` hand512×512 입력을 사용했다.
**Hand Peg1,105점, Box0점**, base 관측0점이다. 실제 sensor는 hand 하나이고,
base는 같은 depth/calibration을 복제한 빈 mask/관측이다. 서로 다른 두 실제 camera가
모두 관측을 내는 경우의 latency가 아니다. 실제 two-view 원본이 없다는 이전 측정의
한계를 유지하며, 그 조건을 바꾸지 않고 비교했다.

기존처럼 **base 검사1회 + hand 검사1회의 합계**를 한 표본으로 삼았다.
각 조합20회 warmup 후100회, frame/mode 실행 순서를 회전·역순으로 교대했다.
CUDA는 시작 전 대기를 timer 밖에서 끝내고, 반환 후 GPU 완료 동기화를 timer 안에
포함했다. Logger/비교/file IO는 timer 밖이다.

| 경로 | 시작과 종료 |
| --- | --- |
| 기존 NumPy | 준비된 CPU 관측 + CPU sensor/mask → CPU 관측 반환 |
| Torch GPU, resident | 준비된 GPU 관측/depth/mask → GPU 관측 반환 |
| Torch GPU, 센서 준비 포함 | 준비된 GPU 관측 + CPU sensor/mask → 검증/준비/H2D → GPU 관측 반환 |
| 위 과정 + CPU compatibility | 위 GPU 검사 후 후속 CPU records용 명시적 readback/검증까지 |

**CUDA 조합의 관측점은 시작부터 GPU에 있다.** 새 기본 packet 경로의 실제 상태다.
NumPy 관측을 GPU로 복사하는 비용을 포함했다고 주장하지 않는다. Sampling/PCA,
packet/관측 생성은 양쪽 모두 제외한다. Resident 행에서는 sensor 준비도 제외하지만,
다음 두 행에는 매번 새 sensor 검증, raw mm/mask/작은 calibration H2D와 Torch 깊이
준비를 포함한다. 기존 NumPy 행은 원래 전체-depth property 변환/검증/반환 복사까지
그대로 포함한다. CPU compatibility는 실제 기본 경로처럼 source camera별로 수행했다.

## 결과

| 경로 | EE 중앙값 | EE p95 | World 중앙값 | World p95 |
| --- | ---: | ---: | ---: | ---: |
| **기존 CPU NumPy** | **5.461ms** | 5.882ms | **5.420ms** | 5.918ms |
| **Torch GPU, resident 입력** | **1.001ms** | 1.134ms | **0.995ms** | 1.185ms |
| **Torch GPU, 센서 준비·전송 포함** | **2.738ms** | 3.034ms | **2.568ms** | 2.907ms |
| **위 과정 + CPU compatibility 반환** | **3.760ms** | 4.249ms | **3.406ms** | 3.697ms |

기본 EE 설정에서 GPU 반환까지 센서 준비를 포함하면 중앙값이 **49.9% 감소**했다.
Resident 검사 자체는81.7% 감소했다. 후속 CPU records를 위해 내려받는 과정까지
포함하면31.1% 감소했다. 서로 다른 endpoint의 중앙값을 빼거나 더해 정확한 내부
구간 시간을 만들지는 않는다.

앞서 별도로 측정한 NumPy EE 중앙값5.382ms와 이번 paired NumPy5.461ms는 실행
차이가 있다. 위 감소율은 **이번 같은 실행의 paired reference**를 기준으로 한다.
World/EE의 미세한 시간 차이를 일반적인 우열로 해석하지 않는다. 이 숫자는 전체
Graph/model latency나 정책 성공률 비교가 아니다.

## 정확성·실행 검증

최종 suite:

```text
visual.runtime.test.test_tensor_view_consistency
visual.runtime.test.test_coordinate_frames
visual.runtime.test.test_graph_perception
visual.runtime.test.test_object_groups
visual.runtime.test.test_vision_features
visual.runtime.test.test_variable_rate_graph_contract
graph.plan_a_geometry.tests
graph.persistent_geometry.tests.test_contracts
graph.plan_a_bc.tests
```

- 로컬 macOS: **105개 중93개 통과, CUDA 전용12개 skip**.
- Vast: **105개 중104개 통과,1개 skip**. Skip은 source-only 배포에 실제
  preregistration 파일이 없는 기존11,487-step schedule 검사다.
- Signed int16의65,536개 깊이 값 전체를 NumPy 기준과 exact 비교했다.
- Strict4mm, 그 양쪽 인접 float, overlap/missing mask, invalid depth,
  outside/behind-camera, empty observations, 비항등 calibration/World/EE,
  잘못된 identity/reference/device/layout을 검사했다.
- GPU 검사에서 `.cpu()` readback을 금지한 테스트가 통과했다. 입력 배열을
  변경하지 않고 source/structural flags를 유지했다. CPU full-depth property를
  사용하지 못하게 막은 Torch 준비/검사 테스트도 통과했다.

실제512 paired 실행에서는 모든 반환 관측 필드/dtype/NaN/source 순서와 audit가
NumPy reference와 exact 일치했다. Untimed 실제 FK/public Graph 비교에서는
legacy NumPy-view 경로와 새 default를 같은 prepared mask/gate로 실행했다:

| 실제 Graph 비교 | World | EE |
| --- | ---: | ---: |
| 노드 / 엣지 | 131 / 1,006 | 131 / 1,006 |
| Node/edge type, index, source provenance | exact | exact |
| Node input 최대 차이 | 0 | 0 |
| Edge attribute 최대 차이 | 0 | 0 |

일반 float 비교 허용 오차는 기존2e-5를 사용했고, 이번 실제 결과는 차이0이었다.
임계값을 완화하거나 epsilon을 추가하지 않았다. 모든 calibration과 경계 입력에서
서로 다른 BLAS/CUDA 부동소수점 경로의 bitwise 회전 불변성을 보장한다는 주장은
하지 않는다. 서로 다른 두 실제 camera rollout, simulator 생성, 학습/평가와 성공률은
이번에 실행하지 않았다. 로봇 필터/이전 gate를 포함한 새로운 full pipeline 성능도
측정하지 않았다.

## 재현성과 산출물

공통 `ExperimentLogger` 재사용, W&B disabled, publishing 없음.
성공 UUID: **`f557a49fa7bd4f7b87f102dde4a52239`**.
원격 root: `/workspace/task10i_view_consistency_cuda_20261103_v1`.
이전 frozen source를 새 root로 복사하고187개 현재 source를 overlay했다.
실행 전/다운로드 후 hash를 검증했고 현재 파일과도 일치한다. 로봇 필터,
SAM2 bridge 및 기존 wrist gate 파일은 작업 전 hash와 동일함을 확인했다.
다른 dirty 작업은 보존했고 package/driver/service 변경이나 다른 작업 중지는 없었다.

첫 benchmark runner는 local 변수명이 `os` import를 가려 측정/Logger 시작 전에
실패했다. Runtime은 바꾸지 않고 runner 변수명을 수정해 `benchmark_v2.py`로
실행했다. 실패 runner/log와 성공 source/raw samples를 모두 보존했다.

- [측정·실제 Graph 검증 summary](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/measured_v2/summary.json)
- [8조합 원본100회씩](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/measured_v2/timings.npz)
- [성공 실행 로그](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/measured_v2.log)
- [CUDA 테스트 로그](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/tests.log)
- [로컬 최종 테스트 로그](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/local_final_tests.log)
- [환경 정보](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/hardware.json)
- [전체 source/input manifest](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/source_manifest.json)
- [Overlay manifest](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/remote/overlay_manifest.json)
- [작업 전 source identity](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/before_identity.json)
- [다운로드·통계/hash 검증 receipt](../../outputs/task10i/view_consistency_cuda_58752018_20261103_v1/download_receipt.json)
- [Runner·실행 방법](../../graph/experiments/view_consistency_cuda_58752018_20261103_v1/README.md)
- [Torch 관측 API](../../graph/persistent_geometry/README.md)

Archive SHA256:
`b4e3e961ab498edb060898c9559b22e48ab12d673fff6b55b2b06aaed2ea60d4`.
741개 source/input manifest 항목과 모든 원본 표본의 중앙값/p95를 재검증했다.

## 커밋 범위 점검

이번 커밋에는 공통 Torch 관측 검사, 기본 `triton_fused` 연결, 관련 테스트와
측정 runner·문서만 포함한다. 별도 작업 중인 `cuda_resident` 입력 prototype과
로봇 필터/SAM2 변경은 제외하고 작업 디렉터리에 보존했다. 위 원격 검증의 frozen
source manifest에는 함께 존재했던 별도 prototype도 기록되어 있다.

다른 미커밋 변경에 의존하지 않는지도 확인했다. Staged tree만 임시 디렉터리에
복사하여 같은105개 테스트를 실행한 결과 **92개 통과,13개 skip**이었다.
CUDA 전용12개와 source-only 복사에 없는 preregistration 검사1개를 건너뛰었다.
이 macOS 점검에서 CUDA를 다시 실행한 것은 아니며, GPU 결과는 위 Vast 검증이다.
