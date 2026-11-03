# Task 10-I SAM2 → 로봇 필터 GPU 입출력 연결

## 결과와 변경 범위

SAM2 출력 마스크를 GPU 텐서로 유지한 채 로봇 필터에 전달하고, clean mask도 같은 GPU의 bool 텐서로 반환하는 직결 경로를 구현했다.
로봇 형상/FK의 CPU 준비와 센서·보정 데이터의 GPU 업로드는 기존처럼 남아 있다. **마스크의 GPU→CPU→GPU 왕복을 제거한 변경**이며,
전체 Graph를 GPU로 이식한 작업이나 CPU 작업이 전혀 없다는 의미는 아니다.

공통 simulator, 기존 필터의 CPU API, 이전 실험 코드, Graph gate·특징·Shape-First·노드 선택·연결 규칙은 수정하지 않았다.
추적 공통 코드 변경은 `SAM2Tracker.track(..., mask_filter=...)`의 선택적 연결점뿐이다. 인자를 생략하면 기존 동작과 동일하다.
작업 중 다른 작업에서 생긴 Shape-First/CUDA sampling 변경은 보존했고 이 작업의 변경 범위에 포함하지 않는다.

## 실제 연결 경로

```text
SAM2 CUDA mask
  → CudaRobotFilter(CUDA mask 입력)
  → CUDA clean mask 반환
  → GPU 소비자
```

이미지 SAM2의 공개 `predict()`는 마지막에 `.cpu().numpy()`를 호출하므로, 새 `image_masks_cuda()`는 설치된 predictor의
동일 `_prep_prompts` / `_predict` 호출 뒤 그 CPU materialization만 생략한다. `multimask_output=False`, `return_logits=True`,
float cast 후 `>0` 판정과 Peg/Box 프롬프트·순서는 기존 Task10-I 이미지 경로와 같다.
`image_clean_masks_cuda()`가 이를 필터에 직접 연결한다. 모델 생성·가중치·autocast·추론 설정은 바꾸지 않는다.
기존 caller의 autocast 설정을 그대로 사용해야 한다. score도 CUDA scalar로 유지하고 없는 프롬프트는 None이다.

추적 SAM2는 원래 GPU tensor mask를 반환한다. 새 callback은 RGB-D CPU 어댑터보다 먼저 실행되며 frame index와 ID 1/2,
mask shape·dtype·device를 보존한다. `SAM2RobotFilter`의 `observation_for_frame(index)`가 해당 시점의 camera와 FK surface를 명시적으로 공급한다.
기존 stride/source-frame 대응은 caller의 것을 그대로 사용하며 새로 추정하지 않는다.

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.sam2_bridge import image_clean_masks_cuda
clean_cuda, scores_cuda = image_clean_masks_cuda(
    predictor, rgb, [peg_box, box_box], camera, current_fk_surfaces
)
```

이것은 live tensor 연결 경로다. 저장 NPZ를 만드는 과거 실험 worker는 변경하지 않았고 원래 정지 실험을 재개하지 않았다.

## 필터 판정 보존

[`gpu_filter.py`](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/gpu_filter.py)는 CUDA bool `[2,H,W]`를 받아
입력과 같은 device/shape/dtype의 새 clean mask를 반환하며 원본을 변경하지 않는다.
고해상도에서는 원본 mask union과 기존 링크 투영 bbox가 겹친 lane만 GPU에서 판정한다.
GPU mask를 CPU에 내려 ROI index를 만들거나 `nonzero`의 동적 크기를 CPU에서 읽지 않는다.

기존 full ray GEMM, 링크별 numerator 연산 순서, FP64, 링크 순서와 엄격한 nearest-depth 동률 처리를 유지한다.
2.5mm tolerance, 6mm band, 0.5mm 내부 margin, overlap 제거도 동일하다.
128² 이하에서는 기존 Stage 9 grouped kernel과 그 연산 방식을 유지한다.
고해상도 threshold 비교는 기존 Torch float64 방식대로 명시적으로 float64 상수를 사용한다.
동적 bbox 정수는 값/정렬별 JIT 특화를 끄고 runtime 인자로 전달해 pose마다 재컴파일되지 않도록 했다.
정책·threshold를 완화하거나 fast-math로 바꾼 것이 아니다.

`depth_cuda`를 제공하는 경우 같은 GPU의 float64 CV-z metres `[H,W]` 계약이다. 생략하면 기존 CPU sensor depth를 업로드한다.
world geometry는 매 호출 현재 FK로 준비하며 이전 프레임의 pose/hull을 재사용하지 않는다.
선택적 GPU diagnostics에는 `evaluated_pixels`를 함께 반환해 high-resolution union 밖의 미계산 영역을 전체 화면 판정으로 오해하지 않도록 했다.

## 나머지 Graph를 유지하는 경계

기존 gate가 원본 mask의 raw_pixels·clean_fraction을 사용하므로, 이미 정리된 mask를 원본처럼 넘기면 안 된다.
[`GpuMaskBoundary`](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/graph_boundary.py)는 raw CUDA mask로
**먼저 GPU 필터를 완료해 clean CUDA mask를 얻은 다음**, 기존 CPU gate/Shape-First/Graph 소비자 앞에서만 raw와 clean을 명시적으로 복사한다.
원래 `PlanA.prepare`에 raw를 보존해 전달하고 사전 계산한 clean을 사용하므로 필터를 두 번 수행하지 않는다.
`boundary.clean_cuda`에는 GPU 출력이 그대로 남는다. private scope 종료나 예외 시 함수 patch를 복구한다.
같은 process에서 concurrent builder는 지원하지 않는다.

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.graph_boundary import GpuMaskBoundary
with GpuMaskBoundary() as boundary:
    graph, records, gate = geometry.build(cameras, data, surfaces, raw_cuda_by_camera)
    clean_cuda_by_camera = boundary.clean_cuda
```

직접 GPU 소비자에는 `CudaRobotFilter` / 이미지 직결 함수 / tracker callback의 GPU 반환값을 사용하면 된다.
위 raw-mask Graph 경로에는 이미 정리된 tracker callback 출력을 원본으로 다시 넣지 않는다.
기존 `attach_rgbd`도 CPU 소비자를 위한 경계로 유지했다.

## 검증 결과

- Mac: 기존 RGB-D 7개 + callback 계약 2개 통과. GPU 전용 5개는 CUDA 미지원으로 skip.
- 기존 원격 RTX 3090: GPU 필터·입력·SAM2 mock 연결 테스트 4개, raw gate 보존·scope 복구 테스트 1개 통과.
- 저장 actual SAM2 mask **112조건 / 224카메라**에서 기존 Stage 9와 clean mask·confirmed·uncertain·candidate·interior·owner가 정확히 일치했다.
  기존 26시점 × 128/512/768/1024의 104조건과 추가 Parent 2개의 8조건이다. 선택 표본이며 독립 장면 112개를 의미하지 않는다.
- ROI predicted depth는 기존 `atol=1e-12, rtol=0` 계약으로 확인했고 실제 최대 차이는 **0.0**였다.
- 모든 실제 입력에서 CUDA 입력/반환과 입력 불변성을 확인했다. Torch dispatch guard로 `.cpu()`에 해당하는 D2H 복사,
  `_local_scalar_dense`와 동적 `nonzero`를 금지한 상태에서도 필터가 동작했다. 검증용 CPU readback은 필터 밖에서만 했다.
  이는 모델/드라이버 전체에 동기화가 전혀 없다는 주장이 아니며 CPU geometry 업로드 비용도 제거하지 않았다.
- 기존 큰 wrist-mask 실패 사례 `0924/t080`의 4해상도에서 CPU 호환 경계를 연결한 Graph의 5 tensor,
  source ID, 모든 record field, gate, Graph.packet metadata가 기존 Stage 9와 일치했다. PCA field의 관측 최대 차이도 **0.0**였다.
- SAM2 실제 추론은 재실행하지 않았다. 설치된 SAM2 API 소스를 직접 확인했고, CUDA fake predictor로 이미지·추적 연결을 검증했다.
  real mask 회귀는 저장 출력을 GPU에 한 번 올린 뒤 수행했다. 실제 SAM2→filter 전체 latency를 측정한 것은 아니다.

초기 개발 검증에서 threshold literal 정밀도와 원격 overlay의 구 tracker 모듈 import 문제를 확인해 수정했다.
이후 bbox 정수의 불필요한 JIT 특화를 제거하는 동안 개발 검증을 중단·재실행했고, 최종 v4 결과만 완료 검증으로 사용했다.
개발 실패/중단 로그는 원자료에 보존했다. 판정 동등성을 위해 tolerance를 넓혀 실패를 숨기지 않았다.

## 실행·산출물

추가 모델 API, grounding/SAM2 추론, 학습, rollout, package 설치, 게시나 원래 프로세스 재개는 하지 않았다.
마지막 확인에서 원래 7개 process는 모두 T(stopped)였다. 새 검증도 종료했다.
공통 `ExperimentLogger`를 사용했고 W&B는 disabled다. filter 검증 run ID는 `f47d2bb821ae4d00a0c684f6d47aaae9`, Graph 경계 검증 run ID는 `db0781708f5843e1b0df5935152d1c89`이다.
이전 47.0→13.4ms는 CPU→CPU 필터 시간으로, 이번 CUDA→CUDA 입출력의 속도나 추가 절감량으로 재사용하지 않는다.

[최종 요약](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/completion.json) ·
[224개 카메라 검증](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/validation_v4/results.json) ·
[Graph 호환 검증](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/boundary_v4/results.json) ·
[원격 테스트 로그](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/validation_v4.log) ·
[실행 소스·설정](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/validation_v4/config.json) ·
[사용 예·제한](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/README.md).

원격 원자료: `/workspace/task10i_robot_filter_gpu_io_20261103_v1`.
로컬 수신 69개 파일 SHA를 확인했다. 공유 simulator 및 다른 Graph 구현은 수정하지 않았다.
