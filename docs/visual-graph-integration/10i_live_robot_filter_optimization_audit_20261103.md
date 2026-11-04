# Task10-I: live 진입점의 로봇 필터 최적화 연결 점검

점검일: 2026-11-03 KST. 대상: 현재 로컬 작업 트리.

기본 `graph.plan_a_bc.rollout.run_episode`에는 최적화한 CUDA 로봇 필터와
SAM2 이전 CPU 준비 제출이 연결되어 있다. 별도 opt-in 없이 기본값으로 동작한다.
다만 다른 과거 평가/실험 진입점까지 모두 연결된 상태는 아니다. 특히 Task10-I
Stage4A2 `frozen_inference`는 SAM2 마스크를 CPU로 먼저 만든 뒤 geometry에
넘기므로 준비와 SAM2가 겹치지 않는다. 선택형 과거 geometry-cache wrapper에는
현재 clean-mask 전달 API와의 호환 문제도 있다.

## 기본 live 호출 흐름

1. [rollout.py](../../graph/plan_a_bc/rollout.py)의 `evaluate`가 모델을 생성하고
   `run_episode`를 호출한다. 현재 작업 트리의 detector는 `LiveGrounding`,
   tracker는 `SAM2Tracker`다. 이번 점검에서는 detection/API나 실제 모델을
   호출하지 않았다.
2. `run_episode`는 `CurrentGeometry`를 생성할 때 coordinate frame만 명시한다
   (104–105행). `perception_backend`와 `robot_filter_backend`를 덮어쓰지 않는다.
3. [geometry.py](../../graph/plan_a_bc/geometry.py)의 기본값은
   `perception_backend='triton_fused'`, 이때 로봇 필터는 `cuda`다(94–102행).
   CUDA가 없으면 명시적으로 실패하며 자동 CPU fallback하지 않는다.
4. `run_episode`는 두 카메라의 vision 호출을 `current_masks` 함수 안에 두고
   **함수 자체**를 `geometry.build`에 넘긴다(118–123행). 먼저 SAM2를 실행하고
   완성된 dictionary를 넘기는 형태가 아니다.
5. `build`는 현재 data/camera/depth/calibration/FK snapshot을 준비한 뒤
   `RobotFilterPreparation.submit`을 호출한다. 그 다음에 `masks()`를 실행한다
   (178–182행). 기록 값은 `before_vision`이다.
6. [preparation.py](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/preparation.py)의
   영속적인 단일 CPU worker가 현재 world vertices, visible ConvexHull 및 카메라
   투영 범위/metadata를 준비한다. SAM2와 CUDA 필터 호출은 caller thread에 남는다.
   outstanding snapshot은 한 개로 제한하며 다음 프레임과 섞이지 않는다.
7. Live vision 호출은 `mask_device='cuda'`를 명시한다(rollout 121행).
   [OnlineVisionAdapter](../../graph/integration_baseline/online_vision_adapter.py)는
   CUDA bool mask를 반환하며 이 분기에서는 dense SAM2 mask의 CPU 복사가 없다.
   기록용 픽셀 수 두 scalar는 host에서 읽는다. Adapter 자체의 일반 default는
   여전히 `cpu`지만 이 live 호출에서는 명시적으로 override된다.
8. `build`가 worker 결과를 받은 뒤 `CudaRobotFilter.filter_cameras`를 호출한다.
   기본 `CudaRobotFilter()`는 `compact=True`, `clip_pixels=32`,
   `depth_from_mm=True`다. 필터 출력 clean mask는 CUDA bool tensor다.
9. 필터 이후 raw/clean을 CPU gate에 넘긴다(geometry 187–190행).
   [PlanA.prepare](../../graph/plan_a_geometry/core.py)는 `clean_masks=`가 있으면
   `classify_robot`를 다시 실행하지 않는다. Raw mask는 raw denominator용으로
   유지한다. 따라서 중복 CPU 로봇 분류는 없다.
10. Episode 종료 시 `geometry.close()`로 worker를 정리한다. Vision 예외가 나도
    pending preparation을 discard한다. Trace에는 `robot_filter_backend`와
    `robot_preparation`이 각각 기록된다(rollout 145행).

## 다른 진입점과 남은 경계

| 진입점 | SAM2 출력 → 필터 | 준비와 SAM2 겹침 | 판정 |
| --- | --- | --- | --- |
| `plan_a_bc.rollout.run_episode` | CUDA → CUDA | 함수 callback 이전 submit | 연결됨 |
| `plan_a_bc.cache.cache_episode` | CUDA → CUDA | 함수 callback 이전 submit | 연결됨; 해당 기존 계약 테스트 통과 |
| `task10i_stage4a2.frozen_inference.run` | adapter 기본 CPU 출력 → geometry에서 CUDA 재업로드 | 완성된 mask dict를 전달 | 미연결; 기본 CUDA 필터 자체는 사용하지만 overlap은 없음 |
| `graph_recurrent10g.eval_runner.episode`, `predstate10i.rollout` | adapter 기본 CPU 출력 → geometry에서 CUDA 재업로드 | 완성된 mask dict를 전달 | 같은 미연결 패턴 |
| `geometry_reach10d.evaluate.live_graph` | adapter 기본 CPU 출력 → geometry에서 CUDA 재업로드 | 완성된 mask dict를 전달 | 같은 미연결 패턴 |

Stage4A2의 근거는 [frozen_inference.py](../../graph/task10i_stage4a2/frozen_inference.py)
172행의 `.process(..., episode, t)` 및 183행의 `geometry.build(..., values)`다.
`CurrentGeometry.build`는 dictionary를 받으면 `masks_already_available`로 기록한다.
SAM2가 이미 끝난 뒤라 이 경우 준비를 과거 SAM2 실행과 겹칠 수 없다.
`live_engineering_runner`와 `final_path_engineering`도 이 모듈을 subprocess로 부른다.
과거 고정된 원격 source copy가 현재 checkout으로 자동 교체되었다는 뜻은 아니다.

Stage4A2가 `--geometry-cache`로 설치하는
[perframe_cache.install](../../graph/task10i_stage4a/perframe_cache.py)의 wrapper는
`prepare(cameras,data,masks,surfaces)` 네 인자만 받는다. 현재 default geometry는
`prepare(..., clean_masks=...)`를 호출하므로 이 조합에는 API 호환 문제가 있다.
로컬 fixture에서 아래 오류를 재현했다.

```text
install.<locals>.prepare() got an unexpected keyword argument 'clean_masks'
```

이는 현재 API와 과거 wrapper 조합의 재현이며, 실제 Stage4A2 모델 실행 전체를
새로 수행한 결과는 아니다. 기존 frozen source나 실행 프로세스는 건드리지 않았다.

기본 live도 전체 입력부터 GPU로 통일된 상태는 아니다.

- `rollout.environment`는 `make_peg_insertion_env(device='cpu', ...)`를 호출하며
  `render_backend='cuda'`/`observation_device='cuda'`를 선택하지 않는다(36–37행).
  공유 factory의 CPU 기본 renderer를 사용하도록 작성되어 있다.
- `sensors`는 RGB/depth/calibration을 `numpy(...)`로 변환한다(55–57행).
- Adapter는 CPU RGB를 JPEG에 쓰고 SAM2 loader로 읽어 GPU에 올린다(87–92행).
- GPU 필터 이후에는 기존 wrist gate 때문에 raw/clean mask가 CPU로 내려간다.

따라서 확인된 연결 범위는 **SAM2 output을 GPU에 유지해 최적화한 필터를 적용하고,
그 앞의 CPU 형상 준비를 vision callback과 겹치는 것**이다. 선택형 simulator CUDA
출력이나 별도 resident ingress prototype까지 기본 live에 연결되었다고 해석하지 않는다.

## 실행 검증과 범위

저장소 `.venv/bin/python`으로 기존 관련 테스트를 실행했다.

```bash
.venv/bin/python -m unittest \
  graph.plan_a_bc.test_robot_overlap \
  graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_preparation \
  graph.integration_baseline.tests.test_online_vision -v
```

결과: **18개 중 14개 통과, CUDA 전용 4개 skip, 실패 0개**.
이 중 `test_real_live_entrypoint_defers_vision_and_uses_cuda_masks`는 실제
`run_episode`를 호출해 provider가 `build` 안에서 실행되는 것과 `mask_device='cuda'`를
확인한다. Simulator/model/geometry 의존성은 fixture로 교체한 계약 테스트다.
별도 테스트가 실제 `CurrentGeometry`와 CPU worker의 submit-before-vision 순서,
vision 예외 정리, snapshot 수명 및 clean-mask 중복 필터 생략을 확인한다.

이 Mac에는 CUDA가 없어 이번 점검에서 실제 GPU SAM2나 물리 simulator/live policy를
다시 실행하지 않았다. 앞선 원격 실제 SAM2 실행은
[실측 보고서](10i_real_sam_overlap_20261103.md)에 별도로 기록되어 있다.
이번 점검은 현재 소스 연결과 호출 계약 검증이며 원격 배포/실행 파일 동일성 검증이나
속도 재측정은 아니다. SAPIEN builtin Vulkan fallback import 경고는 있었고
테스트 실패를 일으키지는 않았다.

Runtime 구현은 수정하지 않고 이 보고서와
[점검 identity](../../outputs/task10i/live_robot_filter_audit_20261103/identity.json)를
추가했다. 공통 logger/publisher는 변경하지 않았으며 모델 평가, 학습, API 호출,
게시 및 중단 실험 재개를 수행하지 않았다. 현재 HEAD와 dirty 소스 hash는 identity에
기록했으므로, 동시 작업으로 바뀔 수 있는 commit명만으로 점검 범위를 추정하지 않는다.
