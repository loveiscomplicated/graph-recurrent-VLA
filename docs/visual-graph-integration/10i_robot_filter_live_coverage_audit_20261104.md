# Task10-I: 로봇 필터 최적화의 live 연결 전수 대조

2026-11-04 KST. **손목 gate의 구현·성능은 점검 범위에서 제외했다.**
기준 HEAD: `d42f777bd8a2a23eef4f814f5922783383c17948`와 당시 dirty 작업 트리.
CUDA 검증 실행 ID: `850b862b7d1743a9b77a4136044c5135`.

## 결론

기본 `graph.plan_a_bc.rollout.run_episode`의 CUDA 실행 경로에는 로봇 필터의
핵심 최적화가 연결되어 있다. 단순히 같은 클래스 이름을 참조하는지 확인한 것이
아니라 native GPU 카메라 입력부터 **gate 직전 raw/clean mask handoff**까지
실제 CUDA에서 실행했다. 공유 준비 1회, 두 카메라 batch filter 1회, 압축 clipping,
CUDA depth/mask, ray cache와 예외 후 재사용을 확인했다.

그러나 **저장 입력 기반 Task10-I 평가/보정 경로와 과거 live 평가 스크립트까지
모두 전환된 것은 아니다.** CPU mask 경계·준비 병렬화 누락, 보정 경로의 CPU
로봇 분류 잔존, 과거 adapter와 새 GPU 카메라의 타입 불일치가 있다.
또한 기본 live의 depth 변환은 필터 전용 Triton 커널 대신 공용 CUDA lookup으로
대체되어 있다. 이 차이는 계산 결과 손실이나 CPU 복귀가 아니며 속도 영향은 미측정이다.

## 변경별 기본 live 대조

| 최적화/변경 | 현재 기본 live | 확인 근거 |
| --- | --- | --- |
| CUDA 로봇 필터 기본 선택 | 연결됨 | `CurrentGeometry`가 CUDA 가능 시 `triton_fused`/`robot_filter_backend='cuda'` 선택 |
| SAM2 raw mask의 GPU 유지 | 연결됨 | live의 `mask_device='cuda'`; `TensorOnlineVisionAdapter` CUDA bool 출력; 필터 전에 dense CPU 복사 없음 |
| GPU clean mask 반환·raw 불변 | 연결됨 | CUDA 필터 출력 그대로 pre-gate handoff; raw tensor 동일성 및 clean bitwise 일치 확인 |
| 두 카메라의 world hull/bounds 준비 공유 | 연결됨 | current-frame `PendingPreparation` 하나와 `filter_cameras(..., prepared=...)` 한 번 |
| plane 수에 따른 kernel grouping | 연결됨 | 기본 `CudaRobotFilter`의 high-resolution 분기를 그대로 사용 |
| mask-union GPU row 압축 | 연결됨 | 기본 `compact=True`; `_compact_rows` → `_clip_compact` 실행 경로 |
| 기본 clip tile 32 | 연결됨 | 실제 생성 객체 `_clip_pixels==32`; 작은 해상도 기존 grouped 분기는 유지 |
| 화면 밖 hull 생략·원래 link slot 유지 | 연결됨 | 동일 `prepare_surfaces` 사용; 별도 live용 축약 classifier로 바꾸지 않음 |
| frame-local equation/normal/group metadata upload 공유 | 연결됨 | 같은 prepared 객체의 `device_equations`를 두 카메라가 공유 |
| 해상도/intrinsic/device별 bounded ray cache | 연결됨 | 영속 filter 객체 유지; 다음 build에서 ray pointer 재사용 확인 |
| 현재 depth/pose/hull 갱신 | 연결됨 | snapshot과 prepared 객체를 매 build 새로 생성; ray 이외 pose 캐시 없음 |
| signed-int16 depth를 GPU에서 float64로 변환 | 연결됨 | native CUDA int16 → 공용 `depth_from_mm` → `depth_cuda=` 전달; CPU `.depth` 변환을 우회 |
| 7.16ms 단계의 필터 전용 단일 Triton depth 커널 | 동일 커널은 미사용 | live는 공용 Torch lookup을 사용; 모든 int16 값의 결과가 fused converter와 정확히 같음 |
| CPU world/ConvexHull/투영 범위를 SAM2와 겹침 | 연결됨 | `submit` 후 callable vision 실행; `before_vision`; 한 worker를 계속 재사용 |
| 필터 중 dense readback/scalar sync 제거 | 연결됨 | `NoReadback` 및 명시적 synchronize 금지 guard 아래 실제 CUDA 통과 |
| 필터 한 번만 적용 | 기본 live에서 연결됨 | 준비 batch 1회; handoff에 이미 clean mask 전달; 기본 경로에 `GpuMaskBoundary`/tracker filter 중복 부착 없음 |
| vision 예외 때 pending 정리·worker 종료 | 연결됨 | 실제 default object에 vision 예외 주입 후 pending 해제 및 다음 build 성공; episode finally에서 close |
| native GPU 센서 출력에서 필터로 연결 | 연결됨 | live가 512 CUDA renderer/output wrapper와 tensor camera/snapshot 사용; 원래 CPU PhysX는 유지 |

초기 CPU strict 13.4ms 구현을 별도 callable로 실행하지 않는 것은 누락이 아니다.
후속 CUDA 필터가 그 판정 계약을 보존하는 대체 경로다. `SAM2RobotFilter`/image bridge와
`GpuMaskBoundary`도 반드시 중첩해야 하는 기능이 아니다. 기본 live는 바깥 builder에서
두 카메라를 묶어 동일 준비/필터 계약을 수행하므로 추가 callback을 붙이면 중복될 수 있다.

단일 준비 worker 안에서 shape별 Qhull은 기존 순서로 계산한다. 과거 시험 후 제외한
Qhull 4-worker 후보나 선택형 `cuda_resident`의 별도 staging arena를 기본 live가
쓰지 않는 것을 이번 채택 최적화의 누락으로 세지 않았다. Native GPU 센서 경로는
CPU image를 한 번에 upload하는 arena와 입력 조건도 다르다.

## 기본 live의 실제 호출 경로

- [rollout.py](../../graph/plan_a_bc/rollout.py) 37–42행: CPU PhysX를 유지하면서
  CUDA renderer, 두 카메라 512 해상도, `ObservationDevice(...,'cuda')`를 선택한다.
- 같은 파일 61–63행: dense RGB/depth를 NumPy로 내리지 않고
  `TensorCameraObservation`으로 전달한다. 작은 calibration/robot metadata는 host다.
- 110–112행: default `CurrentGeometry`와 tensor vision adapter를 생성한다.
- 126–129행: CUDA mask를 만드는 함수를 전달해 vision 실행을 build 안으로 미룬다.
- [geometry.py](../../graph/plan_a_bc/geometry.py) 187–200행:
  `TensorRobotPreparation(CudaRobotFilter())` 생성 → submit → vision → CUDA filter → handoff다.
- [tensor_ingress.py](../../graph/plan_a_bc/tensor_ingress.py): 기존 worker/prepare/filter를
  상속하고 current GPU snapshot 및 `depth_cuda`/`world_cuda`만 연결한다.
- 필터 구현은 [gpu_filter.py](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/gpu_filter.py)의
  동일 high-resolution compact 분기다. `diagnostics=False`가 기본이므로 live는
  owner/full-screen diagnostic을 불필요하게 만들지 않는다.

Raw mask의 CPU export는 Graph 완료 후 영상 artifact를 위한 `numpy_packet` 경계에
있다(rollout 130–131행). 이는 필터 전 raw mask를 CPU로 내렸다 재업로드하는 경로가 아니다.
이 영상 export가 전체 live 지연에 들어가는지는 별도 문제이며 이번 필터 누락 판정과
속도 측정에는 포함하지 않았다. Robot FK/Qhull의 CPU 잔존도 의도한 준비 경로다.

## 남아 있는 미연결 경로

### 1. Task10-I Stage4A2 frozen inference

[frozen_inference.py](../../graph/task10i_stage4a2/frozen_inference.py)는
`geometry.from_h5(..., tensor_images=False)`로 원래 CPU 센서를 유지하고,
기존 `OnlineVisionAdapter.process(..., episode, t)`에서 `mask_device`를 지정하지
않는다. 따라서 adapter는 기본 CPU mask를 반환한다. 이를 완성된 `values`
dictionary로 `geometry.build`에 전달한다.

최종 geometry의 CUDA filter 자체는 사용하지만, **SAM2 GPU mask 유지와
SAM2 이전 준비 제출은 미연결**이다. Geometry는 mask를 다시 CUDA로 올리며,
준비 스케줄은 `masks_already_available`다. 이미 끝난 SAM2와 소급해서 겹칠 수 없다.
이 경로를 호출하는 `live_engineering_runner`/`final_path_engineering`도 같은 상태다.

### 2. Task10-I correction의 CPU 로봇 분류

[tri_model_integration.py](../../graph/task10i_stage4a2/tri_model_integration.py)의
`CausalCorrection.process`는 112/119/197/251행에서 조건에 따라
`geometry.system.prepare(cameras,data,masks,surfaces)`를 직접 호출한다.
`clean_masks=`를 제공하지 않으므로 [PlanA.prepare](../../graph/plan_a_geometry/core.py)의
174–176행은 기존 CPU `classify_robot`를 실행한다.

뒤의 최종 geometry build는 별도로 CUDA filter를 수행한다. 따라서 이 보정 경로에는
**CPU 로봇 분류 + 마지막 CUDA 필터**가 남아 있다. `--geometry-cache`의 memo는
같은 프레임의 CPU classifier 반복을 줄이지만, 마지막 CUDA prepared result와 공유하는
구조는 아니다. 매 프레임 무조건 N회 계산한다고 정량 주장하지는 않는다.

이것은 손목 gate 계산 문제가 아니다. 보정의 `evidence`가
`prepared['filters'][camera]['confirmed_robot'/'uncertain']`를 직접 읽는 기존
로봇 분류 계약이다. 최적화 경로의 clean mask만 넘기면 같은 계약을 충족하지 않는다.
현재 기본 Plan A live는 이 correction 경로를 호출하지 않는다.

### 3. 과거 live 평가 호출자의 image/adapter 연결

아래 호출자는 현재도 기존 CPU `OnlineVisionAdapter`를 사용하며 mask dictionary를
먼저 완성한다.

- [graph_recurrent10g/eval_runner.py](../../graph/graph_recurrent10g/eval_runner.py): 20행 import, 101–102행 호출
- [geometry_reach10d/evaluate.py](../../graph/geometry_reach10d/evaluate.py): 24행 import, 45–46행 호출
- [predstate10i/rollout.py](../../graph/predstate10i/rollout.py): eval_runner의 adapter를 가져와 73–74행에서 호출
- [handoff10h/runner.py](../../graph/handoff10h/runner.py): 같은 adapter를 가져와 73–74행에서 호출

이 호출 패턴에는 CUDA mask 유지와 preparation-before-vision이 없다. 더구나 이들이
공유하는 최신 `rollout.sensors`는 CUDA RGB를 반환하므로 기존 CPU adapter의
`np.asarray(rgb)`와 맞지 않는다. 실제 CUDA uint8 RGB를 이 adapter에 넘겨,
모델을 호출하기 전에 다음 오류가 발생하는 것을 재현했다.

```text
can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

이는 해당 입력/adapter 조합의 호출 계약 오류를 재현한 것이며, 과거 모든 simulator
설정과 정책 rollout를 실제로 실행한 결과는 아니다. CUDA 입력을 만들기 전 별도 환경
문제가 생기는 경우까지 평가하지 않았다. 기존 고정 source copy나 중단된 원격 실험이
현재 checkout으로 자동 교체된다는 뜻도 아니다.

### 4. 마지막 depth 커널의 차이

기본 live의 `TensorRobotPreparation.submit`은 `tensor_fusion.depth_from_mm`을
쓰고, `filter`에서 미리 계산된 `depth_cuda`를 넘긴다. 이 때문에 필터 내부
`_depth_from_mm`/`_depth_mm` 및 `CudaRobotFilter.depth_tensor`의 fused 경로는
live에서 실행되지 않는다.

GPU 변환과 signed-depth/NaN 의미는 유지된다. 실제 CUDA에서 int16 전체
65,536개 값의 NaN 위치와 유효 float64 값을 fused 경로와 대조해 **정확히 일치**했다.
공용 lookup은 후속 단계와 depth를 공유한다. 반면 필터 전용 fused 구현과 실행하는
kernel/임시 배열이 동일한 것은 아니다. 어느 쪽이 현재 live에서 빠른지는 재측정하지
않았으므로 누락으로 인한 속도 저하나 7.16ms 재현을 단정하지 않는다.

## 이전 지적 중 현재 수정된 항목

직전 점검에서 재현했던 `perframe_cache.install`의 `clean_masks` keyword 오류는
현재 코드에서 수정되어 있다. Keyword default와 `clean_masks` 전달을 처리하며,
CPU fixture로 wrapper를 설치한 뒤 supplied-clean 경로가 classifier를 호출하지 않는
것을 확인했다(memo miss 0). 이번 미연결 목록에 그 과거 오류를 반복해서 넣지 않았다.
다만 위 correction 경로는 clean mask를 공급하지 않으므로 다른 문제다.

## 실행 검증

- 로컬 기존 suite: **15개 중 14개 통과, CUDA 전용 1개 skip, 실패 0**.
  실제 `run_episode` 본체의 지연 mask producer/출력 device 요청은 fixture로 검증했다.
- RTX3090 기존 suite: **24개 전부 통과, skip/실패/오류 0**.
  GPU filter 경계·tie·threshold·overlap·invalid depth·compaction/tile·pose/intrinsic
  refresh·batch 공유·signed depth 전수·기존 bridge 및 worker/caller 계약을 포함한다.
- 별도 native CUDA case 2개: `10i_s1_0921_t080`, `10i_s1_0531_track_t011`,
  둘 다 512 두 카메라. 실제 default builder의 robot-filter handoff까지만 실행했다.
  Saved actual SAM2 mask를 사용했고 raw/clean CUDA, direct filter와 bitwise 일치,
  batch 1회/shared prep 1회, ray cache 재사용, 새 prepared 객체 및 vision 오류 후
  재사용을 확인했다. `NoReadback`과 global synchronize 금지 guard도 통과했다.
- Gate 이후 `_build_prepared`는 가로채 이번 새 native probe에서 실행하지 않았다.
  기존 unit suite의 작은 preclean fixture는 raw/clean 전달 계약 확인에만 사용했다.
- GPU depth 공용/fused lookup의 전체 int16 값 비교 및 legacy CUDA-RGB 타입 오류
  재현을 추가했다. 로컬 Mac CUDA/MPS, 실제 SAM2 weight 추론, detector/API,
  simulator 생성·물리 step, policy closed loop 및 새 latency 측정은 하지 않았다.

단순한 source 검토만으로 실제 모델 실행까지 검증했다고 주장하지 않는다.
원격 장치는 RTX3090, Torch 2.14.1+cu130/CUDA13.0의 기존 환경이다. 공통
`ExperimentLogger`를 재사용했고 W&B disabled, 실행 상태는 `finished`다.
기존 SAPIEN Vulkan ICD import 경고는 있었으나 simulator는 실행하지 않았다.

## 소스 고정과 산출물

동시 작업을 보존하려고 1,069개 코드/config 파일을 별도 snapshot으로 고정해
원격 격리 디렉터리에서 검사했다. 기존 실제 SAM2 측정의 전체 소스 복사본을 기반으로
이번 archive를 덮어썼으며, 기반 archive hash도 회수 검증에 기록했다.
원래 paused PID 7개는 모두 `T` 상태를 유지했다.
새 설치·학습·외부 모델/API·게시·프로세스 재개·production 수정은 하지 않았다.
이번 작업의 추가물은 diagnostic runner/README, 결과 및 이 보고서다.

검사 후 주요 live/filter 파일 hash를 다시 대조했다. `tensor_fusion.py`에는 다른
작업의 변경이 생겼지만, 로봇 필터가 사용하는 `_depth_table`/`depth_from_mm`의
AST는 고정 snapshot과 같았다. 그 외 점검한 주요 연결 파일은 snapshot과 일치했다.
원격은 `.git` 없는 복사본이므로 Logger의 null commit/dirty=false는 clean checkout
증거가 아니며, 실제 파일 hash와 archive를 기준으로 삼는다.

- [검사 runner/방법](../../graph/experiments/robot_filter_live_audit_10i_20261104_v1/README.md)
- [CUDA24·native 두 사례·depth/타입 결과](../../outputs/task10i/robot_filter_live_audit_20261104_v1/remote/results.json)
- [CUDA24 전체 로그](../../outputs/task10i/robot_filter_live_audit_20261104_v1/remote/tests.log)
- [원격 실행 로그](../../outputs/task10i/robot_filter_live_audit_20261104_v1/measured.log)
- [로컬 확인·현재 소스 변경 여부](../../outputs/task10i/robot_filter_live_audit_20261104_v1/local_checks.json)
- [고정 source identity](../../outputs/task10i/robot_filter_live_audit_20261104_v1/source_identity.json) 및
  [실행 source archive](../../outputs/task10i/robot_filter_live_audit_20261104_v1/source.tar.gz)
- [회수·소스 검증](../../outputs/task10i/robot_filter_live_audit_20261104_v1/verification.json)
- [종료 상태 점검](../../outputs/task10i/robot_filter_live_audit_20261104_v1/remote_postcheck.txt)
