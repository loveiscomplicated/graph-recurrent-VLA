# Task 10-I — SAM2 실행과 로봇 필터 CPU 준비 겹치기

2026-11-03. 로봇 필터의 **현재 형상 변환·ConvexHull·카메라 투영 범위 준비를
CPU 작업 스레드에서 먼저 시작**하도록 연결했다. SAM2가 마스크를 만든 뒤 그
준비 결과를 받아 기존 GPU 필터를 실행한다. 이번 변경은 이 실행 순서에 한정된다.

```text
현재 카메라·로봇 형상 snapshot ─→ CPU 준비 ────────┐
                              SAM2 추론 ─→ 마스크 ├→ 기존 GPU 필터 → Clean Mask
```

## 구현과 사용

`RobotFilterPreparation`은 한 개의 CPU worker와 한 개의 pending frame만 가진다.
준비에는 기존 `CudaRobotFilter.prepare_surfaces`를 그대로 재사용한다. 모델과
GPU 필터는 호출자의 기존 스레드·CUDA stream·autocast 설정에 유지된다. Worker는
NumPy/SciPy 형상 계산만 하며 CUDA 연산이나 GPU 전체 동기화를 추가하지 않는다.

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.preparation import RobotFilterPreparation

robot_filter = CudaRobotFilter()
with RobotFilterPreparation(robot_filter) as preparation:
    pending = preparation.submit(cameras, current_fk_surfaces)
    raw_masks = run_existing_sam2()  # 기존 호출 스레드에서 실행
    clean_masks = preparation.filter(pending, raw_masks)
```

- 기존 `image_clean_masks_cuda`는 위 순서를 자동 사용한다. 지속 실행 시
  `preparation=`에 같은 context를 전달하여 worker를 재사용할 수 있다.
  `images_clean_masks_cuda`는 동기화된 두 카메라에 준비 결과 하나를 공유한다.
- `SAM2RobotFilter`는 `prepare_frame`/`cancel_prepared`를 제공한다.
  `SAM2Tracker.track`은 정상 `init_state`의 `num_frames`를 이용해 **각 forward
  프레임의 predictor를 진행하기 전에** 준비를 제출한다. 마스크가 나오면 같은
  프레임의 준비 결과를 소비한다. 프레임 순서가 달라지면 잘못된 자세를 쓰지 않고
  오류를 낸다. Prompt 초기화인 `start`는 기존 방식이다.
- `num_frames`가 없는 custom tracking state는 기존처럼 마스크 이후 필터링한다.
  일반 callback이나 callback 없는 tracker의 mask 동작은 유지된다.
- Raw depth·카메라 보정·TCP pose·FK 행렬·local vertices를 제출 시 복사한다.
  이후 원본 버퍼가 재사용되어도 진행 중 프레임의 입력은 바뀌지 않는다. RGB는
  준비에서 읽지 않으며 SAM2에는 원래 제공한 입력을 사용한다. 호출자는 RGB,
  카메라와 FK가 같은 시점이라는 기존 join을 보장해야 한다.
- Pending 결과는 한 번만 소비하며, SAM2 오류·preparation 오류·조기 종료·generator
  close에서 CPU 작업을 정리한다. 다른 프레임이나 이미 소비한 결과를 재사용하지
  않는다. `with SAM2RobotFilter(...) as callback:` 또는 `close()`로 worker를 닫는다.

Async snapshot 경로는 표준 CPU `CameraObservation`/`LinkSurface`를 받는다.
기존 직접 `CudaRobotFilter`의 custom-camera 경로는 그대로다. 로봇 형상·링크
순서/동률·2.5mm/6mm/0.5mm 판정·overlap 제거·원본 raw-mask 통계는 바꾸지 않았다.
Gate, Shape-First, Graph, 물리 시뮬레이터와 SAM2 모델 내부도 이번 작업에서 수정하지 않았다.

## 검증과 측정

로컬 macOS의 CPU 계약·tracking 회귀 테스트는 **8개 중 6개 통과, CUDA 전용
2개 skip**이었다. RTX3090의 관련 suite는 **16개 모두 통과**했다.

- CPU 이벤트로 준비 worker가 SAM2 진입 전에 시작하고, SAM2 역할의 함수가
  반환하기 전에 준비를 끝낼 수 있음을 검사했다. 단순히 추론 후 큐에 넣는
  구현이 통과하지 못하도록 실행 순서를 검사했다.
- Image/video 연결, 두 카메라 공유, CPU 입력의 사후 수정, 보정/자세 갱신,
  중복 제출/소비, 준비 예외, SAM2 예외, 조기 종료와 predictor close를 검사했다.
- 실제 CUDA 필터의 기존 no-readback·depth 경계·owner 동률·invalid depth·빈/겹침
  mask·high-resolution ROI·raw gate 통계 회귀 테스트를 통과했다.
- 저장된 actual repeat0 `10i_s1_0921_t080`의 **128/512/768/1024, 두 카메라
  총 8개 카메라 조건**에서 직렬 준비와 비동기 준비를 비교했다. Hull 계수와
  packed metadata, clean mask, flags, owner, interior, 예측 깊이 및 평가 범위가
  모두 **bitwise exact**였다. 같은 시점의 해상도별 비교이며 독립 장면 8개가 아니다.

같은 저장 입력의 512 두 카메라에서 기존 CPU `prepare_surfaces` 함수만
warmup3 뒤15회 측정한 중앙값은 **3.823ms**였다. 현재 world vertices, 두 카메라
bounds, visible Qhull, packed metadata를 포함한다. Snapshot/submit 비용, GPU
필터, 별도 FK, SAM2·Graph·파일 I/O는 제외했다. RTX3090/AMD EPYC 호스트,
Torch threads4, BLAS threads1의 기존 환경을 사용했다.

이 값은 **병렬화하는 준비 구간의 별도 시간**이다. 역사적인 필터 단독7.16ms에서
3.823ms를 빼서 새 필터 시간이라고 주장하지 않는다. 실제 SAM2 추론은 실행하지
않았고 image/video 연결에는 mock predictor를 사용했다. SAM2의 비동기 실행으로
기존 경로에서도 일부 CPU 준비가 이미 겹칠 수 있으며, 새 snapshot 비용과 CPU
경합도 있다. 따라서 실제 통합 절감량은 SAM2 최적화 경로가 정해진 뒤 같은
입출력 경계로 별도 측정해야 한다. GPU는 원격 RTX3090에서 실행했고 로컬 macOS
CUDA는 미실행이다.

## 재현성과 산출물

공통 `ExperimentLogger`를 재사용했고 W&B disabled, publishing 없음이다.
원격 root는 `/workspace/task10i_robot_filter_sam_overlap_20261103_v1`이다.
Frozen 입력은 수정하지 않았고, 원래 정지된 Task10-I 프로세스도 재개하지 않았다.
이번 범위에서 모델 학습·정책 rollout·외부 모델 API·설치·통합 모델 GPU 전환은 하지 않았다.

- [저장 입력 비교·15회 원자료·source/input hashes](../../outputs/task10i/robot_filter_sam_overlap_20261103_v1/remote/verified/results.json)
- [CUDA 회귀 테스트 로그](../../outputs/task10i/robot_filter_sam_overlap_20261103_v1/remote/tests_gpu_final.log)
- [로컬 CPU 테스트 로그](../../outputs/task10i/robot_filter_sam_overlap_20261103_v1/tests_cpu.log)
- [실행 전 checkout/source identity](../../outputs/task10i/robot_filter_sam_overlap_20261103_v1/source_identity.json)
- [최종 수정 파일 identity](../../outputs/task10i/robot_filter_sam_overlap_20261103_v1/final_identity.json)
- [API와 실행 경로](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/README.md)

재현 명령은 기존 CUDA Python 환경에서 다음과 같다. 새 output 경로를 사용한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled python -m unittest \
  visual.runtime.test.test_tracking_filter \
  graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_preparation \
  graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_gpu_io \
  graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_graph_boundary -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled python -m \
  graph.experiments.robot_filter_gpu_io_10i_20261103_v1.verify_preparation \
  --output /unique/new/output
```
