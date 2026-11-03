# Task 10-I — 시뮬레이터 원본 관측의 CUDA 출력

2026-11-03. 사용자의 범위 정정에 따라 **시뮬레이터 관측 출력 경계만** 완료했다.
물리는 CPU에 유지하고, 이미지의 CPU readback을 생략하며 나머지 CPU 관측을
반환 직전에 CUDA로 옮기는 옵션이다. 통합 모델 전체의 GPU 전환 완료 보고가 아니다.

## 앞서 진행했던 작업과 중단 상태

앞선 요청을 넓게 해석하여 센서 업로드부터 로봇 필터·손목 gate·PCA5·Shape-First·
tensor packet·작은 최종 Graph 소비자까지 연결하는 선택적 후보를 구현했다.
512 두 카메라 예비 비교의 중앙값은 약 **164.2ms → 87.6ms**였다. 이는 저장된
CUDA 마스크와 CPU 센서 입력에서 CUDA Graph를 반환하는 별도 지표이며,
역사적인 **필터 단독 7.16ms**와 비교할 수 없다. 기준의 캐시 수명 등 최종 비교
정리가 남았고, 동시 진행된 object-group 변경도 합쳐서 검증하지 못했으므로
확정 성능이나 채택 결과로 취급하지 않는다.

사용자 정정 시점에 Graph 비교 **439/460조건**까지 차이 없이 진행했고,
별도 필터 비교는 **460/460조건**을 통과했다. 후자는 두 카메라의 유효 평가
픽셀에서 flags·owner·interior·clean mask가 정확히 일치했고 예측 깊이 차이의
최댓값도 0이었다. 이 조건들은 연관된 저장 시점을 포함하므로 독립 표본 수가 아니다.
진행 중이던 이번 후보의 검증 프로세스만 종료했다. 원래 Task10-I 정지 프로세스는
재개하지 않았고 마지막 확인에서도 기존 7개 PID가 모두 T 상태였다.

넓어진 후보의 코드·원자료는 미채택 실험 상태로 보존했다. 기존 기본 backend를
그 후보로 전환하지 않았다. 다른 작업의 변경도 되돌리지 않았다.

- [중단 시점 기록](../../outputs/task10i/cuda_ingress_20261103_v1/interrupted_progress.json)
- [예비 반복 원자료](../../outputs/task10i/cuda_ingress_20261103_v1/paired03_results.json)
- [필터 전체 비교](../../outputs/task10i/cuda_ingress_20261103_v1/filter_all_results.json)
- [Graph 부분 검증 로그](../../outputs/task10i/cuda_ingress_20261103_v1/validation_partial.log)

## 이번에 완료한 출력 경계

```python
from envs.maniskill import make_peg_insertion_env

env = make_peg_insertion_env(
    device="cpu",
    render_backend="cuda",
    observation_device="cuda",
    obs_mode="rgb+depth",
)
try:
    observation, info = env.reset(seed=0)
finally:
    env.close()
```

| 항목 | RTX3090에서 확인한 동작 |
| --- | --- |
| 물리 | `physx_cpu`, underlying task device는 CPU 유지 |
| 카메라 | `sapien_cuda`; RGB/depth가 출력 wrapper 진입 전부터 CUDA |
| 이미지 이동 | 네이티브 CUDA camera accessor를 사용하여 host image readback 생략 |
| qpos·카메라 보정 등 CPU 관측 | 출력 경계에서 각각 CUDA로 한 번 이동 |
| 최종 reset/step 관측 | 배열 leaf가 CUDA tensor로 반환 |
| 원본 형식 | shape/dtype 보존; RGB uint8, depth signed int16 mm 그대로 |

`ObservationDevice`는 원본 관측 tree의 배열만 옮긴다. 이미 같은 CUDA 장치에
있는 tensor는 `.to(device)`가 같은 객체를 반환하므로 다시 복사하지 않는다.
CPU NumPy 변환이나 특징 계산을 끼워 넣지 않는다. Depth의 -32768·0 같은 원본
값도 바꾸지 않는다. CPU에서 생성된 상태에는 최초 H2D가 필요하다.

설치된 ManiSkill `RenderCamera.get_picture`에서 CPU physics와 무관하게
`sapien_cuda` 분기가 `get_picture_cuda(...).torch()`를 쓰는 것을 확인했다.
실제 native-camera 테스트는 wrapper **이전** RGB/depth device가 CUDA이고,
qpos/intrinsic이 CPU이며, 반환 후에는 이 값들이 모두 CUDA임을 검사했다.

`observation_device="cuda"`만 지정하면 기존 CPU renderer 출력의 이미지까지
한 번 업로드한다. 이미지 readback 자체를 피하려면 위 예제처럼
`render_backend="cuda"`도 명시한다. 새 인자를 생략한 기존 호출의 동작은 유지된다.
`reward`, 종료 flag, `info`, 직접 GT accessor, action 인터페이스는 기존 동작이다.
`env.unwrapped.device`는 여전히 CPU이므로 물리 action은 그 장치에 맞춰 전달한다.
이번 수정은 후속 모델의 NumPy 소비자를 바꾸거나 전체 파이프라인의 왕복을
제거하지 않는다. 새 출력 옵션은 명시적으로 선택하는 기능이다.

## 검증과 범위

- 로컬 macOS: 관측 계약 및 기존 simulator suite **11개 중 9개 통과,
  CUDA 전용 2개 skip**. 실제 CPU reset/step/RGB rendering을 실행했다.
- 기존 Linux RTX3090: 관측 계약과 기존 simulator suite **11개 모두 통과**.
  CPU physics에서 CPU-rendered 관측을 CUDA로 반환하는 reset/step도 실행했다.
- 별도 프로세스의 native CUDA camera suite **1개 통과**. CPU physics + CUDA
  renderer + CUDA 관측을 실제 reset/step에서 확인했다. SAPIEN renderer 선택이
  process-global이어서 CPU renderer suite와 분리했다.
- dtype/shape/원본 값 보존, signed-depth 경계값, 같은 GPU tensor의 재복사 없음,
  action/info 비변경, 기본 CPU backend 및 명시적 renderer 전달을 검증했다.
- Simulator API와 raw 출력만 실행했다. 모델 추론·학습·정책 rollout·패키지 설치·
  게시·원래 정지 실험 재개는 하지 않았다. 이번 출력 경계의 latency나 전체 모델
  50ms 달성을 측정·주장하지 않는다. 로컬 macOS의 CUDA는 미실행이다.

공통 `ExperimentLogger`를 재사용했고 W&B는 disabled였다. 첫 검증 runner는
소스 묶음에서 logger 의존 파일 하나를 빠뜨려 테스트 시작 전에 실패했다.
기존 `experiment_artifacts.py`를 함께 전달한 뒤 동일 runtime 코드로 재실행했다.
검증 환경·package 버전·source hash·성공 run identity는 아래 결과와 metadata에 있다.

- [성공 검증 결과](../../outputs/task10i/simulator_observation_cuda_20261103_v1/remote/verified_v2/results.json)
- [CPU physics / CUDA 관측 로그](../../outputs/task10i/simulator_observation_cuda_20261103_v1/remote/verified_v2/cpu_physics_output.log)
- [네이티브 CUDA 카메라 로그](../../outputs/task10i/simulator_observation_cuda_20261103_v1/remote/verified_v2/native_cuda_camera.log)
- [로컬 CPU 검증 로그](../../outputs/task10i/simulator_observation_cuda_20261103_v1/tests_cpu.log)
- [소스 identity](../../outputs/task10i/simulator_observation_cuda_20261103_v1/source_identity_final.json)
- [API 및 실행 명령](../../envs/maniskill/README.md)

원격 검증 root는 `/workspace/task10i_simulator_observation_cuda_20261103_v1`이며,
기존 환경과 인스턴스를 그대로 사용했다. 출력 경계 작업은 여기서 완료·중단했다.
