# 10-I: live 진입점의 GPU 관측 출력 연결·검증 — 2026-11-04

> 후속 변경: 이 보고서의128×128은 당시 검증 해상도다. 현재 live는 [512×512 변경·검증 보고서](10i_live_512_20261104.md)를 따른다.

live `rollout.environment`에서 기존 native CUDA 렌더러와 공통
`ObservationDevice`를 사용하도록 연결을 정리했다. 두 카메라의 RGB·Depth는
GPU에서 생성된 상태로 live 센서 어댑터에 전달된다. reset/step의 나머지
관측 tensor leaf도 공통 출력 wrapper를 거쳐 CUDA로 반환한다.

VastAI **58752018 / RTX3090**에서 실제 환경 reset1회와 zero-action step2회를
실행했고, 관련 테스트 **7개 모두 통과**했다. 물리 backend는 `physx_cpu`,
렌더러는 `sapien_cuda`였다. 학습·정책 rollout·Terra API·실제 SAM2 추론이나
새 성능 측정은 이번 검증에 포함하지 않았다.

## 확인한 기존 구현과 수정

공통 simulator factory에는 이미 `render_backend='cuda'`,
`observation_device='cuda'`와
[ObservationDevice](../../envs/maniskill/observation_device.py)가 있었다.
이 구현은 native CUDA RGB/Depth에 `.to(cuda)`를 적용해도 재업로드하지 않고,
나머지 raw 관측 배열만 같은 출력 장치로 옮긴다.

최신 작업 트리에는 이미 `TensorCameraObservation` 및 tensor live vision
어댑터 변경이 존재했다. 이를 보존·재사용했고 별도 GPU 관측 변환기를 만들지
않았다. 다만 live 환경은 factory 호출에 `observation_device='cuda'`를 주고도
곧바로 `.unwrapped`를 사용해 이 출력 wrapper를 제거하고 있었다.
이 경우 native RGB/Depth는 CUDA지만 보정·관절 등의 raw 관측 출력은 CPU에 남는다.

[rollout.environment](../../graph/plan_a_bc/rollout.py)의 연결을 다음과 같이 바꿨다.

```python
task = make_peg_insertion_env(
    device='cpu', obs_mode='rgb+depth',
    control_mode='pd_ee_delta_pose', render_backend='cuda',
).unwrapped
env = ObservationDevice(TimeLimitWrapper(task, 250), 'cuda')
```

factory의 기본 horizon을 기존 live의250 step으로 교체한 뒤, 공통 출력
wrapper를 바깥에 유지한다. wrapper 구현, 물리 backend, action contract,
controller 설정,250-step horizon 및 sensor 해상도는 새로 바꾸지 않았다.
설치된 ManiSkill `TimeLimitWrapper`와 `BaseEnv` signature/source를 확인하고
실제 reset/step으로 검증했다.

`rollout.sensors`의 기존 tensor 경로는 RGB/Depth를 `.numpy()`로 바꾸지 않고
[TensorCameraObservation](../../graph/persistent_geometry/tensor_camera.py)에
CUDA clone으로 담는다. renderer가 다음 프레임에서 buffer를 재사용해도 현재
관측 snapshot의 소유권을 유지하기 위한 on-device 복사다. dtype·값은 바뀌지 않는다.

작은 robot/FK·calibration 메타데이터는 기존 NumPy 검증 경로에서 사용하므로
센서 어댑터 이후 일부 메타데이터가 다시 CPU로 이동한다. Terra 요청 이미지
인코딩, 명시적인 영상/파일 저장의 host 경계도 이번에 제거한 것은 아니다.
이 변경은 **live 이미지가 CPU 배열로 내려갔다 다시 올라가는 경계와 raw 관측
반환 장치**에 대한 연결·검증이다.

## 실행 결과

| 항목 | 실제 결과 |
| --- | --- |
| 물리 backend / device | `physx_cpu` / CPU |
| 렌더 backend | `sapien_cuda` |
| base_camera RGB | CUDA uint8, `[128,128,3]` |
| base_camera raw Depth | CUDA int16, `[128,128]` |
| hand_camera RGB | CUDA uint8, `[128,128,3]` |
| hand_camera raw Depth | CUDA int16, `[128,128]` |
| reset/step raw 관측 tensor leaf | 모두 CUDA |
| native 이미지의 wrapper 전·후 pointer | 동일: 재업로드/불필요한 출력 복사 없음 |
| sensors 내부 RGB·Depth host 변환 | 금지 guard를 건 상태로 통과 |
| 반환 camera snapshot | CUDA, 원본과 값·dtype 일치, 별도 image buffer 소유 |
| horizon |250 유지;100에서 조기 truncation하지 않음 |

이번 native smoke의 센서 해상도는 설치 환경 기본128×128이다.
512×512로 설정을 변경하거나,512 전체 퍼셉션 성능을 다시 측정하지 않았다.
이전512 Graph benchmark의 시간과 이 smoke 실행을 합쳐 전체 시간으로 보지 않는다.

[새 live 테스트](../../graph/plan_a_bc/test_live_observation_output.py)는 다음을 확인한다.

- custom horizon 바깥에 공통 출력 wrapper가 유지되고 reset/step마다 호출된다.
- 물리/control 설정, action 전달과250-step truncation을 유지한다.
- 실제 native 이미지가 wrapper에 들어오기 전부터 CUDA이고 wrapper 후 pointer가 같다.
- reset1회 + CPU zero-action2회 후 raw 관측 tensor가 모두 CUDA다.
- `sensors`에서 이미지 저장소를 NumPy로 내리면 실패하도록 막아도 실행된다.
- TensorCameraObservation의 RGB·signed-int16 Depth 값과 snapshot 소유권을 유지한다.

기존 `tests.test_observation_device_native_gpu` 및
`tests.test_observation_device.OutputDeviceContracts`도 재사용했다.
Vast에서는 총7개, 실패0/skip0이었다. 로컬 macOS에서는6개 중4개 통과,
CUDA 전용2개 skip이었다. 실제 CUDA/renderer 실행 증거는 Vast에서 얻었다.
공통 simulator/renderer 코드 자체는 수정하지 않았다.

## 실험 식별과 자료

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했다.
run ID **`e00c583b928943bb89edc7ef66e39836`**,
team=`graph`, job_type=`diagnostic`, seed42, `WANDB_MODE=disabled`다.
새 package/driver 설치, API call, artifact publishing, commit은 하지 않았다.
작업 전부터 있던 Terra·gate·tensor ingress 변경을 보존했다.

원격 `/workspace/task10i_live_gpu_observation_output_20261104_v1/source_v3`에서
실행했다. Python3.12.15, Torch2.14.1+cu130, NumPy2.5.4, ManiSkill3.0.1,
SAPIEN3.0.3, Gymnasium1.4.0, CUDA13.0/RTX3090이다.
실행 전 source1,779개 파일의 SHA를 확인했다.

초기 v1/v2의 독립 source bundle은 각각 `tools`/`recurrent` 의존 소스가 빠져
live 테스트 import가 실패했다. 해당 로그를 보존하고 전체 source bundle의
새 v3에서 다시 실행했다. v3 결과만 최종 통과 근거로 사용한다.

- [최종 summary](../../outputs/task10i/live_gpu_observation_output_58752018_20261104_v1/remote/verified_v3/summary.json)
- [실제 CUDA 테스트 로그](../../outputs/task10i/live_gpu_observation_output_58752018_20261104_v1/remote/verified_v3/tests.log)
- [로컬 테스트 로그](../../outputs/task10i/live_gpu_observation_output_58752018_20261104_v1/local_tests.log)
- [실행 source archive](../../outputs/task10i/live_gpu_observation_output_58752018_20261104_v1/source_v3.tar.gz)
- [source SHA manifest](../../outputs/task10i/live_gpu_observation_output_58752018_20261104_v1/source_v3_manifest.json)
- [결과·이전 실패 로그 묶음](../../outputs/task10i/live_gpu_observation_output_58752018_20261104_v1/result_v3.tar.gz)
- [검증 실행 코드](../../graph/experiments/live_gpu_observation_output_58752018_20261104_v1/verify.py)
- [기존 공통 GPU 관측 출력 보고서](10i_simulator_observation_cuda_20261103.md)

최종 source manifest SHA256:
`7c98bfd5988c0f732d8275ee8a79bcfa14b1261d5d7ef957de5b74339a114f07`.
