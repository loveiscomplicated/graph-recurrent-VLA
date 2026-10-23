# RTX 3090 Recurrent GPU 배치 평가 검증

2026-10-23, Vast.ai 57321572, RTX 3090 24GB 한 장에서 실행했다.
시작 브랜치는 `recurrent`, commit은 `7636f52eb04aa4a14b900d9e218d844a9338801c`,
시작 작업 트리는 clean이었다. 검증 당시에는 미커밋 구현 변경을 포함했으며,
해당 구현은 이 보고서와 함께 커밋했다. 실행 metadata의 Git dirty 기록은 보존한다.
로컬 fixture·checkpoint·원본 결과는 outputs 아래에 별도로 보관한다. 8장 launcher와 84개 본 실험,
expert GPU 평가, 데모 수집 GPU 전환 및 원격 게시는 실행하지 않았다.

## 변경 범위

- `recurrent.evaluate`: `--sim-backend {physx_cpu,physx_cuda}`, `--num-envs`,
  `--initial-states`. 기본 CPU·1환경 유지, 잘못된 CPU 복수 환경/GPU expert 요청은 오류.
- `recurrent.gpu_eval`: 배치 GT·feature·정규화·추론·action tensor 경로,
  일곱 모델의 중첩 hidden/history 상태 동결, slot별 손상·단계·이벤트·종료 처리.
- `recurrent.initial_states`: 별도 CPU 프로세스에서 기존 reset fixture 수집,
  slot별 형상 재생성 후 공식 `set_state_dict` 복원 및 엄격한 비교.
- 공통 factory는 선택적 `render_backend` 전달만 확장했다. 생략 시 기존 CPU renderer
  기본값을 유지한다. 모델 구조·feature 의미·checkpoint 선택·학습·데이터 분할은 바꾸지 않았다.
- 기존 `ExperimentLogger`와 checkpoint sidecar 출처 연결을 재사용했다.
  검증 명령은 `WANDB_MODE=disabled`로 로컬 metadata/metrics/manifest만 기록했다.

동기식 배치에서 완료 slot은 통계·정책 상태·손상 상태를 고정한다. 물리 simulator는
전체 배치를 계속 step하지만 완료 episode에는 추가 step을 세지 않는다. 마지막 배치는
실제 남은 수로 생성한다. 배치 GT/action을 NumPy로 왕복하지 않는다. 기존 NumPy RNG의
seed와 draw 순서를 보존하며 작은 noise 계수와 이벤트/요약 scalar만 host를 사용한다.

GPU 출력 기본값은 `outputs/recurrent/eval_gpu/`다. 기존 episode key인
condition·level·seed를 유지하며, checkpoint hash·backend·평가 설정·초기 fixture hash를
`.identity.json`으로 검증한다. 배치 크기는 재개 시 바꿀 수 있다. identity 없는 기존
JSONL에는 이어 쓰지 않는다. GPU `mean_infer_ms`는 null이며 batch 추론 시간은 별도
`performance.batches[].batch_infer_ms`다. 이를 배치 크기로 나눠 단일 추론 지연으로 부르지 않는다.

## 설치 API 및 초기 조건

Python 3.11.17, torch 2.14.1+cu130, ManiSkill 3.0.1, SAPIEN 3.0.3,
Gymnasium 1.3.0, NumPy 2.4.6, NVIDIA driver 580.126.09였다.
GPU 실행은 `physx_cuda`, `cuda:0`, `render_backend="none"`, state 관측,
`pd_ee_delta_pose`, horizon 250이다. CPU는 `physx_cpu`, cpu,
`sapien_cpu` renderer다. 두 backend의 simulator 검사는 별도 프로세스에서 실행했다.

설치 task의 `__init__`/`_load_scene`/`BaseEnv._set_episode_rng`를 확인했다.
단일 환경의 기본 reconfiguration frequency는 1, 다중 환경은 0이다. 단일 seed를
다중 환경에 주면 나머지 seed가 파생되므로 같은 초기 문제라고 가정할 수 없다.
GPU에서는 seed 목록과 `options={"reconfigure": True}`를 명시한다. 형상을 먼저
검사하고 공식 `set_state_dict`로 actor/articulation 상태를 복원한다. 이 API가
GPU apply, articulation kinematics update, fetch를 수행하는 것도 설치 소스로 확인했다.

seed 2000–2029의 CPU fixture에는 simulator state dict, peg half sizes,
hole radius/offset, qpos/qvel 및 actor/TCP/finger/hole GT가 있다. GPU B=1/14/16/30에서
위치·관절값 atol=1e-5, rtol=0을 검사했다. quaternion은 q와 -q를 동일하게 취급한
상대 회전각 1e-5 rad 기준이다. 허용오차를 늘리지 않았다. B=16 첫 확인에서 형상·qpos·qvel
오차는 0, TCP 위치 최대 오차는 약 2.2e-6 m였다. 모든 배치 reset에서도 검사한다.
Hole 이동은 한 slot에만 적용하고 apply/fetch 후 다음 GT 및 물리 step에서 유지되는지 검사했다.

Fixture의 별도 collect() 재수집은 8.885초였고, 최초 fixture의 30개 episode 상태와 bitwise 동일했다.
Python import까지 포함한 별도 새 fixture 프로세스는 14.058초였다.
이 일회성 준비 시간은 아래 반복 평가 처리량과 분리한다. 사용 fixture SHA-256:
`3eff3c28c73466679e17a8f12406d459dd02bf3e00f5938fb19ff9cfc15b236d`.

## 재현

기존 checkpoint가 없어 허용된 격리 검증용 checkpoint를 만들었다.
데모 12개 모두 expert 성공으로 수집됐고 1,734 frame이었다. 학습은 각 모델 20 update,
batch 8, seed 0이며 기존 train/validation 분할과 best.pt 선택을 그대로 사용했다.
이 checkpoint의 성공률은 과학적 정책 성능을 검증하는 수치가 아니다.

```bash
WANDB_MODE=disabled .venv/bin/python -m recurrent.collect --num-demos 12 --workers 2 --out outputs/recurrent/gpu_validation/demos.npz
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.train --model mlp --level none --data outputs/recurrent/gpu_validation/demos.npz --steps 20 --batch 8 --seed 0 --out-dir outputs/recurrent/gpu_validation/mlp_none_s0
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.train --model gru --level none --data outputs/recurrent/gpu_validation/demos.npz --steps 20 --batch 8 --seed 0 --out-dir outputs/recurrent/gpu_validation/gru_none_s0
WANDB_MODE=disabled .venv/bin/python -m recurrent.initial_states --out outputs/recurrent/gpu_validation/initial.pt
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.validate_gpu
.venv/bin/python -m recurrent.gpu_report
```

`validate_gpu`는 CPU 정확성 비교를 독립 단일 환경 프로세스 6개로 나누고,
성능 측정은 순차 실행한다. 각 subprocess의 정확한 명령·wall time·NVML peak는
`outputs/recurrent/gpu_validation/*.process.json`에 있다. 다시 실행할 때 기존 episode를
중복 기록하지 않으며, 완료된 성능 실행의 측정값은 유지하고 재개 검증 시간을 별도로 기록한다.
새로운 성능 반복을 측정하려면 새 출력 경로를 사용한다. 이 호스트에는 NVML Python
binding `pynvml`이 설치돼 있으며 검증 runner만 사용한다.

## 회귀·계산 의미 검사

| 검사 | 결과 |
|---|---|
| Recurrent 기존 검사 + CPU tensor stage/reference | 12 passed |
| 공통 logger/artifact 계약, 별도 프로세스 | 35 passed |
| 공통 CPU simulator/GT + render override 계약 | 10 passed |
| tensor 계산·일곱 모델 상태·event/종료/부분 배치/재개 | 7 passed |
| GPU simulator B=1/14/16/30 초기 복원·slot pose·다음 step | 1 passed |

skip은 없었다. feature·동일 RNG 손상·정규화·실제 MLP/GRU checkpoint action,
일곱 모델의 CPU/GPU 계산 및 배치/독립 실행은 atol=rtol=1e-4를 통과했다.
합성 입력으로 retention/transport, shift delay, blackout 시작, 조기 성공·terminated,
horizon과 완료 slot 통계 고정을 확인했다. batch 3→1 부분 배치 및 batch 크기를 바꾼
재개에서 완료 seed 재실행·중복 기록이 없는 것도 검사했다.

Simulator와 계약 검사를 한 프로세스에 합친 첫 실행은 imageio의 동적 모듈 조회와
logger 테스트의 importlib mock이 충돌해 1 error였다. 별도 계약 프로세스에서는
35개가 모두 통과했다. 실패를 skip으로 바꾸거나 logger 구현을 바꾸지 않았다.

실행 명령은 [Recurrent README](../recurrent/README.md#gpu-배치-평가-검증)에 있다.
새 GPU 모듈은 Ruff 검사와 format 검사를 통과했다.

## 실제 비교 및 성능

seed 2000–2029를 각 조건에서 평가했다. MLP는 clean/normal 30개,
GRU는 4개 손상 단계 × normal/hole_shift와 blackout 0/10/20/40 각 30개로
360개다. CPU1/GPU16/GPU30 합계 1,170개이며 모든 실행의 key 집합이 일치하고
중복은 없었다. 상세 episode 차이는 `outputs/recurrent/gpu_validation/comparison.json`에 있다.

| 정책/구성 | Episode | 성공 | acquisition 실패 | retention 실패 | alignment 실패 | Hole 이동 | Blackout |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP CPU1 | 30 | 0 | 29 | 0 | 1 | 0 | 0 |
| MLP GPU16 | 30 | 0 | 28 | 2 | 0 | 0 | 0 |
| MLP GPU30 | 30 | 0 | 28 | 2 | 0 | 0 | 0 |
| GRU CPU1 | 360 | 0 | 298 | 28 | 34 | 11 | 9 |
| GRU GPU16 | 360 | 0 | 314 | 2 | 44 | 14 | 12 |
| GRU GPU30 | 360 | 0 | 314 | 2 | 44 | 14 | 12 |

transport/insertion 실패는 모두 0이다. 모든 episode가 250 step에 종료됐다.
MLP의 CPU/GPU 실패 단계는 2/30개, transport 시점은 1/30개에서 달랐다.
GRU는 두 GPU 구성 각각 CPU와 실패 단계 40/360개, transport 시점 44개,
hole 이동 시점 14개, blackout 시점 12개에서 달랐다. 이벤트가 한쪽에만 발생한
경우도 시점 불일치로 센다. 성공과 종료 step 불일치는 0이다.

아래는 각 30 episode 중 CPU와 다른 수다. GPU16/GPU30의 CPU 대비 불일치
개수가 같아 한 열에 함께 표시한다. 각 seed의 실제 값은 비교 JSON에 남긴다.

| GRU 조건/단계 | 실패 단계 차이 | Transport step 차이 | Hole 이동 step 차이 | Blackout step 차이 |
|---|---:|---:|---:|---:|
| normal/none | 4 | 4 | 0 | 0 |
| normal/low | 4 | 4 | 0 | 0 |
| normal/mid | 2 | 4 | 0 | 0 |
| normal/high | 2 | 2 | 0 | 0 |
| hole_shift/none | 4 | 4 | 4 | 0 |
| hole_shift/low | 4 | 4 | 4 | 0 |
| hole_shift/mid | 2 | 4 | 4 | 0 |
| hole_shift/high | 2 | 2 | 2 | 0 |
| blackout_0/none | 4 | 4 | 0 | 0 |
| blackout_10/none | 4 | 4 | 0 | 4 |
| blackout_20/none | 4 | 4 | 0 | 4 |
| blackout_40/none | 4 | 4 | 0 | 4 |

GPU16 대 GPU30은 MLP의 위 비교 필드가 모두 일치했다. GRU의 성공·실패 단계·종료
step도 모두 일치했지만 transport 시점 18개, hole 이동 시점 5개, blackout 시점 6개가
달랐다. 배치 모델 계산의 허용오차 통과를 closed-loop 궤적의 bitwise 동일성으로
해석하지 않는다.

설치 controller의 `PDEEPoseController.set_action`과 `Kinematics.compute_ik`를 확인했다.
CPU는 목표 pose를 만든 뒤 Pinocchio 반복 IK를 사용하고, GPU는 현재 Jacobian에서
delta joint position을 계산한다. GPU 물리는 PhysX CUDA다. 동일한 초기 문제와
GT 입력에서의 정책 계산이 검증돼도 이후 물리·IK 경로는 다르다. 이 비교는 IK와
PhysX가 결과 차이에 기여한 비중을 별도 ablation으로 분리하지는 않았다.
관측 손상/event 순서 및 집계 정의를 바꿔 차이를 맞추지 않았다.

### 독립 성능 측정

GRU clean/normal 30 episode를 각 구성에서 3회 새 출력으로 실행했다.
정확성 비교 worker가 모두 끝난 뒤 순차 측정하며, 각 반복의 전체 프로세스 wall time을
분모로 episode/s와 CPU 대비 배속을 계산한다. 환경 생성/reset/rollout/저장을 분리한다.
CUDA 추론 및 rollout 경계는 synchronize한다. NVML은 장치 전체 사용량을 0.1초 간격으로
표본 측정하므로 Torch allocator 밖의 PhysX 메모리와 idle baseline도 포함한다.
305.9 env-step/s probe 수치는 사용하지 않는다.

| Backend | Process s | Create s | Reset s | Rollout s | Save s | Episode/s | Rollout env-step/s | Peak VRAM GiB | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| physx_cpu/1 | 147.98 | 1.74 | 8.12 | 132.81 | 0.0030 | 0.203 | 56.5 | 0.52 | 1.00× |
| physx_cuda/16 | 56.90 | 4.70 | 4.66 | 40.74 | 0.0006 | 0.527 | 184.1 | 4.05 | 2.60× |
| physx_cuda/30 | 32.67 | 2.76 | 2.73 | 20.98 | 0.0005 | 0.918 | 357.5 | 2.98 | 4.53× |

표는 각 3회 평균이다. rollout env-step/s는 유효 step/rollout 시간이다.
전체 프로세스 시간을 분모로 한 env-step/s는 CPU 50.7, GPU16 131.8, GPU30 229.6이다.
모든 반복은 7,500 유효 step이었고 물리 step 수도 같았다. 조기 종료나 완료 결과
재사용으로 처리량을 높이지 않았다. 평균 evaluation 함수 시간은 CPU 142.94초,
GPU16 51.39초, GPU30 27.30초이며 프로세스 시간에는 import/모델 로드 등도 포함된다.
Phase 합계와 전체 시간의 나머지는 초기화·정책 로드·GT 검증·logger·종료 등이다.

| 구성 | 1회 전체 초 | 2회 전체 초 | 3회 전체 초 | 평균 ± 표본 표준편차 초 |
|---|---:|---:|---:|---:|
| CPU1 | 147.265 | 147.575 | 149.099 | 147.980 ± 0.982 |
| GPU16 | 56.486 | 56.662 | 57.559 | 56.902 ± 0.576 |
| GPU30 | 32.510 | 32.402 | 33.099 | 32.671 ± 0.375 |

GPU batch 추론 시간은 평균 GPU16 0.988 ms, GPU30 1.043 ms였다.
이 값은 feature 계산·정규화·정책 추론·action 복원을 포함한 **한 배치 호출** 시간이다.
단일 episode 추론 지연이나 배치 크기로 나눈 단일 추론 지연으로 해석하지 않는다.
GPU16은 16+14 환경 두 배치 및 환경 재생성을 포함하며 GPU30은 한 배치다.
NVML 표의 값은 반복별 peak의 평균이고 최대 표본 peak는 GPU16 약 4.05 GiB,
GPU30 약 2.98 GiB였다. 메모리가 배치 크기에 단순 비례한다고 해석하지 않는다.

`comparison.json`은 조건별 집계, CPU/GPU 및 GPU16/GPU30의 모든 episode 차이,
성능 반복별 값과 평균을 포함한다. 원본 JSONL·identity·summary·process 기록은
`outputs/recurrent/gpu_validation/`에 있으며 테스트 출력은 그 아래 `test_logs/`에 있다.
성능 시작 직전 다른 정확성 worker와 GPU 검사는 모두 끝냈다. 앞서 정확성 실행과
겹쳐 시작된 CPU timing 시도는 중단해 `discarded_contended_timing/`으로 분리했고
집계에 포함하지 않았다. 최종 9회는 모두 새 30 episode 실행임을 집계 도구가 검사했다.

## 판정과 남은 차이

요청 범위에서 초기 문제 구성, 계산 의미, slot별 종료·이벤트·상태·재개 검증을 통과했고,
RTX 3090 실제 평가의 전체 처리량은 CPU 대비 GPU16 2.60배, GPU30 4.53배로 빨라졌다.
이 구현과 검증용 checkpoint 범위에서는 GPU 배치 평가로 평가 병목을 줄였다고 판단한다.

다만 smoke checkpoint의 성공은 모두 0이다. 강한 정책의 성공률·강건성에 대한 과학적
결론은 이 결과로 내리지 않는다. CPU/GPU 실패 단계 및 event 시점 차이와 GPU 배치 크기별
일부 시점 차이는 남아 있으며 위 표와 episode 비교 JSON에 명시했다. CPU와 GPU 결과를
동일한 물리 rollout으로 혼합하거나 GPU 전환을 정책 성능 개선으로 해석하지 않는다.
8장 실행·84개 본 실험·전체 학습·원격 게시와 expert/데모 GPU 전환은 수행하지 않았다.

