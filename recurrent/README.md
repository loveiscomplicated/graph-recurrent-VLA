# Recurrent 팀 prototype: 관측 손상 조건에서 controller 7종 비교

Recurrent 팀 코드입니다. 공용 simulator 코드(`envs/`)와 다른 팀 코드는 수정하지 않았습니다.

**완료된 본 실험:** [84개 학습 실행·11,400 episode 재평가 공유 리포트](../docs/reports/recurrent_main_20261023/README.md).
PNG 그림 12개와 CSV 표 8개, 원본 HF archive와 검증 hash를 포함합니다.
학습 seed별 편차, clean 교차평가, blackout 실제 발생 수를 함께 확인할 수 있습니다.

## 무엇을 확인하는가

Graph embedding이 들어오기 전 단계라서, 입력은 simulator GT state로 만든 38차원 vector를 씁니다. 이 입력으로 아래 세 질문에 답하는 실험을 한 번에 돌립니다.

1. 과거를 보는 것이 closed-loop 제어 성공률을 높이는가 (모든 모델 vs `mlp`)
2. 최근 몇 step을 이어 보는 것과 recurrent하게 기억하는 것은 무엇이 다른가 (recurrent 계열 vs `stack_mlp`)
3. 어떤 모델이 어떤 관측 조건에서 가장 좋고, 어디서 실패하는가

관측이 깨끗하면 현재 상태만 봐도 action을 정할 수 있어 기억이 할 일이 거의 없습니다. 그래서 vision이 담당할 peg와 hole 관측에만 손상(관측 끊김, 위치와 방향 noise, 지연)을 넣고, 손상 강도를 `none`, `low`, `mid`, `high` 4단계로 나눠 학습과 평가에 같은 강도로 적용합니다.

## GPU 서버에서 바로 실행 (Vessl, Vast.ai 공통)

```bash
git clone https://github.com/loveiscomplicated/graph-recurrent-VLA.git
cd graph-recurrent-VLA
git checkout recurrent
bash recurrent/setup_server.sh
tmux new -s recurrent
sh recurrent/run_all.sh smoke
sh recurrent/run_all.sh
```

`setup_server.sh`가 하는 일:

1. `.venv`가 없으면 Python 3.11 venv를 만듭니다 (`python3.11`이 없으면 `uv`로 받습니다).
2. `requirements.txt`, `requirements-tracking.txt`, `recurrent/requirements.txt`를 설치합니다.
3. GPU가 있는데 torch가 CUDA를 못 쓰면, `nvidia-smi`의 CUDA 버전에 맞는 torch를 다시 설치합니다.
4. CUDA가 보이는 상태에서 CPU 시뮬레이터가 뜨는 Vulkan 드라이버를 찾아 `outputs/recurrent/env.sh`에 적습니다. 설치된 lavapipe를 먼저 시도하고, 없거나 실패하면 NVIDIA 드라이버를 시도한 뒤 필요하면 `mesa-vulkan-drivers`를 설치합니다. 상태 관측만 쓰므로 lavapipe로도 충분합니다.
5. CPU 시뮬레이터·tensor 평가·GPU 시뮬레이터 테스트를 별도 프로세스에서 실행합니다. 테스트용 checkpoint는 임시 생성하고, GPU 복원 검사용 CPU 초기 상태도 별도 프로세스에서 임시 수집하므로 기존 검증 산출물이 필요하지 않습니다. 테스트 오류·실패는 설정 실패로 처리합니다. CUDA 등 실행 조건이 없는 검사는 사유와 함께 건너뛰며, 나머지 검사가 통과하면 `setup PASS`를 출력합니다. 건너뛴 backend/device는 검증된 것이 아닙니다.

`run_all.sh smoke`는 작은 규모(데모 12개, 학습 20 step, 평가 1 episode)로 전체 과정이 오류 없이 도는지만 봅니다. 성공률은 의미가 없습니다. `run_all.sh`(main)가 본 실험입니다. 중간에 끊겨도 같은 명령을 다시 실행하면 끝난 학습과 평가는 건너뛰고 이어서 합니다.

W&B 인증 파일(`/root/.secrets/wandb.env`)이 있으면 공용 `ExperimentLogger`가 online으로 기록하고, 없으면 `experiments/` 아래에 offline으로만 남깁니다. 자세한 규칙은 [experiment tracking](../docs/experiment_tracking.md)을 따릅니다. HF 게시는 자동으로 하지 않습니다.

## 실행 순서와 산출물

| 단계 | 명령 | 산출물 |
| --- | --- | --- |
| 데모 수집 | `python -m recurrent.collect` | `outputs/recurrent/data/expert_demos.npz` |
| 파라미터 수 확인 | `python -m recurrent.count_params` | 화면 출력 |
| 추론 지연 측정 | `python -m recurrent.latency` | `outputs/recurrent/latency.json` |
| 학습과 평가 | `python -m recurrent.run_experiments` | `outputs/recurrent/exp/main/{runs,eval,logs}` |
| 리포트 | `python -m recurrent.report` | `outputs/recurrent/exp/main/report/` |

`run_experiments`는 할당된 학습을 모두 끝낸 뒤 평가합니다. 일반 실행의 평가는 기존
CPU·단일 환경이 기본값이며 worker 수는 `--train-workers`, `--eval-workers`로
바꿉니다. `--train-only`와 `--eval-only`는 상호 배타적입니다. 평가 전용 모드는
학습 결과가 없거나 미완료면 실패하며 학습을 시작하지 않습니다. 단일 shard의 일반
실행은 완료 후 검증된 결과로 리포트를 생성합니다. 설정 일부를 바꿔 새 실험을 할 때는
별도 config와 `out_dir`를 사용합니다.

### GPU 여러 장에 본 실험 나누기

서버 설정은 기존 `bash recurrent/setup_server.sh`로 한 번 실행합니다. 그다음
`bash recurrent/smoke_multi_gpu.sh`로 여러 GPU에서 학습 분산과 GPU 배치 평가를 확인합니다.
smoke는 새 데모 12개와 CPU 초기 상태 fixture를 만들고, 7개 모델 × 2개 손상 단계의
14개 학습을 GPU별 shard로 나눠 실행합니다. CPU 수집·초기 상태 저장도 NVIDIA Vulkan 초기화에 필요한 첫 번째 선택 GPU를 보이게 유지하며, 물리 연산은 `physx_cpu`를 사용합니다. 학습이 모두 끝나면 같은 `gru_none_s0`
checkpoint와 seed로 GPU마다 평가 프로세스 하나를 동시에 실행합니다. 기본값은
GPU당 30개 환경·30 episode이며 8장에서는 총 240 episode입니다.

`EVAL_NUM_ENVS=16 bash recurrent/smoke_multi_gpu.sh`처럼 환경 수와 episode 수를
함께 줄일 수 있습니다. 사용할 GPU는 `CUDA_VISIBLE_DEVICES=0,2,4,6`처럼 선택하며
최소 2장이 필요합니다. 새 출력 디렉터리를 매번 생성하고 GPU별 JSONL·로그를 분리합니다.
모든 학습의 CUDA 사용, 평가의 실제 `physx_cuda`/`render=none`, episode key와 중복,
새 배치 실행 완료를 검사한 뒤 `Multi-GPU smoke PASS`를 출력합니다. 결과 경로와
`smoke_summary.json`에 GPU 배정과 평가 처리량이 남습니다. W&B는 기본 `disabled`이며
기존 공용 logger의 로컬 기록을 유지합니다. 설정된 `WANDB_MODE`는 그대로 따릅니다.
이 smoke는 normal/none 조건의 GRU GPU 평가를 확인하며 expert 평가와 비교 리포트는
생성하지 않습니다. 서버 설정을 다시 실행하거나 84개 본 실험을 시작하지 않습니다.

본 실험은 다음 명령으로 실행합니다. **7개 모델 × 4개 손상 단계 × 3개 seed =
84개 CUDA 학습**이며, 학습 3000 step·데이터 split·`best.pt` 선택·평가 조건과 seed는
기존 설정을 유지합니다.

```bash
bash recurrent/run_main_multi_gpu.sh
# GPU 선택과 평가 배치 크기 변경 (episode 수는 조건마다 30개로 유지)
CUDA_VISIBLE_DEVICES=0,2,4,6 EVAL_NUM_ENVS=16 bash recurrent/run_main_multi_gpu.sh
# 다른 출력 위치나 실험 범위는 config에 data와 out_dir를 명시
MAIN_CONFIG=/path/to/main.yaml bash recurrent/run_main_multi_gpu.sh
```

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| `MAIN_CONFIG` | `recurrent/configs/experiment.yaml` | 적용 config; `out_dir` 아래 로그·fixture·latency·결과·리포트를 저장 |
| `CUDA_VISIBLE_DEVICES` | Torch가 보는 모든 GPU | 사용할 GPU ID/UUID 목록, 최소 2개 |
| `EVAL_NUM_ENVS` | `30` | GPU당 동시 평가 환경 수; 총 episode 수·결과 identity와 무관 |
| `EVAL_WORKERS` | `1` | 본 실행에서는 GPU당 평가 프로세스 하나만 허용 |

실행 순서는 다음과 같습니다.

1. `data` 경로의 expert 데모를 검증하거나 없을 때 100개 수집합니다. 기존 데이터의
   episode 수, 길이, shape, 유한값, seed, 수집 조건·환경·패키지 metadata를 확인합니다.
   불일치나 출처 누락은 자동 재수집하지 않고 중단합니다.
2. 평가 seed 2000–2029의 CPU reset fixture를 별도 프로세스에서 한 번 만듭니다.
   기존 fixture는 seed·환경·패키지 버전·tensor를 검증해 재사용합니다. CPU 수집과
   fixture 생성, expert 평가에도 선택 GPU를 보이게 유지해 NVIDIA Vulkan을 지원합니다.
3. 같은 작업 목록을 GPU 수로 분할해 모든 shard를 `--train-only`로 실행합니다.
   **84개 학습이 모두 성공하고 checkpoint/history 검증까지 끝나야 평가를 시작합니다.**
4. 같은 shard를 `--eval-only --sim-backend physx_cuda --num-envs N
   --initial-states ...`로 실행합니다. GPU별 평가 프로세스 하나가 정책들을 순차
   평가하며, 배치가 작아져도 조건마다 30 episode를 유지합니다. shard 0만 expert를
   `physx_cpu`·단일 환경으로 평가합니다. 정책·expert 모두 같은 CPU fixture를 씁니다.
5. 작업별 정확한 condition·corruption level·episode seed 집합과 identity를 검증합니다.
   기본 설정의 기대치는 **학습 84개, 정책 11,340 episode, expert 60 episode**입니다.
   통과한 결과만 리포트에 포함하고 최종 `Main multi-GPU run PASS`를 출력합니다.

정책의 GPU physics와 expert의 CPU physics는 다릅니다. 실제 학습 device,
평가 backend, 초기 상태 출처를 결과·리포트에 기록합니다. Offline–closed-loop 그림은
마지막 학습 step 대신 평가에 사용한 `best.pt`의 validation 오차를 사용합니다.
일반 실행의 기본 CPU 경로와 GPU 평가의 수치가 같다고 가정하지 않습니다.

### 중단 후 재실행과 불일치 복구

복원한 학습 결과를 **다른 패키지 환경에서 평가만** 할 때는
`python -m recurrent.reevaluate --source-root /workspace/recovered-main
--out-dir outputs/recurrent/reevaluation_main_20261023`으로 검증과 명령 계획만
생성할 수 있습니다. `--execute`를 추가해야 실제 평가를 시작합니다.
원본 학습 완료·데이터·설정 검증을 유지하며 새 경로에 현재 환경의 평가 identity를
기록합니다. 학습이나 원격 게시는 실행하지 않습니다. 정확한 실행·재개 명령과
서버 smoke 근거는 [복원 checkpoint 재평가](../docs/recurrent_restored_evaluation.md)를 따릅니다.

같은 명령을 다시 실행하면 완료된 결과도 identity를 검사한 뒤 재사용합니다.
`out_dir/identity.json`, `applied_config.json`에 적용 설정과 데이터 SHA-256을 기록하며,
각 학습의 identity는 모델·손상 설정, seed, 실제 split episode seed, Adam 설정,
목표 step, 데이터 hash, validation 주기를 포함합니다. GPU 수·worker 수·평가 배치
크기는 identity에서 제외하므로 실행 자원은 변경할 수 있습니다.

- 학습 완료에는 `best.pt`, `last.pt`, `history.json`의 identity·step·validation 일관성이
  필요합니다. 같은 설정의 미완료 산출물은 `runs/<name>/incomplete/<id>/`로 보존하고
  처음부터 다시 학습합니다. optimizer 상태부터 이어 학습하지 않습니다.
- 평가 identity에는 checkpoint SHA-256, backend, 조건·손상·seed 범위, fixture hash,
  패키지 버전이 들어갑니다. 완료 metadata와 summary도 검증합니다. JSONL 끝의
  불완전한 마지막 기록만 `.partial-<id>.bak` 원본 백업 후 제거하고 재개합니다.
  유효한 마지막 기록의 개행 누락은 보완합니다. 중간 손상·중복·예상 밖 episode는 오류입니다.
- 설정·데이터·fixture가 다르거나 기존 결과의 identity를 확인할 수 없으면 파일을
  보존하고 중단합니다. 원래 config·데이터·패키지 환경을 복구하거나, 기존 폴더를
  보관한 채 새 `data`/`out_dir`를 지정한 config로 실행합니다. identity 파일만 지워서
  강제로 재사용하지 마세요. 이전 버전의 출처 없는 결과를 자동 변환하지 않습니다.
- Linux `flock`이 같은 출력 경로의 본 실행을 차단합니다. 오류·SIGINT·SIGTERM 시
  shard 프로세스 그룹과 자식 writer를 종료하고 회수합니다. 잠금 FD를 writer까지
  상속하므로 launcher가 SIGKILL되어도 살아 있는 writer가 잠금을 유지합니다.
  잠금 파일은 삭제하지 마세요. 남은 프로세스가 종료되어야 다시 실행할 수 있습니다.

필요하면 `run_experiments`를 직접 두 단계로 호출할 수 있습니다. 모든 GPU의 학습
shard가 성공한 뒤 평가 shard를 시작해야 합니다. 본 launcher와 동시에 수동 shard를
실행하지 마세요. 예를 들어 GPU 4장에서는 shard당 21개를 배정합니다.

```bash
# 선택 GPU마다 index=0..3과 CUDA_VISIBLE_DEVICES를 바꿔 실행하고 모두 완료될 때까지 대기
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.run_experiments --config experiment.yaml --shard-count 4 --shard-index 0 --train-workers 1 --train-only --no-report
# 공통 CPU fixture는 별도 프로세스에서 준비 (기존 파일은 검증 후 재사용)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.initial_states --out outputs/recurrent/exp/main/initial_states.pt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.run_experiments --config experiment.yaml --shard-count 4 --shard-index 0 --eval-workers 1 --eval-only --sim-backend physx_cuda --num-envs 30 --initial-states outputs/recurrent/exp/main/initial_states.pt --no-report
# 모든 평가 shard 완료 후 한 번 실행; backend와 기본 fixture 경로는 결과에서 확인
.venv/bin/python -m recurrent.report --config experiment.yaml
```

리포트 폴더에는 `report.md`와 그림, 표 파일이 생깁니다.

| 파일 | 내용 |
| --- | --- |
| `success_by_level.png`, `success_table.png` | 손상 단계별 성공률 (seed 평균, 표준편차) |
| `success_table_clean_trained.png` | 손상 없이 학습한 모델을 손상 환경에서 평가한 성공률 |
| `paired_vs_mlp.png` | 질문 1: 각 모델 성공률 − mlp 성공률, episode 단위 짝지은 비교와 95% 신뢰구간 |
| `paired_vs_stack_mlp.png` | 질문 2: 각 모델 성공률 − stack_mlp 성공률 |
| `blackout_stress.png` | 질문 2: peg를 든 직후 관측을 0, 10, 20, 40 step 끊었을 때의 성공률 |
| `robustness_drop.png`, `failure_stages.png` | 질문 3: 손상에 따른 성공률 하락, 실패 단계 비율 |
| `learning_curves.png`, `offline_vs_closed_loop.png`, `cost.png` | 학습 곡선, offline 오차와 성공률의 관계, 파라미터와 학습 시간과 추론 지연 |
| `summary.csv`, `episodes.csv`, `paired.csv` | 그림에 쓴 수치 전부 |

## 설계

| 항목 | 설정 | 이유 |
| --- | --- | --- |
| 입력 | GT 36차원 + 관측 신선도 2차원(`obs_fresh`, `obs_age`) | 실제 tracker도 지금 추적 중인지는 알기 때문 |
| 출력 | `pd_ee_delta_pose` 7차원 (이동 3, 회전 3, gripper 1) | Graph 팀과 같은 action 형식 |
| 손실 | 이동과 회전은 z-score MSE, gripper는 BCE | 데모 gripper 명령이 열림과 닫힘 두 값뿐이기 때문 |
| 모델 | mlp, stack_mlp(최근 8 step), gru, lstm, cfc, rkn, mamba | 모두 약 14만~15.5만 파라미터로 맞춤 |
| 학습 | 데모 episode 8개씩 통째로 넣는 BC, 3000 step, Adam 1e-3 | 모든 모델이 같은 batch를 보게 하기 위함 |
| checkpoint | 검증 손실이 가장 낮은 `best.pt`로 평가 | Graph 팀과 같은 선택 기준 |
| 평가 | seed 2000~2029의 새 배치 30개, horizon 250 | Graph v4 fresh 평가와 같은 배치와 시간 제한 |
| 성공 판정 | rollout 중 `info['success']`가 한 번이라도 참 | Graph v2 이후 평가와 같음 |
| 실패 단계 | acquisition, retention, transport, alignment, insertion | Graph v2 `stage_summary`와 같은 기준값 |
| hole 이동 조건 | peg를 든 뒤 10 step 지나서 hole을 옆으로 2cm 이동 | 모든 모델이 같은 작업 단계에서 변화를 겪게 하기 위함 |

정규화 기준(평균, 표준편차)은 깨끗한 학습 데이터에서 계산합니다. 깨끗한 데이터에서 아예 변하지 않는 값(예: 상자가 기울지 않아 항상 0인 hole 방향 성분)은 정보가 없고 손상 noise만 실어 나르므로 입력에서 0으로 고정합니다. 나머지는 표준편차 최소값 0.01을 둡니다.

## 바꾸기 쉬운 곳

| 바꾸고 싶은 것 | 위치 |
| --- | --- |
| 모델 크기 | `configs/models.yaml` 또는 `experiment.yaml`의 `model_overrides: ["gru.hidden=192"]` |
| 손상 강도 | `configs/corruption.yaml` |
| 실험 범위 (모델, 단계, seed) | `configs/experiment.yaml`의 `models`, `levels`, `seeds` |
| 평가 조건 | `experiment.yaml`의 `eval.conditions`, `eval.stress`, `cross_eval_clean` |
| 데모 수 | `experiment.yaml`의 `train.num_demos` |
| 일부만 실행 | `python -m recurrent.run_experiments --models gru cfc --levels mid high --seeds 0` |

## ManiSkill 공식 데모로 같은 실험

```bash
sh recurrent/run_all.sh official
```

공식 데모를 내려받고, 공식 replay 도구로 `pd_ee_delta_pose`로 변환한 뒤, 매 step 상태를 복원해 같은 형식으로 저장합니다. 변환 설정은 Graph 팀 `graph/prototype/replay_demos.py`와 같습니다. ManiSkill 3.0.1 replay가 첫 상태를 잘못 저장하는 문제는 원본 H5의 첫 상태로 바로잡습니다. Graph 팀이 이미 변환한 파일이 있으면 `python -m recurrent.official_demos --converted <변환된 h5> --source <원본 h5>`로 같은 파일을 쓸 수 있습니다. Graph v2~v4에서는 공식 데모로 학습한 BC가 full-task 성공을 내지 못했으므로, 이 실험은 직접 만든 expert 데모의 결과와 비교하는 용도입니다.

## 기존 CPU 실행 확인 기록

- Linux CPU(2코어), Python 3.11, ManiSkill 3.0.1, SAPIEN 3.0.3, torch 2.14.1, Vulkan은 lavapipe에서 `setup_server.sh`, 단위 테스트 9개, `run_all.sh smoke` 전체를 실행했습니다.
- 공식 데모 변환 경로는 직접 녹화한 같은 형식의 H5로 확인했습니다(공식 replay 후 상태 복원 오차 1.2e-7). 실제 공식 데모 다운로드는 이 환경에서 막혀 있어 실행하지 못했습니다.
- CUDA 학습, NVIDIA Vulkan, Vessl과 Vast.ai 환경, W&B online 기록은 실행하지 않았습니다.

## GPU 배치 평가 검증

학습된 정책만 `--sim-backend physx_cuda --num-envs N`으로 평가할 수 있습니다.
기본값은 기존 CPU·단일 환경이며, expert GPU 평가와 CPU 복수 환경은 오류입니다.
GPU 경로는 horizon 250, state 관측, `pd_ee_delta_pose`, 렌더링 비활성화를
사용하며 CUDA 실패를 CPU로 대체하지 않습니다. 모델·학습·정규화는 그대로입니다.

먼저 **별도 CPU 프로세스**에서 기존 reset 결과를 저장합니다. 단일 환경은
reset마다 형상을 재생성하지만 다중 환경의 기본값은 그렇지 않으므로 GPU는
slot별 seed와 `reconfigure=True`로 형상을 생성하고 simulator state dict를 복원합니다.
매 reset에서 형상, hole offset, 관절·actor·TCP·hole GT를 atol 1e-5로 검사합니다.
Quaternion은 부호 동치를 적용합니다. 아래 명령은 각각 별도 프로세스입니다.

```bash
.venv/bin/python -m recurrent.initial_states --out outputs/recurrent/gpu_validation/initial.pt
.venv/bin/python -m recurrent.evaluate --ckpt outputs/recurrent/gpu_validation/gru_none_s0/best.pt --sim-backend physx_cuda --num-envs 16 --initial-states outputs/recurrent/gpu_validation/initial.pt --conditions normal
```

GPU 기본 출력은 `outputs/recurrent/eval_gpu/`입니다. 동기식 배치이며 완료된
slot의 정책·손상·통계 상태는 고정하고 다음 배치까지 slot을 재사용하지 않습니다.
마지막 배치는 남은 episode 수로 생성합니다. GT·관측 기하·정규화·action은
GPU tensor로 유지합니다. 기존 episode별 NumPy RNG draw 순서를 보존하기 위해
작은 noise 계수와 이벤트 정보만 CPU에서 전달합니다.

완료 키는 condition·level·seed입니다. `.identity.json`이 checkpoint 내용,
backend, 조건·손상 설정, horizon, seed 범위, fixture hash를 검증합니다.
배치 크기는 identity에 포함하지 않아 재개 시 변경할 수 있습니다. identity 없는
기존 JSONL에는 이어 쓰지 않고 새 출력 경로를 요구합니다. 평가 writer에도 파일
잠금을 적용해 같은 JSONL로의 동시 쓰기를 차단합니다.

GPU episode의 `mean_infer_ms`와 해당 summary는 null이며, 배치 추론 시간은
summary의 `performance.batches[].batch_infer_ms`에 따로 기록합니다. 이를
배치 크기로 나눈 값을 단일 추론 지연으로 해석하지 마세요. `active_env_steps`는
완료 전 유효 step 합계, `physical_env_steps`는 완료 slot을 포함한 물리 step 수입니다.
Torch peak VRAM은 PhysX의 메모리를 포함하지 않습니다. 검증 runner는 별도로
NVML 장치 전체 peak VRAM을 0.1초 간격으로 측정합니다.

`python -m recurrent.validate_gpu`는 준비된 검증용 MLP·GRU checkpoint와 fixture를
사용해 요청된 CPU/GPU 16/GPU 30 비교 및 GRU clean/normal 3회 측정을 실행합니다.
CPU 정확성 작업은 독립 프로세스 6개로 실행하며 성능 측정은 순차 실행합니다.
원격 게시와 84개 본 실험은 실행하지 않습니다. 12개 데모·20 update smoke checkpoint의
성공률은 과학적 성능 비교가 아니라 구현 검증용입니다.

검사도 CPU/GPU 프로세스를 분리합니다.

```bash
.venv/bin/python -m unittest recurrent.tests.test_recurrent -v
.venv/bin/python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -v
.venv/bin/python -m unittest tests.test_maniskill_env tests.test_graph_gt_nodes -v
.venv/bin/python -m unittest recurrent.tests.test_gpu_eval -v
.venv/bin/python -m unittest recurrent.tests.test_gpu_sim -v
```

실측 비교 보고서: [RTX 3090 GPU 평가 검증](../docs/recurrent_gpu_evaluation.md).

## 본 실행 launcher 검증 (simulator·GPU 없이)

```bash
bash -n recurrent/run_main_multi_gpu.sh
.venv/bin/python -m unittest recurrent.tests.test_main_execution -v
# 공통 logger/artifact 계약은 별도 Python 프로세스에서 검사
.venv/bin/python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -v
```

가짜 자식 프로세스와 임시 데이터·checkpoint로 8-GPU의 84개 작업 분할, 학습/평가
barrier, CUDA 정책/CPU expert 인자와 GPU 가시성, identity 거부·재사용,
미완료 학습 보존·재실행, JSONL 복구·중복 거부, 배치 크기 변경 후 재개,
중복 실행 잠금·종료 신호·자손 회수, `best.pt` 기준 리포트와 정확한 집계를 확인합니다.
이 테스트는 본 실험이나 GPU smoke를 실행하지 않습니다. 이전 launcher 변경 검증에서는 실제
CPU/GPU simulator, CUDA 학습·평가, NVIDIA Vulkan, 원격 게시를 실행하지 않았습니다.

## RKN NaN 수정과 작은 GPU smoke

RKN의 관측 분산 갱신은 `K=P/(P+R)`에 대해 `(1-K)*P` 대신 동치식 `K*R`을
사용합니다. float32에서 `K`가 1로 반올림되어 분산이 0이 되고 `log(0)`이 모델
출력을 깨뜨리던 문제를 방지합니다. 구조·파라미터·checkpoint 형식은 유지하므로
기존 checkpoint를 그대로 읽습니다. 향후 학습도 같은 안정한 forward 계산을 사용합니다.

CPU/GPU 평가는 simulator 관측·정규화 입력·모델 출력·recurrent state·action의
유한값을 확인하며 RKN 분산은 양수인지도 검사합니다. GPU는 활성 slot만 검사합니다.
잘못된 값은 simulator step 전에 checkpoint, condition, level, episode seed와
1부터 시작하는 step, tensor 이름을 포함한 오류로 중단합니다. JSONL에도 NaN/Inf를
기록하지 않습니다. 기존 손상된 행은 자동 수정하지 않고 파일·행·값·seed를 보고합니다.

RKN 평가 identity에는 `settings.rkn_covariance_update="stable_v1"`이 추가됩니다.
이 필드가 없는 이전 RKN 결과에 이어 쓰기는 거부합니다. 이전 JSONL·summary·완료
파일과 identity는 함께 보존하고 새 평가 출력 경로를 사용하세요. 다른 모델과 expert의
identity는 유지됩니다. 이 수정이 기존 RKN 학습을 재실행하거나 결과를 자동 변환하지는
않습니다. 유한값 검사의 실행 시간도 평가의 추론 시간에 포함됩니다.

다음은 **본 평가를 재개하지 않고** 기존 checkpoint로 작은 smoke만 실행합니다.
설치된 환경의 Python을 `PY`로 지정하고, CPU fixture 생성과 GPU 평가는 별도
프로세스에서 실행합니다. fixture는 현재 패키지 환경에서 새로 생성하므로 원본
실험의 초기 상태·수치와 같다고 간주하지 않습니다. GPU batch 1·2에서 각각
2 seed × 2 condition × 2 level, 합계 16 episode를 최대 250 step 실행합니다.

```bash
PY=.venv/bin/python
"$PY" -m unittest recurrent.tests.test_numerics -v
mkdir -p outputs/recurrent/troubleshooting
SMOKE_DIR=$(mktemp -d outputs/recurrent/troubleshooting/rkn-XXXXXX)
"$PY" -m recurrent.initial_states --out "$SMOKE_DIR/initial_states.pt" \
  --episodes 2 --start-seed 2010 --horizon 250
for BATCH_SIZE in 1 2; do
  WANDB_MODE=disabled EXPERIMENTS_DIR="$SMOKE_DIR/tracking" \
    "$PY" -m recurrent.evaluate \
    --ckpt outputs/recurrent/exp/main/runs/rkn_none_s0/best.pt \
    --conditions normal hole_shift --levels none high \
    --episodes 2 --start-seed 2010 --horizon 250 \
    --sim-backend physx_cuda --num-envs "$BATCH_SIZE" \
    --initial-states "$SMOKE_DIR/initial_states.pt" \
    --out "$SMOKE_DIR/smoke_batch${BATCH_SIZE}.jsonl" || exit 1
done
```

두 프로세스 모두 JSONL의 유효성·episode 집합·summary·완료 metadata 저장까지
성공해야 smoke 통과입니다. 성공률 자체는 통과 기준이 아닙니다. 모델 수치 회귀와
기존 평가·완료 검증 테스트는 다음처럼 별도 프로세스에서 실행합니다.

```bash
"$PY" -m unittest recurrent.tests.test_recurrent -v
"$PY" -m unittest recurrent.tests.test_gpu_eval -v
"$PY" -m unittest recurrent.tests.test_main_execution -v
```

2026-10-23 검증: GTX 1650 4GB, Python 3.12.15, torch 2.14.1+cu132,
NumPy 2.5.4와 위 simulator 버전에서 관련 테스트 52개가 통과했습니다. 수정 전의
결정적 수치 회귀는 CPU·CUDA 모두 실패했고 수정 후 통과했습니다. 저장된 RKN
12개 checkpoint는 CPU·CUDA 각각 고정 입력 500 step 동안 유한한 출력·state와
양수 분산을 유지했습니다. 실제 `physx_cuda`/`render=none` smoke 16 episode도
각각 250 step 후 결과 검증을 통과했고, 원본 산출물·기존 백업 관련 파일 1,082개의
SHA-256이 일치했습니다. smoke의 과제 성공은 0/16이며 제어 성능 개선을 입증하지
않습니다. 원본 RTX 3090 8장 환경과 본 평가 재개·재학습·원격 게시는 실행하지
않았습니다. 실행 기록은 `outputs/recurrent/troubleshooting/rkn_stable_20261023T111605Z/`
아래 `verification.json`, 테스트 로그와 smoke JSONL에 보존했습니다.
