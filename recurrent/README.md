# Recurrent 팀 prototype: 관측 손상 조건에서 controller 7종 비교

Recurrent 팀 코드입니다. 공용 simulator 코드(`envs/`)와 다른 팀 코드는 수정하지 않았습니다.

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

`run_experiments`는 학습이 끝난 checkpoint부터 바로 평가를 시작합니다. 학습은 GPU에서 여러 개를 동시에 돌리고(`train_workers`), 평가 rollout은 CPU 프로세스 여러 개로 나눠 돌립니다(`eval_workers`). 둘 다 기본값은 CPU 코어 수에서 자동으로 정하며 `--train-workers`, `--eval-workers`로 바꿀 수 있습니다. 끝나면 리포트를 자동으로 만듭니다.

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

7개 모델 × 4개 손상 단계 × 3개 seed = **84개 학습·평가 조합**입니다. 기존 본 실험 명령은 저장소 루트의 `bash recurrent/run_main_multi_gpu.sh`이며, 이 launcher의 평가는 아직 CPU입니다. 위 GPU 평가 smoke의 변경은 본 실험 launcher에 자동 적용되지 않습니다. 이 스크립트는 데모를 한 번 수집하고(기존 파일은 재사용), GPU마다 shard 하나를 실행하며, 84개 학습·평가와 expert 평가가 모두 끝난 뒤 리포트를 만듭니다. 중단 후 같은 명령을 다시 실행하면 완료된 작업을 건너뜁니다. 평가 CPU 프로세스 수는 기본 GPU당 1개이며 `EVAL_WORKERS=2 bash recurrent/run_main_multi_gpu.sh`처럼 조절할 수 있습니다.

직접 shard별 명령을 실행하려면 아래 예시를 사용합니다. GPU 4장에서는 21개씩 배정됩니다.

```bash
.venv/bin/python -m recurrent.collect --num-demos 100
.venv/bin/python -m recurrent.count_params
.venv/bin/python -m recurrent.latency
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.run_experiments --shard-count 4 --shard-index 0 --train-workers 1 --eval-workers 2
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m recurrent.run_experiments --shard-count 4 --shard-index 1 --train-workers 1 --eval-workers 2
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m recurrent.run_experiments --shard-count 4 --shard-index 2 --train-workers 1 --eval-workers 2
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m recurrent.run_experiments --shard-count 4 --shard-index 3 --train-workers 1 --eval-workers 2
```

네 명령은 각각 다른 tmux 창이나 터미널에서 동시에 실행합니다. GPU가 `N`장이면 `--shard-count N`, `--shard-index 0`부터 `N-1`까지 사용합니다. 같은 설정과 데모 파일, 공유 출력 디렉터리(`outputs/recurrent/exp/main`)가 필요합니다. 서로 다른 서버에서 실행한다면 데모 파일과 결과 디렉터리를 같은 공유 스토리지에 두거나, 완료 후 결과를 합쳐야 합니다. shard별 작업 목록은 `plan.shard-<index>-of-<count>.json`에 기록됩니다.

shard 0만 공통 expert 평가를 실행하며, 분할 실행에서는 리포트 생성을 건너뜁니다. 모든 shard가 성공적으로 끝난 뒤 한 번 실행합니다.

```bash
.venv/bin/python -m recurrent.report --config experiment.yaml
```

중단된 shard는 같은 index/count 명령으로 다시 실행하면 완료된 학습과 평가는 건너뜁니다. 실행 중인 동일 shard를 중복으로 띄우지 마세요. `run_all.sh`는 데모 수집을 포함한 단일 실행 경로이므로 분할 실행에는 위 명령을 사용합니다. 평가 worker는 GPU가 아니라 CPU를 사용하므로 `--eval-workers` 합계를 서버 CPU 용량에 맞추세요.

기본 Recurrent 수집·평가의 물리 연산은 `physx_cpu`입니다. 학습된 정책의 GPU 배치 평가는 아래 별도 절을 따릅니다. ManiSkill의 GPU 물리 연산 가능 여부를 서버에서 별도 확인하려면 `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m recurrent.probe_gpu_sim --num-envs 16`을 실행합니다. 이 probe는 영상 렌더링 없이 16개 상태 관측 환경의 reset, GT, step을 확인합니다. 성공해도 현재 단일 환경용 expert, 관측 손상, 실패 단계 계산, 정책 상태를 배치 방식으로 바꾸고 결과를 비교 검증하기 전에는 본 평가가 GPU로 전환되지 않습니다. RTX 3090의 실제 정책 평가 검증은 아래 GPU 배치 평가 절과 검증 보고서를 따릅니다.

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
기존 JSONL에는 이어 쓰지 않고 새 출력 경로를 요구합니다. 동시 프로세스가
같은 출력 경로에 쓰는 것은 지원하지 않습니다.

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
