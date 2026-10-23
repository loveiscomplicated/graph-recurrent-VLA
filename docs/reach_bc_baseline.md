# Reach BC baseline: first-success dataset / 실제 closed-loop 평가

2026-10-12, Vast.ai **55512455**에서 구현·학습·평가했다. Flat MLP는 held-out
**0/5**, fresh **3/50 (6%)**에 성공했다. 동일 초기화의 Cartesian reference는
**5/5, 50/50**에 성공했다. 데이터·feature·action의 정합성 검사를 통과했으나,
안정적인 learned Reach baseline 확보에는 아직 미달했다. Test 결과로 모델이나
학습 설정을 재선택하지 않았다.

전체 실행 순서와 명령은 [실험 README](../graph/subtasks/reach_bc/README.md),
측정 원본은 `outputs/subtasks/reach_bc_v1/`에 있다. 기존 generated_50 export,
원본 H5, segmentation index와 split, 기존 91개 데이터 및 v1–v4 결과는 보존했다.

## 실행 환경과 범위

- Linux / Python 3.11.16 / ManiSkill 3.0.1 / SAPIEN 3.0.3 / Gymnasium 1.3.0.
- Torch 2.8.0+cpu / NumPy 1.26.4 / Pinocchio 2.7.0 / mplib 0.1.1.
- 학습·추론 CPU, simulation `physx_cpu`, `num_envs=1`.
- RTX 3090은 Vulkan rendering에 사용했다. CUDA 학습·CUDA simulation은 실행하지 않았다.
- 새 인스턴스가 빈 환경이어서 `/workspace/reach-bc-55512455/.venv`를 별도로 설치했다.
  Repository는 기존 absolute path에 복사하여 dataset 내부 참조와 hash를 유지했다.
- Mac에서는 기존 데이터 회귀 테스트와 저장 결과 분석만 수행했다. 신규 데이터의
  audit·학습·최종 SR은 모두 Linux에서 실행했다.

`reports/linux_requirements_freeze.txt`에 실제 dependency resolution을 보존했다.
Simulator factory의 CPU-only Torch RGB readback 오류도 실제로 검출·수정했다.
이 경우 shared factory는 state/RGB 모두 host readback을 사용한다. 원본 source
environment의 별도 `sapien_cuda` 설정과 shared `sapien_cpu` 설정은 SAPIEN의
process-global render device 때문에 별도 테스트 프로세스에서 확인했다.
Success threshold나 물리 backend는 변경하지 않았다.

## 기존 모델 및 인터페이스 조사

`python -m graph.prototype.train --model flat --dataset ... --output ...`은 **v1**을
학습한다. 64 peg / 64 hole surface point, robot 3 node의 14D feature를 flatten하고
81D context를 붙인 1,915D 입력을 사용한다. Encoder는
`1915 → 128 → 128 → 128`, action head는 `128 → 128 → 7`이다.
`export_bc.py`는 이 geometry format으로 데이터를 저장하며 **qvel을 export하지 않는다**.
따라서 기존 v2 CLI의 `qvel_dataset()`이 요구하는 full-task manifest의
`episodes`, `source`, `timestep`을 그대로 기대하면 이 export와 직접 호환되지 않는다.

이번 모델은 기존 **v2 `CommonPolicy(flat, goal, regression)`**로 학습 전에 확정했다.
Encoder, feature 순서·sampling, normalizer, regression BC loss를 재사용했다.
같은 encoder의 128D embedding에 기존 31D bypass를 붙이고
`159 → 128 → 7` action head를 사용한다. 총 **299,655 parameters**이다.
별도 architecture나 hidden width 변경은 없다. 새 experiment CLI는 source replay에서
qvel을 검증·저장하고 기존 모델과 normalizer를 호출하는 얇은 연결 역할이다.

31D bypass는 qvel9와 기존 geometric residual22다: EE-frame grasp translation3,
grasp relative rotation6, peg head in hole3, peg relative rotation6, clearance1,
world insertion axis3. Geometry context에는 기존 qpos·finger position·poses·sizes가
포함된다. 접촉·success·reward·expert phase·현재/미래 expert action은 입력이 아니다.
`goal_geometry()`는 **현재 peg/hole/EE pose와 size, 고정 offset**만 사용한다.
Expert 미래 waypoint나 metadata의 목표를 policy input으로 읽지 않는다.

주의: v2의 goal feature는 canonical **grasp** target이며 실제 Reach approach target과
동일한 pose는 아니다. 구조를 보존하기 위해 바꾸지 않았다. 50개 초기 state에서
둘의 position 차이는 **50.05–59.69 mm, 평균 54.82 mm**였고 rotation은 일치했다.
실제 종료 판정과 reference controller는 기존 Reach target을 사용한다. 이 feature의
task 적합성은 다음 실험에서 검토할 제한이며, offline/live feature 불일치는 아니다.

## Reach-first-success audit

각 episode에서 저장 segment 전체를 실제 Linux simulator로 replay했다. 최초 성공은
기존 `SubtaskSpec('reach')`의 live 판정이다. 평균 step 수로 자르지 않았다.
`s_k`가 최초 성공이면 action `[0,k)`와 state `[0,k]`를 저장했다.

| 항목 | 원본 | First-success |
|---|---:|---:|
| Episodes | 50 | 50 |
| Actions | 2,084 | **1,603** |
| Train / validation / test episodes | 40 / 5 / 5 | 40 / 5 / 5 |
| Train / validation / test actions | 1,663 / 225 / 196 | **1,280 / 173 / 150** |
| Episode 평균 actions | 41.68 | **32.06** |

모든 50 episode에 최초 성공 뒤 action이 있었다. 제거한 sample은 **481개 (23.08%)**,
episode당 2–17개, 평균 9.62개다. 원본 segment 내부에서 최초 성공 이후 다시 success
조건을 벗어난 episode는 **0개**, 모든 endpoint도 success였다. 제거 구간의 gripper
command는 모두 **+1(open)**이었다. 즉, 이번 데이터에서는 Reach segment 내부의
gripper close/Grasp action 혼입은 관측되지 않았다. 제거 이유는 정확한 first-success
Reach label 정의이며, 실제 검증 결과와 달리 혼입을 단정하지 않는다.

EE의 episode별 평균 실제 translation은 성공 전 **6.699 mm/step**, tail
**2.219 mm/step**이었다. 이미 성공한 뒤 목표에 더 가까워지는 움직임도 저장 segment에
남아 있었음을 보여준다. 각 episode의 action 통계, 최초 성공 step, endpoint metric,
tail 길이와 이탈 횟수는 `reports/trimming_audit.json`에 있다.

정합성 결과:

- 전체 2,084 action replay의 simulator state·native observation 최대 오차 **0**.
- 기존 offline geometry와 live `GTAdapter` geometry 최대 오차 **0**.
- Raw flat/state feature 및 train 통계로 정규화한 feature 최대 오차 **0**.
- qpos/qvel은 실제 observation의 `[:9]` / `[9:18]`와 검증했다.
- 50 episode 각각 `len(states) = len(actions) + 1`; terminal success state 포함.
- 원본 export의 input `s_t`와 target `a_t`를 그대로 subset으로 보존했다.
- 원본 H5/index/export hash와 split 유지, seed/source UID 중복 없음.
- 모든 normalization은 train 40 episode에서만 계산한다. Validation/test geometry와
  qvel을 극단적인 값으로 바꿔도 train 통계가 변하지 않는 테스트도 통과했다.

학습용 파일: `data/reach_first_success.pt`. 원본 `generated_50/bc/reach.pt`는 수정하지 않았다.

## Smoke 및 학습

Load, forward/backward, finite gradients, small-batch overfit, checkpoint roundtrip,
offline/live agreement를 먼저 실행했다. 32개 train sample을 300번 업데이트한 smoke는
MSE **0.249586 → 0.000006244**였고, 저장·복원 prediction은 정확히 일치했다.
본 학습은 smoke weights를 사용하지 않고 seed 0으로 새로 초기화했다.

| 설정 | 값 |
|---|---|
| Epochs / batch | 30 / 32 |
| Optimizer | AdamW |
| LR / weight decay | 0.001 / 0.00001 |
| Scheduler | 기존 CosineAnnealingLR, T_max=30 |
| Loss | 7D normalized action의 unweighted MSE |
| Normalization | 기존 population std, floor 0.001; train only |
| Selection | Best validation action MSE |
| Selected epoch | **21** |
| CPU training time | **4.51 s** |

| Selected checkpoint metric | Train | Validation | Test |
|---|---:|---:|---:|
| Action MSE | 0.00082935 | **0.00079100** | **0.00247839** |
| Translation RMSE (normalized) | 0.018456 | 0.019668 | 0.033040 |
| Rotation RMSE (normalized) | 0.039707 | 0.036638 | 0.067556 |
| Gripper MAE | 0.005869 | 0.015506 | 0.016128 |

Test translation command RMSE는 **3.304 mm/component**, rotation command Euler
RMSE는 **0.006756 rad/component**이다. 이는 controller command 차이이며 실제
rollout의 목표 pose 오차와 구분한다. Train 표는 선택 checkpoint의 재측정값이고,
epoch history의 train 값은 optimizer 업데이트 중 batch별 prediction이다.

Epoch 30 validation MSE는 0.00083011로 epoch 21보다 높았다. 마지막 5 epoch에도
일관된 개선이 없었다. 추가 학습이나 hyperparameter search는 하지 않았다.
Test는 checkpoint 선택 이후 한 번 측정했고, test SR로 설정을 바꾸지 않았다.

Checkpoint: `checkpoints/best.pt`. Config·normalization·optimizer/scheduler state·hash·
split·versions는 checkpoint 및 `training/`에 있다. Loss/component curve는
`figures/loss_components.png`다.

## Independent closed-loop 평가

기존 test IDs **15, 29, 31, 33, 41**의 source starts와 fresh seed
**12000–12049**를 고정했다. 로컬의 43 metadata source / 175 seed 값 및 Linux
metadata를 검사하여 기존 train/val/test/smoke/evaluation과 fresh seed 중복이 없었다.
각 시작은 두 번 실제 reset하여 state·geometry·Reach-local target fingerprint가
정확히 일치했다. Manifest에는 seed, reset kwargs, environment config, max_steps,
subtask, simulator version/backend, source/data hash와 fingerprint를 저장했다.
Source prefix 없이 fresh reset하는 경우도 기록된 환경 설정을 유지했다.

매 step 최신 GT를 추출하고 기존 train normalizer → Flat prediction → `[-1,1]`
clip → `pd_ee_delta_pose` controller → 기존 success/failure/timeout 순서로 실행했다.
History/hidden state는 없다. Budget은 기존 Reach **130 steps**다.

| Model | Dataset | Test action MSE | Held-out SR | Fresh SR | Params | CPU forward latency |
|---|---|---:|---:|---:|---:|---:|
| **Flat MLP BC** | Reach first-success | **0.00247839** | **0/5** | **3/50** | 299,655 | **0.136 ms** |
| Cartesian reference (진단용) | N/A | N/A | **5/5** | **50/50** | N/A | 0.382 ms (pose command 계산) |
| Flat + fixed-open (진단용) | Reach first-success | N/A | **0/5** | **5/50** | 299,655 | 0.129 ms |

Flat의 GT extraction + feature normalization + forward 평균은 **5.326 ms**다.
Forward p50/p95는 0.130/0.153 ms이며 latency는 simulator step을 제외한다.

| Run / set | Attempted | Valid | Success | Physical failure | Timeout | Invalid | 성공 평균 step |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat / held-out | 5 | 5 | 0 | 3 | 2 | 0 | N/A |
| Flat / fresh | 50 | 50 | 3 | 22 | 25 | 0 | **34.0** |
| Reference / held-out | 5 | 5 | 5 | 0 | 0 | 0 | **12.8** |
| Reference / fresh | 50 | 50 | 50 | 0 | 0 | 0 | **12.9** |
| Fixed-open / held-out | 5 | 5 | 0 | 3 | 2 | 0 | N/A |
| Fixed-open / fresh | 50 | 50 | 5 | 17 | 28 | 0 | **50.0** |

Physical failure는 모두 기존 `peg_displaced`였다. SR denominator는 valid resets이며
invalid 대체 sampling은 없었다. Flat fresh Wilson 95% CI는 **2.06–16.22%**,
reference는 **92.87–100%**, fixed-open은 **4.35–21.36%**다.
Flat 성공 fresh seed는 **12005, 12027, 12034**이다. 첫 평가의 SSH console 접속이
끊겼지만 process는 55개를 완료하고 결과를 저장했다. 별도 55개 repeat에서 초기
fingerprint, success/failure, step 수, 모든 state/action trace가 **정확히 동일**했다.

## Reference controller와 실패 진단

Installed controller source와 실제 CPU test에서 translation scale **+0.1 m**,
rotation scale **−0.1 rad**, root translation / root-aligned body rotation,
intrinsic XYZ Euler composition, rotation vector norm clipping을 확인했다.
Gripper **+1 → 0.04 m** target도 직접 검사했다. Reference는 현재 EE와 기존
Reach target의 root-frame residual에 고정 gain 0.5를 적용하고 action bounds/norm을
제한하며 gripper +1을 유지한다. IK/planner를 새로 작성하거나 threshold를 바꾸지 않았다.
단일 step에서 위치·방향 오차 감소 테스트와 동일 55-start의 성공으로 interface를 검증했다.
Reference는 학습된 정책과 동일한 연구 성과로 해석하지 않는다.

Flat fresh 실패 episode의 최소 position error 중앙값은 **57.43 mm**, 최종
position error 중앙값은 **182.81 mm**, 최종 rotation error 중앙값은 **34.85°**다.
Held-out 실패의 동일 값은 **30.50 mm / 108.15 mm / 27.06°**다.

- **접근이 제한되는 예:** 12001은 초기 거리 163.1 mm, 최소 126.4 mm 후 28 step에서
  peg 이동으로 실패했다. 단순히 output이 0인 경우는 아니다. Trace에는 action 방향의
  반전과 실제 peg 이동이 기록되어 있다.
- **목표 근처에서 실패하는 예:** 12017의 최소 거리는 12.35 mm였으나 그 state의
  rotation error는 **41.84°**, peg 이동은 **31.87 mm**였다. Reach 조건을 충족하지
  못했으며 90 step에서 기존 peg displacement 판정으로 종료했다.
- **접근 후 이탈 예:** 12000은 초기 126.4 mm에서 **34.74 mm**까지 접근한 뒤
  최종 **388.3 mm**, rotation **67.74°**로 멀어져 130 step timeout이었다.
- **빠른 peg 이동 실패:** held-out 11033은 11 step에서 peg 이동 **41.61 mm**로 종료했다.

Fresh Flat의 4,912 control step 중 motion component clipping은 **977 (19.89%)**,
translation clipping은 591, rotation component clipping은 929,
controller rotation norm saturation은 955 step이었다. Gripper raw clipping은
4,384 step이나 open 경계 +1을 조금 넘는 출력도 포함하므로 motion saturation과
분리한다. 실제 finger open 조건 미달은 fresh **136 step**, held-out **17 step**이었다.
Fixed-open에서도 fresh 110 step에서 실제 open 조건이 미달했다. Command가 +1이어도
접촉·실제 dynamics가 finger state에 영향을 줄 수 있음을 보여준다.

Fixed-open은 fresh 3→5 성공이지만 대부분 실패가 남았다. 그 결과를 main SR로
대체하지 않았다. 이 비교는 gripper 부담만으로 전체 실패를 설명하기 어렵다는
근거이며, 작은 표본에서 인과적 효과나 통계적 우위를 주장하지 않는다.

6개 대표 rollout의 저장 action replay에서 prediction·qpos 최대 오차는 모두 **0**이었다.
Train-normalized 1,946D feature의 nearest train RMS distance는 held-out expert
median **0.265**, p95 **0.315**였다. 성공 12005는 sample median **0.205**,
최대 **0.230**인 반면, timeout 12000은 median **3.158**, 최대 **61.600**이었다.
11033도 후반 최대 **21.321**까지 이탈했다. 초기 12000은 0.213으로 expert 범위에
가깝지만 rollout 중 커졌다. 이 수치는 제한된 demonstration 분포에서 벗어나는
closed-loop state와 출력 불안정의 근거이며, 단독으로 원인을 증명하는 지표는 아니다.

Offline train/validation이 비슷하여 큰 validation overfitting의 뚜렷한 증거는 없다.
Small-batch overfit도 가능했다. Verified expert state에서 feature/action mismatch는
검출되지 않았고 reference는 모든 seed에 성공했다. 남은 병목은 작은 demonstration
coverage, 실제 Reach와 다른 기존 grasp-goal feature의 적합성, off-distribution state에서
누적되는 prediction error 등을 구분하는 것이다. 이번 결과만으로 하나를 확정하지 않는다.
DAgger, RL, recovery 수집, recurrent/graph 모델은 실행하지 않았다.

대표 영상: `videos/flat_12005.mp4` (성공), `flat_12000.mp4` (timeout),
`flat_11033.mp4` (빠른 peg 이동), `flat_12017.mp4` (근접·방향 오차),
`reference_12000.mp4`. 영상에는 seed/step/position/rotation 표시와 JSON trace가 있다.
Video rollout도 main 결과와 trace가 동일하다. Plot은
`figures/representative_errors.png`, `figures/sr_failures.png`에 있다.

## 테스트 및 다음 단계 판단

- Reach Linux 실제 suite **9/9 pass**: trimming/정렬, 원본/split, train-only 통계,
  model gradient/checkpoint, overfit evidence, feature agreement, live success,
  timeout, installed action decoding, seed overlap/fingerprint, all-55 policy repeat.
- Linux source/model regression **52개, 44 pass / 8 expected skips**: generated_50의
  모든 segment 정렬 검사와 실제 다섯 subtask reset/replay 포함. Skip은 복사하지 않은
  legacy artifact 검사와 opt-in old-data integration이다.
- Shared Linux CPU/RGB/GT **8/8 pass**. 초기 RGB-readback 오류 및 한 process의
  render-device 전환 오류는 기존 실패 로그에 보존했다. 수정 후 별도 suites 통과.
- Mac 기존 subtask 실제 CPU integration **32개, 31 pass / 1 expected skip**:
  skip은 Linux generated index가 지정되지 않은 검사다. Linux에서 해당 검사를 실행했다.
- Local data/model regression **53개, 50 pass / 3 opt-in skips**.

데이터 준비부터 실제 rollout까지의 재현 가능한 파이프라인과 Graph 비교에 사용할
고정 dataset/split/manifest는 확보했다. 다만 **6% SR은 안정적인 learned baseline으로
간주하지 않는다**. 현재 상태에서 GNN을 추가하면 baseline의 제어 불안정과 Graph의
기여를 분리하기 어렵다. Graph 성능 우위의 본격 해석 전에 Reach BC의 실패 병목을
확인하는 것이 적절하다. 이번 작업에는 미실행 필수 deliverable이 없으며, 안정적인
학습 정책 확보와 Graph 비교는 이후 연구 과제로 남는다.

## 추가 병목 진단: frozen checkpoint의 제어 피드백

같은 날 사용자 요청에 따라 실패를 더 세밀하게 분석했다. **학습, architecture,
checkpoint 선택, success threshold, 데이터 split을 변경하지 않았다.** 새 결과는
`outputs/subtasks/reach_bc_v1/bottleneck_v1/`에 별도로 기록했다. 아래 결과를 원래
Flat BC의 6% main SR로 대체하지 않는다.

### 방법과 검증 범위

1. 원래 55개 rollout의 모든 5,418 control step을 저장 action으로 Linux에서 다시
   실행했다. 매 state에서 입력 그룹별 값을 바꾼 prediction, 정규화 값, action의
   목표 방향 성분을 계산했다. 원래 prediction·qpos·qvel·EE/peg pose 최대 오차 **0**.
2. 민감한 입력 그룹에 대한 input Jacobian을 일부 초기/근접 state에서 계산했다.
   Optimizer나 weight gradient update는 없었다.
3. 사전에 고정한 입력 intervention 4개를 **각각 같은 55 starts**에서 실행했다.
   Bounds와 평균은 기존 train 40 episode에서만 계산했다. 총 220개 추가 rollout이다.
4. Train seed 11000/11001, validation 11005, held-out 11029/11033, fresh
   12000/12005에서 실제 simulator 1-step finite difference를 실행했다.
5. 실패 rollout의 실제 prefix를 그대로 실행한 뒤 reference로 전환하는 5개 probe를
   실행했다. Snapshot injection, budget/조건 변경, training dataset export는 없었다.

Code는 `bottleneck.py`, 집계·figure는 `bottleneck_report.py`다. Probe input 변경이
원본 feature/weight를 수정하지 않고 지정한 channel만 바꾸는 새 테스트 **2/2**가
Linux 및 Mac에서 통과했다. 실제 실행·초기 fingerprint 일치는 보고서에 보존했다.

### 확인된 병목 1: 작은 gripper 오차가 EE action으로 재유입된다

Train gripper command는 모두 +1이다. 실제 finger position의 train 표준편차는
left **8.34e-8 m**, right **2.70e-7 m**, finger velocity 표준편차는
**1.45e-4 / 1.41e-4 m/s**였다. 기존 normalizer는 이 channel에 **std floor 0.001**을
사용한다. Finger position은 gripper context와 qpos에 중복 포함되고, finger velocity는
v2의 31D bypass를 통해 action head로 직접 들어간다.

여기서 z는 `(값 - train mean) / max(train std, 0.001)`이다. 실제 표준편차가
floor보다 작으므로 raw population 표준편차로 나눈 값과 구분한다. Floor는 오히려
그 경우의 수치 증폭을 줄인다. **Floor 0.001 자체가 계산 오류라고 주장하지 않는다.**
Rollout에서 생기는 finger 움직임과 그것에 대한 policy의 큰 출력 민감도가 문제다.

Seed **11033**의 구체적인 전개:

| 시점 | 실제/예측 값 |
|---|---|
| 첫 policy gripper command | **0.9581** |
| Controller의 finger target | 약 **0.038953 m**, fully-open target보다 1.05 mm 작음 |
| 첫 step 뒤 finger position | 약 **0.039589 m**, 여전히 open 기준 0.035 m 만족 |
| Finger 이동 | 약 **0.411 mm** |
| Finger velocity | **−0.006446 / −0.006310 m/s** |
| 정규화 finger velocity | **−6.44 / −6.43** |
| 첫 yaw action | **0.0712** |
| 다음 yaw action | **1.1208**, controller component bounds 초과 |
| 같은 두 번째 state에서 finger 입력을 train mean으로 대체한 yaw | **0.1501** |
| 같은 state에서 qvel 전부를 physical zero로 대체한 yaw | **0.0716** |

이 same-state probe에서는 simulator state, peg geometry, 목표, model weights가 모두
동일하다. Finger 입력 그룹만 바꾸면 6D motion prediction 차이의 norm이 **1.1501**이다.
Jacobian의 국소 선형 추정으로 yaw에 대한 finger velocity contribution은 **−0.7626**,
finger position contribution은 **+0.000321**이었다. 실제 비선형 mean-replacement의
yaw 차이는 −0.9707이다. 이 state에서는 **finger velocity 경로가 주된 증폭 요인**이다.
Position과 velocity를 동시에 바꾼 intervention 자체를 완전한 physical state로 해석하지 않는다.

Fresh 50개 중 첫 gripper prediction이 +1 미만인 **25개는 모두 실패**했다. 이 그룹의
두 번째 state에서 finger 입력을 평균으로 바꿨을 때 6D action 차이 median은 **0.416**,
velocity z 최대값 median은 **2.223**이다. 첫 prediction이 +1 이상인 나머지 25개의
동일 값은 **0.0118 / 0.148**이고 3개가 성공했다. 이 cohort 비교는 상관관계지만,
same-state intervention과 실제 첫 step의 손가락 움직임이 영향 경로를 직접 뒷받침한다.

다만 +1 이상으로 시작한 25개도 22개 실패했다. Finger coupling만으로 전체 실패를
설명할 수 없다. Finger-qvel input weight의 RMS 변화는 초기 weight RMS의 **71.6%**였다.
따라서 단순히 학습되지 않은 random weight가 남았다고 단정하지 않는다. Train에서는
finger velocity와 일부 EE action이 상관되어 있었다(최대 |Pearson r| **0.502**).
작은 수동적 finger dynamics를 motion 예측 신호로 사용할 수 있었던 것이 실제 gripper
오차로 발생한 능동적 finger 움직임에서는 취약해진 것으로 해석할 수 있다.

### 확인된 병목 2: qvel 의존적 action 피드백이 작은 오차를 확대할 수 있다

실제 환경의 동일 reset state에서 현재 action 6개 component를 각각 ±0.01만 바꿔
한 step 실행한 뒤 다음 policy action을 측정했다. Translation perturbation은 ±1 mm,
rotation Euler component perturbation은 ±0.001 rad다. 모든 비교는 실제 qpos/qvel,
controller 및 geometry response를 포함하며, reset fingerprint가 동일함을 확인했다.

| 다음 action을 계산하는 조건 | 최대 singular gain 범위 | Median |
|---|---:|---:|
| 원래 Flat policy | **1.804–2.934** | **2.254** |
| Finger input train mean | 1.606–2.850 | 2.291 |
| qvel input physical zero | **0.063–0.113** | **0.082** |
| EE Rzz만 train range 제한 | 1.804–2.934 | 2.254 |
| 낮은 분산 channel train range 제한 | 1.583–2.874 | 2.363 |
| 동일 perturbed state의 reference | **0.190–0.276** | **0.196** |

현재 action의 작은 차이가 다음 predicted action의 더 큰 차이로 전달될 수 있다.
Finger 입력을 안정화해도 팔 qvel 경로의 영향이 남는다. 이 현상은 **train 및 validation
seed에서도** 관측돼 fresh 초기화만의 문제로 보기 어렵다.

부호도 확인했다. 현재 normalized X translation action에 대한 다음 X action의
finite-difference coefficient는 policy에서 **+0.782–+1.089**, 같은 state의 reference는
**−0.188–−0.184**였다. Held-out 11029에서는 X action을 0.01 늘릴 때 다음 policy
X prediction은 약 0.01035 증가하고 reference는 약 0.00188 감소한다. 이 초기 state들에서
policy가 이전 움직임을 이어가려는 응답을 보이는 반면, reference는 바뀐 현재 pose를
기준으로 남은 오차를 줄이는 응답을 보였다. 모든 축·state의 전역 제어 법칙으로 일반화하지 않는다.

이는 **한 step의 local action-to-next-action sensitivity**이며 full robot/controller
state의 asymptotic stability를 증명한 결과는 아니다. 원래 probe matrix spectral radius는
0.771–1.175이고, 성공 seed 12005도 singular gain 2.300이다. Gain이 1보다 크다는
사실만으로 모든 episode가 불안정하다고 분류하지 않는다. 실제 rollout 이탈, qvel
intervention, reference 비교를 함께 볼 때 강한 속도 의존성과 오차 증폭 가능성의 근거다.

### 전체 rollout intervention: 입력 제한만으로 해결되지 않는다

| Frozen-checkpoint 조건 | Held-out | Fresh | Fresh peg 이동 실패 | Fresh timeout |
|---|---:|---:|---:|---:|
| **원래 Flat BC** | **0/5** | **3/50** | **22** | **25** |
| Finger position/velocity 입력 → train mean | 0/5 | 5/50 | 11 | 34 |
| qvel 9개 입력 → physical zero | 0/5 | **0/50** | **0** | **50** |
| EE rotation Rzz만 train min/max 제한 | 0/5 | **3/50** | 19 | 28 |
| 기존 std가 0.001 floor인 channel → train min/max 제한 | 0/5 | 5/50 | 8 | 37 |

Finger mean intervention은 learned gripper command를 그대로 사용한다. 기존 fixed-open은
실제 command를 +1로 바꾼 별도 조건이다. 서로 다른 실험이므로 혼동하지 않는다.
Floor-channel intervention은 context 33개와 shared state 6개를 제한한다.
모든 bounds는 train에서만 계산했고, variant별로 다른 threshold를 탐색하지 않았다.

qvel zero는 충격성 peg 이동 실패를 없앴지만 **전부 timeout**이었다. Fresh 최종
position error median **154.9 mm**, rotation error median **9.19°**, 마지막 raw 6D motion
command norm median **0.0609**였다. 속도 입력을 없애면 정책의 이동 능력도 크게 변한다.
실제 이동 중인 state에 qvel zero를 넣는 것은 expert manifold 밖의 진단이므로,
이 결과로 qvel 삭제가 올바른 최종 설계라고 결론 내리지 않는다. 현재 checkpoint의
goal-conditioned imitation이 속도 변화 없이 안정적으로 목표까지 조절하는 controller로
작동하지 않는다는 관측이다.

Rzz channel 단독 제한은 원래 3개 성공을 유지했다. 초기 급증 state에서 Rzz를 바꿔도
거의 효과가 없었다(held-out 두 번째 state 최대 action 차이 약 1.6e-7).
이전 보고서의 큰 geometry z-score가 **실패의 최초 원인임은 입증되지 않았다**.
EE가 크게 기울어진 뒤 이 channel이 확대되는 현상은 후속 증폭 또는 결과일 수 있다.
낮은 분산 channel 전반의 제한도 SR은 5/50에 그쳤다. 단순 normalization clipping이
안정적인 baseline을 만들었다고 해석하지 않는다.

### 실패 위치: 대부분 충분히 가까워지지 못한다

Fresh 50개에서 position 20 mm 조건을 한 번이라도 통과한 것은 **4개**,
rotation 15° 조건을 한 번이라도 통과한 것은 **41개**, 두 조건을 같은 state에서
통과한 것은 **3개**였다. 이 세 episode가 원래 성공 episode다. 실패의 대부분은
미세한 gripper success 판정만 남은 상황이 아니며, 위치 접근·유지부터 실패한다.
기존 workspace predicate의 실패는 **0개**였다. 이 predicate는 peg의 workspace를
검사하며, EE workspace 전체의 안전성/IK reachability 인증으로 해석하지 않는다.

48/50 fresh episode에서 목표로부터 멀어지는 방향의 translation command가 적어도
한 번 관측됐다. Planner 경로는 언제나 거리를 단조 감소시켜야 하는 것은 아니므로
이 count 자체를 control error의 정의로 쓰지 않는다. 최소 거리 이후 큰 이탈과
oscillation trace가 동반되는 episode에서 제어 방향·크기 불안정의 증거로 사용한다.

같은 learner state에서 목표가 여전히 가능한지 확인한 reference-switch 결과:

| Seed | 그대로 replay한 learner prefix | Reference 성공까지 추가 step |
|---|---:|---:|
| 11029 | 30 | **7** |
| 11033 | 1 | **11** |
| 12000 | 14 | **4** |
| 12001 | 23 | **12** |
| 12017 | 50 | **8** |

다섯 곳 모두 성공했다. 이전 learner action과 실제 접촉/controller history를 유지하고
원래 130-step budget과 context를 그대로 사용했다. 이 probe는 진단용이고 recovery
demonstration이나 학습용 dataset은 만들지 않았다. 이 지점에서는 목표 접근 실패를
환경의 도달 불가능성이나 controller scaling 탓으로 설명하기 어렵다. 이미 원래 failure가
발생한 뒤의 상태나 모든 IK 상태를 검사한 것은 아니다.

### 배제한 설명과 아직 확정하지 않은 부분

- **Feature/API/control mismatch:** 모든 baseline step의 live replay·prediction이 정확히
  일치하고 reference도 성공한다. 현재 검증 범위에서 주요 원인이라는 증거가 없다.
- **Reach label / post-success 혼입:** 제거한 tail은 모두 open이며 first-success 판정도
  일치한다. 선택 checkpoint의 마지막 pre-success state MSE는 train 0.000307,
  validation 0.000266, test 0.000448로 전체 MSE보다 작았다. Label 경계 오류가 주된
  병목이라는 증거는 없다.
- **단순히 gripper 닫힘:** 손가락이 아직 open threshold를 만족해도 qvel coupling으로
  EE output이 커졌다. Fixed-open이나 finger input 변경만으로 대부분 실패가 남는다.
- **Rzz normalization 단독 원인:** 같은 state와 전체 rollout에서 영향 범위를 분리했고
  단독 제한으로 SR이 개선되지 않았다. 최초 원인으로 단정하지 않는다.
- **Grasp/Reach goal offset:** 기존 grasp residual이 실제 Reach target보다 50–60 mm
  아래에 있다는 제한은 여전히 존재한다. 이번 진단으로 이 offset이 주원인이라고
  입증하지는 못했다. Frozen checkpoint에 다른 목표 feature를 넣는 것은 학습 분포를
  바꾸므로 성능 개선용 inference 수정으로 사용하지 않았다.
- **유일한 원인 / data coverage / architecture capacity:** 40개 demonstrations의 좁은
  state 분포, velocity와 action의 상관관계, 손가락 신호의 영향, 부족한 error correction을
  완전히 독립적인 원인으로 분리하지 못했다. 데이터 크기나 MLP 용량만 탓할 근거도 없다.

### 다음 실험의 우선순위

우선 확인할 것은 **손가락 velocity가 EE prediction에 재유입되는 취약한 경로**와
**velocity-conditioned action imitation의 closed-loop 민감도**다. 다음 변경은 같은
Flat MLP에서 train/inference를 함께 맞춘 feature/normalization ablation으로 검토하고,
validation 시작 상태의 closed-loop 결과와 control sensitivity를 함께 확인하는 것이
적절하다. 현재 checkpoint에 qvel zero나 clipping을 붙이는 것으로 완료된 해법을
주장하지 않는다. 실제 Reach residual의 명시화는 그 다음의 별도 ablation 후보다.

Graph/Recurrent를 넣으면 동일 state bypass의 취약한 경로를 그대로 물려받을 수 있다.
이번 진단 결과를 model 선택용 fresh-test tuning에 쓰지 않고, train/validation 기준으로
최소 변경을 선택한 뒤 고정된 benchmark에서 별도 결과를 보고해야 한다. 이번에는
새 모델 학습, RL/DAgger, recovery data 수집을 실행하지 않았다.

재현 명령은 아래와 같다. Audit/rollout probes는 기존 result가 존재하면 실행을
거부한다. Offline 집계와 figure는 같은 고정 checkpoint에서 재생성할 수 있다.

```bash
REACH_PY=/workspace/reach-bc-55512455/.venv/bin/python
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --audit
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode finger_mean
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode qvel_zero
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode ee_floor_clip
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode low_variance_clip
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --feedback
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --reference-switch
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --offline
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck_report
"$REACH_PY" -m unittest graph.subtasks.reach_bc.tests.test_bottleneck -v
```

`config.json`에 diagnostic 조건과 checkpoint/data/manifest hash를 기록했다.
`audit.json`, 네 variant JSON, `feedback.json`, `reference_switch.json`, `offline.json`,
`weight_updates.json`, `summary.json`, `results.md`, supervisor logs 및 figure 두 장을
저장했다. Full trace archive도 로컬에 보존했다. 그림은
`bottleneck_v1/figures/finger_coupling.png`, `feedback_interventions.png`다.
