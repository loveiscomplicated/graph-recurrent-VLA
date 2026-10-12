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
