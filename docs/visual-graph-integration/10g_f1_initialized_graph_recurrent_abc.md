# Task10-G — F1-initialized Graph–Recurrent A/B/C

**고정 A/B/C 학습과 Development 선택, 독립 Closed-loop Final 및 Geometry/Action 분석 완료.**

## 실제 실행과 Identity

- Root: `/workspace/task10g_f1_abc_20261031/full`
- Instance57987772 / RTX3090; CPU PhysX + CUDA renderer/vision/policy. GPU PhysX와 로컬 macOS CUDA는 미실행.
- F1 SHA256: `781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6`
- Dataset manifest SHA256: `f1b61e7635be3a9b1533d5158e8be8df869edf67dddee926335c68486d5cf6b1`
- Optimizer updates: 각 11487; seed42 한 개. BC sampling schedule, train-only ActionCodec, batch2/sequence32 공통.
- 선택 병렬도: 3; 실제 3 CUDA process 동시 benchmark를 실행했다. 전체 memory/throughput은 [기록](../../outputs/task10g/final_results/concurrency_benchmark.json)에 보존.
- 단일 geometry λ: 0.0313649216. Train-only gradient median의25% rule/clamp/fallback은 [calibration](../../outputs/task10g/final_results/lambda_calibration.json)에 보존.
- 기존 ExperimentLogger의 실행 UUID와 로컬 기록을 재사용했다. Git commit, 외부 publishing, 새 유료 인스턴스 생성 없음.

## Functional Closed-loop Final

| Condition | Parents | Grasp | Lift | Insertion | Selected step |
|---|---:|---:|---:|---:|---:|
| A | 12 | 0 | 0 | 0 | 9000 |
| B | 12 | 4 | 4 | 0 | 9000 |
| C | 12 | 3 | 3 | 0 | 11487 |

Final seeds: `[98200, 98201, 98202, 98203, 98204, 98205, 98206, 98207, 98208, 98209, 98210, 98211]`. Zero offsets. 최대 250 Action, 공식 success/환경 종료에서 종료.
Development는 고정 Validation Parent에서만 실행했다. Insertion→Lift→Grasp→Alignment→최종 TCP–Peg error→Validation BC→earliest step 순으로 선택하고 Final 전에 checkpoint SHA를 고정했다.
각 조건은 독립 causal vision memory/GRU reset을 사용한다. 같은 seed의 초기 simulator/controller/RNG snapshot과 최초 graph/Robot76 hash가 정확히 일치했다. PhysX의 숨은 solver 상태 전체 직렬화나 모든 장치의 bit-exact trajectory를 주장하지 않는다.
[전체 Final](../../outputs/task10g/final_results/final_rollouts.csv) · [선택 동결](../../outputs/task10g/final_results/model_selection_frozen.json) · [Paired 분석](../../outputs/task10g/final_results/paired_success_analysis.csv)

Strict Reach는 기존 Task10D의 ordered approach/final pose/open-gripper/early-close/peg-motion/collision 계약으로 별도 계산했다. 실제 Insertion 성공보다 우선하지 않는다. End-to-end inference, 전체 policy, CUDA encoder+readout 시간은 각각 다른 열로 기록했다.
![Functional success](../../outputs/task10g/final_results/figures/functional_success.png)

## Geometry와 Action 반응

Geometry는 이미 관찰했던 F1 Final91000–91011의 재사용 진단이다. 새 Closed-loop Blind Final과 구별한다. A는 F1 head, B는 동결 F1 head 진단, C는 공동 학습 head를 기록했다. 주 비교는 세 encoder에 동일 F1 초기 head·동일 train-only500-update budget을 적용한 독립 probe다(Engineering miniature는3-update). B의 동결 head 오차만으로 representation 열화를 단정하지 않는다.
실제 물체 Pair의 ±5/10mm Δ vector error/direction/signed gain/magnitude ratio/zero-response 차이를 기록했다. Action 분석은 기존 Task10D의 실제 Peg 개입과 Expert counterfactual 코드로 같은 Robot76, previous hidden, causal prefix를 고정한다. 물체 이동 mm와 controller command response mm는 별도 물리량이다. Baseline repeat/물리/Expert 계약을 통과한 Pair만 주 해석에 쓴다.
[Geometry metrics](../../outputs/task10g/final_results/geometry_metrics.csv) · [Actual displacement](../../outputs/task10g/final_results/paired_displacement_metrics.csv) · [Action correction](../../outputs/task10g/final_results/action_correction_metrics.csv)

![Geometry](../../outputs/task10g/final_results/figures/geometry_errors.png)

![Action response](../../outputs/task10g/final_results/figures/action_response.png)

## 학습과 실패 분석

A의 encoder/readout은 동결, B는 BC joint, C는 BC+Geometry joint다. C State Head는 Action/GRU 입력에 포함되지 않고 추론에서 실행하지 않는다. 동일 GNN 및 Projection/GRU/Action 초기 hash, lambda0 B/C gradient 일치, GT sidecar 차단 positive control, geometry gradient 전달과 실제 checkpoint 다음 update resume를 검증했다.
정상 Expert prefix는 승인 nominal expert action/robot/timestep의 정확한 비교 후 BC를 활성화했다. 교란 구간은 실제 관측을 history에 포함하며 BC를 끈다. Recovery는 유효 expert labels만 사용했다. 같은 Parent의 branch를 독립 Parent로 세지 않는다.
[Expert phase별 controller 오차](../../outputs/task10g/final_results/phasewise_action_errors.csv)와 [Final 실패 구간](../../outputs/task10g/final_results/failure_analysis.csv)에 초기 action fit, direction reversal, premature close, saturation, missing geometry, collision, error accumulation을 기록했다. Hidden carry/reset은 계약 시험과 trace로 검사했고 특정 실패 원인을 증명했다고 주장하지 않는다. Recovery 추가 rollout은 NOT_TESTED다.

![Train/validation](../../outputs/task10g/final_results/figures/training_validation_curves.png)

## 연구 질문 판정

| Question | Verdict | Evidence scope |
|---|---|---|
| B−A: 공동 BC 업데이트의 실제 제어 이점 | INCONCLUSIVE | paired outcome; 한 training seed/12 Parent |
| C−B: Geometry auxiliary 추가 효과 | INCONCLUSIVE | 성공률·geometry·action response를 함께 해석 |
| C−A: 전체 결합 효과 | INCONCLUSIVE | 제한된 paired 표본; 일반적 우월성 주장 없음 |

모든 functional metric이0이어도 사전 등록한 평가·진단을 끝냈다. Offline BC/geometry loss 감소는 Closed-loop success 또는 정확한 action correction의 증거를 대신하지 않는다. 추가 λ 탐색/학습/데이터 수집은 실행하지 않았다.

## 운영·재현성과 보존

각 stage는 atomic manifest와 입력/출력 SHA, 실제 step/episode 수, 종료코드, traceback을 남긴다. Lock과 자식 process group 관리로 중복 실행을 막고, OOM은 동일 batch/objective로 마지막 검증 checkpoint에서만 제한적으로 재개한다. 정상 worker 산출물은 유지한다. tmux 프로세스는 SSH 종료와 독립적이며 인스턴스 삭제/재생성 복구는 보장하지 않는다.
원본 F1 checkpoint/dataset/source archive의 SHA를 종료 시 재검사했다. Scratch는 인스턴스 파일시스템이며 삭제 시 보존되지 않을 수 있다. 자동 외부 백업/전달 연결은 설정하지 않았으므로 전송 성공을 주장하지 않는다. 보고서 full content는 실행 source의 docs/visual-graph-integration에도 복사한다.
[Verification](../../outputs/task10g/final_results/verification.json) · [Artifact index](../../outputs/task10g/final_results/artifact_index.json) · [Completion audit](../../outputs/task10g/final_results/completion_audit.json)

- B-A / insertion: gained0, lost0, difference0/12; exact discordant p=1.
- B-A / lift: gained4, lost0, difference4/12; exact discordant p=0.125.
- B-A / grasp: gained4, lost0, difference4/12; exact discordant p=0.125.
- B-A / alignment: gained0, lost0, difference0/12; exact discordant p=1.
- C-B / insertion: gained0, lost0, difference0/12; exact discordant p=1.
- C-B / lift: gained2, lost3, difference-1/12; exact discordant p=1.
- C-B / grasp: gained2, lost3, difference-1/12; exact discordant p=1.
- C-B / alignment: gained1, lost0, difference1/12; exact discordant p=1.
- C-A / insertion: gained0, lost0, difference0/12; exact discordant p=1.
- C-A / lift: gained3, lost0, difference3/12; exact discordant p=0.25.
- C-A / grasp: gained3, lost0, difference3/12; exact discordant p=0.25.
- C-A / alignment: gained1, lost0, difference1/12; exact discordant p=1.
## 수치 요약과 해석 범위

아래 표는 동결된 산출물의 추가 집계다. 정책·probe를 다시 학습하거나 checkpoint를 재선택하지 않았다.

| 조건 | 마지막 Train BC | 마지막 Validation BC | 선택 step | 선택 Validation BC | C 평균 Geometry loss | 추가 Geometry graph |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.03811 | 0.08077 | 9000 | 0.08196 | — | 0 |
| B | 0.03967 | 0.06956 | 9000 | 0.06793 | — | 0 |
| C | 0.03888 | 0.07239 | 11487 | 0.07239 | 0.05032 | 367584 |

Train 마지막 행은 공통 BC sampling schedule의 동일한 마지막 mini-batch다. 조건 간 fit 비교에는 동일 Validation 지표와 phase 진단도 함께 사용한다. Offline loss 감소는 Closed-loop 성공을 대신하지 않는다.

| 조건 / 동일 train-only probe | Peg 위치 mm | Hole 위치 mm | Tip–Hole mm | Peg quaternion deg | Hole quaternion deg |
|---|---:|---:|---:|---:|---:|
| A | 14.66 | 51.61 | 56.69 | 4.225 | 8.748 |
| B | 17.27 | 56.59 | 61.18 | 5.151 | 9.225 |
| C | 16.39 | 55.7 | 61.3 | 4.709 | 9.426 |

Geometry 표는 이미 관찰했던 F1 Final의 진단이다. 새 Blind Closed-loop Final과 구별하며, quaternion 대칭의 식별 가능성을 보장하지 않는다.

| 조건 | 물체 | 실제 이동 mm | valid pair | Parent | Δ error mm | Parent-macro error mm | cosine | signed gain | magnitude ratio | zero 대비 개선 mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | peg | 5 | 96 | 12 | 4.982 | 4.982 | 0.4794 | 0.5208 | 0.9151 | 0.01844 |
| A | peg | 10 | 96 | 12 | 8.71 | 8.71 | 0.5788 | 0.5257 | 0.8063 | 1.29 |
| A | box | 5 | 142 | 12 | 5.388 | 5.382 | 0.4392 | 0.4302 | 0.9382 | -0.3879 |
| A | box | 10 | 142 | 12 | 8.467 | 8.499 | 0.6388 | 0.5966 | 0.8503 | 1.533 |
| B | peg | 5 | 96 | 12 | 7.304 | 7.304 | 0.3437 | 0.5086 | 1.277 | -2.304 |
| B | peg | 10 | 96 | 12 | 11.68 | 11.68 | 0.3752 | 0.4301 | 0.9759 | -1.675 |
| B | box | 5 | 142 | 12 | 6.456 | 6.451 | 0.2885 | 0.2958 | 0.9942 | -1.456 |
| B | box | 10 | 142 | 12 | 9.822 | 9.85 | 0.5236 | 0.4254 | 0.7654 | 0.1777 |
| C | peg | 5 | 96 | 12 | 6.835 | 6.835 | 0.3883 | 0.5595 | 1.24 | -1.835 |
| C | peg | 10 | 96 | 12 | 11.28 | 11.28 | 0.4861 | 0.5239 | 1.05 | -1.283 |
| C | box | 5 | 142 | 12 | 6.135 | 6.124 | 0.3637 | 0.3821 | 1.037 | -1.135 |
| C | box | 10 | 142 | 12 | 9.126 | 9.158 | 0.6471 | 0.5815 | 0.88 | 0.874 |

Pair 수는 독립 표본 수가 아니다. cosine이 정의되지 않는 zero response는 평균에서 제외하고 원본 CSV의 빈 값으로 보존했다.

| 조건 | 실제 Action 개입 eligible / 전체 | 독립 Parent | Controller response error mm | cosine | signed gain | zero 대비 개선 mm |
|---|---:|---:|---:|---:|---:|---:|
| A | 16 / 16 | 2 | 0.8408 | 0.2933 | 0.1497 | -0.1503 |
| B | 16 / 16 | 2 | 1.01 | 0.2499 | 0.3064 | -0.3198 |
| C | 16 / 16 | 2 | 0.9394 | 0.201 | 0.3643 | -0.2489 |

이 표의 mm는 controller translation command의 변화다. 위 실제 물체 이동 mm와 같은 물리량으로 해석하지 않는다. 같은 Robot76·previous hidden·causal prefix와 baseline repeat/물리/Expert 유효성 검사를 통과한 행만 집계했다.

| 조건 | Validation phase | frame | Controller translation error mm | Rotation component error rad | Direction reversal rate |
|---|---|---:|---:|---:|---:|
| A | reach | 388 | 1.6 | 0.006871 | 0.002577 |
| A | descend | 158 | 1.501 | 0.002117 | 0 |
| A | close | 96 | 2.294 | 0.00247 | 0 |
| A | lift | 137 | 2.193 | 0.006459 | 0 |
| A | align | 477 | 2.372 | 0.00553 | 0.03774 |
| A | insert | 221 | 2.772 | 0.002542 | 0 |
| A | expert_prefix | 740 | 2.143 | 0.007147 | 0.03649 |
| A | recovery | 1064 | 2.584 | 0.005949 | 0.03195 |
| B | reach | 388 | 1.503 | 0.005441 | 0.002577 |
| B | descend | 158 | 1.175 | 0.002604 | 0 |
| B | close | 96 | 1.991 | 0.002122 | 0 |
| B | lift | 137 | 1.959 | 0.006756 | 0 |
| B | align | 477 | 2.329 | 0.004716 | 0.02935 |
| B | insert | 221 | 2.709 | 0.002698 | 0 |
| B | expert_prefix | 740 | 2.295 | 0.004598 | 0.05 |
| B | recovery | 1064 | 2.518 | 0.006309 | 0.03195 |
| C | reach | 388 | 1.519 | 0.006043 | 0.002577 |
| C | descend | 158 | 1.215 | 0.001857 | 0 |
| C | close | 96 | 1.833 | 0.002504 | 0 |
| C | lift | 137 | 2.134 | 0.006895 | 0 |
| C | align | 477 | 2.53 | 0.005392 | 0.03564 |
| C | insert | 221 | 2.632 | 0.002799 | 0 |
| C | expert_prefix | 740 | 2.405 | 0.005534 | 0.03649 |
| C | recovery | 1064 | 2.547 | 0.005724 | 0.03383 |

Phase 평균은 유효 BC frame 기준이며 branch가 같은 Parent를 공유할 수 있다. 이 frame 수를 독립 Parent 수로 간주하지 않는다. Reversal은 두 translation vector가 정의되고 dot<0인 frame의 비율이다.

| 조건 | Final 초기 Action7 MAE | 조기 Close step 평균 | Saturation step 평균 | Object-missing step 평균 | 마지막 TCP–Peg mm | 마지막 Tip–Hole mm |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.1946 | 217.5 | 164.4 | 71.92 | 401.6 | 378 |
| B | 0.1495 | 152.8 | 153.2 | 76.33 | 188.5 | 362.2 |
| C | 0.141 | 166.5 | 144.6 | 78.17 | 297.5 | 359.7 |

Hidden carry/reset 검사는 통과했지만 실제 실패 원인을 단독으로 배제하거나 확정하는 인과 실험은 아니다. Missing geometry, phase fit 오차와 누적 제어 오차는 별도 근거이며 Geometry loss의 개선만으로 Action Correction을 입증하지 않는다.

| 조건 | End-to-end inference s | Policy s | CUDA encoder+readout s |
|---|---:|---:|---:|
| A | 0.8198 | 0.004673 | 0.002955 |
| B | 0.8665 | 0.004643 | 0.002927 |
| C | 0.8119 | 0.004658 | 0.00294 |

동일 첫 Final Parent98200의 영상(성과에 따른 선택 없음): [A](../../outputs/task10g/final_results/rollouts/final/A/step009000/seed98200/rollout.mp4), [B](../../outputs/task10g/final_results/rollouts/final/B/step009000/seed98200/rollout.mp4), [C](../../outputs/task10g/final_results/rollouts/final/C/step011487/seed98200/rollout.mp4)


## 로컬 전달 및 독립 검증

원격 연구 프로세스 종료 후 SSH로 Task10-G 산출물을 회수했다. 35522개 index 파일의 SHA/크기를 확인하고, Development72회와 Final36회의 trace 및 MP4 첫/마지막 프레임을 읽었다. 공통 BC 노출, 선택 규칙, Best/Last 체크포인트와 독립 geometry probe도 별도로 검사했다. 원격 원문 및 hash manifest는 그대로 보존했다. 원문의 외부 백업 미설정 설명은 원격 pipeline 자체의 설정이며, 이 절은 별도 SSH 수집 결과다. F1 원본 RGB-D/Graph/GT 데이터와 shared host mmap 전체는 이 전달의 범위에 포함하지 않았다.

[독립 검증 기록](../../outputs/task10g/independent_completion_audit.json)
