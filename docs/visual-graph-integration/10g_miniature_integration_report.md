# Task10-G — F1-initialized Graph–Recurrent A/B/C

**ENGINEERING MINIATURE — 연구 결과로 사용하지 않음.**

## 실제 실행과 Identity

- Root: `/workspace/task10g_f1_abc_20261031/mini_v5`
- Instance57987772 / RTX3090; CPU PhysX + CUDA renderer/vision/policy. GPU PhysX와 로컬 macOS CUDA는 미실행.
- F1 SHA256: `781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6`
- Dataset manifest SHA256: `f1b61e7635be3a9b1533d5158e8be8df869edf67dddee926335c68486d5cf6b1`
- Optimizer updates: 각 4; seed42 한 개. BC sampling schedule, train-only ActionCodec, batch2/sequence32 공통.
- 선택 병렬도: 3; 실제 3 CUDA process 동시 benchmark를 실행했다. 전체 memory/throughput은 [기록](../../outputs/task10g/miniature_v5_evidence/concurrency_benchmark.json)에 보존.
- 단일 geometry λ: 0.01. Train-only gradient median의25% rule/clamp/fallback은 [calibration](../../outputs/task10g/miniature_v5_evidence/lambda_calibration.json)에 보존.
- 기존 ExperimentLogger의 실행 UUID와 로컬 기록을 재사용했다. Git commit, 외부 publishing, 새 유료 인스턴스 생성 없음.

## Functional Closed-loop Final

| Condition | Parents | Grasp | Lift | Insertion | Selected step |
|---|---:|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 0 | 4 |
| B | 1 | 0 | 0 | 0 | 4 |
| C | 1 | 0 | 0 | 0 | 4 |

Final seeds: `[98190]`. Zero offsets. 최대 2 Action, 공식 success/환경 종료에서 종료.
Development는 고정 Validation Parent에서만 실행했다. Insertion→Lift→Grasp→Alignment→최종 TCP–Peg error→Validation BC→earliest step 순으로 선택하고 Final 전에 checkpoint SHA를 고정했다.
각 조건은 독립 causal vision memory/GRU reset을 사용한다. 같은 seed의 초기 simulator/controller/RNG snapshot과 최초 graph/Robot76 hash가 정확히 일치했다. PhysX의 숨은 solver 상태 전체 직렬화나 모든 장치의 bit-exact trajectory를 주장하지 않는다.
[전체 Final](../../outputs/task10g/miniature_v5_evidence/final_rollouts.csv) · [선택 동결](../../outputs/task10g/miniature_v5_evidence/model_selection_frozen.json) · [Paired 분석](../../outputs/task10g/miniature_v5_evidence/paired_success_analysis.csv)

Strict Reach는 기존 Task10D의 ordered approach/final pose/open-gripper/early-close/peg-motion/collision 계약으로 별도 계산했다. 실제 Insertion 성공보다 우선하지 않는다. End-to-end inference, 전체 policy, CUDA encoder+readout 시간은 각각 다른 열로 기록했다.
![Functional success](../../outputs/task10g/miniature_v5_evidence/figures/functional_success.png)

## Geometry와 Action 반응

Geometry는 이미 관찰했던 F1 Final91000–91011의 재사용 진단이다. 새 Closed-loop Blind Final과 구별한다. A는 F1 head, B는 동결 F1 head 진단, C는 공동 학습 head를 기록했다. 주 비교는 세 encoder에 동일 F1 초기 head·동일 train-only500-update budget을 적용한 독립 probe다(Engineering miniature는3-update). B의 동결 head 오차만으로 representation 열화를 단정하지 않는다.
실제 물체 Pair의 ±5/10mm Δ vector error/direction/signed gain/magnitude ratio/zero-response 차이를 기록했다. Action 분석은 기존 Task10D의 실제 Peg 개입과 Expert counterfactual 코드로 같은 Robot76, previous hidden, causal prefix를 고정한다. 물체 이동 mm와 controller command response mm는 별도 물리량이다. Baseline repeat/물리/Expert 계약을 통과한 Pair만 주 해석에 쓴다.
[Geometry metrics](../../outputs/task10g/miniature_v5_evidence/geometry_metrics.csv) · [Actual displacement](../../outputs/task10g/miniature_v5_evidence/paired_displacement_metrics.csv) · [Action correction](../../outputs/task10g/miniature_v5_evidence/action_correction_metrics.csv)

![Geometry](../../outputs/task10g/miniature_v5_evidence/figures/geometry_errors.png)

![Action response](../../outputs/task10g/miniature_v5_evidence/figures/action_response.png)

## 학습과 실패 분석

A의 encoder/readout은 동결, B는 BC joint, C는 BC+Geometry joint다. C State Head는 Action/GRU 입력에 포함되지 않고 추론에서 실행하지 않는다. 동일 GNN 및 Projection/GRU/Action 초기 hash, lambda0 B/C gradient 일치, GT sidecar 차단 positive control, geometry gradient 전달과 실제 checkpoint 다음 update resume를 검증했다.
정상 Expert prefix는 승인 nominal expert action/robot/timestep의 정확한 비교 후 BC를 활성화했다. 교란 구간은 실제 관측을 history에 포함하며 BC를 끈다. Recovery는 유효 expert labels만 사용했다. 같은 Parent의 branch를 독립 Parent로 세지 않는다.
[Expert phase별 controller 오차](../../outputs/task10g/miniature_v5_evidence/phasewise_action_errors.csv)와 [Final 실패 구간](../../outputs/task10g/miniature_v5_evidence/failure_analysis.csv)에 초기 action fit, direction reversal, premature close, saturation, missing geometry, collision, error accumulation을 기록했다. Hidden carry/reset은 계약 시험과 trace로 검사했고 특정 실패 원인을 증명했다고 주장하지 않는다. Recovery 추가 rollout은 NOT_TESTED다.

![Train/validation](../../outputs/task10g/miniature_v5_evidence/figures/training_validation_curves.png)

## 연구 질문 판정

| Question | Verdict | Evidence scope |
|---|---|---|
| B−A: 공동 BC 업데이트의 실제 제어 이점 | NOT_TESTED | paired outcome; 한 training seed/12 Parent |
| C−B: Geometry auxiliary 추가 효과 | NOT_TESTED | 성공률·geometry·action response를 함께 해석 |
| C−A: 전체 결합 효과 | NOT_TESTED | 제한된 paired 표본; 일반적 우월성 주장 없음 |

모든 functional metric이0이어도 사전 등록한 평가·진단을 끝냈다. Offline BC/geometry loss 감소는 Closed-loop success 또는 정확한 action correction의 증거를 대신하지 않는다. 추가 λ 탐색/학습/데이터 수집은 실행하지 않았다.

## 운영·재현성과 보존

각 stage는 atomic manifest와 입력/출력 SHA, 실제 step/episode 수, 종료코드, traceback을 남긴다. Lock과 자식 process group 관리로 중복 실행을 막고, OOM은 동일 batch/objective로 마지막 검증 checkpoint에서만 제한적으로 재개한다. 정상 worker 산출물은 유지한다. tmux 프로세스는 SSH 종료와 독립적이며 인스턴스 삭제/재생성 복구는 보장하지 않는다.
원본 F1 checkpoint/dataset/source archive의 SHA를 종료 시 재검사했다. Scratch는 인스턴스 파일시스템이며 삭제 시 보존되지 않을 수 있다. 자동 외부 백업/전달 연결은 설정하지 않았으므로 전송 성공을 주장하지 않는다. 보고서 full content는 실행 source의 docs/visual-graph-integration에도 복사한다.
[Verification](../../outputs/task10g/miniature_v5_evidence/verification.json) · [Artifact index](../../outputs/task10g/miniature_v5_evidence/artifact_index.json) · [Completion audit](../../outputs/task10g/miniature_v5_evidence/completion_audit.json)

- B-A / insertion: gained0, lost0, difference0/1; exact discordant p=1.
- B-A / lift: gained0, lost0, difference0/1; exact discordant p=1.
- B-A / grasp: gained0, lost0, difference0/1; exact discordant p=1.
- B-A / alignment: gained0, lost0, difference0/1; exact discordant p=1.
- C-B / insertion: gained0, lost0, difference0/1; exact discordant p=1.
- C-B / lift: gained0, lost0, difference0/1; exact discordant p=1.
- C-B / grasp: gained0, lost0, difference0/1; exact discordant p=1.
- C-B / alignment: gained0, lost0, difference0/1; exact discordant p=1.
- C-A / insertion: gained0, lost0, difference0/1; exact discordant p=1.
- C-A / lift: gained0, lost0, difference0/1; exact discordant p=1.
- C-A / grasp: gained0, lost0, difference0/1; exact discordant p=1.
- C-A / alignment: gained0, lost0, difference0/1; exact discordant p=1.