# Task 08 — BC 학습 및 자율 Rollout 중간 결과

**기준 시점:** 2026-10-26 16:54 KST (07:54 UTC). Full 학습과 최종 자율 Rollout은 진행 중이다. 이 문서는 중간 결과이며 최종 성능 보고서가 아니다.

## 연구 질문과 통제 조건

기존 Graph–GNN–GRU가 소수 Expert Episode를 충분히 모방할 수 있는지, 그리고 모방한 초기 상태에서 실제 자율 조작을 수행하는지 확인한다. 이어서 Action 학습 방식만 바꿔 Motion6 z-score와 Gripper BCE의 효과를 분리한다. Graph/GNN/GRU 구조, Robot76, Causal Vision Graph, Frozen Split, Expert Action, ManiSkill `pd_ee_delta_pose` Controller는 동일하게 유지한다.

| 방법 | Motion6 학습 | Gripper 학습 | 추론 시 Controller에 전달하는 값 |
| --- | --- | --- | --- |
| A | 원래 Action의 MSE | 연속값 MSE | Native Action7 |
| B | Train Action의 z-score MSE | 연속값 MSE | Motion6 역정규화 + 연속 Gripper |
| C | B와 동일 | 열기/닫기 BCEWithLogits | Motion6 역정규화 + logit 임계값으로 ±1 Gripper |

A↔B는 Motion 정규화, B↔C는 Gripper BCE의 추가 효과를 비교한다. Optimization Loss는 방식 간 직접 비교하지 않는다. 공통 offline 지표는 출력 decode 후 Native Action과 실제 Controller 명령 공간에서 계산한다.

## Small5 Overfit — A/C 완료

Train96 중 미리 정한 완전한 Expert Episode 5개(Seed 30000–30004, Action 611개)를 사용했다. A와 C가 같은 초기 모델 가중치, Episode·Batch 순서, 학습 Seed 및 2,000 optimizer update 예산으로 학습했다. **Small B는 실행하지 않았다.**

| 학습에 사용한 Episode의 offline 지표, 2,000 updates | A | C |
| --- | ---: | ---: |
| Native Action7 MSE | **0.004992** | 0.007975 |
| Native Motion6 MSE | 0.003943 | **0.003848** |
| Controller 위치 명령 평균 오차 | 3.02 mm | **2.05 mm** |
| Gripper 정확도 | **99.67%** | 99.18% |
| 사전 정의한 충분한 Overfit 기준 | 미달 | 미달 |

두 모델 모두 학습 시작보다 Native Action7 오차가 95% 이상 줄었고 Checkpoint Reload 출력이 일치했다. 하지만 사전에 고정한 절대 오차 기준에는 못 미쳐 둘 다 `OFFLINE_FIT_INSUFFICIENT`이다. C는 평균 위치 명령 오차가 작았지만 Native Action7 MSE와 Gripper 정확도는 A보다 나빴다. A/C는 Motion 정규화와 BCE가 동시에 달라 각각의 효과를 분리하지 못한다.

## Small5 Train-seed 자율 Rollout — A/C 완료

수집 당시 Reset Identity가 재현된 Seed 30003·30004만 동일 초기 상태의 Overfit Rollout으로 집계했다. Seed 30000–30002는 현재 Runtime에서 다른 초기 상태를 생성하므로 `BLOCKED_RESET_IDENTITY`로 남겼다. 평가 Horizon은 250 step이며 매 step Student Action을 실제 환경에 적용했다.

| 모델 | 재현된 초기 상태 | 삽입 성공 | `Panda.is_grasping` 파지 판정 | Peg 4cm Lift |
| --- | ---: | ---: | ---: | ---: |
| Small A | 2 | **0/2** | 0/2 | 0/2 |
| Small C | 2 | **0/2** | 1/2 | 0/2 |

유효한 네 Rollout 모두 250 step에서 시간 초과됐다. C의 Seed 30004는 43 step에 파지 판정과 양쪽 손가락 접촉이 확인됐지만, Peg를 4cm 들어 올리거나 삽입하지 못했다. 접촉·파지 판정과 실제 Lift·삽입 성공은 별개다. Reset Identity가 다른 3개 Seed를 성공이나 실패로 합산하지 않는다.

## Full A/B/C 진행 상태

Full 실험은 기존 Train96 / Validation12 / Test12와 동일한 초기 가중치, 학습 순서 및 최대 50 Epoch를 사용한다. 3개 독립 tmux 학습 프로세스가 RTX 3090에서 병렬 실행 중이다. 실제 B2×T32 동시 GPU Probe 3개가 통과했고, 당시 여유 VRAM은 20,263 MiB였다.

**기준 시점의 진행량:** A/B/C 모두 4번째 Epoch, 912 optimizer step. 세 Job 모두 실행 중이었다. Full Test Offline Metric과 Development·Final Held-out Closed-loop 성공률은 아직 나오지 않았다. 첫 Epoch의 Validation 값이나 서로 다른 Optimization Loss로 최종 순위를 판단하지 않는다.

## 현재 해석과 남은 판단

1. 모델이 Expert Action을 학습하는 것은 확인됐으나, Small5를 사전 기준까지 충분히 Overfit하지는 못했다.
2. Small에서 감소한 offline Action 오차는 같은 초기 상태의 자율 삽입 성공으로 이어지지 않았다. 평가 가능한 초기 상태가 2개뿐이므로 일반화 성공률로 확대 해석하지 않는다.
3. Motion 정규화와 Gripper BCE만으로 큰 Closed-loop 개선이 생길 것이라는 기대는 낮다. 아직 B의 Small 결과가 없고, Full의 공통 지표·자율 Rollout도 없어 효과를 단정할 수 없다.
4. Full A↔B와 B↔C의 공통 offline 지표 및 동일 Seed Rollout을 확인한 뒤, 기존 BC 학습을 더 개선할지 Recovery Data/DAgger 또는 Graph Representation 분석으로 넘어갈지 결정한다.

## 실행 기록

- 비교 Run: `/workspace/task08/experiments/3ed77d8b64884f879b9a628e34a9426a/`
- Small A 학습 Run: `/workspace/task08/experiments/2bc5701a80494a72910af0044219ce43/`
- Small C 학습 Run: `/workspace/task08/experiments/9a8cec829aaa4c2aaccc10ed91770c30/`
- Small A Rollout Run: `/workspace/task08/experiments/c8d86c671c754872bbf1bce688cea98b/`
- Small C Rollout Run: `/workspace/task08/experiments/2bc68a41cd574e04a0ab69ea951abc44/`
- 진행 상태: 비교 Run의 `status.json`; Full 진행: 각 학습 Run의 `progress.json`.

이 경로는 Vast 서버의 절대 경로다. 기존 Task 03–07 자료와 학습 Dataset은 수정하지 않았으며, 외부 업로드나 Git Commit을 수행하지 않았다.
