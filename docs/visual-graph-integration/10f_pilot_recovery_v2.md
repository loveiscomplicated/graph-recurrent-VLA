# Task 10-F — Corrective Reach Pilot v4 and Recurrent BC Continuation

## 범위와 원자료

- 이전 실패 Pilot UUID `706f62ce213247e7973102962858f95b`의 raw H5 6개와 기존 보고서는 수정하지 않았다. 이 실행은 새 UUID `59eabfee0ba144cba22a96c53950e1a7`의 matched-duration 재검증을 포함한다.
- 고정 계획 SHA256 `9aefac658d9609bff9cbe30c19a86fbd2a8dd0477be1dee8cb9185d2d5011972`. 인스턴스 57640710의 NVIDIA GeForce RTX 3090, PyTorch 2.8.0+cu128, CUDA 12.8, CPU quota 192000 100000에서 CPU PhysX·GPU renderer, CUDA DINO/SAM2, CUDA E0 GNN+GRU를 실행했다.
- 실행 simulator 버전 ManiSkill 3.0.1, SAPIEN 3.0.3. 로컬 저장소의 Geometry 계약 70개와 공통 Logger/Artifact 계약 42개는 PASS했다. CUDA/renderer/physics 검증은 57640710에서 별도로 실행했다.
- 공통 초기 가중치 Task10-D B step6000 SHA256 `19da92de69b0d9aa66aa210ea759451d0f2b98e2d01c41c587ba986fd60b2cbc`. 두 조건은 새 AdamW 상태로 독립적으로 시작했다.
- 비교 UUID `d7c26fee195e4f32a7ddb99a94877b72`, R0 `5f10e223d4de46ce8d1783078ce72c34`, R1 `afbe26f5f25847f4a80daa12b5d254e3`, Final 평가 `2bfe867c20e340aab2157c5e3ff33e8e`.

## 1. 교란과 안전 판정

- 기존 실패 Pilot은 6개 중 2개만 1mm 안에 들었다. 새 동일조건 고정 8-step 진단은 8개 중 3/8개였다. 기록된 IK 호출 실패는 없었다. World/Base 변환과 action scaling은 일치했고, 초기 운동량·추종 지연·고정 종료 시점이 관찰됐다. 각 요인의 인과 기여율은 분리하지 않았다.
- 실측 feedback 교란의 40개 ±5mm 잔차는 0.300–0.868mm (중앙값 0.650mm). 전 방향을 포함했고, 40/40개가 오차·안정·물리 안전 gate를 통과했다.
- 정상 Expert Reach에서 기존 TCP z>20mm 판정은 1개 timestep을 거부했다. Nominal finger–table 최소 9.897mm, finger–Peg 최소 2.903mm였다. v2는 실제 collision shape, 5개 physics substep contact, 관절 한계·물체 이동을 검사한다. 보수적 형상 간격/접촉 표본 검사이며 연속 시간 swept collision의 수학적 증명은 아니다.
- v4 P/N 40쌍에서 N hold의 최대 anchor 위치 오차는 0.588mm. 새 P raw 배열은 기존 pilot03 P raw와 40/40개 전 항목 동일하다. 과거 N hold 오차를 소급 PASS로 바꾸지 않았다.

## 2. Expert 보정 신호와 데이터

- v4의 P/N 모두 40/40개가 안전하게 Close 직전 Reach target으로 복귀했다. 최초 보정 명령 크기는 0.140–16.875mm이다. 실제 handoff TCP 차이 P−N을 기준으로 한 방향 cosine 중앙값은 0.945, 최솟값은 0.247이다. Y 방향 일부의 방향 신호가 약하며, 포화와 기하학적 투영의 상대 기여는 분리하지 않았다.
- P recovery의 TCP–최종 target 위치 오차는 40/40개에서 감소했다(감소량 40.8–234.0mm). 자세 오차는 31/40개에서 감소했고 나머지는 소폭 증가했지만 최종 자세 gate는 통과했다. 이 감소만으로 새 보정 신호를 주장하지 않고 위 P−N 명령 차이를 함께 사용한다.
![Paired correction direction](../../outputs/task10f_recovery/remote/figures/correction_direction_measured.png)

- 본 수집은 정상 Train96/Development4, corrective Train80/Development32를 고정했다. 정상100/100개가 원래 Reach 기준을 통과했고 최종 위치/자세 오차 최대는 2.564mm / 0.0037rad였다. 10mm 추가40/40개는 목표·안전을 통과했고 3D 잔차는 0.487–0.790mm였다. Development32도 실패 방향을 제외하지 않고 기록했다. 같은 Parent는 Train/Development를 공유하지 않는다.
- 추가 10mm의 P 수집은 N hold v4 변경 전에 시작했으며 P 알고리즘은 동일하다. 해당 N hold는 후행 전수 감사에서 최대 0.481mm였다. Development N은 v4 P-only hold를 사용했다. N 대조 분기는 두 학습 조건의 supervised 데이터에 포함되지 않는다.
- 실제 wall-clock: v4 matched Pilot 23.3분, 본 raw 수집 151.5분, 전체 Graph cache 179.9분. Raw 수집과 cache는 중첩 실행되어 이 시간을 단순 합산하지 않는다.
- Graph cache 212 Episode, 11375 causal timestep. Node21/Edge4/Robot76·C3+A-SF cap128, 원본 Plan A와 pilot 입력/마스크/graph 동등성 PASS. Disturbance 실행 action은 label zero/mask false이고, Prefix→Disturbance→Recovery 관측은 모두 보존했다.

## 3. 동일 Checkpoint A/B

| 조건 | 업데이트 | Supervised timestep | Nominal / Recovery 노출 | Burn-in frame | 선택 step |
|---|---:|---:|---:|---:|---:|
| R0 | 3000 | 153936 | 153936 / 0 | 57357 | 0 |
| R1 | 3000 | 153936 | 76968 / 76968 | 125294 | 3000 |

각 update의 두 window는 동일한 supervised frame 수를 갖고, R1은 50:50, R0는 대응 Parent의 Nominal window로 대체했다. Prefix hidden은 현재 가중치로 매번 재계산하고 Branch 간 재사용하지 않았다. 선택은 Final 접근 전 Development의 Nominal/Recovery 균형 Method B loss로 고정했다.

학습 wall-clock R0 3.8분, R1 4.9분. Final rollout 30개의 개별 실행 시간 합계 49.5분.

![A/B learning curves](../../outputs/task10f_recovery/remote/figures/ab_learning_curves.png)

## 4. Frozen closed-loop 평가

Final seed 65000–65002, Anchor25, X+5mm/Y−10mm를 모델 선택 이전에 등록했다. 기존 Task10-D Final 61000–61005는 사용하지 않았다. Handoff는 Expert Prefix와 교란 관측을 Student GRU에 순서대로 입력했다. Autonomous는 Student가 t0부터 실행하고 t25에 교란했다. 실패한 접근도 분모에 포함했다. 성공은 elevated approach 10mm/0.05rad 선행, pre-close 4mm/0.03rad·opening≥65mm·물리 안전을 Student 3연속 step에서 요구한다. 위치·자세는 현재 Peg에서 다시 계산한 평가 전용 GT target과 비교했다.

| 평가 | R0 성공 | R1 성공 | 평균 위치 오차 mm R0/R1 | 평균 자세 오차 rad R0/R1 | 물리 안전 실패 R0/R1 |
|---|---:|---:|---:|---:|---:|
| handoff | 0/6 | 0/6 | 47.449 / 47.409 | 0.045 / 0.039 | 6 / 2 |
| autonomous | 0/6 | 0/6 | 113.595 / 79.890 | 0.122 / 0.184 | 6 / 4 |
| nominal | 0/3 | 0/3 | 146.585 / 81.025 | 0.163 / 0.024 | 3 / 1 |

실패 분류: handoff R0: 교란 미완료 0, 안전 실패 6, 접근 미도달 2, 목표 미완료 6 / handoff R1: 교란 미완료 0, 안전 실패 2, 접근 미도달 2, 목표 미완료 6 / autonomous R0: 교란 미완료 0, 안전 실패 6, 접근 미도달 6, 목표 미완료 6 / autonomous R1: 교란 미완료 0, 안전 실패 4, 접근 미도달 6, 목표 미완료 6 / nominal R0: 교란 미완료 0, 안전 실패 3, 접근 미도달 3, 목표 미완료 3 / nominal R1: 교란 미완료 0, 안전 실패 1, 접근 미도달 3, 목표 미완료 3.

교란 이후 target 오차 감소 평균 (mm) / recovery overshoot 평균 (mm) / Peg presence 평균: handoff R0 26.819 / 0.000 / 1.000, handoff R1 26.858 / 11.296 / 1.000, autonomous R0 12.080 / 107.463 / 1.000, autonomous R1 30.110 / 118.653 / 1.000. 개별 gate 상태는 rollout trace에 보존했다.

![Closed-loop error curves](../../outputs/task10f_recovery/remote/figures/closed_loop_recovery_error_curves.png)

선택 Checkpoint의 Development Expert controller translation MAE (mm): R0 Nominal 0.775, Recovery 1.166, R1 Nominal 0.751, Recovery 1.072.
Development 첫 Recovery translation 명령의 Expert 대비 평균 방향 cosine: R0 0.634, R1 0.707.
![Expert command error](../../outputs/task10f_recovery/remote/figures/expert_command_error.png)

## 5. 계산 비용과 한계

같은 51-frame cache episode에서 serial / CPU geometry worker1 / worker2 전체 처리량은 1.011, 1.459, 1.545 graph/s였다. 실제 전체 cache의 마지막 기록 worker 수는 1이다. 이는 한 Episode의 검증 benchmark이며 일반적인 배속을 주장하지 않는다.

동일 64-frame P/N Episode의 원본 대비 검증된 optimized cache wall-clock (초)은 4100000: 92.5→57.4, 4100001: 67.4→60.1였다. 새 교란 프레임에서 graph/mask/gate 배열 동등성을 확인한 뒤 optimized 경로를 채택했다.

![GPU and CPU throughput](../../outputs/task10f_recovery/remote/pilot_media/gpu_cpu_throughput.png)

- 물리 판정은 실제 shape/contact와 5개 substep 표본을 썼지만 연속 swept-volume 충돌까지 증명하지 않는다. SAM2 설치의 선택적 `_C` postprocessing은 사용되지 않았다. CPU PhysX는 유지했고 GPU PhysX는 실행하지 않았다.
- Corrective Train Parent는 5개이고 Final Parent는 3개다. 결과는 동일 Checkpoint의 3,000-update 추가학습 효과이며 Scratch 또는 Full96 corrective training의 결론이 아니다.
- 이번 고정 계획의 ±5/10mm 분기는 실패 방향을 제외하지 않았으며 모두 Expert 복귀했다. Parent5와 Expert 성공 궤적에서 출발하는 선택 범위는 남는다. 원래 Pilot 실패와 재검증 비용을 함께 고려했다.

## 6. 연구 질문별 판단

| 질문 | 판단 | 근거 |
|---|---|---|
| Q1 고정 교란 실패 원인 | SUPPORTED | IK 실패 0, 좌표·scale 일치; 8-step 추종 미수렴. 세부 기여율 미분리. |
| Q2 feedback ±5mm | CONFIRMED | 40/40 방향 포함, 3D 잔차 <1mm. |
| Q3 기존 Safety Gate | CONFIRMED | Nominal z20 거부와 v2 실제 형상/접촉 분리. |
| Q4 새 Expert Correction | SUPPORTED | 40/40 명령 차이; Y 방향 신호는 약함. |
| Q5 안전한 Expert 복귀 | CONFIRMED | P/N 40쌍 목표·물리 gate 통과. |
| Q6 causal Vision/Graph/GRU | CONFIRMED | 전 frame cache, 독립 Prefix hidden, masked loss/readback. |
| Q7 Recovery BC | INCONCLUSIVE | Handoff 0→0/6; Autonomous 0→0/6. |
| Q8 Handoff/Autonomous | INCONCLUSIVE | 두 평가를 별도 분모와 궤적으로 보고. |
| Q9 Nominal 보존 | INCONCLUSIVE | 성공 0→0/3; 표본 작음. |
| Q10 추가 데이터/Full96 | INCONCLUSIVE | 현재 교란 성공과 폐루프 결과를 함께 고려. |

## 7. 검증 및 원자료

[verification.json](../../outputs/task10f_recovery/remote/verification.json), [artifact_index.json](../../outputs/task10f_recovery/remote/artifact_index.json), [Controller command/actual trace](../../outputs/task10f_recovery/remote/diagnosis02/disturbance_controller_audit.json), [Safety protocol v2](../../outputs/task10f_recovery/remote/matched_hold_v4/safety_protocol_v2.json), [Pilot correction CSV](../../outputs/task10f_recovery/remote/matched_hold_v4/paired_expert_correction.csv), [sampling manifest](../../outputs/task10f_recovery/remote/sampling_manifest.json), [checkpoint manifest](../../outputs/task10f_recovery/remote/checkpoint_manifest.json), [handoff CSV](../../outputs/task10f_recovery/remote/final_evaluation/expert_handoff_rollouts.csv), [autonomous CSV](../../outputs/task10f_recovery/remote/final_evaluation/autonomous_recovery_rollouts.csv), [nominal CSV](../../outputs/task10f_recovery/remote/final_evaluation/nominal_reach_rollouts.csv), [Pilot MP4](../../outputs/task10f_recovery/remote/pilot_media/expert_P_30000_t10_5_0.mp4), [R0/R1 rollouts](../../outputs/task10f_recovery/remote/final_evaluation/rollouts/).

기존 실패 Pilot UUID `706f62ce213247e7973102962858f95b`와 사후 분석 UUID `f7589cee98f44d75890e1dbcc0409902`의 판정은 그대로 유지했다. 새 산출물은 별도 실험 경로에 저장했다.
