# Task 08-B — Seed 30003 single-episode A/C overfit

Comparison UUID: `60468642a39545318ebf0256727d7a08`. 실행 조건: RTX 3090, 기존 Full A/B/C와 별도 Run, A/C 20,000 Updates.

## 사전 계약과 기준

Episode 30003: 137 Actions / 138 Observations, 성공 True, HDF5 SHA256 `ba578d3e17e804c48c5490dc9970406a2b560d9ea214519094965fa815a09122`.
초기 가중치 SHA256 `386585e094c894e9f3cecc8d57c52a1e3c31f895ffeab8b88e524b0b5c4a45c5`; Robot76 Normalizer SHA256 `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`.
기존 Small5는 Seed 30000–30004, 611 Actions, 2,000 Updates에서 A Action7 MSE 0.004992/C 0.007975로 충분한 Overfit에 미달했다. 이번 결과와 Dataset 크기가 다르다.
정책은 저장된 원본 RGB-D/Predicted Mask 기반 Shape-First/PCA5/F5, Raw EDE, Robot–Object k4/Peg–Box M16, Edge-aware GNN L2/d64, Structured Readout 320D, Presence2D, Robot76, GRU144를 그대로 사용했다.
A는 Native Action7 MSE와 연속 Gripper를 사용한다. C는 Train96 Motion6 z-score MSE와 Gripper BCEWithLogits를 (6×Motion MSE+BCE)/7로 결합하고, 역정규화 및 logit>0→+1/그 외→-1로 디코딩한다.
Task08 기준: Action7 MSE≤0.001, Motion6 MSE≤0.0001, 초기 Action7 MSE 대비 ≥95% 감소, Gripper sign accuracy≥99%.
Train96 Robot76 및 Motion6 통계 재사용. A/C 동일 초기 가중치, AdamW/lr3e-4/B1×T32/clip1/seed42/CosineAnnealingLR(T_max=20000).
기존 config batch_size=2이지만 단일 Episode이므로 실제 유효 Batch는 1×T32다. 두 방식에서 완전히 같다. 저장 Packet에서 생성한 기존 Graph/Robot76/Action 텐서를 정확한 동일성 검사 후 메모리에 재사용했으며 학습 대상과 모델 계산은 바꾸지 않았다.
20,000 Update 고정. 기준을 통과한 milestone 중 Action7 MSE 최저 checkpoint를 선택하고, 없으면 20,000 최종 checkpoint를 선택했다.
기준 미달 시 15k→20k Action7 MSE 개선률 <1%를 OPTIMIZATION_PLATEAU, 그 외를 NOT_CONVERGED_WITHIN_BUDGET로 사전 정의했다.
Controller는 ManiSkill 설치 버전의 pd_ee_delta_pose: Translation ±0.1 m, Rotation 부호 반전 scale -0.1 rad 및 norm clipping, Gripper Target -0.01/+0.04 m. CPU PhysX 20Hz/100Hz, horizon 250, Stop-the-clock.
Reset: PASS_PHYSICAL_WITH_RENDER_QUANTIZATION. 물리 GT reset hash와 qpos/qvel은 정확히 일치; 렌더링/수치 허용 범위는 reset_identity.json에 기록했다.
기존 저장 Action7 재생 평가 Sidecar: 성공 t=136, Close t=49; 저장 TCP와 재생 TCP 위치 차이 최대 7.6064e-05 m. Robot76 전체는 byte-identical하지 않아 재생을 원본 상태 자체로 취급하지 않는다.

## Offline과 Closed-loop 결과

| Method | Fit | Selected update | Action7 MSE | Motion6 MSE | Translation mm mean | Rotation deg mean | Grip sign | First close Expert→Pred | Grasp | Lift 4cm | Insert |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| A | OFFLINE_FIT_SUFFICIENT | 20000 | 1.13101e-07 | 1.08095e-07 | 0.045 | 0.002 | 100.000% | 49→49 | True | True | False |
| C | OFFLINE_FIT_SUFFICIENT | 20000 | 9.13197e-08 | 1.0654e-07 | 0.022 | 0.003 | 100.000% | 49→49 | True | True | True |

A/C Optimization Loss는 다른 목표와 BCE 때문에 직접 비교하지 않는다. 위 Offline 수치는 전부 디코딩한 Native Action 및 설치 Controller 전처리 기준이다.
GT Pose/Contact/Expert stage는 평가와 분석 sidecar에만 사용했다. Student 정책 입력에는 포함되지 않았다.
동일 timestep의 Expert/Student 위치 차이 자체로 정책 오류를 단정하지 않는다.

## Update별 결과

### A

| Updates | Optimization Loss | Action7 MSE | Motion6 MSE | Translation mm | Rotation rad | Gripper accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 8.21499e-05 | 8.21499e-05 | 9.26185e-05 | 1.277 | 0.00057 | 100.000% |
| 5000 | 3.57929e-05 | 3.57929e-05 | 3.93321e-05 | 0.645 | 0.00024 | 100.000% |
| 10000 | 4.12115e-06 | 4.12115e-06 | 3.58954e-06 | 0.271 | 0.00025 | 100.000% |
| 15000 | 2.98355e-07 | 2.98355e-07 | 2.67635e-07 | 0.083 | 0.00006 | 100.000% |
| 20000 | 1.13101e-07 | 1.13101e-07 | 1.08095e-07 | 0.045 | 0.00004 | 100.000% |

15k→20k Action7 MSE 개선률 62.09%; 판정 SUFFICIENT.

### C

| Updates | Optimization Loss | Action7 MSE | Motion6 MSE | Translation mm | Rotation rad | Gripper accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 0.00274374 | 4.50898e-05 | 5.26048e-05 | 0.453 | 0.00100 | 100.000% |
| 5000 | 0.000104925 | 2.29937e-06 | 2.6826e-06 | 0.123 | 0.00024 | 100.000% |
| 10000 | 3.49062e-05 | 1.04403e-06 | 1.21804e-06 | 0.080 | 0.00020 | 100.000% |
| 15000 | 5.73596e-06 | 1.30029e-07 | 1.517e-07 | 0.028 | 0.00006 | 100.000% |
| 20000 | 2.90826e-06 | 9.13197e-08 | 1.0654e-07 | 0.022 | 0.00005 | 100.000% |

15k→20k Action7 MSE 개선률 29.77%; 판정 SUFFICIENT.

## Offline 선택 Checkpoint와 최종 Update 비교

| Method | Selected update | Selected Action7 MSE | 20k Action7 MSE | Selected translation mm mean/p95/max | Selected rotation rad mean/p95/max | Gripper target mm mean/p95/max |
| --- | ---: | ---: | ---: | --- | --- | --- |
| A | 20000 | 1.13101e-07 | 1.13101e-07 | 0.0447/0.1256/0.2856 | 3.67767e-05/0.000100652/0.000127172 | 0.0034/0.0161/0.0312 |
| C | 20000 | 9.13197e-08 | 9.13197e-08 | 0.0217/0.0715/0.2228 | 5.02405e-05/0.000168063/0.000221785 | 0.0000/0.0000/0.0000 |

## Expert 단계별 Offline Action7 MSE

| Stage | A | C |
| --- | ---: | ---: |
| reach | 3.69451e-08 | 1.16359e-07 |
| descend | 2.58564e-08 | 1.25295e-08 |
| close | 1.32965e-07 | 1.08029e-07 |
| lift | 2.57664e-08 | 2.40011e-08 |
| align | 1.96648e-07 | 1.16448e-07 |
| insert | 1.35565e-07 | 5.56659e-08 |

## Same-seed 실제 조작

### A

250-step horizon 중 250 step 실행; 파지 True, 양손가락 접촉 True, 최대 Peg Lift 0.0846 m, 4cm Lift 연속 16 step, 최종 Peg Lift -0.0000 m, 삽입 False.
최초 Close 51; Close 직전 TCP–Peg {'timestep': 51, 'tcp_peg_vector_m': [0.0028590112924575806, -0.05383867025375366, -0.007225764915347099], 'tcp_peg_distance_m': 0.05439658113222448, 'fingertip_peg_distances_m': [0.0699422065432083, 0.07011195955848168]}; Controller 입력 일치 True.
Rollout artifacts: `/workspace/task08b/experiments/f359b53275b84753b67918965dfc7a1b`. 실제 after-action RGB: `/workspace/task08b/experiments/f359b53275b84753b67918965dfc7a1b/episodes/train_seed_30003/after_action_rgb.mp4`.

### C

250-step horizon 중 137 step 실행; 파지 True, 양손가락 접촉 True, 최대 Peg Lift 0.0815 m, 4cm Lift 연속 74 step, 최종 Peg Lift 0.0627 m, 삽입 True.
최초 Close 49; Close 직전 TCP–Peg {'timestep': 49, 'tcp_peg_vector_m': [0.0032783448696136475, -0.05481766164302826, -0.0005713142454624176], 'tcp_peg_distance_m': 0.054918575846982065, 'fingertip_peg_distances_m': [0.07277353775283894, 0.07255667653415698]}; Controller 입력 일치 True.
Rollout artifacts: `/workspace/task08b/experiments/221d95556a2540588a40d609b86d52b8`. 실제 after-action RGB: `/workspace/task08b/experiments/221d95556a2540588a40d609b86d52b8/episodes/train_seed_30003/after_action_rgb.mp4`.

## Q1–Q10 판정

- **Q1 — CONFIRMED**: OFFLINE_FIT_SUFFICIENT
- **Q2 — CONFIRMED**: OFFLINE_FIT_SUFFICIENT
- **Q3 — CONFIRMED**: {'A': {'translation_mm': {'mean': 0.04472612234106097, 'median': 0.031036676838994026, 'p90': 0.10183787494897843, 'p95': 0.12555643618106826, 'p99': 0.24372909605503043, 'max': 0.28555604815483093}, 'rotation_rad': {'mean': 3.677673698139098e-05, 'median': 2.9032263207350835e-05, 'p90': 7.712202368326523e-05, 'p95': 0.00010065239816288432, 'p99': 0.00012021506319791752, 'max': 0.00012717181994561377}}, 'C': {'translation_mm': {'mean': 0.021742161209346644, 'median': 0.01659541204571724, 'p90': 0.039603987336158754, 'p95': 0.07151461243629453, 'p99': 0.13113509207963914, 'max': 0.22281605005264282}, 'rotation_rad': {'mean': 5.024051767935447e-05, 'median': 3.1173138591714766e-05, 'p90': 0.00011460276219536998, 'p95': 0.00016806284479599525, 'p99': 0.00021249656032221065, 'max': 0.00022178529267617605}}}
- **Q4 — CONFIRMED**: {'A': {'first_close': 49, 'expert_first_close': 49, 'transition_timesteps': [49], 'close_segment_accuracy': 1.0}, 'C': {'first_close': 49, 'expert_first_close': 49, 'transition_timesteps': [49], 'close_segment_accuracy': 1.0}}
- **Q5 — CONFIRMED**: {'A': {'at_2k': 8.214992262884855e-05, 'at_20k': 1.1310141940654175e-07, 'trend': 'SUFFICIENT'}, 'C': {'at_2k': 4.508984579479654e-05, 'at_20k': 9.131974819685464e-08, 'trend': 'SUFFICIENT'}}
- **Q6 — CONFIRMED**: {'A': {'success': False, 'grasp': True, 'lifted_4cm': True, 'episode_length': 250}, 'C': {'success': True, 'grasp': True, 'lifted_4cm': True, 'episode_length': 137}}
- **Q7 — SUPPORTED**: {'A': 'Lift/hold during alignment: first open command at autonomous t=79 (native grip=+0.151); grasp ended at autonomous t=79 and was absent at t=80, before Expert insert stage begins at t=122; final Peg lift -0.0000 m.'}
- **Q8 — SUPPORTED**: Single-episode basic BC fit is established for A and C. A loses grasp after lift during autonomous alignment, while C inserts successfully; investigate A closed-loop state departure and method-specific action errors. Held-out generalization remains untested.
- **Q9 — SUPPORTED**: A/C observed differences justify a controlled loss-design follow-up, but one same-seed episode does not select a general method. Motion z-score and gripper BCE change together; a separate Method B comparison is needed to isolate their effects.
- **Q10 — SUPPORTED**: At least one model fits and executes this episode

## 보존 및 한계

시작/종료 보존 감사: PASS (72835 files). 원본 Dataset, 기존 Checkpoint, Full Run은 수정하지 않았다.
시각화: `/workspace/task08b/experiments/60468642a39545318ebf0256727d7a08/single_episode_ac_overfit.html`; Expert/A/C after-action 비교 영상 `/workspace/task08b/experiments/60468642a39545318ebf0256727d7a08/expert_vs_A_vs_C_after_action.mp4`. 같은 timestep의 서로 다른 자율 상태는 행동 오류의 직접 증거가 아니다.
A/C는 Motion z-score와 Gripper BCE가 동시에 다르므로 개별 원인은 분리되지 않는다. 필요하면 별도 Method B가 필요하다.
단일 Episode 성공은 Graph 정보 활용 또는 Held-out 일반화를 증명하지 않는다. CUDA 학습은 본 RTX3090에서 실행했고, simulator는 CPU PhysX stop-the-clock로 실행했다.
실시간 제어 마감시간, CUDA PhysX, MPS는 이 실험에서 검증하지 않았다. 외부 업로드와 Git Commit은 수행하지 않았다.
