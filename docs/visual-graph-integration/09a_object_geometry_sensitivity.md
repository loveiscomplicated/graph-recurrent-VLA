# Task 09-A — Object Geometry Sensitivity & Graph Utilization Diagnostic

ExperimentLogger UUID: `073de7b8e59d4f23b650c87c377c1829`. Vast.ai **57640710 / RTX 3090**에서 여섯 Checkpoint를 추론하고, CPU PhysX에서 실제 Actor Pose를 변경해 새 RGB-D를 얻었다. 모든 Model은 `eval()` / `torch.inference_mode()`이며, 분기 이후 Student Action은 환경에 적용하지 않았다.

**Full 모델이 Graph를 완전히 무시한다는 설명은 지지되지 않는다. 다만 실제 Peg Geometry에 대한 행동 반응의 방향·크기는 불완전했고, 특히 t=25에서 크게 부족했다. Single B/C는 현재 Geometry 변화에 대한 정렬된 반응이 매우 작았다. 별도로, 실제 Peg 관측 누락과 가려진 Box에 대한 잘못된 Point 지원도 확인됐다.** 이 결과만으로 Full 학습 실패의 원인을 GNN–GRU Fusion, Capacity 또는 Optimization 중 하나로 확정하지 않는다.

[정적 HTML](../../experiments/073de7b8e59d4f23b650c87c377c1829/object_geometry_sensitivity.html) · [전체 조건 JSON](../../experiments/073de7b8e59d4f23b650c87c377c1829/per_condition_metrics.json) · [CSV](../../experiments/073de7b8e59d4f23b650c87c377c1829/per_condition_metrics.csv) · [Readback 검증](../../experiments/073de7b8e59d4f23b650c87c377c1829/verification.json)

## 실행 범위와 고정 Identity

Seed 30003의 t=0,10,25에서 각각 독립 Reset과 동일한 저장 Expert Prefix를 사용했다. 모든 분기 직전 실제 Expert Phase는 **reach**, Counter는 0이었다. 예상에 맞춰 시점을 변경하지 않았다.

| 조건 | 물리 상태 수 | 처리 |
|---|---:|---|
| Peg X/Y ±5/±10 mm Primary | 24 | 23개 Geometry 지원, 1개 관측 누락 |
| 추가 Peg X/Y ±20 mm, World Z ±5° | 18 | 17개 Geometry 지원, 1개 관측 누락; 별도 집계 |
| Box X/Y ±10 mm Negative Control | 12 | 8개 관측 가능, t=25의 4개는 완전 가림 |
| Baseline / 독립 Reset Baseline 반복 | 6 | 세 시점 반복 대조 모두 통과 |
| 합계 | **60** | 360개 Model/상태 출력 보존 |

추가 후보는 세 시점 모두 Primary Expert의 Translation 명령이 자체 최대 이동량으로 포화되어 사전 등록한 조건에 따라 실행했다. Primary의 Expert 변화가 없었다는 뜻은 아니다. [고정 Protocol](../../experiments/073de7b8e59d4f23b650c87c377c1829/protocol.json), [추가 조건 실행 근거](../../experiments/073de7b8e59d4f23b650c87c377c1829/additional_decisions.json)를 참조한다.

| Checkpoint | 선택 시점 | SHA256 |
|---|---|---|
| Full A | Epoch 35 | `409a9456740193c3a46337c4e172dac7d64dc22c45250c365d3879d4fb337b5e` |
| Full B | Epoch 48 | `f094f3cc85f8013c37e70d6043a7035071f5733b7db04371207c31fbce7b4e02` |
| Full C | Epoch 38 | `e615ec3ded96b0242004e08ee13273b782ba4af1032c8a84f205144f4ab8d177` |
| Single A | Update 20000 | `5a94542e6d1f973d052fce9598e92a3be5b8c558974cc4e761f17e087a013f4d` |
| Single B | Update 20000 | `72473e581c6c0dfdfa566a26fcf7d73619f310470a6e983a05a69db06cfd8899` |
| Single C | Update 20000 | `1f81b559b5a95fbd2805b952850b896993eda4be2703f2f46a8f43a569c0ecc5` |

정확한 원격 경로와 원본 Receipt/Plan/초기 가중치/Codec Identity는 [identities.json](../../experiments/073de7b8e59d4f23b650c87c377c1829/identities.json)에 있다. Train96 Robot76 Normalizer는 `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`, B/C Motion6 통계 Identity는 `9a475b4a51779a913db563b84495ff6f2189f09013c3e2193863702b352b782c`다. Checkpoint Loader와 기존 A/B/C Decoder를 그대로 재사용했다.

Shape-First/PCA5/F5, Raw EDE, k4/M16, GNN L2/d64, Readout320, Presence2, Robot76, GRU144 및 `pd_ee_delta_pose`는 변경하지 않았다. 실행된 공통 추론 의존 Source 13개는 Task08 실행본과 SHA가 같다. [의존 Source 감사](../../experiments/073de7b8e59d4f23b650c87c377c1829/inference_dependency_identity.json).

## 물리·History·Expert Reference 유효성

설치된 ManiSkill 3.0.1 / SAPIEN 3.0.3의 `Actor.set_pose()`를 사용했다. Pose 변경 뒤 `get_obs()`로 다시 렌더링했으며 `PhysicsAudit`에서 추가 `scene.step`이 없음을 확인했다. Box Hole/Goal은 현재 `box.pose`에서 계산되는 Property였다. [설치 API Source](../../experiments/073de7b8e59d4f23b650c87c377c1829/installed_api_source.txt).

각 개입 전후 Robot qpos/qvel, TCP/Finger/Fingertip, Robot76, Camera Calibration, 비개입 물체의 Pose/Velocity가 정확히 같았다. 독립 Trial 사이에도 Robot76과 각 Model의 Previous Hidden은 정확히 같았다. Tracker와 Hidden은 독립 Reset부터 실제 Prefix를 다시 처리해 구축했으며 다른 Trial에서 복사하지 않았다. 출력 Hidden은 다른 조건으로 이식하지 않았다.

60개 모두 실제 충돌 메시 SAT, 테이블 XY 범위, 기존 접촉 근접 Pair와의 비교를 통과했다. Geometry 허용오차는 0.2 mm였다. **Teleport 뒤 PhysX 접촉 Force를 새로 계산했다고 주장하지 않는다.** Physics Step 없이 얻는 접촉 Cache는 오래된 값일 수 있어, 메시 교차와 새로운 근접 접촉 여부를 독립적으로 검사했다. 분기 이후 동적 안정성은 평가하지 않았다.

현재 Gymnasium은 과거 실행의 1.3.0과 달리 1.4.0이었다. 설치나 Downgrade 없이 별도 진단 Adapter에서 이 차이를 기록했다. 원래 Reset SHA는 일치했고, 현재 t=0/10/25의 qpos/qvel·TCP·양 Finger·Peg·Box Pose는 **Task08-F의 현재 런타임 Expert Replay 기준과 모두 정확히 일치**했다. Controller 설정과 실제 5 Physics Substep/Action도 검증했다. [이전 Prefix 대조](../../experiments/073de7b8e59d4f23b650c87c377c1829/historical_prefix_audit.json).

Expert는 변경된 상태에서 별도 `ScriptedExpert`를 생성해 Grasp Target을 새로 계산했다. 기준 Phase/Counter를 유지했고, Translation 조건의 Target 이동 정합과 독립 Servo Translation 계산을 검증했다. 60개 Reference가 통과했다. 이는 Stage를 고정한 Counterfactual Reference이며, 새 배치에서 처음부터 실행한 Demonstration과 동일하다는 주장은 아니다.

Controller Decoded Translation 차이 ≤0.1 mm는 `NON_DISCRIMINATIVE_REFERENCE`, Model 변화 ≤0.001 mm는 방향 Cosine 계산에서 제외하도록 사전 고정했다. 모든 Peg Perturbation의 Expert Translation 변화는 분별 기준을 넘었다. Baseline/반복/Box-only의 Expert 변화는 0이며 방향 정렬도에 포함하지 않았다.

## Perception/Input Representation 문제 — Case A

Primary의 **t=0 Peg Y+5 mm**에서는 GT Visible Peg Pixel이 **850개**인데 예측 Mask와 Peg Point가 **0개**였다. 추가 조건 **t=0 Peg Y+20 mm**도 Visible Pixel 882개에 예측 Mask/Point 0개였다. 이 두 조건은 물리적으로 유효하고 Expert Reference도 유효하지만, 올바른 Peg Geometry 전달을 검증할 수 없다. Presence 소실에 따른 큰 Action 변화까지 원자료에 남겼으며, 이를 유용한 Geometry 반응 또는 Graph 무시의 근거로 집계하지 않았다.

나머지 Primary 23개는 Baseline/변경 상태 모두 선택 Point의 GT Peg Purity ≥50%, 요청 Translation 방향의 관측 Centroid 변화 ≥1 mm라는 사전 고정 지원 기준을 통과했다. GT Mask/Pose는 이 감사에만 사용했고 Student Graph에는 들어가지 않았다. [입력 Geometry 감사](../../experiments/073de7b8e59d4f23b650c87c377c1829/input_geometry_metrics.json).

![Primary 실제 RGB 및 예측 Mask](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/primary_observations_t00.png)

## Primary: Geometry → Readout → Action → Expert 정렬

아래는 유효한 **23개 물리 Pair**의 기술통계다. 23개를 독립 Episode 표본으로 취급하지 않는다. Expert Translation 변화 평균은 **0.746397 mm**였다. Readout은 각 Checkpoint의 Baseline을 분모로 한 Relative L2다.

| Model | Readout 변화 % | Model 이동 변화 mm | 방향 Cosine | 크기 비율 | Signed Gain | Response 오차 mm | 무반응보다 Response 오차가 작은 Pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full A | 5.333 | 0.374308 | 0.093268 | 0.796498 | 0.054952 | 0.905615 | 10/23 |
| Full B | 6.682 | 0.156326 | 0.395457 | 0.325992 | 0.151466 | 0.698563 | 17/23 |
| Full C | 6.565 | 0.220025 | 0.363145 | 0.461623 | 0.187865 | 0.734754 | 16/23 |
| Single A | 1.322 | 0.095681 | 0.233987 | 0.187906 | 0.009976 | 0.748530 | 14/23 |
| Single B | 3.053 | 0.041855 | 0.045921 | 0.063296 | 0.001643 | 0.745905 | 12/23 |
| Single C | 3.854 | 0.040243 | −0.012338 | 0.068297 | 0.008818 | 0.745433 | 10/23 |

크기 비율은 `||ΔModel|| / ||ΔExpert||`의 Pair별 평균이며, 표의 두 평균 크기를 나눈 값과 다르다. Signed Gain은 `ΔModel·ΔExpert / ||ΔExpert||²`다. 큰 반응이 옆이나 반대 방향으로 나오는 경우를 구분한다. Response 오차는 `||ΔModel−ΔExpert||`이며, 무반응 기준 오차는 `||ΔExpert||`다. 미세한 개선 Pair 수도 보존한 기술 지표이고, 성공률이나 통계적 유의성은 아니다.

TCP·Left Finger·Right Finger·Peg Mean·Box Mean 다섯 Slot, Readout Cosine/Absolute L2, GRU Input/Output Hidden, Native Translation/Rotation/Gripper, Controller Rotation Geodesic/Gripper Target 오차와 Perturbed 상태 자체의 Expert Action 오차는 [전체 CSV](../../experiments/073de7b8e59d4f23b650c87c377c1829/per_condition_metrics.csv), [Slot CSV](../../experiments/073de7b8e59d4f23b650c87c377c1829/readout_slot_metrics.csv)에 있다. Robot Slot도 변해 Peg Mean만으로 판단하지 않았다. 전체 60개 조건에서 Gripper Sign 전환은 없었다.

![Expert 대비 방향·크기 정렬](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/alignment.png)

### Prefix 이후 현재 Graph 반응의 감소

| 시점 | 유효 Pair | Expert Δ mm | Full A Δ mm | Full B Δ mm | Full C Δ mm | Single A Δ mm | Single B Δ mm | Single C Δ mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t=0 | 7 | 0.421266 | 0.832325 | 0.298773 | 0.513857 | 0.246458 | 0.004245 | 0.005787 |
| t=10 | 8 | 0.527816 | 0.252878 | 0.133505 | 0.121202 | 0.026446 | 0.044919 | 0.060666 |
| t=25 | 8 | 1.249468 | 0.094974 | 0.054504 | 0.061745 | 0.032986 | 0.071700 | 0.049969 |

t=25에서 Full A/B/C의 Readout은 각각 **3.2012/5.4376/6.0047%** 변했지만, 이동 명령 크기 비율은 **0.077872/0.057798/0.068991**였다. Full A의 t=10/25 방향 Cosine은 −0.021765/−0.054046이었다. 따라서 현재 관측의 Geometry 변화가 최종 적절한 제어 수정으로 전달되는 데 국소적인 약점이 있다는 근거다. Expert 변화가 없는 조건에서 무반응을 문제 삼은 결과가 아니다.

이는 `SAME_HISTORY_ONE_STEP_GEOMETRY_INTERVENTION`이다. Episode 전체 Graph History를 무시한다거나, Hidden 자체가 원인이라고 단정할 수 없다. t=0과 t=25는 Robot State도 다른 별도 상태이므로, 두 시점의 민감도 차이를 Hidden만의 인과 효과로 해석하지 않는다.

Single B/C의 Signed Gain은 거의 0이고 Response 오차는 무반응 기준과 비슷했다. 이전 단일 Episode 삽입 성공이 현재 Peg Geometry 변화에 대한 유용한 반응을 입증하지는 않는다. Robot/History 암기 가능성과 양립하지만, 암기 메커니즘을 직접 식별한 것은 아니다.

## 추가 후보 및 Box-only 대조

추가 후보는 **17/18** Pair가 입력 지원 기준을 통과했다. Expert 평균 Δ는 **1.542757 mm**, Model Δ는 Full A/B/C **0.559921/0.277409/0.337877 mm**, Single A/B/C **0.103694/0.032120/0.022336 mm**였다. Primary에 합치지 않았다. Translation/Rotation의 개별값은 [Group 요약](../../experiments/073de7b8e59d4f23b650c87c377c1829/group_summary.json)과 원자료에 있다.

Box-only의 Expert Translation 변화는 **12개 모두 0**이었다. t=0/10의 관측 가능한 8개에서는 Full A/B/C 이동 명령이 평균 **0.257389/0.305621/0.238275 mm** 변했다. 이 Reach Reference에는 불필요한 변화다. 다만 Box 이동으로 RGB 기반 전체 탐지/Mask가 달라져 Peg 선택도 변할 수 있어, Box Slot 하나의 의미적 영향만 분리한 결과는 아니다.

t=25에는 **GT Visible Box Pixel 0개인데 Student Box Graph는 69 Point, GT Purity 0**이었다. 네 Box 이동 조건 모두 RGB-D·Graph·Action이 그대로였다. 이 네 조건은 `UNOBSERVABLE_COUNTERFACTUAL`에 해당하며, 무반응을 Graph 무시로 해석하지 않았다. 가려진 물체의 실제 Geometry 지원과 단순한 Graph Presence는 다르다.

![t25 RGB-D·예측 Mask·GT 감사 비교](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/rgbd_masks_t25.png)

## Task 07-G와의 비교

[07-G](07g_point_resampling_stability.md)의 Phase1 Readout Relative L2 평균은 5.774%, 동일 Previous Hidden의 이동 명령 차이는 0.129796 mm, 독립 History에서는 0.153050 mm였다. 이번 Primary Full Readout 변화와 수치 범위가 겹쳐도 같은 강도의 Noise라는 뜻은 아니다.

07-G는 다른 기존 Epoch39 Checkpoint, 네 Episode, 저장된 동일 실제 Pose의 Sampling 변경이다. 이번은 Full/Single 여섯 Checkpoint, 한 Seed의 세 Reach 상태, 실제 Actor 이동 및 새로운 Vision 출력이다. 현재 여섯 Checkpoint 각각의 Sampling 안정성과 Geometry 민감도를 동일 분포에서 함께 검증한 것은 아니다. 07-G 결과만으로 Graph 무시 또는 유용한 Geometry 사용 중 하나를 선택할 수 없다. 이번에는 Expert Response를 함께 측정했기 때문에 무반응·부정확한 반응·관측 실패를 구분할 수 있었다.

## Q1–Q10 판정

| 질문 | 판정 | 검증된 답과 해석 한계 |
|---|---|---|
| Q1 실제 Peg 변화가 Vision Graph에 반영됐는가? | **CONFIRMED, 조건별 제외 있음** | Primary 23/24 지원. t0 Y+5 mm는 실제 보이는 Peg의 Mask/Point 누락. |
| Q2 GNN Structured Readout이 달라졌는가? | **CONFIRMED** | 유효 Pair에서 여섯 Model 모두 Readout 및 Slot 변화. |
| Q3 같은 Robot76/Hidden에서 Action은 얼마나 변했는가? | **CONFIRMED** | 위 표와 모든 Offset 원자료. Exact Robot76/Previous Hidden 대조 통과. |
| Q4 Expert 반응은? | **CONFIRMED** | 독립 Target 재계산, Primary 평균 0.746397 mm. 세 시점 모두 실제 Reach. |
| Q5 Model의 방향·크기는 Expert와 맞는가? | **SUPPORTED: 불완전한 정렬** | Full A 방향 부정확, B/C 부분 정렬·과소 반응. Single B/C의 정렬된 Gain은 매우 작음. |
| Q6 Full A/B/C가 유용하게 반응하는가? | **SUPPORTED: B/C의 일부 조건, 전체적으로 제한적** | 무반응보다 Response 오차가 작은 조건 B 17/23, C 16/23, A 10/23. 일관된 정확성·크기 일치는 아님. |
| Q7 Single과 Full은 다른가? | **CONFIRMED: 측정 차이 / SUPPORTED: 암기 가능성과 양립** | 대체로 Full의 반응이 큼. Single A와 B/C를 동일한 둔감 정책으로 묶지 않음. |
| Q8 07-G 안정성이 적절한 Geometry 민감도와 양립하는가? | **INCONCLUSIVE: 현재 여섯 Model의 결합 강건성** | Checkpoint·상태·Perturbation 분포가 다름. Sampling 안정성 자체는 활용 유무·정확성을 보장하지 않음. |
| Q9 Geometry 활용 부족 가설을 지지하는가? | **SUPPORTED: 국소적 과소/부정확 반응** | Full의 완전한 Graph 무시는 지지되지 않음. Episode 전체 무시 및 Closed-loop 실패의 단일 원인 설명은 INCONCLUSIVE. |
| Q10 무엇을 추가 조사할 근거인가? | **SUPPORTED: Perception 지원과 학습된 Graph→Action Mapping / 원인 분리는 INCONCLUSIVE** | 실제 Peg 누락·가짜 Box 지원과 downstream 제어 반응 오류가 각각 존재. Fusion·Capacity·Optimization 중 하나를 특정하지 못함. 08-E Training Fit 부족은 별도의 기존 근거로 유지. |

## 검증·보호·산출물

로컬 및 RTX 3090 호스트의 계약/회귀 테스트 **23/23 PASS**. 세 독립 Reset 반복과 동일 Observation 재추론, 모든 상태의 Object Row Permutation 대조도 통과했다. [계약 테스트 로그](../../experiments/073de7b8e59d4f23b650c87c377c1829/contract_tests.log), [Reset 대조](../../experiments/073de7b8e59d4f23b650c87c377c1829/repeat_reset_controls.json).

18개 계약 범주 중 **17개는 전체 조건 PASS**다. RGB-D 변화 계약은 가려진 t=25 Box 네 조건에서 실패했고, [contract_audit.json](../../experiments/073de7b8e59d4f23b650c87c377c1829/contract_audit.json)의 전체 `pass_gate=false`를 보존했다. 해당 조건의 Geometry Response 판정을 하지 않았다. 채택한 Peg 비교에는 모든 적용 계약이 통과했다(`accepted_peg_comparisons_pass=true`). 관측 누락 두 Peg 조건도 정렬도 집계에서 제외했다. 전 조건 PASS로 표시하지 않았다.

JSON/CSV/NPZ 실제 Readback, 27개 PNG Decode, HTML 상대 링크, 360개 Model Tensor의 Shape/Finite, Action/Reference Identity, Packet Timestep을 검증했다. 실제 Source Pixel/Depth/Calibration 역투영과 저장 Graph 재구성도 확인했다. [검증 결과](../../experiments/073de7b8e59d4f23b650c87c377c1829/verification.json), [Source/Graph 재검증](../../experiments/073de7b8e59d4f23b650c87c377c1829/source_graph_readback_audit.json). 대표 RGB-D/Mask PNG도 직접 검수했다.

로컬 회수 후 **1,733개 파일 SHA**, JSON/CSV/NPZ/PNG, HTML 링크 46개와 Markdown 링크 80개를 다시 확인했다. 기존 로컬 사용자 작업 12개 파일의 SHA도 그대로다. [로컬 전송·Readback 검증](../../experiments/073de7b8e59d4f23b650c87c377c1829/local_transfer_verification.json). macOS 부가 파일 자동 처리로 생긴 첫 회수 해시 차이는 원본 바이트를 보존하는 재추출로 해결했다. 정렬도 그림의 시점별 색상 범례도 직접 검수해 수정했다.

기존 Source/Input/Checkpoint/Plan/중지 Status **598개 파일**의 시작·종료 SHA가 같았다. [보호 감사](../../experiments/073de7b8e59d4f23b650c87c377c1829/protection_audit.json). 원본 Full Pipeline은 계속 `ERROR / INTERRUPTED_BY_USER_REQUEST`이며 재시작하지 않았다. 기존 프로세스를 Kill/Pause/Suspend하지 않았다.

`experiments/073de7b8e59d4f23b650c87c377c1829/`와 원격 `/workspace/task09a/experiments/073de7b8e59d4f23b650c87c377c1829/`에 Protocol, UUID/Identity, Source Snapshot, Config, Physical GT Sidecar, 원본 RGB-D/Mask, Graph Packet/Source/F5/Raw EDE, 320D Readout, Fused398/GRU Input128/Previous·Output Hidden144, Native Action7/Controller/Expert, JSON/CSV/NPZ, 27개 PNG, HTML, Markdown, Verification 및 Diff를 보존한다.

주요 시각화: [실제 3D Geometry](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/physical_geometry_t00.png), [Graph Point/Edge](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/graph_t00.png), [Readout Slot](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/readout_slots_t25.png), [Expert Offset 반응](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/expert_response_t25.png), [Model Offset 반응](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/model_response_t25.png), [방향 벡터](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/response_vectors_t25.png), [Full/Single](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/full_vs_single.png), [07-G 비교](../../experiments/073de7b8e59d4f23b650c87c377c1829/figures/sampling_comparison.png).

처음에는 분리된 Source Copy의 SAM2 경로 설정, 이후 GT 감사용 SAPIEN ID 접근 오류로 실행 설정/진단 시도가 중단됐다. 실패 로그와 UUID `5dc7faa7fd4e4e79b6c76263fc2652a7`는 별도 보존했으며 최종 60개 상태는 새 UUID에서 실행했다. Student/Vision 알고리즘이나 원본 Checkpoint를 고친 것이 아니다.

공통 `ExperimentLogger`와 기존 Artifact 등록 인프라를 재사용했다. 실행 추가 파일은 opt-in 진단·보고서·테스트뿐이다. 새 학습/Fine-tuning, Architecture/Loss/Codec 변경, 추가 Seed, 새로운 Student Closed-loop 평가, Dependency 설치, Git Commit, 외부 Publishing/Upload는 없었다. CUDA PhysX, MPS, 실제 50 ms 제어 마감시간과 분기 이후 동적 성공은 **NOT_TESTED**다.
