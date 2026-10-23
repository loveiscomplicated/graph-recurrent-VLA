# Graph Prototype v2 — 공통 controller와 closed-loop failure 분석

## 범위와 실행 환경

기존 v1의 GT adapter, 실제 hollow-box/peg surface sampling, node/edge 정의, train-only geometry normalizer, 128D encoder, 데이터 및 episode split을 재사용했다. 새 코드는 `graph/prototype/v2/`에 한정한다. Visual/Recurrent 및 공통 simulator 코드는 변경하지 않았다. 이 구조는 Graph 팀의 feasibility interface이며 최종 팀 공통 architecture를 정한 것이 아니다.

물리 및 PyTorch 학습/평가는 **CPU / physx_cpu**에서 실행했다. CUDA는 없으며, 이번 unrestricted host에서는 MPS availability가 True이지만 MPS 학습/평가는 실행하지 않았다. SAPIEN의 macOS Vulkan renderer는 MoltenVK/Apple M4 Pro를 사용한다. GPU renderer와 GPU simulation/training을 구분한다.

실제 버전: `{"python": "3.11.13 (main, Jun  3 2025, 18:38:25) [Clang 17.0.0 (clang-1700.0.13.3)]", "torch": "2.14.1", "mani_skill": "3.0.1", "sapien": "3.0.3", "gymnasium": "1.3.0", "h5py": "3.16.0", "matplotlib": "3.11.2", "cuda_available": false, "mps_available": true}`.

기존 checkpoint/data/result 주요 artifact 45개 SHA-256가 모두 유지되었다. 검증 목록: `outputs/graph/v2/diagnostics/preservation_check.json`.

## Phase A — 기존 평가와 expert 재검증

### A1. 설정과 action semantics

설치된 `peg_insertion_side.py`는 task를 `max_episode_steps=100`으로 등록한다. 기존 evaluator는 `TimeLimitWrapper(env.unwrapped, horizon+prefix)`로 wrapper를 교체한다. 실제 zero-action probe에서 기본/100 설정은 step 100에 truncated=True였고, 250 설정은 100/101에서 False, 250에서 True였다. 기존 0/9를 100-step 조기 종료 버그로 설명할 근거는 없다. 새 평가도 이 동작을 사용한다. Flags는 batched Torch scalar를 명시적으로 읽고, `info.success`를 매 step OR하며 success/terminated/truncated에서 learner rollout을 종료한다.

이 **250-step 실험은 공식 100-step benchmark score가 아니다**. Warm start에는 expert prefix 이후 추가 250 learner steps를 허용한다. Absolute TimeLimit은 prefix+250이다. 세 모델에 동일하게 적용했다.

Observation/feature ordering/normalization은 동일한 기존 builder와 checkpoint normalizer를 사용한다. 이전 3개 checkpoint의 held-out MSE가 원래 수치와 정확하게 재현됐다. Dataset SHA와 episode split 일치를 강제하고 state_dict를 strict load한다. 전처리와 controller 모두 `pd_ee_delta_pose`이며 별도 IK/joint-policy 변환은 추가하지 않았다.

실제 controller 설정은 Panda wristcam, use_target=False, normalize_action=True, root_translation:root_aligned_body_rotation이다. Action Box는 [-1,1]^7. XYZ는 .1 m scale이다. 설치된 implementation의 회전은 action 3:6의 L2 norm을 1로 cap하고 **rot_lower=-.1을 곱한 XYZ Euler delta**를 root-aligned body rotation으로 적용한다. 이를 axis-angle이라고 부르거나 quaternion 성분 차이로 계산하지 않는다. Gripper -1은 target -.01 m, +1은 .04 m이다. Expert 변환과 learner 실행이 같은 controller이므로 sign을 추가로 바꾸지 않는다. 단위 테스트에서 installed scaling 함수를 직접 호출해 확인했다. Raw prediction을 기록하고 action bounds로 clip한 적용값을 별도로 기록한다.

Task/control 문서는 [공식 task source](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/peg_insertion_side.py), [controller 문서](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html), [replay 문서](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)를 참고했다. API 세부 수치는 main branch를 추측하지 않고 설치된 3.0.1 source 및 runtime audit으로 확인했다.

### A2. 세 종류의 expert initialization

- 원본 expert: 원본 source episode의 초기 snapshot + **8D pd_joint_pos** action 전부 replay. 9/9 success. EE action과 섞지 않았다.
- 변환 expert: 같은 초기 snapshot + 공식 replay로 변환된 **7D pd_ee_delta_pose** action. Full 9/9, alignment-prefix 9/9. 전체 state observation replay 최대 오차 0.0.
- Warm start: original t0 snapshot에서 과거 converted expert action prefix를 실행한 뒤 learner에 handoff. Prefix observation 오차가 .001을 넘으면 중단한다. Contact/controller history를 잃는 단독 중간 snapshot은 새 main 평가에서 사용하지 않았다. 기존 v1의 단독 snapshot expert 4/9 문제는 `outputs/graph/expert_snapshot_control.json`에 보존한다.

기존 설치 버전 replay의 use-first-env-state 초기 저장 t1 문제는 v1 preparation에서 원본 t0 snapshot으로 이미 수정했다. 새 데이터로 바꾸거나 이 오류를 다시 도입하지 않았다. Qvel도 같은 observation의 **현재 t**에서 읽으며 실제 live prefix state와 비교했다. Converted actions T개 / observations T+1개 중 마지막 obs는 target에 연결하지 않는다. Reset seed/reconfigure로 randomized collision shape를 복원하고 기존 snapshot 검증을 재사용한다.

Held-out converted episode의 길이는 105–197 step이고 첫 success는 97–188 step이었다. 9개 중 8개가 100 step 이후 처음 성공한다. 공식 replay는 trajectory action을 계속 실행하며 기본 truncation에서 멈추지 않는 방식이므로 demo 길이와 등록 time limit이 다를 수 있다. 이것은 evaluator가 250을 실제 실행하지 않았다는 증거가 아니다.

### A3. 단계 판정 — 결과를 본 후 threshold 조정하지 않음

`diagnostics.THRESHOLDS`를 baseline 실행 전에 정했다. 모든 JSON에 기준을 기록한다.

- Approach: 현재 peg geometry에서 정한 tail grasp convention에 TCP translation ≤2.5 cm, rotation ≤20°. 평행 jaw의 pi yaw symmetry 두 가지 중 작은 각도를 분석에만 사용한다. Planner의 미래 grasp waypoint를 입력하지 않는다.
- Acquisition: installed Panda `is_grasping(peg)` (default 양쪽 contact force ≥.5 N, closing direction angle ≤85°)가 5개 연속 관측에서 True.
- Transport proxy: 동일 contact retention 5개 + peg centre world_z−local half_z >3 cm. Nominal upright peg의 lift proxy이며 기울어진 peg의 정확한 최저 collision corner 높이 또는 목표 영역 도달을 의미하지 않는다. 이 proxy를 실험 전에 고정했고 결과에 맞춰 바꾸지 않았다.
- Alignment reached: peg head의 hole-local transverse distance ≤1 cm, peg/hole +X axis angle ≤10°, head와 front entrance x 간 거리 ≤4 cm.
- Insertion entered: head가 front plane x≥-box_half_x를 통과하고 hole-local |y|,|z|≤randomized hole radius.
- Success: **공식 info.success만 사용**. Installed task는 head x≥-.015 m, |y|,|z|≤hole radius를 확인하며 orientation/contact를 별도 success 조건으로 요구하지 않는다.

Full-task 각 stage 지표의 denominator는 전체 9 episodes이고 ever reached다. Conditional transition probability가 아니며 stage flags는 반드시 nested하지 않는다. Failure category는 no sustained acquisition → lost final grasp → no transport → no alignment → no success after alignment의 순서로 정한다. 단순 first-contact나 마지막 pose만으로 성공이라고 보고하지 않는다.

Grasp-start prefix는 expert contact가 5번 연속 유지된 최초 state(42–73 step), transport-start는 expert가 그 상태로 nominal lift proxy를 통과한 최초 state(55–100 step)이다. Alignment-start는 v1 dataset의 최초 geometry-proxy phase3 (없으면 phase4)로 정의해 원래 2/9 결과와 직접 비교 가능하게 유지했다. Transport-start는 lift 후 handoff이지 모든 transport가 완료된 oracle state라고 부르지 않는다. 이 시작 index와 새 stricter alignment-reached predicate는 구분한다. 동일 prefix/initial geometry SHA를 모델 사이에 확인했다.

### A4. 기존 정책의 실측 실패

| Model | Full | Acquisition | Transport | Alignment reached | Grasp-start | Transport-start | Alignment-start |
|---|---|---|---|---|---|---|---|
| flat | 0/9 | 5/9 | 5/9 | 3/9 | 0/9 | 0/9 | 0/9 |
| center | 0/9 | 5/9 | 2/9 | 0/9 | 1/9 | 0/9 | 0/9 |
| surface | 0/9 | 8/9 | 7/9 | 0/9 | 0/9 | 1/9 | 2/9 |

Full failure category counts:

- flat: `{'acquisition': 4, 'retention': 1, 'transport': 0, 'alignment': 1, 'insertion': 3}`.
- center: `{'acquisition': 4, 'retention': 0, 'transport': 3, 'alignment': 2, 'insertion': 0}`.
- surface: `{'acquisition': 1, 'retention': 1, 'transport': 0, 'alignment': 7, 'insertion': 0}`.

기존 0/9가 재현됐다. 공통으로 성공 상태에 도달하지 못하지만 같은 failure pattern이라고 단정할 수는 없다. Flat/Center는 acquisition 실패가 각각 4/9였고, Surface는 acquisition 8/9, transport 7/9 이후 alignment에서 많이 실패했다. 따라서 grasp만 개선하면 모든 모델이 해결된다고 볼 수 없다.

Held-out expert-state gripper sign accuracy는 약 99% 이상이다. Dataset gripper 명령은 ±1 두 값만 존재하며 open 36.72%, close 63.28%, intermediate 0%. 기존 expert-state one-step probes(3 episodes × approach/grasp/alignment × 3 models = 27)에서 gripper sign mismatch는 0이다. 동일 prefix를 각각 다시 실행하여 expert/learner action 하나만 바꿨고 초기 geometry hash 일치를 확인했다. 실제 TCP translation movement 차이는 평균 Flat 1.65 mm, Center 1.74 mm, Surface 1.70 mm였다. 이 수치는 normalized action RMSE와 다른 지표다. Rotation bias나 clipping이 유일 원인이라는 결론은 얻지 못했다.

Full rollout에서는 raw translation command 평균 norm이 Flat .601, Center .470, Surface .422였지만 실제 TCP 이동 norm 평균은 3.37 / 3.20 / 3.68 mm/step이었다. 따라서 전체를 “너무 작은 translation command”만으로 설명할 수 없다. IK, joint dynamics, collision, workspace 및 out-of-distribution action의 영향을 분리한 인과 실험은 하지 않았다. Raw 값의 saturation과 실제 이동 차이를 trace에서 확인할 수 있다.

모든 대표/전체 trace에는 t, absolute elapsed step, current/post TCP/peg/hole pose, qpos/qvel, gripper joint state, raw/applied action, realized TCP translation/rotation angle, head relative pose, 양쪽 finger contact force 및 grasp, success/termination flags를 기록했다. 정답 action은 expert replay와 one-step probe에서만 함께 저장한다. Learner 방문 state에 expert의 원래 timestep action을 정답이라고 붙이지 않았다.

## Phase B — 공통 state 보강

### 그대로 둔 입력

Surface graph는 TCP/양쪽 pads + peg64 + hollow-box64 =131 nodes다. Node14는 type one-hot5, EE-relative world xyz3, object-local xyz3, world normal3이다. Edge8은 dst-src xyz3, distance1, relation one-hot4이다. EE–pads, object-local kNN8와 연결 보장 tree, pad–peg 및 peg–hole nearest4를 bidirectional로 유지한다. Center는 5 nodes + 동일 pose/shape context81이다. Flat에는 Surface와 같은 모든 node/context 정보가 fixed order로 있다. Edge-aware residual message passing3 layers, hidden128, 3 robot readout + peg/hole mean readout을 기존 그대로 사용한다. 세부 surface/graph 정의는 [v1 보고서](graph_prototype.md)에 보존했다.

V1 context81에는 이미 qpos9, gripper2, TCP/root pose, peg/box/hole의 position/orientation, sizes/radius/offset이 있다. Surface node에는 EE-relative world-axis position, object-local position, world normal 및 type이 있고 Flat에도 모두 같은 순서로 있다. Qpos/gripper/EE pose를 새 bypass에 중복 추가하지 않았다. 기존 v1의 qpos 내부 finger entries와 gripper2 중복은 baseline 재현을 위해 그대로 보존했다.

### 새 bypass31의 정확한 ordering

| Slice | Feature | Frame / unit |
|---|---|---|
| 0:9 | qvel | Panda active joint order: 7 arm rad/s + 2 finger m/s |
| 9:12 | EE → current tail grasp translation | EE-local metres |
| 12:18 | EE^-1 R_grasp 6D | relative SO(3), first two matrix columns, column-major |
| 18:21 | current peg +X head position | hole-local metres, origin is hole centre |
| 21:27 | R_hole^T R_peg 6D | relative SO(3), same ordering |
| 27:28 | hole_radius − peg_half_y clearance | metres |
| 28:31 | hole +X insertion axis | world unit vector (task robot root is fixed) |

Grasp convention은 peg-local tail x=-max(.05,L/2+.01), y=0, z=half_z-.025, R_grasp=R_peg diag(1,-1,-1)이다. L은 half length이며 episode별 현재 GT 크기를 사용한다. 사전에 정한 현재 물체의 grasp convention이며 expert 미래 waypoint/next action이 아니다.Hole front entry is x=-box_half_x; insertion objective is head near hole centre. Relative head xyz + orientation and existing randomized dimensions retain this information. Quaternions are wxyz; relative rotation uses matrix composition, never component subtraction. World-common rigid transformations leave EE/hole-relative residuals unchanged, verified in tests.

`proprio` ablation supplies only 9 velocities. `goal` supplies all31. Current qvel is read from validated source obs[t,9:18] and live robot.get_qvel; qpos[t,:9] must match the existing dataset. Source current state/action indices and episode splits are unchanged. Each extra continuous feature has train-only mean/std with min std .001. No future state, reward, success, contacts, expert phase, object velocity or nearest expert action enters a policy. Fixed peg-frame grasp convention은 nominal expert pose에서 확인했으며, peg가 넘어지거나 뒤집힌 recovery state의 접근 가능한 top-face 선택/IK feasibility는 검증하지 않았다.

Proprioception/FK and geometric residuals are deployable candidates, assuming Visual supplies object frames, shape, head/hole geometry. Simulator contact/grasp/phase are privileged diagnostics only. Object velocity and TCP velocity were not added because qvel is stable and directly available without a new extraction/interface requirement.

### Common head and interface

Each unmodified Flat/Center/Surface encoder outputs [B,128]. The same head design concatenates normalized shared state: 159→128→7 with ReLU. Capacity is configurable by head_width, default128; no new GNN layer/attention was added. Component loss remains mean squared error over normalized 7D action. Inference clips to actual action bounds. Encoder readout and action head remain separately accessible for later controller replacement.

Binary gripper classification is implemented/tested as an optional head mode (motion squared sum + BCE)/7, hard ±1 deployment. It was **not trained or included in the main experiment**: baseline expert-state signs and matched one-step actions were already reliable, and changing output decisions and loss while testing new state would confound this fixed main comparison. New rollout sign failures after drift are reported rather than hidden. This leaves a controlled binary-head ablation as future work.

```python
geometry = gt_adapter(env)
graph = geometry_normalizer.apply("surface", geometry, graph_builder)
embedding = policy.encoder(graph)  # [B,128]
shared = state_normalizer.apply(geometry, env.unwrapped.agent.robot.get_qvel(), "goal")
action = policy.action_head(torch.cat((embedding, shared), -1))
```

## Phase C — 단계적인 학습 검증

10-epoch curves were still improving: last train/val Flat .003608/.005623, Center .002486/.005347, Surface .003160/.005571. Cosine schedule had already reduced LR near zero. First experiment therefore kept Flat architecture/input/loss exactly and restarted a 30-epoch cosine budget. Then qvel-only and qvel+goal additions were trained separately with 30 epochs each. No test split determined the chosen state or checkpoint.

Validation 선택 결과: `{"selection_criterion": "lowest held-out VALIDATION action MSE among fixed 30-epoch Flat ablations; no test/rollout result used", "candidate_validation_mse": {"flat_duration": 0.005012829788029194, "flat_proprio": 0.003084807191044092, "flat_goal": 0.0027660138439387083}, "selected_state": "goal", "selected_gripper": "regression", "epochs": 30, "seed": 0, "head_width": 128, "notes": "Keep regression: offline signs nearly perfect, all 27 expert-state one-step probes had correct gripper sign. Binary branch implemented and unit-tested, not selected or trained."}`.

공통 main은 동일 train 11,274 / validation 1,325 / test 1,359 samples, AdamW lr .001 / wd 1e-5, cosine30 epochs, batch32, seed0, threads4, 10,590 updates, validation raw action MSE checkpoint selection이다. Same episode split 73/9/9. Data contains91 episodes /13,958 samples. Shuffling uses separate seeded generator so model parameter count does not change minibatch order. Test metrics were computed after each ablation for reporting, not used to choose its checkpoint. The final common setting was locked using validation before main rollout results.

Flat ablation은 v2 config file을 만들기 전에 legacy config를 읽었으므로 checkpoint에 사용하지 않는 max_episode_steps=100 metadata가 남아 있다. Training은 환경을 실행하지 않으며 이 값은 encoder/loss/optimization에 사용되지 않는다. 모든 실제 learner 평가는 명시적인 horizon250 wrapper를 사용했고 JSON에 기록했다. 나머지 v2 config의 metadata는250이다. 이 차이를 runtime time limit과 혼동하지 않는다.

30 epoch로 늘리는 것만으로 Flat offline error와 alignment-start는 일부 개선됐다. 가장 큰 추가 offline gain은 qvel(.004001→.001817)이다. Goal 추가는 validation에서는 .003085→.002766이었으나 test MSE는 .001817→.001820으로 거의 같았다. Full SR 개선은 없었다. Closed-loop에서 관측된 duration-only alignment 0/9→2/9 역시 작은 표본이므로 확정적인 개선이라 부르지 않는다. Proprio/goal은 offline error를 더 낮추었지만 full 성공을 얻지 못했다. State augmentation이 control을 개선했다고 주장하지 않는다. Table의 원래 V0는 10 epochs, 나머지 ablation/main은 30 epochs이므로 old→v2 차이에는 duration도 포함된다. Duration-only row가 이를 분리한다.

## Phase D — 실제 비교 표

### 측정된 결과

CPU, seed 0. Same 91 episodes / 73–9–9 split. Main: 30 epochs, 10,590 updates, qvel9 + geometric residual22, regression head. Validation selection only. Full and each warm-start have 250 **learner** steps; warm starts include prior expert actions.

## Flat ablations

| Configuration | Action MSE | Full SR | Acquisition | Grasp-start SR | Alignment-start SR | Params |
|---|---:|---|---|---|---|---:|
| flat_v0 | 0.004505 | 0/9 | 5/9 | 0/9 | 0/9 | 295687 |
| flat_duration | 0.004001 | 0/9 | 7/9 | 0/9 | 2/9 | 295687 |
| flat_proprio | 0.001817 | 0/9 | 1/9 | 0/9 | 1/9 | 296839 |
| flat_goal | 0.001820 | 0/9 | 2/9 | 0/9 | 0/9 | 299655 |

## Same controller, three encoders

| Model | Action MSE | Translation RMSE | Rotation RMSE | Full SR | Alignment-start SR | Fresh SR | Params | Latency ms |
|---|---:|---:|---:|---|---|---|---:|---:|
| flat | 0.001820 | 0.020477 | 0.036204 | 0/9 | 0/9 | 0/30 | 299655 | 0.406 |
| center | 0.001342 | 0.016685 | 0.034554 | 0/9 | 2/9 | 0/30 | 449031 | 1.024 |
| surface | 0.001566 | 0.017785 | 0.034882 | 0/9 | 1/9 | 0/30 | 449031 | 2.420 |

RMSE is normalized action, not realized TCP motion. Latency is geometry feature/graph + shared-state construction + network, **excludes GT extraction**. No Torch GPU training was used.

## Full-task stage counts (denominator: all 9 episodes)

| Metric | Flat | Center | Surface |
|---|---|---|---|
| approach | 2/9 | 3/9 | 2/9 |
| acquisition | 2/9 | 4/9 | 2/9 |
| transport proxy | 2/9 | 4/9 | 0/9 |
| alignment | 0/9 | 0/9 | 0/9 |
| insertion | 0/9 | 0/9 | 0/9 |
| full-start success | 0/9 | 0/9 | 0/9 |
| grasp-start success | 0/9 | 0/9 | 0/9 |
| transport-start success | 0/9 | 1/9 | 0/9 |
| alignment-start success | 0/9 | 2/9 | 1/9 |

## Uncertainty

- flat: held-out full Wilson 95% [0.0, 0.2991527535509594]; fresh full [0.0, 0.113517091390478]; alignment-start [0.0, 0.2991527535509594].
- center: held-out full Wilson 95% [0.0, 0.2991527535509594]; fresh full [0.0, 0.113517091390478]; alignment-start [0.06322376231222723, 0.5474166563271946].
- surface: held-out full Wilson 95% [0.0, 0.2991527535509594]; fresh full [0.0, 0.113517091390478]; alignment-start [0.019890371327130535, 0.4350062147680601].

Thresholds are fixed in diagnostics.THRESHOLDS and reproduced in each JSON. Acquisition: Panda is_grasping (.5 N, 85° default) for 5 consecutive observations; transport uses nominal upright centre_z−local half_z>3 cm lift proxy, not exact rotated-box clearance or goal-region completion; alignment: head transverse ≤1 cm, axis ≤10°, entry distance ≤4 cm. Insertion flag means aperture entry, **success is exclusively official info.success**, which checks head x≥−.015 m and y,z within randomized hole radius. Counts indicate ever reached and are not conditional transition probabilities. Warm-start stage achievement before handoff is expert-provided.

Single training seed; 9 familiar held-out configurations and 30 fresh seeds (2000–2029), separately reported. Small warm-start samples do not establish graph superiority.


### Latency와 training overhead

| Model | Train seconds | Selected epoch | Feature/graph ms | Shared state ms | Network ms |
|---|---:|---:|---:|---:|---:|
| flat | 18.61 | 29 | 0.1671 | 0.2115 | 0.0274 |
| center | 69.24 | 25 | 0.2265 | 0.2153 | 0.5817 |
| surface | 974.82 | 26 | 0.6170 | 0.2058 | 1.5972 |

Latencies are batch1 CPU, 10 warm-up +50 timed forwards, four Torch threads; graph/geometry normalization, extra-state construction and network are separately timed. GT extraction/rendering/physics is excluded. Models were timed sequentially after training. Training wall times reflect host concurrency and are not isolated throughput benchmarks. Main parameter matching was not performed; Flat has fewer parameters, Center/Surface are equal. No multiple training seeds were run. Fresh 30 episodes expand configuration coverage, **not training seed coverage**.

## Phase E — Distribution shift / correction oracle feasibility

Distance uses train-standardized 33D diagnostic state (extra31 + actual finger qpos2), Euclidean nearest train timestep. It is not probability, a dynamics model, policy-embedding distance, or oracle action. V1 did not consume qvel; the distance is common physical state space for diagnosis. Correlated features and chosen scaling affect values. Held-out expert states provide an out-of-sample reference; learner trace is sampled every5 steps. A large distance supports distribution shift, but does not prove a causal mechanism.

Held-out expert median distance 1.372, p95 3.941.

- flat_goal: median 10.784, p95 515.503, expert p95 초과 82.0% (450 sampled learner states).
- center_goal: median 11.069, p95 76.804, expert p95 초과 78.7% (450 sampled learner states).
- surface_goal: median 11.623, p95 511.999, expert p95 초과 89.1% (450 sampled learner states).

Installed official solution has `env.reset(seed=seed)` before planning, asserts joint-control mode, and imports mplib. Actual import failed: ModuleNotFoundError(mplib). The stock solve entry point is not an arbitrary learner-state oracle. No correction actions/coverage/success rate were measured (N/A), no solver re-planning was executed, and no DAgger/recovery fine-tuning was done. A future oracle must preserve failed state/contact/controller initialization, produce valid corrected joint motion, convert it into the same EE action semantics, and demonstrate correction success before aggregation. Nearest stored demonstration actions were never used as labels.

## 해석 — 실측과 가설 구분

실측: observation/controller/horizon mismatch는 발견하지 못했고 expert는 재현된다. Offline error는 공통 state와 학습 기간으로 감소하지만 full insertion 성공은 얻지 못했다. Warm-start 결과는 별도 표의 작은 표본이다. Failure stage 분포가 encoder별로 다르므로 단일 공통 failure라고 뭉뚱그리지 않는다. 특히 qvel/goal을 더한 Flat에서 acquisition/retention 성능은 악화됐다. Representative traces show wrong opening commands and saturated translation after state drift. 기존 expert-state에서 분류가 잘 된다는 사실은 learner 방문 state에서 gripper가 안전하다는 보장이 아니다.

가설: state-distribution mismatch, velocity-conditioned BC feedback, contact/IK dynamics, expert path timing에 따른 label ambiguity 및 recovery 부재가 관련될 수 있다. 단일 seed / 제한된 configs에서 인과 원인을 분리하지 않았고, qvel이 반드시 나쁘거나 GNN이 무의미하다고 결론 내릴 수 없다. 더 큰 head나 많은 epoch만으로 해결된다고도 주장하지 않는다. 이 결과는 **Case C**에 해당하며 Surface 우월성은 확인되지 않았다. Precision warm-start 연구를 이어갈 제한적 근거는 있지만 full-task 비교 기반은 아직 약하다.

다음 우선순위는 valid failure-state oracle/recovery feasibility, closed-loop 안정성/velocity feedback와 gripper transition을 분리한 작은 실험, 더 넓은 expert configuration coverage이다. 이 세 가지를 확인한 뒤 Recurrent를 붙여 phase/intent memory가 실제 필요한지 검증해야 한다. Recurrent를 추가하면 observation/action timing, controller history, prefix와 hidden-state initialization을 명시해야 한다. 이번에는 Visual/Recurrent를 구현하지 않았다.

Visual 통합 시 replace할 GT는 peg/hole pose와 shape, canonical local surface/normal, hole frame/entry와 uncertainty다. FK/qpos/qvel/gripper는 robot proprioception 후보로 유지한다. Contact/grasp/success/phase는 policy interface에서 제외한다. Dataset point ordering와 quaternion/frame conventions 및 train-only statistics는 버전으로 관리해야 한다.

## 테스트, 산출물, 한계

기존 Graph12 + shared simulator/GT8 =20 regression tests를 실제 실행해 통과했다. V2 16개 tests: shape/finite checks, relative SO(3) and quaternion sign, rigid goal transform, train-only normalization, all three forwards/backwards, each 80-update small-batch overfit, checkpoint save/load and deterministic outputs, exact V0 initialization, optional binary branch, configurable head width, live qvel/goal train-eval equality, installed controller scaling, saved real horizon/expert/paired reproduction, success OR/retention, action bounds/Wilson interval, episode split와 no privileged labels, main normalization/budget/paired configuration 동일성을 통과했다. Analytic fixtures are test-only, not performance data. 실제 로그는 diagnostics/tests_*.log. 추가로 실제 train episodes 27/20의 16 samples에서 각 모델 200-update overfit을 실행해 MSE를 초기 값의 1/10 미만으로 줄이고 checkpoint reload의 bitwise output equality를 확인했다. 이는 held-out metric이 아니며 diagnostics/overfit_real.json에 별도로 기록한다.

초기 goal visualization에서 Torch scalar index list가 multi-axis indexing으로 해석되는 오류를 발견했고 int indices로 수정 후 PNG를 다시 생성했다. 기존 baseline evaluator bug는 발견하지 못했다. 실행하지 않은 GPU/DAgger/multi-training-seed/parameter-matched experiments는 성공했다고 보고하지 않았다.

기존 v1 파일은 보존했고 새 구현: `graph/prototype/v2/{goals,state,models,train,diagnostics,analyze,recovery,overfit,compare,figures,visualize_goals,video,report}.py`, config.toml, README.md, package markers, tests/test_v2.py. 실행 명령은 [v2 README](../graph/prototype/v2/README.md)에 있다. 결과 tables/json, checkpoints/history/loss, rich traces, one-step probes, horizon/expert audit, distribution diagnostics, goal/graph PNGs, action/loss/error figures 및 실제 rollout videos는 `outputs/graph/v2/`에 있다. 기존 Graph README에는 v2 안내 링크만 추가했다.
