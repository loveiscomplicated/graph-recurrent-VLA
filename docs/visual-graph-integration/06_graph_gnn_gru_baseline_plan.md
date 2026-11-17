# Task06 — Minimal Graph–GNN–GRU/BC Integration Baseline 계획

작성일: 2026-10-25. 상태: **PLAN ONLY / 구현 승인 대기**.

조사 기준: `visual-graph` branch, HEAD `63c3f6c9fe0a4ac5b8e0096fc7adca20ca21e4f8` 및 현재 working tree. 기존 dirty/untracked Task04~05 코드를 포함해 읽었다. HEAD만으로 현재 구현을 재현할 수 없다. 이번 변경은 이 계획 파일 하나다.

목표 경로는 `Vision Geometry → Graph Construction → Edge-aware MLP GNN → Structured Readout → Robot State Fusion → GRU → BC Action`이다. 아래 파일명·API·실행값은 승인 후 최소 구현을 위한 제안이며 공통 simulator나 팀 전체의 최종 인터페이스를 정하지 않는다.

## 1. 조사 결론과 여섯 질문의 답

| 질문 | 확인 결과 / 계획 |
|---|---|
| 현재 어디까지 구현됐는가? | Vision v2 F=5/Mask/EE 좌표/Raw EDE 저장 및 수신, 선택적으로 Robot 3행 prepend/+3까지 있다. 별도로 GT Graph→MLP message passing→5개 그룹 readout→feed-forward BC와 Recurrent GRU→BC가 있다. **실제 Vision Graph→GNN→GRU/BC 통합은 없다.** |
| 최소 변경 파일은? | 아래 §7의 opt-in `graph/integration_baseline/` 7개 파일과 인터페이스 문서 보완을 제안한다. 기존 GT 모델·학습 entrypoint·production default·Vision 알고리즘을 수정하지 않는다. |
| 실제 Robot/Action 출처는? | 91-episode GT dataset에는 TCP/양 fingertip/qpos/gripper 및 정합된 delta-pose action이 있다. qvel9는 원본 변환 H5 observation에서 얻는 기존 API가 있다. **현재 Vision episode에는 TCP/qpos/absolute-pose action만 있고 양 fingertip/qvel이 없다.** 서로 다른 episode 자료를 합치지 않는다. |
| 저장 Packet으로 가능한 검증은? | 승인 후 8개 시점의 실제 Point/F=5/Raw EDE에 대한 tensor/GNN/readout 검증이 가능하다. 실제 robot 입력이 없으므로 3 robot node를 포함한 경로에는 synthetic fixture라는 표시가 필요하다. 합성 state/target으로 GRU/backward 연산 계약만 확인할 수 있고 실제 BC 통합 PASS로 보고할 수 없다. |
| 실제 BC 시작에 부족한 것은? | 같은 delta-pose expert episode의 시간순 Vision v2+Raw EDE coverage, 같은 시점 실제 fingertip, qvel, packet↔episode/timestep join, ragged graph sequence collate가 없다. 현재 Vision label의 control mode·시간 방향도 기존 BC와 다르다. |
| L,d,k,M은 어떻게 실험하는가? | 독립 config 필드로 보존하고 smoke 임시값과 성능 선택값을 구분한다. 동일 episode split/입력 hash/state/action/예산에서 connectivity→capacity→소수 interaction 순으로 추후 비교한다. 지금 search는 실행하지 않는다. |

**핵심 Blocker:** 저장 자료의 배열을 연결하는 것과 실제 BC 학습 데이터를 확보하는 것은 다른 단계다. 현재 자료만으로 실제 3-node Robot geometry와 정상 시간축의 delta-pose BC sequence가 준비됐다고 주장할 수 없다.

## 2. 현재 구현과 재사용 범위

관련 지침: [프로젝트](../PROJECT_CONTEXT.md), [engineering](../engineering_rules.md), [tracking](../experiment_tracking.md), [artifact publishing](../huggingface_artifacts.md), [ManiSkill](../../envs/maniskill/README.md). README의 초기 scaffold 설명보다 현재 소스·저장 파일을 우선했다.

### 2.1 Vision 근거와 변경하지 않을 경계

| 근거 | 현재 저장 자료와 재사용 범위 |
|---|---|
| [Task03-K](03k_temporal_shape_first.md), `9185355a32b8433abc70f1738f983377` | 단일 `ep000`의 121 frame × 2 ratio, `selection/` 242 NPZ. Shape-First + Surface-Second, 원본 point/source row 보존. 이것만으로 완성된 F=5/EE/Raw EDE graph sequence가 되지는 않는다. |
| [Task04](04_vision_feature_edge.md), `9f90293b227e4dbfaaa6290afad9cc3b` | `packets/` 16 NPZ: frame 0,18,27,45,56,66,80,120 × ratio 0.5/1.0. Continuous F=5, field availability/validity, EE transform와 receiver 계약. 이 run의 intra edge는 K8이므로 Task06 intra edge 입력으로 잘못 선택하지 않는다. |
| [Task05-B](05b_extended_delaunay_edgewise.md), `9f36f7c89dbd4097a8be9358138d1537` | Extended Delaunay 후보: 비퇴화 Delaunay, 2-point pair, 전체 collinear chain. 기존 Strict 결과와 provenance 근거. |
| [Task05-E](05e_raw_ede_comparison.md), `0f107550c9b64f1a84677a18e988cb51` | `packets/fXXXX_r0.5_raw.npz` 8개를 기본 입력으로 사용. ratio1의 8개는 큰 N 계약 확인용. `edge_policy=raw_extended_delaunay`, `edge_validation=none`, `evaluation_only=true`를 보존한다. |
| [Task05-C](05c_ede_exact_optimization.md), `185bc034567d429daf3f1d09f94dc56c` | 기존 Strict CPU exact-equivalence 이력. Raw 경로에 SAT/Supercover/Mask/Depth 검사를 가져오지 않는다. |
| [Task05-D](05d_gpu_edge_benchmark.md), `be206b04f7b245c4a9436317b8f87987` | 과거 실행은 로컬 MPS B1 cover 및 CPU validation이다. B2/CUDA는 미실행 기록이다. Task06에 B1/B2나 새 GPU kernel을 도입하지 않는다. |

`visual/runtime/vision_packet.py::complete_vision_packet`, `sampling.py`, `pca.py`, `extended_delaunay_edges.py`의 sampling/PCA5/feature/EE transform/Raw 후보 생성 로직은 그대로 둔다. 저장 Packet을 우선 사용하고 이번 기준선의 Graph consumer는 intra edge를 생성하지 않는다. Raw producer는 현재 `visual/runtime/test/raw_ede_comparison_adapter.py::build_raw_ede`에 있다. 추후 데이터 준비에 필요하더라도 이동/승격은 별도 검토하며, 이번 계획에서 새 Vision 실행을 전제하지 않는다.

`keep_ratio=0.5`는 잠정 설정이다. 기존 depth 오염 plateau·멀리 떨어진 finite point도 재필터링하지 않는다. 저장 데이터의 물리 정확성 문제는 진단과 성능 비교로 남긴다.

### 2.2 코드별 현재 인터페이스

| 파일 / 함수 | 현재 구현 | 직접 재사용 / 필요한 차이 |
|---|---|---|
| [graph/vision_input.py](../../graph/vision_input.py): `validate_packet`, `receive_vision_packet` | v1/v2 검증과 deep-copy. `robot_positions=float32[3,3]` 공급 시 `[robot, original point rows]`, `point_node_index=arange(N)+3`, intra edge `+3`. 미공급 시 offset0. | 그대로 호출. 위치의 실제 출처·시점·EE frame은 caller 책임이며 receiver가 확보해 주지 않는다. 실제 모델 경로는 robot 미공급을 허용하지 않는다. |
| `VisionInput` | `positions`, `point_node_index`, `node_kind`, source/evidence, robot mask, edge index. `node_kind`: point0/TCP1/left2/right3. | 이미 tensor GNN schema가 완성된 것은 아니다. Object ID와 결합해 새 모델의 5종 node type으로 명시 매핑한다. |
| [visual/runtime/vision_packet.py](../../visual/runtime/vision_packet.py) | v2 `points[N,3]` float32, `geometric_features[N,5]` float64, field별 mask `[N,3]`; compact values는 float32. | 원본 배열·NaN·dtype·source metadata를 보존하고 별도 finite float32 network view만 만든다. |
| [graph/prototype/graph_builder.py](../../graph/prototype/graph_builder.py) | GT fixed-N `x[B,N,14]`, context81, edge attr8, 4종 edge. world axes/TCP-centred이며 완전한 EE 회전 좌표가 아니다. | index batching 아이디어만 참고. GT local geometry/object pose/hole/context, surface kNN/tree, kinematic edge를 새 경로에 사용하지 않는다. |
| 같은 `GraphBuilder` interaction | 양 finger→Peg kNN; Peg 각 point→Hole kNN. | TCP/양 finger 각각→Peg/Box와 global M pair 계약을 충족하지 않는다. 새 opt-in interaction helper가 필요하다. |
| [graph/prototype/models.py](../../graph/prototype/models.py): `MessageLayer`, `GraphEncoder` | edge-aware MLP, `index_add` mean와 zero-degree clamp, residual+LayerNorm+ReLU. robot3 개별+Peg mean+Hole mean 뒤 context와 projection→128. | mean/index_add와 readout 패턴을 재사용하되 fixed N/attr8/context/LN·ReLU를 그대로 호출하지 않는다. 요청된 residual 수식과 raw `5d` 출력에 맞춘 작은 encoder가 필요하다. |
| [graph/prototype/v2/state.py](../../graph/prototype/v2/state.py) | `none=0`, `proprio=9(qvel)`, `goal=31(qvel+GT goal22)`, train-only `StateNormalizer`, std floor .001. | 최소 후보는 기존 **proprio9**. goal31은 금지된 GT hole/object 관계가 들어가므로 사용하지 않는다. |
| [graph/prototype/v2/models.py](../../graph/prototype/v2/models.py) | `CommonPolicy`의 별도 state bypass, 7D head, `bc_loss` regression/binary. | state bypass 개념과 **regression 7D MSE**를 재사용한다. GT encoder/goal input은 재사용하지 않는다. |
| [graph/prototype/v4/history_models.py](../../graph/prototype/v4/history_models.py) | `159=128+goal31`의 fixed-window MLP. history action/mask 포함 가능. | GRU가 아니다. 이 head나 GT goal31, 과거 expert-action 입력을 새 모델에 연결하지 않는다. |
| `v4/history_dataset.py`, `history_buffer.py` | episode-local contiguous timestep 확인; 현재 target을 history에서 제외; 평가 cache는 detached. | episode 경계/시간 정합 검사 참고. 새 ragged sequence loader는 필요하며 detached graph cache를 학습에 사용하지 않는다. |
| [recurrent/models.py](../../recurrent/models.py): `RNNPolicy` | encoder MLP→`nn.GRU(batch_first=True)`→MLP head. `[B,T,obs_dim]`와 optional state → `[B,T,act_dim], h_n`. | `RNNPolicy(cell='gru', act_dim=7)`를 그대로 감싼다. 기본 constructor hidden256과 실제 `configs/models.yaml` GRU hidden144/layers1/enc_hidden128을 구분한다. |
| [recurrent/train.py](../../recurrent/train.py), `evaluate.py` | episode padding/loss mask, full-episode train 호출마다 state=None; 추론은 state 유지, episode reset에서 None. 기존 loss는 motion6 z-score MSE + gripper BCE. | GRU lifecycle과 padding 참고. 기존 38D GT feature builder·loss/action adapter는 새 Graph regression 경로에 섞지 않는다. |
| [utils/experiment_logger.py](../../utils/experiment_logger.py) / artifacts | UUID4/local scalar/config/metadata/sidecar, 선택적 publishing. | 후속 smoke/eval은 공통 Logger 직접 재사용. 새 run ID/project/group 체계를 추가하지 않는다. |

즉 10개 조사 항목 중 robot 실제 join, 요구된 robot-object kNN/global M, attr4, ragged tensor schema/batching, mask-aware encoder, empty-safe structured readout, Vision sequence join이 미구현이다. MLP message passing·state bypass·GRU·BC loss는 재사용 근거가 있지만 계약 차이를 연결해야 한다.

## 3. 실제 데이터와 시간 정합

이번에는 Python으로 기존 JSON/NPZ/H5/PT를 읽고 key·shape·episode 수를 확인했다. 모델 import를 통한 forward, optimizer, simulator reset/step은 실행하지 않았다. 아래 수치는 새로운 실험 결과가 아닌 저장 자료 inventory다.

### 3.1 Demonstration inventory

| 로컬 경로 | 실제 확인 내용 | Task06 학습 사용 판단 |
|---|---|---|
| `outputs/graph/demos/PegInsertionSide-v1/motionplanning/trajectory.h5` + JSON | 1,000 trajectory group. 원본 `pd_joint_pos`, 대표 action `[178,8]`, env_states `[179,...]`, obs 없음. | 초기 상태 repair와 출처 근거. 7D BC target으로 직접 사용하지 않는다. |
| 같은 디렉터리 `trajectory.state.pd_ee_delta_pose.physx_cpu.h5` + JSON | **91 episode / 13,958 actions**. 91개 모두 actions `[T,7]`, obs `[T+1,43]` header 확인. | 기존 Graph BC의 유효 데이터 출처. RGB-D/mask/estimated graph coverage는 없다. |
| `outputs/graph/dataset.pt` + `dataset.json` | 13,958 row. geometry의 TCP/left/right fingertip/qpos/gripper, GT Peg64/Hole64 surface; `episode_id`, `timestep`, `actions`. 저장 split **73 train / 9 validation / 9 test**. | robot-only whitelist/state/action 공급 후보. GT object geometry는 새 Vision 입력으로 사용하지 않는다. |
| `outputs/graph/demos/PegInsertionSide-v1/rl/trajectory.h5` + JSON | 별도 1,000 group, `pd_joint_delta_pos`, 대표 actions `[50,8]`. | 다른 제어 규약. 91개 motionplanning 변환 episode와 합산해 통합 학습셋이라고 부르지 않는다. |
| `outputs/subtasks/generated_50/converted/dataset.h5`, `raw/dataset.h5`, `segments/index.json` | 별도 generated expert 50 episode, delta-pose, split 40/5/5. 대표 actions `[214,7]`, obs `[215,43]`. | Reach/subtask 데이터·환경 의미가 별도다. 자동 병합하지 않는다. 해당 source에 대한 Vision coverage는 확인되지 않았다. |
| `outputs/recurrent/data/{expert_demos,official_demos}.npz` | 코드의 기본 경로지만 **현재 호스트에 `outputs/recurrent/` 자체가 없다**. | `recurrent.train.load_episodes` 사용 가능 코드와 실제 데이터 존재를 구분한다. 원격/다른 host의 존재는 이번에 조회하지 않았다. |

91개 converted episode의 기존 split은 `graph.prototype.data.episode_split`에서 reset seed group을 같은 split에 묶는 방식이다. 명시 manifest가 있으므로 새 랜덤 frame split으로 바꾸지 않는다. `recurrent.train.split`은 episode 단위 train/val이나 별도 split 방식이고 group 중복을 보장하지 않는다. 통합 경로는 frozen Graph manifest를 사용한다.

### 3.2 Vision coverage와 Action mismatch

실제 원본은 [visual/data/step2/ep000.npz](../../visual/data/step2/ep000.npz) 하나다. `seed=0`, `env_id=PegInsertionSide-v1`, **`control_mode=pd_ee_pose`**, frame_idx 0..120, 20Hz timestamp, RGB-D/calibration/TCP/qpos/action 121행이 있다. `visual/results/runtime_tracking/ep000/masks.npz`는 같은 121 source frame의 추정 Peg/Box mask다. `runtime_tracking/ep001/tracking.json`도 입력이 **같은 ep000.npz**라 두 번째 독립 episode가 아니다.

시간 계약은 [visual/scripts/collect.py](../../visual/scripts/collect.py)의 `Recorder.reset/step/_record`에서 확인했다.

- row0은 reset observation + dummy zero action이다.
- row r>0은 `env.step(command)` **이후** observation과 그 step에 사용한 command를 저장한다.
- 따라서 row r의 observation `o_r`에 대한 다음 expert target은 저장 `action[r+1]`이다. `action[r]`를 같은 row의 next-action target으로 사용하면 이전 전이를 학습한다. 마지막 observation에는 다음 label이 없다.
- 이 shift는 시간 대응 설명일 뿐 **absolute pose를 normalized delta pose로 바꾸지 않는다**. action rotation에 약 ±π가 실제 들어 있으므로 단순 `clip(-1,1)`은 의미를 훼손한다. 현재 Vision episode action을 기존 7D delta BC target으로 사용하지 않는다.

기존 Graph 데이터는 [prepare_dataset.py](../../graph/prototype/prepare_dataset.py)의 `state[t]↔actions[t]`, final state[T] 제외 계약이다. ManiSkill replay 초기 state 오류에 대해 원본 initial source를 사용하고 obs match를 확인한 manifest가 있다. Task06은 이 검증 기록을 재사용하되 현재 조사에서 재실행한 것으로 쓰지 않는다.

Task03-K에는 연속 121-frame selection이 있지만 Task04 F=5/EE와 Task05-E Raw packet은 대표 **8시점 × 2 ratio**뿐이다. ratio0.5/1.0은 같은 observation의 대안이며 두 개의 독립 timestep이 아니다. 대표 frame을 이어도 간격이 18,9,18,11,10,14,40 step으로 불균일하다. 이를 20Hz contiguous GRU 학습 sequence라고 취급하지 않는다. source pixel index 역시 frame 간 물리 point ID가 아니다.

### 3.3 Robot geometry / proprioception의 정확한 출처

| 필요한 입력 | 저장 Vision ep000 | 기존 91-episode GT 데이터 / live API |
|---|---|---|
| TCP world pose | `tcp_pose[121,7]`, packet의 `observation_ee_pose[7]` | `geometry.ee_pose`, `get_gt_state().ee_pose` |
| 양 fingertip 실제 위치 | **저장 key 없음**. finger link pose도 없음 | `geometry.left_fingertip/right_fingertip[13958,3]`; live `left_fingertip_pos/right_fingertip_pos` |
| qpos/gripper position | `qpos[121,9]`; 마지막 두 finger joint는 qpos에 있음 | geometry.qpos9/gripper2, H5 obs의 robot prefix |
| qvel9 | **없음** | `graph.prototype.v2.diagnostics.qvel_dataset(data)`가 source H5 `obs[t,9:18]`를 읽고 `obs[t,:9]`와 geometry.qpos를 대조. live robot.get_qvel |
| expert action | absolute pose, post-step 저장 | converted H5 또는 dataset.actions의 normalized delta-pose `[T,7]` |

기존 [get_gt_state](../../envs/maniskill/gt_state.py)는 fingertip을 finger link의 rubber-pad collision centre로 정의한다. local offsets는 left `[0,+0.00758,0.04525]`, right `[0,-0.00758,0.04525]`m다. 실제 link pose에 적용하는 정의이며 임의 고정 EE fingertip 위치가 아니다. 새 consumer는 object pose/hole 필드를 통째로 전달하지 않고 robot 필드만 whitelist한다.

EE 변환은 같은 시점 TCP pose의 `R,t`를 써서 `p_EE=Rᵀ(p_world−t)`로 한다. TCP node가 EE origin `(0,0,0)`인 것은 실제 동일시점 TCP transform의 결과여야 한다. 양 fingertip은 같은 transform으로 변환하고 receiver에 `[TCP,L,R]` 순으로 공급한다. 관측 identity/시점/pose가 맞지 않으면 거부한다.

저장 qpos로 검증된 robot kinematics/FK를 사용해 fingertip을 복원할 가능성은 있지만 현재 연결 함수·동일시점 검증 결과가 없다. 이번에는 FK, 수치 미분 qvel, 임의 gripper 폭 또는 다른 episode의 fingertip을 만들지 않는다. 승인 후 별도 robot-only 보완으로 검토할 수 있으며, 설치 robot/URDF와 실제 accessor 대조를 통과하기 전에는 real geometry로 표시하지 않는다.

설치된 ManiSkill 3.0.1의 `agents/base_agent.py::get_proprioception`, Panda controller config, `controllers/pd_ee_pose.py`, `envs/tasks/tabletop/peg_insertion_side.py::_get_obs_extra`도 읽었다. qpos/qvel 및 extra TCP 제공 근거와 pose/delta mode 차이를 확인했다. Runnable 근거는 기존 `examples/check_env.py`, `examples/inspect_gt_state.py`와 기록된 README다. **이번 작업에서 CPU simulator, MPS, CUDA를 실행 검증하지 않았다.**

### 3.4 학습 전 Data/Action Alignment Gate

다음이 모두 있어야 실제 BC batch를 허용한다. 파일 부재를 synthetic zero로 채워 Gate를 통과시키지 않는다.

1. immutable manifest: dataset/source hash, source episode ID·reset group, control mode, action schema, timestamp/frame_idx, input provenance, train/validation/test episode lists.
2. observation_t의 Vision packet, robot-only geometry, proprioception과 target action_t를 **동일 episode/timestep**으로 join. 숫자 `ep000`이나 seed만으로 다른 수집계열을 결합하지 않는다. pixel source index는 episode identity가 아니다.
3. source별 pre/post-step label 계약, T/T+1, terminal row, initial-state repair를 검사한다. missing/duplicate/nonmonotonic timestep과 mode mismatch를 명시 오류로 처리한다.
4. 허용 입력은 Point/F=5/mask/node type/edge geometry와 robot state뿐이다. GT object pose/hole/goal/contact/phase/reward/success, expert/future action은 model input dictionary에서 제외한다. GT는 독립 audit/비교 label에만 둔다.
5. episode-local 연속 구간과 실제 관측 간격을 확보한다. 대표 8 frame의 gap, 두 ratio 중복, terminal crossing을 허용하지 않는다. 누락된 observation을 stale geometry로 보충하지 않는다.
6. train split만으로 normalizer를 fit하고 validation/test에서 freeze. test는 checkpoint 선택/HPO에서 제외한다.

91개 episode를 선택하면 같은 episode의 RGB-D/mask/F=5/Raw EDE가 추가로 필요하다. Vision ep000을 학습 대상으로 선택하려면 별도의 absolute-pose 학습 계약·여러 독립 episode split·실제 robot 필드가 필요하므로 현재 delta BC baseline 범위를 확대하게 된다. **이번 계획은 기존 delta-pose BC 규약을 유지하며 이 우회 경로를 자동 채택하지 않는다.** 전체 episode Vision 재처리/새 수집/replay는 이번에도, 단순 구현 승인만으로도 자동 실행하지 않고 별도 데이터 준비 승인 범위로 둔다.

## 4. 최소 통합 설계

### 4.1 책임 분리와 제안 API

```text
stored Vision packet (unchanged) ── receiver ── interaction construction/tensor view
same-time RobotObservation ────────┘                         │
                                                     EdgeMLP GNN
                                                           │
                                  [TCP,L,R,PegMean,BoxMean] = z_G[5d]
same-time robot_state[qvel9] ───── normalization ─────────────┤
object_present[2] ───────────────────────────────────────────┤
                                                    concatenation
                                                           │
                                            existing RNNPolicy(GRU)
                                                           │
                                         existing 7D regression BC output
expert action_t ─────────────────────────────────────── loss only
```

제안 interface는 `construct_graph(packet, robot_observation, connectivity_cfg) → GraphTensor`, `collate_graphs(graphs) → PackedGraphBatch`, `encode_graph(batch) → (z_G[G,5d], object_present[G,2])`다. 실제 모델 호출은 `forward_sequence(packed_frames, frame_map, robot_state[B,T,9], lengths, h0=None) → (raw_action[B,T,7], h_n)`와 `step(graph_batch, robot_state[B,9], h=None, reset_mask=None) → (raw_action[B,7], h_next)`로 감싼다. 아직 존재하는 API가 아니다.

Graph module에는 robot_state tensor나 action target을 넘기지 않는다. RobotObservation에는 robot geometry와 source identity만 전달하고 qvel/proprioception은 별도 인자로 fusion 단계에서만 받는다. CPU 저장 자료부터 시작하며 PyTorch 기본 `index_add`, tensor indexing, stable sort, `nn.Embedding/MLP/GRU`로 충분하다. PyG/torch-scatter/새 의존성은 필요하지 않다.

### 4.2 Node / tensor schema

| 필드 | shape / dtype | 의미 |
|---|---|---|
| `node_pos_ee` | `[N+3,3]` float32 | robot3 앞, packet object point 행은 원래 순서 그대로. metres/EE. |
| `geometric_features_raw` | object `[N,5]` float64 | 원본 packet/audit에서 유지. robot 원본 view를 확장하면 NaN5와 mask false. |
| `field_available`, `field_valid` | `[N+3,3]` bool 각각 | packet `feature_field_available/valid`의 3개 field(normal vector, variation, depth). 5개 scalar mask로 오해하지 않는다. robot은 둘 다 false. |
| `finite_geometric_features` | `[N+3,5]` float32 | network 전용 view. field mask를 `[3,1,1]` 폭으로 broadcast해 사용. |
| `node_type` | `[N+3]` int64 | TCP0 / Left1 / Right2 / Peg3 / Box4. 기존 receiver enum과 explicit mapping. |
| `object_id`, `source_index`, `point_row` | node별 int64 | object는 기존 0/1, source index/원래 row 보존. robot은 해당 값 -1. model geometry selection에 source evidence/GT를 추가하지 않는다. |
| `edge_index` | `[2,E]` int64 | **source→destination**. `+3`는 receiver에서 단 한 번 적용. |
| `edge_attr`, `edge_type` | `[E,4]` float32 / `[E]` int64 | distance+unit direction / intra0, robot-object1, peg-box2. |
| `object_present`, `packet_observed` | `[2]` bool 각각 | readout에 point가 있는지와 packet.valid 원본을 분리. packet.valid는 observed support 존재이며 selected point 수와 의미가 같다고 일반화하지 않는다. |

finite view는 field별 `available & valid`인 값만 정규화해 사용하고 나머지는 **정규화 후** 0으로 둔다. 유효하다고 표시된 값이 nonfinite이면 입력 오류로 중단한다. `NaN * 0`은 NaN이므로 사용하지 않는다. 유효한 true zero와 unavailable/invalid는 두 mask의 차이로 구분한다. 원본 NaN/mask는 고치지 않는다.

Node encoder 입력은 `concat(pos3, finite_F5, available3, valid3, node_type_embedding)`이다. 두 mask를 실제 MLP 입력에 넣고 unit test에서 소비 경로를 확인한다. robot의 finite zero normal은 missing-value placeholder이며 유효한 영벡터 normal이 아니다. Smoke에서는 identity normalization을 명시하고, 실제 학습에는 train-only valid-field 통계와 기존 train-only/std-floor 원칙을 적용한다. normal/radian/metre의 원본 정의와 단위를 바꾸지 않는다. 새 clipping/outlier 제거는 없다.

Object identity, point 수, row/source/geometry/F=5는 불변이다. `packet_observed`는 원본 진단으로 보존하고 readout 존재 mask `object_present=(count>0)` 2개를 별도 fusion 입력으로 소비한다. 두 mask의 차이가 있으면 보고한다.

### 4.3 Edge construction

**Intra-object:** Task05-E raw packet의 `intra_object_edge_index`를 receiver가 +3한 결과 그대로 첫 edge block에 둔다. 순서/양방향/row metadata를 보존하고 Graph에서 후보나 component를 재계산하지 않는다. 나머지 interaction만 append한다. robot-robot kinematic edge, self-loop, 새로운 surface edge는 추가하지 않는다.

**Robot–Object:** robot 3개 각각 object 0/1에 대해 EE 3D squared Euclidean distance를 계산하고 `min(k,N_obj)`개를 선택한다. k는 양의 정수 config다. 빈 object에는 선택하지 않는다. pair마다 정방향/역방향을 한 번씩 만든다. 최대 directed 수는 `6×(min(k,N_peg)+min(k,N_box)) ≤ 12k`다.

동일 거리 정렬 key는 `(distance_squared, source_index, original_point_row)` 오름차순이다. 원본 float32 좌표를 float64로 승격해 CPU에서 squared distance와 stable lexicographic order를 계산하는 기준 경로를 둔다. tolerance bucket이나 epsilon perturbation을 넣지 않는다. 최종 node row를 정렬해 바꾸지 않고 선택 index만 결정한다. 비동일 거리 near-tie를 같은 거리로 합치지 않는다. GPU selection 최적화는 범위 밖이며 나중에 추가할 경우 이 reference와 대조한다.

**Peg–Box:** 전체 Cartesian pair에서 전역 `q=min(M,N_peg×N_box)`개를 선택한다. M은 0 이상 정수이고 `M=0`/한쪽 empty이면 0개다. canonical pair는 `(Peg row, Box row)`이며 tie key는 `(distance_squared, Peg source_index, Peg row, Box source_index, Box row)`다. 두 방향으로 확장하고 중복은 interaction block 안에서 deterministic하게 제거한다. global top-M은 Peg별 top-M이나 reciprocal nearest가 아니다.

최소 구현은 CPU exact block scan을 사용한다. P=N_peg, Q=N_box, B=block point-row 수라 할 때 거리 계산은 **O(PQ)**, 임시 거리 buffer는 **O(BQ)**다. block의 pair를 안정 정렬해 상위 min(M,block_pairs)를 유지하고 이전 best-M과 같은 total-order key로 merge하면 full matrix sort와 동일한 전역 선택을 얻는다. pair enumeration이 유일하므로 원래 중복은 없고 양방향 변환 뒤 invariant로 재확인한다. 단순 block sort 시간 상한은 `O(PQ log(BQ) + ceil(P/B)×M log M)`, selection memory는 `O(BQ+M)`이며 작은 integer source/row 배열과 sort workspace가 추가된다. M이 PQ에 가까우면 output 자체가 O(PQ)라는 점도 기록한다. M=0이면 scan을 생략한다.

기존 저장 범위에서 ratio0.5 최대 P×Q는 **499×451=225,049**, ratio1은 **896×798=715,008** pair다. float64 distance matrix 하나는 약 1.72/5.46MiB지만 index/sort workspace는 별도다. block_rows=64이면 최대 distance buffer는 ratio1에서 64×798×8=408,576 bytes(약 .39MiB)다. broadcast xyz temporary도 block 안에서만 생성한다. 121-frame selection의 N 범위는 ratio0.5 327..958, ratio1 552..1712지만 완성 Raw packet의 범위는 각각 371..950 / 630..1694다. 한 frame의 비용을 전체 episode batch에 곱해 materialize하지 않는다. 이 수치는 크기 산술이며 실행 시간/peak memory 실측이 아니다.

모든 `i→j`의 feature는 `delta=p_j−p_i`, `d=||delta||₂`, `u=delta/d if d>0 else zero3`, `[d,u]`다. edge type은 별도 embedding이다. zero-distance도 node를 합치거나 edge를 버리지 않는다. reverse edge는 동일 d/반대 u, zero pair는 양쪽 zero3다. NaN 방지를 위해 안전한 분모를 사용하고 d=0 분기를 적용한다. feature 연산 dtype/tolerance를 기록하며 reverse 검사와 기준 float64 비교를 계획한다. Normal/depth/mask 필터나 contact probability는 없다.

### 4.4 Ragged graph batching과 structured readout

Graph마다 N/E가 달라 `[B,N,F]`로 강제하지 않는다. frame graph들을 disjoint union으로 pack한다.

- node/edge tensor를 concatenate하고 `ptr[G+1]` 누적 node offset으로 각 graph edge에 batch offset을 더한다. robot+3 offset과 batch offset을 혼동하지 않는다.
- `node_batch[ΣN]`, `edge_ptr[G+1]`, `robot_index[G,3]=ptr[:-1,None]+[0,1,2]`, object/type/source/provenance를 함께 보존한다.
- 모든 edge의 source/destination `node_batch`가 같음을 검사한다. Packed batch 결과와 graph별 독립 forward 결과를 대조한다.
- `frame_map[g]=(batch_slot,timestep_slot)`으로 real frame readout을 `[B,T,5d]`에 scatter한다. padded timestep에는 fake point/graph를 생성하지 않고 sequence mask/length로 분리한다.

각 message passing layer는 요청한 수식을 그대로 따른다.

```text
src=j, dst=i
m_ji = shared_MLP_msg([h_i, h_j, edge_attr_ji, edge_type_embedding])
mbar_i = index_add(messages at dst) / max(in_degree_i, 1)
h_i_next = h_i + MLP_upd([h_i, mbar_i])
```

세 edge type이 layer 내 하나의 Message MLP를 공유한다. zero-degree node의 mbar는 zero다. E=0일 때도 `[0,d]` message와 finite update를 처리한다. 기존 `MessageLayer`의 추가 LayerNorm/ReLU 출력 경로는 자동 상속하지 않는다. GAT/Transformer/EGNN, edge-type별 별도 network는 도입하지 않는다.

Readout은 **`z_G=[h_TCP,h_L,h_R,mean(h_Peg),mean(h_Box)]`**, 정확히 `[G,5d]`다. object별 segmented `index_add`와 count clamp를 쓰며 count0은 정확한 영벡터로 유지한다. 그 뒤 bias가 있는 readout projection으로 zero slot을 바꾸지 않는다. robot3 slot은 항상 남는다. 전체 node global pooling은 없다. Frame18/56 Box empty, 단일 point, 두 object 모두 empty를 검사한다. empty object 관련 robot-object/peg-box edge는 없어야 하며 stale geometry는 추가하지 않는다.

### 4.5 Robot State / GRU / Action의 최소 연결

첫 후보 `robot_state_schema=graph_v2_proprio_qvel9`는 기존 v2의 실제 별도 state 입력을 재사용한다. state 차원은 **9**, finger joint velocity 포함이며 전체 qpos/gripper command를 graph에 복사하지 않는다. 새 state 필드를 추측해 더하지 않는다. 더 넓은 proprio schema가 필요하다는 후속 근거가 생기면 별도 변경·비교로 다룬다. qpos가 원본에 있다고 해서 현재 v2 proprio가 qpos+qvel18이라고 설명하지 않는다.

91-episode data에서 `qvel_dataset`의 ordering 검증 및 `StateNormalizer`의 proprio 모드 통계를 재사용한다. 이 모드는 goal geometry를 참조하지 않는다. 새 wrapper는 robot-only tensor 경로로 기존 평균/표준편차 적용을 연결하고 dummy GT object를 만들어 normalizer에 공급하지 않는다. 저장 Vision ep000에 qvel이 없으므로 실제 fusion 테스트는 Gate에서 막고 합성 state fixture로만 대체 가능하다.

fusion은 단순 `concat(z_G, normalized_qvel9, object_present2)`로 `[B,T,5d+11]`이다. observation mask는 state 자체가 아니라 별도 readout metadata다. `RNNPolicy`의 기존 입력 projection MLP를 포함해 그대로 재사용한다. d를 바꾸면 obs_dim만 산출하고 기존 GT obs_dim38을 하드코딩하지 않는다.

GRU state 계약은 `[num_layers,B,H]`, `h0=None`이면 PyTorch zero init이다. 실제 `configs/models.yaml`의 hidden144/layers1/enc_hidden128을 재사용 후보로 명시한다. `step`은 T=1로 감싼 동일 모델을 호출하고 h를 다음 실제 observation으로 전달한다. episode reset/terminated/truncated 또는 batch slot의 episode 교체 전에 해당 slot h를 0으로 reset한다. observation 한 장을 여러 번 넣는 refinement는 sequence로 세지 않는다.

승인 후 첫 sequence smoke는 episode 경계를 넘지 않는 equal-length contiguous chunk로 단순화한다. padding을 지원할 때는 실제 길이까지만 GRU를 진행하거나 length-aware 경로로 최종 hidden을 얻는다. loss mask만 붙이고 padded transition을 실제 hidden으로 반환하지 않는다. sequence 전체 forward와 같은 초기 h의 step 반복이 일치하는지 검사한다. 독립 shuffled chunk는 h0=0임을 명시하고, stateful chunk는 같은 episode의 바로 다음 chunk에만 h를 전달한다. inference/no-grad와 TBPTT chunk 경계에서 `recurrent.models.detach_state`를 사용하되 chunk 내부는 GNN까지 gradient를 유지한다.

Action은 기존 Graph BC의 **`pd_ee_delta_pose`, 7D continuous regression**을 유지한다. raw head는 normalized controller command `[translation3, rotation3, gripper1]`이고 loss는 `graph.prototype.v2.models.bc_loss(..., gripper_mode='regression')`의 7성분 평균 MSE다. 실제 valid timestep만 모아 호출해 padding을 loss에서 제외한다. gripper를 logits/thresholded binary로 바꾸지 않고 target은 원래 -1..1 controller 값이다. arm ±1↔±.1m, rotation controller 설정/부호, gripper -1↔-.01m / +1↔.04m의 기존 convention을 따른다. raw prediction을 loss 전에 clip하지 않는다. 추후 rollout action 적용 시 기존 clip convention을 유지하며 실제 applied command를 별도 기록한다.

Recurrent 팀의 motion6 z-score+BCE/thresholding은 위 Graph regression과 별개다. GRU class를 재사용한다고 그 loss/normalization/38D GT feature builder를 함께 끌어오지 않는다. 기존 checkpoint도 node/edge schema·input dimension이 달라 그대로 load할 수 있다고 가정하지 않는다.

후속 checkpoint는 새 model state/config와 schema/action/state ID, train-only normalizer, frozen split/input hashes, seed, optimizer/scheduler/RNG를 보존한다. 초기 inference hidden은 checkpoint에서 이어받지 않고 episode별 reset한다. validation minimum checkpoint 선택과 test 분리는 기존 방식에 맞춘다. 과거 v4/Recurrent checkpoint format과 production evaluator를 변경하지 않고 필요한 wrapper만 opt-in으로 둔다.

## 5. provisional_smoke_config와 예상 크기

아래는 **아직 실행하지 않은 임시값**이다. 학습 성능상 최적값이나 최종 architecture 선택이 아니다.

```toml
[provisional_smoke_config]
L = 2
d = 64
k = 4
M = 16
keep_ratio = 0.5
node_type_dim = 4
edge_type_dim = 4
pair_block_rows = 64
gru_hidden = 144
gru_layers = 1
gru_enc_hidden = 128
device = "cpu"
normalization = "identity_for_contract_smoke"
robot_state_schema = "graph_v2_proprio_qvel9"
action_schema = "pd_ee_delta_pose_regression7"
sequence_batch = 2
sequence_length = 4
```

L2는 둘 이상의 message update 검증, d64는 tensor/gradient 비용을 작게 유지, k4는 기존 interaction 크기와 가까운 소수 연결, M16은 global pair 선택·양방향 검사를 가능하게 하는 값이다. GRU 크기는 기존 명시 config에서 가져온다. sequence2×4는 실제 연속 자료가 준비됐을 때만 real batch로 쓰고 현재는 synthetic 계약용이다. M0도 별도 계약 테스트한다.

기본 8 packet에서 node 수는 **374..953**, Raw directed intra E는 **1,990..5,174**다. 두 object가 충분하면 interaction 최대 `12×4+2×16=80`, 따라서 frame0은 N_total953/E_total5,254다. Frame18 Box empty는 N456/E2,622(2,598+24), Frame56은 N374/E2,014다. ratio1 stress frame0은 N1,697/E9,502다. z_G는320, mask 포함 fused dim은331, GRU h는 `[1,B,144]`, action `[B,T,7]`이다. 실제 k-NN/M pair 위치·선택은 승인 후에만 계산한다.

frame0의 float32 node h 한 개는 953×64×4=243,968 bytes, edge message 한 개는 5,254×64×4=1,345,024 bytes다. 두 layer·MLP activation·autograd·8 frame pack·optimizer가 추가되므로 이를 전체 peak로 부르지 않는다. Empty graph/edge라도 `5d`와 action shape는 바뀌지 않는다.

## 6. 승인 후 단계적 구현·검증 Gate

| 단계 | 승인 후 수행 / 확인 | 통과 의미와 중단 조건 |
|---|---|---|
| G0 입력 audit | 기존 source/packet/sidecar hash, 정책·ratio·F=5·좌표·row/source 보존; robot source/sequence manifest 확인 | Raw가 아니거나 provenance/shape/identity가 다르면 중단. 잘못된 입력을 재추정하거나 필터링해 통과시키지 않는다. |
| A synthetic construction 계약 | robot3, original point order/+3 once, raw intra exact, object별 kNN, global M, tie/source/row, duplicate/reverse/index/type/attr4, zero distance | independent 작은 brute-force oracle로 global M 비교. k>N/M>PQ/M0, empty 둘/한쪽, invalid mask, symmetric distance를 포함. GT object pose/action을 바꿔도 graph selection이 변하지 않아야 한다. |
| B 저장 Packet forward | 8×ratio0.5와 ratio1 최대 N, real Point/F=5/Raw edge tensor→GNN, empty18/56, finite layer output, zero-degree, structured5d | robot이 없으면 **REAL_PACKET_WITH_SYNTHETIC_ROBOT_CONTRACT**로만 표기. object-only 진단도 완전 3-node graph 통합과 구분. 실제 관측 3-node forward PASS는 same-time real geometry 확보 후에만 가능. |
| C sequence / GRU | variable-N 다중 graph pack, object mask+별도 state concat, `[B,T,7]`, state carry/reset, full-sequence↔step 비교, mixed episode reset, padding 최종 h | 현재 저장 8 frame은 정규 시계열 Gate 미충족. 합성 시간축 검증과 실제 contiguous sequence PASS를 구분한다. hidden mismatch/nonfinite/index cross-graph이면 중단. |
| D backward smoke | §3.4 데이터 Gate 충족 시 작은 train-only real batch의 기존 BC loss→backward→단 한 번 optimizer step; GNN node/msg/update/type embeddings, GRU, head gradient 확인 | finite gradient와 각 모듈의 적어도 일부 nonzero gradient, 실제 parameter update 검사. zero-degree/missing feature 때문에 모든 개별 parameter가 nonzero여야 한다고 요구하지 않는다. optimizer/model state는 smoke용으로 독립 생성. |
| D 대체 경로 | 실제 aligned batch가 없으면 synthetic state/target/sequence로 같은 연산·gradient 계약만 검사 | **SYNTHETIC_BACKWARD_CONTRACT_ONLY**. 실제 delta label loss, real sequence 또는 BC 통합 성공 수치를 만들지 않는다. |
| E 결과 검증 | 저장 JSON/NPZ readback·finite/shape/byte/NaN 위치·source hash, PNG decode/시각 검토, HTML 상대 링크, 실행/미실행 상태 | 보고서 오류는 같은 run의 수치 보존 후 report-only 복구. 수치와 config를 바꿔 PASS로 덮지 않는다. |

표의 실제 데이터 Gate는 §3.4를 뜻한다. G0/A/B/C/D는 모든 테스트를 한 번에 PASS로 합치는 점수가 아니다. 승인 범위·자료 부재에 따라 `PASS`, `BLOCKED_DATA`, `SYNTHETIC_CONTRACT_ONLY`, `NOT_EXECUTED`를 개별 기록한다. Smoke PASS는 모델이 조작을 학습했다는 뜻이 아니다.

**중단 기준:** nonfinite tensor/loss/gradient, frame/action mismatch, missing real robot geometry, GT/future leakage, preserved point/feature/edge 변경, cross-graph index, unsupported backend 연산 또는 memory failure. missing data는 해당 real 통합 단계만 멈추고 synthetic 결과로 격하해 명시한다. 점 수 감축·새 cutoff·Strict 복원·loss 변경·CPU fallback 은폐·자동 데이터 재처리로 해결하지 않는다.

### 6.1 Latency / memory 측정의 경계

이번 plan-only 단계에서는 CPU/GPU benchmark를 실행하지 않았다. 후속 승인 범위에 포함될 때만 correctness 완료 후 측정한다.

- 별도 구간: packet disk load/decode, receiver validation+copy, robot join/EE transform, robot-object selection, global-M selection, attr/tensor packing, host→device, GNN layer별/총합, readout, fusion, GRU step/sequence, head. backward+optimizer는 forward inference와 분리한다.
- 저장 Packet upstream sampling/PCA5/EE/Raw EDE 생성 비용은 **제외**라고 표시한다. Task05-E 8.261ms 등 과거 수치를 새 경로 시간과 더해 end-to-end 실측이라고 부르지 않는다. I/O/그림/log/HTML 생성도 core 밖이다.
- device/dtype/threads/N/E/L,d,k,M/B,T/train-or-eval/inference_mode/warmup/repetition을 기록한다. CPU wall/process time, GPU는 sync된 host wall과 지원되는 event time을 구분한다. transfer 포함/제외, cold/steady-state를 분리하고 raw samples/mean/median/p95/max를 보존한다.
- memory는 tensor logical bytes, model parameter/optimizer bytes, CPU baseline/peak RSS, GPU allocated/reserved/peak(지원 시)와 측정 제한을 구분한다. MPS API의 boundary allocation을 실제 peak라고 쓰지 않는다.
- GPU는 대상 host와 별도 benchmark 승인 후에만 한다. 현재 CPU/MPS/CUDA forward/benchmark 모두 **NOT_EXECUTED**다. 비용 추정과 smoke 실행 시간으로 20Hz나 latency 보장을 주장하지 않는다.

## 7. 최소 변경 파일 제안

전면 재작성 대신 team-owned opt-in package로 기존 GT prototype과 새 Vision schema를 분리한다. 아래는 **후속 구현 예상 파일이며 이번에는 생성하지 않는다**.

| 파일 | 최소 책임 |
|---|---|
| `graph/integration_baseline/__init__.py` | 명시 opt-in package. 기존 import/default routing 불변. |
| `graph/integration_baseline/graph.py` | tensor dataclass/finite-mask view, receiver 호출, deterministic exact kNN/M, attr4, disjoint-union collate와 object readout indexing. |
| `graph/integration_baseline/model.py` | 작은 요청 수식의 EdgeMLP encoder, raw structured5d, state+mask concat, 기존 `RNNPolicy` sequence/step/reset wrapper. |
| `graph/integration_baseline/data.py` | 저장 NPZ reader 연결, same-time robot whitelist, packet/episode/action manifest join·alignment Gate, ragged sequence frame_map/length 처리. 데이터 생성기는 포함하지 않음. |
| `graph/integration_baseline/config.toml` | provisional_smoke_config, schema/policy/source refs, L,d,k,M 독립값. production default와 HPO grid 없음. |
| `graph/integration_baseline/smoke.py` | 승인된 construction/forward/backward scope별 실행, 공통 Logger, 독립 JSON/NPZ/PNG 및 간단 정적 HTML 생성. 학습/rollout launcher로 자동 연결하지 않음. |
| `graph/integration_baseline/tests/test_baseline.py` | construction brute-force oracle, schema/mask/empty/packed/GRU state/loss alignment/gradient 계약. simulator/GPU를 요구하지 않는 기본 fixture. |
| `docs/visual-graph-integration/00_interface.md` | 승인 후 opt-in 모델 입력/robot/state/action/시간 계약만 보완. 현재 dirty 내용을 보존하며 기존 receiver 계약을 덮지 않음. |
| `docs/visual-graph-integration/06_graph_gnn_gru_baseline.md` | 구현 후 실제 결과·상태와 artifact 상대 링크. 지금은 만들지 않음. |

기존 NPZ `schema_json` reader는 `visual.runtime.test.run_vision_feature_edge.read_packet`이다. 처음 저장-fixture smoke에서 그대로 호출할 수 있다. 모델 runtime이 평가 entrypoint에 의존하는 것이 문제가 되면, 동일 serializer/reader를 `visual/runtime/vision_packet.py` 등에 공용 I/O로 이동하는 **작은 공유 확장과 roundtrip 계약 테스트를 먼저 검토**한다. format/geometry 알고리즘을 새로 만들지 않는다. 이 추가 변경은 필수 core 7파일과 구분한다.

`graph/vision_input.py`는 현재 3행/+3 요구를 이미 지원하므로 필수 수정 대상으로 잡지 않는다. `graph/prototype/{graph_builder,models,train}.py`, `v4/`, `recurrent/{models,train,evaluate}.py`, `envs/`와 공통 Logger/Publisher도 기존 인터페이스를 재사용하며 기본 수정 대상이 아니다. 실제로 발견한 interface gap이 있을 때만 작은 opt-in 확장과 기존 회귀 테스트를 추가한다. training/eval full entrypoint는 데이터 Gate와 별도 학습 승인 이후에만 설계·연결한다.

## 8. 후속 학습 / HPO / 공정성의 경계

Config는 `graph.message_passing.L/d`, `graph.connectivity.robot_object_k/peg_box_M`, `vision.packet_policy/keep_ratio`, `robot_state.schema`, `action.schema`, `sequence`와 `training_budget`, `split_manifest`, `seed`를 독립적으로 snapshot한다. `provisional_smoke_config`와 나중에 검증된 training config를 별도 이름으로 기록한다. L,d는 양의 정수, k≥1, M≥0을 검사한다. M0은 정상 ablation이다. config mismatch checkpoint 재개를 막는다.

1. **Phase1 connectivity:** 나머지 임시 설정·동일 episode/action/state/optimizer schedule을 고정하고 k,M을 비교한다. M0 포함. 작은 범위/예산은 추후 승인으로 정한다.
2. **Phase2 capacity:** 선택된 연결 설정을 고정하고 L,d만 비교한다. parameter 수/학습·추론 비용도 기록한다.
3. **Phase3 interaction:** 남은 소수 k,M,L,d 조합만 확인한다. validation으로 선택하고 test는 최종 비교에서만 읽는다.

Grid search, 장기 학습, 새 objective/goal graph/GCSL/pretraining은 실행하지 않는다. Minimal baseline 검증 후에도 실제 BC 학습은 별도 승인 대상으로 유지한다. 실제 batch가 없으면 synthetic smoke 완료와 data blocker를 보고하고 끝낸다.

추후 비교는 Flat vs Graph, feed-forward vs GRU, 필요 시 Raw vs Strict, M0 vs M>0, GT vs Estimated로 분리한다. 가능한 한 동일 demonstration episode/source reset group, frozen split, action target/loss, robot state fields/normalization, train budget/selection rule/evaluation seed를 사용한다. Graph와 Flat의 point/feature/mask 정보량 차이가 있으면 기록한다. GT-based geometry와 estimated Vision geometry는 source ID와 schema에서 구분하고 GT object pose가 state bypass로 들어가 비교를 오염시키지 않도록 한다.

지표는 validation BC loss와 component별 action prediction error, 별도 승인된 closed-loop success/collision/failure, inference latency, GPU memory/parameter 수다. action MSE 감소를 조작 성공으로 해석하지 않는다. 현재 `graph.prototype.metrics.action_metrics`와 기존 episode evaluation/ExperimentLogger 기록 패턴은 재사용 후보지만, 기존 evaluator의 GT builder/goal31은 Vision adapter 없이 그대로 실행할 수 없다. collision metric이 실제로 계측되지 않았다면 N/A로 남긴다.

## 9. 후속 결과 Markdown / 정적 HTML / 재현성

후속 승인된 실행에서만 `ExperimentLogger(team='graph', model='vision_graph_gru_baseline', job_type='evaluate', config=실제설정, seed=...)`를 생성한다. Logger의 UUID4를 사용하고 W&B project/group/ID 규칙을 그대로 따른다. local/offline/disabled 기능을 유지하며 별도 W&B logger는 만들지 않는다. 이번에는 Logger 인스턴스·UUID·결과 HTML·Forward Output·benchmark artifact를 생성하지 않았다.

예정 결과:

- Markdown: `docs/visual-graph-integration/06_graph_gnn_gru_baseline.md`
- HTML: `experiments/<run_id>/graph_gnn_gru_baseline.html`
- Markdown→HTML: `../../experiments/<run_id>/graph_gnn_gru_baseline.html`
- HTML→Markdown: `../../docs/visual-graph-integration/06_graph_gnn_gru_baseline.md`

HTML은 외부 CDN 없이 local CSS/HTML/table/PNG로 동작하는 간단한 정적 문서다. 실제 HTML을 생성한 뒤 상대 링크·누락 asset·PNG decode와 시각 검토를 수행한다. 미실행 browser/device 검사는 PASS로 기록하지 않는다.

| HTML 필수 내용 | 계획한 표시 |
|---|---|
| schema와 연결 | 실제 node/edge/type/offset, object별 N/E와 degree, Raw EDE+robot-object+Peg-Box를 색으로 구분한 동일-view 예시, robot 출처/EE frame |
| attr/mask/empty | distance/direction/reverse/zero-distance 검사, field_available vs field_valid, NaN 원본과 finite model view, Frame18/56 Box empty 및 object_present |
| 모델 tensor | layer별 h/message/update shape, 다섯 readout embedding과5d, graph+proprio+mask fusion dimension, GRU B/T/H/reset trace |
| forward/backward | action dimension/convention/loss, finite/gradient/optimizer step, sequence/step 일치, cross-batch index와 label alignment Gate |
| 데이터 신뢰 경계 | REAL_ALIGNED / REAL_PACKET_WITH_SYNTHETIC_ROBOT / SYNTHETIC_CONTRACT_ONLY를 표·그림·수치마다 표시. 미확보 robot/sequence/action은 BLOCKED_DATA |
| 비용과 한계 | 승인된 CPU/GPU stage latency/memory와 측정 경계; 미실행은 null/NOT_EXECUTED. 미구현·미검증, 다음 k,M/L,d ablation |

동일 UUID에 `config.yaml`, `metadata.json`, `metrics.json`, `manifest.json` 외에 명시적 input references/hashes, graph/tensor 검사 JSON, loss/gradient/shape JSON, 보존이 필요한 NPZ/PNG, test log, `verification.json`, `source_hashes.json`, `source_snapshot/`, before snapshot, `implementation.diff`를 둔다. dirty/untracked 구현 파일도 명시 목록으로 snapshot/hash에 포함한다. 기존 Task03~05 source artifact/sidecar는 read-only dependency로 참조하고 소유권을 새 run으로 바꾸지 않는다. runtime 측정이 승인되면 raw samples/config/device/thread 정보도 같은 UUID에 둔다.

새 결과만 `logger.link_output`으로 등록한다. HTML/PNG/log/diff는 reports, JSON은 metrics, NPZ는 trajectory 등의 기존 routing을 따른다. 원본 checkpoint를 나중에 참조하면 기존 sidecar를 보존하는 `checkpoint=True` 연결을 사용한다. publisher는 `utils.experiment_artifacts`와 `tools.publish_experiment`만 사용하며, **외부 게시·ZIP dry-run·사후 등록도 이번에는 수행하지 않는다**. HF 실패는 별도 publishing 작업에서 처리하고 성공한 로컬 smoke/학습 결과와 분리한다. credentials/전체 환경변수는 config/log/manifest에 넣지 않는다.

## 10. 이번 plan-only 작업의 완료 범위

완료한 것은 repository/source 및 기존 저장 JSON/NPZ/H5/PT의 읽기 조사와 이 계획 작성이다. 확인한 주요 수치는 91개 converted episode/13,958 action/73·9·9 split, Vision 1 episode/121 frame, 대표 Raw packet16개와 robot/state/action 결손이다. 이는 조작 성능·forward·backward 검증 결과가 아니다.

문서 정적 확인: 로컬 링크 25개 존재, code fence 짝, `git diff --check` 통과. 기존 dirty/untracked 파일과 위 6개 source run의 JSON/NPZ/Python/diff/config 등 1,730개 보호 파일의 경로+내용 SHA-256 집계가 작성 전후 동일했다. 기존 source/sidecar를 등록하거나 수정하지 않았다. 설치 metadata는 ManiSkill3.0.1/SAPIEN3.0.3/Torch2.14.1이며 backend 실행 증거와 구분한다.

실행하지 않은 항목: 신규 Graph/GNN/GRU 코드, forward/backward/optimizer step, 테스트 suite 실행, 학습/HPO, CPU/GPU benchmark, simulator/rollout, Vision 재처리, dependency 설치, 외부 업로드, commit, production default 변경. 후속 검증 명령·그림·보고서는 실제 승인 범위와 데이터 Gate 충족 여부에 맞춰 작성한다.

계획 제출 후 여기서 중단한다. 구현 승인 전까지 §4~9의 제안 동작을 실행하지 않는다.
