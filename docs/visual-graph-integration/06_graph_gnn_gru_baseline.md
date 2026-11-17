# Task06 — Graph–GNN–GRU/BC 최소 통합 및 Expert Replay

Run UUID: `c184b85dadf14db5953f11cdcacb6674`. **최소 실제 Prefix 경로 검증 완료, 전체 Replay Dataset은 BLOCKED_DATA.**

[Static HTML](../../experiments/c184b85dadf14db5953f11cdcacb6674/graph_gnn_gru_baseline.html) · [검증](../../experiments/c184b85dadf14db5953f11cdcacb6674/verification.json) · [전체 결과](../../experiments/c184b85dadf14db5953f11cdcacb6674/summary.json) · [테스트 로그](../../experiments/c184b85dadf14db5953f11cdcacb6674/tests.log) · [Source/Diff](../../experiments/c184b85dadf14db5953f11cdcacb6674/implementation.diff)

## 실제 구현

Opt-in [패키지](../../graph/integration_baseline/README.md)에 Robot 3 Node, 원본 Point/F=5/Availability/Validity 보존, 기존 Raw EDE 수신, 물체별 Robot k-NN, exact global top-M, Attr4, disjoint-union Batch를 구현했다. GNN은 Node/Edge Type Embedding, 공유 Message MLP, destination mean과 residual MLP다. Readout은 `[TCP,Left,Right,mean(Peg),mean(Box)]` 그대로 5d이며 Empty Slot은 0이다.

`L=2,d=64,k=4,M=16,keep_ratio=0.5`는 임시 Smoke 설정이다. Node 입력에는 위치3+finite F5+available3+valid3만 들어간다. 원본 NaN/dtype/source/component/행은 보존하고 별도 finite view를 만든다. Consumer는 Delaunay/Strict Validation을 재계산하지 않는다.

Robot State는 **76차원**, Schema `panda_observable_robot_state_v1`다. qpos9/qvel9, Gripper position2/velocity2/opening1/opening velocity1, TCP·Left/Right Finger·Robot Base 각각 Pose7/linear velocity3/angular velocity3을 모두 포함한다. 실제 `get_proprioception()`의 모든 필드(qpos/qvel)를 사용하고 현재 Controller State는 빈 dict다. 중복을 제거하지 않았다. Gripper opening은 두 finger qpos의 합이며 pad 간 거리 추정이 아니다. Link 출처는 `agent.tcp`, `agent.finger1_link`, `agent.finger2_link`, `agent.robot.get_root()`이며 각각 `.pose.raw_pose`, `.linear_velocity`, `.angular_velocity`를 읽는다. Gripper 필드는 같은 qpos/qvel의 마지막 두 값과 그 합이다. Robot link kinematics는 같은 Live API를 추론에도 사용한다. Joint effort/acceleration/contact는 기존 Proprioception에 없는 항목이며 추정해 추가하지 않았다. [필드·순서·단위·출처](../../experiments/c184b85dadf14db5953f11cdcacb6674/robot_state_schema.json).

기존 `StateNormalizer`의 mean/std 저장 구조와 `.001` std floor를 재사용한 schema-sized Adapter다. 확보한 Train Prefix만으로 Fit하고 Freeze했다. Validation/Test Episode는 수집하지 못했으며 이 Split의 통계/모델 성능을 검증했다고 주장하지 않는다. GNN 이후 `[5d, object_present2, normalized robot76]`=398차원을 기존 `RNNPolicy`(GRU hidden144/layer1/encoder128)에 전달한다. 기존 regression7 MSE를 재사용한다.

## Replay와 실제 Coverage

완주 0/91 Episode. 검증된 Prefix 3개, 관측 473개 / Target 470개. 전체 Replay Pilot Gate: BLOCKED_DATA. 나머지 88개 Replay는 실행하지 않았다.

Source는 기존 `trajectory.state.pd_ee_delta_pose.physx_cpu.h5`: **91 Episode / 13,958 Action**, Frozen Split **73/9/9**를 재확인했다. 원본 Initial State Repair와 기존 source episode mapping을 재사용했다. Snapshot Restore 허용오차 `2e-5`, 순차 Replay `1e-3`은 기존 `prepare_dataset.py` 기준을 그대로 유지했다.

Episode 0/1/2의 순차 Replay는 각각 t=175/117/181에서 qvel 최대 오차 0.0010336712/0.0011354163/0.0011414587로 실패했다. State-only 재실행도 동일하고, 원본 추출 설정의 Torch threads=4/seed=0도 같은 실패 시점/오차였다. 반면 저장 Snapshot 복원 최대 오차는 약 3e-7이다. 따라서 RGB-D 추가나 이번 thread 수가 원인이라는 근거가 없고 정확한 drift 원인은 미확정이다. Snapshot Teleport로 순차 Replay 실패를 숨기거나 허용오차를 완화하지 않았다. [진단](../../experiments/c184b85dadf14db5953f11cdcacb6674/replay_diagnostic.json) · [원본 Thread 설정 재검사](../../experiments/c184b85dadf14db5953f11cdcacb6674/before_snapshot/replay_threads4_diagnostic.json).

각 실패 Frame 및 이후는 저장 입력에서 제외했다. 유효 Prefix의 마지막 관측도 다음 Target이 없는 행으로 남겨 보수적으로 Target 470개를 확보했다. 실제 입력에 쓰는 Prefix는 기존 bound 내에서 source Observation/Action과 대응하며 두 번의 동일 Replay에서 Robot State/Geometry 오차는 0이었다. 완전 Episode 정합 성공으로 해석할 수 없다. [Coverage/Alignment](../../experiments/c184b85dadf14db5953f11cdcacb6674/replay_alignment_coverage.json).

RGB-D는 실제 CPU 환경 기본 Camera 128×128 두 개의 RGB, raw int16 mm depth(overflow/invalid 포함), floating metre view, CV Intrinsic/Extrinsic/OpenGL calibration을 기록했다. Raw mm 값은 보존하며 Camera-to-World CV는 Extrinsic의 역행렬이다. Robot-only 상태와 실제 rubber-pad collision centre를 같은 timestep에서 읽는다. Timestamp는 wall-clock이 아닌 `t/20Hz` simulation time이다. 원본 `a_t`를 재생성/정규화하지 않고 `o_t → a_t → step → o_(t+1)` 순서로 저장한다.

원본 TimeLimit은 100 step인데 Expert 저장 Episode는 그 이후도 계속된다. 실제/source truncated flag가 일치함을 검사하고 보존했다. Sequence Loader는 terminated/truncated 뒤를 가로지르는 sequence를 거부한다. 이번 실제 Smoke는 t=0..3이며 이 경계를 지나지 않는다. 후속 전체 학습 전 Timeout 의미의 별도 결정이 필요하다.

## Offline Vision Dataset

검증 Prefix의 **473/473 관측 전체**에 연속 Packet을 생성했다. RGB-only Grounding DINO와 forward SAM2를 기존 경로로 실행했다. 초기화는 현재 Prefix만 입력하고 현재 Frame 결정만 허용해, 기존 episode-wide late initialization의 미래 Box 참조 가능성을 제거했다. SAM2에도 매 시점 `num_frames=t+1`만 노출하고 한 Frame씩 전파한다. 설치 SAM2의 `min(num_frames,max_obj_ptrs_in_encoder)` 정규화 때문에 전체 영상 길이가 앞부분 1픽셀을 바꾼 첫 `vision_v1` 생성본은 거부·보존했다. `vision_v2`로 재생성하고 실제 RGB Prefix/미래내용변경/영상길이변경 간 앞부분 Mask exact 일치를 별도로 검증했다. [실제 인과성 검사](../../experiments/c184b85dadf14db5953f11cdcacb6674/before_snapshot/actual_mask_causality.json) · [거부 기록](../../experiments/c184b85dadf14db5953f11cdcacb6674/rejected_vision_v1_audit.json). 미래 Frame을 이용한 backfill·backward refinement·GT segmentation은 사용하지 않았다. 기존 SAM2 optional C extension이 없어 설치 Predictor가 hole-filling postprocess를 건너뛴 경고를 로그에 보존했다. Dependency 설치는 하지 않았다.

`RGB-D → predicted Mask → 기존 Shape-First/Surface-Second → 기존 PCA5/F5 → 같은 시점 EE → 기존 observed component → Raw EDE`를 재사용한다. Surface 추가량은 기존 Peg97/Box95이고 총 Point Budget/신규 Geometry Filtering은 없다. 공통 Producer가 실험 Adapter에 있으므로 opt-in Builder에서 기존 Helper를 직접 호출한다. 기존 Source를 이동/수정하지 않았다. [Dataset Manifest](../../experiments/c184b85dadf14db5953f11cdcacb6674/dataset_manifest.json), [Join 검증](../../experiments/c184b85dadf14db5953f11cdcacb6674/dataset_validation.json).

Missing/Invalid/Suffix는 Manifest에 구분한다. 전체 원본 91개 데이터셋이 완성된 것이 아니다. Source Hash+Episode+Timestep으로 RGB-D/State/Target/Graph를 Join하고 Transform·Raw Policy·Source Row·Split을 검사한다. GT Object/Hole/Goal, Reward/Success, Expert/Future Action은 모델 입력에서 제외한다.

## 검증 결과와 신뢰 구분

- **A — SYNTHETIC_CONTRACT_ONLY:** 작은 독립 brute-force oracle, tie/source/row, k/M overflow와 M=0, reverse/zero-distance, zero-degree/empty, +3/packed offset, GT/action 변경 비의존성, 실제 mask encoder 소비와 Readout 계약 PASS.
- **B — REAL_PACKET_WITH_SYNTHETIC_ROBOT:** Task05-E 기본 8 Packet + ratio1 stress 8 Packet의 GNN finite/Variable N/Mask/5d PASS. 기존 Packet의 Robot은 실제값이 없어 명시적 Synthetic Fixture를 사용했다. 실제 연속 시계열로 간주하지 않았다.
- **C — REAL_ALIGNED (검증 Prefix 한정):** 실제 Train Episode 2개 × 연속 4 Frame, Variable N/Packed mapping, Full Robot State와 Action7, 초기 nonzero Hidden의 Sequence↔Step max error `1.79e-07`, Episode/reset/terminated/truncated reset PASS.
- **D — REAL_ALIGNED (검증 Prefix 한정):** 기존 7D MSE `0.142219`, finite GNN Node/Message/Update·Type Embedding·GRU·Head gradient 및 실제 Parameter Update PASS. **Optimizer Step은 정확히 1회**. Synthetic Backward 대체를 사용하지 않았다.
- **BLOCKED_DATA:** 전체 91개 순차 Replay와 완전 Dataset, Validation/Test 실데이터 Coverage.
- **NOT_EXECUTED:** CUDA/MPS, 장기 BC/HPO, Closed-loop Policy Rollout/Success Rate, 외부 게시/Commit/Production 변경.

[Forward/Backward 수치](../../experiments/c184b85dadf14db5953f11cdcacb6674/real_forward_backward.json) · [Tensor/Gradient/Before/After NPZ](../../experiments/c184b85dadf14db5953f11cdcacb6674/real_forward_backward.npz) · [저장 Packet 결과](../../experiments/c184b85dadf14db5953f11cdcacb6674/stored_packet_forward.json).

사전 실행한 기존 GT Prototype 회귀 suite에는 자체 합성 overfit 단위 테스트(기존 모델 3종 × 80 step)가 포함돼 있었다. 이 결과는 [별도 Preflight Log](../../experiments/c184b85dadf14db5953f11cdcacb6674/preflight_legacy_tests.log)에 보존하고, 신규 통합 모델의 실제 데이터 1-step Smoke와 구분한다. 최종 suite에서는 해당 overfit 검사를 반복하지 않았다. 새 Expert 장기 BC 학습은 실행하지 않았다.

![실제 관측 Graph](../../experiments/c184b85dadf14db5953f11cdcacb6674/plots/graph_rgb.png)
![실제 RGB 기반 Mask](../../experiments/c184b85dadf14db5953f11cdcacb6674/plots/actual_masks.png)
![Feature/Mask](../../experiments/c184b85dadf14db5953f11cdcacb6674/plots/features_masks.png)
![Replay 오차](../../experiments/c184b85dadf14db5953f11cdcacb6674/plots/replay_alignment.png)

## 비용과 한계

CPU packet→action(inference, B=1,T=1, warmup3+10회) wall mean **6.676 ms**, p95 **6.756 ms**, max **6.779 ms**. Receiver/Robot k-NN/Global-M/Tensor packing/GNN/Readout/Fusion/Encoder/GRU/Head를 나눴다. 이 시간은 원본 RGB-D 수집·Mask·Offline Graph·IO를 제외한다. [단계별 Latency/Memory](../../experiments/c184b85dadf14db5953f11cdcacb6674/latency_memory.json).

수집과 Offline Vision의 실제 처리시간은 별도 Manifest에 Frame/Episode 단위로 남겼고 [집계/측정 경계](../../experiments/c184b85dadf14db5953f11cdcacb6674/data_pipeline_costs.json)에서 확인할 수 있다. Replay 시간은 저장한 마지막 반복의 reset/initial restore 이후 capture/check/step 구간이며 수집 전체 총시간이 아니다. SAM2 시간은 초기 검출/초기화가 포함된 Frame과 steady frame이 섞이며, RGB JPEG 파일 준비/모델 로딩/압축·IO는 별도로 계측하지 않은 overhead다. 수집/Offline Vision의 isolated peak memory도 NOT_EXECUTED다. 과거 Task05-E 수치를 합쳐 End-to-end 속도나 20Hz를 주장하지 않는다. 통합 RSS는 import/이전 backward를 포함하는 process high-water이며 isolated inference peak가 아니다.

![CPU stages](../../experiments/c184b85dadf14db5953f11cdcacb6674/plots/latency.png)

후속 BC 전 조건: 전체 순차 Replay drift의 원인/재현 환경을 해결하고 기존 Gate로 다시 Pilot·91개 Coverage를 검증할 것, Timeout/Reset 의미를 확정할 것, Validation/Test까지 같은 Causal Mask/Schema/Manifest를 확보할 것, 관측 Mask/Depth 오염과 Empty Object 분포를 검토할 것. 현재 Mask/Graph의 물리 정확도나 BC 성공률은 검증하지 않았다. 최소 계약 통과는 학습 성능의 근거가 아니다.

재현 명령과 책임 구분은 [패키지 README](../../graph/integration_baseline/README.md)에 있다. 이전 Task03~05 Source/Artifact/Sidecar와 기존 GT/Recurrent Prototype은 보호 SHA-256 검사로 불변 확인했다. [보호 결과](../../experiments/c184b85dadf14db5953f11cdcacb6674/source_protection.json), [Source Snapshot/Hashes](../../experiments/c184b85dadf14db5953f11cdcacb6674/source_hashes.json), [Input Hashes](../../experiments/c184b85dadf14db5953f11cdcacb6674/input_manifest.json).
