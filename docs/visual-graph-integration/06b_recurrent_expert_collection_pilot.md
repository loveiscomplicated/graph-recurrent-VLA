# Task06-B — Recurrent ScriptedExpert Direct Collection Pilot

Run UUID: `9fec4f02fc9d4172a48e97d5eff90abd`. **현재 Runtime 직접 수집 5/5 성공, 669/669 Action 정합.**

[Static HTML](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/recurrent_expert_collection_pilot.html) · [검증](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/verification.json) · [Pilot Manifest](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/pilot_manifest.json) · [Alignment/Coverage](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/alignment_coverage.json) · [비용](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/data_pipeline_costs.json) · [질문별 JSON](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/questions.json)

## 1. 기존 Source와 과거 기록

현재 branch `visual-graph`, HEAD `63c3f6c9fe0a4ac5b8e0096fc7adca20ca21e4f8`, 시작부터 dirty/untracked 작업이 있었다. origin/recurrent `a5acd129fd249f53223c2428018fea1656a647fe`는 git show로만 조사했다. checkout/fetch/설치/commit/외부 upload는 하지 않았다.

Expert는 [recurrent/expert.py](../../recurrent/expert.py)::ScriptedExpert, SHA-256 `3cc854654d16b2462886c64fa12f146dd8a51718913d354f86cd7d6e1c0bcee9`. origin/recurrent와 해당 파일 bytes가 같다. 기본 설정 max_trans=0.02m, max_rot=0.15rad, insert_speed=0.006m, gain=3.0을 그대로 사용한다. `reset()` 후 `act()`가 현재 live task를 읽는다. 기존 진입점은 recurrent.collect::run_episode, recurrent.evaluate::ExpertPolicy다. 기존 Collector는 noise=0.02/perturb_frac=0.5 기본값, 성공만 keep, T observations/T actions이므로 이번 계약에 그대로 쓰지 않았다. 신규 [opt-in Recorder](../../graph/integration_baseline/collect_recurrent_expert.py)는 noise=0, hole shift 없음, 실패 포함, T/T+1이며 원본 Expert를 수정하지 않는다.

`docs/reports/recurrent_main_20261023/`는 현재 working tree에 없고 origin/recurrent에 있다. [읽은 원본 보고서](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/source_snapshot/origin_recurrent/docs/reports/recurrent_main_20261023/README.md)와 [CSV 집계](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/historical_expert_evidence.json)를 보존했다. 과거 Expert **60/60 = normal30 + hole_shift30**, seed2000–2029를 조건별 재사용, horizon250, physx_cpu, obs state, corruption none. Hole shift는 transport 후10 step에 seed 짝홀 부호로 ±0.02m를 적용한다. 과거 100 이내 성공10/60, 성공 길이85–198이다. 250은 cap이며 실제 최대길이250이라는 뜻은 아니다. 원본 CPU fixture SHA `c62d0cc9c56706be49e050cfcfd8097693181a684d4a77bcff247ef7066a6a3d`는 로컬에 없으므로 이번에는 reset(seed)를 사용한다. 과거 report 평가 Torch2.14.1+cu130/Linux/Python3.11.17과 현재 host가 다르다. 원본 private JSONL/fixture는 내려받지 않았고 CSV·provenance 근거만 확인했다. 과거 정책11,340 Episode/84개 학습 실행은 이 Expert60회와 별도이며 이번에 학습 결과를 재현하지 않았다. Visual ScriptedPolicy5/5 및 공식 Converted91은 이번 수집·성공률에 합치지 않는다.

## 2. Expert 입력과 Student 입력 분리

Expert reset은 GT peg actor OBB, peg_half_sizes, 현재 TCP closing 축으로 grasp를 결정한다. act는 TCP world pose, peg world pose, goal_pose(peg 삽입 목표), peg half-length, 내부 phase/counter/lift_target을 사용한다. Goal은 hole task GT에 근거한다. qpos/qvel과 fingertip은 Expert의 직접 입력이 아니며, 현재 TCP 및 물리 Controller를 통해 간접 관계가 있다. RGB-D/과거 observation buffer/미래 frame은 읽지 않는다. 상태 기계는 reach→descend→close(8회)→lift→align(조건3회)→insert다. 첫 grasp/lift target과 counter만 과거 정보를 기억한다. 종료는 Expert 자체가 아닌 Collector의 info.success/terminated/truncated/horizon이다.

Student 입력은 같은 시점 predicted-mask RGB-D Graph와 robot-only76D다. GT peg/hole/goal, reward/success, Expert phase/action은 입력에 전달하지 않는다. phase는 `episodes/*.expert_diagnostics.json`에 진단 전용으로 분리했다. raw 저장파일도 GT object pose를 포함하지 않는다.

## 3. Action Schema

[Expert Action Schema](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/expert_action_schema.json)의 mode는 pd_ee_delta_pose, float32[7], translation3/rotation3/gripper1이다. native output→env.step tensor→실제 CombinedController entry가 exact 일치한다. Target adapter/clipping/z-score/gripper BCE는 추가하지 않았다. Task06의 기존 continuous regression7을 그대로 쓴다.

Translation은 root-frame normalized 값이며 Controller가 [-1,1] clip 후 ±0.1m로 scale한다. Rotation은 norm>1이면 norm1로 제한한 뒤 `rot_lower=-0.1`을 곱하고 **XYZ Euler** delta로 바꿔 root에서 left-compose한다. Expert는 relative quaternion의 shortest axis-angle를 계산해 부호 반전/0.1 나눗셈을 하지만 이것이 Euler와 수학적으로 동일하다고 주장하지 않는다. 원본 제어 알고리즘의 근사이며 이번 BC target 의미는 **실제로 받는 원본 Controller 명령**이다. Gripper -1→-0.01m, +1→0.04m mimic target. Native rotation 성분이 명목 Box ±1을 넘을 수 있다. 해당 행수는 표와 NPZ에 그대로 남겼으며 임의 정규화하지 않았다. 실제 scaled arm/gripper input을 별도로 저장하고 설치 Controller 수식과 검증했다. 물리 axis-angle target 자체를 다른 Controller에 이식하는 호환성은 NOT_TESTED다.

## 4. 환경과 수정 후 Pilot

PegInsertionSide-v1 / `panda_wristcam` / `BackendInfo(device=device(type='cpu'), sim_device=Device("cpu"), sim_backend='physx_cpu', render_device=Device("cpu"), render_backend='sapien_cpu')`. Control 20Hz, Physics 100Hz, 5 physics steps/control. [실행 조건](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/executed_conditions.json). shared default100은 그대로 두고 `TimeLimitWrapper(env.unwrapped,250)`만 적용했다. 첫 Task success 또는 terminated/truncated에서 즉시 중지하며 post-success tail은0이다. First success는 Action count 기준1-based, action index는0-based다. 이번 normal 5개는 과거 두 조건60개 평가와 다르며 전체 성공률을 확정하지 않는다.

| Seed | Success | First success step | Terminated | Truncated | Actions | Observations | Failure | native Box 초과 행 |
|---|---|---|---|---|---|---|---|---|
| 2000 | True | 139 | True | False | 139 | 140 | None | 44 |
| 2001 | True | 87 | True | False | 87 | 88 | None | 25 |
| 2002 | True | 153 | True | False | 153 | 154 | None | 45 |
| 2003 | True | 146 | True | False | 146 | 147 | None | 18 |
| 2004 | True | 144 | True | False | 144 | 145 | None | 3 |

최초 시도 UUID `5a71640e4a9f4036b109bcaea7c17824`는 5개 seed마다 첫 Action 후 Recorder hook 누락으로 중단됐다. reset이 Controller를 교체해 pre-reset hook이 없어졌으며 **Expert task 실패율이 아닌 관측 중단/censored attempt**다. 각1 action 적용/1 pre-observation 보존/0 action row 저장. 원본 counter는 audit 뒤 갱신되어0으로 남았고 [정정 annotation](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/aborted_attempts.json)에 실제 적용 수를 기록했다. 원본 실패 Artifact/Source/log는 보존했다. 사용자가 같은5 seed 재실행을 명시 승인한 후 현재 Pilot을 수행했다. 이 중단5회를 숨기거나 60/60 재현으로 표현하지 않는다.

## 5. 동시 기록 및 Robot76D

계약: `o[t] → expert.act() → env.step(exact a[t]) → o[t+1]`. 모든 Episode의 T actions/T+1 observations를 보존하며 마지막 observation에는 Target이 없다. 각 reset 이후 실제 Controller에 감사 hook을 설치한다. command entry에서 읽은 robot state/geometry와 저장한 pre-action state/geometry가 exact 일치하고 elapsed_steps=t다. reset/step 반환 RGB-D와 같은 물리 시점에서 독립 get_obs를 호출한 RGB-D가 exact 일치한다. capture 전후 elapsed counter가 같고 env.step 외 physics stepping/reset/teleport가 없다. 같은 시점 hand/base calibration, image와 robot 상태를 함께 저장한다. Timestamp는 t/control_freq로 산출한 simulation relative seconds이며 별도 PhysX clock 직접 측정이 아니다.

RGB uint8[128,128,3], depth raw int16[128,128]mm, float metre view, intrinsic_cv, extrinsic_cv(world→camera), inverse CV cam2world, cam2world_gl를 두 Camera 모두 기록한다. 설치 shader는 position*1000을 int16로 cast하므로 precision/range 손실이 있고 nonpositive/-32768 count를 보존한다. raw만으로 모든 positive overflow를 식별·복구할 수 있다고 주장하지 않는다. 원본 depth를 보존하며 nonpositive는 downstream invalid다.

Robot Schema [필드/순서/차원/단위/frame](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/robot_state_schema.json)는 Task06 저장 Schema와 **dict exact,76D**다. qpos9/qvel9(rad,m/rad/s,m/s), gripper pos2/vel2/opening1/openingvel1, TCP·left/right finger·base 각pose7/world linear3/angular3다. Pose xyz+wxyz, velocities world; gripper/joint는 joint frame이다. [기존 robot_observation](../../graph/integration_baseline/robot.py)의 observable accessor를 그대로 사용한다. 실제 fingertip은 finger collision pad center local offsets (0,±0.00758,0.04525)m를 link pose로 변환한다. 각 frame의 link pose에서 independently reconstruction 및 EE transform 일관성을 검사했다. 없는 effort/acceleration/contact는 생성하지 않았다. Controller state는 빈 dict이며 필드 변경은 거부한다. Nonfinite0.

## 6. Alignment와 Provenance Gate

669/669 Target에 대해 action bytes3경로, actual scaling, pre-state/geometry, timestep/episode/reset seed, 카메라·depth conversion/calibration, finite76과 boundary를 검사했다. RGB-D/Robot/action은 final NPZ와 **실행 중 append된 별도 HDF5**를 모든 dataset별 dtype/shape/bytes로 다시 비교했다. HDF5는 NPZ를 읽어 복제한 파일이 아니다. [독립 Reader](../../graph/integration_baseline/validate_direct_expert.py)는 source NPZ/H5 hash와 같은 gate를 다시 검사한다. Metadata에는 source hash/ref/config/runtime/backend/schema가 있다. 이전 Converted Replay gate2e-5/1e-3은 이 새로운 direct recording에 적용하지 않는다. Timestamp equality와 반복 camera read는 같은 live state의 동시 기록 근거이며 외부 센서 지연 측정은 아니다.

## 7. 실제 Vision Graph와 Coverage

성공 seed2000에서 **t0..15 prefix만** 처리했다. Status `PASS_PARTIAL_COVERAGE`, Packet 16/16, 두 물체 nonempty 1/16. 전체관측674 중 미처리658는 Missing/NOT_EXECUTED이며 성공적인 full-episode Vision dataset으로 표현하지 않는다. [Coverage/실패/시간](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/vision_coverage.json).

**학습용 두 물체 관측 Coverage는 BLOCKED다.** [Mask/Depth/Overlap 진단](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/mask_coverage_diagnosis.json)에서 Box mask가15/16 Frame 전체에서 Peg mask와 겹친다. 기존 `OrganizedGeometry.observed_mask`가 ambiguous overlap을 양쪽 support에서 제외하여 Box0 Point가 된다. t11만 nonoverlap valid support2/selected2 Point다. 이 관측된 결손은 depth invalid보다 overlapping predicted masks로 설명된다. 어떤 mask가 의미적으로 틀렸는지 GT 점수로 확정하지 않았으며, Box GT 복원/새 mask 알고리즘/수정은 하지 않았다. Forward t0..3은 실제 결손 Box presence0 상태를 통과한 것이므로 양쪽 물체를 충분히 보는 Sequence 검증으로 해석할 수 없다.

Overlay를 직접 확인하면 Peg로 분류된 Mask가 Robot도 포함하는 것으로 보인다. 이는 RGB와 예측 Mask의 시각적 관찰이며 GT segmentation 기반 정량 정확도 평가는 아니다. Raw Dataset은 정합됐더라도 현재 Mask 결과를 올바른 Peg/Box geometry로 간주해 학습을 시작할 근거가 부족하다.

Task06 `causal_masks`와 `packet_from_frame`을 수정 없이 재사용했다. 현재 Prefix RGB-only DINO 초기화, current-frame decision, SAM2 num_frames=t+1, one-frame forward/no backward/no backfill, GT segmentation 없음. Sampling Shape-First, keep_ratio0.5, optimized PCA5/F5, availability/validity, same-time EE, observed component, intra Raw EDE/validation none, source row/index와 variable N을 보존했다. 빈 물체는 presence/missing으로 남기며 GT로 복원하지 않는다. 이전 인과성 실험은 Task06 기록이며 이번에는 그 검증된 코드 경로를 호출했다. 새 prefix/future perturbation 실험을 반복하지 않았다. Mask/Graph 물리 정확도 점수는 GT와 새로 비교하지 않았다.

## 8. 기존 Graph–GNN–GRU/BC

기존 build_graph/pack_graphs/collate_sequences/IntegrationPolicy/EpisodeStream을 그대로 사용했다. Robot node3, F5+availability3+validity3, Raw EDE+3, robot-object k4, peg-box exact globalM16, object presence, L2/d64, readout5d, robot76, GRU144/layer1, regression7이다. [실제 Forward](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/sequence_forward.json) `PASS_REAL_SEQUENCE_FORWARD`. Timesteps [0, 1, 2, 3], output [1, 4, 7], node counts [253, 246, 241, 239], hidden carry max error 5.960464477539063e-08. Random initialized 기존 모델의 finite forward만 확인했으며 예측 성능을 뜻하지 않는다. 정규화는 과거 Task06 train-prefix에서 이미 frozen된 mean/std를 재사용하고 hash를 기록했다. 새 Pilot을 임의 train/test split하거나 새로운 학습 통계에 포함하지 않았다. Reset 검사는 실제 frame tensor에서 explicit identity/flag를 주입한 lifecycle contract이며 실제 Policy rollout은 아니다. Backward0/Optimizer0.

기존 ReplayDataset는 source H5·공식 split·replay gate에 묶여 있어 이 자료를 직접 읽지 못한다. 신규 DirectExpertReader는 독립 진단용이며 production loader/reset 정책을 변경하지 않는다. 실학습용 direct dataset loader와 source-kind별 provenance validation은 추가 구현이 필요하다.

## 9. 전체 Dataset 방향과 남는 결정

이번 5개는 feasibility pilot이며 성공률/seed/shape/camera/phase coverage와 최종 split 비율을 결정할 표본이 아니다. 원본 Expert default Collection의 action noise/perturbation을 쓸지 별도 결정하고, noisy label과 executed command가 다르면 둘을 분리해야 한다. 이 Pilot에서는 noise0로 exact 동일하다. Raw native Action7를 계속 쓸지와 명목 Box 초과 convention을 manifest에 고정해야 하며 자동 clipping은 금지한다.

새 dataset은 실제 reset 시작→첫 success/실제 termination까지 Transition과 final observation을 보존하는 방향이 맞다. 관리상 truncation은 physical reset과 별도 metadata로 저장하고 실제 reset/terminated에서 hidden을 reset해야 한다. Pilot은 horizon250 전에 성공한 경우 실제 timeout boundary를 실행하지 않았으므로 250 timeout 시료의 lifecycle은 NOT_TESTED다. timeout 이후 수집이 필요해지면 physical episode ID와 administrative segment ID를 분리하고 기존 loader/collate/stream의 truncated reset 정책을 별도 승인·검증해야 한다. 이번에는 수정하지 않았다.

공식91의73/9/9 ID는 재사용하지 않는다. 새 dataset UUID+reset seed/geometry 초기상태 hash/perturbation family로 reset group을 정의하고 episode/group 단위 별도 split manifest를 만들 것을 제안한다. 동일 초기상태 계열의 train/test 중복과 frame random split을 피하고 최종 비율은 Pilot만으로 정하지 않는다. Train split 확정 후에만 normalizer fit/freeze, validation/test로 통계를 갱신하지 않는다.

Full 수집 전 필요한 것은 opt-in Collector의 interruption/exception partial-finalization, stable controller lifecycle, 독립 ledger/hash/atomic manifest, 전체 Episode causal mask/graph 및 missing 정책, phase별 coverage/shape 난이도, frame/action/EE calibration gate, group split/normalizer/sequence loader와 IO/memory 검증이다. Graph 알고리즘이나 Expert 제어 개선으로 확대하지 않는다.

## 10. 실측 비용과 규모 추정

[실측 JSON](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/data_pipeline_costs.json): 수집 Episode 합 121.296s, NPZ 35,866,279 bytes, 독립 H5 26,009,353 bytes, raw array 199,635,811 bytes. 수집 시간은 reset/Expert/twice-camera-read/H5/NPZ/검증을 포함하며 env startup은 제외한다. 중복 H5는 진단 비용이며 full dataset 저장 기본값으로 처방하지 않는다.

Vision prefix wall 39.594996083993465s(loading 2.255817625031341s), packet 내 mask 합 36.834s, Shape-First/F5/Raw EDE 합 0.242s, serialized packet 635,559 bytes. Mask 타이밍은 초기 검출과 steady frame이 섞인다. SAM2 JPEG 준비/압축/IO·모델 load overhead가 있어 합산 core 시간과 wall은 다르다. CPU high-water RSS 3396583424 bytes는 process imports/model load를 포함하며 stage peak가 아니다. GPU 비용은 미실행.

후속 예산은 `예정 Action수/관측수 × 이번 per-row 저장량/처리량`으로 조건부 추정할 수 있으나 prefix의 작은 N과 SAM2 horizon cache·가림·후기 시점을 전체 Episode 비용으로 실측처럼 외삽하면 안 된다. 예: RGB 두 camera만 98,304 bytes/observation(128×128×3×2); raw depth두개65,536 bytes/observation; 기록된 float depth/calibration/robot/audit와 압축비를 추가해야 한다. Sequence loading은 ragged N/edge 수에 좌우되며 위 Forward4frame 외 대규모 throughput/CPU-GPU memory는 NOT_TESTED다. 장기수집·학습을 실행하지 않았다.

## 11. 검증·미실행·보존

새 lifecycle/abort rejection tests, raw readback의 corruption negative tests와 기존 integration regression을 실행했다. 로그와 [verification](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/verification.json)에서 실제 결과를 확인한다. 공통 Logger/Publisher 소스를 변경하지 않았고 ExperimentLogger 및 register_existing_output의 로컬 manifest 등록만 재사용했다. W&B disabled. CUDA/MPS/다른OS, 과거60개 재실행/hole-shift, Full Dataset, 장기 BC/HPO/Closed-loop Policy rollout, external upload/commit/production default 변경은 NOT_EXECUTED다. 현재 CPU physics와 sapien_cpu host readback만 실행했다(렌더러 Vulkan/Metal device는 Apple GPU일 수 있음).

최종 **31 tests PASS**(새8개, 기존23개). 첫 test 실행은 TestCase의 `run` 이름을 Path attribute로 가린 테스트 코드 오류로 중단됐고 [원본 실패 로그](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/test_initial_failure.log)를 보존했다. `run_dir`로 고친 뒤 재실행했다. 이는 Recorder/Expert 실패와 별개다. 실제 Causal Mask 실행은 설치 SAM2 `_C` optional extension 부재로 hole filling을 건너뛴 warning이 있으며 로그에 보존했다. FakeTracker suffix-invariance 단위 테스트와 실제 DINO/SAM2 실행 결과를 구분한다.

최초 실패와 수정 후 실행 로그, 재현명령, Source snapshot/hash·implementation diff, PNG/JSON/NPZ/H5를 보존했다. 정적 HTML은 CDN 없이 생성하고 링크·JSON·NPZ·PNG를 실제 readback한다. 기존 source/artifact 보호 manifest는 작업 시작SHA와 비교하며 Task03–06A를 덮어쓰지 않는다.

PNG는 직접 열어 검토·수정했다. CUA에서 Browser 연결은 없었고 native Chrome 접근도 `Computer Use permissions are not granted`로 거부되어 실제 Browser layout 검증은 NOT_EXECUTED다. 권한 우회나 추가 설정 변경을 하지 않았다.

## 12. 질문별 답

**Q1 — CONFIRMED.** 현재 recurrent/expert.py::ScriptedExpert는 origin/recurrent와 동일하다. 기존 진입점 recurrent.collect / recurrent.evaluate의 ExpertPolicy, 신규 opt-in collect_recurrent_expert가 원본을 호출한다.

**Q2 — CONFIRMED.** 원본 native 7D를 그대로 BC Target으로 보존한다. pd_ee_delta_pose Controller와 Task06 regression7 의미가 같다. Expert axis-angle 계산과 Controller XYZ Euler 적용은 다르지만 새 변환 없이 동일 Controller 명령을 모사하는 계약이다. 명목 Box 범위 초과 회전은 별도 계수한다.

**Q3 — CONFIRMED.** 수정 후 normal Pilot 5/5 성공, first-success steps [139, 87, 153, 146, 144]. 100-step 이내 1/5. 과거 60/60 재현이 아니다.

**Q4 — CONFIRMED.** 669 actions / 674 observations. 유효 Target 669. Exact step/controller input, pre-action live state, same-time returned/fresh RGB-D, H5↔NPZ bytes와 episode/timestep identity를 검사했다.

**Q5 — PARTIALLY_CONFIRMED.** 성공 Episode의 prefix 16 Packet; 두 물체 nonempty 1. 실제 Sequence Forward PASS_REAL_SEQUENCE_FORWARD. Box가15/16 Frame에서 사라져 학습용 Vision Coverage는 BLOCKED다. ReplayDataset는 공식 H5/split 전용이며 신규 Reader가 필요하다.

**Q6 — PARTIALLY_CONFIRMED.** 새 Observation과 Action을 같은 live 실행에서 만들므로 과거 Converted Trajectory의 state 재현을 필요로 하지 않는다. 과거 Replay Drift 원인 해결은 아니다. Mask/Depth 품질, GT expert 의존성, seed coverage와 새 dataset 계약이 남는다.

**Q7 — NOT_TESTED.** 대규모 수집 전 split/reset group·action raw target·first success/tail·timeout boundary를 확정하고, 전체 Episode causal Vision coverage 및 graph 품질, loader/normalizer/hidden reset과 비용을 검증해야 한다. Full Dataset/BC 학습은 별도 승인 대상이다.

## 13. 재현

새 collection은 새 UUID를 만들고 기존 결과를 덮어쓰지 않는다. [명령](../../experiments/9fec4f02fc9d4172a48e97d5eff90abd/reproduction_commands.txt)을 사용한다. 사전 보호 manifest는 이번 세션 전용이므로 다른 작업환경에서는 먼저 그 환경의 시작SHA를 생성한다. 보고서/reader만 재실행하는 것은 Simulator rollout을 추가하지 않는다. **이번 Pilot 종료 후 중단하며 Full Dataset 생성과 BC 학습은 사용자 별도 승인 대상이다.**
