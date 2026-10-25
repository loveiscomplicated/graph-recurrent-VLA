# Task 06-A — Expert Replay Drift / TimeLimit 진단

Run UUID: `9a83ae221b9345589e639ddd0a9fe33f`. **조사 완료 / Replay 원인의 단일 확정은 불가 / 전체 Dataset은 계속 BLOCKED_DATA.**

[정적 HTML](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/replay_timeout_diagnosis.html) · [검증](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/verification.json) · [상세 수치](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/drift_analysis.json) · [경계 원자료](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/boundary_and_failure_details.json) · [입력 SHA-256](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/input_manifest.json) · [Source SHA-256](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/source_hashes.json) · [변경 Diff](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/implementation.diff).

보존 검증 예외: 기존 Task06 HTML `experiments/c184b85dadf14db5953f11cdcacb6674/graph_gnn_gru_baseline.html`의 시작 SHA와 현재 SHA가 다르다. 관찰된 수정 시각은 2026-10-25 14:27:26 KST이며 사용자가 “별도 편집함”으로 확인했다. 현재 파일은 덮어쓰거나 복구하지 않았다. 시작 hash와 실제 mismatch를 그대로 보존하고 확인된 해당 SHA만 명시적 사용자 편집 예외로 기록했다. 나머지 보호9,114개 및 직접 진단 입력 SHA는 일치한다. 검증 상태는 PASS_WITH_USER_CONFIRMED_EXTERNAL_CHANGE이며 모든 파일이 불변이었다는 뜻은 아니다.

## 1. 기존 사실과 이번 새 증거

Task06 `c184b85dadf14db5953f11cdcacb6674`의 91 Episode / 13,958 Action / Frozen 73·9·9, 완주 Replay 0개, 검증 Prefix 3개 / 관측473·Target470을 기존 Artifact에서 재확인했다. 기존 State-only 및 threads4 재현 결과와 Snapshot 최대 약3e-7도 확인했다. 기존 `verification.json`의 PASS는 통합·Artifact 검사이며 Full Replay PASS가 아니다. Source H5/JSON 및 원본 H5, frozen manifest는 읽기 전용으로 열었다.

이번에는 첫3 Episode의 전체 저장 길이까지 진단 Replay 8조건×3=24회를 실행했다. 실패 뒤 구간은 **audit-only**이며 BC 입력이나 검증 Dataset으로 저장하지 않았다. 추가로 첫3의 Snapshot 복원/독립 one-step probe, 기존 raw action의 메모리 내 재변환을 실행했다. 91개 신규 수집, Vision 재생성, 학습, 정책 Rollout, Production 수정은 실행하지 않았다.

새 증거는 (a) t=2의 작은 joint 차이와 후기 qvel spike, (b) 100→1000 TimeLimit 변경에도 동일한 물리 결과, (c) 같은 조건 반복 간 모든 기록 배열의 exact 일치, (d) Snapshot의 관측 정합과 다음 Dynamics 정합의 차이, (e) 과거 exact Replay 기록과 현재 Runtime의 불일치다.

## 2. Source / 변환 / Controller 계약

경로: 원본 공식 Motion Planning `trajectory.h5`의 pd_joint_pos action8 → [기존 replay_demos](../../graph/prototype/replay_demos.py) → 설치 ManiSkill `replay_cpu_sim` / `from_pd_joint_pos_to_ee` → converted pd_ee_delta_pose action7. 원본 raw에는 obs가 없고 actor13 / robot31의 env_states가 있다. Converted에는 `actions[T,7]`, `obs[T+1,43]`, `env_states[T+1]`, `success/terminated/truncated[T]`가 있다. obs43은 qpos9, qvel9, TCP7, peg7, peg_half_size3, hole7, radius1이다.

변환기는 원본 joint action을 먼저 실행해 **원본 controller target qpos의 FK**로 목표 EE를 구하고, target 환경의 현재 EE에 대한 delta를 계산한다. 위치 clip / rotation norm clip이 있으면 최대4회의 target step을 실행할 수 있다. 따라서 raw와 converted의 같은 index가 같은 물리시각이라는 계약은 없다. 실제 mapping은 converted0→raw0(seed2,T178→178), 1→raw3(seed6,T117→123), 2→raw4(seed9,T189→189)다. 전체91 mapping/seed/길이는 [inventory](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/source_inventory.json)에 보존했다. 기존 frozen repair mapping을 재사용했으며 seed만으로 다른 raw episode를 새로 선택하지 않았다.

**시간 계약:** o[t]가 a[t]의 입력 상태이고, `step(a[t])` 뒤 o[t+1] 및 flags[t]다. T action / T+1 observation을 유지한다. state0은 raw initial repair, state[t>0]는 converted의 같은 index다. 설치 `replay_cpu_sim`은 `ori_env_states=ori_env_states[1:]` 후 첫 buffer state를 그 배열0으로 덮어 **raw s1을 converted env_states[0]에 저장**한다. observation0은 별도로 현재 raw s0의 get_obs다. [prepare_dataset](../../graph/prototype/prepare_dataset.py)의 repair는 이 저장 오류를 보완하며 Snapshot 정합에는 충분하지만 현재 순차 Replay 정합까지 보장하지 않는다.

저장된 reference는 qpos/qvel/TCP/object 관측과 actor/articulation pose·velocity, action, flags다. Finger geometry는 별도 측정으로 저장돼 있지 않아 **같은 index의 저장 robot snapshot을 현재 robot/URDF로 복원한 FK 값**과 비교했다. 이 재구성은 robot18 및 TCP 정합(최대 약3e-7)으로 검증했고 recorded fingertip GT라고 부르지 않는다. Link pose/velocity15×13은 Live Replay에 기록했지만 source reference link velocity가 없어 그 오차는 산출하지 않았다. Source controller target/IK 결과·contact·solver cache도 저장돼 있지 않다. 추정 reference나 qvel 수치미분을 만들지 않았다.

실제 CombinedController 입력은 action7과 모든 step에서 exact 같다. Translation ±1→±0.1m, XYZ rotation 성분은 norm≤1 clip 후 **rot_lower=-0.1**을 곱하고 Euler XYZ delta quaternion을 만든다. Root translation / root-aligned body rotation, use_delta=True / use_target=False / interpolate=False다. Gripper 한 값은 -1→-0.01m / +1→0.04m, 두 finger mimic이며 action을 재정규화하지 않는다. Target pose/qpos와 scaling 후 action은 NPZ에 남겼지만 **원본 source target과 같은지 직접 증명할 저장값은 없다.** 현재 controller config 일치와 당시 binary·IK 출력의 동일성을 구분한다.

Control20Hz / PhysX100Hz / control step당5 substep. CPU PhysX, TGS/PCM True, solver position15/velocity1, enhanced_determinism False, cpu_workers0, float32 robot state다. 자세한 config·joint/link 순서·wrapper chain은 [runtime](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/runtime.json), 설치 구현은 [보존 Source](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/source_snapshot/installed/mani_skill/trajectory/replay_trajectory.py)에 있다. raw metadata의 commit은 `9d5e0e01c3b13796dd1c9fc83213b315ce8a1946`, converted commit_info는 null이다. 당시 conversion 전체 config/의존 binary hash는 없다.

## 3. Drift 수치와 시간축

모든 timestep은 **observation index t**다. t=175 실패는 a[174] 실행 뒤다. 통계의 qvel mean/p95는 매시점9성분 중 최대 절대오차의 시간분포이며, 단위가 다른 arm/finger의 scalar gate는 기존 기준의 재현용이다. 상세 JSON은 전체 원소 통계와 성분별 통계도 제공한다. qpos9 최대는 기존 rad/m 혼합 요약이며 단위별 곡선·성분을 함께 확인한다.

| Episode | T | 첫 qpos/qvel nonzero | 첫 >1e-3 | 최대 qvel Joint | qvel max | time-max mean | time-max p95 | qpos9 max | TCP xyz max(m) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 178 | 2 | 175 | panda_joint6 | 0.0010336712 | 5.55046643e-05 | 0.000226588175 | 1.52587891e-05 | 5.39422035e-06 |
| 1 | 123 | 2 | 117 | panda_joint7 | 0.00113541633 | 7.19528565e-05 | 0.00029989332 | 1.31130219e-05 | 5.453825e-06 |
| 2 | 189 | 2 | 181 | panda_joint5 | 0.00114145875 | 5.21242836e-05 | 0.000158097316 | 2.62260437e-05 | 7.05942512e-06 |

최초 전체 차이는 t=0 TCP의 약0.6~1.3e-7 성분 차이다. qpos/qvel은 t=0,1에서 exact하고 t=2에서 함께 달라진다. t=2의 qvel 최대는 Episode0 `4.97725841e-6`(right finger, m/s), 1 `2.25286931e-6`(joint2, rad/s), 2 `6.41860515e-6`(right finger, m/s)다. 전체 component9를 JSON에 보존했다. 작은 차이의 출발과 **gate를 넘는 Arm Joint**는 다르다.

qvel9 최대치(전체 source 길이, scalar끼리 단위 혼합 해석 금지):

| Joint | 단위 | Ep0 | Ep1 | Ep2 |
|---|---|---:|---:|---:|
| panda_joint1 | rad/s | 0.000264062546 | 0.00011464674 | 0.000104464591 |
| panda_joint2 | rad/s | 6.97821379e-05 | 0.000247351825 | 4.68147919e-05 |
| panda_joint3 | rad/s | 0.00030305516 | 0.000182387885 | 8.03302974e-05 |
| panda_joint4 | rad/s | 0.000140903518 | 0.000197933987 | 0.000225111842 |
| panda_joint5 | rad/s | 0.000679694116 | 0.000643553212 | 0.00114145875 |
| panda_joint6 | rad/s | 0.0010336712 | 0.00104061142 | 0.000320866704 |
| panda_joint7 | rad/s | 0.00093267858 | 0.00113541633 | 0.000827351585 |
| panda_finger_joint1 | m/s | 0.000311568379 | 0.000293180346 | 0.000443864614 |
| panda_finger_joint2 | m/s | 0.000301159918 | 0.000325670466 | 0.000436265022 |

위치 오차는 초기에 작고 시간이 지나며 커지지만 velocity 오차는 단조 누적하지 않고 oscillation/spike가 있다. 실패 후 마지막 qvel max는 Ep0 `0.0004636124`, Ep1 `0.0010406114`, Ep2 `0.0008273516`다. 따라서 실패 뒤 계속 발산한다고도, 작은 오차가 무해하다고도 단정할 수 없다. TCP orientation 최대는 각각 `1.69166708e-05`, `1.03293749e-05`, `1.72649615e-05` rad(부호 불변 상대 quaternion angle)이다.

첫 source terminated/success는 post-action observation 기준 Ep0 t166(a165), Ep1 t113(a112), Ep2 t181(a180)이다. Drift gate 초과는 각각 그 이후 t175, 이후 t117, 같은 시점 t181이다. 따라서 first-terminated 정책을 논의할 때 성공 뒤 tail과 그 이전 구간을 구분해야 한다. 다만 Ep2는 첫 성공 전이의 결과 관측 자체가 gate를 넘으므로 종료 지점만 줄이는 것으로 pilot 정합이 해결되었다고 볼 수 없다. 이 관찰로 Source Gate나 Episode 길이를 바꾸지 않았다.

| Episode | 재구성 left tip xyz max(m) | right tip(m) | peg xyz(m) | peg linear velocity(m/s) | peg angular velocity(rad/s) |
|---|---:|---:|---:|---:|---:|
| 0 | 5.69224358e-06 | 5.08129597e-06 | 6.15417957e-06 | 0.000157910981 | 0.00432166457 |
| 1 | 5.7220459e-06 | 5.27501106e-06 | 4.67896461e-06 | 0.000344572589 | 0.013927713 |
| 2 | 6.98119402e-06 | 7.01192766e-06 | 7.36862421e-06 | 0.000507727498 | 0.00929670036 |

Peg velocity 오차는 저장 env_states에서 직접 비교했다. qvel1e-3 통과 여부만으로 물리 전체의 정확도를 판단할 수 없으며, 기존 state43 gate에는 actor velocity가 없다. Box state 오차는 세 Episode에서0이다. FK reference는 현재 Snapshot 복원 결과라는 한계를 유지한다.

![Ep0 오차](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/plots/ep000_errors.png)
![Ep1 오차](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/plots/ep001_errors.png)
![Ep2 오차](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/plots/ep002_errors.png)

## 4. t=100 및 TimeLimit의 의미

설치 `tasks/tabletop/peg_insertion_side.py:50`의 `@register_env(...,max_episode_steps=100)`가 출처다. `utils/registration.py:127–168`의 ManiSkill TimeLimitWrapper가 Gym TimeLimit을 제거하고 batched Torch truncation을 계산한다. 실제 chain은 **ManiSkill TimeLimitWrapper → Gym OrderEnforcing → task**이며 Gym TimeLimit/AutoReset은 없다. `BaseEnv.step`은 먼저 action/physics5substep을 실행하고 `_elapsed_steps+=1`, task success/fail로 terminated를 만들며 내부 truncated는 False다. 그 반환 뒤 wrapper가 `elapsed_steps>=100`을 적용한다. Timeout에 controller reset, physics 정지, state 변경은 없다.

정확한 발동은 **a[99] (100번째 action) → o[100]에서 truncated=True**다. a[98]→o[99]는 False, a[100]→o[101]도 True이고 이후 계속 True다. 첫3에서 이 세 지점의 terminated는 False, elapsed_steps는99/100/101이다. Source 및 Replay의 flags는 전체 길이 exact, counter는1..T로 연속이다. 초기화는 episode당 한 번, step 내부 reset 호출은 없다. Controller input/target과 pose는 100 이후도 같은 task에서 계속 적용됐다.

qvel max의 t=99/100/101 값은 Ep0 `2.46763229e-5 / 2.07489356e-5 / 2.96030194e-5`, Ep1 `1.27665699e-5 / 3.78992409e-5 / 3.86117026e-5`, Ep2 `1.98855996e-5 / 1.79708004e-5 / 1.25262886e-5`다. t90..110 및 실패±5 모든 source/replay 관측43·action·flags·counter를 별도 JSON/NPZ에 보존했다. 이 구간에 gate 수준의 불연속은 없다. **TimeLimit만1000으로 바꾼 독립 실험은 세 Episode의 state43·Controller target qpos 전체와 exact 일치**했고 truncation만 달라졌다. TimeLimit을 drift 원인으로 보는 가설은 이 실행 범위에서 배제한다.

원본 raw0/3/4도 a[99]부터 truncated=True이며 그 이후 state/action이 저장돼 있다. 공식 planner의 `follow_path/open_gripper/close_gripper`는 반환 flags에 break하지 않는다. Collector `panda/run.py`는 planner 최종 success로 저장 여부를 판단하며 converter의 `discard_timeout` 기본값은 False다. Converter도 action 전체를 step한다. RecordEpisode는 flags/done을 기록하지만 자동 reset하지 않는다. 즉 source는 **100 이상 계속 실행하는 수집 경로**다. 원본 당시 commit의 모든 바이너리·실행 로그는 없지만 저장 데이터가 현재 설치 코드의 이 동작과 부합한다.

질문별 판단: (1) 100 이상 수집됐음은 데이터로 확인. (2) 100은 관리상 제한이며 물리 종료 아님은 코드+실험으로 확인. (3) 첫3의 timeout 이후도 물리적으로 연속이라는 근거는 단순 flag 일치가 아니라 counter·무reset 제어 흐름·100/1000 동일 상태·source 연속 저장이다. 전체91은 저장자료/collector 경로 근거이며 live91 검증은 아님. (4) 최종 success=True로 유지된 공식변환91개이며 `2/91`개만100 이내에 성공 flag가 존재한다. 이는 tail의 expert provenance를 지지하지만 **현재 Runtime Replay gate나 모든 post-success sample의 BC 적격성을 승인하지 않는다.** (5) shared factory 기본 closed-loop는100이고 done에서 멈추는 evaluator는 이를 적용한다. 다만 기존 Graph evaluator는 명시 horizon으로 TimeLimitWrapper를 재구성하며 Graph pilot/Recurrent 기본250도 존재한다. Task06에는 새 closed-loop evaluator가 없고 이번에 실행하지 않았다. (6) 100-step 평가로는 대부분 expert 성공 길이를 감당하지 못하므로 시연 완주/학습 coverage와 평가 성공률의 의미가 달라진다. 학습 tail을 유지해도 rollout horizon은 자동 연장되지 않는다.

## 5. 원인 후보와 확정 수준

| 후보 | 상태 | 근거 |
|---|---|---|
| RGB-D readback / Torch 1 vs 4 threads | REJECTED (tested scope) | State43 and controller targets identical across the 3 complete diagnostic trajectories. |
| 100-step TimeLimit causes drift | REJECTED (tested scope) | 100 vs 1000 limit: full state43 and controller-target arrays identical; counters keep increasing. |
| Wrong stored initial row without repair | SUPPORTED defect; not sufficient cause | Official replay slices away raw s0 then writes raw s1 into initial state buffer. Repaired initial qpos/qvel exact, but drift remains. |
| Missing controller reset after initial restore | REJECTED for tested controller | Explicit controller.reset changes neither trajectory nor targets; use_target=False/interpolate=False overwrites targets each step. |
| Normalized action values / gross timestep shift | REJECTED in current replay | Captured CombinedController input equals stored action exactly; T/T+1 and flags/counters match. Original target/IK outputs were not recorded. |
| FK/IK / float32 feedback and integration amplification | SUPPORTED mechanism; INCONCLUSIVE root | TCP component differences already at t0; joint state exact through t1, diverges at t2. Small position errors grow with oscillatory velocity spikes; source targets unavailable. |
| Incomplete PhysX state / contact or solver cache | SUPPORTED incompleteness; INCONCLUSIVE causal share | H5 stores actor/articulation kinematics, not solver caches; snapshot-started one-step errors are much larger during contact. |
| Original/runtime binary or backend mismatch | INCONCLUSIVE | Current version strings match prior extraction metadata; original raw commit differs from installed package provenance. Original conversion commit/config/binaries were not archived. |
| Other CPU/OS, CUDA, enhanced-determinism settings | NOT_TESTED | No dependencies installed, no backend changes, no GPU simulation. |

Snapshot 복원 state43 max는 Ep0 `3.72936483e-07`, Ep1 `2.99885869e-07`, Ep2 `3.57627869e-07`로 기존 Gate2e-5 안이다. qpos/qvel은 전 Snapshot에서 exact다. 그러나 snapshot-started 독립 one-step qvel max는 `0.0437275469`, `0.0623038039`, `0.0521915331`다. 이 Probe는 매 step 상태만 덮고 controller/PhysX history는 복원하지 않으므로 source dynamics가 완전 복원됐다는 검사가 아니다. one_step NPZ row0은 o[1]이고, 요약 first_nonzero 등에 **+1하여 source observation index로 해석**한다. 이 결과를 서로 연결해 Episode로 만들거나 순차 Replay PASS로 사용하지 않았다.

추가 메모리 내 재변환에서 generated action max 차이는 Ep0 `0.0481718704`, Ep1 `0.00832123868`, Ep2 `0.0020313384`이며 모두 action0부터 nonzero다. 길이는178/123/189로 같고 최종 success는True지만 저장 converted action/state를 exact 재현하지 못했다. raw pd_joint_pos replay도 actor/robot state 차이가 있었다. 따라서 현재 코드/metadata만으로 **당시 변환 Runtime까지 재현했다는 주장은 불가**하다. 기존 file은 덮지 않았다.

문서/Artifact 불일치: 기존 `outputs/graph/dataset.json`의 `replay_checks`는 같은 첫3 max_obs_error=0.0 및 success=True이나 지금 재현과 다르다. 버전 문자열은 같지만 당시 source/binary snapshot이 없어 왜0이었는지 확정할 수 없다. Task06 결과문서는 원인 미확정이라고 올바르게 제한했고, 과거 Graph README의 validated replay 설명은 해당 과거 기록으로 한정해야 한다. 현재 `PROJECT_CONTEXT`의 빈 팀 디렉터리 설명도 실제 구현과 다르며 공통 환경 범위 원칙만 적용했다. 이전 Task06 Source hash와 현재 값은 input_manifest에서 대조했다.

**원인 확정 여부:** 100-step timeout의 출처/비중단 의미는 확인했다. Drift는 같은 action에도 작은 FK/TCP 차이와 joint integration 차이가 초반부터 생기고 후기 접촉 구간에서 velocity spike로 커지는 양상까지 확인했다. FK/IK feedback, float32 연산, pose round-trip, 물리 solver/contact history 및 당시 binary 차이의 기여를 분리하지 못했으므로 단일 근본원인은 **INCONCLUSIVE**다. 접촉 count는 replay 진단값일 뿐 source contact reference가 없어 원인 확정 label로 쓰지 않는다.

## 6. Sequence / Hidden Reset 정책 후보 — 자동 채택 없음

저장91개 길이 min105 / p25 131 / median149 / p75 175 / max219 / mean153.385다. 100-only cap은 91/91개에서 prefix를 남기지만 완전 episode는0/91이며 **9,100/13,958 전이 유지,4,858(34.8044%) 제외**다. Train7,300/11,274, validation900/1,325, test900/1,359. 기존 `dataset.pt`의 phase label과 모든 episode/timestep/action을 H5에 exact 대조한 뒤 아래 손실을 집계했다. **기존 geometry heuristic label이며 실제 Align/Insert GT가 아니다.** 새 phase 판정이나 접촉 reference를 만들지 않았다. [phase 집계](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/stored_phase_coverage.json).

| 기존 heuristic phase | 전체 | 100 cap 유지 | 제외 |
|---|---:|---:|---:|
| approach | 3082 | 3082 | 0 |
| grasp | 4530 | 4367 | 163 |
| transport | 514 | 359 | 155 |
| alignment | 4082 | 1255 | 2827 |
| insertion | 1750 | 37 | 1713 |

| 후보 | 물리/정합 및 Hidden 의미 | 저장자료상 potential 전이 | Rollout/수정 범위 |
|---|---|---|---|
| A: 100에서 종료 | o[0..100]/a[0..99] T/T+1 유지. 물리 reset은 없지만 평가100의 인위적 boundary로 hidden reset을 정의 | 9,100; 실제 terminated도 선행 종료하면 **9,097**. Ep33 a96 성공으로3개 추가 제외, Ep40 a99는100과동일 | 평가100과 종료 규약 일치. cap/filter를 별도 명시하고 loader/stream 기본 reset을 유지. 삽입 성공 tail 대부분 배제 |
| B: administrative timeout 분리 | Timeout에는 상태 연속성이 있으므로 hidden carry가 물리 의미와 일치할 수 있음. terminated/실제 reset/episode 교체는 boundary 유지 | 첫 terminated까지만이면 **13,060**, 원본 대비898 post-success 전이 제외. A와 같은 terminated 규칙 대비 **3,963** 추가 유지 | flags 원본 보존+별도 semantic boundary, loader/collate/stream 세곳 검토 및 tests 필요. 평가100과 horizon 차이는 남음 |
| C: sequence와 평가 horizon 별도 계약 | Training에는 B의 연속성, 평가에는 선택한100 또는 별도 budget과 reset 명시. 관측-action join은 불변 | terminated 보존 시 B와같이13,060; 원본13,958을 모두 쓰려면 success 이후의 별도 계약도 필요하며 이번 제안은 아님 | 데이터 boundary metadata와 학습/추론 lifecycle, 평가 horizon/metric을 각각 명시·검증해야 함. 타 환경에 timeout 무시를 일반화하지 않음 |

현재 Loader `sequence`/`collate_sequences`는 result[:-1]의 terminated/truncated 중 하나라도 True면 거부한다. source truncated가 a99 이후 **모두 True**이므로 단지 t100을 가로지르는 chunk뿐 아니라 이후 length>1 chunk도 거부한다. 연속길이4 window의 저장자료 potential은 현재 규칙 `8824`, administrative timeout만 분리하고 first terminated를 지키면 `12787`다. 실제 생성/로딩한 새 학습 window 수가 아니다. `EpisodeStream`도 previous_truncated마다 reset하므로 timeout 뒤 매 step hidden을 지울 수 있다. 현재 Smoke t0..3에는 영향이 없다.

Q4 답: 평가100을 Episode 계약으로 선택하는 A에서는 끊기가 일관되나, **물리 reset에 필수라는 설명은 틀리다**. 같은 물리 Episode를 학습하려면 B/C의 reset 구분이 더 잘 맞지만 별도 사용자 정책 결정과 수정 검증이 필요하다. 이번에는 어느 규칙도 변경하지 않았다.

## 7. Gate 판단과 Full Replay 전 필요한 조치

기존 snapshot2e-5 / sequential1e-3은 `prepare_dataset.py`가 observation 정합 및 독립 action replay sanity check를 위해 사용한 hard-coded 값이다. 단위별 허용 budget이나 task 성능 기반 calibration 문서/측정 근거는 찾지 못했다. 원본 controller와 현재 설정은 같은 mode/default이나 source target·IK output 기록이 없어 동일 계산이었다고 증명할 수 없다. qpos/TCP/object도 차이가 있고 actor angular velocity 차이는 별도이므로 **근소한 qvel 초과만을 이유로 tolerance 상향을 권고하지 않는다. 숫자 Gate는 그대로다.**

Full91 재개 전에 필요한 조치는 다음 순서다.

1. 당시 conversion/extraction Runtime의 정확한 package/binary/config/host 근거를 확보하거나, 별도 승인한 작은 controlled capture로 source action·scaled input·target qpos/pose·state0/1/2를 함께 보존하고 same-runtime re-replay를 검증한다. FK/IK 결과 차이와 pose/state round-trip 효과를 구분해야 한다. 접촉/solver cache는 지원 API로 얻을 수 있는 범위만 조사하고 불완전 snapshot을 완전 State로 간주하지 않는다.
2. 기존 Gate로 첫3 complete sequential pilot을 다시 통과해야 한다. 현재 결과는 이 조건을 충족하지 않는다. Teleport/새 action 생성/수치 clipping으로 pass를 만들지 않는다. 순수 수치차이라고 분리된 뒤에만 joint 단위별 예산, actor pose/velocity·contact/task 영향, 여러 source episode 및 backend별 검증을 갖춘 tolerance 제안을 별도 검토한다.
3. A/B/C, terminated 이후 tail, 평가 horizon과 metric의 계약을 사용자 검토로 결정한다. 이번 증거는 **timeout을 물리 reset과 분리해 검토할 근거**로 충분하지만 Loader/GRU 수정의 correctness나 긴 horizon 성공률을 검증한 것은 아니다.
4. 승인된 수정 후 loader/stream boundary 회귀, complete pilot, 91 coverage/원본 split·action identity를 검증하고 그때 causal Vision Dataset 구축을 재개한다. 현재 Vision473/Target470을 새13,060/13,958 dataset처럼 보고하지 않는다.

Q1: 전체 첫차이는 TCP t0 약1e-7, qpos/qvel은 t2부터이며 후기 Arm Joint6/7/5 spike가 각각 t175/117/181에서1e-3을 넘는다. Q2: 단일 근본원인은 미확정, Runtime/kinematics/physics-state 후보가 남는다. Q3:100은 관리상 truncation이며 timeout 이후 physical continuity는 지지되지만 BC 유효성은 replay gate·terminated 경계와 별도로 판단해야 한다. Q4: 정책별 근거/손실은 §6, 자동 reset 변경 없음. **Q5: 현 결과로 Full91 재개·tolerance 변경을 승인할 수준은 아니며 위 추가 evidence/pilot 및 사용자 정책 결정이 필요하다.**

## 8. 실행·미실행·검증과 결정 요청

실행 host는 `macOS-27.0.1-arm64-arm-64bit` / Apple M4 Pro, Python3.11.13, ManiSkill3.0.1 / SAPIEN3.0.3 / Torch2.14.1 / Gymnasium1.3.0 / NumPy2.4.6 / pin4.1.0이다. **실행은 physx_cpu/sapien_cpu뿐**, MPS 사용가능 여부와 실제 실행을 구분하며 MPS/CUDA/다른OS/다른 PhysX 설정은 NOT_EXECUTED다. seed0, threads1(threads4 조건만4), Torch deterministic algorithms=False. Source episode seed2/6/9를 각각 reset에 사용했다. Renderer는 Vulkan validation layer 부재 경고가 있었지만 CPU replay를 완료했다. Source 당시 binary/remote host는 검증하지 못했다.

재현(새 UUID 생성, 입력 덮어쓰기 없음):

```bash
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MPLCONFIGDIR=/tmp/task06a-mpl .venv/bin/python -m graph.integration_baseline.diagnose_replay_timeout \
  --preflight experiments/9a83ae221b9345589e639ddd0a9fe33f/before_snapshot
MPLCONFIGDIR=/tmp/task06a-mpl .venv/bin/python -m graph.integration_baseline.report_replay_timeout \
  --run experiments/NEW_RUN_ID
```

`before_snapshot`은 이번 세션 시작의 보호 manifest이며 새 작업환경에서는 그 환경의 기존 source/artifact SHA를 먼저 생성한다. [execution.log](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/execution.log), [console.log](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/console.log), [실패 첫실행](../../experiments/d272fc182468487a8bb3ee1ac89ee546/execution_failure.json), [첫실행 Source](../../experiments/d272fc182468487a8bb3ee1ac89ee546/source_snapshot/diagnose_replay_timeout.py)를 보존했다. 첫 실행은 snapshot 상관계수가 undefined인 exact0 오차에서 NaN JSON serialization로 중단됐다. 진단 통계만 null 처리로 고쳤고 수치/Gate는 바꾸지 않았으며 새 UUID로 전체 진단을 완료했다.

새 진단 계산3개 및 기존 Loader regression과 공통 Logger/Artifact tests를 실행했다. 한 프로세스 통합49 tests에서는 `imageio.plugins.__getattr__`와 importlib mock/assertWarns의 충돌1 ERROR가 나왔다. 원본 tests를 고치지 않고 **공통 인프라 suite와 simulator-import suite를 별도 process**로 실행해 결과를 보존했다. [combined 실패](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/tests_combined.log) · [인프라](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/tests_infrastructure.log) · [진단/Loader](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/tests_diagnostic.log). 검증 성공은 Artifacts/readback/tests 범위이며 drift gate PASS를 뜻하지 않는다. 최종 JSON에는 기존 보호9,115개 파일의 SHA, 상대링크, NPZ/PNG decode, source snapshot, Git diff 확인 범위를 명시한다.

분리 실행은 공통42 tests와 진단/Loader7 tests가 PASS했다. 정적 HTML은 CUA 브라우저가 이 세션에서 제공되지 않아 실제 browser layout 검증은 **NOT_EXECUTED**다. PNG를 직접 열어 확인하고 HTML/Markdown의 상대경로와 이미지 decode를 별도로 검증했다.

기존 runnable example `examples/check_env.py --control-mode pd_ee_delta_pose`도 실행해 CPU backend, state43/action7, reset 및 한 step을 확인했다. [example log](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/example_check_env.log). 실행 설정·재현 명령은 [commands](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/reproduction_commands.txt), [조건 기록](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/executed_conditions.json)에 보존했다.

공통 ExperimentLogger 및 experiment_artifacts 사후 로컬 등록을 재사용했다. W&B disabled, 외부 Upload/Publishing/설치/Commit 없음. 이 작업에서 작성한 파일은 새 진단/보고 script2개, 독립 test1개, 이 문서 및 새 실험 Artifact다. 기존 dirty/untracked source의 SHA는 일치한다. 기존 Task06 HTML의 변경은 사용자 별도 편집으로 확인됐고 시작 hash를 갱신하지 않은 명시적 보존 예외다. 사용자 확인 전 보존 검사의 FAIL과 로그도 남겼다. [변경 관찰·사용자 확인](../../experiments/9a83ae221b9345589e639ddd0a9fe33f/prior_artifact_change_observation.json).

사용자 결정이 필요한 것은 (a) A/B/C 및 별도 평가 horizon, (b) first terminated 뒤898개 tail 처리, (c) 정확한 Runtime 확보 또는 제어된 추가 capture 실험 승인이다. **실제 Replay/Loader/GRU 수정 및 Full Dataset 재구축은 사용자 검토 후 별도 승인 대상이다.**
