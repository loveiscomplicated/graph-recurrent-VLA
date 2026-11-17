# Task 08-C — Seed 30003 A Policy Gripper 개입

ExperimentLogger UUID: `dc1b894f67af4e16a378dc02c3c38a97`. Vast.ai 57640710의 RTX 3090에서 A 정책을 추론하고 CPU PhysX에서 세 조건을 각각 새 Reset으로 250 step 실행했다. Control 20 Hz, physics 100 Hz, `pd_ee_delta_pose`와 원래 Stop-the-clock 입력 경로를 유지했다. 학습, 가중치 변경, Expert 제어, 외부 게시, Git commit은 없었다.

## 고정 대상과 시간 규약

- Task 08-B Seed 30003 단일 Episode A Selected 20,000 update checkpoint: `/workspace/task08b/experiments/cac57daccc2044a8a8e9de4a5e8f7adc/checkpoints/step020000-epoch4000.pt`; SHA256 `5a94542e6d1f973d052fce9598e92a3be5b8c558974cc4e761f17e087a013f4d`.
- Robot76 normalizer identity: `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`. Action codec은 Task 08-B A의 Native Action7이다. 체크포인트의 Episode·Method·Plan·Codec·Normalizer identity를 로딩 시 검증했다.
- `Observation[t] → Policy Action[t] → env.step(Action[t]) → Observation[t+1]`. 개입은 Observation 79에서 추론한 Action 79의 gripper 성분을 Controller 전달 직전에 교체한다. 모든 Arm6 성분은 그 시점의 A 예측을 그대로 적용했다. Hidden은 각 Episode t=0에만 초기화했다.
- A0는 전 step 원본 Action7, A1은 Action 79에만 gripper `-1`, A2는 Action 79부터 종료까지 적용 gripper `-1`이다. A2는 **gripper assistance가 들어간 mixed control**이며 자율 성공으로 해석하지 않는다.

## 동일 Prefix와 원본 재현

실행 전 고정한 허용오차는 FP32·물리 수치 최대 절댓값 `1e-6`, RGB 2 LSB, raw depth 1 mm, 정수 mask/graph index 및 reset hash 정확 일치였다. [prefix audit](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/prefix_audit.json)은 Reset 및 Observation 0–79의 RGB-D·calibration·Robot76·Mask·Graph, Action 0–78의 적용값, Action 79 직전 qpos/qvel·TCP·손가락·Peg/Box pose/velocity, Model Action7과 GRU Hidden을 확인했다. **A0/A1/A2 모두 PASS**이고 t=79 관측 입력과 물리 상태는 감사 필드에서 정확히 일치했다.

새 A0와 원본 Task 08-B A의 prefix 감사도 PASS다. 추가로 250개 Action7과 250개 GRU Hidden 배열 전체가 원본 trace와 byte 단위로 일치했다(최대 절댓값 차이 0; [전체 재현 감사](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/baseline_reproduction.json)). A0는 원본의 Action 79 `+0.151109`, Action 80 직후 파지 상실, 최대 lift 84.63 mm, 250-step 삽입 실패를 재현했다. 원본 08-B trace의 `t`는 Action index이며 해당 프레임의 파지·접촉 진단은 **Action 적용 후** 상태다. 따라서 Action 79 직전에는 파지가 성립하고, Action 79 직후에는 아직 `Panda.is_grasping=True`, Action 80 직후에는 False다.

## 실제 결과

| 조건 | Action 79 적용 gripper / target | Action 79 직후 양측 접촉력 | 최초 파지 상실 | `Panda.is_grasping` 구간 (Action index) | Peg ≥4 cm 구간 | 최대 lift | 삽입 |
| --- | --- | --- | ---: | --- | --- | ---: | --- |
| A0 | `+0.1511` / `+0.0188 m` | 2.12 / 1.99 N | 80 | 51–79 | 65–80 | 84.63 mm | 실패 |
| A1 | `-1` / `-0.0100 m` | 28.72 / 28.85 N | 80 | 51–79, 112–166 | 65–80, 135–168 | 136.79 mm | 실패 |
| A2 | `-1` / `-0.0100 m` | 28.72 / 28.85 N | 177 | 51–176, 190–249 | 65–91 | 84.63 mm | 실패 |

세 조건 모두 Action 79 직전 파지 판정은 참이었다. A0의 접촉력은 Action 78 후 8.15/8.21 N에서 Action 79 후 2.12/1.99 N, Action 80 후 0/0 N으로 감소했다. A0의 두 finger joint는 Action 79 후 약 20.9 mm, Action 80 후 약 22.1 mm로 벌어졌다. Peg lift는 Action 79 후 74.5 mm에서 Action 80 후 50.9 mm, Action 81 후 7.9 mm로 하락했다.

A1은 같은 원본 Model Action 79 `+0.1511`을 기록하면서 적용값만 `-1`로 바꿨다. Action 79 후 force는 28.72/28.85 N, lift는 77.4 mm였다. 그러나 변경된 Observation 80에서 A가 다시 `+0.2405`를 예측했고 그대로 적용하자 접촉력이 0/0.12 N으로 떨어져 파지를 잃었다. A1은 Action 112에서 재파지해 136.79 mm까지 다시 들었다. Action 166의 모델 gripper `+0.163` 후 접촉력은 1.54/1.43 N, Action 167 후에는 0/0 N이 되어 다시 놓쳤다. 한 프레임의 수정만으로는 지속 파지나 삽입에 충분하지 않았다.

A2는 Action 80에서도 모델이 `+0.2405`를 예측했지만 적용값을 `-1`로 유지했다. 양쪽 힘은 약 28.8 N이고 Peg–TCP 상대 위치가 거의 일정한 채 파지가 Action 176까지 유지됐다. 그러나 A의 Arm 출력에 따라 Peg 자체가 낮아졌다. Lift는 Action 79 후 77.4 mm, Action 91 후 40.8 mm, Action 120 후 약 0.1 mm였다. **접촉 유지와 들어 올린 상태의 운반은 다르다.** Action 177에서는 적용 gripper `-1`, target `-0.010 m`인데도 양쪽 Peg 접촉력이 0/0 N이 되고 finger joint가 21.2/24.3 mm로 벌어졌다. Action 190부터의 재파지 판정은 바닥 높이에서 발생했고 삽입은 없었다. Action 177의 정확한 충돌 원인은 이 실험만으로 특정하지 않았다.

개입 후 A의 후속 출력은 실제 상태 변화에 반응했다. A0와 A1/A2의 Action 79까지 Model Action7과 Hidden은 동일했다. Action 80의 A1/A2 GRU Hidden은 A0와 L2 거리 0.4303, Arm6 출력 차이는 L2 0.0239였고, 이후 조건별 차이가 더 커졌다. 이는 다른 실제 관측에서의 출력 차이이며 동일 상태의 Policy 비교로 해석하지 않는다.

## Q1–Q9 판정

| 질문 | 판정 | 답 |
| --- | --- | --- |
| Q1 원본 t=79 개방/상실 재현 | **CONFIRMED** | 전체 Action·Hidden 정확 일치, Action 79 `+0.1511`, Action 80 후 파지 상실. |
| Q2 개입 직전 동일성 | **CONFIRMED** | 세 독립 Reset의 동일 prefix, Observation 79/Hidden/물리 상태 감사 PASS. |
| Q3 한 step Close | **CONFIRMED** | Action 79 후 force·파지는 유지되지만 Action 80의 재개방 후 상실. 지속 방지는 실패. |
| Q4 지속 Close | **CONFIRMED** | 최초 파지는 Action 176까지 유지; Action 177에 상실, 190부터 바닥 근처 재파지. 종료까지 안정적 운반은 실패. |
| Q5 Lift·삽입 | **CONFIRMED** | A1은 재파지 후 136.79 mm 재상승, A2는 Action 91 이후 4 cm 아래로 내려감. 둘 다 삽입 실패. |
| Q6 후속 A 출력 | **CONFIRMED** | Action 80에 양수 gripper 재출력. Hidden·Arm6은 달라진 관측을 따라 분기. 원본 전체 출력은 각 trace에 기록. |
| Q7 실패 시 접촉·Geometry·Motion | **SUPPORTED** | A0/A1은 target 증가·finger 개방·접촉력 붕괴·Peg 낙하. A2는 유지 접촉 중 Peg가 내려가고 Action 177에 target Close에도 접촉 소실. 정확한 충돌 원인은 미확정. |
| Q8 물리적 인과 기여 | **SUPPORTED** | 동일 prefix에서 Action 79 Close는 그 step의 접촉력을 회복한다. 지속 Close는 즉각적인 낙하를 지연시킨다. 한 step만의 개입은 충분하지 않다. |
| Q9 H1 대 H2 구분 | **INCONCLUSIVE** | 개입은 학습 후 Controller 경계에만 작용한다. 연속 Gripper Regression 자체와 t=79 이전 Motion6/GRU history 분포 변화의 근본 원인을 분리하지 못한다. |

## 산출물 및 검증

- [정적 HTML](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/a_gripper_intervention.html), [세 조건 after-action 비교 MP4](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/a0_a1_a2_after_action_comparison.mp4), [t=75–90 확대 PNG](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/figures/t75_90.png), [전체 타임라인 PNG](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/figures/timeline.png), [3D 궤적 PNG](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/figures/trajectory3d.png), [Hidden/Arm 차이 PNG](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/figures/hidden_arm.png).
- [frame metrics CSV](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/frame_metrics.csv), [summary JSON](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/summary.json), [analysis detail](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/analysis_detail.json), [verification](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/verification.json), [source/protection audit](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/protection_audit.json). 조건별 폴더에는 Raw/Applied Action7, GRU Hidden, Observations, Graph/Packet, Controller target와 GT/접촉 trace, MP4, PNG가 있다.
- 새 계약 테스트 + Stop-the-clock 테스트 **11/11 PASS** ([log](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/contract_tests.log)). 설치된 ManiSkill 3.0.1/SAPIEN 3.0.3에서 runnable example의 `pd_ee_delta_pose` Reset/Step 및 CPU PhysX 확인 ([log](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/simulator_api_check.log)). [Readback 검증](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/verification.json)은 NPZ/JSON/CSV/PNG/MP4 프레임 수 및 HTML 상대 경로 검사를 통과했다. 보호 대상의 시작·종료 SHA도 동일했다.
- 사용하지 않은 08-B C 체크포인트도 실행 후 SHA가 원본 receipt와 일치한다([확인 기록](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/c_checkpoint_identity_check.json)).
- 시작·종료 [자원 감사](../../experiments/dc1b894f67af4e16a378dc02c3c38a97/resource_audit.json)에서 기존 Full A/B/C PID 43369/43376/43381이 계속 실행 중임을 확인했다. 다른 프로세스에 중단·일시정지·suspend 명령을 내리지 않았다.

실제 실행 경로는 `/workspace/task08c/experiments/dc1b894f67af4e16a378dc02c3c38a97`이고 로컬 `experiments/`에 동일 산출물을 복사했다. CPU PhysX와 RTX 3090 추론은 실행했다. CUDA PhysX, MPS, 실제 50 ms 제어 마감시간은 검증하지 않았다.
