# Task07-E — Autonomous Reach–Descend Failure Causal Diagnosis

2026-10-26 KST. **Seed 41005에서는 정상 Expert 접근 뒤 Model Close가 성공했지만,
동일한 자율 Model 접근 뒤에는 Expert Close와 Model Close가 모두 실패했다.**
닫기 직전 모델 TCP의 Expert Grasp Target 오차는 **84.967mm / 2.576°**였다.
따라서 이 Seed의 최초 파지 실패는 Close 명령만으로 설명하기보다,
**Close 전에 파지에 불리한 상대 위치를 형성한 문제**로 보는 것이 지지된다.
Vision의 경계 오염이 이 접근 오차에 기여했는지는 이번 개입만으로 분리하지 못했다.

ExperimentLogger UUID: **`c76d3cf8f8354c7199e13f3068d2d1c3`**.
이 결과는 `MIXED_CONTROL_CAUSAL_DIAGNOSTIC_NOT_POLICY_SUCCESS_RATE`다.
Expert 접근과 공통 Lift/Hold를 포함하므로 기존 Held-out 본평가나 Policy Closed-loop Success Rate에 합산하지 않는다.
추가 Seed, 학습, 모델 개선 또는 기존 본평가 결과 수정은 수행하지 않았다.

[정적 HTML](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/autonomous_reach_diagnosis.html) ·
[실행 결과](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/results.json) ·
[기하·Vision 분석](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/analysis.json) ·
[프레임 CSV](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/diagnostic_frames.csv) ·
[Q1–Q7 판정](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/interpretation.json).

## 실제 실행 및 조건

기존 학습 Run `2be5e62357a44b14a5651a1a346edc7a`의 Best epoch39/global step8959,
SHA256 `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`를 고정했다.
원본 Train-only Robot76 Normalizer, Shape-First/F5/Raw EDE, Graph, GRU Hidden144,
Action7 및 DINO/SAM2 Causal Streaming을 그대로 사용했다.

Vast57640710의 기존 평가 Python 환경에서 CUDA Vision/GNN/GRU와 CPU PhysX를 실제 실행했다.
ManiSkill3.0.1/SAPIEN3.0.3/Gymnasium1.3.0, Panda `panda_wristcam`,
`PegInsertionSide-v1`, `pd_ee_delta_pose`, Control20Hz/Physics100Hz,
Action당 Physics5회, Horizon250이다. 환경·Controller 설정은 원본 학습 조건과 대조했다.
실행 디렉터리는 `/workspace/task07-autonomous-reach`이며 기존 checkout을 수정하지 않았다.

각 Trial은 새 환경을 Seed41005로 Reset했다. Snapshot Teleport, Hidden 이식 및 저장 Action 재생은 없다.
매 실제 Observation에서 Student가 새 Action을 계산하고 Hidden을 한 번 업데이트한다.
A/Positive Control의 Expert 접근 중에도 Student는 같은 순서로 관측하지만 환경은 Expert가 제어한다.
GT Pose·접촉·Segmentation·Expert Stage·Grasp Target은 Student Input에 전달하지 않았다.
GT Segmentation은 현재 Camera render texture만 읽었으며 별도의 physics/render step을 추가하지 않았다.

| 조건 | Reach/Descend | Close8 | Close 시작 t | Close8 직후 파지 | Hold20 유지 | Hold 최소 Peg/TCP 상승 |
| --- | --- | --- | --- | --- | --- | --- |
| Positive Control | Expert | Arm0 / Gripper−1 | 40 | 성공 | 성공 | 79.240 / 78.800mm |
| A | Expert | 실제 Model Arm + Gripper | 40 | 성공 | 성공 | 78.828 / 79.085mm |
| B | Model | Arm0 / Gripper−1 | 27 | 실패 | 실패 | 약0 / 79.860mm |
| C | Model | 실제 Model Arm + Gripper | 27 | 실패 | 실패 | 약0 / 79.615mm |

네 Trial 모두 실행 계약을 완료했다. **실험 실행 PASS와 파지 성공/실패는 구분한다.**
Positive Control은 A/B/C 성능 수치에 합산하지 않는다. B/C는 t<27에서 음수 Gripper가 없고,
t27에서 최초 음수가 출력됨을 검증했다. 닫기 시점을 이동하거나 Seed를 추가하지 않았다.

## 공통 Lift/Hold와 최초 접촉

07-D의 사전 정의 기준을 그대로 유지했다. Close 시작 TCP World Pose를 기준으로
World +Z8cm, 동일 Orientation을 목표로 삼았다. Close8 뒤 추가 settling 없이
Lift20/Hold20을 수행하고, 기존 Expert `_servo`의 TCP 피드백·gain3·최대 이동2cm·최대 회전0.15rad를 사용했다.
Lift/Hold Gripper는 모두 −1이다. Hold20 전체에서 Peg≥4cm 상승, TCP≥6cm 상승,
설치된 `Panda.is_grasping`이 모두 참이어야 유지 성공이다.
`is_grasping`은 양측 각각 힘≥0.5N 및 접촉 방향각≤85°를 사용한다.

- A와 Positive Control은 **첫 Close Action t40 후부터** 양측 접촉·파지 판정이 참이었다.
  Close8 전체와 Hold20 전체에서 파지가 성립했다. 공통 후속 닫기가 나중에 처음 회복시킨 결과가 아니다.
- B/C는 Close8 동안 파지 판정이 0/8이고, Close8 직후 양손가락 접촉력 벡터가 모두 `[0,0,0]N`이었다.
  Lift/Hold 중에도 최초 양측 접촉·파지가 관측되지 않았다. TCP는 실제 상승했지만 Peg는 상승하지 않았다.
- Model Close가 B와 다른 실제 TCP 경로를 만들었으나, 두 조건 모두 파지 실패였다.
  Close 종료 Target 위치 오차는 B86.499mm, C107.129mm였다. Model Close가 항상 무해하다는 주장은 하지 않는다.

[실제 After-action 비교 영상](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/close_lift_hold_comparison.mp4)과
[Close/Lift/Hold Montage](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/close_lift_hold_montage.png)는
각 조건의 자기 Close 시작을 맞춘다. 서로 다른 t40/t27을 같은 절대 Timestep의 Action 비교로 취급하지 않는다.
각 Trial 원본 영상은 Action 전 RGB이며 비교 영상은 해당 Action 후 RGB다.

## B/C 개입 이전 동일성

B/C의 Reset Identity와 t0..27의 RGB-D/Calibration, Robot76, TCP/Fingertip Geometry,
Predicted Mask, Graph Node/Edge/Source Index는 exact였다.
FP32 Action/Hidden은 고정 `atol=1e-6, rtol=1e-5`를 적용했고 실제 최대 차이는 각각 **0**이었다.
매 Step의 qpos/qvel, TCP/Finger/Peg/Box Pose 및 Linear/Angular Velocity도
`atol=1e-6, rtol=0`으로 검사했고 최대 성분 차이는 **0**이었다.
위치 단위는 m, Quaternion은 무차원, 속도는 m/s·rad/s, 관절은 rad/m·각 속도 단위다.
동일성은 첫 Close 추론까지이며 그 Action을 적용한 이후에는 강제 동일화하지 않았다.

A와 Positive Control의 t0..40도 같은 기준으로 통과했다.
추가로 새 C의 t0..27은 **기존 Held-out41005의 RGB-D·Mask·Graph/Source Index와 exact**였고,
기존 Action/Hidden도 최대 차이0으로 재현했다.
Inference 중 Physics/elapsed step 불변, Action 후 Physics5/Control1을 실제 계측했다.
총326 Control Actions, 1,630 Physics Steps, 326 Hidden Updates를 실행했다.
Readback에서는 각 현재 TCP로 공통 Servo 명령을 다시 계산해 실제 Applied Action과 대조했다.

## Close 직전 기하와 접근 궤적

Reference는 같은 Seed 초기 상태에서 원본 `ScriptedExpert`가 계산한 `grasp`다.
World 위치 오차와 Peg 기준 TCP Relative Pose 오차를 각각 보고한다.
회전은 정규화 Quaternion의 `2 acos(|q1·q2|)`로 계산해 부호 모호성을 제거했다.
아래 위치는 mm, 회전은 degree다. Quaternion·속도 원시값은 Trace에 보존했다.

| Close 직전 | A / Positive Control, t40 | B/C, t27 |
| --- | --- | --- |
| TCP→Expert Grasp Target 위치 오차 | 1.760 | **84.967** |
| World 위치 차이 x/y/z | −0.935 / +0.093 / −1.488 | **+60.157 / −49.120 / +34.462** |
| 회전 오차 | 0.157° (0.002733rad) | 2.576° (0.044959rad) |
| Peg 기준 상대 위치 오차 | 1.760 | 84.967 |
| Peg 기준 위치 차이 x/y/z | +0.811 / +0.475 / −1.488 | **−77.543 / +4.351 / +34.462** |
| Peg 기준 상대 회전 오차 | 0.157° | 2.576° |
| Left/Right Fingertip 중심→Peg 표면 거리 | 23.723 / 24.673 | 33.205 / 40.499 |
| Gripper Opening | 80.000 | 68.317 |
| 양 Pad 중심 연결 Segment가 Peg Cuboid와 교차 | 예 | **아니오** |

Fingertip 거리는 실제 Rubber-tip collision box **중심점**에서 GT Peg Cuboid 표면까지의 signed 거리다.
유한한 Pad 표면 간 Collision Clearance와 혼동하지 않는다. Segment 교차는 손가락 사이 위치를 설명하는
기하 지표이며, 그 자체가 충분한 물리적 Graspability 판정은 아니다.
B의 실제 Full Close 실패와 Positive Control의 실제 파지가 이 기하 비교를 보완한다.
B/C는 Target보다 World Z34.462mm 높고 XY도 약77.7mm 어긋난 상태에서 닫기 시작했다.
A와 B/C의 실제 Close Pose를 직접 비교한 Peg-relative 위치 차이는
약 `[−78.353,+3.876,+35.950]mm`다.

[3D TCP 궤적](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/reach_trajectory_3d.png)과
[Close 위치/회전 비교](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/close_geometry_comparison.png)를 보존했다.
Expert의 기록 Stage는 reach t0..25, descend t26..39, close t40..47이다.
Student에 동일 Phase Label을 붙이지 않았다.

Student는 시작 Target 거리141.453mm에서 t3에127.038mm로 줄어 최초 유의미한 접근을 보였다.
이 시점의 기준은 실행 전에 고정한 초기 Target 거리 대비10mm 감소다.
TCP–Peg 중심 거리는 t13에102.438mm로 최소였지만, Target 위치 오차는 당시65.736mm / 회전28.894°였다.
Target 위치 오차의 최소는 별도로 t19의56.400mm였다. 이후 t26에78.701mm,
첫 음수 Gripper t27에84.967mm로 다시 커졌다.
즉 회전은 상당히 맞춰 갔지만 위치를 유지하며 충분히 하강해 파지 상태를 만들지 못했다.
Close8 종료·Lift 종료·Hold 종료를 포함한 전 시점 수치는 각 Trial `diagnosis.json`의 `key_states`에 있다.

**최초 결정적인 관찰 지점은 첫 Close 직전 t27**이다. 이미 큰 위치 오차와 불리한 손가락–Peg 배치가 있었고,
Arm0/Full Close로도 잡지 못했다. t13이나 t19를 “실패가 확정된 최초 원인 시점”이라고 지정하지는 않는다.
접근 중 거리가 큰 것은 정상일 수 있고 다른 유효한 파지 Pose의 존재를 완전히 검사한 실험도 아니다.

## Vision/Graph 진단

모든 Step에서 원본 RGB-D/Calibration, Predicted Peg/Box Mask, 겹침, 객체별 Point 수·Presence,
Source Index, Shape-First Point·F5/Validity, Raw EDE와 소비 Graph를 저장했다.
GT Segmentation은 기존 Mask를 보정하거나 새로운 Point를 만드는 데 사용하지 않았다.
기존에 선택된 Point의 Source Pixel이 어느 실제 객체인지 사후 분류했다.

B/C의 t0..27에서 Peg/Box Point 관측 누락은 없었고 Mask Overlap은0이었다.
닫기 시작 t27의 결과는 다음과 같다.

| 항목 | Peg | Box |
| --- | --- | --- |
| GT / Predicted Pixels | 398 / 429 | 524 / 534 |
| IoU | 0.883827 | 0.977570 |
| Precision / Recall | 0.904429 / 0.974874 | 0.979401 / 0.998092 |
| Shape-First Point 수 | 172 | 126 |
| 선택 Point의 Source가 올바른 GT 객체인 비율 | 91.279% | 95.238% |
| 상대 객체 GT 영역으로 잘못 들어간 Mask Pixel | 0 | 0 |

접근 전체 평균 IoU는 Peg0.908365 / Box0.914828이며 최소는0.752874 / 0.767442였다.
객체 소실이나 Peg/Box의 전면적인 Identity Swap은 관측되지 않았다.
하지만 Peg Point의 GT 표면 절대 거리 t27은 **평균6.566mm / p95 50.179mm / 최대113.284mm**였다.
2D 경계의 일부 Background/비대상 Pixel이 실제 큰 3D Outlier를 만든다.
따라서 높은 IoU·Presence만으로 제어에 충분히 정확한 Geometry라고 판정하지 않는다.
정상 Expert 접근에서도 경계 오염이 존재했다는 사실은 그 오염이 자율 행동에 미치는 영향을 배제하지 않는다.
t27에서 올바른 GT Peg Pixel에 속한 기존157개 Point는 표면 거리 평균0.272mm / 최대0.610mm였지만,
나머지15개는 평균72.440mm / 최대113.284mm였다. 전체 Point를 같은 정확도로 취급할 수 없다.
이 분류는 저장 Point를 진단용으로 나눈 것이며 입력 Point를 제거·보정하지 않았다.
올바른/잘못된 GT Source별 통계는 `analysis.json`에 보존했다.

각 Trial `vision_geometry.png`에는 같은 프레임의 RGB, Predicted Mask, 독립 GT Mask,
Shape-First/Raw EDE 투영을 나란히 배치했다. 기존 Rollout 영상에는 소비 Graph inset이 있다.
Mask 경계·Depth·Point 선택·Graph 집계가 접근 Action에 얼마나 기여했는지는 개입하지 않았다.

## 질문별 판정과 남은 가설

| 질문 | 판정 | 이 Seed에서의 답 |
| --- | --- | --- |
| Q1 정상 Expert 접근 뒤 Model Close가 잡는가? | **CONFIRMED** | A 성공. Positive Control도 성공. |
| Q2 Model 접근의 Close 직전 오차는? | **CONFIRMED** | Target84.967mm / 2.576°, Z+34.462mm 및 XY 이탈. |
| Q3 Model 접근 뒤 Expert Close로 잡는가? | **CONFIRMED** | B 실패. Close/Lift/Hold 중 양측 파지 없음. |
| Q4 같은 접근에서 두 Close 결과가 다른가? | **CONFIRMED** | 둘 다 실패. 경로는 달라지지만 교체가 파지를 회복시키지 않음. |
| Q5 실패가 이미 Close 전 상태 형성에 있었는가? | **SUPPORTED** | A 성공+B/C 실패, 큰 사전 위치 오차와 실제 Full Close 실패가 지지. |
| Q6 Vision Geometry가 충분히 정확했는가? | **INCONCLUSIVE** | Identity/관측 지원은 있었지만 큰 3D 경계 Outlier가 남음. |
| Q7 다음 진단 우선순위는? | **SUPPORTED** | Reach/Descend 상태 형성과 Action을 먼저 추적하고 Vision Outlier 기여를 함께 분리. Gripper 단독 교체 우선순위는 낮음. |

현재 결과는 이 Seed의 실패를 **초기 Gripper 약함만**, **Close 중 Arm 잔차만**, 또는
**정상 접근 뒤 Model Close가 반드시 실패함**으로 설명하는 가설을 지지하지 않는다.
공통 Lift가 실행되지 않아 실패했다는 설명도 실제 TCP 상승으로 배제된다.
고정 입력 경로·Hidden 재현 불일치와 객체 관측 완전 소실/전면적 Identity Swap의 증거도 없었다.

남는 질문은 **관측된 비대상 Vision Point가 접근 Action의 위치 이탈을 얼마나 유발했는지**, 그리고
정확한 Geometry에서도 학습된 접근 응답이 같은 오차를 만드는지다.
정밀한 Vision 입력 통제나 접근 상태만의 별도 개입 없이는 순수 Action Learning Error와 Vision 원인을 인과적으로 순위화할 수 없다.
GT-corrected Mask/Point 실험, 추가 Pose 보정 실험, 추가 Seed 및 이후 Transport/Align 평가는 **NOT_TESTED**다.
한 Seed 결과를 전체 성공률이나 일반 인과 효과로 확대하지 않는다.

## 검증·보존·재현

새 Opt-in Script/Config, 분석 Script 및 계약 테스트만 추가했다.
기존 `grasp_intervention`의 Logger-independent I/O·Retention 기준을 재사용하고,
기존 Simulator/Observation/Vision/Graph/Model/Expert를 호출했다. Production Default나 공통 API는 변경하지 않았다.
공통 `utils.experiment_logger.ExperimentLogger`로 새 UUID를 만들고 로컬 Artifact를 등록했다.
외부 W&B/HF 업로드·Publishing, Dependency 설치, Git Commit, Training/Fine-tuning/HPO는 수행하지 않았다.

- 로컬 관련 계약 테스트 **21개 PASS**, 서버 신규 계약 테스트 **7개 PASS**.
- 실제 네 Trial의 물리 시간·개입·Hidden·공통 Servo·Retention을 독립 Readback으로 대조했다.
- Checkpoint/Normalizer의 메모리 값 전후 hash가 같고, 원본 Source·학습·기존 Artifact 보호 Inventory **23,581개가 불변**이었다.
- 원본 Dataset120개 HDF5도 고정 Checkpoint와 함께 보존된 기존 Manifest SHA256에 모두 일치했다.
  실험 후 새로 만든 기준 hash와의 비교가 아니다.
- 실행 전 기존 평가 Process는 종료 상태였고 GPU는1MiB 사용 중이었다. 다른 Process를 중지하거나 자원을 강제로 확보하지 않았다.
  종료 후 GPU도1MiB로 돌아왔다.
- 로컬 독립 Readback에서 NPZ704개, PNG19개, MP4 5개/374프레임, CSV1개와 HTML 상대 링크47개를 검증했다.
  3D 궤적·실제 Close/Hold Montage·Vision 비교 그림도 직접 확인했다. Packet의 원래 미정의 Feature는
  기존 Availability/Validity 계약으로 검사하고, 실제 모델 입력·Action·Hidden의 finite 검사는 별도로 적용했다.
- macOS Simulator/MPS, CUDA PhysX, 추가 Seed 및 완전 자율 본평가는 이번 작업에서 **실행하지 않았다**.

[서버 검증](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/verification.json) ·
[로컬 독립 검증](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/local_verification.json) ·
[Dataset120개 불변 검사](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/dataset_integrity.json) ·
[최종 불변 감사](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/final_protection.json) ·
[Source Hashes](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/source_hashes.json) ·
[Implementation Diff](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/implementation.diff) ·
[로컬 테스트](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/local_contract_tests.log) ·
[서버 테스트](../../experiments/c76d3cf8f8354c7199e13f3068d2d1c3/remote_contract_tests.log).

실제 실행 명령과 환경은 Artifact의 `run.sh`, `bootstrap.py`, `reproduce.txt`에 있다.

```bash
bash /workspace/task07-autonomous-reach/run.sh
```

이 명령은 고정 Seed41005의 새 진단 Run을 만든다. 원본 평가 checkout을 import하되
새 Script는 별도 디렉터리에서 로드하고 CUDA를 먼저 초기화한다.
실제 실행된 Bootstrap/Config/Source Snapshot·Hash·Diff·로그를 함께 보존했다.
각 Trial의 RGB-D/Robot76/Mask/Packet/Graph, 예측·적용 Action/Controller,
Hidden, GT Pose/Contact/Velocity 및 Episode/Condition Identity는 `trials/<condition>/`에 있다.
기존41005의 Readback 입력 복사본은 `dependencies/original_41005/`로 분리했고 원본은 수정하지 않았다.
실험은 종료했으며 추가 물리 실험을 자동으로 시작하지 않는다.
