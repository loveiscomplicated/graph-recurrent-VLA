# Task07-F — Vision Geometry Oracle Reach Ablation

2026-10-26 KST. **Seed41005에서 비대상 Pixel을 제거하거나 현재 GT Segmentation을 사용해도,
고정 Checkpoint의 자율 Reach–Descend와 파지는 회복되지 않았다.** A/B/C 모두 실제 250-step을
완료했고 양측 파지·Peg Lift·Task Success가 관측되지 않았다. Mask 개입은 Action과 궤적을
실제로 바꿨으며, 첫 Close 직전 위치 오차는 A84.967mm, B193.529mm, C106.380mm였다.

ExperimentLogger UUID: **`a878851dd4ee4b7a853502915115454c`**.
실험의 범위는 **`VISION_ORACLE_CAUSAL_ABLATION_NOT_STANDARD_POLICY_EVALUATION`**이다.
B/C는 Oracle 진단이며 기존 일반 Vision Policy 성공률에 합산하지 않는다.
이 세 Episode로 일반적인 성공률이나 모집단 인과 효과를 추정하지 않는다.

[정적 HTML](../../experiments/a878851dd4ee4b7a853502915115454c/vision_oracle_reach_ablation.html) ·
[실행 결과](../../experiments/a878851dd4ee4b7a853502915115454c/results.json) ·
[전체 프레임 분석](../../experiments/a878851dd4ee4b7a853502915115454c/analysis.json) ·
[프레임 CSV](../../experiments/a878851dd4ee4b7a853502915115454c/frames.csv) ·
[검증](../../experiments/a878851dd4ee4b7a853502915115454c/verification.json).

## 실제 실행과 통제

Vast57640710의 기존 CUDA 평가 Python 환경을 재사용했다. 새 실행 위치는
`/workspace/task07-vision-oracle`이다. 학습 Pipeline Run은
`2be5e62357a44b14a5651a1a346edc7a`, 기존 Checkpoint Manifest의 학습 실행 UUID는
`002abd72baf84c4da185fd7134c3146f`다. Best epoch39/global step8959와
SHA256 `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`를 그대로 사용했다.
Train-only Robot76 Normalizer, GNN/GRU Hidden144, Model Schema, Action7,
Graph k4/M16, Shape-First/PCA5/F5, keep_ratio0.5, Variable N, Raw EDE는 고정했다.

실제 환경은 ManiSkill3.0.1/SAPIEN3.0.3/Gymnasium1.3.0,
`PegInsertionSide-v1`, `panda_wristcam`, `pd_ee_delta_pose`, PhysX CPU,
Control20Hz/Physics100Hz/Action당5 physics steps/Horizon250이다.
Vision과 GNN/GRU는 RTX3090 CUDA에서 실행했다. `base_camera`128×128의 원본 RGB-D를 사용했다.
SAM2 optional `_C` 후처리 확장 부재 경고는 기존 실행과 동일하며 설치나 알고리즘 변경으로 우회하지 않았다.

| 조건 | 실제 Graph 입력 Mask | Tracker 입력·상태 | 제어 |
| --- | --- | --- | --- |
| A / `PREDICTED` | 현재 DINO/SAM2 예측 | 원본 RGB의 기존 causal streaming | 전체 모델 자율 |
| B / `GT_FP_FILTERED` | 현재 예측 AND 현재 GT visible mask | 원본 예측을 그대로 유지 | 전체 모델 자율 |
| C / `GT_MASK_ORACLE` | 현재 GT visible mask | 원본 예측을 그대로 유지 | 전체 모델 자율 |

매 프레임에 세 조건 모두 기존 DINO/SAM2 경로를 호출했다. B/C 개입은 출력 Mask의 복사본에만
적용했으며 Tracker prompt, detection decision, temporal memory에 GT Mask를 되넣지 않았다.
개입 뒤 기존 `packet_from_frame`으로 Sampling·PCA5·F5·EE Transform·Raw EDE를 재실행했다.
이전 A의 Packet에서 Point만 삭제하거나 기존 Edge/Feature Tensor를 재사용하지 않았다.

각 조건은 Seed41005의 새 환경 Reset에서 시작했다. 시작 전에 독립 Reset 세 번으로
qpos/qvel, TCP/Finger/Peg/Box Pose, RGB-D/Calibration, Robot76 및 Reset Identity의 exact 일치를
검사했고, 각 실제 Trial의 시작도 이 기준과 대조했다. Hidden은 조건별 t0에서만 초기화하고
매 실제 Observation마다 한 번씩 갱신했다. Snapshot, Action 재생, Hidden 이식,
Expert Close, Lift/Hold 또는 위치 보정은 사용하지 않았다.

## 질문별 답과 판정

| 질문 | 판정 | 이 Seed에서 확인한 답 |
| --- | --- | --- |
| **Q1. A에서 기존 실패가 재현됐는가?** | **CONFIRMED** | 250프레임 전체 RGB-D/Mask/Graph/Source Index exact. Action·Hidden 최대 차이0. 첫 Close t27/84.967mm, 파지 실패 재현. |
| **Q2. FP 제거만으로 Close 직전 오차가 얼마나 변했는가?** | **CONFIRMED** | B193.529mm로 A보다 **108.562mm 증가**. 첫 Close도 t27→43. 서로 다른 실제 Close 상태의 결과 비교다. |
| **Q3. Full GT Mask로 파지 가능한 위치까지 접근했는가?** | **CONFIRMED — 회복되지 않음** | C 첫 Close t28/106.380mm, A보다 **21.413mm 증가**. 손가락 중심 Segment가 Peg와 교차하지 않았고 실제 파지·Lift도 없음. 모든 가능한 파지 Pose를 열거해 배제한 판정은 아니다. |
| **Q4. 첫 Close의 위치·회전·파지는 어떻게 다른가?** | **CONFIRMED** | A27/84.967mm/2.576°, B43/193.529mm/18.568°, C28/106.380mm/4.577°. 세 조건 모두 이후8 Action 및 전체250 Action 동안 양측 파지 없음. |
| **Q5. 어떤 Outlier와 Graph가 바뀌었는가?** | **CONFIRMED** | 같은 A-t27에서 Peg 비대상15개·Box6개가 B/C에서0개가 됨. Peg p95 50.179→0.542/0.533mm. Sampling·F5·Raw EDE도 함께 바뀜. 아래 동일 상태 표 참조. |
| **Q6. One-step Probe와 Free-running 영향이 일치하는가?** | **SUPPORTED** | Mask→Action 영향과 초기 t0 출력 일치는 확인. A-t27 고정 History에서는 B/C도 여전히 음수 Gripper지만 실제 Free-running 첫 Close는43/28이었다. 즉 누적된 자기 상태·History의 영향이 추가됨. 개선 방향의 일치로 해석할 수 없음. |
| **Q7. Vision 원인과 Action Learning을 어디까지 구분하는가?** | **INCONCLUSIVE** | 현재 Segmentation 오차 제거만으로 회복되지 않는 것은 확인. 정확한 관측 표면도 이 Checkpoint가 성공적으로 이용하지 못한 결과를 지지하지만, 순수 BC 오류·History·Feature/Topology·분포 차이의 기여율은 분리하지 못함. Vision 오류가 무관했다는 결론도 불가. |
| **Q8. 다음 연구 우선순위는?** | **SUPPORTED** | Reach–Descend의 Action Learning/상태 분포와 History 진단을 우선한다. Vision Outlier 개선은 입력 품질 과제로 유지하되 그것만으로 파지가 회복될 것으로 기대할 근거는 약하다. Graph는 표현·Topology/학습 분포 감사를 우선하며 구조 변경 우선이라는 근거는 없음. |

다른 Seed, 재학습, Oracle로 학습한 Policy, Graph/GNN/GRU 구조 비교 및
Vision/Action/History의 독립적인 추가 개입은 모두 **NOT_TESTED**다.

## Reach–Descend와 첫 Close

첫 Close는 사전에 정의한 **Raw Action7의 Gripper가 처음 음수가 된 Observation t**다.
조건별로 관측된 이벤트를 그대로 사용했다. t27을 B/C에 강제하지 않았다.
아래는 해당 Action을 적용하기 **직전**의 Geometry다. Reference는 같은 초기 상태의 원본
`ScriptedExpert.grasp`이며 평가에만 사용했다. World Target과 현재 Peg 기준 상대 Target 오차를
별도로 보존했다. Expert의 `act()`나 Phase를 Student 제어에 사용하지 않았다.

| 항목 | A / `PREDICTED` | B / `GT_FP_FILTERED` | C / `GT_MASK_ORACLE` |
| --- | --- | --- | --- |
| 첫 음수 Gripper t | 27 | 43 | 28 |
| Raw Gripper | −0.335427 | −0.439738 | −0.014578 |
| World Target 위치 오차 mm | 84.967 | 193.529 | 106.380 |
| World 차이 x/y/z mm | +60.157/−49.120/+34.462 | +160.337/−101.586/+37.757 | +85.754/−51.596/+36.067 |
| 회전 오차 degree | 2.576 | 18.568 | 4.577 |
| Peg-relative Target 위치 오차 mm | 84.967 | 193.529 | 106.381 |
| Left/Right fingertip 중심→Peg 표면 mm | 33.205/40.499 | 132.534/155.325 | 56.109/57.017 |
| Gripper opening mm | 68.317 | 65.567 | 63.180 |
| 양 Pad 중심 Segment–Peg 교차 | 아니오 | 아니오 | 아니오 |
| Close 전 최소 Target 오차 mm / t | 56.400 /19 | 72.313 /15 | 59.316 /17 |
| 최초 양측 접촉력≥0.5N | NOT_OBSERVED | NOT_OBSERVED | NOT_OBSERVED |
| 최초 `Panda.is_grasping` | NOT_OBSERVED | NOT_OBSERVED | NOT_OBSERVED |
| 파지 유지 실패 시점 | NOT_OBSERVED: 파지 자체 없음 | 동일 | 동일 |
| Peg≥4cm Lift / Task Success | 관측 없음 /실패 | 관측 없음 /실패 | 관측 없음 /실패 |
| Episode length / First Success Step | 250 /NOT_OBSERVED | 250 /NOT_OBSERVED | 250 /NOT_OBSERVED |

Fingertip 거리는 Rubber-tip collision box **중심점** 기준이다. 유한한 Pad 표면 간 clearance나
충분한 파지 가능성 판정으로 취급하지 않는다. 설치된 `Panda.is_grasping`은 양측 힘≥0.5N과
접촉 방향각≤85°를 검사한다. 전체 Episode에서 양측 힘 기준도 성립하지 않았고,
Peg 높이의 최대 변화는 각 조건 모두0.00002mm 미만의 수치 수준이었다.
운반·정렬·삽입의 성공적인 진행은 관측되지 않았다. Student에 Expert Stage를 붙이지 않았다.

각 최초 Close 이후8 Action7/Controller input, TCP·Peg 상대 Motion,
양손가락 힘·Pose·속도와 전체250-step Trace를 보존했다.
`analysis.json`의 `first_close_next8_actions`, `first_close_next8_grasp`와 각 Trial의
`trace.json`, `actions.npz`에서 읽을 수 있다.

[전체/접근 구간 3D 궤적](../../experiments/a878851dd4ee4b7a853502915115454c/reach_trajectory_3d.png) ·
[Action7](../../experiments/a878851dd4ee4b7a853502915115454c/native_action7.png) ·
[오차·Hidden·Topology Timeline](../../experiments/a878851dd4ee4b7a853502915115454c/timelines.png).

## 같은 상태에서의 Vision Geometry와 Representation 변화

**아래 비교는 모두 A의 동일한 t27 RGB-D/Robot76/Hidden으로 재생성한 Read-only Probe다.**
Free-running B-t43/C-t28과 혼동하지 않는다. GT Mesh는 표면 거리 **측정에만** 사용했다.
모델의 Point는 항상 원본 Depth·Source Pixel·Calibration에서 복원했고, occluded surface를 추가하지 않았다.

| A-t27 고정 상태 | A Mask | B Mask | C Mask |
| --- | --- | --- | --- |
| Peg/Box Point 수 | 172/126 | 162/132 | 162/131 |
| Peg/Box 비대상 Source Point | 15/6 | 0/0 | 0/0 |
| Peg 표면 거리 mean/median/p95/max mm | 6.566/0.273/50.179/113.284 | 0.296/0.274/0.542/0.614 | 0.279/0.273/0.533/0.690 |
| Box 표면 거리 mean/p95/max mm | 18.235/1.013/421.278 | 0.428/0.901/1.023 | 0.434/0.914/0.983 |
| Raw EDE 유향 Edge 수 | 1532 | 1602 | 1594 |
| 전체 Graph 유향 Edge 수 | 1612 | 1682 | 1674 |
| Graph connected components | 7 | 6 | 6 |
| Robot–Object / Peg–Box 유향 Edge 수 | 48/32 | 48/32 | 48/32 |
| F5 available 각 Field 수 | 298/298/298 | 294/294/294 | 293/293/293 |
| F5 valid 각 Field 수 | 276/276/298 | 284/284/294 | 283/283/293 |

원본 A-t27에서 올바른 GT Peg Pixel을 Source로 가진157개 Point의 표면 거리 평균은0.272mm였고,
비대상15개는 평균72.440mm였다. Box의 비대상6개는 평균374.215mm였다.
B/C는 이런 Source 오염을 제거했다. 같은 Depth에서도 Mask의 경계 Pixel은 큰 3D Outlier를 만들 수 있다.
특히 Box는 오염 비율이5% 미만이어서 p95가 작더라도 최대421mm Outlier가 남아 있었다.

그러나 **B는 A의 Point에서21개만 삭제한 Graph가 아니다.** 같은 t27에서 A와 공통인
Source Point는 B126개/C108개였고, B는172개 Source가 빠지고168개가 새로 선택됐다.
C는190개가 빠지고185개가 선택됐다. B의 공통 Source에서도 유효 F5 값25개,
C에서는106개가 달라졌다. Source Identity로 대조한 유향 Edge의 제거/추가는
B1328/1398개, C1418/1480개다. Feature별 단위와 차이는 JSON에 따로 보존했다.
따라서 특정 Outlier 하나의 인과 효과나 단순 Point 삭제 효과로 해석하지 않는다.
Component 수 감소나 Edge 수 증가를 Graph의 물리적 연결 정확성 향상으로 단정하지 않는다.

Free-running 전250프레임에서도 B/C의 Peg·Box 비대상 Source Point는 모두0이었다.
각자의 첫 Close에서 B/C Peg 표면 거리 p95는0.537/0.567mm,
Box p95는0.909/0.942mm였다. C의 applied Peg/Box Mask IoU는1이었다.
입력 정합이 회복됐어도 실제 접근/파지 개선으로 이어지지 않았다.

[같은 상태의 Geometry](../../experiments/a878851dd4ee4b7a853502915115454c/same_state_geometry.json) ·
[Sampling/F5/Edge 차이](../../experiments/a878851dd4ee4b7a853502915115454c/same_state_graph_changes.json) ·
[같은 상태 Outlier 분포](../../experiments/a878851dd4ee4b7a853502915115454c/same_state_t27_outliers.png) ·
[첫 Close RGB/Mask/Graph 나란히 보기](../../experiments/a878851dd4ee4b7a853502915115454c/first_close_vision_geometry.png).

## One-step와 Free-running의 구분

A의 t0/13/19/26/27/34 Observation과 **그 Observation 전의 A Hidden**을 고정해
각 Mask의 새 Graph를 모델에 한 번 입력했다. Probe Action을 환경에 적용하지 않았고,
출력 Hidden을 실제 Trial에 이식하지 않았다. A Probe 출력은 저장된 실제 A Action과 일치했다.
t0에서는 B/C Probe도 각각 실제 B/C의 첫 Action과 일치해, 동일 초기 상태의 Mask 개입이
처음부터 Action 차이를 만들었다는 것을 확인했다.

| 같은 A 상태 t | B−A translation command 차이 norm mm | C−A 차이 norm mm | B−A raw Gripper | C−A raw Gripper |
| --- | --- | --- | --- | --- |
| 0 | 0.190 | 1.044 | −0.007549 | −0.036972 |
| 13 | 0.715 | 0.494 | −0.016611 | −0.013233 |
| 19 | 0.253 | 0.196 | −0.003529 | −0.001125 |
| 26 | 1.591 | 1.641 | +0.078348 | +0.081146 |
| 27 | 0.876 | 1.037 | +0.057973 | +0.063566 |
| 34 | 0.527 | 0.537 | +0.003187 | +0.003052 |

Translation 값은 native XYZ 차이에 Controller의0.1m scale을 적용한 **명령 차이**이며,
실제로 달성한 TCP 변위가 아니다. Rotation과 전체 Action7은 원본 Probe JSON/NPZ에 있다.
A-t27 고정 상태에서는 Gripper A−0.335427/B−0.277454/C−0.271861로 모두 음수였다.
그런데 Free-running에서는 B의 첫 음수가43, C가28이었다. 순간 Mask만의 출력과
누적된 자기 Observation/Action/Hidden의 결과는 다르다. 물리 상태와 History의 개별 기여율까지
이 Probe로 분리하지 않았다.

[One-step Probe](../../experiments/a878851dd4ee4b7a853502915115454c/one_step_probes.json).

## 학습 분포와 해석 한계

고정 Train96개 Episode의 실제 Action-target11975프레임을 원본 Vision Manifest에서 대조했다.
학습 Peg/Box Presence는0.514/0.752, Point Count 중앙값은69/123이었다.
이번 A/B/C는 모두 Peg/Box Presence1.0이었다. 전체 Episode Point Count 중앙값은
A176/145, B177/134, C179/134였다. 세 조건의 Count는 모두 학습의 객체별 p01–p99
범위(Peg0–212, Box0–482) 안에 있었다. **과도한 Point 수 증가가 관측되지는 않았다.**

이것은 Mask·F5·Topology·Geometry·History의 Joint Distribution이 학습과 같다는 증명은 아니다.
학습은 성공 Expert 궤적을 포함하며 이번 실패 Rollout은 다른 상태에 오래 머물렀다.
Stage를 맞추지 않은 전체 분포 요약을 Oracle만의 OOD 증거로 해석하지 않는다.

현재 증거가 지지하는 좁은 결론은 **현재 비대상 Pixel/Segmentation 오류를 제거하는 것만으로는
이 Checkpoint·Seed의 실패가 해소되지 않는다**는 것이다. Vision 오류의 Action 영향은 확인했지만,
원래 실패에 대한 순수한 기여율이나 BC 오류와의 상대 크기는 **INCONCLUSIVE**다.
BC Action Learning, 상태/History 의존성, 남은 Depth/Feature/Topology 한계, Oracle 표현의
분포 차이는 추가로 구분해야 하는 가설이다. GNN/GRU Architecture가 원인이라는 결론은 내리지 않는다.

[학습 분포 대조](../../experiments/a878851dd4ee4b7a853502915115454c/training_distribution.json).

## 계약·Artifact·보존 검증

실제750 Control Actions/3750 Physics Steps/750 Hidden Updates를 확인했다.
Observation·GT Texture 획득·Vision·Graph·모델 추론 중 추가 Physics Step은0이었다.
Controller entry는 원본 native Action7과 exact였고 설치된 translation/rotation/gripper scale도 다시 계산했다.

GT ID는 Peg18/Box19였다. 설치된 Camera의 현재 `get_obs()` texture에서 RGB와 raw depth가
현재 Observation과 exact인지 매번 검사했다. 초기 visible GT Pixel을 Calibration으로 복원해
독립 GT Surface와 대조한 p95는 Peg0.568mm/Box0.919mm였다. Actor ID·Pixel 정합과
원래 Source Index로부터 camera→world→EE 복원을 검증했다. GT Pose/Mesh/Goal/Contact와
Expert Reference는 진단 Sidecar에만 남기고 모델 입력으로 제공하지 않았다.

- 로컬 관련 계약 테스트 **63개 PASS**, 서버 신규 계약 테스트 **6개 PASS**.
- 같은 저장 t27 Fixture에서 A의 원본 Packet/Graph exact, B/C 전체 재생성,
  A Graph 재사용 거부, 미래/과거/다른 Episode GT·Pixel 정합 오류 거부,
  Hidden 이식·시간 불연속 거부를 검사했다.
- 로컬 macOS의 저장 Frame 재계산은 F5의 미세 수치와 Raw EDE에서 Linux와 차이가 있었다.
  같은 로컬 Runtime의 Adapter/direct builder는 exact였고, **실제 Linux 실행 전에는 저장된
  Linux Packet/Graph exact 일치를 별도로 요구하여 통과했다.** 이를 플랫폼 간 exact라고 보고하지 않는다.
- 실제 Run의 Packet/Graph를 독립 Readback하고 모델 입력·Raw EDE 계약·Controller·Physics·Hidden·
  초기 상태·Oracle Source Pixel을 다시 검사했다. JSON/NPZ/CSV/PNG/MP4 및 상대 링크의 최종
  Decode 결과는 `local_verification.json`에 있다.
- Checkpoint/Normalizer의 메모리 hash 전후가 같았다. 기존 Source·Checkpoint Bundle·학습 Run·
  07-E·기존41005 Artifact **18,695개 불변**, 원본 학습 HDF5 **120개**가 고정 Manifest SHA256에 일치했다.
- 추가 감사의 첫 시도는87개 상대 경로를 SSH 로그인 디렉터리에서 찾아 오탐했다.
  원래 실행 cwd로 재검사해 변경0개를 확인했다. 실패한 감사와 정정 이유도 보존했다.
- 실행 전후 GPU는1MiB 사용 상태였고 기존 Process를 중지하거나 강제로 메모리를 확보하지 않았다.
  Runner는 정상 종료했다. 원본 파일·학습 데이터·가중치·Normalizer·Production Pipeline을 수정하지 않았다.
- 신규 Adapter/Runner/Report/Config/Test/이 문서만 추가했다. Dependency 설치,
  Training/Fine-tuning/HPO, Git Commit, 외부 Publishing/Upload는 수행하지 않았다.
  기존 서버로의 실행 소스 전달과 결과 회수만 수행했다.
- macOS Simulator/MPS, CUDA PhysX, 추가 Seed 및 새로운 공식 성공률 평가는 **NOT_EXECUTED**다.

[최종 Decode/Link Readback](../../experiments/a878851dd4ee4b7a853502915115454c/local_verification.json) ·
[보호 감사](../../experiments/a878851dd4ee4b7a853502915115454c/final_protection.json) ·
[학습 HDF5 검사](../../experiments/a878851dd4ee4b7a853502915115454c/dataset_integrity.json) ·
[로컬 테스트](../../experiments/a878851dd4ee4b7a853502915115454c/local_contract_tests.log) ·
[서버 테스트](../../experiments/a878851dd4ee4b7a853502915115454c/remote_contract_tests_final.log) ·
[실행 Source Hashes](../../experiments/a878851dd4ee4b7a853502915115454c/source_hashes.json) ·
[구현 Diff](../../experiments/a878851dd4ee4b7a853502915115454c/implementation.diff).

각 조건 `trials/A`, `trials/B`, `trials/C`에 원본 RGB-D/Calibration/Robot76,
Predicted/Cleaned/GT/Applied Mask, Segmentation/Actor ID, Packet/F5/Validity/Source Index/Raw EDE,
Graph Input, Action7/실제 Controller, Hidden 및 GT Pose/Contact/Velocity Trace를 보존했다.
`probes/`는 A의 같은 상태에 대한 독립 Read-only 입력이다. `dependencies/original_41005/`는
기존 평가와의 대조용 복사본이다. 보고서 이미지·영상은 각자의 실제 상태를 표시한다.

[Action 후 RGB 비교 영상](../../experiments/a878851dd4ee4b7a853502915115454c/rollout_comparison_after_action.mp4) ·
[각 최초 Close에 맞춘 영상](../../experiments/a878851dd4ee4b7a853502915115454c/close_aligned_after_action.mp4).
Close 정렬은 관찰용이며 서로 다른 시점을 동일 물리 상태로 취급하지 않는다.

실행 명령은 Artifact의 `run.sh`/`bootstrap.py`에 보존했다.

```bash
bash /workspace/task07-vision-oracle/run.sh
```

이 Opt-in 명령은 새 Logger UUID로 동일한 고정 프로토콜을 실행한다. 이번 작업의 실험은 종료했으며
추가 물리 실험이나 Production Oracle 채택을 자동으로 시작하지 않는다.
