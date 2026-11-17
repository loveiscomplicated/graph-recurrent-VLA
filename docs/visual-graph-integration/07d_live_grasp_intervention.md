# 고정 Checkpoint — 실제 닫기 8프레임의 혼합 제어 진단

## 실행 목적과 범위

07c에서 발견한 덜 닫는 명령과 닫기 중 팔 움직임이 실제 Peg 파지를 깨뜨리는지,
정상적인 Expert 접근 이후 네 가지 조건으로 비교한다. 이 실험은
**`MIXED_CONTROL_CAUSAL_DIAGNOSTIC_NOT_POLICY_SUCCESS_RATE`**이며,
완전 자율 Policy 성공률이나 삽입 성공률로 합산하지 않는다.

- ExperimentLogger UUID: `d046e3a679a3430aab5e69f279c3f0a7`.
- 고정 Best Checkpoint: epoch39, global step8959.
- SHA256: `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`.
- 학습 Run: `2be5e62357a44b14a5651a1a346edc7a`, 원본 Train-only Normalizer 그대로 사용.
- 기존 07c의 Train30000·30003, Validation30007·30012. 결과를 보고 seed를 추가·교체하지 않는다.
- 4개 seed × 4개 조건 = 16회. 기존 Held-out 본평가와 별도 Run·디렉터리다.
- 실제 실행 위치: Vast57640710의 `/workspace/task07-grasp-intervention`.

## 시간·상태·개입 계약

ManiSkill3.0.1/SAPIEN3.0.3/Gymnasium1.3.0, physx_cpu, Panda,
`pd_ee_delta_pose`, control20Hz/physics100Hz, Action당 physics5회, horizon250을 유지한다.
모델·DINO·SAM2는 기존 CUDA 평가 환경에서 실행한다. wall-clock 추론 시간은 제한하지 않는다.

각 실행은 새 환경을 같은 seed로 Reset하고 원본 `recurrent.expert.ScriptedExpert`가
reach→descend를 실제 실행한다. 원본 Dataset의 Expert source hash와 같은지 검사한다.
그 동안 모델도 **현재 실제 RGB-D/Robot76**을 매 Step 한 번씩 받아 Graph를 만들고 Hidden을 누적한다.
저장 Expert 영상·Graph·과거 Hidden을 이식하지 않는다.

Expert가 close로 전환된 다음 Observation부터 정확히 8개의 Action만 바꾼다.

| 조건 | close 팔6D | close Gripper |
| --- | --- | --- |
| `expert` | 0 | -1 |
| `model` | 현재 모델 | 현재 모델 |
| `replace_gripper` | 현재 모델 | -1 |
| `stop_arm` | 0 | 현재 모델 |

close 도중에는 바뀐 실제 상태에서 새 Observation을 받고 모델/Hidden을 계속 진행한다.
서로 다른 조건에 같은 예측 Action 열을 재생하지 않는다. Hidden Reset은 각 실행 t0에서만 한다.
같은 seed의 Reset identity, Expert prefix의 RGB-D/Calibration/Robot76/Mask/Graph,
첫 close 추론까지의 Action·Hidden을 대조한다. Observation/Graph는 exact,
Hidden/Action은 사전 고정 FP32 `atol=1e-6, rtol=1e-5`, close 직전 위치는 `atol=1e-6m`,
quaternion 성분은 `atol=1e-6`이다.
불일치가 생기면 그 이후 결과를 같은 초기 조건의 비교로 통과시키지 않고 중단한다.

모든 Step에서 실제 `scene.step()` 횟수를 계측해 추론 중 0회, Action 후 5회를 검사한다.
실제 Controller entry와 전처리 결과도 적용한 native Action7에 대조한다.

## 공통 들어 올리기와 판정

8개 close 직후 추가 정지 없이 공통 lift20프레임, hold20프레임을 실행한다.
목표는 **개입 전 close 시작 TCP 위치 + world z8cm, 같은 방향**이다.
기존 Expert `_servo`의 Robot TCP feedback/gain/최대 이동2cm를 그대로 사용한다.
조건마다 같은 절차·같은 목표를 쓰며, 실제 servo 명령은 조건별 TCP 상태에 반응한다.
Peg/Goal GT를 이용한 운반·정렬·삽입·Rescue는 하지 않는다.

lift/hold의 Gripper는 전 조건 -1이다. 이 공통 후속 닫기가 초기의 약한 닫기를 회복할 수 있으므로,
최종 유지 판정만으로 close8 동안 이미 파지가 성립했다고 해석하지 않는다.
**close8 직후 접촉/손가락 힘, lift 중 최초 접촉, hold 중 높이·접촉**을 구분해 보존한다.

실행 전에 고정한 유지 기준은 hold20프레임 모두에서 다음 세 조건을 충족하는 것이다.

1. Peg 중심이 close 시작보다 4cm 이상 상승.
2. TCP가 close 시작보다 6cm 이상 상승해 실제 lift 실행이 확인됨.
3. 설치된 `Panda.is_grasping`을 충족: 양손가락 접촉력 각각 0.5N 이상, 접촉 방향각 85도 이하.

GT Peg pose·양손가락 접촉력은 독립적인 결과 기록에만 사용한다.
Model input은 기존 `observation()`의 RGB-D/Camera/Robot whitelist다.
Presence는 point 지원 유무이며 Vision의 의미적 정확성을 뜻하지 않는다.

## 결과

**16회 모두 실제 실행을 완료했고, 네 조건 모두 파지 유지 기준을 충족했다.**
모든 조건에서 close8 직후 이미 양측 파지 판정이 참이었다.
따라서 최종 lift 성공을 모두 뒤늦은 공통 닫기의 회복으로 설명할 수는 없다.
다만 lift/hold 동안의 -1 유지가 이후 미끄러짐을 방지한 영향은 이 설계에 포함된다.

| Seed /Split | Expert 대조군 | 현재 모델 | Gripper만 대체 | 팔0만 대체 | 현재 모델 hold 최소 Peg 상승 |
| --- | --- | --- | --- | --- | --- |
| 30000 /Train | 유지 | 유지 | 유지 | 유지 | 7.899cm |
| 30003 /Train | 유지 | 유지 | 유지 | 유지 | 8.204cm |
| 30007 /Validation | 유지 | 유지 | 유지 | 유지 | 7.952cm |
| 30012 /Validation | 유지 | 유지 | 유지 | 유지 | 8.290cm |

각 조건의 hold20프레임 모두에서 양측 파지 기준을 충족했다. 총 제어1,540회,
Physics7,700회, 모델 Hidden 업데이트1,540회를 실제 실행했다.
위 표는 **고정된 4개 정상 접근 사례의 혼합 제어 결과**이며 Policy 성공률이 아니다.

독립적인 Artifact readback에서 16회 모두 계약을 통과했다.
조건 간 첫 close 추론까지의 RGB-D/Camera/Robot76/Mask/Graph/Source Index가 exact였고,
Action과 Hidden의 최대 절대 차이도 각각 **0**이었다.
모든 Trace에서 Physics5회/Control1회 및 실제 적용 Action과 Controller entry 일치를 확인했다.
Weight와 Normalizer의 전후 hash도 같았다.

### 닫기 오차의 실제 물리적 영향

- 30000·30003·30007에서는 Expert/Gripper 대체 조건이 첫 close Action 후에 양측 파지 판정을
  받았고, 현재 모델/팔0 조건은 두 번째 Action 후에 받았다. **Gripper 대체가 판정 시점을
  1 control step=50ms 앞당겼지만 최종 유지 여부를 바꾸지는 않았다.**
- 30012에서는 네 조건 모두 두 번째 Action 후에 파지 판정을 받았다.
  이 Live 실행의 첫 모델 gripper는 -0.02243이었다. 07c의 저장 입력 -0.03474와 같은
  수치라고 가정하지 않는다. 각 조건의 Live prefix를 서로 대조하는 실험이다.
- 현재 모델의 close 중 평균 이동 명령 norm은 Seed 순서대로
  5.667 /4.560 /4.284 /4.490mm/control step이었다. 실제로 이 명령들을 환경에 적용했다.
  close 직전 대비 close8 이후 실제 TCP 위치 변화 norm은
  7.950 /1.558 /7.439 /7.246mm였고, 네 사례 모두 Peg를 유지했다.
- Arm delta0은 실제 TCP를 물리적으로 고정한다는 뜻이 아니다. Expert 대조군도
  동역학/접촉 아래에서 close 중 약0.529 /2.441 /1.384 /2.089mm의 TCP 위치 변화가 있었다.

**이번 네 정상 접근 상태에서는 약한 초기 닫기와 팔 움직임이 각각 또는 함께 존재해도
파지가 성립했다.** 이 결과는 두 모방 잔차가 41005 실패의 직접 원인이라는 가설을 지지하지 않는다.
그렇다고 잔차가 더 불리한 접근 자세나 다른 접촉 조건에서도 무해함을 입증한 것은 아니다.

현재 자료에서는 닫기 표현/구조 변경의 우선순위를 정하기보다,
자율 Rollout의 reach·descend에서 close 직전 위치/자세가 어떻게 어긋났는지 확인하는 편이 타당하다.
이 실험은 41005의 같은 seed Expert counterfactual을 실행한 것은 아니므로,
그 Episode의 실패 원인을 특정하거나 Vision 추적 기여를 배제하지 않는다.

## Artifact와 재현

본 실험의 결과 링크:

- [정적 HTML](../../experiments/d046e3a679a3430aab5e69f279c3f0a7/grasp_intervention.html)
- [전체 결과 JSON](../../experiments/d046e3a679a3430aab5e69f279c3f0a7/results.json)
- [프레임별 CSV](../../experiments/d046e3a679a3430aab5e69f279c3f0a7/intervention_frames.csv)
- [Artifact/실행 계약 검증](../../experiments/d046e3a679a3430aab5e69f279c3f0a7/verification.json)
- [로컬 독립 readback](../../experiments/d046e3a679a3430aab5e69f279c3f0a7/local_verification.json)
- [전체 자료 ZIP /121.46MB](../../outputs/graph/integration_baseline/grasp_intervention/live_grasp_intervention_d046e3a6.zip)

각 Trial은 원본 RGB-D/Calibration/Robot76, Predicted Mask, Packet/Graph,
모델 Action과 실제 Applied Action/Controller, Hidden, 접촉력/Peg pose와 20FPS 영상이 있다.
비교 영상과 Montage는 **Action 후 실제 RGB**를 사용하며, 원본 Trial 영상은 Action 전 RGB다.
수치·상대 링크·PNG/MP4는 생성 후 Decode한다.

로컬 관련 계약 테스트10개, 서버 신규 계약 테스트5개 PASS.
서버에서 NPZ3,144개·PNG24개·MP4 20개(총1,732프레임), HTML 상대 링크64개를 readback했다.
Graph 값/Source Index와 개입 Action 성분, 실제 Physics/Hidden Trace를 독립적으로 다시 검사했다.
원본 학습 파일·평가 source·앞선 Held-out 본평가 결과의 보호 hash를 전후 대조했다.
보호 대상52개 파일 모두 동일했다. 로컬 ZIP SHA256은
`4b8070d102ab4693f9773a46c297f7b02beddc5304f7a90e38e5b5bd82548775`이며,
CRC와 내부 파일 SHA를 검증한 뒤 로컬에서도 NPZ/PNG/MP4/상대 링크와 16회 실행 계약을 재검사했다.
그림과 실제 close/hold RGB Montage를 직접 확인했다. 실험 Process는 정상 종료했고
GPU 메모리 사용량은 종료 후1MiB로 돌아왔다. 다른 Process를 중지하지 않았다.

실제 서버 명령은 별도 Supervisor `task07-grasp-intervention-57640710`으로 실행한다.
CUDA를 먼저 초기화한 Bootstrap이 원래 평가 checkout을 import path에 추가한 뒤
독립 디렉터리의 script를 실행한다.

```bash
cd /workspace/task07-stop-clock
VK_ICD_FILENAMES=/workspace/graph-recurrent-VLA/visual/vulkan/nvidia_icd_egl.json \
HF_HOME=/workspace/graph-recurrent-VLA/visual/cache/hf \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
MPLCONFIGDIR=/tmp/task07-mpl WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/workspace/task07-stop-clock-work/eval-venv/bin/python \
  /workspace/task07-grasp-intervention/bootstrap.py \
  --config /workspace/task07-grasp-intervention/graph/integration_baseline/grasp_intervention_config.json \
  --checkpoint-manifest /workspace/task07-stop-clock-work/bundle/checkpoint_manifest.json \
  --output-root /workspace/task07-grasp-intervention/experiments
```

학습 소스·Checkpoint·Normalizer·Dataset·기존 본평가 source는 수정하지 않는다.
새 dependency 설치·학습·HPO·외부 게시·Git commit은 수행하지 않는다.
CUDA 모델/Vision과 CPU physics를 서버에서 실행하며, 이 진단의 macOS Simulator/MPS/CUDA physics는 미실행이다.
