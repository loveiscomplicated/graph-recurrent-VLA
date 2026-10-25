# Task07 — Stop-the-clock Vision–Graph–GNN–GRU Closed-loop Rollout

## 1. 결과와 범위

**구현 및 CPU Simulator/Online Vision smoke 완료. 학습된 Policy 본평가는 `ROLLOUT_NOT_EXECUTED`.**

공식 평가명:

`Stop-the-clock / Zero-simulation-time-inference-delay / 20Hz simulation-control Closed-loop Evaluation`

- ExperimentLogger UUID: `0ab6be4b9b0d410c93593a33d6fdb2db`.
- 시작 HEAD: `c07b12e29470b70ddcbf5f64fd9d5ed6ee7054c0`.
- 실제 작업 브랜치: `feature/stop-the-clock`; 요청의 참고 브랜치 `visual-graph`와 구분한다.
- 시작 Working Tree는 clean. 최종 소스는 미커밋 상태이며 run의 source snapshot/hash/diff로 보존한다.
- [정적 HTML](../../experiments/0ab6be4b9b0d410c93593a33d6fdb2db/stop_the_clock_rollout.html),
  [결과 JSON](../../experiments/0ab6be4b9b0d410c93593a33d6fdb2db/rollout_results.json),
  [decode/link 검증](../../experiments/0ab6be4b9b0d410c93593a33d6fdb2db/verification.json).

RTX3090 학습에는 접속하지 않았다. 학습 완료 확인, 원격 상태 polling, tmux/GPU process 조작,
dependency 설치, 학습 데이터·체크포인트 수정, 재학습/HPO, Git commit, 외부 업로드를 수행하지 않았다.
이번 실행은 Random Weight와 identity normalizer를 사용하는 기능 계약 smoke다. 그 결과로
학습된 모델의 성공률이나 Confidence Interval을 산출하지 않는다.

## 2. 실행 구조

```text
현재 env.reset / env.step의 base_camera RGB-D + 같은 시점 robot kinematics
  → 기존 DINO Tiny / SAM2.1 Small의 current-prefix causal mask
  → 기존 Shape-First / PCA5 / F5 / EE transform / Raw EDE
  → Robot3 + Peg/Box variable-N graph, k4 / global M16
  → 기존 Edge-aware GNN L2,d64 / Structured Readout320
  → Presence2 + Full Robot76 / 저장된 train-only normalizer
  → 기존 GRU hidden144 / Action7
  → 실제 pd_ee_delta_pose controller
  → 다음 환경 상태와 새로운 관측
```

`rollout_stop_clock.py`는 single environment에서 관측 → Vision/geometry/graph →
`EpisodeStream.step()` → `env.step()`을 순서대로 실행한다. Expert나 Rescue Action을 호출하지 않는다.
Policy 입력은 명시적으로 선택한 카메라 데이터와 robot-only state다. `obs['extra']`는 복사하지 않는다.
GT accessor는 별도 reset identity 진단 함수에서만 사용하며 반환 hash는 입력 row/graph에 넣지 않는다.

기존 모델·graph·robot·Vision helper·학습 코드는 수정하지 않았다. 신규 평가 모듈,
테스트, config/template과 integration README만 추가·수정했다.

## 3. 실제 Simulation Time 계약

실행 환경: macOS, Python3.11.13, Torch2.14.1, ManiSkill3.0.1, SAPIEN3.0.3,
Gymnasium1.3.0, NumPy2.4.6. `physx_cpu` / `sapien_cpu`, `panda_wristcam`,
`pd_ee_delta_pose`, base_camera128×128 RGB-D를 사용했다. CUDA/MPS는 실행하지 않았다.

| 검사 | 실제 결과 |
| --- | --- |
| Control / Physics Frequency | 20Hz / 100Hz |
| 실제 PhysX timestep | 0.009999999776482582s, float32의 10ms 표현 |
| Action당 physics calls | 정확히5회 |
| 제어 간격 | 50ms simulation time |
| 추론 중 물리 호출 / elapsed step 변화 | 0회 / 0 |
| 명시적인70ms wall-delay probe | 시뮬레이션 시간 불변 |
| 별도 Horizon 검사 | 250 actions /1250 actual scene.step /12.5s simulation |
| TimeLimit | step250에서 truncated; 전역100-step 기본값 변경 없음 |
| Episode reset | 이후 다른 seed의 live episode를 t0부터 실행 |

`PhysicsAudit`는 실제 `scene.step()`을 감싸고 control elapsed step도 함께 검사한다.
추론 전후에는 같은 robot state인지 추가 검사한다. 동기식 `env.step()`만으로 시간 계약을
만족하므로 별도 pause/resume은 추가하지 않았다. Physics timestep의 부동소수점 표현 오차를
50ms simulation-control 계약과 구분한다.

## 4. Online Vision과 검증 구분

`OnlineVisionAdapter.process(rgb, episode_identity, timestep)`은 현재 RGB만 받는다.
Episode 전체 배열이나 미래 길이를 받는 API가 없다. 새 RGB를 JPEG로 한 번 등록하고,
설치된 SAM2 `_load_img_as_tensor`와 기존 mean/std 전처리를 그대로 사용한다.

SAM2의 최초 디렉터리 로딩 결과를 파일 추가만으로 갱신된다고 가정하지 않는다.
`state['images']`를 bounds-checked append-only 입력 컨테이너로 교체한다.
각 state의 `num_frames=t+1`을 유지하며 기존 feature cache, conditioning/non-conditioning
output, temporal memory, tracker ID와 forward propagation 구현은 유지한다.
알 수 없는 SAM2 state schema는 `BLOCKED_ONLINE_VISION`으로 중단한다.

초기화는 기존 `detect_initial()`을 현재 prefix에 호출한다. 최초 scan과5-frame 재탐색 간격,
Prompt/Threshold/Assignment와 두 scan 일치 규칙을 변경하지 않았다. 늦게 발견된 객체는
offline baseline과 동일하게 별도 객체 그룹 state로 시작하며 기존 그룹을 계속 유지한다.
재탐색 중 과거 RGB를 다시 검사할 수 있으나 미래 프레임은 존재하지 않는다.
Backward refinement, GT segmentation/pose 보충, speed revision 자동 활성화는 없다.

| 검증 종류 | 입력과 의미 | 결과 |
| --- | --- | --- |
| Live Simulator | 각 reset/step이 새로 제공한 RGB,2 seeds×3 control steps | PASS |
| Recorded real sequence | 위 smoke의 실제 RGB4프레임을 한 장씩 공개 | offline causal과 픽셀 차이0, Presence 차이0 |
| Prefix/Suffix causality | 독립 stream의 마지막 RGB만 검은 영상으로 변경 | 이전3프레임 픽셀 차이0 |
| Real SAM2 late-object fixture | 실제 RGB를 반복한7프레임, 과거 t0 DINO box로 box prompt만 t5에 추가 | 기존 객체 memory 유지, 객체 그룹 일치, offline 픽셀/Presence 차이0 |
| Synthetic contracts | 미래/음수 index 거부, duplicate/gap/identity/shape 오류, Empty, reset, 늦은 객체 | PASS |

Late-object 검사는 **실제 SAM2 + 제어된 prompt schedule fixture**다. 자연 발생한 지연 Detection이나
7-step live trajectory로 보고하지 않는다. 모든 비교 배열과 수치 차이는
[recorded comparison](../../experiments/0ab6be4b9b0d410c93593a33d6fdb2db/recorded_vision_comparison.json)과
그 NPZ, `late_object_sam2_fixture.npz`에 보존했다.

SAM2 optional `_C` postprocessing extension 부재 경고가 실제 발생했다. 설치·알고리즘 변경으로
숨기지 않았고 같은 호스트/동일 경고 조건에서 online/offline을 비교했다. CUDA 호스트에서의
동일성·숫자 차이는 별도 검증 대상이다. CPU의 짧은 sequence 검증이250-frame Vision 안정성을
입증한 것은 아니다.

## 5. Graph, GRU, Action 계약

현재 RGB-D의 calibration/TCP와 같은 시점의 robot fingertip geometry를 사용해 기존
`packet_from_frame`, `geometry_in_ee`, `build_graph`, `pack_graphs`를 호출한다.
기존 Source Row/Index, Raw EDE 방향/order, F5와 availability/validity view, Empty Object,
k4/M16, structured5d readout을 유지한다. source-row나 GT geometry로 point를 보충하지 않는다.

`EpisodeStream`은 첫 hidden을 None→zero 초기화하고 episode 내부의 매 frame에 딱 한 번 업데이트한다.
Empty Object에서도 hidden을 유지한다. 중복·불연속 timestep을 거부한다. Episode 시작/종료에서
reset event를 기록한다. 각 episode의 `hidden.npz`는 post-observation/pre-action hidden,
episode identity, consumed timestep을 저장한다. `restore_stream_hidden()`의 다음 step 출력
동일성도 테스트했다. 이는 GRU 복원 기능이며 simulator/tracker 전체 resume 기능은 아니다.

Action7은 FP32/finite/shape를 검사하고 그대로 simulator device의 controller에 보낸다.
Raw 출력, env.step 입력, 실제 controller entry를 비교한다. 실제 설치된 전처리 결과도 저장하고
translation clipping/scaling, rotation norm clipping과 `rot_lower=-0.1`, gripper -0.01..0.04m를
검사했다. 저장·전달 전에 별도 수동 clipping이나 좌표계 변환을 추가하지 않았다.

## 6. 실제 Closed-loop smoke 결과

| Smoke seed | Live actions | 실제 physics calls | Hidden updates | Peg/Box mean presence | Mask missing | Robot state 변화 |
| --- | --- | --- | --- | --- | --- | --- |
| 39000 | 3 | 15 | 3 | 1.0 /1.0 | 0 /0 | 확인 |
| 39001 | 3 | 15 | 3 | 1.0 /1.0 | 0 /0 | 확인 |

두 episode 모두3-step smoke limit으로 끝났다. 성공 여부의 원시 bool은 남기지만 성능 집계에서 제외한다.

| CPU wall-clock 평균 | Seed39000 | Seed39001 |
| --- | --- | --- |
| Vision mask | 2145.23ms | 2085.12ms |
| Geometry/PCA5/Shape-First/F5/Raw EDE | 13.39ms | 12.32ms |
| Graph construction | 5.22ms | 4.94ms |
| GNN + readout | 0.749ms | 0.724ms |
| GRU | 0.078ms | 0.064ms |
| 전체 기존 모델 inference | 0.953ms | 0.906ms |
| Episode simulation time | 0.15s | 0.15s |
| Episode wall time | 6.60s | 6.40s |
| Wall throughput | 0.454 control steps/s | 0.469 control steps/s |

짧은 CPU smoke 평균이며 latency 최적화나 benchmark 결과가 아니다. GNN/GRU는 기존 module의
일시적인 timing hook으로 측정한다. Wall throughput에는 episode IO/진단 비용이 포함된다.
20FPS MP4는 simulation control 순서를 표현한다. **50ms 이하 wall inference나 실시간 제어를
달성했다고 보고하지 않는다.**

- [39000 영상](../../experiments/0ab6be4b9b0d410c93593a33d6fdb2db/episodes/smoke_39000/rollout.mp4),
  [39001 영상](../../experiments/0ab6be4b9b0d410c93593a33d6fdb2db/episodes/smoke_39001/rollout.mp4).
- 각 episode에 원본 RGB/raw int16-mm depth/metre view/calibration/Robot76/TCP,
  predicted masks, packet/graph, native/scaled action, hidden, latency/flags/initialization trace가 있다.
- PNG/MP4에는 현재 RGB와 predicted mask, EE x/y graph inset, timestep, gripper, success, presence를 표시한다.
  마지막 terminal RGB-D도 별도 저장하며 불필요한 terminal-frame GRU update는 하지 않는다.

## 7. Checkpoint 이관과 본평가 준비

실제 RTX3090 Best Checkpoint는 수령하지 않았다. `direct_train.save_checkpoint()`로 만든
**실제 direct_bc_v1 형식의 생성 fixture**를 로드해 weight/normalizer 동일성과 거부 계약을 검증했다.
Optimizer나 training RNG를 실행·복원하지 않는다.

검증 항목:

- Best pointer + immutable checkpoint receipt + SHA256 + epoch/global-step 일치.
- 원본 Run UUID, config/model/graph/robot/action schema, dataset/split/Vision hash.
- 정확한 train episode ID의 normalizer mean/std와0.001 std floor, FP32/finite/shape.
- 모델의 모든 key/shape/dtype와 strict load. 누락/손상/partial checkpoint 거부.
- `weights_only=True`와 고정된 NumPy RNG 타입 allowlist. 임의 Python payload 실행 거부.
- 원본 정책 source inventory, DINO weight/revision, SAM2 checkpoint/source identity.
- collection의 simulator version/robot/controller/frequency/horizon과 실제 환경 일치.

필수 bundle 파일과 전체 CLI는 [integration README](../../graph/integration_baseline/README.md#task07-opt-in-stop-the-clock-closed-loop-evaluation)에 있다.
완료된 원본 환경에서의 Vision runtime inventory는 독립 transfer bundle에만 저장한다.
나중에 생성한 inventory가 과거 학습 provenance를 자동 증명한다고 간주하지 않는다.
실제 원본 run의 유효 Source/Weight Revision 근거와 함께 제공해야 한다.

```bash
# Checkpoint가 없어도 실행 가능한 로컬 계약 smoke
MPLCONFIGDIR=/tmp/task07-mpl .venv/bin/python -m graph.integration_baseline.rollout_stop_clock \
  --mode smoke --config graph/integration_baseline/rollout_config.json

# 완전한 bundle을 이미 받은 후, original UUID와 Best receipt에 manifest를 연결
.venv/bin/python -m graph.integration_baseline.rollout_checkpoint \
  --bundle /absolute/transfer/bundle --source-run-id ORIGINAL_TRAINING_UUID \
  --output /absolute/transfer/bundle/checkpoint_manifest.json

# 실제 frozen-test12와 held-out40 집합을 실행 전에 고정
.venv/bin/python -m graph.integration_baseline.rollout_stop_clock \
  --mode prepare-seeds --config graph/integration_baseline/rollout_config.json \
  --checkpoint-manifest /absolute/transfer/bundle/checkpoint_manifest.json \
  --seeds-manifest /absolute/transfer/bundle/seeds_manifest.json

# 명시적으로만 실행하는 본평가
.venv/bin/python -m graph.integration_baseline.rollout_stop_clock \
  --mode evaluate --config graph/integration_baseline/rollout_config.json \
  --checkpoint-manifest /absolute/transfer/bundle/checkpoint_manifest.json \
  --seeds-manifest /absolute/transfer/bundle/seeds_manifest.json
```

현재 seed template은 의도적으로 본평가용 manifest가 아니다. Smoke39000/39001,
timing39002와 held-out 후보41000..41039를 구분했다. Test12는 원본 frozen split에서만 해석한다.
원본 dataset/collection/pilot/smoke seed와 held-out 중복을 거부하고, 실제 reset group도
검사한다. Frozen test와 held-out 결과를 구분해 Wilson95% CI를 계산한다.
모든 계획 seed에 status를 남기며 runtime error/미실행이 있는 집합은 완성된 성공률을 보고하지 않는다.
First success step, episode length, termination/timeout, presence/missing, DINO 초기화·재탐색,
native/scaled action, hidden reset, wall latency를 저장한다. 미계측 collision/task-failure는 null이다.

## 8. 검증과 한계

| 검증 | 결과 |
| --- | --- |
| 최종 smoke가 실행한 Task07 + 기존 baseline 계약 | 39 tests PASS |
| Integration 전체 discovery | 81 tests PASS |
| 공통 ExperimentLogger/Artifact | 42 중41 PASS, 기존1 FAIL |
| 실제 CPU250-step 시간 계약과2×3 live Vision closed-loop | PASS |
| RGB4프레임 online/offline, changed suffix, real SAM2 late-object fixture | PASS |
| JSON/NPZ/PNG/MP4 decode 및 HTML 상대 링크 | PASS, verification.json 참고 |
| 실제 학습 checkpoint, CUDA/MPS, 전체52-episode 정책 평가 | NOT_EXECUTED |

공통 테스트 실패는 `tests.test_experiment_artifacts.ArtifactTests.test_reject_running_run_symlink_and_outside_organization`
line426의 외부 organization repository에 대한 `ValueError not raised`다. 해당 테스트와 공통 모듈은
시작 snapshot 대비 바이트가 같고 단독 재실행에서도 재현했다. 이번 평가에서는 업로드를 호출하지
않으며 이 기존 문제의 수정을 Task07에 포함하지 않았다. 전체 테스트를 PASS로 보고하지 않는다.

최초 sandbox 실행은 simulator 초기화에서 SIGABRT(exit134)로 끝났다. 해당 시도의 UUID
`efa094bcb33944af8b7ef26d2d773f72`와 로그/소스는 보존했다. 이후 권한이 열린 환경에서 실제 CPU
smoke가 성공했다. 중간 검증 run `4651d1fa7e4b47dca6bc4ab9bcf38560`도 보존하며 공식 최종
증거는 이 문서 상단의 `0ab6…` run이다.

초기 보호 inventory는 기존 tracked 파일869개와 기존 실험 artifact368개를 hash했다.
기존 학습·실험 artifact의 변경은0개다. 기존 tracked 파일에서는 integration README만 변경했다.
RTX3090의 원격 process 상태 자체는 조회하지 않았으므로 원격 학습이 정상 진행 중이라는
실시간 상태 주장은 하지 않는다.

## 9. 최종 질문에 대한 답

| 질문 | 답 |
| --- | --- |
| 매 Control Step에 실제 Online Vision을 처리할 수 있는가? | 예. CPU에서 새 simulator RGB-D로6회 확인.250-frame Vision 안정성은 미실행. |
| 추론 중 simulation time 불변을 검증했는가? | 예. 실제 physics 호출0회, elapsed 불변,70ms 지연 probe와250-step 시간 계약 통과. |
| Episode 안의 GRU Hidden 유지가 정상인가? | 예. 실제 episode별3회 carry, 시작/종료 reset; sequence/step·empty·reset·복원 계약 통과. |
| Policy Action7이 환경을 변화시키는가? | 예. 실제 controller entry/scale 검증 및 양 episode robot state 변화 확인. |
| RTX3090 Best/Normalizer를 그대로 사용할 수 있는가? | direct_bc_v1 형식은 변환 없이 지원. 실제 파일·원본 source/config/schema·Vision identity 호환성은 수령 전 미검증. |
| 실제 checkpoint 수령 후 추가 구현이 필요한가? | 계약이 일치하는 완전한 bundle이면 추가 구현 없이 manifest 고정 후 CLI 실행 가능. 누락·상이한 source/runtime는 자동 우회하지 않고 차단한다. |
| 무엇이 미실행인가? | 학습된 policy 성공률/CI, frozen-test12·held-out40, 실제 RTX3090 checkpoint strict load, CUDA/MPS 및 real-time 성능. |

본평가를 자동 시작하거나 학습 완료를 기다리지 않는다. 이번 작업은 위 구현·최초 검증·산출물 보고로 종료한다.
