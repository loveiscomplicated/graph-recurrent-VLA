# Task 10-F — Mid-trajectory Corrective Demonstration Pilot

**판정: PILOT_FAILED_EARLY_STOP. 본 수집·Graph Cache·A/B Fine-tuning·Closed-loop 평가는 실행하지 않았다.**

실제 Pilot UUID `706f62ce213247e7973102962858f95b`. 지정 Vast instance **57640710**, RTX 3090 24GB에서 실행했다. Task10-D의 기록된 SSH 주소는 다른 인스턴스57987772였으므로 그곳에서는 읽기 전용 확인만 수행했다. 지정 서버로 코드와 입력을 복사하여 독립 경로 `/workspace/task10f_corrective_reach`를 사용했다. 승인 없는 Git commit, 외부 publishing, 메시지 발송은 수행하지 않았다.

## 실행 범위와 중단 사유

Parent5 × Anchor2 × Offset4, 최대40개를 사전 계획했다. 실제 정상 Parent는 seed30000 한 개, 완료 Branch는 **6개**, 중단 중인 H5는 1개다. 중단 H5는 성공·실패율 분모에서 제외하고 파일 자체를 보존했다. 완료 Branch에서 실제 변위의 3D 잔차가1mm 미만인 것은 **2/6**다. 즉 요청한 ±5mm 교란을 안정적으로 만들지 못했다. 핵심 물리 교란 gate 실패로 상위 단계를 진행하지 않았다. 이 결과는 BC의 학습 가능성에 관한 음성 결과가 아니다.

처음 네 분기의 실패 확인 후 운영상 중단했다. 중단 명령과 자료 회수 사이 완료된 두 분기도 모두 포함한다. 조건을 유리하게 골라 남기지 않았다. 연구 실행 소스는 첫 gate 실패 후 즉시 종료하도록 추가 보완했지만 이 보완은 이미 실행된 원자료를 변경하지 않는다.

## 실제 환경과 소스 재사용

Torch `2.8.0+cu128`, wheel CUDA `12.8`, CPU quota `192000 100000`. CPU PhysX, GPU SAPIEN renderer,20Hz control/100Hz physics를 사용했다. Task10-D 실행 서버의 Torch2.14.1+cu130/13.17core와 다르다. CUDA Physics, DINO/SAM2, GNN/GRU 학습·추론은 이번 실행에서 NOT_TESTED다. GPU renderer가 실행됐다는 사실이 다른 호스트와 RGB-D byte 동등성을 의미하지 않는다.

`geometry_reach10d.runtime.environment`, `collect.target`, `collect_recurrent_expert.capture/append_h5`, `geometry09a.physical_state`, `ScriptedExpert`, `ExperimentLogger`, `profile.Resources`를 재사용했다. 기존 수집·캐시·학습 코드를 재작성하거나 모델 구조·Loss·Normalizer·Controller를 변경하지 않았다. 신규 코드는 task-local snapshot, disturbance, pilot gate와 사후 진단뿐이다. 로컬 inherited11개 파일 SHA 보존을 확인했다. 기존 원격 Checkpoint/Cache에는 쓰지 않았으며 전체 원격 artifact의 사후 재해시는 수행하지 않았다.

후속 공통 초기 후보는 Task10-D B step6000, SHA256 `19da92de69b0d9aa66aa210ea759451d0f2b98e2d01c41c587ba986fd60b2cbc`다. 이번 Pilot은 정책 가중치를 로드하지 않았고 새 호스트에서 Checkpoint readback/strict-load를 검증하지 않았다. 따라서 A/B 초기화 동등성은 NOT_TESTED이며 training protocol·final seeds는 아직 동결하지 않았다.

## Snapshot·시간·이력

seed30000의 t10과 t25는 모두 `reach`였다. 두 anchor에서 Robot/Actor/목표·속도를 포함한15개 physical 필드는 복원 최대 오차0, 독립 복원 후5-step feedback replay의 physical field 및 command 최대 차이0이었다. 명시적으로 Controller target/interpolation fields, joint drive targets, Episode clock, Expert fields, Python/NumPy/Torch CPU RNG와 task RNG를 복원했다. 설치된 `set_state_dict()`는 Controller를 복원하지 않아 별도 복원이 필요했다.

이 검사는 저장한 필드와 관찰한5-step 동작에 한정된다. PhysX contact solver 내부 warm-start와 모든 Controller 내부 필드의 완전 serialization 증거는 아니다. Snapshot은 실행 중 메모리에만 있었고 전체 binary snapshot 파일은 저장하지 않았다.

모든 완료 H5에서 T+1 Observation/T Action, Robot76, 양 카메라 RGB-D, t=0..T 및 timestamp=t/20을 readback했다. 동일 anchor의 Prefix는 Robot/Action/RGB-D byte가 같다. Prefix와8개 disturbance frame의 supervision은 False, 이후 Recovery는 True다. GRU hidden은 저장·주입하지 않았다. 실제 Graph 캐시 및 mask-aware loader/GRU 실행 검증은 수행하지 않았다. 중단 raw는 승인 Dataset이 아니다.

## 실제 교란과 Recovery

TCP나 joint를 teleport하지 않았다. anchor의 TCP world pose에서 X/Y ±5mm 목표를 만들고, 원래 `_servo`의 최대 translation을2mm로 제한해 **8회 실제 env.step(0.4초)**을 실행했다. 교란 동안 목표 Orientation과 gripper open을 유지하고 관측을 전부 저장했다.

교란 후 새 ScriptedExpert가 현재 Peg OBB/TCP에서 목표를 다시 계산하도록 했다. 원래 목표와 translation2µm/orientation10µrad 이내 일치를 assertion으로 검사했다. 이후 phase를 `reach`로 명시적으로 재설정하여 elevated approach를 재획득하고 원래 feedback/threshold로 descend를 수행했다. 이는 counter만 이어가는 방식이나 미래 Action 복사가 아니다. Task10-B의 단일 counterfactual reference를 연속 Teacher로 간주하지 않았다.

| Branch | 요청 mm | 실제 mm | 3D 잔차 mm | 최종 위치 mm | 최종 자세 rad | Recovery steps |
|---|---|---|---:|---:|---:|---:|
| 30000_t10_5_0 | [5, 0, 0] | [3.129, -2.075, -0.363] | 2.817 | 2.000 | 0.00310 | 40 |
| 30000_t10_-5_0 | [-5, 0, 0] | [-4.446, -1.217, -0.209] | 1.353 | 1.919 | 0.00294 | 40 |
| 30000_t10_0_5 | [0, 5, 0] | [-0.629, 0.062, -0.512] | 5.004 | 2.105 | 0.00331 | 41 |
| 30000_t10_0_-5 | [0, -5, 0] | [-0.024, -4.993, -0.039] | 0.046 | 2.156 | 0.00341 | 40 |
| 30000_t25_5_0 | [5, 0, 0] | [2.508, -2.184, -0.38] | 3.335 | 2.009 | 0.00312 | 25 |
| 30000_t25_-5_0 | [-5, 0, 0] | [-4.822, -0.625, -0.101] | 0.658 | 1.886 | 0.00288 | 25 |

완료6개 모두 최종 pose-only4mm/0.03rad 이내로 돌아왔다. 그러나 strict success는 **0/6**이다. 모든 Branch에서 broad safety predicate가 실패했다. 이 probe의 z>20mm 하한은 최종 TCP가 약19.7mm로 낮아지는 정상 descend와 충돌할 수 있으므로 이것만으로 실제 충돌이라고 주장하지 않는다. 기록된 robot/external contact impulse event는 0개였지만, 검사 이름 필터·control boundary 관찰에는 제한이 있다. Substep collision freedom/전체 self-collision은 NOT_TESTED다. Gripper opening≥65mm, Peg/Box/Hole translation<2mm는 완료 raw에서 확인했다. Object orientation 보존의 독립 정량 감사는 NOT_TESTED다.

Recovery 최초 Action 및3/5/10-step translation/rotation command sum·error 감소량을 CSV로 보존했다. 처음 Reach는 목표까지 멀어20mm translation command saturation이 대부분을 차지한다. 원본 nominal suffix Action을 별도 저장하지 않아 **nominal 대비 정확한 Action 변화·교란 상쇄 방향/이득의 전체 비교는 NOT_TESTED**다. Pose 회복만으로 유효한 새 Corrective supervision이 충분하다고 주장하지 않는다.

![Recovery error curves](../../outputs/task10f/analysis/recovery_error_curves.png)

## Contract 결과

PASS: 기록된 Parent reset/physical restore/5-step replay, 완료 raw SHA256/readback, Prefix identity, timestep/Action alignment, supervision mask 값, gripper opening, object translation.

FAIL: 요청 교란 변위 정확도(2/6만 통과), broad safety predicate, strict Recovery success. 안전 판정의 보수적 높이 기준과 실제 충돌을 구분해야 한다.

NOT_TESTED: 전체 Controller/PhysX 내부 serialization, 새로운 관측의 causal DINO/SAM2·Graph equivalence, Node21/Edge4 cache, GRU carry/detach/reset, 실제 masked loss, 동일 초기 weight/Normalizer/노출량·update·save-resume, GPU training, handoff/autonomous/nominal rollout 및 Final 평가.

기존 infrastructure test42개는 로컬 실제 저장소 환경에서 PASS였다. 최초 원격 사본에서 실행한42개는 fixture/context와 의존성 문제로4 failures/4 errors/1 skip였다: Git context 없음, scripts/ 누락, python-dotenv 미설치다. 이 결과를 숨기거나 원격 PASS로 바꾸지 않았다. 공통 logger/publisher 코드 자체는 수정하지 않았다. 실제 Pilot Logger는 disabled W&B/local mode이며 외부 Publishing은 없었다.

## 계산 비용과 보존

실행 Pilot의 resource sampling window는 110.0초,106 samples다. 완료 raw349개 Action에는 중복 Prefix90개, disturbance48개, recovery211개가 포함된다. Supervisor startup·이전 실패·자료 회수는 별도이며 이 window를 전체 Task wall-clock으로 주장하지 않는다. 평균 GPU util0.45%, peak VRAM555MiB, 부모 CPU100.35%였다. 이번 병목 관찰은 CPU physics/capture 진단에 한정하며 CUDA Vision/Training 처리량이나 기존 최적화 speedup을 검증하지 않았다. [측정 JSON](../../outputs/task10f/analysis/throughput_summary.json).

6개 완료 H5 SHA256은 생성 receipt와 일치했고 전체 Dataset을 readback했다.6개 MP4도 frame count를 readback했다. PNG는 육안 확인했고 보고서 상대 링크10개가 존재함을 확인했다(계산 비용 링크 추가 전). 실행된 Pilot 소스는 `outputs/task10f/evidence/executed_pilot.py`에 별도로 보존하여 조기 gate 종료를 추가한 현재 소스와 구분한다. 원격 Pilot은 STOPPED이며 종료 후 GPU memory1MiB/util0%를 확인했다. 기존 서비스는 중지하지 않았다. 원격 Logger는 SIGTERM 중단으로 완료 status를 기록하지 못했으며, 사후 분석은 별도 ExperimentLogger UUID `f7589cee98f44d75890e1dbcc0409902`로 source_run_id를 연결했다.

## Q1–Q10 연구 판정

| 질문 | 판정 | 근거 |
|---|---|---|
| Q1 안전·재현 교란 | INCONCLUSIVE | snapshot/replay는 정확하나 변위·안전 gate 실패 |
| Q2 실제 Expert 보정·복귀 | SUPPORTED | 실제 state feedback으로6/6 pose-only 복귀; strict safety 미충족 |
| Q3 새 상태–Action 관계 | INCONCLUSIVE | 실제 disturbed 상태/새 feedback Action 존재; nominal 대비 정량 비교 미완료 |
| Q4 Causal Vision/Graph/GRU | NOT_TESTED | Raw temporal contract만 CONFIRMED |
| Q5 Action correction 개선 | NOT_TESTED | Student fine-tuning 없음 |
| Q6 Expert handoff 개선 | NOT_TESTED | 정책 평가 없음 |
| Q7 Autonomous 개선 | NOT_TESTED | 정책 평가 없음 |
| Q8 Nominal 손상 | NOT_TESTED | 정책 평가 없음 |
| Q9 실패 원인 후보 축소 | CONFIRMED | 현재 stop 원인은 데이터 생성 gate이며 모델 fit/Geometry/GRU 원인 판단 불가 |
| Q10 Full Training 정당화 | INCONCLUSIVE | 현 protocol로 확대 불가; 생성 gate 수정 Pilot이 먼저 필요 |

## 가능한 최소 수정 범위

정책 결과를 보지 않은 상태에서, 교란8-step 고정 대신 실제 world displacement·velocity 수렴을 확인하는 bounded feedback disturbance를 별도 protocol로 사전 고정하고 nominal hold control과 함께 검증할 수 있다. 양의 Y에서0.062mm밖에 이동하지 않은 원인은 command clipping·현재 관성·IK target/실제 joint tracking을 추가 계측해야 좁힐 수 있다. 원인이 IK 실패라고 단정하지 않는다.

안전 검사는 사전 고정20mm TCP 높이 대신 nominal reach와 로봇 collision mesh/테이블 clearance를 대조해 검증할 필요가 있다. 새 Target 교란·모델 구조·Loss 변경이나 TCP teleport로 우회할 근거는 없다. Nominal continuation과 matched-duration hold trace, serializable full snapshots와 controller command audit를 저장한 뒤 Pilot을 다시 통과해야 한다. 현재 자료는 Full Training·새 Recovery 대량 수집의 근거로 사용하지 않는다.

## 산출물

[검증 JSON](../../outputs/task10f/analysis/verification.json) · [Raw readback](../../outputs/task10f/analysis/raw_readback.json) · [보정 Action CSV](../../outputs/task10f/analysis/corrective_action_analysis.csv) · [Parent/Branch Manifest](../../outputs/task10f/evidence/attempt04/parent_branch_manifest.json) · [사전 Protocol](../../outputs/task10f/evidence/attempt04/perturbation_protocol.json) · [실행 로그](../../outputs/task10f/evidence/pilot.log) · [실행 오류](../../outputs/task10f/evidence/pilot.err.log) · [전체 원자료 archive](../../outputs/task10f/execution_evidence.tar.gz) · [정적 HTML](../../outputs/task10f/analysis/research_report.html).

각 완료 Branch의 Raw H5/Action trace/receipt와 실행 실패 시도02/03은 evidence/에 있다.6개 Expert Recovery MP4와 error curves는 analysis/에 있다. Cached Graph, checkpoint, A/B learning curves, R0/R1 영상·평가 CSV는 미실행이므로 생성하지 않았다.
