# Task10-F1 — GT-supervised Graph Encoder Pretraining

## 현재 판정과 실제 실행 범위

**PREPARATION_VERIFIED / WAITING_VALIDATED_TASK10F_DATA. 본 GNN 사전학습과 독립 성능 평가는 아직 수행하지 않았다.**

비교·준비 ExperimentLogger UUID는 `dcbafd9fd31847a3a83fef6830f87834`다. RTX3090 실행 위치는 지정된 **Vast.ai57987772**, 독립 root는 `/workspace/task10f1_gt_pretrain_20261030`이다. Task10-F의57640710은 읽기 전용 조사만 했으며, 그 수집·학습 프로세스나 Source를 변경하지 않았다. 새 인스턴스 생성, GRU/Action 학습, Git commit, 외부 publishing은 수행하지 않았다.

2026-10-30 추가 steering을 반영했다. 기존 Full-task 자료·완료 캐시·검증을 보존하면서 **Task10-F의 승인된 GT-complete 데이터를 우선 사용한다.** 초기120-episode 계획과12,000-update 설정은 원래 기록으로 보존한다. 실제 데이터 지원 범위·runtime 정합성을 확인한 뒤 별도의 final release에서 데이터 구성·학습 예산을 동결한다. Full-task 자료만으로 본 학습을 임의로 시작하지 않는다. 데이터 대기는 프로젝트 실패로 처리하지 않는다.

이 문서는 현재 준비·감시·인계 보고서다. 사전학습 모델이나 공간 복원 성능이 확보됐다는 최종 연구 보고서가 아니다. 향후 DATA_READY callback은 실제 학습·독립 평가 후 이 문서와 산출물을 갱신해야 한다.

## 1. 원본 조사와 GT38D 계약

Task09-D, Plan A BC,10-B,10-C1/C2,10-D,10-E 보고서와 현재 branch의 실제 API를 조사했다. Task10-E에 복원된 원본 `recurrent.features.build_features`, Clean corruption, normalizer, 학습·추론 코드 및 Clean seed0/1/2의38D checkpoint를 확인했다. 세 checkpoint는 **계약·normalizer 확인용**이며 GRU나 BC 가중치를 GNN 초기화에 사용하지 않는다. 원본 source backup이 역사적 dirty 학습 실행과 완전히 같다는 보증은 없다는10-E의 제한도 유지한다.

원본 encoder SHA256은 `80de6fd742be4e7c9a3477cde0419e803b72253c09d1227fc0fec23b965b0761`다. 전달받은38D 표와 실제 원본 순서는 일치했다.

| Index | Feature | 추론 Source / 역할 |
|---|---|---|
|0–2|TCP world position|A: Robot76 measured TCP|
|3–6|TCP world quaternion|A: measured wxyz, canonical w≥0|
|7–9|Left fingertip − TCP|A: measured finger link + 검증된 pad offset|
|10–12|Right fingertip − TCP|A: 같은 measured FK|
|13–14|두 finger joint position|A: Robot76 gripper position, metres|
|15–17|Peg position − TCP|B: GNN estimate, world-aligned metres|
|18–21|Peg world quaternion|C: GNN conditional estimate; 대칭성 한계|
|22–24|Peg head − Hole|예측 position/quaternion/half-length로 계산|
|25–27|Hole position − TCP|B: GNN estimate; Box centroid와 구분|
|28–31|Hole world quaternion|C: GNN conditional estimate; 가시성·대칭성 한계|
|32–34|Peg local half extents|C: GNN estimate, episode-randomized shape|
|35|Hole radius|C: GNN estimate; square-hole half opening|
|36|Observation freshness|Clean 계약1|
|37|min(age,20)/20|Clean 계약0|

위 상대 위치는 **world 위치의 차**다. inverse TCP rotation을 적용한 좌표가 아니다. 반면 Graph XYZ는 TCP axes의 metres이므로 두 좌표를 혼동하지 않는다. Quaternion은 wxyz이고 원본 encoder는 부호를 canonicalize한다. 원본 unit quaternion을 전제로 하며 Head는 unit normalize와 q/−q 동치 loss를 사용한다.

Peg head는 `peg_p + R(peg_q)·[peg_half_x,0,0]`이다. Head-to-Hole3D를 별도 prediction target으로 중복 추가하지 않는다. 설치된 simulator에서 `hole_pose = box.pose * box_hole_offsets`를 확인했다. Hole offset·Peg half sizes는 scene 재구성 때 무작위이며, Hole radius는 Peg cross-section half-size에 clearance를 더한 값이다. 현재 Graph의 물체 중심을 GT로 대체하거나 episode GT 형상을 추론 상수로 넘기지 않는다.

로봇15D는 측정값, Object Head는 **18D(positions6 + quaternions8 + shape scalars4)**, 파생 head vector3D와 freshness2D를 더해38D를 조립한다. Shape scalar 간 환경 관계가 있다는 사실과18개 출력값의 순서는 구분한다.

**UNSUPPORTED:** RGB 색상을 포함하지 않는 현재 Graph에서 arbitrary Peg pose의 대칭 roll/end polarity를 항상 유일하게 구분하는 것, 물체가 완전히 사라졌을 때 그 현재 pose를 관측으로 식별하는 것, 보이지 않는 rim·extent의 확정적 복원이다. 현재 태스크 분포의 통계적 추정과 보편적 관측 가능성을 구분한다. 수치형 추정값을 내더라도 이 한계를 제거했다고 주장하지 않는다. 추출 API는 identifiability metadata를 제공하며 이를 Graph/38D 입력의 추가 feature로 넣지 않는다.

근거: [차원별 Feature Contract](../../outputs/task10f1/evidence/gt38_feature_contract.json), [Observability Audit](../../outputs/task10f1/evidence/target_observability_audit.json), [Source/원본 checkpoint identity](../../outputs/task10f1/evidence/source_identity.json).

## 2. 확보한 Full-task 보완 자료

기존 Train96 HDF5/Graph Cache는 필요한 per-timestep Object GT가 없어 그대로 사전학습에 재사용하지 않았다.09-D Train8/Validation4는 소규모 기능 검증 후보이며 다른 historical runtime과 혼합하지 않았다.10-B 평가 개입22개는 fitting에 사용하지 않는다.10-D 자료는 Reach 직전까지만 기록하므로 Full-task를 대신하지 않는다. Recurrent 자료는 원본 GT38 계약 확인에 사용했다.

추가 steering 이전에 새 Full-task 수집을 시작했다. 현재120개 원자료 수집은 완료됐고, 이 자료는 보완 cohort로 보존한다. 기존 parent의 Train96/Validation12 구조와 신규 Final91000–91011을 사전 등록했다. Split은 parent 단위이며 과거 Test/Final을 fitting에 사용하지 않는다. 서로 다른 timestep의 GT를 Graph centroid로 대체하지 않는다.

| Split | Episodes | Action-timestep observations | ScriptedExpert insertion |
|---|---:|---:|---:|
|Train|96|11,973|96/96|
|Validation|12|1,477|12/12|
|New Final|12|1,350|12/12|
|합계|120|14,800|120/120|

이는 GT를 사용하는 **수집 Expert의 결과**다. GNN이나 learned policy의 성공률이 아니다. 원자료에는 각 episode의T+1 terminal observation도 별도로 보존한다. 원본 RGB-D·Robot76과 같은 pre-action clock에서 GT sidecar를 기록하고, 원본 GT38 encoder와 sensor-only robot + oracle object assembly를 비교했다. 전체 최대 절대 차이는 **5.960464477539063e−08**, 사전 허용 오차1e−7 이내다. Final 라벨 접근은 정합성·물리 계약 확인용이며 모델 선택·학습에는 넣지 않았다.

| 실제 Expert phase | Action frames |
|---|---:|
|Reach|3,885|
|Descend|1,591|
|Close|960|
|Lift|1,354|
|Alignment|4,866|
|Insertion|2,144|

별도의 같은 Robot/Camera/causal prefix 조건에서 실제 Peg/Box를 이동하고 새 RGB-D를 저장했다. Train offset은±7.5mm axis, Validation은 별도6mm diagonal, Final은미학습±5/10mm axis다. Train12개 parent와 Validation4개 parent, Final12개 parent를 분리했다. 원래 SAT 기준을 통과하지 못한 개입은 보존하고 Δ loss에서 제외한다.

| Split | Paired candidates | 물리 유효 |
|---|---:|---:|
|Train|192|192|
|Validation|96|80|
|Final|576|476|

2026-10-30 **01:54:18 UTC snapshot**에서 완료 Graph Cache는43 episode/5,288 frames, paired cache는4 episode/64 graphs다. 나머지 캐싱은57987772의 `task10f1_data` Supervisor에서 계속된다. 이 수치는 snapshot이며 실시간 완료 개수와 다를 수 있다. 완료 파일만 receipt SHA/readback으로 재사용·회수한다. GPU physics는 사용하지 않았고 CPU PhysX + GPU renderer/vision을 사용했다.

근거: [Dataset manifest](../../outputs/task10f1/evidence/dataset_manifest.json), [Split](../../outputs/task10f1/evidence/split_manifest.json), [Paired protocol](../../outputs/task10f1/evidence/paired_intervention_manifest.json), [실제 준비 snapshot](../../outputs/task10f1/evidence/preparation_snapshot_steering_v1.json), [기존 자료 감사](../../outputs/task10f1/evidence/existing_data_audit.json).

## 3. Task10-F 우선 재사용 감사와 현재 대기 사유

57640710 `/workspace/task10f_recovery_v2/pilot`의 완료 receipt·승인 HDF5·manifest를 읽기 전용으로 조사했다. 당시 완료 P branch는20개/계획40개, P/N receipt는40개였으며 모두 branch receipt에는 승인으로 기록됐다. Graph Cache는 engineering2개뿐이었다. 실행 중 파일을 단순히 열어 READY로 판단하지 않았다.

RGB-D/보정/Robot76/Parent·Branch/Timestep/Segment/Action Mask는 확인했다. 그러나 다음 필드는 매 timestep 저장되어 있지 않았다.

- Peg world pose/quaternion7D
- Hole world pose/quaternion7D
- Peg half sizes3D와 Hole radius1D
- 그 상태를 정확히 복원할 직렬화 simulator/controller/RNG snapshot

물체 이동 norm이나 “거의 정지”라는 사실로 pose를 완성할 수 없다. In-memory replay 검사 통과와 디스크 snapshot 보존은 다르다. Seed-only 근사 replay나 Graph centroid로 GT를 만들지 않는다. 기존 Safety Gate 실패 Pilot도 자동 승인하지 않는다. Action Mask OFF인 disturbance frame은 **별도의 GNN physical/GT mask가 승인할 때만** 사용할 수 있다.

추가로 실제 tensor runtime은 F1 `Torch2.14.1+cu130`, F `Torch2.8.0+cu128`로 다르다. ManiSkill3.0.1·SAPIEN3.0.3·Gymnasium1.4.0 일치만으로 자동 혼합하지 않는다. `release.py`는 simulator/camera/robot/geometry/vision/source/tensor runtime 계약을 비교하며 불일치하면 `WAITING_RUNTIME_COMPATIBILITY`를 기록하고 학습을 열지 않는다. 현재 다른 인스턴스의 환경을 수정하지 않았다.

따라서 현재는 **WAITING_DATA**다. Task10-F 전체 BC 실험 완료가 아니라, 승인한 수집 범위의 GT·Graph·물리 판정·정합성·sealed manifest가 준비되는 시점이 재개 기준이다.

[읽기 전용 감사 원문](../../outputs/task10f1/task10f_dataset_audit.json), [Task10-F 담당 작업용 최소 GT Sidecar/READY 계약](../../docs/visual-graph-integration/10f1_task10f_gt_sidecar_handoff.md), [F1이 요구하는 실제 runtime 계약](../../outputs/task10f1/evidence/expected_task10f_runtime_contract.json).

## 4. 구현·실행 검증

새 모듈은 [graph/gt_pretrain10f1](../../graph/gt_pretrain10f1/README.md)에 한정한다. 기존 Plan A E0 GraphEncoder L2/d64를 직접 재사용하고 Node encoder만 기존 Plan A와 같은21+type4 입력으로 설정한다. TCP/Left/Right/Peg/Box5×64 Readout320은 그대로다. Head는 Readout320 + Presence2 + train-standardized measured Robot76을 받는398→256→128→18 MLP다. GNN+Head는193,906 parameters이며 GRU·Action Head가 없다.

Absolute objective는 world displacement6D를0.1m scale, shape4D를0.02m scale로 회귀하고 sign-invariant rotation loss를 사용한다. Paired position Δ는0.01m scale이다. GT-GRU normalizer와 이 target scale은 별도다. 사전 설정은 AdamW3e−4/weight decay1e−5, cosine, absolute64 frames +16 pairs/update, Δ weight1,12k updates,250-update validation, first minimum episode-macro objective다. **이 설정의 본 실행은 아직0회**이며 실제 우선 데이터 지원을 확인한 final release가 필요하다.

Raw Geometry/Robot-only/Train-mean baseline, position mm·orientation degree·normalized field error·mean/median/p95·episode macro·present/missing, 같은-state Δ error/cosine/gain/ratio/zero-response, oracle/predicted38 assembly, frozen hashes·extraction cost를 계산하는 평가 코드를 구현했다. 실제 pretrained checkpoint가 없으므로 이들의 성능 수치를 만들지 않았다.

| 검증 | 결과와 근거 |
|---|---|
|원본38D 순서·단위·world displacement·wxyz·Clean|PASS: 원본 encoder/checkpoints와 대조|
|Measured Robot + oracle objects assembly|PASS: 최대5.96e−08|
|새 Full-task original/optimized Graph·Gate|PASS: t0/20/60/100 exact|
|DINO Detection Memo 원본 동등성|PASS|
|독립 CUDA worker 동등성|PASS:105 frames,24 payload arrays exact|
|GPU packing vs original packing|PASS: 다른 episode·중복 frame 포함|
|GNN/두 message layer/Head gradient·finite loss|PASS: 실제 cached train frames|
|Checkpoint/Optimizer/RNG next-update resume|PASS: 연속 실행과 weight hash exact|
|빈 물체·5-slot order·GT input whitelist|PASS|
|Raw/GT/graph/mask 승인·분리, 중복 READY 차단|PASS: isolated contract fixtures|
|Frozen same-checkpoint A/B CUDA API|PASS:4 real frames, 폐기용 engineering weights|
|본 GNN 사전학습 및 최종 checkpoint|NOT_TESTED|
|Held-out absolute/Δ·baselines·trained extraction cost|NOT_TESTED|
|실제 Task10-F DATA_READY 전송·최종 혼합 release|NOT_TESTED|
|CUDA PhysX / macOS simulator / MPS|NOT_TESTED|

실제 CUDA smoke의 gradient norm은 GNN0.07884, Head1.05050, message layers0.04156/0.03840이었다. 이것은 engineering test이며 일반화 성능이 아니다. 추가 frozen API 첫 시도는 deterministic reduction을 명시하지 않아 최대1.79e−07 차이로 exact-repeat 검사에 실패했다. 실패 checkpoint/log를 보존했다. Freeze/CLI에서 deterministic mode를 명시한 v2는 **허용 오차를 완화하지 않고** exact 출력과 동일 weight hash를 통과했다.

Freeze API: `encode_graph`, `predict_state`, `extract`, `assemble_gt38`. Sensor-only CLI는 GT dataset/sidecar를 읽지 않고 같은 checkpoint에서 State38과 Readout320/Presence2/Robot76을 추출한다. 보편적 식별 불가 차원은 metadata로 표시한다. 기존 inheritance source711개 파일 hash는 변경이 없었다.

[Engineering GPU/Resume 증거](../../outputs/task10f1/evidence/engineering_verification.json), [독립 worker 증거](../../outputs/task10f1/evidence/concurrent_worker_benchmark.json), [Frozen CUDA API v2](../../outputs/task10f1/evidence/frozen_API_engineering_verification_v2.json), [실패 로그 보존](../../outputs/task10f1/evidence/engineering_extract.log), [계약 테스트](../../outputs/task10f1/contracts_tests.log), [공통42 tests](../../outputs/task10f1/infrastructure_tests.log).

## 5. DATA_READY 자동 연결과 중복 방지

로컬 Mac의 독립 tmux `task10f1_data_ready`에서30초 간격으로 sealed `gnn_dataset_ready.json`을 읽는다. 기존 프로젝트 `event_ops.py`의 fingerprint/flock/CLI command/main lifecycle helper를 재사용하며 원본 monitor 코드는 변경하지 않는다. 일반 진행 중 모델 호출은0이다. Marker가 없다는 이유로 Task10-F를 중지·재시작하거나 Codex를 호출하지 않는다.

Marker가60초 이상 안정적이고 collection/Graph readback/physical/GT38 alignment/parent provenance/sealed-file gate가 PASS일 때만 전체 파일을 검증한다. 실제 HDF5·GT·Graph·mask·parent split과 SHA/size를 읽고 검사한다. 파일 존재·정상 action mask만으로 READY를 승인하지 않는다.

유효한 데이터 content로 Event ID를 계산하고 `DATA_READY`를 한 번 기록한다. 완료 파일만 로컬 import staging과57987772 독립 import 경로로 복사하고, source/local/target SHA 및 readback을 다시 비교한다. 기존 primary/root artifact를 덮어쓰지 않는다. 그 다음 main turn이 idle이고 인증 검증이 PASS일 때 Invocation Journal claim을 먼저 기록하고 Codex callback을 최대1회 시작한다. 같은 Event ID는 반복 실행하지 않는다. Source 수집 프로세스를 조작하는 경로는 없다.

실제 로컬 CLI0.165.0의 ChatGPT 인증,57987772 read-only SSH 및 지정 proof 파일 쓰기를 **격리된 fixture에서1회** 실행해 PASS했다. 이것은 실제 DATA_READY나 GNN 학습 실행이 아니다. [공식 non-interactive 안내](https://learn.chatgpt.com/docs/non-interactive-mode)와 실제 설치 CLI/help를 확인했으며, 실제 가능 여부는 fixture 증거를 기준으로 한다.

현재 watcher는 WAITING_DATA, production READY invocation0회다. 실제 READY-triggered 전송·최종 release·본 학습까지의 end-to-end 실행은 아직 NOT_TESTED다. Watcher는 로컬 Mac에서 실행하므로 Mac이 꺼지거나 잠들면 관측·CLI 재개가 지연될 수 있다. 다른 인스턴스에 인증 정보를 복사하지 않았다.

근거: [CLI/Auth/SSH 실제 검사](../../outputs/task10f1/operations/callback_verification.json), [현재 heartbeat](../../outputs/task10f1/operations/watch_state.json), [Invocation Journal](../../outputs/task10f1/operations/invocation_journal.json), [운영 설정](../../outputs/task10f1/operations/watch_config.json).

## 6. 안전한 재개 명령과 남은 작업

현재 source marker가 없으므로 아래 학습 명령을 바로 실행하지 않는다. DATA_READY callback은 원본 사용자 지시와 추가 steering 전체, source/transfer evidence, 진행 중 F1 상태를 먼저 읽는다. 완료 stage를 재실행하지 않으며 다른 작업과 중복 소유하지 않는다.

```bash
# 57987772, 독립 source에서만. 실제 sealed import와 source marker SHA가 필요하다.
python -B -m graph.gt_pretrain10f1.release \
  --root /workspace/task10f1_gt_pretrain_20261030 \
  --imported /workspace/task10f1_gt_pretrain_20261030/imports/EVENT_ID \
  --marker-sha SOURCE_MARKER_SHA256

# release가 READY_FOR_GNN_PRETRAINING이고 final 파일 SHA가 맞을 때만 허용된다.
WANDB_MODE=disabled CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -u -B -m graph.gt_pretrain10f1.train --root /workspace/task10f1_gt_pretrain_20261030
python -u -B -m graph.gt_pretrain10f1.evaluate --root /workspace/task10f1_gt_pretrain_20261030
```

`release`는 별도의 `final_dataset_manifest.json`, `final_pretrain_config.json`, `data_ready_release.json`을 만들며 초기 파일을 수정하지 않는다. Raw GT를 원본 encoder와 대조하고 normalized training sidecar를 독립 import 영역에 만든다. Action Mask와 별도 GNN Mask를 사용하며 같은 parent의 branch를 다른 split에 넣지 않는다. 실제 배포 데이터의 release는 아직 검증 전이다. 일치하지 않는 runtime을 임의로 승인하거나 전체 Task10-F dataset 대신 engineering2개만 넣어 학습하지 않는다.

본 학습 이후 남은 필수 산출물은 selected `gnn_pretrained.pt`(GNN+Head+optimizer/scheduler/RNG), train/validation CSV, 독립 absolute/Δ/Robot-only/Raw Geometry metrics, 최종 oracle/predicted38 검증, 같은 pretrained checkpoint의 A/B 추출·cost, 그림·static HTML·최종 verification/artifact index다. 이 파일들을 빈 placeholder로 만들어 완료로 표시하지 않는다. 새 GRU/Action/BC 학습은 별도 지시 전까지 시작하지 않는다.

## 7. Q1–Q10 현재 결론

| 질문 | 판정 | 현재 답 |
|---|---|---|
|Q1 GNN이 예측해야 할 항목|CONFIRMED|Robot15D 재예측 제외, Object18D estimate; head vector는 파생, freshness는 계약|
|Q2 Graph 상태 복원 정확도|NOT_TESTED|본 학습과 held-out 복원 평가 전|
|Q3 Raw/Robot-only 대비 우위|NOT_TESTED|비교 코드만 구현; 성능 수치 없음|
|Q4 새 Peg/Box 이동 방향·크기|NOT_TESTED|물리 유효 paired 자료 확보, pretrained response 평가는 전|
|Q5 Readout320 공간 정보|INCONCLUSIVE|구조·gradient·frozen 출력은 확인, 학습된 의미 정보는 미평가|
|Q6 GT38 호환·GT 누출 방지|SUPPORTED|Oracle·sensor whitelist·후보 predicted assembly API 검증; 정확도/보편적 관측 가능성은 미확인|
|Q7 동일 Frozen GNN으로 A/B 인계|SUPPORTED|같은 폐기용 checkpoint의 실제 CUDA API 검사 PASS; 최종 pretrained checkpoint는 아직 없음|
|Q8 재학습 없는 추출 비용|NOT_TESTED|최종 pretrained representation의 비용을 아직 측정하지 않음|
|Q9 부족·비관측 Geometry|CONFIRMED|Task10-F per-step GT 누락, symmetric orientation/hidden rim·extent 한계, runtime 차이|
|Q10 후속 GRU 학습 근거|INCONCLUSIVE|본 GNN 학습·absolute/Δ 평가가 먼저 필요하며 GRU를 시작할 근거가 아직 없음|

## 8. 보존·산출물

기존 ExperimentLogger identity/project/group 규칙을 유지하고 local/disabled logging을 사용했다. 원본 Dataset/Graph Cache/BC 및 GT-GRU checkpoints/Source는 읽기 전용이다. 초기 CUDA worker 전환 때 미완료 파일은 독립 interrupted 영역에 보존했고, 완료 cache는 SHA 검사 후 재사용했다.

120개 raw/GT와 snapshot 시점의 완료 cache·검증·engineering checkpoint를 [evidence](../../outputs/task10f1/evidence/)에 회수한다. 현재 단계의 파일별 SHA는 [Preparation artifact index](../../outputs/task10f1/evidence/preparation_artifact_index.json)에 있다. [로컬 회수 readback](../../outputs/task10f1/local_preparation_readback.json), [준비 verification](../../outputs/task10f1/verification.json), [준비 static HTML](../../outputs/task10f1/research_report.html)은 이 인계 상태의 증거다. 최종 성능·checkpoint artifact index와 구분한다.

실제 회수 SHA 검사는 **864개 파일/882,882,952 bytes PASS**다. [Runtime source·robot/camera inputs archive](../../outputs/task10f1/prepared_runtime_source_steering_v1.tar.gz)와 [최신 팀 코드 overlay](../../outputs/task10f1/current_team_source.zip), [복원 순서](../../outputs/task10f1/source_restore_instructions.json)를 보존했다. 원격 원본 metadata는 바꾸지 않고 준비 run의 로컬 사본에 공통 artifact 등록 함수를 통해 보고서를 연결했다. 최종 테스트는 계약13개 + 공통42개, 합계55개 PASS다.
