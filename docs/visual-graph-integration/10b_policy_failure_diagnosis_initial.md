# Task 10-B — Plan A BC Policy Failure Diagnosis

정상 Expert 상태에서부터 controller 명령 오차가 확인됐다. 따라서 현재 실패를 자율 상태의 Distribution Shift만으로 설명할 수 없다. 저장 rollout에서는 접근 후 조기 Close와 외향 이동이 함께 관찰됐다. 다만 Geometry-to-Action의 적절성, GNN 표현 한계, feedback 증폭의 독립 기여는 이번 분석으로 분리하지 못했다.

ExperimentLogger UUID: `35877452d6d94d9691964fc8a8490858`. Parent comparison: `3788798a719f4ec4a66c224ca42f5d63`.

## 실행 범위와 근거

Phase 1과 2를 실행했다. Train30003 + Dev42000–42004의 기존 scored rollout 108회(E0/E1 각54회, 고유 seed6개), Train96/Validation12의 13,452개 action 상태를 네 frozen checkpoint에서 분석했다. 53,808개 모델별 예측을 다루지만 독립 상태 수는13,452다. Frame·checkpoint 반복을 독립 Episode로 취급하지 않는다. step500 smoke와 Test12 rollout/cache/prediction은 열지 않았다.

네 checkpoint의 SHA256·model tensor identity·config·Robot76 normalizer·ActionCodec이 frozen receipt와 일치했다. E0 245,447 parameters, E1 194,759 parameters로 구조와 용량이 다르다. Readout320+Presence2+Robot76 → encoder128 → GRU144 → Action7을 유지했다. legacy Graph14·Full ABC 가중치는 이식하지 않았다.

원격 독립 source는 `/workspace/task10b_diagnosis_20261029/source`, 원본은 `/workspace/plan_a_bc_3788798a719f4ec4a66c224ca42f5d63`다. source manifest677개 파일 모두 일치했다. 새 코드는 별도 `diagnose.py`, `supplement.py`뿐이며 원래 policy/geometry는 수정하지 않았다. `report.py`는 로컬 결과로 보고서를 만든다. 공통 ExperimentLogger와 기존07-G cosine/stats,08-E extra_metrics, ActionCodec/controller_commands, Plan A graph_at/pack_graphs를 재사용했다.

## Phase 1 — 실제 시간축

Observation[t]와 Robot76[t]에서 Action[t]를 예측하고 trace pose/거리는 post-state[t+1]이다. 저장 hidden[t]는 forward 이후 값이므로 previous hidden은 t=0에서 zero, 이후 stored[t−1]이다. 다음 Robot76 TCP와 이전 trace post-TCP가 일치하고, action 배열 및 저장 Reach/Grasp/Lift 집계가 일치했다. Grasp 접촉 predicate는 저장 bool을 재집계했으며 물리를 재실행하지 않았다.

최초 peg pose가 trajectory에 없으므로 정확한 초기 TCP–Peg 거리와 t=0 목표 방향은 NOT_TESTED다. t>=1에서는 이전 post-peg와 현재 pre-TCP를 사용했다. Native action을 installed controller로 전처리한 뒤 Robot base quaternion으로 world 방향에 회전했다. 최소거리의 post-state 번호와 명령의 t를 구분한다.

아래 외향 후보는 target-to-command cosine<0이 3프레임 연속인 최초 시점이다. 인과적 최초 오류나 공식 PASS/FAIL 기준이 아니다. Grasp 후 목표가 바뀌는 경우에도 같은 기하 지표가 나올 수 있으므로 성공 사례의 외향 시점을 곧바로 실패로 해석하지 않는다.

| 모델 | Seed | 최소거리 mm | 최소 post-state | 첫 3연속 외향 명령 t | 첫 Close t | 최종거리 mm |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 30003 | 92.8 | 33 | 35 | 34 | 361.0 |
| E0 | 42000 | 87.8 | 39 | 38 | 39 | 365.1 |
| E0 | 42001 | 106.2 | 69 | 39 | 40 | 152.3 |
| E0 | 42002 | 144.0 | 24 | 24 | 28 | 296.7 |
| E0 | 42003 | 270.4 | 12 | 11 | 8 | 501.1 |
| E0 | 42004 | 85.2 | 34 | 35 | 55 | 435.2 |
| E1 | 30003 | 177.4 | 14 | 14 | 14 | 441.2 |
| E1 | 42000 | 193.2 | 16 | 15 | 12 | 415.7 |
| E1 | 42001 | 92.3 | 49 | 23 | 49 | 170.2 |
| E1 | 42002 | 159.9 | 16 | 15 | 16 | 316.2 |
| E1 | 42003 | 179.5 | 20 | 20 | 18 | 508.6 |
| E1 | 42004 | 166.0 | 27 | 26 | 28 | 461.8 |


seed30003 Last: E0는 최소92.8mm 이후 t35에서 연속 외향 명령, t34 Close가 관찰됐다. E1은 최소177.4mm, t14 외향 명령과 Close가 관찰됐다. 두 시점 모두 Peg 존재·wrist gate 채택이며 seed30003 전체250프레임에서 Peg 누락은0이다. 단순 누락만으로 이 사례를 설명할 수 없다. Node/Edge 존재는 mask purity·정확한 geometry를 보장하지 않는다.

Last Dev42000의 Peg 누락은 E0 219/250, E1 246/250이고, Dev42001은 E0 20/250, E1 173/250이다. E1의 두 seed에서는 외향 후보 시점에 Peg가 이미 누락됐다. 다른 seed에서도 누락이 필수로 동반되는 것은 아니다. Gate·노드·간선·희소 저장 graph의 실제 point 수는 `stored_graph_gate_audit.json`을 따른다. Live graph에는 완전한 pixel/source purity 근거가 없으므로 GT 오염 판정은 NOT_TESTED다. Gate의 GT_used/future_used와 association 기록은 보존했으나 독립적인 source purity 검증을 대신하지 않는다.

## Phase 2 — Expert-state Action Fit

전체 저장 예측을 우선 재사용했다. 네 모델 각각 seed30003의16개 연속 cached graph에서 원래 forward와 진단 adapter의 Readout, GRU input/hidden, Native Action7, decoded controller command가 CPU에서 bit-exact하게 일치했다. 전체137-step sequence를 hidden carry로 다시 실행해 저장 예측과의 최대 절대 차이도 기록했다(약5.5–6.2e−6 native unit). 이는 CPU와 원래 저장 추론의 수치 비교이며 bit-exact PASS로 표시하지 않았다. 최초 엄격 tolerance 검사가 실패한 로그도 보존했다. 어댑터 동등성의 exact 기준은 변경하지 않았다.

Train/Validation 432개 예측 파일에서 target·timestep이 원본108개 cache와 정확히 일치했다. cache SHA256·Node21/Edge4·Robot76 길이를 검사했다. Checkpoint weight hash는 추론 전후 동일했고 inference_mode를 사용했다. 재학습은 없다.

| 모델/Checkpoint | Train MSE | Val MSE | Val translation mm | Val t0–40 mm | Val 반전 mm |
|---|---:|---:|---:|---:|---:|
| E0 validation_best | 0.004475 | 0.006416 | 1.924 | 1.595 | 8.335 |
| E0 last | 0.004463 | 0.006433 | 1.927 | 1.602 | 8.287 |
| E1 validation_best | 0.005261 | 0.007035 | 1.838 | 1.570 | 7.560 |
| E1 last | 0.005252 | 0.007053 | 1.837 | 1.562 | 7.510 |


위 평균은 frame 수 가중이며 독립성에 대한 통계 추론이 아니다. 단위는 installed PDEEPoseController clipping/scaling 이후 translation vector L2 mm다. rotation은 intrinsic XYZ command rotation의 상대 SO(3) geodesic rad로 계산했으며 실제 TCP orientation 오차가 아니다. Gripper target도 controller 전처리 후 값이다. 실제 TCP 이동량은 Phase1에서 별도 기록했다.

구간명은 정확한 Expert phase label이 아니며 t0–40, 최초 Close 이전, Close±5, 작은 translation<1mm, command 방향 반전으로 정의했다. Expert HDF5 inventory에 phase/stage label을 찾지 못했다. 방향 cosine은 expert translation<1e−6m 또는 zero denominator에서 미정의(null/빈 CSV)로 남겼다. 1mm는 기존08-E에서 가져온 진단용 기준이며 성능 성공 기준이 아니다.

### Q1. 정상 Initial Reach부터 틀리는가?

**CONFIRMED (관측):** seed30003의 네 checkpoint 모두 최초 translation error>1mm가 t0이다. Last의 t0–40 평균은 E0 1.761mm, E1 1.429mm이고 1mm 초과 비율은 각각21/41,15/41이다. 다만 이를 전체 환경의 최초 인과 실패 시점으로 확장하지 않는다. 이 시점들의 정확한 phase label이 없어 initial time window로 기술한다.

### Q2. 전체 Action7 MSE가 숨기는 정밀도 문제가 있는가?

**CONFIRMED (관측):** E0의 낮은 전체 MSE가 controller translation 우위를 뜻하지 않는다. Val Last에서 E0 1.927mm, E1 1.837mm다. Val 방향 반전48개 frame에서 각각8.287mm,7.510mm이며 1mm 초과46/48,47/48이다. 모든 expert 방향 반전을 이전 방향 고집으로 해석할 수 없다. Train+Val 반전417개 중 Last가 이전 방향 cosine>0, 현재 방향 cosine<0인 경우는 E0 69개, E1 57개다. 이는417개의 독립 실험이 아니다. seed30003 반전4개에서는 평균 현재 방향 cosine이 두 모델 모두0.98 이상이어서 큰 오차에 크기 문제도 포함된다.

### Q3. 실제 rollout은 언제 잘못 움직이는가?

**CONFIRMED (관측):** 위 Last 표와 `stored_rollout_step_metrics.csv`에 접근·반전·최소거리·명령·실제 이동을 저장했다. seed30003의 t35(E0),t14(E1)는 3연속 외향 명령 후보이며 첫 미충족 Reach predicate를 최초 제어 오류로 사용하지 않았다. Autonomous와 Expert의 같은 timestep action을 동일 상태 pair로 비교하지 않았다.

### Q4. 그때 Graph는 어떻게 변했는가?

**혼합 증거:** seed30003 두 Last에서는 Peg 존재와 gate 채택이 유지됐다. E1 Dev42000/42001에서는 candidate 시점 Peg가 누락됐다. `stored_graph_gate_audit.json`은 candidate±2 gate·노드·간선과 실제 저장 frame의 Peg/Box point 수를 보존한다. 모든 frame의 point/source purity는 복원할 수 없다. 누락의 존재는 확인되지만 그것이 원인인지 상태 이탈의 결과인지 분리하지 못했다.

### Q5. 실제 Peg Geometry 변화에 적절히 반응하는가?

**NOT_TESTED:** 새 동일 상태 물리 Peg±5/10mm 개입은 실행하지 않았다. 기존09-A는 legacy 모델 결과이며 이번 E0/E1 결과로 옮기지 않는다.

### Q6. Raw Geometry, Readout, GRU/Action 중 어느 단계인가?

**INCONCLUSIVE:** 현재 Expert input에서도 출력 오차가 있다는 것과 일부 autonomous graph 누락은 확인했다. Graph→Readout→Action의 올바른 반응 방향·gain·counterfactual reference는 검증하지 않아 bottleneck 위치를 분해할 수 없다. Frozen probe도 실행하지 않았다. 기존 readout 정보 소멸이나 GNN 단독 원인이라는 결론은 근거가 없다.

### Q7. E0/E1과 중간 성공은 어떻게 다른가?

seed30003 Last의 E0가 E1보다 가까이 접근했지만, Expert 상태 translation은 E1이 더 정밀하다. 따라서 한 지표의 일관된 우위를 주장할 수 없다. Best→Last 차이는 작다. Val translation은 E0 1.924→1.927mm, E1 1.838→1.837mm, Val t0–40은 E0 1.595→1.602mm, E1 1.570→1.562mm다. 정식 Best 선택은 그대로다. Best10750의 Train/Dev stored rollout은 일정상 없으므로 폐루프 Best 대 Last 직접 비교는 NOT_AVAILABLE이다.

실제 초기 성공은 **E1 step1000 / Dev42000**이다(seed30003 성공으로 오인하지 않는다). Close t44, Grasp post46, 최소6.28mm(post90), 최대84.18mm lift가 기록됐다. 같은 seed Last는 Close t12, 최소193.21mm, Grasp 없음이다. 초기 성공에서도 Peg 누락205/250과 gate 채택0이 기록돼, 이 사례는 gate 채택이나 presence 하나가 성패를 결정한다는 주장도 지지하지 않는다. E0 step10000의 lift 사례는 Dev42004다. Reach origin30mm는 미충족이면서 Grasp/Lift를 기록했으므로 서로 다른 predicate임을 다시 확인했다. 성공 사례는 모델 재선택에 사용하지 않았다.

### Q8. 초기 오류가 feedback에서 증폭되는가?

**INCONCLUSIVE / 인과 검증 NOT_TESTED:** Expert-state 오차와 실제 접근 실패가 함께 존재하지만, 이것만으로 동일 작은 초기 오류의 Model suffix 증폭을 입증하지 않는다. E/E·EM·ME·MM 분기는 실행하지 않았고 과거08-F 수치를 새 checkpoint에 귀속하지 않았다. Live Close가 더 이른 사실도 물리상태·현재 graph·GRU history의 독립 효과를 분리하지 못한다.

### Q9. 가장 유력한 원인 1–3개

| 우선 | 원인 후보 | 증거 등급 | 해석 |
|---|---|---|---|
| 1 | Expert-state action fit 및 전환/정밀 명령 오차 | CONFIRMED 오차 존재 / SUPPORTED 실패 기여 | t0 오차, 반전 구간 큰 controller error. 기여율이나 단독 충분원인은 미확정 |
| 2 | 자율 상태의 조기 Close·외향 이동과 누적 이탈 | SUPPORTED 현상 / INCONCLUSIVE feedback 메커니즘 | 같은-state branch 없이 증폭 인과를 확정하지 않음 |
| 3 | 일부 Dev에서 Perception/Geometry 전달 누락 | CONFIRMED 누락 / INCONCLUSIVE 인과 기여 | 다른 실패 seed는 Peg 유지, 초기 성공 seed도 누락 다수 |

Geometry-to-Action 활용, GNN/Readout 표현, GRU history 각각의 원인 기여는 INCONCLUSIVE이며 직접 개입은 NOT_TESTED다. 복수 원인이 공존할 수 있다.

### Q10. Task10-C 우선순위

1. **Controller 명령과 전환 구간의 BC fit 개선을 우선 후보로 삼는다.** 현재 자료의 반전 구간7.5–8.3mm 오차와 t0 오차가 근거다. 전체 Action7 loss만 비교하지 말고 stage/전환별 translation vector, rotation geodesic, Close timing을 함께 고정 평가한다. 손실 weighting·전환 표본 분포·학습 fit 개선 중 무엇이 효과적인지는 후속 비교 실험이며 이번에는 적용하지 않았다.
2. **모델 구조 변경 전에 최소09-A 인과 개입으로 Geometry 활용을 분리한다.** Train30003, 동일 Robot76/previous hidden/causal vision history, 유효 Peg XY±5/10mm, 반복 baseline과 expert counterfactual 검증을 사전 고정한다. seed30003 실패에서 Peg presence가 유지된다는 사실만으로 GNN 문제를 확정할 수 없기 때문이다.
3. **그 뒤 필요하면08-F의10-action 분기로 feedback 기여를 검증한다.** Current reset/prefix identity가 통과한 경우에만 E/E·EM·ME·MM을 비교한다. Test12 결과로 설정을 조정하지 않는다. DAgger·새 training·hidden 이식은 이번에 실행하지 않았다.

과거 Full B seed30003 Best MSE0.003438041, 마지막0.003433573에 비해 현재 E0 Last0.005194677, E1 Last0.005042027로 약1.51×,1.47×다. Full B Best controller translation1.525mm 대비 현재 Last E0 1.473mm/E1 1.299mm다. 기준은 보존된08-E 보고서이며 기존 checkpoint 재실행이 아니다. Geometry/feature/parameter/runtime와 selection cadence 차이가 있어 순수 Geometry 효과로 해석할 수 없다.

## 실행 생략과 검증 한계

요청의 Phase1/2 중단 규칙을 적용했다. 정상 Expert 상태의 오차로 Distribution-Shift-only 설명은 이미 배제되므로 새 physics rollout을 확대하지 않았다. Phase3 물리 geometry 개입과 Phase4 증폭 분해는 여전히 미검증이며 `scope_decision.json`에 남겼다. 이 보고서는 전체 인과 분해 완료나 모든11항 계약 PASS를 주장하지 않는다. 초기 상태의 현재 재현성도 새로 검사하지 않았으므로 Task10-A의 reset 결과를 가정하지 않는다.

실행환경은 Linux Python3.12 / Torch2.14.1+cu130 / NumPy2.5.4이며 **진단 추론과 CSV 분석은 CPU만 실행**했다. 이번 Task에서 CUDA policy/vision, MPS, PhysX/SAPIEN simulator/render backend는 실행하지 않았다. installed controller 전처리 source는 `installed_controller_source.json`에 보존했다. 원본 runtime guard와 reset hash를 삭제하거나 우회하지 않았다. Task10-A 프로세스/코드/작업환경, Plan B, 원본 artifact는 변경하지 않았다. Git commit·외부 publishing·메시지 전송도 없다.

`verification.json`에 PASS/NOT_TESTED, `checkpoint_identity.json`에 네 checkpoint 실제 경로와 hash, `frozen_source_hash_audit.json`에 원본 source 대조 결과가 있다. 실패한 진단 실행2건의 Logger와 로그도 남겼다(어댑터 metadata 중복 key 수정, CPU–저장 예측의 과도한 tolerance 검사). 마지막 성공 UUID를 본문에 사용한다.

## 자료

- [단계별 원자료](stagewise_action_errors.csv), [전체 Expert 오차](expert_state_action_errors.csv), [Controller command 오차](controller_command_errors.csv)
- [저장 rollout 분석](stored_rollout_failure_analysis.csv), [시점별 거리·명령·이동](stored_rollout_step_metrics.csv), [시간 정렬 검사](trajectory_alignment_audit.json)
- [반전 분석](correction_transition_analysis.json), [검증](verification.json), [Artifact Index](artifact_index.json)

![Expert-state 명령](expert_vs_student_actions.png)
![Expert-state correction error](reach_correction_error.png)
![Stored distance](reach_distance_curves.png)
![Stored command direction](action_direction_curves.png)
![Wrist gate](presence_gate_timeline.png)
