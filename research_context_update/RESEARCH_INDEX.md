# RESEARCH_INDEX — 필요한 원문 찾기

> 기준: 2026-10-29 기존 패키지에 2026-11-03 KST 확인한 로컬 보고서를 증분 반영했다. 각 상태의 관측 시각은 원문을 따른다. `TEAM_CONTEXT.md`와 함께 사용한다. 이 문서는 원문을 대체하는 전체 연구 요약이 아니라 **질문에 맞는 문서를 골라 읽기 위한 안내**다.
> 아래 모든 경로는 `아카이브.zip`의 루트 기준이다. 예: 아카이브.zip :: `09d_plan_a_geometry_suitability.md`. ZIP 안으로 자동 이동하는 웹 링크를 뜻하지 않는다.

## 먼저 구분할 최신 기준

| 질문 | 우선 확인할 원문·구간 | 현재 판단 |
|---|---|---|
| 연구를 왜 시작했나? | `context_sources/VLA_GNN_Recurrent_연구_스토리_v0.md` §1–5 | Slow semantic reasoning / fast embodied control 분리. 최근 BC 실패가 최초 동기는 아님 |
| 전체 제안과 지금 구현은 어떻게 다른가? | `TEAM_CONTEXT.md` §2–3 + 아래 설계 v0/09-D/10-B | VLM semantic path·full adaptive re-anchor는 전체 구상, 현재까지는 fast-path feasibility 및 후속 Perception·정책 진단이며, 전체 Slow/Fast 우위는 미입증 |
| Plan A 학습·최종 평가가 끝났나? | `10_plan_a_cached_graph_bc.md`의 실행 범위·최종 Test12·Reset 정합성 | **후속 보고서 기준 최종 48회 완료**. 초기 FAILED snapshot보다 이 보고서를 우선 |
| 왜 조작에 실패하나? | `10b_policy_failure_diagnosis.md`, `10c1_frozen_representation_probe.md`, `10h_phase_conditioned_handoff_diagnosis.md`, `10i_stage4a_final_report.md` | 각 checkpoint·조건의 fit/표현/관측/제어 문제를 구분. 전체 실패의 단일 원인은 미확정 |
| Frozen probe 등 후속 후보가 실제 실행됐나? | `10c1_frozen_representation_probe.md`, `10c2_learning_signal_audit.md`, 아래 9–13절 | C1/C2와 D~H, Predicted State38 Final120회는 후속 결과가 있다. Stage4A-2 채택은 미완료이며 초기 권고를 현재 대기 작업으로 읽지 않는다 |
| Plan A Hybrid가 Plan B인가? | `09d_plan_a_geometry_suitability.md` §3, `09e_plan_b_box_persistence.md` §3–4 | 전자는 현재 point selection, 후자는 Box 과거 surface memory. 서로 다름 |
| 왜 Shape-First를 사용했나? | 03-J/K, 09-C Legacy, `09d_plan_a_geometry_suitability.md` §6·9·11 | 표면 정확도와 boundary 보존의 trade-off. 보편적 우승 표현이 아님 |
| 캐시/전처리 최적화는 적용됐나? | `10_plan_a_cached_graph_bc.md`의 P1–P6 | 조합 후보 약1.198배와 exact 비교를 보고했으나 **기존 frozen BC 실행에는 미적용** |
| 예전 진단을 재사용할 코드는? | 07-G Action / 08-E / 08-F / 09-A / 09-B, 10-B Phase1/2 | 기존 진단 계산을 재사용하되 새 모델의 schema/codec/hidden과 정합 검증 |
| 가장 최근의 Perception·정책 상태는? | `10i_stage4a2_paused_scope_review.md`, `10i_stage4a2_perception_readiness_bottleneck.md`, `10i_stage4a2_qwen35_rejection_mechanism.md`, `10i_predicted_state38_graph_recurrent_bc.md` | Stage4A COMPLETED/NO_GO, Stage4A-2 PAUSED. State38 Final120회 완료; 자율 삽입 우위는 미확인이고 H2 표본의 S6/12 대 R/J2/12는 제한적으로 해석 |

**중요한 갱신 관계**


- **10-B→10-C1/C2→10-D~H:** 당시 frozen probe 우선 권고는 후속 실행으로 이어졌다. C2의 조건 충족 pair0은 가설 실패가 아니라 근거 부족이다. 새 실험 결과를 이전 E0/E1 checkpoint에 소급 귀속하지 않는다.
- **10-F:** `10f_corrective_reach_bc.md`의 초기 물리 gate 실패 뒤 `10f_pilot_recovery_v2.md`에서 별도 수집·BC·평가를 완료했다. 성공률 개선은 미입증이며 동일 step 해석은 `10f_matched_checkpoint_handoff_diagnosis.md`를 우선한다.
- **F1/Stage4A:** 준비·데이터 NOT_READY 및 재개 전 기록보다 `10f1_gt_supervised_gnn_pretraining_final.md`와 `10i_stage4a_final_report.md`가 해당 완료 범위의 최신 기준이다. F1 자체의 policy 미실행과 후속10-G의 실제 BC 완료를 구분한다.
- **Qwen/SAM2:** 국소 Prompt 개선·좁은 Gate PASS를 일반적 선택기 승인으로 읽지 않는다. `10i_stage4a2_sam2_mask_verifier_tracking.md`가 후속 Mask/추적 결과이며, readiness 보고서 당시의 SAM2 미평가는 이 범위에서 갱신됐다. Mask 표시C 개선은 없고 추적 경로도 오초기화·Wrist 실패가 남아 채택·정책 성능 근거로 확대하지 않는다.
- **State38:** 시작→Vulkan 중단→평가 재개→17:33 KST 부분 결과→23:23 KST 완료 순서다. `10i_predicted_state38_graph_recurrent_bc.md`와 실제 완료 CSV/감사를 최신 기준으로 사용하고 중간 Final36/120·Handoff 미실행을 현재 상태로 읽지 않는다.
- `TEAM_CONTEXT.md`는 **2026-10-29 바이트를 그대로 보존**했다. 그 파일의 C1 미실행·당시 다음 우선순위는 위 후속 원문으로 갱신할 대상이며 이번 작업에서는 편집하지 않는다.

- `10_plan_a_cached_graph_bc.md`는 초기 Test reset 실패 뒤의 **복구 최종 보고서**다. `evidence/plan_a_bc_20261029T032648Z/`는 실패 당시 자료이므로 현재 평가가 여전히 0회라고 해석하지 않는다.
- `10b_policy_failure_diagnosis.md`가 최신 추가 진단이다. `history/10b_policy_failure_diagnosis_phase1_2.md`의 “물리 개입·분기 미실행”과 loss-first 우선순위는 해당 시점 기록이며 최신 결론을 덮어쓰지 않는다.
- 09-D의 “BC 미실행”, 09-E의 “후속 비교 미실행”은 **각 실험의 범위**다. 09-D 뒤 별도 Plan A BC를 실제 수행한 사실과 구분한다.
- 최종 Test 완료·reset 복원은 후속 **보고서 근거**다. 최종 48회 CSV/영상·복원 코드는 이 패키지에 없으며, 여기서 독립 재검증한 것으로 표시하지 않는다.

## 원문 목록 — 주제별

`결과/진단`은 실제 실행 범위를 원문에서 확인해야 한다. `계획/요청`은 동봉된 과거 실행 지시이며, 후속 결과보다 현재 우선권을 갖지 않는다. 아래 간략한 판단은 해당 조건 안의 관찰이지 일반 성능 주장이나 현재 설정을 바꾸라는 지시가 아니다.

### 1. 초기 Geometry 설계와 normal/feature

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `00_interface.md` | 설계·계약 | Vision–Graph 역할과 interface. Task04의 현재 관측 v2 packet 구현과 전체 설계안을 구분한다. |
| `00_search.md` | 과거 코드 조사 | 기존 Vision E9와 GT Graph prototype의 입력 경로·단절을 조사한 기록. 현재 통합 상태로 읽지 않는다. |
| `00_ultimate_goal.md` | 하위 연구 구상 | Point cloud의 구조·경계·budget 개선 동기. 프로젝트 전체의 최초 문제의식은 연구 스토리 v0를 따른다. |
| `00_ultimate_goal_2.md` | 하위 연구 구상 | Pose fitting 자체보다 GNN에 유용한 surface/relative geometry 입력을 만들려는 설계 의도. |
| `01_organized_geometry.md` | 구현·결과 | Pixel-aligned XYZ/normal/관측 경계/variation 추출. 이 단계는 sampling·GNN·GRU 구현 결과가 아니다. |
| `02a_normal_reliability.md` | 진단 | 작은 normal stencil과 1mm depth 양자화가 Peg 평면의 높은 variation을 대부분 설명한다. |
| `02b_normal_comparison.md` | 비교 결과 | PCA5를 후속 후보로 제안. 평면 안정성·crease/rim 반응·비용의 trade-off를 기록한다. |
| `02c_structural_edges.md` | 비교 결과 | PCA5 orientation과 depth 경계 증거를 구분. Raw depth OR variation의 무조건 채택은 지지하지 않는다. |
| `pca5_optimization.md` | 최적화 결과 | PCA5 의미를 유지한 CPU/CUDA 최적화와 tracking 연계 측정. 초기 grounding·최종 Graph·controller 전체 지연은 제외. |

### 2. Point sampling과 구조 보존

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `03_sampling.md` | 비교 결과 | Structural/surface sampling의 seam·rim 보존을 비교. Peg 개선과 낮은 Box rim 보존이 함께 나타났다. |
| `03ab_adaptive_node_budget.md`<br>계획: `03ab_adaptive_node_budget_plan.md` | 비교 결과 | Adaptive budget이 node/시간을 줄였지만 seam/rim 손실도 컸다. 작은 geometry 손실 가설을 지지하지 않는다. |
| `03d_box_rim_bottleneck_phase1.md` | 진단 | 관측→후보→priority→surface 단계별 rim 손실을 분리. 후보 coverage가 높아도 최종 선택에서 역전될 수 있다. |
| `03d_priority_fps_trace_phase2.md`<br>계획: `03d_priority_fps_trace_phase2_plan.md` | 진단 | 원래 FPS 선택 순서를 exact 재현하고, 거리 경쟁과 단계별 rim 보존을 추적한다. |
| `03d_priority_fps_seed_control_phase3.md`<br>계획: `03d_priority_fps_seed_control_phase3_plan.md` | 통제 비교 | 공통 FPS 초기점에서도 후보 집합에 따른 차이가 남았다. 초기점 하나로 일반 기여율을 추정하지 않는다. |
| `03e_priority_fps_metric_control.md`<br>계획: `03e_priority_fps_metric_control_plan.md` | 통제 비교 | 동일 support·첫 점·quota에서 depth 가중 FPS 거리만 바꾼 frame0 비교. XY-only와 UV 거리도 구분한다. |
| `03f_local_structural_sampling.md`<br>계획: `03f_local_structural_sampling_plan.md` | 통제 비교 | UV 공간 분할로 국소 후보 보존을 비교. Grouping·quota·초기화가 함께 달라 단일 효과로 분해하지 않는다. |
| `03g_geometry_aware_sampling.md`<br>계획: `03g_geometry_aware_sampling_plan.md` | 제한된 결과 | Observed patch 기반 grouping의 첫 feasibility. 일부 거리 개선과 GT 구조 보존 감소가 함께 나타났다. |
| `03h_candidate_count_matched.md`<br>계획: `03h_candidate_count_matched_plan.md` | 진단 | 같은 후보 수에서도 PCA5의 기대 rim coverage가 높아, 후보 수 증가만으로 차이를 설명할 수 없었다. |
| `03i_voxel_structural_sampling.md`<br>계획: `03i_voxel_structural_sampling_plan.md` | 제한된 결과 | Voxel precompression 뒤 FPS cap을 적용한 비교. 순수 voxel이나 일반적인 Shape-First 우위의 증거는 아니다. |
| `03j_variable_rate_shape_first.md`<br>계획: `03j_variable_rate_shape_first_plan.md` | 구현·결과 | PCA5 구조 후보와 외부 silhouette를 이용한 variable-rate sampling. 고정 frame·ratio에서 geometry 보존을 평가. |
| `03k_temporal_shape_first.md` | 연속 관측 검증 | 한 Episode의 121프레임에서 Shape-First를 검증. Source-index 겹침을 물리적 point identity로 해석하지 않는다. |

### 3. Feature packet과 Graph edge

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `04_vision_feature_edge.md`<br>계획: `04_vision_feature_edge_plan.md` | 계약 검증 | EE normal3+variation+depth jump의 F5, availability/validity 및 point/source 정렬을 구현·검증. |
| `05_triangle_intra_object_edges.md`<br>계획: `05_triangle_intra_object_edges_plan.md` | 비교 결과 | TMD가 관측 규칙상 invalid edge를 줄였지만 고립·fragmentation·비용이 증가. 최종 채택 보류. |
| `05b_extended_delaunay_edgewise.md`<br>계획: `05b_extended_delaunay_edgewise_plan.md` | 비교 결과 | Edge-wise DE/EDE로 연결을 일부 회복. 관측 규칙 통과가 물리적 surface 정확성을 보장하지는 않는다. |
| `05c_ede_exact_optimization.md`<br>계획: `05c_ede_exact_optimization_plan.md` | 최적화 결과 | 동일 16조건에서 Edge exact equivalence와 CPU 속도 개선을 확인. 과거 host 수치와 분모를 섞지 않는다. |
| `05d_gpu_edge_benchmark.md`<br>계획: `05d_gpu_edge_benchmark_plan.md` | 최적화 결과 | 실제 MPS hybrid EDE의 exact 비교·wall timing. GPU kernel/event latency와 peak memory는 미지원/미측정. |
| `05e_raw_ede_comparison.md`<br>계획: `05e_raw_ede_comparison_plan.md` | 정책 비교 | 검증 경로를 생략한 Raw EDE와 Strict EDE 비교. 빨라진 이유는 수행 작업 차이이며 동일 연산 최적화가 아니다. |

### 4. Expert 수집·통합·실행 경로

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `06_graph_gnn_gru_baseline.md`<br>계획: `06_graph_gnn_gru_baseline_plan.md`<br>요청: `06_graph_gnn_gru_baseline_request.md` | 구현·당시 차단 | Graph–GNN–GRU 최소 prefix 통합은 검증했으나 원래 replay dataset은 BLOCKED_DATA였음. |
| `06a_replay_timeout_diagnosis.md` | 진단 | 공식 Expert replay의 drift/TimeLimit을 조사. 단일 원인 확정 및 전체 dataset 복구를 주장하지 않는다. |
| `06b_recurrent_expert_collection_pilot.md` | 파일럿 결과 | 현재 runtime에서 ScriptedExpert 직접 수집 5/5 성공, 669개 action 정합. 이후 직접 수집 방식의 근거. |
| `06c_direct_expert_bc_training.md` | 과거 실행 기록 | 120개 성공 raw Episode를 수집했으나 해당 run은 Vision 중 사용자 중단. 그 run의 학습은 시작되지 않았다. |
| `06c_vision_speed_check.md` | 후보 검증·전환 보류 | DINO 재사용/SAM2 경로 개선을 검토했지만 mask 1픽셀 차이로 해당 출력 일치 gate 미통과. |
| `06d_visual_stream_reuse.md` | 구현·재사용 검증 | 기존 Visual streaming을 opt-in backend로 연결. 원래 cache/가중치를 자동 전환하지 않음. 이전 ZIP에서 복원. |

### 5. 초기 Closed-loop 실패와 Reach 진단

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `07_stop_the_clock_rollout.md` | 구현·smoke | Zero-simulation-time inference delay 평가 경로 구현. 이 문서 시점의 trained-policy 본평가는 미실행. |
| `07b_stop_clock_vast57640710.md` | 과거 실행 snapshot | 학습된 Best로 smoke 후 held-out 평가를 시작한 당시 기록. 최종 40회 결과로 읽지 않는다. |
| `07c_fixed_checkpoint_grasp_diagnostic.md` | Offline 진단 | 저장 입력의 evaluation/streaming step 경로 정합. Close 부호 외 강도·팔 정지 명령 오차를 확인. |
| `07d_live_grasp_intervention.md` | 혼합 제어 개입 | 정상 Expert 접근 뒤 Close의 팔/그리퍼 제어를 교차 비교. 완전 자율 성공률이 아니다. |
| `07e_autonomous_reach_diagnosis.md` | 조건부 인과 진단 | Seed41005에서 Model 접근 뒤 Expert Close도 실패. Close 전 불리한 상대 위치 형성을 지지. |
| `07f_vision_oracle_reach_ablation.md` | Oracle 진단 | 같은 실패 seed에서 FP 제거·GT segmentation만으로 Reach/파지가 회복되지 않았다. |
| `07g_point_resampling_stability.md` | Sampling 진단 | Readout 변화에 비해 작은 local/phase sampling 변경의 action 영향은 작았다. Geometry 활용 여부와는 별개. |
| `07g_reach_action_distribution_analysis.md` | Action 진단 | Expert 상태의 방향 전환·보정 오차와 autonomous 외향 이동을 분석. 같은 07-G라도 앞 문서와 다른 진단. |

### 6. BC fit·단일 Episode·오류 누적

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `08_bc_normalization_overfit_training.md` | 과거 진행 snapshot | Small5와 Full A/B/C 파이프라인을 시작한 기록. 최종 상태·성과는 08-B~F 결과를 우선. |
| `08_bc_training_interim_summary.md` | 과거 중간 요약 | BC 진행 중 결과와 미완료 항목 정리. 이후 single/full 보고서를 덮어쓰는 상태 문서가 아니다. |
| `08b_single_episode_ac_overfit.md` | 학습·rollout 결과 | Seed30003의 A/C overfit과 live 조작 비교. 단일 궤적 성공을 Graph 활용·일반화 증거로 쓰지 않는다. |
| `08c_a_gripper_intervention.md` | 인과 개입 | Single A의 t79 opening이 파지 상실에 기여했지만 close 유지로도 삽입은 해결되지 않았다. |
| `08d_method_b_single_episode_overfit.md` | 학습·rollout 결과 | Motion6 정규화+continuous gripper인 Method B도 single-episode 삽입 성공. BCE 필수 주장을 지지하지 않음. |
| `08e_full_trainseed_diagnostic.md` | 진단·중지 기록 | Full A/B/C가 train seed에서도 파지 실패. Single 대비 fit 차이와 사용자 중단·미실행 Final40을 기록. |
| `08f_error_accumulation_diagnostic.md` | 분기 진단 | 이전 Full A/B/C의 Expert/Model 분기. 새 Plan A E0/E1에 수치를 귀속하지 않는다. |

### 7. Geometry 활용·정보 복원·Plan A/B

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `09a_object_geometry_sensitivity.md` | 실제 물체 개입 | 이전 Full 정책의 Graph 반응은 있으나 방향·크기가 불완전. 현재 BC 모델의 대응 실험은 10-B 추가본. |
| `09b_readout_geometry_probe.md` | Frozen probe | 이전 GNN Readout의 부분적인 위치 decodability와 제한된 일반화. 정보 소멸을 증명하지는 않는다. |
| `09c_persistent_geometry_graph_benchmark.md` | Geometry/Encoder 비교 | Robot filter·multi-view·persistence·진단 Encoder 비교. Peg tracking drift와 Wrist Box 오염으로 단일 우승 없음. |
| `09c_legacy_fps_extension.md` | 100-update 파일럿 | Erosion→SOR→FPS 원형 재현. 표면 정확도와 경계 trade-off. 09-C/D의 2,000-update 결과와 직접 순위 비교 금지. |
| `09d_plan_a_geometry_suitability.md` | CONDITIONAL | C3 current-only 및 SF/Legacy/Hybrid 비교. Hybrid는 시간 누적이 아님. 당시 policy BC·rollout은 미평가. |
| `09e_plan_b_box_persistence.md` | NO-GO | Box-only B1/B2의 Geometry Gate 실패. Coverage/rim·이동 대응 제약으로 후속 GNN/probe/Final 미실행. |

### 8. 기존 Plan A BC와 당시 실패 메커니즘

| 원문 경로 | 종류 | 무엇을 확인할 때 읽나 / 핵심 판단 |
|---|---|---|
| `10_plan_a_cached_graph_bc.md` | 최종 결과·복구 완료 보고 | E0/E1 각11,487 updates, 중간 각54회, 복구 Test48회. 네 조건 Insertion0/12. 최적화·운영 결과도 포함. |
| `10b_policy_failure_diagnosis.md` | 해당 E0/E1 추가 인과 진단 | Expert-state 오차, 약한 geometry correction, k0 feedback 증폭. 이 보고서 시점에는 GNN 표현과 GRU/head 활용이 미분리였으며, 후속 C1/C2는 아래9절. |

### 9. Task 10-C~F — 표현·학습 신호·Reach와 복구

근거 수준은 해당 질문의 실행 조건에 한정한다. `CONFIRMED`도 이 아카이브 작업에서 실험을 재실행했다는 뜻은 아니다.

| Task / 질문 | 실제 실행 범위와 핵심 판단 | 근거 수준 | ZIP 내부 원문 |
|---|---|---|---|
| 10-C1: 학습된 표현이 Peg 위치·이동을 보존하나? | 네 BC checkpoint의 Expert13,452-frame 동결 추출; 별도 dense09D Train8/Val4와 held-out30003 paired22/secondary36 probe. Raw에는 이동 신호가 있으나 learned downstream의 paired 복원은 불안정하다. 원본 Train96 actor-GT probe와 새 BC는 미실행이다. | SUPPORTED; 모듈별 원인 INCONCLUSIVE | `10c1_frozen_representation_probe.md` |
| 10-C2: 조건을 맞추면 Geometry에 따른 Expert 보정 신호를 확인할 수 있나? | Train96/Val12 총13,452 Graph를 감사했으나 Robot·이력·Peg/Box 품질 조건을 동시에 만족한 cross-episode pair는 양쪽 모두0개다. 신호 부재와 표현 문제를 분리할 수 없으며 새 학습·policy forward는 없다. | INCONCLUSIVE | `10c2_learning_signal_audit.md` |
| 10-D: 동일 초기 Robot의 Geometry 다양성만 늘리면 Reach가 개선되나? | 수집·캐시·A/B 동일525,090-timestep 학습·live 평가 완료. 미학습 대각선 Reach는 A/B 모두0/6; B의 보정 개선은 미확인이다. Parent 노출까지 동일한 단일요인 비교는 아니며 Grasp/Insertion은 미평가다. | 실행 CONFIRMED; 개선 INCONCLUSIVE | `10d_geometry_conditioned_reach_bc.md` |
| 10-E: 기존 GT38D GRU는 같은 조건에서 제어할 수 있나? | 기존 Clean seed0/1/2로54회 실행: Reach13/54, Insertion35/54. Graph와 기록된 물리 필드는 맞췄지만 controller/full simulator snapshot 동등성은 미확인이다. 엄격한 full-state paired 분모는0이며 Reach와 full-task 성공을 섞지 않는다. | CONFIRMED / COMPLETED_WITH_PAIRING_LIMITATION | `10e_gt_gru_vs_graph_reach.md` |
| 10-F 초기: 실제 ±5mm 교란을 만들 수 있나? | 완료6 branch 중 잔차<1mm는2/6; 물리 gate 실패로 중단했다. 이 시점의 본 수집·학습·평가는 미실행이며 BC의 음성 결과가 아니다. | CONFIRMED; 후속 정책 NOT_TESTED | `10f_corrective_reach_bc.md` |
| 10-F 복구: 보정 Demonstration과 추가 BC가 Recovery를 개선하나? | 후속 feedback 교란40/40 gate와 P/N40쌍 복귀를 검증하고 수집·캐시·R0/R1 각3,000-update 학습·평가까지 완료했다. 선택 checkpoint의 handoff/autonomous 모두0/6, nominal0/3; 일부 오차·안전 지표 개선을 성공률 개선으로 바꾸지 않는다. | 실행 CONFIRMED; 효과 INCONCLUSIVE | `10f_pilot_recovery_v2.md` |
| 10-F 동일 step 후속: checkpoint 차이 때문인가? | R0 step3000의15회를 추가 평가하고 R1 step3000의15회를 검증 후 재사용했다. 동일 step에서도 성공은0이며 handoff 오차의 일관된 R1 우위는 없다. 공개 Final seed의 후속 진단으로 새 blind test가 아니다. | CONFIRMED; 일반 우열 INCONCLUSIVE | `10f_matched_checkpoint_handoff_diagnosis.md` |

### 10. Task 10-F1~H — GT 사전학습·정책 비교·Handoff

| Task / 질문 | 실제 실행 범위와 핵심 판단 | 근거 수준 | ZIP 내부 원문 |
|---|---|---|---|
| 10-F1: GT 지도 GNN이 제어에 충분한 공간 정보를 복원하나? | GNN/State Head 사전학습과 독립 Final12 Parent/1,350 frame 완료. Peg15.291mm·Hole51.071mm 평균 오차와 부분 개선은 확인했으나 작은 이동 Δ 복원은 불충분하다. 이 Task 자체의 GRU/BC/closed-loop는 미실행이다. | SUPPORTED; GT 정책 입력 충분성 INCONCLUSIVE | `10f1_gt_supervised_gnn_pretraining_final.md` |
| F1 이전 준비·데이터 인계는 왜 차단됐나? | 최초 NOT_READY → GT sidecar 보완·212 Episode 재캐시 → 최종 사전학습으로 이어진 기록이다. 초기 대기/차단을 현재 전체 미완료로 읽지 않는다. | 단계별 기록; 최신은 위 최종본 | `10f1_gt_supervised_gnn_pretraining.md`, `10f1_task10f_gt_sidecar_handoff.md`, `10f1_task10f_dataset_ready_audit.md`, `10f1_gt_sidecar_remediation.md`, `10f1_direct_continuation_20261030.md` |
| 10-G: F1 초기화 뒤 A Frozen/B Joint/C Joint+Geometry의 BC는 성공하나? | 각11,487 updates(seed42), Development 선택 후 Final12씩36회 완료. Grasp/Lift는 A0/12·B4/12·C3/12, Insertion은 모두0/12다. 이전 Plan A E0/E1과 다른 checkpoint·데이터·조건이다. | CONFIRMED; 일반적 구조 우열 INCONCLUSIVE | `10g_f1_initialized_graph_recurrent_abc.md` |
| G의 miniature/실행 이력은 성능 증거인가? | mini_v3/mini_v5는 engineering fixture다. 실행 로그의 Final 진행 중 서술은 위 최종 보고서로 갱신된다. | engineering만 CONFIRMED | `10g_miniature_v3_integration_report.md`, `10g_miniature_integration_report.md`, `10g_pipeline_engineering_log.md` |
| G 실패가 모두 초기 인식 실패인가? | 98200은 초기 Peg/Box mask가 대체로 정상; 98202는 Peg 후보가 Null gate에서 탈락하고 Wrist 오배정은 최종 Graph에 진입하지 않았다. 98207은 초기 가시성이 제한됐다. 서로 다른 실패 사례를 하나의 원인으로 묶지 않는다. | 사례별 SUPPORTED | `10g_seed98200_recognition_diagnosis.md`, `10g_seed98202_initialization_diagnosis.md`, `10g_seed98202_peg_mask_diagnosis.md`, `10g_seed98207_initial_visibility_diagnosis.md` |
| G 학습 자료의 Main-visible 조건을 충족했나? | 320 logical episode 전수 감사에서 Train190/272·Val25/48이 가시성 gate 탈락이다. 기존 데이터·평가를 사후 제외하지 않고 후속 분포 정의의 근거로 남긴다. | CONFIRMED | `10g_initial_main_visibility_dataset_audit.md` |
| 10-H: Expert prefix가 B의 후반 제어를 회복시키나? | B step9000 동결, 독립12 Parent×E/B0/H1/H2=48회. Insertion E12/12·B0/H1 0/12·H2 6/12다. H2는 hole 앞105–142mm의 축/lateral 정렬 상태이며 접촉 직전 몇 mm만의 시험이 아니다. | CONFIRMED; 원인 분해 INCONCLUSIVE | `10h_phase_conditioned_handoff_diagnosis.md` |

### 11. Task 10-I — 데이터 준비와 Stage4A

| Task / 질문 | 실제 실행 범위와 핵심 판단 | 근거 수준 | ZIP 내부 원문 |
|---|---|---|---|
| Stage1: 초기 Main-visible 후보를 확보했나? | 실제 reset926회에서 최종 유효 고유 Parent300개를 확보하고 별도 replay 검증했다. 초기 상태 후보 풀이며 action0회, Expert 성공·학습 데이터 완료를 뜻하지 않는다. | CONFIRMED; Expert NOT_TESTED | `10i_stage1_main_visible_candidate_pool.md` |
| Stage1.5: 같은 후보 풀을 원격에 복원했나? | 원본 identity를 유지한 원격 mirror/인계를 완료했다. Stage2_READY는 정책 학습 완료가 아니다. | CONFIRMED | `task10i_stage1_mirror_recovery.md` |
| Stage2: 다양성을 고려한 R/D 데이터 선정을 끝냈나? | 기존 Train33/82·Val7/23 logical 자료와 신규300 후보를 검증해 분할·선정 계획을 동결했다. 이 Task 자체에서는 새 Expert trajectory를 만들지 않았다. | CONFIRMED; 수집은 후속 Stage3 | `10i_stage2_diversity_aware_dataset_selection.md` |
| Stage3: 승인된 Expert 데이터가 실제 존재하나? | R/D 각각96 Parent·272 logical, 공통Val16 Parent·48 logical을 수집·finalize했다. R/D는78 Parent·236 동일 logical source를 공유한다. 최초 선택87/90 Expert 성공과 reserve3개를 기록했으며 학습·정책 성능 검증은 아니다. | CONFIRMED | `10i_stage3_expert_dataset_collection.md` |
| Stage3 원자료·복원 의존성은 어디 있나? | 58576054의 raw/GT 존재를 확인했고, 후속 별도 요청으로 누락 F1 백업과 네 백업 의존성의 HF 복원 검증을 완료했다. 전체 과거 연구의 게시 완료와는 다르다. 이번 아카이브 작업에서는 접속·게시하지 않았다. | 기록상 CONFIRMED; 현재 원격 NOT_TESTED | `10i_stage3_presence_58576054_20261102.md`, `10i_stage3_hf_dependency_completion_20261102.md` |
| Stage4A: Perception 실패 조사·독립 검증이 완료됐나? | Development64/Internal96/Val48 source의13,150관측, Oracle·후보·보호 감사를 완료했으며 채택은 NO_GO다. Graph object recall/presence는 geometry 정확도나 policy 성공률과 다르다. | 실행 CONFIRMED; 채택 NO_GO | `10i_stage4a_final_report.md` |
| Stage4A 이전 복원·중간 기록과 최신 결론은? | 환경 복원과101-frame Mask/Graph 동등성 검증 뒤 전체 연구를 완료했다. 중간 IN_PROGRESS/재개 미실시 문구보다 위 최종본이 최신이다. | 단계별 CONFIRMED | `10i_stage4a_perception_failure_investigation.md`, `10i_stage4a_resumption_check_58576054.md`, `10i_stage4a_environment_restore_58576054.md`, `10i_stage4a_forward_equivalence_58576054.md` |

### 12. Task 10-I Stage4A-2 — 중지된 통합 후보와 완료된 국소 진단

**Stage 전체는 PAUSED_SCOPE_REVIEW이며 채택·Stage4B Handoff는 미완료다.** 아래 개별 진단 완료를 Stage 전체 완료로 확대하지 않는다. 상세 원문은 누적 실행 기록보다 정정·범위 검토 보고서를 먼저 읽는다.

| 질문 / 개별 Task | 실제 실행 범위와 핵심 판단 | 근거 수준 | ZIP 내부 원문 |
|---|---|---|---|
| DINO/DINOv2/SAM2 통합 후보의 실제 범위는? | Development36/88 Episode/5,845 frame 비교·Graph 등록 평가 완료; 안전 gate29 PASS/3 FAIL/3 INCONCLUSIVE. Internal B0 60/60, 후보38/60 완료와21-frame PARTIAL만 보존됐으며 독립 GT/paired 평가·신규 Final·Stage4B는 미실행이다. | 부분 실행 CONFIRMED; 채택 미완료 | `10i_stage4a2_paused_scope_review.md`; 누적 이력 `10i_stage4a2_dino_dinov2_sam2_integration_optimization.md` |
| Mask→Graph의 최초 손실은? | 지정4프레임에서 Box overlap 제거와 Peg cap128 손실을 확인했다. Main Peg는 t30 늦은 최초 초기화부터 Box였다. 안전한 Owner/patch는 입증하지 못했다. | 국소 CONFIRMED; 안전 수정 INCONCLUSIVE | `10i_stage4a2_mask_to_graph_surgical_fix.md`, `10i_stage4a2_box_semantic_ownership_fix.md` |
| 더 많은 점·구조점 보존이면 해결되나? | t6/t86 budget sweep과2×2 ablation에서 coverage 이득·trade-off를 측정했다. 6프레임 geometry 진단에서는 B256도 <3mm 위치1/6이며 t86 축오차는 개선되지 않았다. Hole 추정 실패와 FPS 이전 손실도 남는다. | 국소 CONFIRMED; 정렬 해결 NOT_SUPPORTED | `10i_stage4a2_point_budget_sensitivity.md`, `10i_stage4a2_structural_budget_2x2.md`, `10i_stage4a2_graph_geometric_precision.md`, `10i_stage4a2_stagewise_geometry_precision.md` |
| 초기 후보 재검증·두 Detector면 안전한 규칙이 있나? | 고정 P/N/C1의 Temporary SAM2·YOLO-World 진단에서 상충 증거를 확인했지만 NO_SAFE_RULE/TRADEOFF_UNRESOLVED다. 새 benchmark·Production 적용은 없다. | 국소 SUPPORTED; 채택 INCONCLUSIVE | `10i_stage4a2_initial_candidate_verification.md`, `10i_stage4a2_dual_grounding_feasibility.md` |
| Qwen 초기 실패를 Prompt로 바꿀 수 있나? | 같은3후보에서 OTHER/BOX/OTHER와 형식 실패가 기능적 역할 Prompt의 PEG/BOX/PEG로 교정됐다. 국소 성공이며 일반화·안전 통합을 의미하지 않는다. | 국소 CONFIRMED | `10i_stage4a2_qwen35_semantic_verifier_feasibility.md`, `10i_stage4a2_qwen35_functional_role_prompt.md` |
| Frozen Qwen verifier의 좁은 Semantic Gate는? | 24 Parent·192후보에서 PEG30/31·BOX30/30으로 gate는 PASS했으나 ROBOT73/77을 BOX로 오분류했다. GPU 단기 상주·연결 smoke와 실질 누락 감소·종합 안전성을 구분한다. | 좁은 gate CONFIRMED; Production NOT_SUPPORTED | `10i_stage4a2_qwen35_frozen_verifier_independent_evaluation.md` |
| Qwen이 DINO 없이 위치를 직접 만들 수 있나? | 24 Parent/96요청의 NF4 full-image 실험은 strict JSON2/96, GT지원 Peg1/56·Box0/54다. 상세 Prompt 재평가·정밀도3종도 각 고정 조건의 VLM_GROUNDING_INSUFFICIENT이며 모델 일반 능력의 판정은 아니다. | 고정 조건 CONFIRMED; DINO 대체 NOT_SUPPORTED | `10i_stage4a2_qwen35_full_image_grounding.md`, `10i_stage4a2_qwen35_grounding_implementation_audit.md`, `10i_stage4a2_qwen35_detailed_prompt_retry.md`, `10i_stage4a2_qwen35_precision_comparison.md` |
| Native512와 Prompt는 무엇을 개선했나? | 기존4 Parent·16 Camera-frame의 Qwen2×2에서512/P1 좌표 진단은 개선됐으나 정식 JSON0/16이다. DINO 원본128/512 비교는 Peg/Box 각각6/16→8/16으로 국소 이득을 보였다. | 파일럿 SUPPORTED; 일반화 INCONCLUSIVE | `10i_stage4a2_qwen35_native_resolution_prompt.md`, `10i_stage4a2_dino_native_resolution.md` |
| DINO 후보를 Qwen으로 거르면 좋아지나? | 같은4 Parent/251후보에서 Peg10/16→8/16, Box8/16→11/16이다. 후기 정답 Peg의 BOX 오분류가 손실을 만들었고 요청 간 컨텍스트 누적은 확인되지 않았다. 초기 RGB 추가298요청도 후기 Peg0/8·Box3/8로 개선이 없었다. | 고정 조건 CONFIRMED | `10i_stage4a2_dino_qwen_candidate_verifier.md`, `10i_stage4a2_dino_qwen_verifier_failure_audit.md`, `10i_stage4a2_qwen35_temporal_reference.md` |
| 모델의 기본 인식 불능인가, 외형 Prompt 효과인가? | 알려진9후보에서 역할5/9→외형9/9; 후속 Parent-held-out15/60이미지·986후보에서 Peg31/60→37/60, Box33/58→45/58이다. Peg 차이 CI는0을 포함하고 후기 Main Peg0/15여서 일반적으로 안정적인 선택기는 아니다. | 국소 CONFIRMED; 제한적 개선 SUPPORTED | `10i_stage4a2_qwen35_vla_research_and_perception_diagnostic.md`, `10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md` |
| 지금 관측 경로를 학습에 연결할 근거가 충분한가? | 동시 Box 성공30/58, 후보 존재52/58→정답 승인36/58→선택30/58로 verifier/selection 손실이 남는다. 후속326회 거부 진단은 Peg 오분류와 Box proposal의 주물체·Query 충돌을 구분했다. 이 시점의 SAM2 미평가는 아래 후속 Mask 결과로 갱신하며 policy 성공은 여전히 미검증이다. | 진단 SUPPORTED; readiness INCONCLUSIVE | `10i_stage4a2_perception_readiness_bottleneck.md`, `10i_stage4a2_qwen35_rejection_mechanism.md` |
| SAM2 Mask를 보여주거나 초기 정체를 추적하면? | 기존15 Parent/60장·986후보에서 Mask 표시C의 동시 Mask 성공24/58은 A/B27/58보다 개선되지 않았다. 추적8 Parent의 동일 t80에서는 현재 분류2/16→초기화 후 추적11/16, 전체623/801이지만 Robot 오초기화81프레임 지속과 Wrist 실패가 남는다. 경로·이력·Codec이 다른 비교이며 Hole/Graph/학습·closed-loop는 미평가다. | 국소 결과 CONFIRMED; 추적 방향 SUPPORTED; 채택 INCONCLUSIVE | `10i_stage4a2_sam2_mask_verifier_tracking.md` |

### 13. Predicted State38와 Task 10-J

| Task / 질문 | 실제 실행 범위와 핵심 판단 | 근거 수준 | ZIP 내부 원문 |
|---|---|---|---|
| 10-I Predicted State38: Frozen F1 출력으로 새 GRU를 학습하면? | Frozen F1+새38D GRU의11,487-update 학습, Dev24회·선택step9000 뒤 Final120회 완료(11-02 23:23 KST). S/R/J의 자율·H1 insertion은 모두0/12; H2는 S6/12·R/J2/12이며 S−R gained5/lost1, nominal paired p=.21875다. H2 첫10-step lateral drift는 S6.28mm·R13.31mm지만12 Parent/한 checkpoint의 일반적 우위나 단일 실패 원인은 미확정이다. | 실행·표본 결과 CONFIRMED; 일반 우위 INCONCLUSIVE | 최신 `10i_predicted_state38_graph_recurrent_bc.md`; 과거 중간 `10i_progress_20261102_1733.md`; 시작 `10i_predicted_state38_graph_recurrent_bc_launch.md`; 오류 `10i_predicted_state38_bc_failure_20261102.md`; 재개 `10i_predicted_state38_bc_resume_20261102.md` |
| 10-J: DINO Prompt만 바꾸면 기존 실패 후보가 복구되나? | 평가 가능한13이미지/25물체 관측에서 탐색8/17→13/17, 동결 확인2/8→4/8이다. PARTIAL_INPUT_GT이며98202/t0의 exact segmentation 부재 장면은 정량 분모에서 제외했다. 런타임 Prompt 변경·정책 개선은 없다. | 제한적 SUPPORTED / FOLLOWUP_WORTHWHILE | `10j_dino_prompt_feasibility.md` |

### 14. 추가 세부 원문·운영 근거 찾기

아래는 실험 성능과 별개인 상세/운영 기록이다. 게시·복원·환경 준비를 새 연구 실험 완료로 세지 않는다.

| 확인할 내용 | ZIP 내부 원문 / 해석 |
|---|---|
| 10-B 추가 분석 원문 | `10b_policy_failure_diagnosis_extension.md`, `10b_details/extension/causal_findings.md`, `10b_details/extension/phase3_findings.md`, `10b_details/extension/phase3_t0_findings.md`, `10b_details/extension/phase4_findings.md`. 종합 최신 판단은 기존 `10b_policy_failure_diagnosis.md`와 함께 읽는다. |
| 10-B 최초 세부 Findings | `10b_details/initial/phase1_findings.md`, `10b_details/initial/phase2_findings.md`, `10b_details/initial/phase3_findings.md`, `10b_details/initial/phase4_findings.md`. 초기 Phase3/4 미실행은 당시 범위다. 최초 보고서의 동일 바이트 사본은 `history/10b_policy_failure_diagnosis_phase1_2.md` 한 곳에 보존했다. |
| 10-A 복구와 별개 운영 기록 | `10_plan_a_event_65c4ff249b1a39c5c91911ef.md`는 이미 연구 완료한 뒤 monitor 통신 오류가 해소된 기록이다. `10_first_frames_20261029.md`는 첫 프레임 추출이며 새 정책 평가가 아니다. |
| 데이터 미디어 위치 | `10g_training_video_preview_export.md`에서96개 원본 episode의 MP4·index 위치를 찾는다. 영상은 이번 ZIP에 새로 포함하지 않았다. |
| 과거 게시·의존성 정정 | `10cd_commit_and_hf_publication_20261030.md`, `hf_unpublished_audit_20261031.md`, `hf_remaining_publication_20261031.md`. 마지막 문서의 IN_PROGRESS를 전체 게시 완료로 읽지 않는다. Stage3 의존성 범위의 후속 완료만 `10i_stage3_hf_dependency_completion_20261102.md`가 갱신한다. |
| 원격 이전·공간·새 환경 준비 | `vast57987772_safe_disk_cleanup.md`, `vast57987772_to_58576054_transfer_verification.md`, `10i_instance_setup_58752018_20261102.md`. 당시 검증 범위이며 현재 원격 실행 상태를 보장하지 않는다. |
| 구현 연혁 | `10f1_to_stage4a2_commit_review_20261102.md`는 기존13커밋의 정리다. 연구 동기를 이 구현 연혁이나 BC 실패로 대체하지 않는다. |

## 원자료·실험 상태를 찾는 위치

- `evidence/archive_update_20261103/experiment_inventory.json`: 로컬에서 읽은 기존 ExperimentLogger metadata의 위치·SHA·run_id·원래 status·시각만 담았다. `finished`는 해당 실행의 종료 기록이며 과학적 성공이나 전체 Task 완료가 아니다. 남아 있는 `running` snapshot을 완료로 승격하지 않았다.
- `evidence/archive_update_20261103/artifact_locations.json`: 보고서의 상대 링크를 원본 저장소 위치로 해석한 locator다. 로컬 존재 여부를 별도로 기록하며, 미존재가 원격 소실을 뜻하지 않는다. 보고서 본문의 절대 원격 root·고정 HF revision도 함께 사용한다.
- 새 대형 checkpoint·dataset·영상·전체 중간 로그는 넣지 않았다. 기존 ZIP의 초기 오류 전달본은 그대로 유지했다. 각 보고서는 원래 바이트를 보존하므로 본문의 상대 artifact 링크가 ZIP 안에서 열리지 않을 수 있다. **인덱스의 원문 경로와 보고서 외부 artifact 경로를 구분**한다.
- Predicted State38은 완료 보고서와 로컬 전달26개 파일의 SHA readback, 자율48회+Handoff72회의 CSV를 대조했다. 선별된 작은 완료 근거는 `evidence/task10i_predicted_state38_final_20261102/`에 있으며 MP4는 원격에 남아 있다. 원격 재접속·전체 영상 재생이나 실험 재실행은 하지 않았다.


## 원래 문제의식·팀 자료·과거 컨텍스트

| ZIP 내부 경로 | 역할 / 주의점 |
|---|---|
| `context_sources/VLA_GNN_Recurrent_연구_스토리_v0.md` | **원래 연구 동기의 우선 원문**. Chunking·stale observation·Slow/Fast·response/latency 가설. 현재 성능 증거는 아님 |
| `context_sources/VLA_GNN_Recurrent_모델_설계_v0.md` | 초기 전체 아키텍처. VLM·task node·adaptive re-anchor·큰 hidden/층수를 현재 구현으로 오인하지 않음 |
| `context_sources/SHARED_TECHNICAL_FOUNDATION.md` | Graph/recurrence 공통 개념. 개인 연구 확장이나 후보를 팀의 확정 범위로 가져오지 않음 |
| `context_sources/1023-visual-prototyping-result.txt` | 별도 Vision v1 결과. Tracking/point-cloud 속도와 vision+TCP scripted 성공을 learned BC 성능과 구분 |
| `context_sources/#-1주차-—-세부-팀별-구현-모듈-정의.txt` | 초기 Slow/Fast 역할, Visual/Graph/Recurrent 모듈 분담 |
| `context_sources/2,3주차-해야-할-일.txt` | GT adapter로 각 모듈을 독립 prototype한 계획. GT prototype과 현재 estimated student 입력을 구분 |
| `context_sources/pasted.txt` | 초기 주차별 일정표. 현재 진행 상태를 판단하는 자료가 아님 |
| `context_sources/PROJECT_INSTRUCTIONS.txt` | 교체 전 지침의 역사적 사본. 현재 세 파일 구성과 새 팀 결정이 우선 |
| `context_sources/SETUP_README.txt` | 이전 프로젝트 설치 안내. 현재의 추가 필수 파일을 뜻하지 않음 |
| `context_sources/history/TEAM_CONTEXT_2026-10-15.md` | 초기 컨텍스트 백업. 현재 구현·진행 상태에 우선하지 않음 |
| `context_sources/history/TEAM_CONTEXT_2026-10-29_before_bundle.md` | 이번 통합 직전 수정본. 당시 ‘10-A 진행 중’은 최신 최종 보고로 갱신됨 |
| `context_sources/history/research_handoff_2026-10-28.txt` | 과거 대화 인계문. 09-D/Plan B 미완료 서술은 이후 보고서로 갱신됨 |
| `history/10b_policy_failure_diagnosis_phase1_2.md` | 10-B 최초 분석 원문. 추가 진단본과 바뀐 결론을 추적할 때만 우선 참조 |

## 복구 전 BC 원자료 전달본

`evidence/plan_a_bc_20261029T032648Z/` 아래에는 별도로 전달된 ZIP의 40개 파일을 원래 상대 구조와 바이트 그대로 보존했다. 전체 내용을 매번 읽지 말고 필요한 파일을 선택한다.

| 확인할 내용 | ZIP 내부 경로 |
|---|---|
| 최초 오류 traceback | `evidence/plan_a_bc_20261029T032648Z/failed_rollout_traceback.txt` |
| 실패 시점 pipeline 상태 | `evidence/plan_a_bc_20261029T032648Z/pipeline_state.json` |
| 초기 reset 조사 결과 | `evidence/plan_a_bc_20261029T032648Z/diagnostics/research_findings.md` |
| 원래 모델 선택·step | `evidence/plan_a_bc_20261029T032648Z/model_selection_frozen.json` |
| 원래 split/seed | `evidence/plan_a_bc_20261029T032648Z/seeds_manifest.json` |
| E0 중간 실제 rollout | `evidence/plan_a_bc_20261029T032648Z/E0/rollout_metrics.csv` |
| E1 중간 실제 rollout | `evidence/plan_a_bc_20261029T032648Z/E1/rollout_metrics.csv` |
| Offline 수치 | `evidence/plan_a_bc_20261029T032648Z/final_offline_metrics.json` |
| 학습 시간축 | `evidence/plan_a_bc_20261029T032648Z/learning_timeline.json` — 큰 파일이므로 필요한 키/구간만 읽기 |
| 실행 소스 목록·hash | `evidence/plan_a_bc_20261029T032648Z/execution_source_manifest.json` |
| 보존된 rollout 코드 | `evidence/plan_a_bc_20261029T032648Z/source/graph/plan_a_bc/rollout.py` |
| 당시 검증 상태 | `evidence/plan_a_bc_20261029T032648Z/verification.json` |

이 전달본의 reset 원인 미확정·Test0회 상태는 **이전 기록**이다. 후속 복원 방법과 최종 평가 상태는 `10_plan_a_cached_graph_bc.md`를 따른다. 전달본 README·manifest에 표시된 checkpoint/영상 위치가 이 ZIP 안의 실제 파일 존재를 뜻하지 않는다.

## 자료 포함 범위와 보존

- **기존2026-10-29 통합 패키지의 포함 내역:** 원래 ZIP의71개 Markdown과19개 tracking JSON을 보존했다. 이전 ZIP에만 있던 `06d_visual_stream_reuse.md`를 추가 복원해, 기존 시리즈 Markdown은 72개다. 동봉된 이름이 _plan.md/_request.md로 끝나는 문서 17개는 과거 계획으로 표시했다.
- 기존 통합본은 당시 최신10-B와 별도 이전10-B, 프로젝트 원문9개, 이전 컨텍스트/인계문, BC 초기 오류 전달본을 포함한다. 동일 바이트의 중복은 한 번만 보존하고 출처를 합쳤다.
- 이번 증분은 고유 연구 Markdown86개를 추가했다. 동일 바이트10-B 초기본1개는 기존 경로에 출처만 합쳤다. 기존 연구 보고서의 같은 이름 버전 교체는0개다. 이전 인덱스·Manifest·README는 history에 보존하고 현재 Manifest와 안내만 갱신했다.
- 원문 내용은 수정하지 않았다. __MACOSX, ._* 및 .DS_Store 같은 filesystem metadata만 제외했다. 파일별 SHA256과 원래 출처는 `ARCHIVE_MANIFEST.json`을 따른다.
- 보고서 안의 repository·원격·상대 경로는 작성 당시 근거다. **모든 checkpoint·RGB-D·NPZ·영상·전체 코드가 포함된 재현 패키지는 아니다.** 없는 파일을 있는 것처럼 인용하거나 링크가 열린다고 가정하지 않는다.
- 현재 읽는 문서가 과거 명령/계획인 경우 자동 실행하지 않는다. 원문 검색·확인과 새로운 작업 승인은 구분한다.

## 갱신 규칙

새 보고서를 ZIP에 추가하고 이 인덱스의 해당 행과 최신 관계만 갱신한다. 같은 이름의 수정본은 이전 파일을 `history/`에 보존한다. 현재 결정·완료 상태가 바뀐 경우 이후 별도 대화에서 `TEAM_CONTEXT.md`의 해당 부분을 갱신한다. 이번 작업에서는 해당 파일을 수정하지 않았다. 인덱스를 전체 보고서의 복사본으로 키우거나 별도 상태 파일을 추가할 필요는 없다.
