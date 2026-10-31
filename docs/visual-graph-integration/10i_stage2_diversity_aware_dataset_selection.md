# Task10-I Stage 2 — Diversity-aware Initial State Dataset Selection

상태: **COMPLETED**. ExperimentLogger UUID `34d10bf652b344a38d54b7672dbfe0c0`. 본 결과는 **선정 계획**이며 신규 Expert action/trajectory가 있는 학습 데이터셋이 아니다. Stage 3는 자동 실행하지 않았다.

Engineering 이력: 첫 `/workspace/task10i_stage2_20261031/run`은 분포 summary의 `maximum` 출력 키 중복 오류로 BLOCKED를 보존했다. 수정은 분포 CSV 출력에만 적용했고 seed/weight/scale/selection은 변경하지 않았다. 회귀 테스트에 분포 출력 계약을 추가한 뒤 별도 Source/Output Root에서 전 단계를 재검증했다. 이전 선정 ID/순서와의 동등성은 별도 완료 감사로 확인한다.

## 1. 입력과 Source Identity — CONFIRMED

Vast.ai instance **57987772**에서 CPU NumPy/SciPy로 실행했다. Canonical `/workspace/task10i_stage1_20261031/verified_pool_v2`는 읽기 전용이다. Stage 1.5 `handoff/stage2_input_receipt.json`과 그 Artifact Index 전체를 검증했고, **8,514개 전체 파일**을 원본 path/size/SHA256과 다시 대조했다. Stage 1의 8,506 indexed files와 8개의 기존 bookkeeping 파일을 구분했다. Missing/Mismatch/Extra/Unreadable은 모두 0이다.

926행 aggregate와 300개 eligible initial NPZ, Descriptor, RGB-D, Visibility evidence를 실제로 읽고 Stage 1.5 fingerprint와 대조했다. 기존 Primary Gate, 1.5mm depth / 0.0001px frustum tolerance, Physical/Replayed PASS는 변경하지 않았다. 기존 JSON의 역사적 macOS 절대경로는 provenance이며 새 로더는 실제 remote pool 상대경로로 읽는다.

입력 [Identity](../../outputs/task10i/stage2/run/stage2_input_identity.json), [Receipt 검증](../../outputs/task10i/stage2/run/stage2_input_receipt_verified.json), [300개 Readback](../../outputs/task10i/stage2/run/input_candidate_readback.json), [Source 보존](../../outputs/task10i/stage2/run/source_preservation_audit.json), [실행 Source](../../outputs/task10i/stage2/run/execution_source_identity.json).

## 2. 기존 Train33 / Validation7 — CONFIRMED

Task10-G final release manifest, 실제 host metadata, dataset identity, 승인 Eligibility IDs, 320개 원본 HDF5 및 GT sidecar SHA를 대조했다. episode→parent seed→split→branch를 실제 자료로 연결했다. 기존 감사 NPZ에는 Hole GT가 없으므로 원본 GT sidecar의 `raw_hole_pose`, `raw_hole_radius`에서 복원했다. Peg/TCP/RGB-D/calibration을 저장 첫 프레임과 exact 대조하고, Parent 안 모든 branch의 초기 Peg/Hole/TCP/qpos/qvel/camera/image를 exact 비교했다. Box actor pose는 기존 raw GT에 없으며 추정하지 않았다. 거리 계산에는 실제 Hole frame을 사용한다.

Train 33개는 **82 logical episodes**, Validation 7개는 **23 logical episodes**다. 원본 272/48 중 190/25 visibility-failed episodes는 Stage 3 계획의 제거 inventory로만 기록했고 원본은 수정하지 않았다. Existing Final/Test는 사용하지 않았다. [Parent inventory](../../outputs/task10i/stage2/run/existing_valid_parent_inventory.json), [보존/제거 episode inventory](../../outputs/task10i/stage2/run/existing_episode_preservation_manifest.json), [원본 SHA 목록](../../outputs/task10i/stage2/run/existing_source_file_inventory.json).

## 3. 신규300 분할과 무작위 추출 — CONFIRMED

정렬된 ID에 PCG64 seed `1010202601`를 적용한 uniform permutation으로 Validation Cohort20 / Diagnostic Holdout60 / Training220을 먼저 동결했다. Validation 앞9는 Primary, 뒤11는 Reserve다. Validation20 전체와 Holdout60은 Train/scale/greedy에서 제외했다. Source snapshot은 그대로 UNASSIGNED이며 Stage2 manifest에만 새 역할을 부여했다.

Training220에서 Common44와 잔여 Random19를 독립 uniform permutation으로 비복원 추출했다. 최종 신규63의 분포는 uniform subset63과 동등하다: 각 63-subset을 만드는 44/19 분할 수가 동일하고 각 순서의 확률도 같다. 가장 좋은 Monte Carlo 결과로 R을 바꾸지 않았다. [분할/전체 trace](../../outputs/task10i/stage2/run/candidate_pool_split_manifest.json), [RNG](../../outputs/task10i/stage2/run/random_seed_manifest.json), [추출 trace](../../outputs/task10i/stage2/run/sampling_trace.json).

## 4. Feature와 거리 계약 — CONFIRMED

GT는 Dataset construction/분석 전용이다. Student 입력·모델은 바꾸지 않았다. 모든 pose는 world xyz,wxyz, m; quaternion 정규화 후 sign-invariant SO(3) angle `2 acos(abs(dot(q1,q2)))`을 사용한다. 180° reversal은 π 거리이며 Red/White 끝을 합치지 않는다. Yaw는 [-π,π] 주기 진단이며 objective에 중복 추가하지 않는다.

| Group | 정의 | Weight |
| --- | --- | ---: |
| G1 | world peg−TCP XYZ / coordinate scale의 RMS | .30 |
| G2 | Peg world SO(3) geodesic / angular scale | .20 |
| G3 | 평균(position RMS, hole⁻¹·peg SO(3) / scale) | .25 |
| G4 | Full length, square thickness / scale의 RMS | .15 |
| G5 | 8개 OBB corner 평균 u,v /128 / scale의 RMS | .10 |

최종 거리 `d=Σ weight_g·distance_g`. Group 차원 수로 가중치가 커지지 않도록 vector RMS를 쓴다. G3 안 position/orientation은 각 1/2이다. 원점 이동에 무관한 world displacement이며 TCP local frame으로 혼합하지 않는다. G5는 mask centroid/actor center와 다른 명시적 OBB-corner 평균이다.

Scale은 기존 Train33+Train220 **253개만** fit한다. Vector는 coordinate IQR, rotation은 253개 내 upper-triangle pair-angle IQR이다. Floor는 위치1cm, length1cm, thickness2mm, angle10°, image4px이다. Nearly constant axis에도 이 floor를 사용한다. 실제 clearance가 thickness/2+3mm임을 모든 state에서 검사했으므로 hole radius를 독립 가중하지 않는다. scalar TCP distance·Yaw·image extent·boundary margin도 중복 또는 품질 진단으로만 둔다. 이것은 사전 정의 heuristic이며 검증된 최적 weight가 아니다.

[Schema](../../outputs/task10i/stage2/run/feature_schema.json), [Scale 실제 값과 fit IDs](../../outputs/task10i/stage2/run/feature_normalization.json), [Distance](../../outputs/task10i/stage2/run/distance_protocol_frozen.json), [Selection](../../outputs/task10i/stage2/run/selection_protocol_frozen.json). Holdout positive-control 차단·q/−q·wraparound·방향 반전·degenerate floor·miniature E2E는 [테스트 로그](../../outputs/task10i/stage2/run/contract_tests.log)를 따른다.

## 5. R/D/Validation 선정 — CONFIRMED

R=기존33+Common44+Random19=**96**. D=동일 기존33+동일 Common44+Facility-location19=**96**. Shared Validation=기존7+Random9=**16**. D는 Training220의 `J(S)=mean_x min_s d(x,s)` 감소량을 매번 최대화했다. Pure farthest-point는 사용하지 않았다. Float64와 ID 정렬 tie-break로 전체 greedy continuation을 독립 두 번 실행하여 exact 일치했다.

실제 R/D 교집합 **78**, R-exclusive **18**, D-exclusive **18**. 추가19 중 **1개**가 겹친다. R tail을 D 후보에서 배제하지 않았다. 신규 Expert 수집 고유 Parent union은 Validation9 포함 **90개**다(상한91).

[R manifest](../../outputs/task10i/stage2/run/dataset_R_parent_manifest.json), [D manifest](../../outputs/task10i/stage2/run/dataset_D_parent_manifest.json), [동일 Validation](../../outputs/task10i/stage2/run/shared_validation_parent_manifest.json), [Hash freeze](../../outputs/task10i/stage2/run/selection_identity.json), [전체 greedy 순서](../../outputs/task10i/stage2/run/coverage_selection_order.json), [공통/차이 감사](../../outputs/task10i/stage2/run/selection_overlap_audit.json).

## 6. Coverage 분석

### Training Candidate220

| dataset | n | mean | median | p90 | p95 | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| existing33 | 220 | 0.429777 | 0.429303 | 0.54995 | 0.587937 | 0.691092 |
| base77 | 220 | 0.303793 | 0.354593 | 0.467144 | 0.493459 | 0.602347 |
| R | 220 | 0.266372 | 0.331853 | 0.446726 | 0.476141 | 0.602347 |
| D | 220 | 0.249332 | 0.321596 | 0.411577 | 0.440338 | 0.500637 |

기존33 대비 mean 감소는 R **38.02%**, D **41.99%**다. D의 R 대비 mean 감소는 **6.40%**다. Greedy 자체 objective에서의 개선은 독립 일반화 증거가 아니다. p90/p95/max는 표의 실제 값으로 별도 해석한다.

### 독립 Diagnostic Holdout60

| dataset | n | mean | median | p90 | p95 | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| existing33 | 60 | 0.412119 | 0.39433 | 0.560701 | 0.566366 | 0.730803 |
| base77 | 60 | 0.375171 | 0.368505 | 0.475431 | 0.510871 | 0.586552 |
| R | 60 | 0.361772 | 0.364346 | 0.453996 | 0.473598 | 0.508257 |
| D | 60 | 0.367467 | 0.364346 | 0.457905 | 0.475047 | 0.567599 |

Holdout에서 D의 R 대비 mean 감소는 **-1.57%**다. 판정은 **INCONCLUSIVE**: 이 60개 초기 상태의 근접 Coverage에 한정한다. Tail 개선의 유무는 mean과 별개다. Holdout은 최종 Policy blind test가 아니며 어떤 결과도 seed/weight/algorithm 재선정에 사용하지 않았다.

[Training metrics](../../outputs/task10i/stage2/run/random_vs_diversity_coverage.csv), [Holdout metrics](../../outputs/task10i/stage2/run/coverage_holdout_metrics.csv), [Group별 nearest 및 total-nearest에 대한 group 값](../../outputs/task10i/stage2/run/feature_group_coverage.csv), [개별 nearest parent](../../outputs/task10i/stage2/run/nearest_set_distances.csv).

## 7. Conditional Random Monte Carlo 200회

Common44를 고정하고 잔여 후보에서 Random19를 200회 추출했다. 200개 독립 학습 실험이 아니다. Percentile은 random mean scores 중 해당 score 이하의 비율이며 낮을수록 더 나은 coverage다.

| cohort | mean | median | variance | p5 | p95 | R_percentile | D_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selection | 0.263499 | 0.263512 | 3.81638e-06 | 0.260437 | 0.266704 | 93.5 | 0 |
| holdout | 0.364247 | 0.364528 | 1.28212e-05 | 0.358162 | 0.369795 | 21 | 83 |

[200회 모든 metrics](../../outputs/task10i/stage2/run/random_monte_carlo_metrics.csv), [sample IDs trace](../../outputs/task10i/stage2/run/random_monte_carlo_sampling_trace.json). 동결 R을 유지했다.

## 8. 분포 왜곡·경계·극단 상태 감사

Candidate300은 native random reset을 고정 Main visibility/physical/replay 조건으로 제한한 경험 분포다. R/D는 기존33을 포함하므로 300 신규만의 분포와 완전히 같을 필요는 없다. R/D 전체와 각 추가19, Common44를 따로 비교했다. Peg XYZ, periodic yaw/SO3 pair angle, length/thickness/radius, peg−hole, peg−TCP, image center/extent, margin을 기술 통계·KS statistic·Wasserstein distance로 보존했다. 중복 pair/부분집합의 의존성을 감안해 유의성 p-value를 주장하지 않는다.

| population | n | minimum | mean | median | fraction_below_0.1px | fraction_below_1px | fraction_below_4px | fraction_below_8px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate300 | 300 | 0.084652 | 18.5424 | 19.7605 | 0.00333333 | 0.01 | 0.0566667 | 0.17 |
| existing33 | 33 | 0.861119 | 19.1158 | 19.2457 | 0 | 0.030303 | 0.0909091 | 0.0909091 |
| R_tail19 | 19 | 1.77446 | 15.8606 | 17.2716 | 0 | 0 | 0.0526316 | 0.157895 |
| D_tail19 | 19 | 5.83355 | 19.1361 | 20.0536 | 0 | 0 | 0 | 0.105263 |
| R | 96 | 0.861119 | 19.333 | 19.9118 | 0 | 0.0104167 | 0.0520833 | 0.104167 |
| D | 96 | 0.861119 | 19.9813 | 20.5331 | 0 | 0.0104167 | 0.0416667 | 0.09375 |

Training253의 Peg XYZ/size p5/p95 밖인 축이 하나라도 있는 Parent를 극단 위치·크기 diagnostic으로 미리 정의했다. Candidate300 **37.3%**, R tail19 **26.3%**, D tail19 **42.1%**, R 전체 **37.5%**, D 전체 **40.6%**다. D tail만과 전체를 혼동하지 않는다. Margin은 objective에 포함하지 않았다.

Coverage 개선만으로 ‘분포 왜곡 없음’을 확정하지 않는다. **INCONCLUSIVE**: 이 감사는 측정한 주변 분포·경계·SO3 통계 범위의 집중도를 보여주며, 통계적 동등성이나 고차원 분포 보존을 입증하지 않는다. 분포 차이가 정책 성능을 악화하는지는 **NOT_TESTED**다. [분포 summary](../../outputs/task10i/stage2/run/initial_distribution_summary.csv), [shift](../../outputs/task10i/stage2/run/distribution_shift_metrics.csv), [boundary](../../outputs/task10i/stage2/run/boundary_margin_comparison.csv), [extremes](../../outputs/task10i/stage2/run/extreme_distribution_audit.json).

## 9. 누출·중복 — CONFIRMED

Train/Validation exact ID overlap0, Train/Holdout0, Validation/Holdout0. 기존 split 재할당0, Final/Test 사용0, candidate ID/seed 충돌0. 신규 vs 기존33/7, R vs D, 각각 Train vs Validation/Holdout의 Peg/Hole/TCP pose·shape·qpos/qvel을 실제 비교했다. R/D 공통 Parent의 exact overlap은 의도된 것이다. Near 기준은 위치1mm, orientation1°, half-size/radius0.1mm, robot qpos0.001이다. 서로 다른 Split의 동일 초기 물리 상태는 차단한다. Near만으로 GT 누출을 단정하거나 후보를 제거하지 않는다.

| comparison | pairs | exact_same_parent | exact_distinct_parent | near_distinct_parent |
| --- | --- | --- | --- | --- |
| new_vs_existing_train | 9900 | 0 | 0 | 0 |
| new_vs_existing_validation | 2100 | 0 | 0 | 0 |
| R_vs_D | 9216 | 78 | 0 | 0 |
| R_train_vs_validation | 1536 | 0 | 0 | 0 |
| D_train_vs_validation | 1536 | 0 | 0 | 0 |
| R_train_vs_holdout | 5760 | 0 | 0 | 0 |
| D_train_vs_holdout | 5760 | 0 | 0 | 0 |

[실제 거리·차원 CSV](../../outputs/task10i/stage2/run/near_duplicate_audit.csv), [duplicate summary](../../outputs/task10i/stage2/run/physical_duplicate_summary.json). CSV는 exact/near 전체와 각 left의 가장 가까운 physical threshold-ratio 상대를 기록한다. 전체 pair 수를 독립 표본 수로 해석하지 않는다.

## 10. Reserve와 Stage 3 Handoff

Validation11, Common176, R157, D157의 순서를 동결했다. Common/R은 별도 seed의 uniform order, D는 같은 greedy trajectory의 20번째 이후 continuation이다. Reserve는 역할 간 겹칠 수 있으며 초기 선정이나 미래 성공을 의미하지 않는다. Validation은 자기 Cohort만 사용한다. Common은 현재 R/D union에 없는 ID를 함께 대체한다. R/D tail은 자기 dataset에 없는 상대 dataset ID를 공유할 수 있다. 실패 Parent는 모든 queue에서 제외하고 skip 사유를 기록한다. 충돌 처리 순서는 Validation→Common→R→D, 각 initial slot rank/ID 순이다. 원래 실패 상태를 삭제하지 않고 모든 attempted/failed/replacement/final ID와 coverage 변화를 새 버전으로 추적한다. Exhaustion은 차단하며 임의 seed를 보충하지 않는다. 현재 실제 대체는 **0회**다.

[reserve protocol](../../outputs/task10i/stage2/run/stage3_collection_reserve_protocol.json), [Validation11](../../outputs/task10i/stage2/run/validation_reserve.csv), [Common](../../outputs/task10i/stage2/run/common_reserve.csv), [R](../../outputs/task10i/stage2/run/random_reserve.csv), [D](../../outputs/task10i/stage2/run/coverage_reserve.csv), [고유 수집 union](../../outputs/task10i/stage2/run/stage3_parent_collection_union.json), [Stage3 input receipt](../../outputs/task10i/stage2/run/stage3_input_receipt.json).

## 11. Logical Episode 구성과 Stage 3 검증 과제

| split | original_logical | preserved_logical | new_parent_count | logical_deficit | target_logical |
| --- | --- | --- | --- | --- | --- |
| train | 272 | 82 | 63 | 190 | 272 |
| validation | 48 | 23 | 9 | 25 | 48 |

기존 full-task/corrective source 및 branch별 비율과 모든 실패 parent의 실제 branch inventory는 [수집 계획](../../outputs/task10i/stage2/run/stage3_demonstration_plan.json)에 있다. 목표272/48과 preserved82/23의 차이는190/25 logical episodes다. 새 Parent63/9를 선정한 사실만으로 그 수가 복원되지 않는다. 실패한 기존 Train63/Validation9 parent를 ID 정렬하여 신규 slot에 대응한 **Parent별 branch 수·유형 template**도 저장했다. Common44는 같은 앞44 template, R/D tail은 같은 뒤19 template를 각자 계승한다. Tail overlap이 서로 다른 template를 받으면 수집 union과 dataset별 포함을 별도 기록한다. 이 계획의 logical deficit는 각각190/190/25이며 성공한 episode가 있다는 뜻은 아니다. 모든 신규 Parent의 Full-task 성공 확인이 선행되어야 한다.

Collector에서는 prefix/disturbance supervision=false, recovery=true다. Task10G BC는 승인 nominal의 actions/robot/timestep과 exact 일치한 prefix만 재활성화하고 disturbance는 항상 false다. Stage3는 이 원본 계약을 확인해야 한다. Parent마다 동일한 임의 branch 수를 만들지 않았다.

Stage3는 실제 Expert 성공, Full-task/Corrective lineage, copied-prefix·supervision mask·action/RGB-D/GT clock 정합, reset(seed)+snapshot readback, 실패 attrition과 실제 collection 후 coverage를 검증해야 한다. 고정33/7이 무효가 되는 경우 silent replacement하지 않고 차단한다. 새 Final Evaluation은 이300과 기존 Final/Test에 없는 별도 seed가 필요하다.

## 12. 연구 질문과 해석 범위

- Q1/Q2: 기존33과 신규300의 실제 위치·방향·크기·상대관계 범위는 [분포 summary](../../outputs/task10i/stage2/run/initial_distribution_summary.csv)와 아래 그림에 있다. 신규 후보가 기존 nearest distance를 줄일 수 있는 미대표 영역을 제공함은 measured selection/holdout baseline로 확인한다.
- Q3/Q4: 고정 R/D의 기존33 대비 변화와 독립 Holdout 차이는 6절의 수치 범위에서만 해석한다.
- Q5/Q6: 분포 동일성은 INCONCLUSIVE다. 극단/작은 margin 비율과 orientation 통계를 별도 제시했고 margin 최적화는 하지 않았다.
- Q7: ID·seed·physical state·split 감사 범위에서 계획 identity는 CONFIRMED다.
- Q8: 고정 reserve와 충돌/실패 기록 규칙은 CONFIRMED다. 실제 대체 후 성공/coverage는 NOT_TESTED다.

**NOT_TESTED:** 실제 Expert 성공률·action 품질·정책 성공률·성능상 악영향. Simulator reset/action, GPU PhysX/CUDA/MPS inference, DINO/SAM2, Graph recache, GNN/GRU 학습·closed-loop 평가는 실행하지 않았다. Linux CPU 저장자료 분석만 수행했고 로컬 simulator도 실행하지 않았다. 기존 Task10-H source/process/checkpoint를 수정하지 않았다. 공통 ExperimentLogger를 disabled transport로 재사용했고 외부 publishing·Git commit·새 인스턴스 생성은 하지 않았다.

## 13. 그림과 이미지 근거

[전체 Gallery](../../outputs/task10i/stage2/delivery/gallery.html)는 원본 Main RGB와 별도로 그린 Peg OBB overlay, Descriptor, Visibility evidence, original snapshot/source를 연결한다. Stage1 annotated preview를 raw로 사용하지 않았다.

### Peg world XY (m)

![Peg world XY (m)](../../outputs/task10i/stage2/run/figures/peg_xy.png)

### Peg orientation distribution

![Peg orientation distribution](../../outputs/task10i/stage2/run/figures/peg_yaw.png)

### Peg full length/thickness (m)

![Peg full length/thickness (m)](../../outputs/task10i/stage2/run/figures/peg_shape.png)

### Peg minus hole world XY (m)

![Peg minus hole world XY (m)](../../outputs/task10i/stage2/run/figures/peg_hole.png)

### Peg minus TCP world XY (m)

![Peg minus TCP world XY (m)](../../outputs/task10i/stage2/run/figures/tcp_peg.png)

### Mean OBB-corner center (image fraction)

![Mean OBB-corner center (image fraction)](../../outputs/task10i/stage2/run/figures/main_center.png)

### selection nearest-set coverage

![selection nearest-set coverage](../../outputs/task10i/stage2/run/figures/coverage_selection.png)

### holdout nearest-set coverage

![holdout nearest-set coverage](../../outputs/task10i/stage2/run/figures/coverage_holdout.png)

### Nearest distance distributions

![Nearest distance distributions](../../outputs/task10i/stage2/run/figures/nearest_ecdf.png)

### Conditional Random19 Monte Carlo (200 repeats)

![Conditional Random19 Monte Carlo (200 repeats)](../../outputs/task10i/stage2/delivery/random_mc_labeled.png)

### Boundary margin distribution

![Boundary margin distribution](../../outputs/task10i/stage2/run/figures/boundary_margin.png)

### Selected parents: shared and exclusive

![Selected parents: shared and exclusive](../../outputs/task10i/stage2/run/figures/selection_overlap.png)

## 14. 최종 검증

[Artifact Index](../../outputs/task10i/stage2/run/artifact_index.json), [Final Verification](../../outputs/task10i/stage2/run/final_verification.json), [Stage3 Receipt](../../outputs/task10i/stage2/run/stage3_input_receipt.json). 선정·Config hash를 다시 읽고 모든 JSON/CSV/PNG와 SHA/size를 검증했다. 단계별 atomic proof와 progress를 보존했다. 완료된 실행은 재선정 없이 readback만 수행한다. 미완료 resume는 동일 source/protocol에서 deterministic stage를 재검증하며 frozen 결과가 달라지면 차단한다.

## 15. 독립 완료 감사와 최종 해석

원격 실제 실행 Root는 `/workspace/task10i_stage2_20261031_v2/run`, UUID `34d10bf652b344a38d54b7672dbfe0c0`다. [독립 원격 감사](../../outputs/task10i/stage2/remote_execution/independent_completion_audit_v2.json)는 원본340 Parent, 전체8,514 Canonical 파일, 764개 불변 Stage2 산출물, 별도 프로세스의 Greedy176 전체 trace exact 재현, Coverage와 Monte Carlo200회, Reserve 순서 및 Parent별 branch budget190/190/25를 확인했다. 첫 실패 Run의 선정 Identity/Order/Scale/Split/CSV는 새 Run과 SHA256 exact 일치한다.

독립 SO(3) relative Rotation 구현의 objective와 frozen 구현은 최대 **1.39×10⁻¹⁷** 차이를 보인 Reserve 후반 동률 후보가 있다(rank95/129). 같은 원격 float64 구현·Runtime에서는 전체 순서가 exact 재현된다. macOS의 수치 재추출은 10⁻¹² 이내이고, 늦은 Reserve tie의 bitwise 순서까지 cross-platform 일치를 주장하지 않는다. Stage3는 다시 계산하지 않고 저장된 Queue를 사용한다. 최종 D19에는 이 동률 차이가 없다. [로컬 전수 Readback/재계산](../../outputs/task10i/stage2/local_completion_audit.json).

**Selection Pool:** D mean은 R보다 **6.40% 낮고** p90/p95/max도 낮다. **독립 Holdout:** D mean은 R보다 **1.57% 높고**, median은 동일하며 p90/p95/max도 더 높다. Conditional Random200에서 D의 Holdout mean percentile은 **83**, R은 **21**이다(낮을수록 좋음). 따라서 **독립 Holdout Coverage 개선은 INCONCLUSIVE이며 이 표본에서는 관측되지 않았다**. 이 결과를 보고 R/D나 weight·seed를 바꾸지 않았다. 실제 정책 성능·Expert 성공은 계속 **NOT_TESTED**다.

D tail19의 작은 margin 집중은 이번 기술통계에서 관측되지 않았다: margin<4px는 D tail0/19, R tail1/19이며 D tail 최소 margin5.834px다. 반면 사전 정의 위치·크기 extreme 비율은 D tail8/19(42.1%)로 R tail5/19(26.3%)보다 높다. 전체 D39/96(40.6%), R36/96(37.5%), Candidate112/300(37.3%)이다. 그러므로 경계 집중과 위치·크기 분포 변화는 구분해야 하며, 분포 보존이나 성능상의 무해성을 확정하지 않는다.

기존33과 신규300은 범위가 완전히 분리된 두 분포가 아니다. 후보가 일부 방향·크기·상대관계 범위를 확장하며 내부의 빈 영역도 보충한다. 아래는 주변 범위의 예이고, 이 표만으로 다변량 Coverage를 판정하지 않는다.

| feature | existing33_range | candidate300_range |
| --- | --- | --- |
| peg_xyz_0_m | -0.074122 … 0.095769 | -0.099564 … 0.098491 |
| peg_xyz_1_m | -0.127294 … -0.005111 | -0.124994 … -0.000103 |
| peg_yaw_rad | 0.562992 … 2.563091 | 0.530176 … 2.607494 |
| g4_0_m | 0.172873 … 0.248118 | 0.170179 … 0.249159 |
| g4_1_m | 0.031389 … 0.048830 | 0.030015 … 0.049815 |
| g3p_0_m | -0.133357 … 0.150143 | -0.144527 … 0.175658 |
| g3p_1_m | -0.445586 … -0.238876 | -0.492843 … -0.207464 |

기존 Full-task/Corrective 구성은 다음과 같다. Corrective source의 `nominal`도 독립 초기 상태 수로 더하지 않았다.

| split | inventory | logical | full_task | corrective | full_task_fraction | corrective_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| train | original | 272 | 96 | 176 | 0.352941 | 0.647059 |
| train | preserved | 82 | 33 | 49 | 0.402439 | 0.597561 |
| validation | original | 48 | 12 | 36 | 0.25 | 0.75 |
| validation | preserved | 23 | 6 | 17 | 0.26087 | 0.73913 |

수집 계획은 새 Train63마다 기존 실패63 slot의 실제 branch template를 대응하여 Full-task63+Corrective127=190개를 계획한다. Validation은 기존 실패9 slot의 Full-task6+Corrective19=25개를 계획한다. 모든 신규 Parent의 Full-task 성공 확인은 별도로 선행되어야 한다. Validation template에 Full-task logical entry가 없는 Parent도 성공 확인을 생략한다는 의미는 아니다. 자격 확인용 추가 trajectory를 학습 inventory에 자동 포함하지 않는다. 공유 tail이 다른 template를 받으면 필요한 branch union과 각 dataset inclusion을 분리한다.

로컬 최종 Markdown/Static HTML은 원격 원문 전체와 이 감사 결과를 포함한다. 원격의 최소 HTML fallback은 표/그림을 펼치지 않으므로 로컬에서 정식 Markdown renderer로 HTML을 생성했다. 원본은 변경하지 않았다. Monte Carlo 그림은 겹친 subplot 제목만 별도 전달 그림으로 수정했다. 모든 원본 결과 SHA는 그대로다. Gallery의 Raw RGB/OBB와 원본 NPZ/Evidence/Snapshot 링크는 현재 저장소의 Canonical 자료로 연결했다.

프로젝트 코드는 Git dirty 상태였고 새 Commit을 만들지 않았다. 원격 isolated source에는 `.git`가 없어 Logger의 commit이 null이므로 clean checkout 증거로 해석하지 않는다. [로컬 Git/Source provenance](../../outputs/task10i/stage2/local_source_provenance.json)와 [실제 frozen source](../../outputs/task10i/stage2/run/execution_source_identity.json)를 함께 사용한다. 분석은 Linux CPU, 전달 QA는 macOS 저장자료 분석으로 수행했다. Shared simulator/model/logger 코드는 변경하지 않았고 기존 ExperimentLogger를 재사용했다. 외부 Publishing·새 인스턴스·Stage3 실행은 없다.

원격 실행 전 계약 테스트 **11개 PASS**, 전달 후 계약 테스트 **12개 PASS**다. 추가 테스트는 늦은 Resume가 새 Logger UUID로 기존 결과에 연결할 때 원래 sidecar/SHA를 보존하는 경우를 검증한다. 이 운영 복구 보강과 별도 감사·전달 helper는 현재 코드에 있고 원격 연구 실행 당시 Source는 별도로 보존했다. [추가 계약 테스트 로그](../../outputs/task10i/stage2/postrun_contract_tests.log). PNG 그림을 실제 열어 확인했고 모든 HTML 표/이미지 및 로컬 링크를 정적으로 검사했다. 이 세션에 사용할 브라우저가 없어 브라우저 스크린샷 검사는 실행하지 않았다.
