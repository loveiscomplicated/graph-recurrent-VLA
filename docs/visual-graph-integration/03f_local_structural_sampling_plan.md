# Task 03-F — Generalizable Local Geometry Preservation: 수정 계획

**계획만 작성한 상태다. 구현·신규 진단 계산·테스트·실제 FPS는 실행하지 않았다.**
Task03-E의 geometry-level offline ablation과 검증은 완료로 인정한다.
이번 변경은 이 Markdown 하나이며, 아래 실행과 산출물 생성은 별도 구현 승인 이후의 계획이다.

### 연구 목표와 이번 수정 범위

목표는 PegInsertionSide의 Box rim retention 최대화가 아니라, **제한된 point budget에서 다양한
국소 기하 구조와 일반 표면을 안정적으로 보존할 sampling 규칙의 가능성과 취약성을 검증하는 것**이다.
다음 두 단계는 상호 배타적인 대안이 아니라 확장 가능한 구조다.

- **A. General Geometric Representation**: 물체 이름·작업 목적·알려진 형상 없이 관측 geometry의
  국소 구조와 일반 표면을 보존한다. 이번 Task는 이 단계의 sampling만 검증한다.
- **B. Task-Relevance Scoring**: 향후 A의 후보·geometric information에 task context, robot state,
  interaction 정보를 결합해 점별 중요도를 평가할 수 있다. Score는 최종 node feature 또는
  별도 승인된 sampling 우선순위에 쓰일 수 있다. A에서 일찍 제거한 정보는 B가 복구할 수 없다.

이번에는 B의 score 계산·학습·attention·task-conditioned quota·node feature 변경을 하지 않는다.
기존 후보 evidence와 전체 C/X의 source index, group·Priority·Surface 선택 경로를 보존해
향후 B를 별도 비교할 여지를 남기되, 새로운 production interface는 도입하지 않는다.

기존 **A의 코드·artifact 근거와 C의 재현·원본 보존 계약은 유지**한다. 아래 B는 방법의 의미와
일반성 한계를 명시하고, D는 GT-free 평가·최소 stress test·실험 완료와 채택의 구분을 추가한다.
A의 코드/API 확인 기록은 기존 계획의 근거를 유지한 것이며, 이번 수정에서 repository/API 탐색이나
artifact 진단을 다시 수행한 것은 아니다.

## A. 기존 candidate evidence와 FPS 코드의 재사용 가능성

### 확인한 저장 자료와 계약

| 자료 | 재사용 내용 |
| --- | --- |
| [Task03 comparison](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json), [Frame 0 samples](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz) | `evidence/pca5/{0,1}/20/orientation_flat`, 원본 선택 source·역할·23개 metadata field |
| [Task03-E JSON](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.json), [NPZ](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.npz), [verification](../../visual/results/diagnostic_previews/03e/verification.json) | `S3/{peg,box}/candidate_flat`, `candidate_xyz`, `support_flat`, `support_xyz`와 `0.0/{P,S,F}/*`; λ=0 global FPS를 저장 baseline으로 읽기 |
| [Task01 dense geometry](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.npz) | 원본 XYZ, validity, object mask, overlap, raster shape와 camera provenance |
| [Task02-C pairs](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_pairs.npz), [보고서](02c_structural_edges.md), [기존 Frame 0 그림](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_comparison.png) | 원본 2D 이웃과 관측 evidence의 성격 확인; 저장 reference와 negative ROI 재사용 |
| [normal comparison Frame 0](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0000.npz) | 평가용 Peg face map·visible mask·seam endpoints |

문서·현재 코드·기존 JSON/PNG 및 NPZ header를 읽었다. 새로운 연결 성분 수, 구간별 후보 수,
coverage, 거리, 선택은 계산하지 않았다. 아래 수치는 저장 결과와 배열 shape의 인용이다.

| S3 대상 | Structural 후보 C | 관측 support X | 기존 첫 Priority flat index | Priority / Surface / Final |
| --- | ---: | ---: | ---: | --- |
| Peg (`object_id=0`) | 703 | 11,780 | 214225 | 32 / 97 / 129 |
| Box (`object_id=1`) | 599 | 11,665 | 96553 | 32 / 95 / 127 |

### Raster·normal variation·validity·pixel-pair evidence

[sampling.py](../../visual/runtime/sampling.py)의 후보는 유효하고 tracker overlap이 없는 관측점이며,
원본 raster 오름차순 `flat_index=v*W+u`를 유지한다. S3의 Structural은 orientation-only다.
[edge_evidence.py](../../visual/runtime/edge_evidence.py)의 `PairEvidence`는 raw object mask에 닿는
오른쪽/아래쪽 1px pair를 한 번씩 저장한다. `supported[P,2]`는 endpoint별 관측 support,
`normal_angle_deg`는 양쪽 normal이 유효할 때만 정의된다. 20°를 **초과**한 pair의 supported endpoint가
orientation 후보가 된다. Variation은 기존 4-neighbour 최대 oriented angle과 관련되며,
mask boundary·invalid boundary 자체는 orientation 후보 조건이 아니다.

Task03 저장 metadata에는 `normal_variation_rad`, `normal_variation_valid`, `normal_valid`,
`valid_points`가 있지만 **선택된 256행에만** 있다. 모든 후보의 exact optimized-PCA5 feature map은 없다.
Task02-C에는 PCA5 Peg 23,876개 / Box 23,645개의 pair 행과 `pixels`, `supported`, `overlap`,
`invalid_depth_boundary`, `normal_angle_deg`가 저장되어 있다. 이는 reference PCA5의 evidence이며
Task03 optimized PCA5의 모든 실수 feature와 bit-exact하다고 가정하지 않는다.
Task02-C pair를 다시 threshold해 Task03 후보를 대체하지 않는다.

### 두 grouping 방식 검토와 추천 근거

| 방식 | 현재 evidence에서 가능한 일 | 이번 판단 |
| --- | --- | --- |
| A. Image-space 연결 성분·연속 구간 | 저장 source index로 4-neighbour 후보 인접성을 구성할 수 있다. 고각도 pair만 연결할지 후보 mask의 모든 인접점을 연결할지에 따라 다른 topology가 된다. | 고각도 pair는 선을 따라 이어지는 링크가 아니라 normal 변화의 양쪽 endpoint 증거다. 후보 띠·접점은 여러 구간을 합칠 수 있고 validity/지원의 끊김은 잘게 나눌 수 있다. 성분을 연속 구간으로 바꾸려면 추가 thinning·분기·gap 규칙이 필요하다. |
| **B. Deterministic spatial stratification** | 동일 후보의 원본 UV만으로 작은 수의 공간 group을 만들고 기존 FPS를 group별 호출할 수 있다. | **이번 최소 구현으로 추천한다.** Exact dense normal feature나 안정적인 component topology를 요구하지 않는다. |

기존 그림에는 이어진 후보 띠·접점과 rim 밖 반응이 함께 보이고, 기존 보고서는 mask/occlusion 혼동을 기록한다.
그러나 S3 후보가 실제 몇 개 성분으로 분절되는지는 아직 측정하지 않았다.
B의 선택은 심한 fragmentation을 새로 입증했다는 뜻이 아니라, 현재 확인된 저장 입력만으로
추가 topology 보정 없이 하나의 결정적 control을 정의하기 위한 판단이다.
A는 구현하거나 추가 ablation으로 실행하지 않는다. B가 geometry를 개선한다는 전제도 두지 않는다.

### 기존 함수 재사용과 interface gap

- `fps_indices()`를 변경 없이 재사용한다. Float64 제곱거리, 중복 row 방지,
  `argmax` 동률 시 입력의 첫 row, `initial`이 총 count에 포함되는 계약을 유지한다.
- Task03-E [runner](../../visual/runtime/test/run_priority_fps_metric_control.py)의
  `metric_coordinates(..., 0.)`, `feature_rows()`, `evaluation_inputs()`, `score_case()`,
  `peg_labels()`, `rim_labels()`, `segment_spacing()`을 필요한 범위에서 호출한다.
  기존 `main`/`diagnose`/`load_inputs`/`execute_controls`는 S2나 λ=1 replay를 포함하므로 호출하지 않는다.
- [sampling_evaluation.py](../../visual/runtime/test/sampling_evaluation.py)의 `CoverageMatcher`,
  `staged_reference_coverage`와 기존 ROI/occupancy helper를 재사용한다.
  Normal·backprojection·candidate detector·reference 생성 경로를 실행하지 않는다.
- Production `allocate_budget()`는 두 물체용 shape `(2,)` 계약이다. 임의 group 수에 쓸 수 없다.
  이번 group allocation은 새 offline runner 내부의 작은 helper로 한정한다.
  공통 API 확장은 이 한 조건의 실험에 필요하지 않으며 production sampler를 복제하지 않는다.

## B. 추천 방법 하나: UV spatial stratification control

**UV 공간 이분할 + capacity-capped 균등 배분을 유일한 구현 추천으로 유지한다.**
현재 저장된 GT-free 입력과 기존 FPS만으로 공간별 예산 분산의 효과를 비교할 수 있기 때문이다.
이는 일반적인 geometry sampling을 검토하기 위한 **spatial stratification control**이며,
structural grouping이나 실제 구조 보존을 이미 달성한 방법이라는 뜻은 아니다.
Image-space 연결 성분 대안은 A의 검토에만 남기고 구현하지 않는다. 다른 grouping을 추가하거나 sweep하지 않는다.

### UV 분할의 의미와 일반성 한계

UV의 공간 분할은 실제 국소 기하 구조의 분할과 동일하지 않다.

| 상황 | 예상 취약성과 해석 |
| --- | --- |
| 하나의 긴 연속 edge가 여러 group에 걸침 | 동일 구조가 여러 quota를 받는다. Group 수는 독립 구조 수가 아니며, 다른 구조보다 예산을 많이 받을 수 있다. |
| 서로 다른 구조가 한 group에 들어감 | 같은 quota 안에서 Camera XY FPS로 경쟁한다. 깊이만 다른 구조나 작은 구조의 보존을 보장하지 않는다. |
| 작은 noisy cluster와 큰 실제 구조가 공존 | 최소 capacity는 noise 판별이 아니다. 4개 이상 noise가 leaf를 차지하거나 실제 구조와 섞일 수 있고, 균등 quota가 noise에 과도하게 배분될 수 있다. |
| Camera distance·viewpoint 변화 | 투영 span·정수 pixel·동률·occlusion과 후보 밀도가 바뀌면 split/group이 바뀐다. 이상적인 연속 UV의 균일 scale과 달리 실제 raster 관측의 불변성은 보장하지 않는다. |
| 후보 밀도·Priority budget 변화 | 후보 수 균형 split은 조밀한 영역을 여러 leaf로 나눌 수 있다. 균등 quota는 길이·면적·복잡도에 비례하지 않으며 희소한 긴 구조를 충분히 덮는다는 근거가 없다. |

**최대 8개 group과 split 최소 capacity 4개는 이번 Priority 32점 budget에 맞춘 실험 설정**이다.
8×4는 group별 약 4점을 배정할 여지를 주지만, 4후보가 하나의 실제 구조이거나 4선택점이 충분하다는
기하학적 근거는 없다. Capacity-capped 배분은 유한 budget·capacity 준수와 결정성은 보장하지만,
일반적인 budget/candidate 조건에서 최적 또는 균형적인 구조 보존을 보장하지 않는다.
다른 budget에 8/4를 자동 비례 조정하는 규칙도 이번에 도입하지 않는다.
작은 budget에서 일부 group이 quota 0을 받는 등의 동작 계약과 성능 타당성은 별개이며,
이번 실제 비교와 geometry stress test는 K=32로 한정한다. 일반 budget의 성능은 미검증이다.

다음 가설만 검증한다: 같은 C와 K에서 이 공간 배분이 global FPS보다 후보의 국소 공간 보존을
균형 있게 만드는가, 그 효과가 density·구조 크기·관측 변화와 false response에 얼마나 취약한가.
후보가 남아 있는 공간을 보존한다는 것과 올바른 실제 구조를 보존한다는 것은 별도로 평가한다.

### Group 생성 규칙

물체별로 **동일한 규칙**을 적용한다. Group 생성 입력은 `candidate_flat`과 원본 `(H,W)`뿐이다.
UV는 정수 `u=flat%W`, `v=flat//W`로 얻는다. XYZ·normal score·GT·evaluation label은 grouping 입력이 아니다.
Object ID는 입력 집합 분리와 기존 seed stream 식별에만 사용한다.

1. 모든 후보를 하나의 leaf로 시작한다. 목표는 **최대 8개 leaf**, 분할 후 최소 capacity는 **4후보**다.
   8은 Priority 32점을 group당 약 4점으로 나누는 이번 control의 고정 granularity다.
   Peg face 수나 Box 물리 선분 수에서 정하지 않았으며 group 수 sweep을 하지 않는다.
2. 분할 가능한 leaf 중 후보 수가 가장 큰 것을 고른다. 동률은 `min(flat_index)`가 작은 leaf 우선이다.
3. 각 축에서 서로 다른 정수 좌표 사이만 cut 후보로 삼고, 양쪽에 각각 최소 4후보가 남아야 한다.
   유효 cut이 있는 축 중 UV span이 큰 축을 택한다. Span 동률은 u축 우선이다.
4. 해당 축에서 양쪽 후보 수 차이가 가장 작은 cut을 택한다. 동률은 더 작은 cut 좌표 우선이다.
   좌표가 같은 후보를 cut 양쪽으로 임의 분산하지 않는다. Cut 양쪽 좌표 사이의 중점을 경계로 기록한다.
5. 8개 leaf에 도달하거나 유효한 split이 없으면 끝낸다.
   최종 group ID는 leaf의 `min(flat_index)` 오름차순으로 부여한다.
   각 group 안의 후보 row는 다시 원본 flat index 오름차순으로 유지한다.

성분을 만들지 않으므로 connectivity fragmentation에 따른 무제한 group 증가는 없다.
작은 고립 후보도 삭제하지 않으며, 최소 capacity 조건 때문에 작은 cluster 하나가 별도 leaf가 되는 일을 제한한다.
노이즈 여부를 판별하거나 morphology·depth filter로 합치지 않는다.
큰 noisy 영역이 group을 차지하거나 작은 구조가 같은 leaf에 묻힐 위험은 남으며 평가에서 드러내야 한다.

### Priority 32점 allocation

`N=|C|`, `K=min(32,N)`으로 두고 group capacity `n_g`를 사용한다.
초기 `q_g=0`에서 group ID 오름차순으로 순회하며 `q_g<n_g`인 group에 한 점씩 배분한다.
총합이 K가 될 때까지 같은 순회를 반복한다. 포화 group은 건너뛴다.
이는 capacity에 제한된 균등 배분이며 GT·score·깊이·물체별 구조 가중치는 없다.
예를 들어 8개 group 모두 capacity가 충분하면 각각 4점이지만, 실제 group 개수·capacity·quota는 승인 후 기록한다.

항상 `sum(q_g)=K`, `0<=q_g<=n_g`다. 작은 capacity의 잔여 quota는 같은 순회에서 다른 group에 배분한다.
선택 결과나 coverage를 본 뒤 group·split·quota를 다시 조정하지 않는다.
작은 budget의 계약 테스트에서 K가 group 수보다 작으면 group ID 순서의 첫 K개에만 1점이 간다.
이때 anchor group quota가 0이면 anchor를 강제로 추가하지 않고 quota가 양수인 group만 ID 순서로 잇는다.
이는 off-budget 경계 동작이며 anchor-first 비교 계약과 성능 해석은 실제 K=32 조건에 한정한다.

### 초기점·group 내부 FPS·Surface

- 기존 object seed는 `SeedSequence([0, object_id]).generate_state(1)[0]`이다.
  원본 전체 후보의 seeded 첫 점을 anchor로 유지한다. 현재 저장 anchor는 Peg 214225, Box 96553이다.
  승인 후 저장 첫 점과 seed 계약을 대조하되, 이를 위한 global FPS replay는 하지 않는다.
- Anchor가 속한 group은 quota가 양수일 때 그 점의 group-local row를 `initial=[row]`로 전달한다.
  나머지 group은 동일 object seed와 `initial=()`로 기존 FPS를 호출한다.
  나머지 group의 각 호출은 정렬된 row에서 기존 `default_rng(object_seed).integers(group_capacity)`로 첫 점을 정한다.
  Anchor group은 `initial`을 사용하므로 첫 점을 위한 RNG draw를 수행하지 않는다.
  전역 RNG 상태나 Python hash에 의존하지 않고, group별 seed sweep·새 seed는 도입하지 않는다.
- Group별 거리 입력만 `[x,y,0]`의 별도 float64 배열이다. 호출은 기존
  `fps_indices(metric_group_xyz, q_g, seed=object_seed, initial=...)`다.
  XY 동률은 group 내 원본 source 순서로 해결한다. 원본 XYZ를 덮어쓰지 않는다.
- Quota 0인 group에는 FPS를 호출하지 않는다. Priority 순서는 quota가 양수인 anchor group의
  FPS 결과를 먼저, 나머지 group은 ID 오름차순으로 이어 붙인다.
  이번 K=32에서는 anchor가 전체 P의 첫 점이고, P의 source index는 모두 고유하다.
- P를 원본 support row로 매핑한 뒤
  `fps_indices(support_xyz, final_budget, seed=object_seed, initial=priority_rows)`를 호출한다.
  **기존 실제 3D Surface FPS**로 Peg 97점 / Box 95점을 추가한다. Surface의 선택점 자체는 P에 따라 달라질 수 있다.
  λ=0은 이 비교의 distance control이며 최종 metric 채택이 아니다. Camera XY 거리와 image UV 거리는 구분한다.

### Fallback과 alignment

- Group이 하나면 같은 anchor로 후보 전체에 기존 λ=0 FPS를 호출한다.
  충분한 후보가 있는 이 경우 P/F는 저장 global 방식과 같은 선택 계약이다.
- `0<N<32`이면 전 후보를 고유하게 P로 선택하고 부족분을 기존 Surface FPS에서 채운다.
  이때 `q_g=n_g`이며 위 group 내부 FPS·anchor-first 이어 붙이기로 P의 순서도 결정한다.
  `N=0`이면 P는 비우고 동일 object seed의 3D Surface FPS로 전체 budget을 채운다.
  빈 C에는 group·anchor를 만들거나 Priority FPS를 호출하지 않는다.
  Support까지 부족하면 padding·가상점을 만들지 않고 가능한 수와 shortfall을 기록한다.
  **이번 실제 ablation은 저장 703/599후보·고정 support·32점 조건이 필수다.**
  해당 입력 계약이 깨지면 fallback 수치를 정상 paired 결과로 내지 않고 중단한다. 부족 사례는 synthetic 테스트로 검증한다.
- Group 입력에 중복 source가 있으면 조용히 dedup하지 않고 오류로 처리한다.
  모든 C가 정확히 한 group에 속하고, 선택 local row → C flat → X row의 대응을 보존한다.
  최종 XYZ는 X에서 gather하고 pixel `(v,u)`·object ID·frame·camera·P/S role을 같은 row에 맞춘다.
- Feature는 Task03-E의 `feature_rows()`와 같은 unavailable 계약을 유지한다.
  동일 S3 baseline condition의 저장 metadata에 있는 source만 23개 field를 exact gather한다.
  `feature_available`, `feature_available_positions`, compact `available_features/*`를 저장한다.
  없는 값은 생성하지 않으며 dense reference PCA cache·근사 feature·새 encoder로 보충하지 않는다.
  선택되지 않은 C의 기존 evidence·source 및 선택 경로도 참조 가능하게 보존하며,
  미가용 feature를 이유로 후보를 추가 제거하지 않는다.

## C. 최소 offline ablation의 변경 파일과 정확한 실행 범위

### 변경 파일

| 경로 | 승인 후 역할 |
| --- | --- |
| `visual/runtime/test/run_local_structural_sampling.py` (신규) | S3 저장 입력 loader, B의 group/allocation helper, 기존 FPS·평가 재사용, GT-free 공간 평가와 D의 고정 synthetic fixture 비교, JSON/NPZ/PNG/HTML 기록 |
| `visual/runtime/test/test_local_structural_sampling.py` (신규) | Group·allocation·선택·provenance·GT 독립성의 synthetic 계약 테스트 |
| `docs/visual-graph-integration/03f_local_structural_sampling.md` (신규) | 최종 결과·trade-off·검증·제약 및 새 run 링크 |

기존 sampler·detector·normal estimator·평가 helper·Task03-D/E runner·공통 logger/publisher는 수정하지 않는다.
현재 작업 트리의 기존 수정 및 미추적 파일도 보존한다. 계획 파일 외의 위 세 파일은 아직 생성하지 않는다.

### 비교와 실행 순서

실제 관측 비교 범위는 **ep000 Frame 0 × S3(PCA5) × Peg/Box × {저장 global λ=0, 신규 local λ=0}**다.
이와 별도로 D의 작은 synthetic candidate stress test만 수행할 계획이다. 추가 episode 수집이나 simulator 실행은 없다.
고정값은 `num_points=256`, `ratio=.25`, `threshold=20°`, `seed=0`이다.
변경은 B의 grouping·quota 및 이에 따른 group 내부 선택뿐이다. 후보 C, support X, 원본 XYZ/source,
Priority 32점, Final 129/127점, 3D Surface 방식, reference·joint tolerance는 고정한다.

1. 새 실행 시작 시 입력·보호 source·기존 artifact/sidecar의 hash 및 Git dirty 상태를 기록한다.
   Task03-E 저장 hash·source 계약과 현재 사용 함수의 호환성을 확인한다.
   S3 candidate/support 배열은 Task03/Task01과 exact 대조한다. 불일치하면 신규 선택 전에 중단한다.
2. Task03-E NPZ의 `S3/{peg,box}/0.0/{P,S,F}` 순서·원본 XYZ·역할을 그대로 읽는다.
   **Global λ=0 FPS와 λ=1 FPS는 재실행하지 않는다.** λ=1은 기존 결과 링크/수치의 별도 참고 baseline이다.
3. 관측 입력만 전달한 순수 helper로 두 물체의 group·quota·local P를 확정한다.
   Group별 FPS는 한 번씩, 3D Surface FPS는 물체별 한 번만 수행한다.
   실제 데이터의 FPS를 synthetic 테스트나 시각화 생성에서 반복하지 않는다.
4. 두 물체의 선택을 저장하고 고정한 뒤 평가 annotation·구간 label을 읽는다.
   기존 matcher로 저장 global 결과를 재평가하여 Task03-E 수치와 대조한다.
   Baseline 평가가 다르면 tolerance나 reference를 수정하지 않고 실패로 보고한다.
5. 같은 평가로 local 결과 및 global 대비 paired 차이를 기록한다. D의 GT-free 평가와
   미리 고정한 synthetic 비교도 기록한다. 결과를 본 재선택·hyperparameter 조정은 없다.

위 순서에 앞서 B의 split·quota·seed·tie·fallback, D의 평가 구역·fixture 생성 상수·평가 규칙을
config/source snapshot에 고정한다. 실제 Box rim이나 synthetic 성능을 보고 유리하게 바꾸지 않는다.
Phase1–3 분석, Task03-E ablation 및 실제 λ=1/0 global FPS를 재실행하지 않는다.

승인 후 제안 entrypoint는 아래와 같다. **현재 없는 신규 모듈의 예정 명령이며 이번에는 실행하지 않는다.**
Runner는 위 고정 조건만 실행하도록 만들고 sweep 옵션을 추가하지 않는다.

```bash
WANDB_MODE=disabled MPLCONFIGDIR=/private/tmp/task03f-mpl \
  .venv/bin/python -m visual.runtime.test.run_local_structural_sampling
```

`ExperimentLogger(team='visual', job_type='evaluate', seed=0, ...)`로 **새 UUID**를 생성한다.
기존 UUID/project/group 규칙을 유지하고 Task03와 Task03-E를 provenance로 기록한다.
`source_run_id`에는 직접 baseline인 Task03-E를, 별도 config 항목에는 Task03 원본 run을 기록한다.
새 파일만 `link_output()`으로 로컬 inventory/sidecar에 연결한다.
원본 artifact를 새 run 소유로 재등록하지 않는다. Publishing 경로는 기존
`utils.experiment_artifacts`/`tools.publish_experiment`를 유지하며 이번 작업에서는 호출하지 않는다.

## D. 테스트·완료 기준·결과 문서 및 시각화 계획

### 1. GT-free 일반 공간 보존: primary sampling 진단

실제 비교에서는 같은 후보 C, Priority P, 관측 support X, Final F를 사용한다.
Global P/F는 저장 배열을 읽고 local P/F만 신규 선택한다. 다음은 **승인 후 계산할 지표**다.

| 관찰할 항목 | 최소 측정 규칙 |
| --- | --- |
| C → P 국소 거리 | 모든 c∈C에 대해 `min_p ‖xyz(c)-xyz(p)‖₂`(mm)와 `min_p ‖uv(c)-uv(p)‖₂`(px)를 각각 계산한다. 서로 독립적인 최근접점이며 joint GT matching과 혼동하지 않는다. Raw 거리 분포/ECDF, p50/p95/max 및 global 대비 paired 차이를 저장한다. |
| 가장 크게 비어 있는 관측 국소 영역 | 각 거리의 max를 만드는 후보의 source·XYZ·UV와 아래 평가 구역을 표시한다. 동률은 원본 flat index 우선이다. 구역별 p95/max와 P=0인 후보 점유 구역도 표시한다. 이는 관측 C 위의 최대 미피복 위치이며, 후보 사이의 연속 빈 공간이나 실제 구조의 최대 결손 길이를 복원한 값은 아니다. |
| 국소 집중과 공간 분산 | 아래 독립 구역별 `n_r=count(C∩r)`, `p_r=count(P∩r)`, `p_r/K`, `p_r/n_r`, P=0 여부를 기록한다. 후보 점유 구역 중 선택점이 있는 구역의 비율, 최대 구역의 Priority 점유율, 구역별 거리 p95/max와 구역을 동등 가중한 평균 p95를 함께 비교한다. |
| 후보 수·밀도·예산 | 전체 N, K, K/N과 구역별 n_r·pixel 면적당 후보 수·p_r/n_r를 보인다. B의 group capacity/quota/선택률은 별도 동작 진단이다. N=0 또는 분모 0은 N/A로 기록하며 선택 수 자체도 함께 보인다. |
| 일반 Surface | 같은 X에서 F까지 실제 3D 최근접 거리 mean/p95(mm)를 기존 방식으로 비교한다. P의 후보 coverage 개선이 X → F의 악화를 상쇄한다고 자동 판단하지 않는다. |

**평가 구역은 sampler group과 독립적으로 정의한다.** 물체별 X의 UV bounding box를 각 축 4등분한
4×4 격자를 사용한다. 추가로 X의 camera XYZ bounding box를 각 축 2등분한 최대 2×2×2 구역을
사용해 같은 pixel 영역의 depth 차이도 관찰한다. 경계는 낮은 쪽 포함·높은 쪽 제외이며 최댓값은
마지막 구역에 포함한다. Span 0인 축은 한 구간으로 둔다. 경계는 같은 X에서 한 번만 고정하고,
global/local에 동일 적용한다. 빈 구역은 개수와 함께 구분하고 후보 점유 구역만 coverage 분모에 쓴다.
격자 해상도는 평가 전에 고정하며 group/cut·GT·선택 결과로 옮기거나 추가 세분하지 않는다.

거리 p50/p95는 후보 밀도에 가중된 통계다. 조밀한 후보 구역에 전체 지표가 지배되는 문제를
구역별 거리와 동등 구역 가중 요약으로 함께 드러낸다. 물체/scale 간 참고 비교에는 동일 X의
3D·UV bbox 대각선으로 나눈 거리도 함께 기록하되, 분모가 0이면 N/A로 둔다.
절대 거리와 정규화 거리를 모두 보존하고 임의 성공 threshold나 합성 종합 점수는 만들지 않는다.

Group quota 충족만으로 local coverage 개선을 주장하지 않는다. 격자 분산 역시 실제 구조별 분산과
같지 않다. **C에는 false structural response가 포함될 수 있으므로 C → P나 분산 개선은
실제 구조 보존 정확도의 자동 개선이 아니다.** 아래 GT 보조 평가, noise·관측 경계·negative ROI와
Surface trade-off를 함께 해석한다. 평가 label은 grouping·quota·FPS에 전달하지 않는다.

### 2. 기존 GT geometry 보조 평가: selection 이후에만 GT 사용

Peg seam과 Box rim은 일반적인 geometry preservation을 관찰하는 진단 사례다.
GT coverage를 알고리즘의 최적화 목표나 group·quota 결정 근거로 사용하지 않는다.

`R(A)`를 기존 **같은 node/reference pair의 2px AND 3mm** 조건으로 A가 덮은 unique reference 집합으로 둔다.
C/P/S/F에 대해 다음을 기존 `score_case()`와 staged helper로 기록한다.

- Peg 전체 seam 및 face-pair별, Box 전체 rim 및 선분별 C/P/S/F coverage와 numerator/denominator.
  분모는 기존 관측 가능 Peg 462 / Box 103이고, GT visible Peg 464 / Box 103도 함께 보존한다.
  Reference 좌표·ID·순서·ambiguous membership은 Task03-E와 같다.
- 실제 matching node 수와 unique reference 수, joint pair 목록, 중복으로 덮인 reference 수를 구분한다.
  여러 평가 label에 걸친 node는 기존처럼 ambiguous로 남긴다.
- `R(C)\R(P)`, `R(C)\R(F)`로 후보가 있었지만 선택/최종 보존되지 않은 reference를 구간별 표시한다.
  `R(P)`, `R(S)\R(P)`, `R(P)\R(S)`, `R(P)∩R(S)`로 Priority 보존·Surface 추가·각 경로의 독자 기여와 중복을 구분한다.
- 동일 X → F 실제 3D 최근접 거리의 mean/p95(mm), 기존 known-negative ROI별/union 점유를 P/S/F로 비교한다.
  ROI는 겹칠 수 있으므로 단순 합산하지 않는다. 일반 Surface point를 자동으로 오선택으로 분류하지 않는다.
- 기존 X depth median/p95 경계의 band별 C/P/S/F 점유·선택률과 depth 통계를 재사용한다.
  원거리 점을 삭제하거나 새 depth cutoff로 샘플링하지 않는다.
- Group별 capacity·quota·선택률 및 실제 선택점 UV/XYZ 분포, 기존 matched-node 최근접 간격,
  reference별 covered/missing 위치를 보여 국소 집중과 빈 구간을 확인한다.
  Group quota의 충족은 동작 계약일 뿐, 구조 또는 일반 공간 보존 향상의 증거로 단독 사용하지 않는다.
  Box는 기존 `segment_spacing()`으로 실제 matching node 투영 간격·tail·missing-reference run span을 평가한다.
  Peg 물리적 중심선 순서는 검증되지 않았으므로 물리 길이/arc 간격은 N/A다.
  Reference 배열 순서를 물리 순서로 바꾸어 해석하거나 missing span을 연속 미보존 길이로 보간하지 않는다.

### `rim_axis1_-1`: 보조 실패 사례의 후보 지원과 선택 실패를 구분

S3 Box λ=0의 기존 Final은 이 구간에서 **0/26**이다. 전체 C가 91/103 rim reference를 덮는다는
저장 수치만으로 이 구간에 충분한 후보가 있었다고 판단하지 않는다. 승인 후 **선택이 확정된 평가 단계에서만**:

1. 기존 matcher의 `2px_3mm` adjacency에서 이 구간에 속한 저장 reference column만 평가용으로 조회한다.
2. X/C/P/S/F 각각에 대해 해당 column과 joint match하는 고유 node 수·flat index,
   덮인 unique reference ID와 `/26` coverage를 기록한다. 복수 label node는 별도 표시한다.
3. `R(C)`와 해당 구간 reference의 교집합을 후보 지원 상한으로 보여준다.
   같은 곳에 몰린 많은 후보와 구간 전체를 덮을 수 있는 후보를 구분하기 위해 위치와 covered/missing ID를 함께 표시한다.
4. 후보로도 못 덮는 reference, 후보는 덮지만 global P/F가 놓친 reference,
   local P/F가 새로 보존한 reference와 여전히 놓친 reference를 분리한다.

새로운 “충분한 후보 수” cutoff나 rim quota는 만들지 않는다. 이 평가로 얻은 label·수치·위치를
group 생성·allocation에 되돌려 보내지 않는다. Local에서도 0/26이거나 다른 구간이 악화되어도 그대로 보고한다.
이 구간의 개선은 필수 성공 조건이 아니다. 개선되더라도 다른 seam/rim이나 Surface를 심하게 잃으면
그 손실을 함께 보고하며, 특정 구간의 개선으로 일반적인 보존 성능이 입증됐다고 해석하지 않는다.

### 3. 최소 synthetic geometry stress test: sampler 강건성만 검증

ep000 Frame 0의 개선만으로 일반화 성능을 입증할 수 없다. 승인 후 **주어진 후보 집합에 대한
sampler의 강건성**을 보는 작은 offline 비교를 추가한다. 새 episode·simulator·센서 수집,
합성 환경용 normal estimator/candidate detector는 만들지 않는다.

아래 **3종, 총 7개 고정 fixture**만 비교한다. 모든 fixture는 K=32, seed=0, B의 8/4 규칙과
Camera XY λ=0을 그대로 쓴다. Synthetic global P는 같은 전체 후보와 anchor에서 기존 FPS를
한 번 호출하고, local P는 B의 group별 기존 FPS로 한 번 선택해 저장한다.
모든 fixture는 geometry label과 무관하게 동일 `SeedSequence([0, 0])` stream을 사용한다.
Anchor는 원본 flat 순서 C에 대한 해당 seed의 첫 RNG row로 정하며 두 방식에 공유한다.
이는 실제 Task03-E baseline의 replay와 별개다. 시각화·평가에는 저장한 선택을 사용한다.

| Fixture 쌍/변형 | 고정할 입력과 관찰할 취약성 |
| --- | --- |
| 긴 얇은 연속 구조: 균일 / 불균일 밀도 (2개) | 같은 길이의 직선 구조에 기본 128개 균일 후보를 둔다. 불균일 버전은 첫 1/4 구간에만 128개 후보를 추가한다. 겹치는 endpoint/source는 생성 시 피한다. 길이·support 범위는 동일하며 density 때문에 leaf와 예산이 조밀한 끝에 쏠리는지 본다. |
| 떨어진 여러 구조: clean / 작은 noise 추가 (2개) | 서로 떨어진 길이 비 1:2:4의 선분 3개에 각각 64개 균일 후보를 둔다. Noise 버전에는 짧고 조밀한 고립 cluster 8후보만 추가한다. 작은 noise가 최소 capacity를 통과할 수 있는 상황에서 실제 구조의 길이별 보존과 noise 점유를 본다. |
| 같은 geometry의 near / far / 부분 가림 (3개) | 동일한 분리 선분 배치를 같은 pinhole camera에서 거리 d와 2d로 투영한다. 세 번째는 near 관측에서 화면 중앙 1/4 폭의 관측을 제거한다. 같은 geometry의 scale·정수 raster 변화와 가림으로 group·선택이 바뀌는지 본다. |

각 fixture의 실제 좌표·선분 배치·작은 cluster 위치, raster 크기·intrinsics·d, 가림 경계 및
균일 표본의 endpoint 규칙은 구현 시 **어떤 FPS나 성능 평가보다 먼저** 고정 manifest에 명시한다.
두 방식은 fixture별 같은 C·XYZ·UV·source 순서·anchor를 공유한다. 정수 raster로 투영할 때
source가 유일하도록 하는 생성 규칙과 동일 pixel 충돌 처리도 사전 명시하고 생성 전후 후보 수를
기록한다. 이 fixture 생성 규칙을 성능을 보고 바꾸거나 후보 수를 맞추기 위해 결과별 보정하지 않는다.
추가 distance·noise 비율·budget·seed 값이나 grouping 알고리즘은 sweep하지 않는다.

Synthetic 선택 단계에는 C의 XYZ·UV/source와 고정 seed/budget만 전달한다.
어느 후보가 실제 구조인지, noise인지, 어느 선분에 속하는지는 evaluator 전용 별도 annotation이다.
가림은 관측 생성 단계에만 적용하며 sampler에 occlusion label을 전달하지 않는다.
새 segmentation·skeletonization·mesh reconstruction·learned clustering도 없다.

평가는 다음을 사용한다.

- D.1의 C → P 거리 분포·구역 분산·예산 비율을 적용한다. Synthetic 평가 구역의 bbox는
  fixture의 관측 후보 C에서 고정하며 두 방식에 동일 적용한다. 대응 near/far에서는 raw와
  bbox 정규화 거리를 함께 보고, visible geometry가 달라지는 가림 사례는 별도 표시한다.
- 생성 시 알려진 실제 선분 위에 후보 밀도와 독립적인 균일 reference 128개씩을 둔다.
  선택 후에만 reference → P 실제 3D 거리 p50/p95/max를 전체·선분별로 측정한다.
  가려진 reference는 visible 평가에서 제외하고 제외 수를 보존한다. 관측 불가능 부분의 복원을
  sampler에 요구하지 않는다. 이 합성 거리 평가는 실제 Task03-E의 joint tolerance 평가와 구분한다.
- 평가용 구조 ID별 실제 선택 수·선분 길이·거리, noise cluster의 P 점유율을 기록한다.
  관측 bbox 외곽 10% 띠와 알려진 가림 경계 1px 띠의 C/P 점유율도 고정된 평가로 기록한다.
  경계에 실제 구조가 있을 수 있으므로 경계 점유 자체를 오류로 단정하지 않는다.

Synthetic에서는 후보 선택 P의 성능만 비교한다. 일반 Surface X → F 성능은 실제 고정 X에서 평가하고,
synthetic support/normal/feature를 새로 만들어 perception 전체의 일반화로 확대하지 않는다.
동일 XYZ/source·budget·seed·tie·fallback을 확인하는 **구현 계약 테스트**와 위 **geometry 성능 비교**는
구분한다. 성능이 악화돼도 계약 테스트 실패로 바꾸거나 성공할 때까지 fixture를 조정하지 않는다.
계약 위반·입력 불일치는 비교를 무효화하지만, 유효한 비교의 성능 실패는 결과로 남긴다.

이 작은 검증도 실제 독립 episode·다른 물체·다른 센서의 일반화를 입증하지 않는다.
다른 viewpoint 전반과 일반 budget의 성능 역시 미검증으로 남기며 대규모 benchmark를 계획하지 않는다.

### 4. 승인 후 구현 계약 테스트와 실패 기준

| 검증 | 필요한 확인 |
| --- | --- |
| 후보·budget | 두 방식의 C/X 및 source 순서·XYZ가 exact 동일. 실제 Peg/Box 각각 P=32, S=97/95, F=129/127, 합계 256 |
| Group partition | C의 disjoint exhaustive partition, group 내부/사이 중복 없음, 최대 8개, split 최소 capacity, 축/cut/leaf 동률 규칙과 반복 결정성 |
| Allocation | `sum(q)=K`, 실제 K=32 또는 후보 부족 시 min(32,N), 모든 quota가 capacity 이내. 포화·uneven capacity·단일 group·빈 후보·32 미만 fallback 및 작은 budget의 quota 0/anchor 동작 검증. 일반 budget 성능 검증으로 해석하지 않음 |
| 선택 | Group마다 기존 FPS 호출 확인, P⊆C, P/S 및 전체 선택 source 중복 없음. Anchor·group 순서·XY tie·동일 XYZ row 처리 검증 |
| 단일 group | Synthetic 입력에서 기존 λ=0 global FPS와 P/F 순서 exact 일치. 실제 baseline은 재실행하지 않음 |
| GT 독립성 | Synthetic GT 좌표·pose·rim/face-pair label·ROI annotation을 변경/삭제해도 group ID·quota·P/S/F가 불변. 선택 helper에 annotation 인자가 없고 GT 접근도 없음을 확인 |
| Source alignment | Local→candidate→support row 매핑, 원본 XYZ·pixel·object·frame·camera exact 대응, 변환 거리 배열이 원본을 변경하지 않음 |
| Feature gap | Exact 가용 row만 gather. Availability mask/compact position 정렬·미가용 값 부재 확인. 근사 복원 없음 |
| 저장 baseline 보존 | Task03-E λ=0 저장 P/S/F·XYZ·role·수치와 artifact hash 불변. λ=1 포함 기존 결과도 덮어쓰지 않음 |
| 평가 동일성 | Reference UV/XYZ·ID·관측 분모 exact 일치, 동일 joint tolerance와 같은 pair 사용, node/reference dedup, `R(F)=R(P)∪R(S)` 및 기여·overlap 검증 |
| GT-free 지표 | 작은 analytic synthetic 입력에서 최근접 3D/pixel 거리·p50/p95/max·격자 경계·점유 raw count 확인. 같은 평가 구역 사용, sampler group을 평가 구역으로 대체하지 않음. 빈 집합/0 span/N/A 처리 명시 |
| Stress test 분리 | 7개 fixture와 설정 사전 고정·source 및 평가 annotation 분리·선택 재사용 확인. 성능의 개선 여부는 assertion으로 강제하지 않음 |
| 입력/source/artifact 보존 | 실행 전후 SHA-256 대조. 기존 dirty source/미추적 파일·Task03/03-E·관련 입력/sidecar 불변. 역사 source snapshot과 현재 hash의 차이는 구분해 기록 |
| 산출물 | NPZ readback alignment, JSON raw counts, HTML/Markdown 상대 링크와 PNG 실제 표시 확인 |

신규 테스트의 FPS 입력은 synthetic으로 한정한다. 아래 관련 기존 suite와 함께 한 번 실행하고,
통과 뒤 변경/실패가 없으면 같은 검사를 반복하지 않는다. 지금은 어느 테스트도 실행하지 않는다.

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest \
  visual.runtime.test.test_local_structural_sampling \
  visual.runtime.test.test_sampling \
  visual.runtime.test.test_sampling_geometry \
  visual.runtime.test.test_sampling_evaluation \
  visual.runtime.test.test_priority_fps_metric_control -v
```

공통 logger/publisher는 변경하지 않는다. 기존 공통 계약을 바꿀 필요가 발견되면 먼저 gap을 보고하며,
그 변경을 이번 최소 구현에 조용히 추가하지 않는다.

### 5. 산출물과 완료 기준

새 `experiments/<new_uuid>/`에 다음을 만들 계획이다.

| 산출물 | 내용 |
| --- | --- |
| `local_structural_sampling.html` | GT 없이 구분한 candidate group과 실제 선택점, global/local의 일반 공간 거리·분산 지표, 7개 synthetic stress test의 간단한 비교, Peg seam/Box rim GT 보조 평가, Surface·negative ROI·depth trade-off, 원본 및 신규 JSON/PNG·NPZ 상대 링크 |
| `local_structural_sampling.json` / `.npz` | 고정 config·fixture manifest·run provenance, C/X와 기존 evidence 참조, group membership·split·capacity·quota·초기점, P/S/F source/XYZ/pixel/object/role, feature availability, 독립 평가 구역·GT-free raw 거리/점유·paired 차이, synthetic 선택/평가, reference별 joint coverage |
| PNG | GT 없는 candidate group 및 독립 평가 격자 overlay, P/S 선택 overview, 일반 거리 분포·최대 미피복 위치, synthetic 비교, Box rim·Peg seam 확대, 구간별 coverage, Surface·negative ROI·depth trade-off, 검증 가능한 Box 간격 |
| 검증·재현 자료 | `verification.json`, 입력/source/artifact 전후 hash, source snapshot, dirty diff와 신규 파일 patch, 테스트 로그, 결과 Markdown snapshot |

HTML의 GT 없는 group 패널과 평가 annotation 패널은 구분한다. 하나의 group을 물리 seam/rim이라고 표시하지 않는다.
결과 Markdown은 요청한 `docs/visual-graph-integration/03f_local_structural_sampling.md`에 작성하고,
새 UUID의 HTML·원본 JSON/PNG·선택 NPZ·검증 자료 및 Task03/03-E에 링크한다.
Markdown의 HTML 링크는 `../../experiments/<new_uuid>/local_structural_sampling.html` 형태의
상대경로로 연결하고, 원본 artifact는 A의 상대경로를 재사용한다. HTML에서도 자기 파일 위치를
기준으로 JSON/NPZ/PNG와 원본 artifact로 이동할 수 있게 연결한다. 구현 승인 후에만 생성·표시를 확인한다.

실험 완료 조건은 계약 테스트와 원본 보존을 충족하고, 같은 실제 입력 및 고정 synthetic fixture에서
선택·평가·raw evidence를 재현 가능하게 남겨 다음 질문에 답하는 것이다.

1. 전체 후보의 국소 공간 보존이 global FPS보다 균형 있게 되는가? 거리·독립 구역 분산이 같은 결론을 지지하는가?
2. 효과가 특정 Box rim 구간에만 국한되는가? Peg/Box 다른 구간과 GT-free 지표에서는 어떤가?
3. 후보 밀도·구조 크기·camera distance/scale·가림 변화에서 어떤 취약성이 드러나는가?
4. 실제 Peg/Box 구조와 일반 Surface mean/p95 보존 사이의 trade-off는 무엇인가?
5. Noise 또는 관측 경계, known-negative ROI·원거리 영역에 선택 예산이 과도하게 배분되는가?
6. 이 근거가 일반적인 geometry sampling 방법의 추가 검증을 정당화하는가, 어떤 가설이 남는가?

일부 구조 악화, local에서도 `rim_axis1_-1` 0/26, synthetic 성능 실패를 모두 그대로 기록한다.
성능 개선이나 일반화 성공은 실험 완료의 의무 조건이 아니다. 구간 coverage 개선과 함께 다른 구조·
Surface 손실을 보고하고, 후보 coverage 향상을 구조 정확도의 향상으로 치환하지 않는다.

**실험 완료와 최종 알고리즘 채택은 별개다.** 새 일반 sampling 방식으로 채택하려면 이후 별도 승인으로
독립 형상·관측·episode에서 동일 설정을 검증해야 한다. 이번 한 frame과 제한된 synthetic 비교만으로
production sampling·distance metric·모델 구조를 확정하지 않으며 GNN/BC 성능 개선도 주장하지 않는다.

이번 및 위 구현 계획의 범위에서 GT component·rim별 quota·hole 특화 규칙, 새 detector/normal estimator,
task-relevance scoring·학습·attention·task-conditioned quota·node feature 변경,
λ/threshold/ratio/budget/seed sweep, depth filtering·outlier 삭제, production sampler·feature encoder·Vision–Graph interface 변경, intra-object edge,
Graph/GNN/BC, PCL/GPU 최적화, dependency 설치·외부 업로드·Git commit은 제외한다.

**현재는 이 기존 계획 문서만 수정하고 중단하며 구현 승인을 기다린다.**
새 진단 계산·FPS·테스트·실험을 실행하지 않고, 새 ExperimentLogger UUID·결과 HTML·결과 보고서·구현 파일도 생성하지 않았다.
Simulator·renderer·CPU/GPU 실험 및 backend/device 검증도 이번에는 실행하지 않았다.
