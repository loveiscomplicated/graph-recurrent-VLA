# Task03-H — Candidate Count-Matched Structural Evidence Analysis 계획

이 문서는 **구현 전 계획**이다. ep000 Frame 0 Box의 FD2/PCA5 저장 orientation 후보를 후보 수와 공간 분포 관점에서 비교한다. 연구 질문은 “PCA5의 높은 Box rim candidate coverage가 더 유용한 위치 분포에서 오는가, 더 많은 후보에서 오는가?”이다. Candidate-centric sampling 개선 제안은 보류한다.

이번 작업에서는 기존 코드·NPZ·JSON을 읽고 이 문서만 작성했다. 새 coverage 계산, 무작위 추출, 테스트, 진단, 결과 HTML, ExperimentLogger UUID 및 실험 artifact는 생성하지 않았다. 아래 구현·검증·산출물은 별도 승인 후 수행할 계획이다.

## 고정 조건

| 항목 | 고정값 |
| --- | --- |
| 입력 | ep000, source Frame 0, `base_camera`, Box만 |
| 후보 | S2 FD2 / S3 PCA5, orientation-only, 기존 threshold 20° |
| 관측 support | 동일 RGB-D·tracker mask·validity·overlap 처리·camera geometry의 11,665개 source node |
| GT reference | 저장된 동일 rim line reference 103행, 원래 행 번호를 reference ID로 유지 |
| Matching | 동일 point/reference pair에서 pixel 거리 ≤2px **AND** camera XYZ 거리 ≤0.003m |
| Count matching | k=297, 균등 비복원 추출 |

`structural_ratio`, 최종 256점 budget, allocation, FPS 초기점은 비교 변수가 아니다. Priority/Surface/Final 선택 집합을 평가하거나 FPS를 재현하지 않는다. Task02-C의 orientation+depth union을 Task03 orientation-only 대신 사용하지 않는다.

## A. 기존 저장 데이터 재사용

### 확인한 입력과 배열

아래 파일은 현재 호스트에서 존재와 저장 schema를 읽어 확인했다. Hash의 현재 파일 일치 여부와 배열 간 exact equality는 승인 후 검증하며, 기존 검증 기록을 이번 실행의 검증으로 대체하지 않는다. 대용량 원본과 `experiments/`는 Git에서 관리되지 않아 다른 checkout에는 없을 수 있다.

| 자료 | 재사용 경로·키 | 역할 |
| --- | --- | --- |
| Task03 | [comparison.json](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json), [Frame0 NPZ](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz), 같은 디렉터리 `f0000_metrics.json` | 원본 config, source snapshot/hash, 후보와 GT reference의 기준 |
| FD2 후보 | Task03 NPZ `evidence/fd2/1/20/orientation_flat`: `(297,) int64` | 전체 후보 source flat index, 저장 순서 유지 |
| PCA5 후보 | Task03 NPZ `evidence/pca5/1/20/orientation_flat`: `(599,) int64` | 전체 후보 source flat index, 저장 순서 유지 |
| GT reference | Task03 NPZ `evaluation/box/reference_uv`: `(103,2) float64`, `reference_xyz`: `(103,3) float64` | 새로운 reference를 만들지 않고 그대로 사용 |
| Task01 dense geometry | [f0000.npz](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.npz), 같은 디렉터리 `geometry.json` | `xyz` `(512,512,3) float32`, `valid_depth`, `overlap_mask`, `box/object_mask`, `box/valid_points`, `intrinsic`, `cam2world`, `source_frame`, `camera` |
| Task03-E | [저장 NPZ](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.npz), [저장 JSON](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.json) | `S2/box/` 및 `S3/box/` 아래 `candidate_flat`, `candidate_xyz`, `support_flat`, `support_xyz`를 직접 로드 |
| Task03-E 후보 matching | 같은 NPZ의 `{S2,S3}/box/1.0/C/` 아래 `joint_node_row`, `joint_reference_id`, `matched_node`, `covered`, `reference_count_per_node` | 후보–reference joint pair를 저장 정보에서 재구성. `1.0`은 저장 namespace이며 FPS 실행 지시가 아님 |
| Task03-E reference/구간 | 같은 NPZ `box/reference_uv`, `box/reference_xyz`, `box/observable`, `box/reference_segment` | reference exact 대조, 기존 네 rim 구간과 ambiguity 재사용 |
| Phase1 | [diagnostic.json](../../visual/results/diagnostic_previews/03d_phase1/diagnostic.json), 같은 디렉터리 `verification.json` | `frames[0].conditions`, `frames[0].S2_vs_S3.C`의 기존 결과 대조 |
| Phase2/3 | [Phase2 JSON](../../visual/results/diagnostic_previews/03d_phase2/priority_fps_trace.json), [Phase3 JSON](../../visual/results/diagnostic_previews/03d_phase3/priority_fps_seed_control.json) | 기존 후보 provenance 확인 자료. Trace/seed 분석 재실행 없음 |
| Task02-C 저장 평가 영역 | [f0000_pairs.npz](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_pairs.npz) | `evaluation/box/`의 기존 ROI와 `rim_uv/xyz`, 다른 physical `seam_uv/xyz`만 사용. 후보 union은 사용하지 않음 |

Task03-E의 `candidate_xyz`는 FD2 `(297,3)`, PCA5 `(599,3)`이며, 두 `support_flat`은 `(11665,)`, `support_xyz`는 `(11665,3)`이다. 원본 XYZ를 변환·재추정하지 않고 source index로 gather하여 exact 대조한다. UV는 같은 source raster의 `(u,v)`, 저장 `pixels_vu`는 `(v,u)`임을 구분한다.

Task03 config에 기록된 원본은 `visual/data/step2/ep000.npz`와 `visual/results/runtime_tracking/ep000/masks.npz`다. 저장된 SHA-256은 각각 `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`, `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`다. 승인 후 실제 파일 hash를 다시 대조한다.

### 기존 helper와 재사용 경계

- [sampling_evaluation.py](../../visual/runtime/test/sampling_evaluation.py)의 `CoverageMatcher(geometry, BOX, reference, tolerances=((2., .003),))`, `reference_hits()` 및 `adjacencies['2px_3mm']`를 사용한다. 기존 기본값의 다른 tolerance는 실행하지 않는다. `reference_hits()`는 source membership/중복 검사와 node/reference 구분을 제공한다.
- `matcher.flat`에 candidate source index를 exact 매핑한 adjacency 행에서 reference별 `getnnz(axis=0)`와 node별 `getnnz(axis=1)`를 구할 수 있다. Task03-E의 저장 COO pair 목록도 같은 boolean adjacency로 재구성해 대조할 수 있다. 기존 helper에 expected coverage 기능은 없으므로 새 평가 runner의 작은 순수 함수로 추가한다.
- [run_rim_bottleneck.py](../../visual/runtime/test/run_rim_bottleneck.py)의 `source_contract()`, `require_equal()`, `occupancy()`를 재사용한다. `occupancy()`에는 `X`와 이번 네 비교 집합만 전달하며, 개별 ROI와 중복 제거 union을 분리한다.
- [run_structural_edges.py](../../visual/runtime/test/run_structural_edges.py)의 `negative_regions()`는 기존 Box ROI와 rim+기타 physical seam reference를 받아 기존 구조 주변 제외 규칙을 재현한다. Task03-E `evaluation_inputs()`의 **Box 처리 부분**을 근거로 호출한다. 새 ROI·오염 판정 규칙을 만들지 않는다.
- [run_priority_fps_metric_control.py](../../visual/runtime/test/run_priority_fps_metric_control.py)의 저장 배열·구간 label을 우선 사용한다. `rim_labels()`는 기존 label identity 대조가 필요할 때만 평가용으로 재사용한다. GT reference 생성 helper인 `rim_reference()`는 의미 확인에만 참조하고 새 reference를 만들지 않는다.
- Task03-E의 `diagnose()`, `select()`, `execute_controls()`, `baseline_gate()`, `score_case()`나 Phase2/3 상위 runner는 호출하지 않는다. FPS와 Surface 평가를 포함하는 `load_inputs()` 전체 및 `staged_reference_coverage()`도 이번 비교를 위해 호출할 필요가 없다. Dense XYZ가 있으므로 `geometry_evidence()`의 backprojection도 필요 없다.

### 기존 결과로 답할 수 있는 부분

다음은 이번에 계산한 값이 아니라 Phase1/Task03-E 저장 결과와 사용자 제공 사실이다.

| 집합 | node 수 | rim matching node | rim matching node fraction | unique reference |
| --- | ---: | ---: | ---: | ---: |
| 전체 관측 X | 11,665 | 153 | 153/11,665 | 103/103 |
| FD2 original | 297 | 101 | 34.01% | 57/103 |
| PCA5 original | 599 | 127 | 21.20% | 91/103 |

기존 후보 reference 교집합은 56개, FD2-only 1개, PCA5-only 35개다. 저장 Task03-E의 구간별 covered reference 수(FD2/PCA5)는 `rim_axis1_+1` 3/24개, `rim_axis1_-1` 26/26개, `rim_axis2_+1` 2/15개, `rim_axis2_-1` 26/26개다. 여기서 slash는 두 estimator의 값 구분이며 비율이 아니다. 구간 reference 수는 각각 25·26·26·26개다. 이번 진단의 구간별 coverage 비율도 전체 103개를 분모로 사용하고, 구간 크기는 별도 count로 표시한다.

원래 후보의 matching node/reference ID, ROI/depth 요약, 구간별 coverage는 기존 JSON에 있다. Reference별 multiplicity는 저장 pair 목록으로 복원할 수 있다. **Multiplicity 전체 표·candidate source 교집합/차집합·count-matched 기대값·random control·고정 subset 평가·새 overlay는 승인 후 새 계산/집계가 필요하다.** 현재 자료만으로 동일 297개에서 PCA5의 이점이 유지되는지 결론 내리지 않는다.

## B. Count-matched 진단 설계

### 비교 집합과 GT-blind 추출

| 조건 | 모집단 | 평가할 node 수 | 용도 |
| --- | --- | ---: | --- |
| FD2 original | 저장 FD2 orientation 후보 | 297, 전부 | 고정 기준점 |
| PCA5 original | 저장 PCA5 orientation 후보 | 599, 전부 | 원래 coverage와 중복 분포 |
| PCA5 matched | 같은 599개 후보 | 297 | 후보 수를 맞춘 위치 분포 비교 |
| Observed random control | 동일 Box 유효 관측 X 11,665개 | 297 | 임의 관측점 대비 rim 선택 집중도 대조 |

선택 함수에는 모집단 길이 N, k, RNG seed만 전달한다. `numpy.random.Generator(PCG64(SeedSequence(...)))`의 `choice(N, size=297, replace=False)`로 행을 균등 추출한다. XYZ·normal score·GT·matching·ROI·구간 label을 선택 함수에 전달하지 않는다. 추출 행을 정렬해 저장 source 순서를 유지할 수 있으나, 선택 membership과 원본 index/XYZ 값은 바꾸지 않는다. 관측 대조군도 candidate/normal-valid 조건을 추가하지 않고 전체 유효 Box 관측 support에서 고른다.

Stochastic 보조 분석은 **조건별 16회**로 사전 고정한다. Repeat ID `r=0,…,15`, 명시적 seed 목록 `20261024,…,20261039`, `SeedSequence([20261024+r, stream])`를 사용한다. Stream은 PCA5 matched=0, observed random=1이다. 매 repeat/조건마다 새 PCG64를 생성하므로 조건 실행 순서와 GT annotation이 draw를 바꾸지 않는다. 반복 사이 subset 중복은 허용되며 각 subset 내부에는 중복이 없다.

공간 overlay 대표는 두 조건 모두 **r=0**으로 고정한다. 좋은 결과의 seed를 고르거나 반복 수/seed를 결과에 따라 바꾸지 않는다. 16회 coverage/node fraction/ROI 등의 mean·sample SD·min/max와 개별 결과를 모두 보존한다. 이 변동은 고정 frame·후보 집합의 부분추출 변동이며 detector 일반화의 신뢰구간이 아니다. 기대 coverage의 추정에는 이 표본 평균을 사용하지 않는다.

### Reference multiplicity 기반 정확한 기대값

각 모집단에 대해 boolean joint adjacency를 A라 하자. `A[i,j]`는 node i와 저장 reference j가 **동일 pair**에서 두 tolerance를 동시에 통과했음을 뜻한다. `h_j = sum_i A[i,j]`이고 reference j의 missing/covered 여부는 `h_j=0`/`h_j>0`이다. 한 node의 복수 reference matching과 한 reference의 복수 node matching을 모두 보존한다.

고정 N개 중 k개를 균등 비복원 추출할 때:

\[
p_j(k)=1-\frac{\binom{N-h_j}{k}}{\binom Nk},\qquad
E[U_k]=\sum_{j=0}^{102}p_j(k),\qquad
E[\mathrm{coverage}_k]=E[U_k]/103.
\]

Reference 사건들이 독립일 필요는 없다. 기대값의 선형성으로 합산이 정확하다. 다만 reference 간 공유 node 때문에 `sum p_j(1-p_j)`를 전체 coverage 분산으로 사용해서는 안 된다. 이번 최소 계획에서는 정확한 분산/공분산 구현을 추가하지 않고 위 16회로 변동을 기술한다.

큰 조합 수나 factorial을 float으로 만들지 않는다. 다음 유한곱의 log를 float64로 계산한다.

\[
\log q_j=\sum_{i=0}^{k-1}\log1p\!\left(-\frac{h_j}{N-i}\right),
\qquad p_j=-\operatorname{expm1}(\log q_j).
\]

대칭인 `q_j = product_{i=0}^{h_j-1}(1-k/(N-i))`도 같으므로 짧은 쪽 `min(k,h_j)`을 택한다. 합은 `math.fsum`으로 누적한다. `-expm1`로 작은 coverage 확률에서 `1-exp(...)`의 cancellation을 피한다. `h_j=0` 또는 `k=0`이면 p=0, `k>N-h_j`이면 p=1을 먼저 처리한다. `0≤h_j≤N`, `0≤k≤N`을 검증하고 N=0은 k=0인 경우만 허용한다. 이 값은 조합론적 정확식의 수치 평가이며 무작위 근사값과 구분해 저장한다.

PCA5 `(N=599,k=297)`와 observed control `(N=11665,k=297)`에 같은 식을 적용한다. FD2 `(N=k=297)`는 저장 coverage와 같아야 한다. Rim matching node 총수가 H인 모집단의 기대 matching node 수는 `kH/N`, 기대 node fraction은 `H/N`이며, reference별 기대 multiplicity는 `kh_j/N`이다. 실제 subset의 정수 node/reference 수와 비정수 기대값을 별도 필드·열에 기록한다. GT matching은 선택 이후의 평가 정보이며 선택 확률을 바꾸지 않는다.

Multiplicity로 coverage 포화를 분석할 수 있다. **새 subset 실행 없이** FD2는 정수 k=0…297, PCA5는 k=0…599, observed control은 k=0…297의 위 기대값을 계산한 곡선을 그린다. 고정 103개 분모와 k=297 선을 사용하고 추가 node당 기대 coverage 증가도 표시한다. 이는 고정 후보의 analytic rarefaction이며 threshold/budget/FPS seed sweep이 아니다. `h_j=0`인 위치는 후보 수를 원래 N까지 늘려도 덮을 수 없고, 큰 h가 같은 reference에 쌓이면 해당 reference의 추가 기여는 포화된다는 점을 보여준다.

### 지표·공간 분포

- 각 집합의 총 node 수, 적어도 한 reference와 매칭된 실제 node 수, **rim matching node fraction(특정 rim 집중도)**, unique reference 수/103을 기록한다. 이 fraction을 Box 전체 structural edge에 대한 detector precision으로 부르지 않는다.
- Reference 103행의 h, covered/missing ID, `h=0`, `h=1`, `h≥2` 분포 및 node별 reference 수를 보존한다. `sum(h)`는 pair 수이며 matching node 수도 unique reference 수도 아니다.
- FD2/PCA5 original의 candidate source index 교집합·양방향 차집합과 covered reference 교집합·양방향 차집합을 따로 저장한다. Subset은 같은 reference 공간에서 FD2와 비교하며, 기존 56/1/35 reference 분류에 대한 보존 확률/실제 보존도 표시한다.
- 기존 `box/reference_segment`를 기준으로 네 rim 구간의 covered/missing reference와 기대 coverage를 기록한다. 각 구간 covered 수/103을 표시하고 구간의 고정 reference 수는 별도 count로 기록한다. Corner/multiple membership은 기존 ambiguous 처리를 유지하고 node 수를 구간별로 중복 합산하지 않는다.
- 기존 negative ROI별 node 수·집합 대비 fraction·관측 ROI 대비 점유와 OR union의 중복 제거 점유를 따로 기록한다. ROI 이름 `contamination`을 포함해 ROI 안의 모든 후보를 false edge로 분류하지 않는다.
- Depth min/median/p95/max와 전체 관측 X의 median/p95로 고정한 depth band별 점유를 `occupancy()`로 기록한다. Camera XYZ의 XY/XZ 산점도, source UV 분포와 rim 구간별 reference heatmap으로 위치 차이를 보여준다. 원거리 positive depth 자체는 오염 정답이 아니다.

Primary 비교는 `E[U_PCA5,297] − U_FD2,297`이다. 별도로 PCA5 original−FD2 original, PCA5 original−PCA5 matched 기대값, 두 estimator−observed random 기대값을 node 수/fraction/reference coverage/ROI 관점에서 표로 기록한다. Original−matched 차이는 동일 PCA5 모집단의 무작위 후보 수 감소 효과이고, matched−FD2 차이는 두 후보 집합의 위치·중복 분포 차이를 포함한다. 서로 다른 estimator의 차이를 독립적인 인과 기여율로 분해하지 않는다.

기대 coverage의 PCA5 우위가 남으면 후보 수만으로 원래 차이를 설명하기 어렵다는 근거다. 우위가 사라지면 원래 높은 coverage가 후보 수에 상당히 의존한다는 근거다. 추가 PCA5 후보가 모두 noise라는 결론은 내리지 않는다. 두 estimator가 observed random보다 rim matching fraction/coverage가 높으면 임의 관측점과 다른 rim 주변 선택 선호의 근거로 해석한다. 한 subset의 승패로 결론을 바꾸지 않으며, 한 frame/Box의 기술적 비교를 모든 형상·독립 detector 일반화·GNN/BC 효용의 우열로 확장하지 않는다.

## C. 최소 변경 파일

승인 후 추가할 Python 파일은 다음 두 개다. 평가 로직·수치 함수·정적 보고서 생성은 작은 runner 안에 두고 공용 인터페이스를 확장하지 않는다.

| 경로 | 역할 |
| --- | --- |
| `visual/runtime/test/run_candidate_count_matched.py` | 저장 입력 gate, GT-blind subset, expected coverage, adjacency/multiplicity 평가, JSON/NPZ/PNG/HTML·결과 Markdown 작성 |
| `visual/runtime/test/test_candidate_count_matched.py` | 선택 독립성·비복원·수치 기대값·same-pair·집계 일관성·불변성의 synthetic 계약 테스트 |
| `docs/visual-graph-integration/03h_candidate_count_matched.md` | 승인 후 작성할 결과 보고서. 기존 결과/계획 문서 덮어쓰기 없음 |

이번에 추가하는 파일은 `docs/visual-graph-integration/03h_candidate_count_matched_plan.md` 하나다. Production estimator/normal/candidate/sampling/allocation, 기존 D/E helper·runner·테스트, 공통 logger/publisher는 변경하지 않는다. 저장 schema가 실제 gate와 맞지 않으면 재검출·허용오차 변경·다른 후보 대체 없이 gap을 보고하고 중단한다.

## D. 검증 및 산출물 계획

### 실행 전·후 검증

1. **입력/provenance:** Task03 UUID/config, Task01/03의 input/mask SHA-256와 camera, Frame 0, intrinsic/cam2world, source frame 대응을 확인한다. `source_contract()`로 기존 평가 의미와 원본 snapshot/hash를 확인한다. 실제 사용 helper와 새 runner의 source snapshot·SHA-256, Python/NumPy/SciPy 버전, Git commit/dirty를 기록한다. 기존 미커밋 작업은 보존하며 commit만으로 재현 가능하다고 하지 않는다.
2. **동일 support/XYZ:** Task03-E 양쪽 support의 source index/order/XYZ가 exact 일치하고 dense `box/valid_points` 및 `OrganizedGeometry.observed_mask(BOX)`와 같은지 확인한다. XYZ의 Z와 원본 depth, 유효성·mask·overlap 및 source membership을 대조한다. FD2/PCA5 후보 수는 297/599, 모두 unique이며 동일 support 안에 있어야 한다. Candidate 배열은 Task03 orientation-only와 exact 일치하고 XYZ는 dense source gather 및 Task03-E 저장값과 같아야 한다.
3. **선택 독립성/중복:** GT를 전혀 받지 않는 선택 함수를 먼저 실행해 seed·모집단 hash·draw 행·source index를 저장한다. 이후 GT/ROI annotation을 평가한다. Synthetic GT/label/score/XYZ를 바꾸거나 순서를 바꿔도 N/k/seed가 같은 draw membership을 유지하는지 확인한다. 매 subset은 정확히 297개 unique이며 올바른 모집단의 부분집합이어야 한다. 원본 후보/source 값과 XYZ는 그대로여야 한다.
4. **정확식 synthetic 검증:** 작은 N의 모든 k-subset을 열거해 기대 unique reference 수와 exact rational 조합식/안정적 log 계산을 비교한다. h=0, h=N, k=0, k=N, k>N−h, N=0/k=0, 잘못된 범위와 작은 p를 포함한다. 한 node가 여러 reference를 덮고 여러 node가 같은 reference를 덮는 fixture로 reference 독립성 가정 없이 기대합이 맞는지 확인한다. k별 단조성, 0≤E[U]≤103 및 k=N에서 원래 unique coverage 복원도 검사한다.
5. **Same-pair/집계:** Pixel-near node와 XYZ-near node가 다를 때 match로 합성하지 않는 synthetic fixture와 tolerance 경계·중복 pair 계약을 확인한다. 새 matcher와 Task03-E 저장 C pair/covered/matched-node 배열을 exact 대조한다. `count(h>0)=unique refs`, `count(row degree>0)=matching nodes`, `sum(h)=boolean adjacency nnz`를 확인한다. 전체 관측/후보의 153/101/127 matching nodes와 103/57/91 refs, 기존 reference 교집합 56/1/35도 재확인한다.
6. **ROI/분모:** 기존 original 후보의 ROI/depth 요약과 구간 coverage를 저장 JSON과 대조한다. ROI union은 OR이며 개별 ROI 합으로 만들지 않는다. Subset reference ID는 원래 모집단 covered ID의 부분집합이어야 한다. Coverage는 모두 /103이며 실제 node 수·reference 수·pair 수·기대값 필드를 구분한다. Depth bands는 같은 X에서만 정한다.
7. **보존/보고서:** 원본 episode/mask/Task01/Task03/D/E NPZ·JSON과 관련 source/기존 sidecar를 실행 전후 hash로 비교한다. 원본 입력에는 `link_output()`이나 sidecar 갱신을 하지 않는다. NPZ readback으로 dtype/shape/source/선택 배열을 확인하고 Markdown↔HTML 및 JSON/NPZ/PNG 상대 링크, PNG decode와 HTML 이미지 로드를 검증한다. 실패 시 새 결과의 실패 상태만 남기고 원본/평가 정의를 조정하지 않는다.

승인 후 root `.venv`에서 새 `test_candidate_count_matched`와 기존 `test_sampling_evaluation`의 관련 계약을 실행하고 `tests.log`를 남긴다. FPS를 실행하는 테스트 모듈은 이번 필수 검사에 포함하지 않는다. 공통 인프라 코드의 변경 필요가 생기면 먼저 gap을 보고하고 공통 계약 테스트 범위를 재검토한다. 현재 계획 단계에서는 어떤 테스트도 실행하지 않았다.

### ExperimentLogger와 정적 산출물

승인 후 `ExperimentLogger(team='visual', job_type='evaluate', ...)`를 `WANDB_MODE=disabled`로 사용해 **새 UUID**의 `experiments/<run_id>/`에 저장한다. 기존 run ID/project/group 규칙을 유지한다. Config에는 k/seed 전체 목록/stream 규칙/조건·source UUID/입력 hash/tolerance/후보 종류·selection에 GT를 사용하지 않는다는 계약을 기록한다. `record()`에는 simulator/GPU 미실행과 source provenance를 보존한다.

| 산출물 | 내용 |
| --- | --- |
| `candidate_count_matched.json` | 원래/동일 개수/대조군 지표, 기대값·개별 16회·요약, 차이, source/reference 집합 비교, 해석 한계 |
| `candidate_count_matched.npz` | 원본 후보/관측 source index·XYZ, 각 draw 행/index/XYZ, 고정 103개 UV/XYZ/구간 label, boolean joint pair 목록·node-hit·reference multiplicity·coverage 확률과 곡선 |
| `candidate_original.png` | FD2 297/PCA5 599의 전체 Box 위치와 같은 rim crop 비교 |
| `candidate_matched.png` | FD2 original/PCA5 matched/observed random 각각 297개, r=0의 동일 crop overlay |
| `reference_multiplicity.png` | 같은 reference 행/기존 구간의 중복·missing heatmap, 원래 집합과 matched 기대값/대표 subset 구분 |
| `expected_coverage.png` | candidate 수에 따른 정확한 기대 coverage 포화 곡선, k=297 및 FD2 고정값 표시 |
| `roi_depth_spatial.png` | 기존 known-negative ROI/union, 동일 depth band 분포, camera XY/XZ 분포 |
| `candidate_count_matched.html` | 아래 정적 HTML 구성, 결과 Markdown/JSON/NPZ/PNG 상대 링크 |
| `verification.json`, `source_hashes.json`, `tests.log`, source snapshot | 검증 상태·불변성·재현성 근거 |

Crop은 결과를 보기 전 고정한다. 기존 저장 Box `hole_rim` mask bounding box에 12px padding, 이미지 경계 clip을 적용하고 모든 패널에 같은 extent/marker 크기·alpha·색상 정의를 사용한다. 전체 Box/support view도 함께 제공해 crop 밖 후보를 감추지 않는다. Matching node와 nonmatching 후보, covered/missing reference를 다른 기호로 표시한다. Reference는 기존 구간 label과 검증된 local 축 순서로 표시하며 reference 행 번호를 물리 순서로 간주하지 않는다. GT/ROI 색은 선택 완료 후 annotation이다.

HTML에는 원래 FD2/PCA5 후보 위치, 같은 297개 overlay, node 수와 unique reference coverage 표, 기대 coverage와 고정 반복 변동, reference별 중복·누락, 기존 ROI/depth 분포, 원본 provenance 및 결과 링크를 포함한다. 별도 web 서비스·새 dependency 없이 간단한 정적 HTML/PNG로 만든다.

결과 Markdown `docs/visual-graph-integration/03h_candidate_count_matched.md`에서 `../../experiments/<run_id>/candidate_count_matched.html`로 연결하고, HTML에서 `../../docs/visual-graph-integration/03h_candidate_count_matched.md`로 역연결한다. HTML 내부 JSON/NPZ/PNG는 같은 디렉터리의 상대 파일명으로 연결한다. 실제 UUID는 승인 후 실행 때 생성하며 지금 placeholder 디렉터리를 만들지 않는다.

새 산출물만 기존 `link_output()`/`utils.experiment_artifacts`의 로컬 등록을 재사용한다. `tools.publish_experiment`와 외부 전송은 실행하지 않는다. Simulator/renderer/CUDA/MPS/GPU는 이번 계획에서 실행하지 않았고, 승인 후 진단도 저장 geometry의 CPU offline 평가로 한정한다.

## 중단 범위

계획 제출 후 중단한다. 구현·diagnostic 실행은 별도 승인 대상이다. Priority/Surface FPS, 새로운 detector/normal estimation, threshold 변경·sweep, GT 기반 subset 선택·top-K·quota, 공간 thinning, PCL, Adaptive Budget 재실험, Task-Relevance Scoring, Graph/GNN/BC·encoder 변경, 추가 frame/episode·simulator/GPU 실험, production sampler 변경, dependency 설치, 외부 업로드 및 Git commit은 수행하지 않는다. 기존 03-D/G 원인 분석이나 repository 전체 재탐색도 반복하지 않는다.
