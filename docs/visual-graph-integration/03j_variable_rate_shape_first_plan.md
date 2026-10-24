# Task03-J — Variable-Rate Shape-First Sampling：구현 전 계획

상태: **계획 제출. 별도 승인 전까지 구현·Sweep을 실행하지 않는다.**
이번 산출물은 이 Markdown 하나다. Production Sampling, Graph, 기존 실험 source와 결과를 변경하지 않는다.
ExperimentLogger UUID, 결과 Markdown, HTML은 아직 만들지 않는다.

최소 구현안은 **PCA5 orientation 후보와 outer-mask silhouette 후보를 독립 Voxel 압축 →
동일 keep_ratio로 deterministic thinning → 원본 source 합집합 → 전체 observed support의 Surface FPS**다.
기본 검토 조건은 `keep_ratio=0.5`이며, 가변 N과 형상 보존의 trade-off를 확인한다.
`num_points=256`, 물체별 Priority 32, `structural_ratio=.25`는 새 경로의 상한으로 사용하지 않는다.
이 문서는 이번 feasibility용 계약 제안이며 팀 전체의 최종 모델·Graph 구조를 확정하지 않는다.

## 1. 현재 구현과 재사용 범위

루트 README, ManiSkill README, PROJECT_CONTEXT, engineering/tracking/artifact 지침,
Visual/runtime README, Task03-I 계획·결과와 아래 실제 source를 확인했다.
기존 dirty/untracked 변경은 작업 시작 상태로 보존한다.

| 확인한 파일/API | 현재 동작 | 03-J 재사용·변경 경계 |
| --- | --- | --- |
| [rgbd.py](../../visual/runtime/rgbd.py): `TrackedRGBDFrame`, `iter_saved_rgbd` | 저장 RGB-D/mask를 `source_frame`으로 연결. CV camera m, tracker 1/2 → PEG 0/BOX 1 | 관측·camera·TCP kinematics만 사용. detector/tracker/simulator 재실행 없음 |
| [geometry.py](../../visual/runtime/geometry.py): `OrganizedGeometry.observed_mask` | `object_mask & finite_positive_depth & ~overlap` | X의 유효성·원본 XYZ/pixel 계약 유지 |
| [sampling.py](../../visual/runtime/sampling.py): `ObjectCandidates`, `SampledPoints` | raster source, object ID, role와 metadata 보존. 내부 NumPy/CV 좌표 출력 | 기존 형식은 참고하되, 새 실험 출력에 별도 provenance를 추가. 기존 `sample_points`는 호출·변경하지 않음 |
| 같은 파일: `sample_points`, `SamplingConfig` | 전역 budget/structural ratio 배분, 후보 FPS cap, 남은 Surface FPS | 새 선택 경로에는 사용하지 않음. 256/32/.25를 우회하지 못하는 wrapper를 만들지 않음 |
| 같은 파일: `fps_indices(points, count, seed, initial)` | initial을 포함한 **전체** 목표 count. unique 원본 row. initial이 비면 RNG로 첫 row 선택 | 전체 X의 3D Surface 추가에만 재사용. `count=len(P)+requested_surface` 의미로 호출 |
| [Task03-I source](../../visual/runtime/test/run_voxel_structural_sampling.py): `nominal_scale`, `voxel_keys`, `representatives` | X median Z 기반 nominal metric scale, camera-origin grid, centroid-nearest 원본 후보 대표점 | 이 세 pure helper를 그대로 import. source별 독립 실행. 기존 파일 수정·복사 구현 없음 |
| 같은 파일: `choose_priority`, `select`, `run` | anchor mapping, cap32 FPS, final target 129/127, 기존 실험 실행 | **호출하지 않음.** 기존 cap/anchor/dyadic 진단 경로를 가져오지 않음 |
| [Task03-E feature helper](../../visual/runtime/test/run_priority_fps_metric_control.py): `feature_rows` | 같은 S3 저장 source에 있는 23개 metadata field만 exact gather | source join/availability 방식 재사용. 다른 estimator/조건 feature 혼합 없음 |
| [evaluation](../../visual/runtime/test/sampling_evaluation.py), Task03-I 평가 helper | 3D 거리, 단계별 reference coverage, ROI/depth 보조 평가 | 선택 freeze 후 평가 함수만 재사용. Frame0 전용 loader/고정 count assertion은 재사용하지 않음 |
| [Vision–Graph 설계](00_interface.md), [sampling API 설명](../../visual/runtime/sampling.md) | 전자는 설계 문서, 후자는 최종 EE tensor/edge packaging 미구현 명시 | 기존 256/259 설명은 이전 정책의 기록. 03-J에는 적용하지 않음 |
| [GraphBuilder](../../graph/prototype/graph_builder.py), [v3 Graph](../../graph/prototype/v3/interaction_graph.py) | GT 기반 peg/hole geometry 입력, robot-first, 내부 edge 재구성, 현재 node feature 생성 | Vision packet 입력 경로가 아님. provenance를 받는 API 없음. 기존 GNN 경로 수정·강제 연결 없음 |
| [GT node accessor](../../graph/gt_nodes.py) | simulator GT로 robot/물체 geometry 생성 | 선택 입력으로 사용하지 않고 simulator 호출 없음 |

[Task03-I 결과](03i_voxel_structural_sampling.md)는 Frame0의 nominal R을 Peg 432 / Box 356개,
cap 이후 P를 각각 32개로 보고했다. R→P에서 추가 reference 손실이 컸다는 관측이 출발점이다.
이는 이번 ratio의 성능 예측이나 silhouette 추가의 인과적 근거가 아니다.
PCA5와 silhouette를 함께 도입하므로 예전 실험 대비 차이를 thinning만의 효과라고 해석하지 않는다.

**실제 연결 gap:** repository의 Graph prototype은 Vision의 `points`, `object_id`,
`geometric_features`, `intra_object_edge_index`를 수신하는 consumer가 아니다.
Sampling도 EE 변환, 최종 feature encoding, intra-object edge를 완성하지 않는다.
따라서 이번에는 새 실험 packet과 최소 Graph 수신 계약까지 검증하고, 기존 GNN/BC end-to-end 완료를 주장하지 않는다.

## 2. 사전 고정 관측과 비교 조건

| 항목 | 고정값 |
| --- | --- |
| Episode/camera | 기존 `ep000`, `base_camera` |
| 평가 source frame | **0, 45, 80**, 이 순서. 3개 모두 보고하며 결과에 따른 교체·추가 없음 |
| PCA5 | 기존 optimized PCA5 orientation **strict >20°**, 저장 후보만 사용 |
| Voxel | Task03-I nominal 공식, multiplier 1, camera origin `(0,0,0)`, 물체별 a |
| Main arms | `keep_ratio ∈ {1.0, 0.75, 0.5, 0.25}`, phase 0 |
| Diagnostic arm | ratio .5, phase 1. 모든 지정 frame, 양쪽 source에 동일하게 적용 |
| Surface 추가 요청 | 모든 지정 frame/ratio에서 Peg 97 / Box 95 |
| 최종 point 수 | `N_total = N_structural_unique + N_surface_added`, 전역 cap 없음 |
| Randomness | Structural RNG 없음. Surface 첫 initial도 아래 규칙으로 명시; 최적 seed 선택 없음 |

주 실험은 3 frame × 4 ratio = 12개 frame-condition, phase 진단은 3개다.
물체별 결과와 frame 전체 결과를 함께 기록한다. Phase 1을 최종 정책으로 고르지 않는다.
Surface 97/95는 Frame0의 과거 추가 수를 유지한 첫 비교용 설정이며 최종 상한·고정 정책이 아니다.
다른 frame도 같은 요청 수를 적용하여 ratio 외 변수가 바뀌지 않게 한다.
물체가 비거나 capacity가 부족하면 실제 수와 shortfall을 기록하고 다른 물체로 재분배하지 않는다.

| 기존 입력 | 사용할 내용 |
| --- | --- |
| [episode](../../visual/data/step2/ep000.npz), [tracker masks](../../visual/results/runtime_tracking/ep000/masks.npz) | 원본 RGB-D/K/cam2world/TCP, raw object mask, source frame mapping |
| [Task01 dense directory](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a) | `f0000.npz`, `f0045.npz`, `f0080.npz`: XYZ/valid_depth/overlap/object masks |
| [Task03 sampling directory](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f) | 같은 frame의 `fNNNN_samples.npz`: `evidence/pca5/{0,1}/20/orientation_flat`, `S3_n256_r0.25_t20_s0/*` feature와 과거 선택 |
| [Task03-I run](../../experiments/92f29716e76f4b998fe924f08e80357d) | Frame0 원본 후보·nominal scale/대표점과 기존 결과의 read-only 대조 |

이번 확인은 파일 존재와 NPZ field/shape/dtype inspection뿐이다. 지정 frame별 후보 수·윤곽선·새 선택·coverage는 계산하지 않았다.
Dense 저장 frame 목록은 0/15/18/45/80/120이다. 20은 dense 파일이 없어 평가 목록에 넣지 않았다.
0/45/80은 필요한 저장 입력이 있는 소수 관측으로 사전 고정했으며 metric 결과로 선정하지 않았다.
승인 후 입력 hash·source_frame·mask·K·XYZ exact alignment를 먼저 검사한다.
불일치/누락 시 해당 원인을 보고하고 중단하며 결과에 맞춰 frame을 바꾸지 않는다.

## 3. 후보와 최소 관측 유효성 규칙

물체별 원본 observed support를 X, PCA5 후보를 `C_pca`, silhouette 후보를 `C_sil`이라 한다.
모든 집합은 `(episode, camera, source_frame, object_id, flat_index)`로 식별한다.
`flat_index=v*W+u`이며 point는 반드시 저장 X에서 exact gather한다.

### 3.1 PCA5

저장 `orientation_flat` 순서인 source ascending을 유지한다. `C_pca ⊆ X`와 중복 없음,
원본 XYZ 일치를 확인한다. normal/score/threshold를 재계산하거나 다른 PCA 구현으로 대체하지 않는다.
PCA5는 5×5 same-object connected support의 법선을 인접 pair에서 비교한 evidence다.
PCA5 evidence 자체도 physical sharp edge 정답으로 간주하지 않는다.

### 3.2 Outer-mask silhouette

**depth 유효성이나 overlap을 적용하기 전 raw tracker mask**에서 외부 contour를 구한다.
기존 OpenCV 사용 방식의 `RETR_EXTERNAL`과 `CHAIN_APPROX_NONE`을 사용하며 설치된 API의
반환형·좌표형을 테스트로 확인한다. 새 dependency를 설치하지 않는다.

1. Raw mask의 외부 contour만 추출한다. Internal hole contour는 후보에 넣지 않는다.
   depth-valid mask에서 contour를 구해 invalid-depth 구멍을 새 silhouette로 만드는 것을 금지한다.
2. 각 contour를 원본 mask pixel의 `(v,u)`/flat source walk로 저장한다.
   반환 중복 closure를 제거하고, 시작점은 최소 flat으로 canonicalize한다.
   같은 시작점을 갖는 forward/reverse cyclic sequence 중 lexicographic 최소를 방향으로 택한다.
   퇴화 walk의 동일 source 재방문은 기록하고 후보 순서에는 첫 출현만 유지한다.
3. Contour/component는 canonical sequence의 최소 source, 이어 전체 sequence로 정렬한다.
   별도 walk 위치와 segment ID를 보존하여 떨어진 성분을 하나의 연속선으로 보지 않는다.
4. 그 뒤 contour pixel ∩ X만 `C_sil`로 사용한다. Invalid depth/overlap 때문에 빠진 위치는
   원본 walk에 exclusion marker로 남긴다. 이 틈을 가로질러 contour를 보간·연결하지 않는다.
   Image border에 닿는 구간은 truncation flag를 남기며 frame 밖 윤곽을 추정하지 않는다.

Silhouette는 **관측상 물체 mask 외곽**이다. Robot occlusion에 의해 생긴 외곽도 이 evidence에
포함될 수 있지만 physical edge/crease로 분류하지 않는다. Robot GT segmentation, Box rim,
known-negative ROI로 해당 구간을 제거하지 않는다. 작은 mask 성분·가는 구조도 크기로 제거하지 않는다.

### 3.3 최소 GT-free outlier 제안

추가 제거는 **저장 센서 계약에 명시된 invalid/saturation code 또는 per-pixel invalid flag**가
실제로 있을 때 그 관측만 제외하는 수준으로 제한한다. 근거가 없는 finite-positive depth를
depth 크기, ROI, 이웃과의 거리만으로 observation failure라고 판정하지 않는다.
현재 inspected episode에는 별도 validity/saturation/near-far clip field가 없으므로,
이번 고정 ablation의 **추가 outlier 제거는 0개**로 계획한다. 센서 근거가 새로 발견되면 적용 전 계획을 갱신한다.

기존 finite-positive/mask/overlap 제외와 contour non-observed 제외는 계속 수행한다.
단계마다 원본 source, object/frame, `pca_candidate`/`raw_outer_contour` 소속,
제외 사유와 사유별 count를 저장한다. 사유 중첩 count와 unique 제외 수를 구분한다.
후보가 X 밖에 있다는 저장 계약 위반은 조용히 drop하지 않고 실패로 보고한다.
Erosion, statistical/radius outlier filter, depth percentile cutoff, isolated component 제거는 하지 않는다.
Task03-I의 30.726m 같은 큰 depth도 수치만으로 제거하지 않으며 depth tail/coverage 왜곡을 보고한다.

## 4. 독립 Voxel → deterministic thinning → 합집합

### 4.1 Nominal metric Voxel

물체별 X의 원본 camera Z median과 저장 intrinsic으로 Task03-I 공식을 그대로 사용한다.

```text
z_ref = median(X[:, 2])
epsilon = z_ref * sqrt((2/fx)**2 + (2/fy)**2)
a = epsilon / sqrt(3)
key = floor(float64(original_cv_xyz) / a).astype(int64)
```

Camera origin과 축은 고정하고 bbox/world/GT 기준으로 이동·회전하지 않는다.
같은 물체의 두 source는 **동일 a/grid**에서 각각 독립 압축한다.
각 cell의 float64 centroid에 가장 가까운 원본 후보를 대표로 고르며 exact squared-distance tie는 최소 source다.
평균 좌표는 계산에만 사용하고 출력 node를 만들지 않는다.
`representatives` helper에는 ascending source copy를 넘기고 원래 contour 순서 mapping은 따로 보존한다.

`R_pca`는 기존 대표 source ascending 순서다. `R_sil`은 각 대표 source가 `C_sil` canonical contour
순서에서 등장한 위치로 정렬한다. Voxel 계산의 cell key 순서를 silhouette thinning 순서로 쓰지 않는다.
Component 사이에도 동일 global source-list thinning을 적용하며 별도 최소 quota를 주지 않는다.
한 voxel이 가까운 별개 contour를 합칠 수 있고 작은 성분이 thinning에서 사라질 수 있음을 기록한다.

각 source의 candidate→cell/대표 source, capacity, 원래 source 순서, contour 위치,
대표점 순서와 same-cell 거리를 저장한다. 두 source 대표점이 같은 cell에 있어도 서로의 원본 index가
다르면 그대로 두며 마지막 중복 제거는 source index로만 한다.
`sqrt(3)*a=epsilon`의 same-cell bound는 압축 단계에서만 검사한다.
Thinning 이후 coverage나 물리적 edge localization의 상한으로 주장하지 않는다.

### 4.2 Thinning index 규칙

유효 ratio 범위는 `(0,1]`, NaN/Inf/0/음수/>1은 오류로 한다. M=0이면 empty다.
두 source에 같은 함수와 ratio를 사용한다. 각 arm의 voxel 결과는 동일하고 한 번만 계산한다.

```python
# Proposed rule: rational arithmetic, no floating rounding drift.
# keep_ratio = p/q in lowest terms; 1, 3/4, 1/2, 1/4 for this sweep.
indices = [(k*q)//p for k in range(ceil(M*p/q))]
# keep only indices < M; main phase=0.
```

즉 ratio 1은 전부, .5는 `[0,2,4,...]`, .75는 `[0,1,2,4,5,6,...]`, .25는 `[0,4,8,...]`다.
기타 ratio도 입력 decimal의 rational 표현으로 결정적인 균등 index 간격을 만들며,
count는 `ceil(M*ratio)`로 고정한다. 작은 M에서 실제 비율의 반올림 차이를 기록한다.
이는 index 간격의 균등성이며 **3D arc length/등간격을 보장하지 않는다**.
Raster PCA ordering과 2D contour ordering 모두 camera depth, 경사, 성분 길이의 영향을 받는다.

Phase 진단은 ratio .5에만 `[1,3,5,...]`로 한 칸 이동한다. Wrap/refill하지 않는다.
홀수 M이면 두 phase의 count가 하나 다를 수 있으므로 실제 count를 함께 보고한다.
Source별·union Jaccard, coverage/최대 누락 구간 변화와 최종 Surface 변화를 모두 공개하며
더 좋은 phase를 택하거나 object별 phase를 섞지 않는다.

### 4.3 합집합과 Surface

`T_pca`, `T_sil`을 concat한 뒤 동일 원본 source 중복을 제거하여 P를 만든다.
순서는 PCA 선택 순서 먼저, 이어 silhouette 선택 순서에서 아직 없는 source다.
물체 최종 순서는 PEG 0, BOX 1이며 물체마다 P 다음 S를 둔다.

```text
N_P = len(T_pca) + len(T_sil) - len(intersection(T_pca, T_sil))
surface_added = min(requested_surface, len(X) - N_P)
F = P + S
N_total = sum_objects(N_P + surface_added)
```

합집합의 실제 보존율은 원본 후보 union 또는 대표점 union 분모를 명시하여 별도로 보고한다.
한 source가 반대편 source 선택으로 유지될 수 있으므로 ratio와 정확히 같다고 가정하지 않는다.
32점 FPS, Structural FPS, global budget allocation을 실행하지 않는다.

Surface는 **전체 X의 원본 3D Euclidean FPS**를 재사용한다.
X 행으로 매핑한 모든 P를 `initial`로 넣고 `count=len(P)+surface_added`로 호출한다.
P가 있으면 기존 함수는 RNG 첫 점을 사용하지 않는다. P가 없고 Surface가 필요하면
X의 최소 source row를 명시 initial로 주고, 그것을 **Surface 1개**로 세어 requested count 안에 포함한다.
따라서 Structural과 Surface 모두 random sampling을 쓰지 않는다. FPS argmax tie는 기존 X 첫 row다.
이미 선택한 source는 다시 선택하지 않는다. 서로 다른 source의 동일 XYZ는 허용한다.
X empty는 empty, capacity 부족은 shortfall만 기록하며 padding·복제·관측 밖 점 생성은 없다.

## 5. Vision → Graph provenance 계약과 실제 구현 한계

승인 후 별도 실험용 per-node `provenance` 구조를 정의한다. Selection role과 candidate evidence를 분리한다.

| 필드 | point N개에 대한 shape/type | 의미 |
| --- | --- | --- |
| `source_index` | `[N] int64` | 원본 flat source. frame/camera/episode identity는 packet-level 보존 |
| `object_id` | `[N] int64` | PEG 0 / BOX 1. 좌표/feature와 동일 row |
| `candidate_evidence` | `[N,2] bool` | 열 순서 PCA5 normal-change, outer-mask silhouette. 원본 후보 membership로 gather |
| `both_evidence` | `[N] bool` | 두 evidence의 AND, 단독 semantic class로 사용하지 않음 |
| `selected_by_structural_source` | `[N,2] bool` | thinning에서 PCA/silhouette 어느 경로로 유지됐는가. dedup 시 OR |
| `selection_stage` | `[N] enum` | Structural union 또는 Surface FPS. 서로 다른 선택 단계 |
| `surface_selected` | `[N] bool` | Surface selection role. PCA/silhouette candidate evidence와 동시에 true 가능 |
| `feature_available` | `[N] bool` 및 필요시 field별 mask | 실제 같은 조건에서 저장된 feature의 존재 여부 |

후보 evidence는 **원본 `C_pca`, `C_sil` membership**이다. 두 후보에 속했지만 한쪽 voxel/thinning에서
탈락한 source도 선택된 node에서는 both evidence가 유지된다.
Surface가 선택한 후보도 evidence를 지우지 않는다. 반대로 같은 voxel의 다른 후보 evidence를
대표 node의 원본 membership에 합성하지 않는다. Cell 내 evidence 요약은 별도 진단값이다.

`points[N,3]`, `object_id[N]`, `geometric_features[N,F]`, provenance 모든 row의 순서를 유지한다.
`intra_object_edge_index[2,E]`는 이 최종 point 배열의 `0..N-1`을 참조해야 한다.
Graph receiver는 전달받은 point/provenance를 resample하거나 정렬하지 않는다.
Robot-first 표현을 사용하면 graph node `0,1,2`는 TCP/left/right, Vision point i는 `3+i`다.
Point metadata와 `point_node_index[N]` mapping을 보존하고 robot을 `node_kind`로 구분한다.
Robot source는 존재하지 않으며 전체 node metadata가 필요하면 source=-1/관측 evidence=false와 명시적 robot mask를 쓴다.
기존 edge를 붙일 때에만 point index에 +3을 적용하며 입력 edge 자체는 변경하지 않는다.

**최소 receiver 제안:** 기존 `GraphBuilder`에 임의 GT geometry를 만들어 넣는 대신,
opt-in `graph/vision_input.py`에 point packet 검증·보존용 수신 container를 추가한다.
Point provenance와 기존 feature를 독립 필드로 보관하고 현재 GNN `x`에 concatenate하지 않는다.
이 container는 새 edge topology, GNN feature encoder, action interface를 정의하지 않는다.

현재 실제 final `geometric_features` encoding과 Vision intra-object edge 생성기는 없다.
따라서 저장 23-field metadata를 source로 exact join하고 compact values + positions + availability를 유지한다.
수신 계약의 `[N,F]` 검증용 view는 저장 field 이름/shape 순서를 명시하고 unavailable을 NaN/mask로 표시한다.
누락 normal/score를 0, reference PCA, 다른 arm feature로 채우지 않는다.
기존 최종 GNN feature가 완성됐다고 부르지 않으며 incomplete packet을 GNN에 전달하지 않는다.

실관측 audit packet의 `intra_object_edge_index`는 명시적으로 empty `[2,0]`,
`edge_status=not_implemented`로 보낸다. Empty를 유효 topology 완성으로 계산하지 않는다.
알려진 소형 test packet의 supplied edge로만 index 보존/robot offset/metadata 정렬을 검증한다.
실제 robot 좌표는 새 simulator/kinematics 수집 없이 만들지 않는다. 이번 실험의 `N_total+3`은
**계약상 예상 Graph node 수**이고, 세 robot을 넣은 테스트 수신 container 검증과 구분한다.
기존 GNN 연결, 실제 robot/point EE 좌표 결합, stale cache, 실제 edge 성능은 미구현 범위로 보고한다.

## 6. 최소 파일 변경 계획 — 별도 승인 후에만

| 새 파일/문서 | 최소 책임 |
| --- | --- |
| `visual/runtime/test/run_variable_rate_shape_first.py` | 저장 입력 loader, external contour/order, 기존 Voxel helper 호출, thinning/union/Surface, freeze, 평가, JSON/NPZ/PNG/HTML |
| `visual/runtime/test/test_variable_rate_shape_first.py` | source/contour/voxel/thinning/Surface/provenance/GT 격리 계약 테스트 |
| `graph/vision_input.py` | 실험용 packet 수신·row 정렬/가변 N/provenance 보존. 기존 builder/encoder 수정 없음 |
| `visual/runtime/test/test_variable_rate_graph_contract.py` | 새 receiver를 직접 호출하는 전달 계약 테스트. 실제 GNN 통합 테스트라는 명칭을 쓰지 않음 |
| `docs/visual-graph-integration/03j_variable_rate_shape_first.md` | 결과·한계·재현·검증 범위. 새 계약 설명 포함 |
| `experiments/<run_id>/...` | 새 결과만 저장. 기존 run/sidecar/source read-only |

Pure helper 재사용으로 해결하므로 공통 logger/artifact API 확장은 필요하지 않다.
향후 production 채택은 별도 검토다. 기존 sampling defaults, detector/normal, Graph prototype,
shared simulator, 기존 Task03 source/report를 수정하지 않는다.

## 7. 보고할 지표와 해석

모든 frame/ratio/object에서 아래 표를 만들고 raw 배열을 NPZ로 보존한다.
평균뿐 아니라 frame별 결과를 공개하며 frame aggregate는 동일 가중치와 분모를 명시한다.
Empty query는 분모 0/N/A, nonempty query→empty target은 coverage 0와 distance unavailable 수를 표시한다.
Undefined distance를 0으로 평균하지 않는다. Percentile은 NumPy linear를 유지한다.

| 항목 | 고정 정의 |
| --- | --- |
| 후보 count | `len(C_pca)`, `len(C_sil)`, both intersection, raw candidate union. 관측 제외 이전 contour 수도 별도 |
| Voxel count | source별 R, 대표점 source 교집합/union, occupied cells와 same-cell 거리/bound |
| Thinning/Structural | source별 M/선택 count/실제 ratio, 선택 교집합/unique P, 원본 후보·대표점 union 대비 보존율 |
| Nodes | object별 P/S/F, 전체 Point N, 계약상 Robot 포함 N+3. 실제 topology 없는 Graph 비용과 구분 |
| 3D candidate coverage | source별 및 후보 union에 대해 C→R(source별/union), C→T(해당 source), C→P, C→F 거리 mean/p95/max(mm). 원본 source 중복 분모 제거 |
| Silhouette coverage | 관측 가능한 ordered contour source가 P 및 F에 exact 포함된 비율, silhouette source의 T_sil 선택률, source별 C_sil→P/F 3D 거리와 고정 2px AND 3mm 관측점 matcher coverage |
| 연속 누락 | 원본 contour walk의 유효 관측 run별 exact-missing 최대/분포 및 tolerance-missing 최대. Pixel count와 원본 CV XYZ 인접 길이(mm) 모두 저장 |
| Silhouette 연결 한계 | invalid/overlap 틈과 component 사이를 gap 길이에 합산하지 않음. 완전 유효 closed loop만 closure를 잇고, 반복 source walk의 거리 해석 한계 표시 |
| Peg seam / Box rim | 모든 선택 freeze 후 기존 physical reference evaluator와 same-pair ≤2px AND ≤3mm. visible/observed/unique reference, matching node와 pair count를 구분 |
| 단계별 보조 GT | C_pca/C_sil/both/union, source별 R/T, P/S/F의 reference 보존·손실, Surface가 복구한 reference와 중복. 분모와 ID 저장 |
| 전체 Surface coverage | 전체 원본 X→F distance mean/p95, raw tail/max, object별 count/shortfall |
| 기존 ROI/depth 점유 | 선택 후 기존 ROI 함수 재사용. X/C/R/T/P/S/F 및 source별 점유, depth median/p95/max, X 기반 고정 bands. ROI를 selector에 전달하지 않음 |
| Feature/provenance | field별 available/unavailable count, source exact join, 모든 point의 evidence/role 일치, Graph 수신 전후 packet row/hash 일치 |
| 시간 | contour/nominal scale/Voxel/thinning/union/Surface/packet 처리 wall time. 공통 전처리 1회 비용과 arm별 비용 분리. I/O·GT·plot·HTML 시간 별도 |
| 메모리/edge | NumPy array nbytes(입력/중간/packet/산출), 가능하면 process peak RSS와 측정 방법/단위. 실제 edge E=0는 미구현 상태와 함께 보고 |

Contour coverage matcher는 원본 관측 contour를 reference로 삼는 GT-free 평가이며 물리적 edge 검증이 아니다.
순서 간격의 3D 길이는 관측 walk 길이이지 실제 물체 둘레가 아니다.
Peak RSS는 process 누적값이므로 ratio별 peak로 잘못 비교하지 않는다.
새 topology를 만들어 E를 측정하거나 기존 GT GraphBuilder에 node를 끼워 넣지 않는다.
선택된 N에 대한 알려진 `N²` pairwise float32 거리행렬의 bytes 같은 계산은 **이론값**으로만 표시할 수 있다.
Graph/GNN latency·GPU memory·20Hz 가능 여부는 측정하지 않은 것으로 남긴다.

해석의 주축은 N 대비 GT-free candidate/contour coverage와 X→F 거리의 trade-off다.
Ratio가 작아도 Surface가 일부 후보를 다시 선택할 수 있으므로 P와 F를 구분한다.
GT rim 점수는 보조 평가이며 ratio/phase 선택·필터 조정·frame 교체의 목적함수로 사용하지 않는다.
기존 Task03-I/Task03 S3 결과는 historical reference이고 같은 node budget 비교라고 주장하지 않는다.
성능 우월성이나 특정 ratio 채택을 테스트 PASS 조건으로 두지 않는다.

## 8. 계약 테스트와 승인 후 실행 순서

1. **입력·보존:** HEAD/status, 기존 tracked/dirty/untracked source와 기존 실험/sidecar SHA-256,
   입력 hash와 실제 dependency version을 기록한다. missing dependency는 보고하고 설치하지 않는다.
   지정 3 frame의 X/C/mask/K/source identity를 exact 검사한다.
2. **후보 테스트:** 외곽/내부 hole 구분, contour canonical 방향/시작점, 여러 component,
   1pixel/가는/퇴화 mask, invalid depth 구멍, overlap, image boundary, occlusion-shaped mask.
   유효 원본 contour point만 선택하고 candidate를 physical edge로 바꾸지 않는지 확인한다.
3. **선택 테스트:** 재사용 Voxel helper의 floor/음수/centroid tie/원본 좌표/bound;
   source별 independent grouping과 정렬 복구; ratio 1/.75/.5/.25 및 기타 ratio의 expected index,
   empty/odd M/phase1 count, 중복 source union, Surface initial/count API/빈 P deterministic first,
   support 부족/no-padding, **N>256 및 P>32 허용**을 검증한다.
4. **Provenance/Graph 테스트:** both evidence의 두 bit, 한 source thinning 탈락 후 다른 source로 보존,
   Surface candidate의 evidence 유지, feature unavailable 보존, frame identity, dtype/row mismatch 실패,
   서로 다른 N 및 N=0, point/robot 구분, supplied edge remap/index 범위/동일 물체 참조를 검사한다.
   기존 GNN x의 생성 함수에 provenance를 추가하지 않았음을 확인한다.
5. **GT 격리:** selector/receiver는 GT pose/segmentation/rim/negative ROI를 인자로 받지 않는다.
   새 runner는 관측 키만 읽는 선택 단계와 GT reader를 분리한다.
   기존 `run_sampling.evaluate`처럼 선택 전에 GT 배열을 읽는 전체 runner는 호출하지 않는다.
   GT 평가 진입을 막는 sentinel/mock 아래 selection이 완료되는 테스트를 둔다.
6. **회귀 검증:** 아래 관련 CPU 테스트 후 실패를 해결한다. 공유 API 변경은 없지만 새 runner의
   logger/artifact 계약 사용을 확인하기 위해 공통 infrastructure 테스트도 실행한다.
7. **한 번의 고정 평가:** 새 `ExperimentLogger(team='visual', job_type='evaluate', seed=0, ...)`,
   `WANDB_MODE=disabled`로 UUID를 생성한다. Project/group/run ID 규칙을 override하지 않는다.
   실제 config·frames·ratio·phase·Surface 추가 수·source snapshot·Git dirty/diff·hash를 저장한다.
   세 frame의 네 ratio와 phase 진단의 선택/metadata를 **전부** 저장·readback·hash한 다음 GT를 연다.
   `selection_before_gt.npz`/JSON freeze 이후 selector 재호출·선택 변경을 하지 않는다.
8. **결과·링크:** 같은 UUID에 평가/HTML/PNG를 저장하고 `logger.log/record/link_output`으로 연결한다.
   공통 artifact manifest만 사용하며 publisher 호출·upload·online sync는 없다.
   선택 hash, 기존 source/run hash와 HEAD 불변을 재확인하고 결과 문서를 제출한다.

승인 후 실행할 관련 테스트 묶음(현재는 **미실행**):

```bash
.venv/bin/python -m unittest \
  visual.runtime.test.test_variable_rate_shape_first \
  visual.runtime.test.test_variable_rate_graph_contract \
  visual.runtime.test.test_voxel_structural_sampling \
  visual.runtime.test.test_sampling \
  visual.runtime.test.test_sampling_geometry \
  visual.runtime.test.test_sampling_evaluation \
  tests.test_experiment_logger tests.test_experiment_artifacts -v
```

새 함수명/CLI는 구현 시 위 책임 범위 안에서 정의한다. 제안하는 API를 기존 API로 표기하지 않는다.

## 9. 승인 후 산출물과 완료 조건

- 결과 Markdown: `docs/visual-graph-integration/03j_variable_rate_shape_first.md`
- 정적 HTML: `experiments/<run_id>/variable_rate_shape_first.html`
- 같은 run에 `preregistered_config.json`, `selection_before_gt.npz/.json`,
  `variable_rate_shape_first.json/.npz`, ratio별 PNG, `verification.json`, test log, source snapshot/hash를 저장한다.

HTML에는 ratio별 원본 후보, source별 Voxel 대표점, thinning 후 Structural, Surface,
silhouette coverage/연속 누락, node count, geometry trade-off, provenance/feature availability와
JSON/NPZ/PNG 상대 링크를 포함한다. 같은 frame/시점의 overlay와 3D 비교를 함께 제공한다.
Markdown→HTML은 `../../experiments/<run_id>/variable_rate_shape_first.html`,
HTML→Markdown은 `../../docs/visual-graph-integration/03j_variable_rate_shape_first.md`로 한다.
HTML 내 JSON/NPZ/PNG는 같은 run 기준 상대 경로로 연결한다.

생성 후 링크 존재·상대 경로 해석·양방향 도달, JSON/NPZ readback, PNG decode를 검증한다.
실제 browser 표시 확인은 실행한 경우에만 PASS로 기록하고 미실행이면 구분한다.
N/count 식·원본 XYZ·source/role/evidence/feature 행 정렬·수신 일치·GT freeze·기존 파일 불변이 완료 조건이다.
Graph 수신 container 검증과 기존 GNN end-to-end 통합은 별도로 보고한다.

이번에 실시한 작업은 source/기존 보고서 독해, 저장 artifact 존재·schema 확인과 계획 문서 추가뿐이다.
Simulator/renderer/CUDA/MPS/GPU/PCL, sampling/contour 계산, 새 계약 테스트·Sweep은 미실행이다.
GT 최적화, 새 detector, normal threshold sweep, Adaptive Budget, GNN/BC 학습,
새 edge topology, 추가 episode, dependency 설치, 외부 upload, Git commit은 범위 밖이다.
**이 계획과 최소 구현 범위를 제출하고 별도 승인 전까지 중단한다.**
