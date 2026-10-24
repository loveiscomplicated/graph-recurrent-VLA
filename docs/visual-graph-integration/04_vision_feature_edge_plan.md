# Task04 — Vision Geometric Features & Intra-object Edge Integration Plan

상태: **구현 전 계획 / 승인 대기**. 이번 변경은 이 문서 한 개다. Feature 계산,
새 Edge/Graph 생성, 검증 실행, ExperimentLogger 생성은 후속 구현 승인 후 진행한다.
이 계획은 현재 관측을 전달하는 opt-in feasibility 경로이며 최종 모델 구조를 정하지 않는다.

## 1. 확인한 구현과 적용 범위

먼저 루트 `README.md`, `envs/maniskill/README.md`, `docs/PROJECT_CONTEXT.md`,
`AGENTS.md`, `docs/engineering_rules.md`, tracking/publishing 지침을 읽었다.
아래 판단은 현재 checkout의 tracked/dirty/untracked 구현을 기준으로 한다.
이전 문서의 “visual 구현 없음” 같은 초기 설명보다 현재 소스와 저장 계약을 우선했다.

| 확인 대상 | 현재 완료된 사항 | Task04에서 필요한 사항 |
| --- | --- | --- |
| [Task03-J](03j_variable_rate_shape_first.md), [Task03-K](03k_temporal_shape_first.md), `run_variable_rate_shape_first.py`, `run_temporal_shape_first.py` | PCA5 + 외부 Silhouette + metric voxel + deterministic thinning + Surface FPS, 가변 N, source/XYZ/선택 근거 보존 | 선택 결과를 바꾸지 않고 런타임 Feature와 Edge를 붙이는 경로 |
| `visual/runtime/sampling.py::extract_sampling_evidence` | 전체 observed support에 optimized PCA5 analytic normal, `cropped_features`, `pair_evidence` 계산 | 결과 `SamplingEvidence`를 최종 선택 source에 gather |
| `visual/runtime/pca.py` | 5×5 patch-local 연결 성분, 최소 6점, rank/고유 eigenvector gate; 전체 support의 normal/validity 출력 | estimator 변경 불필요. covariance/eigenvalue는 내부 계산 후 반환하지 않음 |
| `visual/runtime/geometry.py`, `edge_evidence.py` | normal, variation, depth jump, 경계 원인, 인접 pixel pair 진단 | 측정값과 각각의 validity를 최종 행에 정렬 |
| J/K `make_packet` | 원래 `S3_n256_r0.25_t20_s0` 저장 행과 정확한 source join; 없는 Feature는 NaN/false | 이 저장 조건의 Feature availability를 런타임 계산 가능성으로 해석하지 않음 |
| `graph/vision_input.py` | NumPy packet 검증/깊은 복사, point 행 불변, optional robot 3행 prepend와 Edge +3 remap, N=0 수용 | 런타임 Feature validity/frame 및 새 Edge 정책 검증 추가 |
| `graph/prototype/graph_builder.py::GraphBuilder` | 물체별 Euclidean k-NN, 기본 `surface_k=8`, 양방향 Edge | GT 기반 고정 batch geometry/robot/inter-object 처리와 결합되어 직접 호출 불가 |
| [00_interface.md](00_interface.md) | 역할·EE frame·Feature 3모드·Vision 내부 Edge의 설계 | Tensor 출력, 3모드, RGB-D Edge filtering, stale 처리까지 구현됐다는 뜻은 아님 |

`SHARED_TECHNICAL_FOUNDATION.md`는 저장소 전체에서 hidden/ignored 파일을 포함해
이름으로 검색했지만 찾지 못했다. 내용을 확인했다고 간주하지 않는다. 현재 확인 가능한
계약을 기준으로 이 계획을 작성하며, **후속 구현 착수 전 해당 문서의 위치/내용을 확보해
대조하거나, 현재 계약 기준 진행이 명시적으로 승인되어야 한다.** 충돌하면 계획을 먼저 수정한다.

이번에는 저장 NPZ/JSON의 schema만 읽었다. 예를 들어 K frame45/ratio .5 packet은
points `[514,3]` float32, Feature view `[514,25]` float64, 23개 원본 field,
availability `[514,23]`, Edge `[2,0]` int64다. 23개 field를 학습 Feature 23차원으로
해석하면 안 된다. bool 진단·선택 근거·normal 벡터가 섞인 audit view다.
AB frame45 cache에는 물체별 모든 `GeometricFeatures` field가 실제로 저장되어 있다.
Feature/Edge correctness를 새로 실행 검증한 결과는 아니다.

### 고정 조건

- 잠정 `keep_ratio=0.5`, phase 0. ratio 1.0/phase 0은 같은 대표 프레임의 비교 기준으로 보존한다.
- 기존 PCA5 strict >20° 후보, Outer Silhouette, nominal metric voxel, thinning,
  Surface FPS와 최종 source/XYZ/행 순서를 바꾸지 않는다. 256/32 상한을 복원하지 않는다.
- 대표 입력은 K가 저장한 선택 결과다. Surface 추가 요청 Peg97/Box95도 해당 결과 그대로이며
  새로운 최종 budget으로 채택하는 작업이 아니다. 관측에 따른 실제 N과 shortfall을 유지한다.
- GT geometry/물체 pose/mesh/segmentation, task relevance를 Feature/Edge 선택에 사용하지 않는다.
  Camera calibration과 TCP kinematics는 좌표 변환 입력이다. RGB는 overlay용이며 GNN Feature에 넣지 않는다.
- Robot Node 및 inter-object interaction Edge는 Graph 책임이다. stale geometry, 학습,
  threshold/ratio sweep, GPU/PCL 최적화, 업로드·설치·commit은 범위 밖이다.

## 2. Feature 우선 구현: 새 estimator 없이 계산 결과 재사용

### 2.1. 계산과 gather

런타임에서는 현재 frame의 `extract_sampling_evidence(..., estimator='pca5')` 결과를
sampling과 Feature packaging이 공유한다. 이 함수나 `local_plane_optimized`를
최종 선택점마다 다시 호출하지 않는다. `process_geometry`의 baseline normal로
PCA5 normal을 바꾸지도 않는다. 이미 얻은 `SamplingEvidence`를 소비하는 작은 adapter가 필요하다.

1. episode/camera/source_frame, H/W, mask, depth validity, overlap, estimator/parameters를 확인한다.
2. 최종 packet 행 i의 `(object_id[i], source_index[i])`를 사용한다.
   `source_index=v*W+u`는 원본 C-order raster index이며 최종 point 행 번호가 아니다.
3. 해당 object의 `observed_mask`가 true인지, source가 범위 내인지, object-source 쌍이
   유일한지 확인한다. `geometry.xyz.reshape(-1,3)[source]`와 입력 camera XYZ는 정확히 일치해야 한다.
4. 각 dense field를 원본 raster로 flatten한 뒤 같은 source 배열로 gather한다.
   object 0→1 및 물체별 P→S의 기존 행 순서를 유지한다. source를 정렬한 결과로 point를 재배열하지 않는다.
5. gather를 마친 normal만 point와 같은 EE frame으로 회전한다. scalar의 원래 의미와 단위는 유지한다.
6. 입력은 수정하지 않고 새 packet을 만든다. Feature 유효성 때문에 point를 제거하거나
   sampling을 다시 수행하지 않는다.

Silhouette-only와 Surface-only도 같은 gather 경로를 쓴다. PCA5 후보가 아니어도
normal/variation이 유효할 수 있고, 유효 point라도 normal은 계산 불가할 수 있다.
관측 support가 작거나 선형/퇴화한 patch에서는 기존 estimator의 NaN/false를 유지한다.
“선택된 모든 유효 point에 Feature 제공”은 **모든 행에 값 또는 명시적 invalid와 원인이
정렬되어 있음**을 뜻한다. 존재하지 않는 normal을 보장하는 조건이 아니다.

### 2.2. 최소 측정 Feature

최초 구현은 `continuous`의 다음 3 field, 총 F=5로 한정한다. 이는 비교용 런타임
측정 계약이며 최종 GNN encoder의 feature 구성을 확정하지 않는다.

| field / flatten 순서 | Shape / dtype | 유효성 | 좌표계·단위 |
| --- | --- | --- | --- |
| `normals` | `[N,3]` float32 | `normal_valid` | EE frame 단위벡터. camera-facing 방향을 회전해 보존 |
| `normal_variation_rad` | `[N]` float32 | `normal_variation_valid` | 동일 물체의 유효 4-neighbour normal과 최대 각도, rad |
| `depth_jump_m` | `[N]` float32 | `depth_pair_valid` | camera-z의 유효 4-neighbour 최대 절대 차이, m |

Normal variation은 PCA curvature가 아니다. `depth_jump_m`은 다른 label/배경의 유효 depth도
비교하는 관측 진단이다. EE z 차이나 물체의 physical crease로 재해석하지 않는다.
`normal_neighbor_count`, `depth_discontinuity`, `mask_boundary`, `depth_validity_boundary`,
`image_boundary`, `normal_valid`, `normal_variation_valid`, `depth_pair_valid`, `valid_points`는
동일하게 gather해 `feature_diagnostics`로 전달한다. GNN Feature에 자동 concatenate하지 않는다.

새 estimator는 현재 최소 범위에 필요하지 않다. Curvature를 추가하려면 PCA에서 이미
계산한 eigenvalue를 반환하도록 확장하는 방안을 먼저 검토해야 한다. 현재 API에는 없으므로
사용 가능하다고 주장하거나 eigen decomposition을 별도로 재실행하지 않는다. 이번 최소 변경에는 넣지 않는다.

### 2.3. 세 모드의 구현 가능 범위

| mode | 현재 근거로 가능한 것 | 필요한 입력 / 이번 처리 |
| --- | --- | --- |
| `continuous` | 위 normal/variation/depth 진단과 validity | 현재 RGB-D, tracker mask, 기존 PCA5 evidence, calibration/TCP. 최소 구현 대상으로 추천 |
| `onehot` | 선택 provenance와 orientation 후보 여부만 존재. Corner/Edge/Surface 의미 분류는 없음 | 명시적 분류 정의, Corner 판별 근거, unknown/invalid 규칙이 필요. 현재는 지원 불가 오류로 종료 |
| `probability` | calibrated class probability 없음 | classifier 또는 명시적 확률 모델, 클래스 정의와 calibration/평가 근거가 필요. 현재는 지원 불가 오류로 종료 |

`candidate_evidence`, `selection_stage`, Surface FPS 선택 여부를 의미 class로 바꾸지 않는다.
Silhouette는 가림 외곽일 수 있고, Surface FPS가 PCA 후보를 다시 선택할 수도 있다.
20° threshold나 정규화 angle을 확률로 만들지 않는다. 모드 요청을 조용히 continuous로
대체하지 않는다. `00_interface.md`의 onehot 기본값은 설계상의 기본값으로 남기되,
Task04 entrypoint는 `continuous`를 명시하고 3모드 구현 여부를 정확히 정정한다.

## 3. Vision→Graph packet 계약

기존 unversioned audit packet은 legacy v1로 그대로 수용한다. 새 runtime packet에만
`schema_version=2`, `feature_mode='continuous'`를 명시하고 strict 검증을 적용한다.
알 수 없는 version/mode는 거부한다. 기존 J/K artifact와 reader/runner를 덮어쓰지 않는다.

| 필드 | v2 최소 계약 |
| --- | --- |
| identity | `episode`, `camera`, `source_frame` 유지. 원본 timestamp/frame_idx가 있으면 보존 |
| `points` | float32 `[N,3]`, finite, `coordinate_frame='EE_metres'` |
| `object_id` | int64 `[N]`, PEG=0/BOX=1 |
| `geometric_features` | float64 `[N,5]` 연속값 view. 기존 receiver의 view dtype 유지; NaN 허용, Inf 금지 |
| `feature_fields` | 위 순서의 이름·shape `[3],[],[]`·원본 dtype float32 및 단위/좌표계 명시 |
| `compact_features` | field별 final-row `positions:int64`와 원본 dtype values. availability=true인 행에 대해 dense view와 equal-NaN 일치 |
| `feature_field_available` | bool `[N,3]`: 계산 결과 field가 제공됐는지. 정상 runtime에서는 invalid normal 행도 true |
| `feature_available` | bool `[N]`, 위 availability의 all. 유효성 보장으로 사용 금지 |
| `feature_field_valid` | bool `[N,3]`: normal/variation/depth pair validity. valid=true이면 해당 field 전 성분 finite |
| `feature_valid` | bool `[N]`, field validity의 all. 부분 유효 Feature는 field mask로 구분 |
| `feature_diagnostics` | §2.2의 행별 원본 측정 validity·경계 원인·이웃 수. validity와 서로 일치해야 함 |
| `provenance` | 기존 source/object/evidence/selection 배열을 행별 그대로 전달 |
| transform metadata | `T_EE_from_camera:float64[4,4]`, 해당 source의 `observation_ee_pose[7]` wxyz 및 Feature별 frame |
| `valid` | bool `[2]`, 현재 물체별 observed support 존재 여부. Feature 전체 validity와 별개 |
| `intra_object_edge_index` | int64 `[2,E]`, 최종 point 행 참조. `edge_status='supplied'` 및 §4 정책 metadata |

Availability=false 또는 validity=false인 측정 float field는 전체를 NaN으로 남긴다.
Invalid를 0, `[0,0,1]`, Surface one-hot 등 정상 관측처럼 채우지 않는다.
`feature_field_valid <= feature_field_available`를 검증하고 vector 일부만 유효한 normal은 허용하지 않는다.
Compact field는 **제공된 행**을 담으므로 invalid NaN도 포함할 수 있다.
N=0에서도 schema는 고정하고 `[0,3]`, `[0,5]`, `[0,3]` masks, `[2,0]` Edge를 반환한다.

기존 provenance의 `feature_available`/`feature_field_available`는 **과거 artifact row의 존재**다.
새 top-level runtime availability와 다른 이름 공간으로 유지하며 덮어쓰지 않는다.
J/K packet을 입력으로 사용하면 원래 23개 field 설명을 `source_artifact_feature_fields`에
함께 보존해 이 과거 mask의 해석을 유지한다. 실제 runtime에서 과거 artifact가 없으면
이 선택적 audit metadata를 새로 만들지 않는다. v2 validator는 필수 생성 provenance와
선택적 과거 availability를 구분한다. 원래 audit Feature 값은 원본 artifact에 그대로 남고
그 hash/경로를 실행 metadata에 연결한다.

`candidate_evidence[:,0:2]`는 PCA5/Silhouette 자격,
`selected_by_structural_source[:,0:2]`는 thinning 후 선택 경로,
`both_evidence`는 자격 교집합, `selection_stage` 0/1은 Structural/Surface,
`surface_selected`는 Surface FPS 선택이다. 서로 대체하지 않는다. 모두 기본 GNN Feature와 분리한다.

Graph receiver는 기존 깊은 복사·행 불변을 유지한다. 별도 robot 좌표를 받는 경우에만
point node mapping 및 Edge를 +3 offset하며, 원 packet Edge는 그대로 보존한다.
Supplied Edge를 다시 계산하지 않는다. 현재 receiver는 opt-in audit consumer이며
기존 `GraphBuilder`/GNN 입력까지 연결된 상태가 아니다. NaN/mask를 실제 encoder가 소비하는
정책은 후속 Graph 협의 대상이다. v2 PASS를 “그대로 학습 가능한 x 완성”으로 보고하지 않는다.
이번에는 Tensor/PyG adapter나 robot/inter-object 정책을 추가하지 않는다.

### 3.1. Camera→EE 변환

`TrackedRGBDFrame.cam2world`는 CV-camera→world이고 `tcp_pose`는 world의
`[x,y,z,qw,qx,qy,qz]`다. `collect.py`는 CV extrinsic의 역행렬을 저장한다.
OpenGL `cam2world_gl`를 이 값으로 대신 쓰지 않는다.

```text
T_EE_from_camera = inverse(T_world_from_EE(tcp_pose)) @ T_world_from_camera
p_EE = R @ p_camera + t
n_EE = R @ n_camera
```

계산은 float64, 최종 point/normal은 float32로 전달한다. 원본 camera XYZ와 source는
입력 artifact/검증 자료에 보존한다. normal에는 translation을 적용하지 않고 회전 후
camera-facing 부호를 EE 원점 기준으로 다시 뒤집지 않는다. invalid normal은 변환 대상에서 제외한다.
Variation은 회전에 불변이다. Camera-z depth jump는 이름·frame metadata로 고유 의미를 보존한다.

기존 `visual/scripts/pointcloud.py::pose_mat/to_world`와
`graph/prototype/geometry.py::rotation/transform/inverse_transform`에 같은 수학이 있다.
전자는 legacy plotting/pointcloud 의존성, 후자는 Torch GT geometry 모듈이므로 runtime adapter가
그 모듈 전체를 import하지 않는다. 작은 좌표 변환은 `vision_packet.py`에 두고 기존 식과 대조한다.
입력 pose/calibration의 shape, finite, homogeneous row, orthonormal R/det+1,
quaternion norm을 검사한다. 잘못된 calibration을 identity로 대체하지 않는다.
현재 observation의 TCP만 사용하며 누락 입력은 오류다. stale pose/age_steps를 추정하지 않는다.

## 4. Intra-object Edge: 관측 연결 성분으로 제한한 대칭 k-NN 하나

Feature 계약과 정렬 검증을 통과한 뒤 구현한다. 기존 GraphBuilder의 물체별
Euclidean k-NN 방식을 참고하되, Vision 전용 pure helper로 분리한다.
기존 GT batch builder를 호출하면 재포장/robot/inter-object 정책이 섞이므로 재사용하지 않는다.
`edge_evidence.py`의 pixel pair는 관측 진단이며 최종 point index Edge가 아니다.

추천은 **동일 object의 동일 observed component 안에서 k=8 nearest neighbours를
선택하고 양방향 합집합으로 만드는 방법**이다. k=8은 기존 Graph prototype의 출발값을
재사용한 고정 feasibility 설정이며 최적값·최종값으로 주장하지 않는다. sweep은 하지 않는다.

1. 기존 `PairEvidence`에서 두 endpoint 모두 같은 object의 observed support이고
   `depth_jump=False`인 수평/수직 pair만 연결한다. invalid depth/overlap/외부 label은 제외한다.
   기존 `0.01m + 0.02*min(z1,z2)` gate를 그대로 쓰며 새 threshold를 도입하지 않는다.
2. 전체 원본 support X에서 연결 성분을 계산한다. 선택된 점만으로 pixel 연결성을
   검사하면 thinning으로 연결이 끊기므로 X를 사용한다. 기존 SciPy sparse connected-components를
   사용할 수 있으며 새 dependency는 없다. source→component mapping을 최종 source로 gather한다.
3. object/component별 최종 point 행 집합 I를 만든다. 거리 기준은 **선택된 3D camera XYZ의
   Euclidean distance(m)**다. Camera 좌표를 사용해 EE float32 변환 반올림이 topology를 바꾸지 않게 한다.
   EE에서도 rigid transform에 의해 같은 거리여야 함을 검증한다.
4. 각 행에서 자기 자신을 제외하고 `min(8, len(I)-1)`개 선택한다.
   float64 squared distance와 최종 point 행 index의 lexicographic 정렬로 동률을 결정한다.
   normal validity/선택 근거/GT를 이 선택 기준에 넣지 않는다.
5. 각 쌍에 역방향을 더한 뒤 directed `(src,dst)`를 unique하고 Edge 열만 lexicographic 정렬한다.
   point/Feature/provenance 행은 정렬·변경하지 않는다. 동일 XYZ의 서로 다른 source는
   거리 0인 정상 후보이며 source 행을 병합하지 않는다.

출력은 point-local int64 `[2,E]`다. N=0, 한 점 component, 빈 Box는 Edge 0개를 허용한다.
Disconnected component 사이를 억지로 채우거나 관측 밖 point를 만들지 않는다.
Self-loop와 동일 directed pair 중복은 생성하지 않으며 v2 receiver도 거부한다.
역방향은 중복이 아니라 별개의 directed Edge다. 범위 밖/음수/cross-object index는 오류다.
대칭성도 v2 정책 계약으로 검사한다. Legacy v1 검증 의미는 바꾸지 않는다.

### 대안과 선택 근거

| 방식 | 거리·규모 특성 | 예상 실패 / 이번 선택 |
| --- | --- | --- |
| object별 무제한 k-NN | metric 3D, Edge O(Nk), 별도 radius 없음 | 떨어진 관측 조각을 강제 연결할 수 있어 단독 사용 제외 |
| metric radius | radius 내 모든 쌍, sparse search는 가능하지만 출력 최악 O(N²) | 새 거리 threshold 필요, 가변 밀도에서 고립 또는 Edge 폭증. 이번에는 제외 |
| **observed-component 제한 대칭 k-NN** | 기존 depth gate + metric 3D, E≤2kN | 관측 불연속 조각 사이 연결을 차단하고 Edge 수 상한을 명확히 하므로 추천 |

최소 CPU 구현은 행별 distance/lexsort로 한다. Component의 최종 점 수를 n_c라 하면
거리 계산 O(Σn_c²), 전체 정렬 포함 O(Σn_c² log n_c), temporary memory O(max n_c),
Edge 저장 O(Nk)다. Support 연결 성분 계산은 O(|X|+P), memory O(HW+|X|+P)이며
P는 지원된 인접 pixel pair 수다. 전체 N×N 행렬은 저장하지 않는다.

이것은 물체 안의 **관측 기하 근접 연결**이다. Physical edge/crease/contact 검출이 아니다.
같은 component가 먼 경로로 이어져 있으면 얇은 벽 양쪽이나 hole을 가로지르는 shortcut이
남을 수 있다. Depth gate 아래의 crease도 통과할 수 있고, noisy depth는 과도한 단절을 만든다.
작은 component는 고립될 수 있으며 대칭화 후 degree는 8을 넘을 수 있다. k-NN은 component의
전체 연결성을 보장하지 않는다. 길이 분포/연결 성분/고립점/긴 연결을 보고하고 결과가 부적절하면
이 단계에서 멈춘다. radius나 normal-angle gate를 결과를 보고 즉시 덧붙이지 않는다.

## 5. 정확한 최소 변경 파일

아래는 **후속 승인 후** 변경할 파일 목록이다. 이번에는 첫 행만 생성한다.
Feature 단계가 통과하기 전 Edge 파일/Edge 생성 작업은 시작하지 않는다.

| 파일 | 변경 목적 |
| --- | --- |
| `docs/visual-graph-integration/04_vision_feature_edge_plan.md` | 현재 계획 문서 |
| `visual/runtime/vision_packet.py` (신규) | 기존 SamplingEvidence와 최종 선택 행을 받아 gather, validity, EE 변환, v2 packet 생성. estimator/selector 추가 없음 |
| `graph/vision_input.py` (기존) | legacy 유지, v2 Feature/frame/Edge 계약 검증 및 기존 복사/offset 수신 재사용 |
| `visual/runtime/test/test_vision_features.py` (신규) | gather/모드/validity/변환 계약 테스트 |
| `visual/runtime/intra_object_edges.py` (신규, Feature PASS 후) | 기존 pair support component mapping과 deterministic k=8 대칭 Edge pure helper |
| `visual/runtime/test/test_intra_object_edges.py` (신규, Feature PASS 후) | component/인덱스/중복/self-loop/빈 입력/동률/불변성 테스트 |
| `visual/runtime/test/test_variable_rate_graph_contract.py` (기존) | v1 회귀 유지, v2 전체 packet/Feature/provenance/Edge 및 robot offset 보존 테스트 추가 |
| `visual/runtime/test/run_vision_feature_edge.py` (신규) | 저장 대표 프레임 재사용, cache/런타임 parity, CPU 측정, 결과/HTML/JSON/NPZ/PNG 생성 |
| `docs/visual-graph-integration/00_interface.md` (기존) | 구현 상태/모드 지원/Task04 opt-in NumPy 계약/가변 N 예외 링크 정정. 기존 256/259 예시가 Task04 상한이 아님을 명시 |
| `docs/visual-graph-integration/04_vision_feature_edge.md` (신규 결과) | 실제 실행한 Feature/Edge 검증 및 한계 보고 |

`sampling.py`, `pca.py`, `geometry.py`, J/K runner, 기존 sampling test/evaluation,
공통 simulator/logger/publisher, 기존 GraphBuilder/model/config는 수정 대상이 아니다.
확인하지 못한 foundation 문서는 이 목록에 임의로 만들지 않는다. 대조 후 추가 변경이 필요하면
변경 목록과 계약을 먼저 갱신한다.

## 6. 계약 테스트 및 CPU 검증 절차

### 6.1. 고정 입력과 보존

K run `9185355a32b8433abc70f1738f983377`의 대표 frame
`0,18,27,45,56,66,80,120`과 ratio `.5,1.0`, phase 0을 사용한다.
`selection/f{frame:04d}_r{ratio:g}.npz`에 저장된 최종 packet/source/XYZ를 읽는다.
K의 `selection_before_gt.json` hash는
`198b9d1ffe965be49b81632f53fde5903dcc1e4dff0f934c2f94f74f2521e923`다.
필요한 provenance/schema는 K source와 freeze metadata에 대조한다.
Sampling 함수를 재호출해 새로 선택하지 않는다. **121프레임 Sweep은 재실행하지 않는다.**

기하 입력은 AB run `f68c59f210b5499dadc7e962c7f64492/cache/f{frame:04d}_evidence.npz`와
그 원본 RGB-D/mask다. AB의 저장 provenance JSON, K 입력 hash manifest, 원본 Task03
config/저장 candidate를 통해 camera/source/parameters/estimator를 확인한다.
기존 `run_adaptive_node_budget.py::load_evidence`는 runner에서만 재사용 가능하다.
Production `vision_packet.py`가 test runner를 import하지 않게 한다.
GT key를 받지 않는 frame/evidence 입력만 Feature/Edge helper에 전달한다.

실행 시작 때 현재 HEAD, dirty diff, untracked 목록, 사용 source와 input artifact hash를
새 run에 기록한다. 기존 dirty/untracked 파일과 입력 artifact/sidecar는 수정하지 않는다.
신규 파일이 이미 존재하면 덮어쓰기 전에 중단한다. 종료 시 보호 파일 hash를 대조한다.

### 6.2. 필수 계약

| 항목 | 확인 내용 / PASS 조건 |
| --- | --- |
| source·XYZ·object 정렬 | 비정렬 source fixture, 물체별 같은 source 번호, 두 object 혼합에서도 직접 dense gather와 행별 일치. 음수/범위 초과/중복 object-source/다른 frame·mask·XYZ는 reject |
| 생성 근거 | 입력과 출력의 candidate/both/selected_by/stage/surface 배열 bytewise 동일. Feature validity와 독립 |
| Silhouette-only / Surface-only | PCA5 자격 없는 Silhouette와 Surface FPS 행도 dense field와 equal-NaN 일치. Surface가 후보를 선택한 경우도 별도 검증 |
| availability / validity | field 존재하나 normal invalid, normal valid이나 이웃 없어 variation invalid, depth pair 없음 등을 분리. NaN→0, bool mask 뒤집기, valid+NaN/Inf, partial normal은 reject |
| Feature mode | continuous schema/F/dtype/field 순서 고정. onehot/probability 요청은 명시적 unsupported 오류 |
| 가변·빈 관측 | K frame18/56 빈 Box, frame27/66 재관측, ratio에 따른 N 보존. synthetic N=0/1/2/불균등 object 지원. padding/복제/과거 geometry 삽입 없음 |
| Camera/EE | identity, 알려진 90° 회전+translation, quaternion wxyz, round-trip, normal 회전/길이/부호, invalid 보존. point·normal 절대 오차 ≤1e-6, normal norm 오차 ≤1e-5; 원 camera source XYZ 비교는 exact |
| 거리·변환 | rigid transform 전후 거리 오차 ≤1e-6m. camera XYZ에서 만든 Edge와 같은 계산을 float64 EE에서 수행한 결과 비교; 동률/roundoff near-tie는 거리와 tie 규칙을 별도 진단하고 미설명 불일치는 중단 |
| Edge | 모든 index `[0,N)`/동일 object/동일 support component, self-loop·directed 중복 0, 양방향, E≤16N, 단일점/빈 object 출력 일치 |
| discontinuity fixture | 같은 object의 분리 mask, invalid/overlap gap, 기존 depth gate 초과 pair가 component를 잇지 못함. normal invalid만으로 point/Edge를 제거하지 않음 |
| Graph 수신 | 전체 packet equal-NaN/hash 보존과 입력 불변. deep copy 확인. Robot supplied case는 frame 일치 전제 아래 point mapping/Edge만 +3, point-feature-provenance 행과 원 Edge 보존 |
| GT-free | object GT key 접근 시 실패하는 sentinel fixture, pure helper 입력 검토. K GT reference를 Feature/Edge 데이터나 거리 filter로 사용하지 않음 |

실제 대표 프레임에 특정 Silhouette-only/Surface-only invalid 사례가 없으면 N/A를
명시하고 synthetic fixture로 보완한다. 사례를 만들기 위한 threshold/ratio 변경은 하지 않는다.
Frame 번호는 물리 point의 시간상 identity를 뜻하지 않는다.

대표 frame의 dense cache gather를 먼저 검증하고, 같은 8개 frame에서 기존
`extract_sampling_evidence`를 **frame당 한 번** 실행해 cache와 normal validity·후보·측정값을
대조한다. 이는 저장 Feature의 공백을 새 runtime에서 메울 수 있는지 확인하는 검증이다.
기존 고정 sampling 결과를 다시 선택하지 않는다. Mask/validity/candidate는 exact,
float normal/variation/depth는 명시한 수치 오차 기준으로 비교한다. Variation/depth의
cache parity 절대 허용오차는 각각 1e-6 rad/1e-6 m다. Estimator/parameters가 다르거나
차이가 설명되지 않으면 새 Feature로 기존 sampling 근거를 바꿔 덮지 않고 중단한다.

후속 구현 후 repository root의 기존 CPU `.venv`로 실행할 명령은 다음과 같다.
아래 새 module/CLI는 **계획이며 현재 존재하는 API가 아니다.**

```bash
.venv/bin/python -m unittest visual.runtime.test.test_vision_features visual.runtime.test.test_variable_rate_graph_contract -v
.venv/bin/python -m unittest visual.runtime.test.test_intra_object_edges -v
.venv/bin/python -m unittest visual.runtime.test.test_variable_rate_shape_first visual.runtime.test.test_sampling_geometry -v
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task04-mpl .venv/bin/python -m visual.runtime.test.run_vision_feature_edge --frames 0 18 27 45 56 66 80 120 --keep-ratios 0.5 1.0 --phase 0 --device cpu --repeats 5
```

Logger/Publisher 구현은 바꾸지 않는다. 해당 공통 테스트의 재실행은 새 runner의 산출물
등록을 실제로 바꾸거나 관련 실패가 생길 때만 추가한다. 설치 누락은 보고하고 설치하지 않는다.

### 6.3. 계산 시간·메모리

CPU에서 3회 warmup 후 5회 측정한다. 저장 선택 결과를 사용하며 I/O/plot/압축/logger 시간을
core latency에서 분리한다. 실제 runtime extraction 비용은 cached gather 비용과 별도로 측정한다.
대표 frame/ratio/object별 N, support 수, component 수, E, degree/길이 분포를 함께 기록한다.

- 단계: geometry/PCA5/features/pairs, Feature gather, EE 변환, support components,
  k-NN/dedup, packaging, receiver, 합계. Extraction은 두 ratio가 공유하며 중복 합산하지 않는다.
- 시간: sample 수와 mean/median/p95/max(ms). 대표 frame 소수의 p95가 sequence p95가 아님을 명시한다.
- 메모리: 입력 evidence/point/Feature/mask/compact/provenance/Edge의 `nbytes`, receiver 복사본,
  temporary 추정, 별도 CPU worker의 peak RSS 및 기준 RSS. Python heap 측정만으로 NumPy native memory를
  대체하지 않는다. macOS RSS byte/Linux KB 차이는 platform 정보와 함께 정규화한다.
- 규모 참고: K 기록상 ratio .5 최대 N=958, ratio 1 최대 N=1712다. E≤16N일 때
  int64 Edge만 최대 각각 245,248 / 438,272 bytes다. 이는 이론 상한이며 실제 메모리 측정이 아니다.
  Dense covariance/support 및 receiver copy 메모리도 따로 포함한다.

성능 수치를 미리 PASS로 가정하지 않는다. 실패/OOM/비유한 값이면 해당 frame과 단계,
마지막 N/E를 남기고 중단한다. 50ms를 넘으면 20Hz 미입증/초과로 보고하며 같은 작업에서
point cap·policy 변경·GPU 최적화를 시작하지 않는다. CPU validation PASS는 GNN/BC 성공,
physical edge 정확도, 실시간 20Hz 또는 다른 backend 검증을 의미하지 않는다.

## 7. 구현 순서와 중단 조건

1. **입력·계약 확인:** foundation 공백 해소, 현재 파일/dirty 상태·입력 hash·source schema 확인.
   승인 범위와 문서 충돌 또는 입력 누락이면 구현 시작 전 중단한다.
2. **Feature만 구현:** `vision_packet.py` gather/diagnostics/validity와 pure contract test 작성.
   기존 PCA5 결과만 소비한다. source/XYZ/object/근거 mismatch, invalid 은폐 또는 모드 오해가 있으면 중단한다.
3. **EE·receiver 연결:** Feature frame/pose 검증과 v2 opt-in receiver 추가, v1 회귀 확인.
   Frame/normal/compact/availability/깊은 복사 계약이 모두 통과해야 다음 단계로 간다.
   Feature 검증용 중간 packet의 Edge는 `not_implemented`/`[2,0]`로 명시한다.
4. **Feature CPU 대표 프레임 확인:** cache gather와 실제 extraction parity, 빈 Box/가변 N 확인.
   여기까지 PASS 전 Edge 생성은 시작하지 않는다. 새 estimator가 필요하다는 증거가 생기면
   필요한 확장과 근거를 보고하고 이 계획을 수정한다.
5. **Edge helper 구현:** §4 한 방법만 적용하고 index/component/대칭/고립점·shortcut 진단과 receiver 보존 확인.
   오연결 문제가 최소 baseline의 한계를 넘으면 정책을 임의 조정하지 않고 결과와 함께 중단한다.
6. **CPU 비용·산출물 검증:** 고정 8프레임×2조건만 측정, JSON/NPZ 재읽기와 HTML/PNG/상대 링크 확인,
   기존 보호 파일 hash 확인 후 결과 Markdown 작성. Partial/FAIL은 통과 단계와 미실행 단계를 분리한다.

현재 계획 문서의 승인은 구현 진행 허용의 경계다. 이후 동작은 승인된 위 순서를 따르며,
추가 sweep/GT 사용/학습/게시로 범위를 확장하지 않는다.

## 8. 승인 후 결과 경로와 정적 HTML 계획

결과 Markdown 경로는 `docs/visual-graph-integration/04_vision_feature_edge.md`다.
후속 승인 후 한 번의 `ExperimentLogger(team='visual', job_type='evaluate',
model='task04_vision_feature_edge', config=actual_config, seed=0)` context에서 새 UUID를 생성한다.
기존 run ID/project/group 규칙을 유지하며 `WANDB_MODE=disabled`로 로컬에 기록한다.
입력 K/AB run은 `record()`에 dependency 경로/hash로 명시한다. 입력 artifact에 `link_output()`을
호출해 원본 sidecar 소유권을 바꾸지 않는다. 계획 단계에는 UUID를 생성하지 않는다.

```text
experiments/<new_run_uuid>/
  vision_feature_edge.html
  vision_feature_edge.json
  vision_feature_edge.npz
  verification.json
  tests.log
  source_hashes.json
  plots/f0000_alignment.png ...
  plots/f0000_edges.png ...
  plots/validity.png
  plots/cost.png
  config.yaml / metadata.json / metrics.json / manifest.json
```

HTML은 외부 CDN 없이 열리는 정적 보고서다. Frame/ratio/object별 다음 내용을 담는다.

- 최종 행 index↔원본 `(v,u)`/source/XYZ/object/EE XYZ/Feature/validity 대응표와 mismatch 수.
- Normal/variation/depth별 availability와 validity, NaN 원인별 수, Silhouette-only/Surface-only 사례.
- PCA5/Silhouette 자격 교집합, Structural 선택 경로, Surface 선택을 서로 다른 범례로 표시.
- 최종 point 번호를 기준으로 같은 물체 내부 Edge를 2D overlay와 3D PNG에 표시.
  Object/component/고립점/긴 연결을 구분하고 physical edge 검출 결과가 아님을 명시.
- 빈 Box, 관측 복구 및 실제 가변 N/E, ratio 1.0 대비 .5의 고정 결과.
- 단계별 실제 CPU 시간·메모리와 이론 상한, cache/extraction 비용의 구분 및 실행하지 않은 backend.
- `vision_feature_edge.json`, `vision_feature_edge.npz`, `verification.json`, `plots/*.png`의
  상대 링크. NPZ에는 모든 행/Edge/Feature/mask/근거/transform을 저장하고 JSON에 schema를 남긴다.
  HTML의 표시를 줄이더라도 원본 데이터 행이나 Edge를 제거하지 않는다.

새 결과만 기존 `logger.link_output`으로 등록한다. HTML/PNG/log/Markdown은 `kind='reports'`,
JSON 수치는 `kind='metrics'`, NPZ는 기존 `kind='trajectory'` inventory로 등록한다.
정적 HTML의 로컬 링크를 readback 검사하고 대표 PNG를 눈으로 확인한다.
게시 CLI, `--dry-run --report-files`(로컬 manifest 변경 포함), 원격 sync는 실행하지 않는다.
기존 `utils.experiment_artifacts`/`tools.publish_experiment`는 향후 명시적 게시 요청이 있을 때만 재사용한다.

## 9. 이번 계획 단계의 완료 범위

기존 구현·계약·대표 저장 schema를 읽고 이 계획만 작성했다. 새 Feature/Edge/Graph,
실험 UUID, 결과 Markdown/HTML/PNG는 생성하지 않았다. 121프레임 Sweep, 계약 테스트,
CPU feature/edge benchmark, simulator/renderer, CUDA/MPS/PCL, GNN/GRU/BC는 실행하지 않았다.
설치·외부 업로드·Git commit은 수행하지 않는다. 후속 실행의 PASS 여부는 아직 미검증이다.
