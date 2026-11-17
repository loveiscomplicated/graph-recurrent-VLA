# Task05 — Observation-Constrained Triangular Graph Edges Plan

상태: **계획만 제출 / 후속 구현 승인 대기**. 이번 산출물은 이 Markdown 한 개다.
새 Edge·삼각형 생성, 실험·benchmark, ExperimentLogger UUID 및 결과 HTML 생성은 하지 않았다.

## A. 기존 입력 재사용과 타당성

### A.1. 판단

**저장된 입력으로 2D Delaunay 후보와 관측 기반 검증을 구현할 수 있다.**
다만 관측 mask·depth가 함께 오염된 경우까지 물리적 표면 연결성을 보장하지 못한다.
특히 아래 Frame 0 사례는 국소 depth jump가 없는 원거리 관측이므로, 새 방식으로
263.469mm 연결이 반드시 제거된다고 기대해서는 안 된다. 최종 채택은 보류한다.

[MeshGraphNets 논문 §3.1](https://arxiv.org/html/2010.03409v4#S3.SS1)은 주어진
simulation mesh의 node/edge를 graph node/양방향 mesh edge로 옮긴다. 여기서는 그
connectivity 활용 아이디어만 참고한다. 우리의 연결성은 불완전한 단일 RGB-D 관측에서
추정하므로 논문의 mesh, remeshing, dynamics 정확도·속도 결과를 그대로 적용할 수 없다.

루트 지침, `README.md`, `envs/maniskill/README.md`, `docs/PROJECT_CONTEXT.md`,
[engineering rules](../engineering_rules.md), [tracking](../experiment_tracking.md),
[artifact 지침](../huggingface_artifacts.md), Visual README와 아래 현재 소스를 읽었다.
현재 checkout의 dirty/untracked Task04 구현을 기준으로 하며 초기 scaffold 설명보다
실제 소스·저장 배열을 우선한다.

### A.2. 재사용 정보와 실제 공백

| 기존 자료/API | 재사용할 정보 | 제약·추가 작업 |
| --- | --- | --- |
| [Task03-K](03k_temporal_shape_first.md), run `9185355a32b8433abc70f1738f983377` | PCA5 + Outer Silhouette + metric voxel + keep_ratio, phase 0의 고정 선택과 source/행 순서 | sampler를 다시 실행하지 않음 |
| [Task04](04_vision_feature_edge.md), run `9f90293b227e4dbfaaa6290afad9cc3b`의 `packets/f####_r*.npz` | EE points, camera points, F=5 Feature·validity·provenance·기존 directed Edge·component·변환·관측 identity | 원본을 읽고 복사. Edge 관련 출력만 새 후보에 맞게 교체 |
| 같은 run의 `runtime_evidence/f####.npz` | 512×512 camera XYZ, raw object masks, valid_depth, overlap, dense 기존 normal 진단, 물체별 전체 PairEvidence | RGB·calibration 전체는 이 cache에 없으므로 packet/원본 관측과 join |
| `visual/runtime/geometry.py::OrganizedGeometry.observed_mask` | `mask & valid_depth & ~overlap`; valid_depth는 finite-positive depth | 유한 양수인 원거리 outlier는 support에 남음 |
| `visual/runtime/edge_evidence.py::PairEvidence` | 원본 right/down pixel pair, supported, signed depth delta, depth_jump, mask/invalid/overlap 원인 | 최종 point끼리의 pair가 아님. normal angle은 진단만 사용 |
| `visual/runtime/intra_object_edges.py` | `observed_components`, 기존 symmetric k-NN(k=8), 기준선 진단 | 전체 support component를 그대로 재사용. 기존 k-NN 함수·정책은 보존 |
| `visual/runtime/test/run_vision_feature_edge.py::read_packet`, `save_packet` 및 기존 evidence loader | pickle 없는 NPZ/schema JSON 복원·저장 형식 | 평가 runner 전체를 호출하면 extraction/새 결과가 생기므로 호출하지 않음 |
| `graph/vision_input.py` | v2 packet 검증·deep copy, Point-local `[2,E]`, optional robot 3행 공급 시 +3 | **현재 `_validate_v2_edges`는 k-NN `EDGE_POLICY`와의 일치만 허용한다.** 새 policy의 명시적 분기가 필요 |
| 원본 `visual/data/step2/ep000.npz`, `visual/results/runtime_tracking/ep000/masks.npz` | RGB, camera-z depth, tracker mask, frame identity·timestamp, camera/TCP calibration | GT segmentation·mesh·물체 pose로 연결을 선택하지 않음 |

원본 경로는 기존 sampling `comparison.json`의 input/masks 설정에서 확인했다. 두 파일과
16개 Task04 packet, 8개 runtime evidence가 존재한다. 대표 저장 NPZ의 배열 schema를
읽었으며 원본 전체 hash·모든 행의 정합성 재검증은 후속 실행의 입력 관문으로 남긴다.
`source_index=v*W+u`에서 `(u,v)`를 복원한다. 이 source는 최종 Point row와 다르다.
backprojection의 pixel centre `(u+.5,v+.5)` 관례도 유지한다.

현재 루트 `.venv`에서 SciPy **1.17.1**, `scipy.spatial.Delaunay`와 `QhullError`의
import만 확인했다. Delaunay는 호출하지 않았다. Visual의 별도 lock에는 SciPy 1.18.1이
있으므로 후속 실행은 실제 interpreter/version을 기록하고 환경을 혼용하지 않는다.
[SciPy 공식 문서](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.spatial.Delaunay.html)에
따라 2D simplices를 final row로 매핑하고 Qhull 퇴화 오류·누락 vertex를 처리해야 한다.
새 dependency 설치는 필요하지 않은 후보이나 실제 생성 성공은 아직 미검증이다.

### A.3. 고정 조건과 저장 기준선

- Frame `0, 18, 27, 45, 56, 66, 80, 120`, primary ratio **0.5**, reference **1.0**, phase **0**.
- 기존 가변 N, Point XYZ·object_id·Feature·availability/measurement validity·provenance·row ordering·EE 변환을 exact 보존한다. NaN을 포함한 field는 equal-NaN 및 원본 배열 hash로 확인한다.
- RGB는 시각화용이다. normal angle·PCA5 구조 후보·선택 source는 연결 금지 조건으로 사용하지 않는다.
- 하나의 기존 episode에서 고른 8 frame × 2 ratio이며 독립 episode 16개가 아니다. 추가 episode/simulator 실행은 없다.

아래는 **기존 Task04 packet을 읽은 수치**이며 Task05 결과가 아니다. E는 directed다.

| frame | ratio .5: Peg / Box N | ratio .5: E | ratio 1: Peg / Box N | ratio 1: E |
| --- | ---: | ---: | ---: | ---: |
| 0 | 499 / 451 | 8486 | 896 / 798 | 15138 |
| 18 | 453 / 0 | 4112 | 807 / 0 | 7304 |
| 27 | 416 / 123 | 4888 | 733 / 150 | 7952 |
| 45 | 348 / 166 | 4490 | 593 / 237 | 7318 |
| 56 | 371 / 0 | 3220 | 636 / 0 | 5518 |
| 66 | 305 / 119 | 3666 | 505 / 125 | 5516 |
| 80 | 249 / 228 | 4128 | 398 / 349 | 6704 |
| 120 | 207 / 239 | 3774 | 316 / 378 | 6168 |

## B. 단일 후보: 2D Delaunay + 관측 support / local depth validation

### B.1. 후보 삼각형

동일 object의 **기존 observed depth-gated component별** 최종 선택 Point에 2D
Delaunay를 적용한다. 이는 face segmentation이 아니라 Task04의 관측 성분 제한 재사용이다.
3D Delaunay/사면체, 새 Point, Steiner vertex, 보간 3D Node는 만들지 않는다.

1. 기존 evidence에서 `observed_components`를 재사용해 component map을 얻고 저장 packet과 exact 대조한다. source/XYZ/support/identity가 맞지 않으면 중단한다.
2. object/component의 final row 목록을 만들고 원본 UV를 float64 작업 배열로 전달한다. 입력 packet 행은 정렬하거나 수정하지 않는다.
3. 2D Delaunay simplex의 작업 index를 final Point index로 되돌린다. 최종 출력은 canonical triangle row tuple/edge tuple 순서로 정렬해 포장한다.
4. 아래 mask·depth 규칙을 모두 통과한 삼각형의 세 변만 합친다. 한 변이 두 삼각형에서 나와도 unique undirected pair는 한 번만 세며 양방향으로 출력한다.

Qhull 옵션은 최초 구현에서 명시적으로 고정·기록한다(후보: 2D 기본 옵션
`Qbb Qc Qz Q12`). `QJ` perturbation, 점 추가, 실패 시 k-NN/2점 연결 fallback은 쓰지 않는다.
동일 입력·동일 설치 버전에서 재현성을 검사한다. Cocircular diagonal의 선택은 버전 간
동일함을 가정하지 않으며 실제 simplices와 library version을 저장한다.

### B.2. Mask와 관측 support: 꼭짓점 외에 삼각형 전체 검사

물체별 support `S = raw_mask & finite_positive_depth & ~overlap`를 사용한다.
선택되지 않은 원본 관측 pixel도 **검증 근거**로 사용하며 Graph Node로 추가하지 않는다.

각 삼각형 T와 세 변에 대해 raster **supercover**를 사용한다. UV 정수 좌표를 pixel
centre로 놓고 pixel cell을 `[u-.5,u+.5] × [v-.5,v+.5]`로 정의한다. 이는 모든 좌표에
`.5`를 더하는 원본 관례와 동치다. 닫힌 T 또는 변과 cell이 접촉하는 모든 pixel을
포함한다. 꼭짓점·중심 몇 개만 검사하거나 일정 간격 subsampling하지 않는다.

- 삼각형 내부와 경계의 모든 supercover pixel이 같은 object의 S여야 한다.
- 하나라도 mask 밖, invalid depth, overlap, image 밖이면 triangle을 reject한다. Hole filling, erosion/dilation, 허용 비율은 도입하지 않는다.
- 대각선이 pixel cell 모서리를 정확히 통과하면 접촉하는 cell을 모두 포함한다. 임의의 한쪽 raster 경로만 택해 1px gap을 건너뛰지 않는다.
- half-pixel 정수 배율 좌표의 판정으로 경계 포함 규칙을 고정하고 synthetic hole/concavity/diagonal 사례로 검사한다.

이는 **현재 해상도의 관측 공백**을 보수적으로 막는 규칙이다. 경계에 스치는 unsupported
cell 때문에 정상 silhouette triangle을 과도하게 버릴 수 있다. Subpixel hole, 잘못 채워진
tracker mask, 같은 mask에 섞인 배경은 완전히 검출하지 못한다. Mask에 표시된 hole과
물리적 hole을 구분하며, 구멍 안에서 실제 관측된 안쪽 표면까지 무조건 공백으로 취급하지 않는다.

### B.3. Depth: 이미지 경로상의 국소 jump를 검사

기존 PairEvidence의 **signed adjacent depth delta와 depth_jump**를 우선 재사용한다.
선택점의 `depth_jump_m` scalar만으로 edge 경로를 추정하지 않는다.

1. 각 변의 supercover에서 right/down 4-neighbour pair를 조회한다. 역방향은 같은 pair를 사용한다. 대각선 corner-touch에서는 두 직교 경유 cell을 모두 포함한 보수적 경로를 검사한다.
2. 세 변뿐 아니라 T 전체 supercover 안의 right/down 인접 pair도 검사한다. 변은 정상이어도 내부에 가려진 depth seam이 있는 triangle을 막기 위함이다.
3. 검사 대상 pair는 양 끝 supported, finite delta, `depth_jump=false`여야 한다. 필요한 pair가 없거나 delta가 finite하지 않으면 검증 불가로 triangle을 reject한다. Cache support/identity 자체의 모순은 입력 오류로 실행을 중단한다.
4. T의 vertex component 일치만으로 이 검사를 생략하지 않는다. 다른 곳으로 우회해 같은 component가 된 두 점도 직선 경로에서는 gap/jump를 가로지를 수 있다.

기존 gate는 인접 pixel의 camera-z에 대해
`abs(z_b-z_a) > 0.01 m + 0.02 * min(z_a,z_b)`다. Task04와 같은 값을 고정하고
cache/source 설정을 기록한다. 이는 기존 관측 gate이며 물리적 연속성의 증명은 아니다.
변 양 끝의 총 `abs(z_i-z_j)`에 이 threshold를 적용하지 않는다. 누적 depth 변화가 커도
각 인접 변화가 gate 안인 경사진 표면은 허용한다.

Supercover 내 모든 pair를 검사하는 방식은 연속 경로의 존재만 찾는 것보다 보수적이다.
Pixel cell 접촉 때문에 실제 T 밖에 가까운 jump까지 검출할 수 있고, grazing view의 정상
표면도 1px depth 변화가 크면 reject될 수 있다. 반대로 작은 변화가 길게 누적되는 잘못된
관측은 통과할 수 있다. 새로운 length/z cap이나 normal gate로 이 한계를 숨기지 않는다.
새 물리 threshold는 추가하지 않는다. Unsupported pixel/jump의 허용 개수는 0인 보수적
관측 정책이며, 그로 인한 silhouette/corner 과분리를 반드시 별도 평가한다.

### B.4. Corner/crease 허용과 Edge 출력

**서로 다른 normal을 갖는 이웃 면도 위 관측 검사를 통과하면 허용한다.**
20°는 기존 PCA5 Structural 후보의 기준이며 Edge veto가 아니다. Face label,
동일 평면 검사, plane residual, RANSAC, GT mesh/face labeling은 사용하지 않는다.
Corner를 가로지르는 3D 직선 변은 message-passing 연결이며 실제 surface geodesic이나
평면 mesh facet이라고 주장하지 않는다.

삼각형 하나라도 검증에 실패하면 그 삼각형 전체를 버린다. 유효해 보이는 두 변만 따로
salvage하지 않는다. 단, 이웃의 **다른 accepted triangle**이 같은 변을 가지면 그 변은 남는다.
최종 undirected pair 집합 U에서 `(i,j),(j,i)`를 만들며 **directed E=2|U|**다.
삼각형은 별도 Graph Node가 아니다. 공유 변 때문에 일반적으로 E를 `6T`로 세면 틀린다.
유효한 triangle에 속하지 못한 기존 node는 삭제하지 않고 isolated로 남긴다.

### B.5. 퇴화·실패 상황의 고정 처리

| 상황 | 최소 처리와 한계 |
| --- | --- |
| convex hull이 오목한 외곽을 메움 | full supercover support 검사로 reject. 경계 복구/retriangulation은 안 함 |
| mask 내부 hole | 꼭짓점이 모두 유효해도 내부의 unsupported pixel로 reject. Filled/오염 mask hole은 보장 불가 |
| 분리된 mask/depth component | 기존 component별로 독립 triangulation. 우회 연결이 있는 component 내부도 local 검증 |
| depth seam/occlusion | 경계·내부의 기존 jump/invalid 검증. 같은 오염 plateau나 subthreshold seam은 남을 수 있음 |
| silhouette의 불규칙 밀도 | N/위치는 고정. 긴 boundary triangle rejection·node 고립을 보고하고 보간/재sampling 안 함 |
| 긴·가느다란 triangle | length/aspect/min angle은 진단값. 면적 양수이고 관측 검증을 통과하면 임의 quality threshold로 제거 안 함 |
| 정확한 일직선·중복 UV | 정수 UV의 cross product로 affine rank와 zero area를 판정. 동일 object의 중복 source는 입력 오류. collinear set은 T=E=0, 모든 node 보존 |
| 거의 일직선·cocircular | 실제 integer UV의 양수 면적은 임의 epsilon으로 없애지 않음. QhullError/omitted vertex/coplanar 진단 기록. 보정 jitter 없이 실패 component는 빈 Edge로 표시 |
| component에 N=0/1/2 | T=U=E=0. 2점에 예외 Edge를 만들지 않음. N=0의 component/isolated는 0, N=1/2는 그대로 고립 |
| 원거리 finite-positive outlier | Point/Feature를 보존. 주변 jump가 있으면 reject될 수 있으나 일관된 원거리 plateau는 통과 가능. 최종 물리성 판정 불가 |

일반 Qhull/코드 예외를 정상 퇴화로 뭉뚱그리지 않는다. 입력 오류·예상하지 못한 계산 오류는
해당 실행을 실패로 표시한다. 오류 component의 빈 Edge도 성공적인 관측 reconstruction으로
집계하지 않는다. 단순 vertex 미포함은 원인·고립 node를 공개하고 채택 검토를 보류한다.

## C. 최소 offline 비교·계약·변경 파일

### C.1. 후속 승인 후 실행 순서와 ablation

1. 원본 dirty/untracked 및 Task04 artifact hash/sidecar를 보호하고 새 run에 입력·소스 hash, 실제 config·버전·Git dirty diff/source snapshot을 기록한다. 기존 run은 입력 dependency다.
2. 16개 저장 Task04 packet에서 points/Feature/provenance를 직접 읽고 evidence·원본 RGB-D identity와 대조한다. K 선택 hash/행 정합성을 확인한다. **Sampling과 Feature extraction은 재실행하지 않는다.**
3. 기존 k-NN 함수를 같은 입력과 기존 component로 실행해 저장 Task04 Edge·정책·component를 exact 재현한다. 실패하면 비교를 진행하지 않는다.
4. synthetic 계약 테스트를 통과한 후 동일 UV의 Delaunay 후보를 한 번 생성하고 아래 validation stage를 비교한다. GT 데이터는 열지 않는다.
5. 수치·Edge/triangle·rejection witness를 저장한 후 시각화/성능 계측/상대 링크 readback을 검증한다. 결과에 따라 채택 또는 보류를 보고하고 baseline은 유지한다.

| arm / stage | 목적 | 최종 후보 여부 |
| --- | --- | --- |
| K8 | Task04 관측 성분 제한 3D symmetric k-NN(k=8) | 고정 비교 기준 |
| T0: raw | object/component별 동일 2D Delaunay 후보 | mask/depth 위험 확인용 중간 상태 |
| TM: support | T0에 full-triangle/edge support 검증만 적용 | mask rejection 효과 분리용 중간 상태 |
| TMD: support+depth | TM에 boundary/interior PairEvidence gate 적용 | **유일한 새 비교 후보** |

T0/TM은 같은 candidate set의 stage별 진단이며 별도 정책 sweep/채택 후보가 아니다.
모든 물체·frame·ratio에 같은 규칙을 적용한다. Ratio끼리는 원래 N이 다르므로 방법 비교는
각 frame/ratio/object 안의 paired 차이로 수행한다. 빈 Box 조건도 빠뜨리지 않는다.

### C.2. 측정 지표와 검증 가능한 정확성

| 질문 | 기록할 값 |
| --- | --- |
| 크기 | object/component별 실제 N, candidate/accepted/rejected T, unique undirected U, directed E; K8의 triangle 수는 N/A |
| 길이 | unique undirected Edge의 camera64 Euclidean mean/p95/max(m·mm), UV length, top-10 source/UV/XYZ; EE 표현과 구분 |
| 관측 공백 crossing | K8/T0/TM/TMD의 unique edge를 같은 supercover/path 감사기로 검사: mask 밖·invalid·overlap·jump·검증불가의 수/비율, numerator/denominator와 witness |
| 내부 위험 | triangle boundary는 통과했으나 내부 support/jump로 reject된 수; Edge만 비교해서 놓치는 영역 별도 표시 |
| 모서리 허용 | 양 끝 normal 유효성, normal angle 분포, 20° 초과 Edge의 유지/소실 및 stage별 사유. 이는 cross-face 오검출률/GT 정확도가 아님 |
| 연결성 | 최종 Graph의 undirected degree histogram/min/median/p95/max, isolated 수·비율, connected component 수와 전체 크기 목록, largest-component/N |
| 과분리 | 기존 observed component별 K8 대비 Graph component 분할·새 isolated row/source 수; 관측 성분과 최종 Graph component를 명확히 구분 |
| 제거 이유 | candidate triangle별 모든 reason bit와 최초 reject stage; unique candidate edge별 local invalid 사유 및 `no_accepted_incident_triangle`; 남은 shared edge와 구분 |
| 원거리 연결 | 두 방식 top-10 길이, 그 endpoint camera-z, 기존 observed support camera-z 분포와 전체 finite extent. Outlier를 clipping해 통계에서 숨기지 않음 |

성분 수는 단순히 `observed_component_id`의 unique 수로 대신하지 않는다. 모든 Point를
포함한 실제 최종 Edge adjacency에서 다시 구한다. 빈 graph의 길이/degree quantile/비율은
정의에 따라 null/N/A, N=0의 component 수는 0이다. Isolated node도 component 크기 1에 포함한다.
Micro 합산과 object/frame별 paired 값은 분리하며 긴 Edge max를 pooled mean으로 숨기지 않는다.

같은 감사 규칙으로 TMD가 violation 0인지 확인하는 것은 **규칙 준수 검사**다.
물리적 mesh 정확성 평가는 아니다. 독립적인 작은 synthetic expected mask/pair 사례와
원본 RGB-D 시각 검토를 함께 사용하고, 실제 표면의 true/false Edge precision으로 부르지 않는다.

### C.3. 필수 시각 진단

Frame 0 Box의 Task04 Edge **약 263.469mm**, `(251,206)↔(259,206)`를 baseline의
동일 row/source로 추적한다. [기존 확대 PNG](../../experiments/9f90293b227e4dbfaaa6290afad9cc3b/plots/f0000_r0.5_shortcut_detail.png)를 읽어 확인했다.
이번 읽기에서 해당 9개 pixel은 모두 mask=true, valid=true, overlap=false이고 저장된
camera-z는 모두 `8.430999755859375 m`였다. 그 사이 기존 right pair 8개는 supported 양 끝이
true, signed delta=0, depth_jump=false였다. **Edge 후보 생성/검증 실험 없이 기존 배열만 읽은 사실**이다.

따라서 해당 선분의 1D 경로는 기존 support/local-depth 조건으로 걸러질 근거가 없다.
후속 비교에서는 (a) Delaunay candidate에 그 변이 존재하는가, (b) incident triangle 내부가
다른 관측 문제로 reject되는가, (c) 최종 Edge로 남는가를 별도 표시한다. 변이 없어져도
같은 원거리 point 사이에 다른 긴 Edge가 생길 수 있으므로 원거리 component 전체를 함께 본다.
이 pixel이나 Box hole 좌표를 selection/threshold에 넣지 않는다.

정상 corner 예시는 고정 frame 전체에서 원본 RGB-D상 연속 관측된 굽은 영역을 시각적으로
확인하고, baseline/후보의 양쪽 normal과 경로 support를 표시한다. 20° 초과 값은 진단 강조에만
사용한다. 고정 row/source와 검토 근거를 남기며 face label이나 새 점을 만들지 않는다.
실데이터에서 명확한 사례를 확보하지 못하면 `NOT_ESTABLISHED`로 표시하고, normal만
바꿔도 Edge가 불변인 synthetic folded-surface 테스트와 구분한다. 선택된 node 배치 자체가
연결을 제공하지 않으면 그 한계도 공개한다.

### C.4. 계약 테스트 계획

| 테스트 | 기대 계약 |
| --- | --- |
| object/component/index | 같은 object·기존 observed component 내부, 원본 final Point 범위, triangle vertex↔source 매핑 정확 |
| Edge algebra | int64 `[2,E]`, self-loop/directed duplicate 없음, 모든 변 양방향, E=2U; shared triangle edge dedup |
| support | 꼭짓점은 유효하지만 내부 1px hole/concavity/invalid/overlap을 갖는 triangle reject; diagonal corner-touch와 image boundary 포함 |
| local depth | endpoint delta는 0이어도 중간 seam이면 reject; endpoint 총 delta가 커도 모든 국소 pair가 통과하는 ramp는 유지; 내부 seam만 있는 triangle도 reject |
| corner/normal | 관측 support·local depth가 이어지는 folded surface에서 20° 초과 normal과 PCA candidate flag 변경으로 T/Edge가 변하지 않음; invalid normal도 단독 veto 없음 |
| 퇴화/소수점 | empty object, N=1/2, collinear, nearly collinear integer UV, cocircular, duplicate source 오류, QhullError/누락 vertex 처리 |
| 불변성 | Point·Feature·compact fields·validity·provenance·행 순서·EE/calibration/timestamp가 원본과 동일; 입력/원본 cache hash 보존 |
| GT 독립성 | GT 접근 시 실패하는 observation-only fixture, GT/face/object-pose import·접근 금지 검사; object ID를 바꾼 동일 관측에도 같은 규칙 |
| 일관된 validation | 모든 object/frame에 같은 규칙; object명·특정 UV/hole 위치에 따른 분기 없음; triangle reject와 최종 surviving shared edge 구분 |
| receiver/offset | 기존 v1/v2 회귀, 새 policy만 명시 수용, unknown policy 거부; robot 미공급 offset 0, 명시적으로 실제 공급한 synthetic robot `[3,3]` 테스트에서만 +3 |
| baseline 재현 | 16개 저장 packet의 k-NN Edge와 component exact 일치; 기존 테스트와 원본 baseline policy 검증 유지 |
| artifact | JSON/NPZ exact readback, 원본 보존 hash, PNG decode·시각 검토, Markdown↔HTML 상대 링크; 미실행 검사는 NOT_EXECUTED |

후속 테스트는 기존 `test_intra_object_edges`, `test_vision_features`,
`test_variable_rate_graph_contract`와 새 triangle 테스트를 실행한다. 실제 simulator robot
좌표 수집은 하지 않는다. PASS는 계약과 관측 규칙 준수만 뜻하며 물리적 reconstruction,
GNN 성능, BC 성공, 최종 정책 채택을 의미하지 않는다.

### C.5. 최소 변경 파일 제안 — 지금은 생성/수정하지 않음

| 파일 | 후속 최소 역할 |
| --- | --- |
| `visual/runtime/triangle_intra_object_edges.py` (신규) | Delaunay, integer UV degenerate 처리, supercover·PairEvidence validation, final-row Edge와 audit 반환. 기존 component helper 재사용 |
| `visual/runtime/test/test_triangle_intra_object_edges.py` (신규) | 위 synthetic/receiver/불변성/GT 독립 계약 |
| `visual/runtime/test/run_triangle_intra_object_edges.py` (신규) | 저장 입력 전용 geometry ablation, baseline 재현, 계측·HTML/JSON/NPZ/PNG. sampler/estimator/GT reader 호출 없음 |
| `graph/vision_input.py` (좁은 수정) | `_validate_v2_edges`에 새 명시 policy 분기. 공통 index/object/component/self-loop/symmetry 검증은 유지. Graph topology 생성 책임 추가 없음 |
| `docs/visual-graph-integration/00_interface.md` (좁은 수정) | opt-in triangle policy·metadata·지원 한계 명시; 기존 k-NN 계약 보존 |
| `docs/visual-graph-integration/05_triangle_intra_object_edges.md` (신규, 승인 후) | 실제 결과·수치·판단과 HTML 양방향 링크 |

기존 `visual/runtime/intra_object_edges.py`, `vision_packet.py`, Sampling/Feature estimator,
shared simulator, robot/inter-object 정책을 변경할 필요는 없다. 필요 없는 공통 policy registry나
parallel receiver를 새로 만들지 않는다. v2 Feature schema와 수신 함수는 재사용한다.
신규 helper의 예정 입력은 기존 camera_points/object_id/source_index/component_maps와
저장 evidence이며, 출력은 Point-local `edge_index`, row-aligned `component_id`, 별도
triangle/rejection `audit`다. Audit의 triangle vertex는 int64 `[T,3]` final row로 보존하고
기존 `intra_object_edge_index[2,E]`에만 Edge를 넣는다. 좌표 변환을 다시 수행하지 않는다.
신규 policy metadata는 method/version, 2D UV convention, component gate, Qhull options,
support/path raster rule, 기존 depth gate와 evidence hash, no-normal-veto를 명시한다.
원래 generation provenance는 그대로 두고 Edge 정책/audit namespace만 추가한다.
기존 k-NN `E≤16N` 분기는 유지하고 triangle 분기에는 `E=2U`, `U≤3*T_accepted` 등
해당 생성 방식의 검사를 적용한다. Receiver는 dense mask 검증을 재실행한 것처럼 주장하지 않는다.

## D. CPU·메모리 평가, 실패·중단 기준과 산출물

### D.1. CPU와 메모리

후속 계측은 같은 CPU/interpreter에서 입력을 메모리에 읽은 뒤 **3회 warmup + 5회 측정**한다.
K8/TMD 실행 순서를 반복마다 교대하고 thread/host/버전과 raw sample을 기록한다.
5개 표본의 p95는 기술통계이며 통계적 우월성 근거로 사용하지 않는다.

- 공통 비용: support/component 생성·검증, source/UV 정렬 확인·lookup 준비. 두 ratio가 공유하는 frame 작업은 1회로 기록한다.
- K8: 기존 k-NN+dedup 호출, packaging, receiver를 분리한다. 기존 함수 내부를 benchmark 때문에 수정하지 않는다.
- TMD: triangle 생성, boundary/interior rasterization·mask validation, PairEvidence lookup 준비·depth validation, edge dedup/양방향 packaging, receiver를 분리한다.
- Core 전체 wall time: 공통 준비+각 방법 전체 처리. Stage timer 합과 실제 wall subtotal을 함께 기록한다. 검증 lookup 준비/validation 비용을 누락하지 않는다.
- Offline 전체 비용: NPZ decode/read, diagnostics, 그림/HTML/JSON 압축·쓰기·logging을 core 밖 별도 계측한다. 이번에는 extraction을 실행하지 않으므로 새 end-to-end sensor latency로 해석하지 않는다.
- 메모리: 방법/조건별 fresh CPU worker의 baseline/peak RSS와 증가분, native Qhull 포함 한계, packet/evidence/triangle/edge/lookup/scratch 배열 bytes를 구분한다. macOS RSS 단위를 명시한다.

길이 진단·전체 invalid witness 저장 등 보고서 전용 작업은 core와 분리하되 실제 core가
사용하는 validation·dedup 비용은 모두 포함한다. Memory worker에서도 sampling/추출은 없다.
기존 Task04 시간은 참고값으로 남기고 속도 비교는 같은 새 실행의 baseline 재계측으로 한다.

2D Delaunay의 일반적인 `O(N log N)` 생성 비용만으로 빠르다고 결론 내리지 않는다.
Raster 검증은 triangle bbox 검사량, supercover 면적·경로 길이·PairEvidence 수에 좌우된다.
긴/가느다란 triangle의 bbox 비용도 기록하고 frame 크기를 넘는 전체 거대 broadcast allocation은
피한다. 이는 CPU 구현의 최소 메모리 처리이며 GPU/PCL 최적화가 아니다.
E 감소·mean/p95/max latency·peak RSS는 각각 독립적으로 비교한다. 기존 50ms 기준 초과도
표시하되 이 offline edge 작업만으로 20Hz 통합 성능을 주장하지 않는다.

### D.2. 실패·보류·중단 기준

| 구분 | 기준과 처리 |
| --- | --- |
| 입력/재현 실패 | 파일·identity·hash·source/Feature/행 정합성 또는 k-NN exact parity 실패 시 즉시 중단. 재sampling·tolerance 확대·기존 run 수정으로 통과시키지 않음 |
| 계약 실패 | cross-object/component, 범위 밖 index, duplicate/self-loop/asymmetry, 입력 mutation, GT 의존 또는 normal-only veto가 하나라도 있으면 후속 비교 중단 후 수정 필요 |
| validation 실패 | accepted T/Edge에서 규칙상 support/jump 위반이 하나라도 있거나 raster oracle와 불일치하면 구현 실패. 관측 오염 plateau 통과는 별도 알려진 관측 한계로 기록 |
| 계산 실패 | 예상 퇴화는 명시 상태와 빈 Edge로 기록. 예상 밖 Qhull 오류/미포함 node/검증 불가가 남으면 해당 조건을 성공으로 채우지 않고 채택 보류. OOM/timeout 등은 조건 실패이며 N/ratio를 줄여 대체하지 않음 |
| 연결성 손실 | K8에서 비고립이던 node의 신규 고립, 기존 selected Graph component의 추가 분할, 확인한 정상 corner 연결 소실이 하나라도 있으면 해당 source와 사유를 의무 검토하고 채택 보류. 공백을 막기 위한 정당한 분리인지 결과에서 판단 |
| 정확성 개선 미확립 | 알려진 의심 연결이 남거나 다른 원거리 연결로 대체되면 그대로 공개. 위반 감소가 없거나 규칙 감사 이외의 정상 corner 근거가 없으면 개선을 주장하지 않음 |
| CPU·메모리 | TMD 전체 core mean이 K8 이상이면 속도 개선 없음, max/peak RSS 악화도 별도 표기. 조건별 50ms 초과를 보고. 정확성을 희생하는 임계값 조정/최적화로 이번 ablation을 연장하지 않음 |

연결성/속도의 단일 임계값으로 최종 채택을 자동화하지 않는다. 위 조건은 실패와 검토 대상을
놓치지 않기 위한 기준이다. 신규 고립이 없더라도 물리적으로 맞는 Graph라는 뜻은 아니다.
원거리 plateau 해결에 range prior/새 filtering/새 threshold가 필요하다고 판단되면 **별도
설계·승인 대상으로 남기고 이번 실행은 종료**한다. 특정 Box hole에 맞춘 값은 도입하지 않는다.

### D.3. 승인 후 결과 Markdown / HTML 계획

승인된 geometry 평가 시작 시에만 공통 `utils.experiment_logger.ExperimentLogger`를
`team='visual'`, evaluation job으로 한 번 생성한다. UUID는 Logger가 생성하며 임의 지정하거나
Task04 UUID를 재사용하지 않는다. 기존 project/group 규칙과 local logging을 유지하고,
외부 업로드 없이 `WANDB_MODE=disabled`로 실행한다. 입력 run ID/hash와 소스 snapshot을
기록하고 새 결과만 `link_output()`으로 연결한다. Publisher 실행 및 로컬 manifest를 바꾸는
publishing dry-run도 이 작업에 포함하지 않는다.

후속 결과 위치와 링크는 **예정 경로**다. `<run_id>`는 이번에 생성하지 않는 placeholder다.

| 출발 파일 | 후속 상대 링크 |
| --- | --- |
| `docs/visual-graph-integration/05_triangle_intra_object_edges.md` | `../../experiments/<run_id>/triangle_intra_object_edges.html` |
| `experiments/<run_id>/triangle_intra_object_edges.html` | `../../docs/visual-graph-integration/05_triangle_intra_object_edges.md` 및 `../../docs/visual-graph-integration/05_triangle_intra_object_edges_plan.md` |
| 위 결과 Markdown | `../../experiments/<run_id>/triangle_intra_object_edges.json`, `.npz`, `plots/*.png`, `verification.json` |
| 위 HTML | `triangle_intra_object_edges.json`, `.npz`, `plots/*.png`, `verification.json`, `tests.log`, `source_hashes.json` |

정적 HTML에는 다음을 포함한다.

- 동일 Point의 K8/TMD 2D/3D 비교, raw RGB·depth·mask와 accepted/rejected triangle overlay. Ratio/object/frame 선택과 row/source 표기를 맞춘다.
- 정상 corner/crease의 서로 다른 normal과 유지된 연결, hole·unsupported·depth seam의 reject witness. 20° 초과를 오류 색상으로 분류하지 않는다.
- Frame 0 의심 pair의 후보 포함/최종 유지/incident triangle 사유, 전체 원거리 범위와 표시용 확대를 나란히 제공한다. 축 범위 축소가 데이터 제거로 해석되지 않게 한다.
- T0→TM→TMD triangle/edge 수, rejection 이유, 길이·degree·isolated·실제 Graph component 분포와 K8 대비 paired 차이.
- CPU stage/전체 비용·5회 raw timing·peak RSS/array bytes, 미정의·실패·미실행 상태.
- triangle→vertex row→source/UV, candidate edge→accepted incident triangle, pair depth witness를 추적할 JSON/NPZ와 PNG **상대 링크**.

PNG decode, JSON/NPZ readback, 모든 상대 링크, 원본 artifact hash, 실제 브라우저 표시와
시각 검토 여부를 따로 기록한다. 보고서 생성 실패 시 저장된 수치만으로 같은 새 UUID의
보고서를 복구하며 재sampling/새 Edge·benchmark를 자동 재실행하지 않는다.

### D.4. 이번 작업의 종료 범위

계획 검토에서는 기존 소스·문서·저장 NPZ와 PNG를 읽고 SciPy import만 확인했다.
새 triangle/Edge·실험·benchmark·결과 UUID/HTML은 생성하지 않았다. 코드 테스트는 실행하지
않고 문서 링크·변경 범위·원본 보존만 확인한다. Simulator/renderer, CPU simulation,
CUDA/MPS/GPU/PCL, 학습·rollout은 모두 이번 호스트에서 실행하지 않았다.

Sampling/keep_ratio/voxel, Feature encoding/estimator, stale geometry, Graph inter-object
Edge, GNN/GRU/BC, dependency 설치, 외부 업로드, Git commit은 수행 범위 밖이다.
기존 dirty/untracked 상태와 Task04 artifact를 보존하고 **이 계획 제출 후 중단한다.**
