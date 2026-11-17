# Task03-I — Shape-First Voxel Sampling & PCA5 Spatial-Support Scale：구현 전 계획

상태: **계획만 작성. 구현·새 선택·진단 계산·실험·테스트는 승인 후 진행한다.**
이번 변경은 이 Markdown 하나다. 결과 Markdown, HTML, ExperimentLogger UUID는 만들지 않는다.
기존 dirty/untracked 파일, source, 입력, 실험 결과와 sidecar를 보존한다.

추천하는 첫 비교는 **물체별 고정 metric nominal voxel → score-free 원본 대표점 →
초과 시 기존 3D FPS로 Priority 최대 32점 → 전체 X의 기존 3D Surface FPS**다.
Voxel 크기를 키워 선택하는 대안은 이번 선택 arm에 포함하지 않는다.
해상도와 예산의 양립 여부를 보기 위한 dyadic occupied-count 진단은 선택과 분리한다.
이는 Shape-First, Surface-Second 가설을 검증하는 최소 control이며 채택된 정책이 아니다.

## 1. 고정 입력·기존 근거·확인 범위

ep000 Frame 0, base_camera, S3 optimized PCA5, orientation-only **>20°**, seed 0을 고정한다.
C는 저장 Structural 후보, X는 원본 observed support다. C/X의 순서·source·XYZ를 바꾸지 않는다.
전체 Final 상한 256, Structural Priority 상한 64, 이번 frame의 물체별 Priority 상한 32를 유지한다.
Final 목표는 Peg 129 / Box 127이다. 32점과 structural_ratio=.25는 현재 조건이지 최적값이 아니다.
물체 간 남는 Priority를 재분배하거나 Adaptive Budget/ratio 자동 변경을 하지 않는다.

| 물체 | X | C | Priority 상한 | Final 목표 |
| --- | ---: | ---: | ---: | ---: |
| Peg | 11,780 | 703 | 32 | 129 |
| Box | 11,665 | 599 | 32 | 127 |

위 count는 [Task03-F](03f_local_structural_sampling.md)와 [Task03-G](03g_geometry_aware_sampling.md)의 저장 결과다.
[Task03-H](03h_candidate_count_matched.md)는 같은 297개에서도 PCA5 expected rim coverage
79.898700/103 대 FD2 57/103을 보고했다. 후보 품질·분포의 해석 근거로만 사용한다.
Task03-H의 subset을 새로운 sampling 입력으로 쓰거나 다시 만들지 않는다.
Task03-F UV stratification과 Task03-G surface-patch는 기존 실패 비교군으로 보존한다.

| 저장 source/artifact | 재사용 내용 |
| --- | --- |
| [Task03 원본](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json), [samples](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz) | `evidence/pca5/{0,1}/20/orientation_flat`, S3 현재 조건의 선택·23개 feature field, 원본 λ=1 P/S/F |
| [Task01 dense geometry](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.npz) | 원본 camera XYZ, validity, mask, overlap, intrinsic/cam2world |
| [Task03-E NPZ](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.npz), [JSON](../../experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.json) | S3 Peg/Box C/X 및 저장 Global Camera XY λ=0 P/S/F; λ=1 저장 대조 |
| [Task03-F NPZ](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/local_structural_sampling.npz), [fixture manifest](../../experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/fixture_manifest.json) | 기존 UV 선택, 기존 7개 synthetic의 저장 입력·선택·평가 정의 |
| [Task03-G NPZ](../../experiments/495a67210c5c4cbfab91d013239f7ce9/geometry_aware_sampling.npz) | 기존 surface-patch 선택과 비교 결과; reference/optimized evidence 한계 |

이 계획에서는 source와 저장 보고서를 읽고, NPZ의 intrinsic 및 저장 field 이름·shape만 확인했다.
새 epsilon 수치, occupied count, 거리 분포, Voxel/FPS 선택은 계산하지 않았다.
저장 intrinsic은 `[[256,0,256],[0,256,256],[0,0,1]]`, XYZ shape은 `(512,512,3)`이다.
Simulator·renderer·CPU/GPU sampling 실험·CUDA/MPS/PCL은 이번 작업에서 실행하지 않았다.

## A. PCA5의 실제 지원 영역과 nominal metric epsilon

### A1. 코드에서 확인한 normal·pair 반응 범위

[sampling.extract_sampling_evidence](../../visual/runtime/sampling.py)는 S3에서
`local_plane_optimized(..., solver='analytic')`를 사용한다.
[pca.local_plane_optimized / covariance_patches](../../visual/runtime/pca.py)의 규칙은 다음과 같다.

1. Normal 중심의 `du,dv ∈ [-2,2]`인 5×5 patch만 참조한다. 완전 patch는 box moments,
   부분 patch는 patch 내부의 중심 연결 성분을 쓴다. 연결은 same-object observed support의
   4-neighbour이며 `|z2-z1| > .01 + .02*min(z1,z2)`이면 끊는다. Equality는 끊지 않는다.
2. X는 finite positive depth, 해당 mask, overlap 제외의 기존 계약이다. Partial patch를 허용하되
   중심 포함 최소 6점, covariance rank≥2, unique smallest eigenvector 조건을 요구한다.
   부족하거나 퇴화하면 normal-invalid이며 보간·padding·새 fallback normal을 만들지 않는다.
   Float64 covariance와 analytic solver의 residual/degeneracy `eigh` fallback,
   camera-facing normal 방향 및 float32 normal 저장을 그대로 해석한다.
3. [edge_evidence.pair_evidence / PairEvidence.candidates](../../visual/runtime/edge_evidence.py)는
   right/down 인접 pair를 각 한 번 기록하고, 양쪽 normal이 유효할 때 angle을 구한다.
   `angle >20°`인 pair의 **양 끝 supported pixel 모두** orientation 후보가 된다.
   Pair angle 비교 자체에 depth-jump veto를 새로 가정하지 않는다. Normal patch 내부 gate와
   pair의 orientation/depth evidence는 서로 다른 단계다.
4. 한 horizontal pair의 두 5×5 normal support 합집합은 최대 **6×5**, vertical은 **5×6**이다.
   한 endpoint의 네 이웃 비교를 모두 고려하면 offset은 최대 한 축 ±3, 다른 축 ±2까지 간다.
   Bounding box는 7×7이지만 네 `(±3,±3)` 모서리는 포함하지 않는다.
   이것은 가능한 입력 의존 영역이며 유효 support/gate 때문에 실제 사용점은 더 적을 수 있다.
5. [geometry.compute_geometric_features](../../visual/runtime/geometry.py)의 `normal_variation_rad`는
   유효한 인접 normal angle의 최대값이며 validity/count를 별도로 가진다.
   양 끝 후보 표시와 두 patch의 normal mixing 때문에 반응 띠가 넓어질 수 있다.
   5×5 normal window를 전체 detector 반응 범위나 물리적 edge 오차 폭으로 읽지 않는다.

계산용 crop의 halo 2(normal), halo 1(feature)은 ROI 선택이나 voxel origin이 아니다.
Mask/image 경계, 적은 support, 곡면, gate 미만 crease 혼합, flying pixel에서 높은 variation이
실제 모서리를 뜻하지 않을 수 있다. 지원 영역의 확인은 코드 독해이며 새 normal 실행 결과가 아니다.

### A2. 서로 다른 세 값

| 값 | 이번 정의 | 주장하지 않는 것 |
| --- | --- | --- |
| PCA5 spatial support | 법선 fitting이 참조할 수 있는 관측 pixel들의 metric 범위; 아래 식은 동일 depth 근사 | edge localization의 실측 오차나 신뢰구간 |
| Localization error | 물리적 edge/rim와 검출 후보 위치 차이; 이번 GT 평가에서도 관측·reference·matcher 영향을 포함 | 모든 후보가 참 edge에 대응한다는 가정, support 크기와의 동일성 |
| Voxel approximation error | C의 한 점을 같은 cell의 원본 대표점으로 대체할 때의 거리 | budget으로 cell을 버린 이후의 전역 최근접 거리 보장 |

### A3. 결과를 보기 전에 고정할 scale 정의

물체마다 **전체 원본 X의 camera Z median**을 `z_ref`로 사용한다. 후보 C의 median은 비교용으로만
보고한다. C의 밀도나 detector 반응에 따라 scale을 다시 정하지 않는다. X는 이미 기존 계약에 따라
finite positive이므로 추가 cutoff·clipping·trim·erosion을 하지 않는다. 짝수 개 median은 중앙 두 값의
평균, depth 요약 p95는 기존 NumPy linear percentile을 쓴다.

\[
r_5(z)=z\sqrt{(2/f_x)^2+(2/f_y)^2},\qquad
\varepsilon_{nominal}=r_5(z_{ref}),\qquad
a_{nominal}=\varepsilon_{nominal}/\sqrt{3}.
\]

Scale multiplier는 1로 사전 고정한다. 반 window 대각선인 `r5`를 nominal compression tolerance의
설계 proxy로 택한 것이며 detector 오차를 추정한 것이 아니다. Full-window 직경 `2*r5`와 구분한다.
실제 fx/fy는 저장 intrinsic에서 읽고 Task01/episode와 exact 대조한다. 숫자 epsilon/voxel size 산출은
승인 후에만 한다. 전체 pair 반응 영역 때문에 epsilon을 사후 확대하지 않는다.

추가 footprint는 설명용으로
`r_pair(z)=z*max(sqrt((3/fx)^2+(2/fy)^2), sqrt((2/fx)^2+(3/fy)^2))`를 함께 기록할 수 있다.
이는 endpoint 기준 입력 footprint의 동일 depth 범위이고 선택 scale로 사용하지 않는다.
경사진 표면은 window 안 Z가 달라지며, depth jump/occlusion에서는 고정 Z 평면 근사가 깨진다.
X 또는 C의 depth tail을 이 식 하나로 설명하거나 모든 localization error가 epsilon 이내라고 하지 않는다.

Voxel 내부에서 원본 두 점의 거리는 최대 `sqrt(3)*a`이므로 nominal cell 내부 후보→대표점에는
`d ≤ sqrt(3)*a_nominal = epsilon_nominal` bound가 있다. 대표점을 cell 밖으로 옮기지 않는 경우의
보장이다. Float64 연산 오차 allowance는 검사에만 적용하고 grid나 선택 tolerance로 쓰지 않는다.
Cell을 버린 뒤 최종 P까지의 거리와 same-pair `2px AND 3mm` GT 통과에는 이 보장이 없다.

X가 비면 scale은 N/A, C도 비어야 한다. Nonempty C/empty X, nonfinite intrinsic/XYZ,
양수가 아닌 scale은 입력 계약 실패로 중단한다. 임의 기본 voxel size로 대체하지 않는다.

### A4. Depth·오염·비교 공정성

아래 값은 [Task03-G 저장 depth 표](03g_geometry_aware_sampling.md)의 인용이며 새 진단 결과가 아니다.

| 물체/집합 | median m | p95 m | max m |
| --- | ---: | ---: | ---: |
| Peg X | .206 | .32705 | .365 |
| Peg C | .181 | .316 | .350 |
| Box X | .496 | .535 | 30.726 |
| Box C | .488 | .6665 | 1.426 |

Median은 한 극단값이 max/mean을 지배하는 영향을 줄이지만 mask의 대규모 오염에는 강건성을 보장하지 않는다.
30.726m observed point와 모든 기존 후보를 유지한다. 승인 후 X/C/R/P/S/F의 median/p95/max 및
candidate별 `r5(z)` 분포를 보고하되 선택에는 물체별 고정 `a_nominal` 하나만 쓴다.
Peg/Box는 같은 공식이지만 서로 다른 metric size가 된다. 이를 동일 절대 거리 해상도 비교라고 부르지 않는다.
물체 내부 가까운 면/먼 면에는 같은 metric grid가 적용되어 pixel당 해상도와 support 대비 비율이 달라진다.
물체별 raw mm와 `distance/epsilon_nominal`을 함께 보되 정규화 지표만으로 개선을 판정하지 않는다.

Mask 경계의 unstable normal/flying pixel도 occupied voxel과 FPS 예산을 소비할 수 있다.
같은 cell이 다른 물리 표면, 얇은 구조 양쪽, seam 양쪽 띠를 합칠 수 있다. 이번 voxel은 topology나
surface identity를 분리하지 않는다. GT/normal/depth gate를 추가해 cell을 분할하지 않는다.
Depth-adaptive variable-size voxel은 향후 별도 후보이며 이번 최소 구현에서 제외한다.

## B. 단일 GT-free 선택 규칙과 budget-resolution 진단

### B1. Grid·대표점·source 순서

물체별 C에 같은 규칙을 독립 적용한다. 좌표는 **원본 CV-camera XYZ(m)**, 원점은 camera `(0,0,0)`,
축은 원본 camera 축이다. World/task pose·물체 bbox 최소점·GT·rim/ROI로 grid를 이동/회전하지 않는다.
Camera grid이므로 viewpoint 변화에 대한 불변성은 보장하지 않는다.

1. C의 원본 flat source `v*W+u`가 unique/ascending이고 C⊆X이며 XYZ가 exact gather인지 검사한다.
   내부 계산만 float64로 승격하고, 출력 XYZ는 원본 dtype/value를 그대로 gather한다.
2. `k = floor(float64(XYZ)/float64(a_nominal))`를 int64 cell key로 쓴다.
   각 cell은 축별 `[k*a,(k+1)*a)`다. 음수는 truncation이 아닌 floor이며, 경계점은 저장 좌표의
   float64 quotient에 대한 floor로 결정한다. `round`, bbox recenter, 경계 jitter/epsilon 이동을 쓰지 않는다.
   Nonfinite quotient/int64 overflow는 중단한다. Key tuple은 lexicographic 순서로 기록한다.
3. Cell 내 C를 source 오름차순으로 두고 float64 산술 centroid를 계산한다.
   Centroid에 대한 squared 3D distance가 최소인 **원본 후보 하나**를 대표점으로 고른다.
   Exact distance tie는 source index가 작은 후보다. `isclose` 기반 tie band는 두지 않는다.
   Centroid는 선택용 계산값일 뿐 node/feature 좌표로 출력하지 않는다.
4. 전체 대표점 R을 대표 source 오름차순으로 정렬한다. 각 C의 cell key, 해당 representative source,
   cell capacity, 대표 source/원본 XYZ/pixel/provenance와 candidate→동일-cell 대표점 거리를 저장한다.
   서로 다른 source의 동일 XYZ는 허용하되 같은 source를 중복 선택하지 않는다.

**Score-free만 사용한다.** 현재 `sampling.extract_candidates`에는 normal variation과
`orientation_score_deg` 생성 경로가 있지만, Task03 NPZ의 현재 S3 조건에는 최종 256행에만
그 field가 저장돼 있다. C 전체 703/599행에 대응하는 dense optimized score 또는 전체 optimized
pair-angle 배열은 해당 artifact에 없다. 선택된 일부 행의 score가 있다는 사실은 모든 C에 대한
score-max 선택 가능성을 뜻하지 않는다. 다른 조건/estimator의 feature를 섞어 누락 score를 채우지 않는다.

Task02 reference PCA5 normal/pair와 optimized PCA5는 후보 endpoint membership 일치와
연속 angle의 bit-exact 일치가 다르다([Task03-G evidence 한계](03g_geometry_aware_sampling.md)).
Reference angle로 optimized score를 대체하거나 새 score를 추정하지 않는다.
향후 score-max는 완전한 optimized C-source 대응과 저장 provenance를 확보한 뒤 별도 단일 변경으로
제안한다. 이번 voxel 변경과 동시에 넣지 않는다.

### B2. 초과 예산 처리: 방향 1만 추천

| 방향 | 장점 | 한계 | 이번 결정 |
| --- | --- | --- | --- |
| 1. Nominal R 후 기존 선택 함수로 cap | PCA5에서 정한 절대 거리 scale 유지, 기존 FPS 재사용, `P⊆R`로 압축/추가 손실 분리 가능 | FPS의 전역 거리 경쟁과 원거리 noise 민감도 재도입; 버린 cell에는 epsilon 보장 없음 | **채택할 사전 고정 규칙** |
| 2. 예산을 만족할 때까지 voxel 확대 후 대표점 | 모든 최종 occupied cell을 표현하고 원본 후보당 같은-cell 대표점 존재 | nominal 해상도 희생, 정확히 32 cell 불보장, Priority underfill 및 origin/boundary 영향 | 선택 arm에서 제외; 필요한 해상도 완화량은 count-only 진단 |

방향 1에서 cap 함수는 **기존 `sampling.fps_indices`의 원본 Euclidean 3D**로 고정한다.
λ=0/다른 metric을 결과에 따라 선택하지 않는다. 이번 가설은 voxel으로 공간 중복을 먼저 줄이면
남은 전역 FPS 경쟁의 결과가 달라지는지다. FPS 경쟁이 사라졌다고 주장하지 않는다.

초기화는 전체 C의 기존 object stream `fps_seed(0, oid)`가 고른 anchor의 **cell 대표점**이다.
Frame0 저장 anchor Peg 214225 / Box 96553과 원본 seeded anchor 대응을 확인한다.
원래 anchor가 centroid-nearest가 아니면 source 자체는 유지하지 않고 같은 cell의 대표점을 사용한다.
Anchor를 강제로 두 번째 대표점으로 넣지 않는다. 대표점만 허용한다는 규칙을 우선한다.
R source 순서와 이 mapped anchor를 FPS `initial`로 전달하며, tie는 기존 first input row 규칙이다.
동일 source anchor 비교가 아니라 같은 원래 anchor cell에서 출발하는 규칙임을 보고한다.

| 상황 | Priority P | Surface / Final |
| --- | --- | --- |
| `len(R)>32` | mapped anchor로 시작하는 기존 3D FPS로 R에서 32개 | P를 initial로 전체 X 3D FPS |
| `len(R)=32` | R 전부; mapped anchor 먼저, 나머지 source ascending | 동일 |
| `0<len(R)<32` | R 전부; 같은 순서, 추가 C로 refill하지 않음 | 미사용 Priority slot은 같은 물체 Surface로 전환 |
| `0<len(C)<32` | cell 압축 후 R 전부; 실제 P는 C보다 작을 수도 있음 | 부족분 Surface; padding/복제 없음 |
| `len(C)=0` | P=빈 배열, mapped anchor=N/A | X에서 기존 seeded Surface FPS만 수행 |
| X도 없거나 Final 목표보다 적음 | 위 availability 계약 유지 | `min(Final 목표,len(X))`, shortfall 명시; 타 물체 증액 없음 |

P를 X row로 exact 매핑한 뒤 `fps_indices(X_xyz, final_target, seed=object_seed, initial=P_rows)`를 쓴다.
F는 P-prefix와 나머지 S로 나누며 S는 C/R에 속한 점도 선택할 수 있다. 선택 경로와 후보 flag를 구분한다.
Actual P≤32/물체, 전체≤64; actual F≤129/127 및 합≤256이다.
Frame0 X가 충분하면 F=129/127을 유지한다. P 부족 시 Surface 수는 `F-len(P)`로 늘어난다.
이를 adaptive ratio 채택으로 부르지 않으며 cap과 actual count를 함께 기록한다.

### B3. 해상도와 budget의 충돌을 숨기지 않는 진단

선택에서 사용한 크기는 항상 `a_used=a_nominal`, `a_used/a_nominal=1`이다.
`N_occupied(a_nominal)>32`이면 **현재 nominal 해상도에서 모든 cell을 표현하기에 Priority가 부족**하다고
명시한다. FPS로 32점이 나왔다는 이유로 resolution-budget compatible이라고 판정하지 않는다.

별도로, cell을 버리지 않고 budget을 맞추려면 얼마나 coarse해져야 하는지 다음 count-only rule을 고정한다.
`a_l=2^l*a_nominal`, `k_0`는 B1의 nominal key, `k_{l+1}=floor_divide(k_l,2)`로 parent를 만든다.
`l=0,1,...` 순서로 모든 occupied count를 기록하고 **최초 `N_l≤32`**에서 멈춘다.
이 grid family는 원점이 고정된 nested dyadic partition이므로 count가 nonincreasing이다.
음수 key에도 floor division을 써야 한다. 임의 연속 크기 grid의 단조성을 가정한 이분 탐색은 하지 않는다.
일반 size 변화의 boundary crossing과 nested parent rule을 구분한다.

`a_budget_dyadic=a_l`, `a_budget_dyadic/a_nominal`, `N_l`, `sqrt(3)*a_l`을 저장하되
**선택에 적용하지 않는다.** 이는 검사한 dyadic family에서 필요한 첫 크기이며 전역 최소 크기나
정확히 32 cell을 만드는 해답이 아니다. `N_l<32`도 정상이다. Nonempty finite int64 key는 반복 halving으로
축별 -1/0에 도달하여 최대 8 cell이 되므로 이번 K=32 진단은 종료한다. Arithmetic/종료 계약 위반은 실패다.
K=0이면 nonempty C의 all-cell 표현은 불가능으로 명시하며 이 반복을 사용하지 않는다.

결과 표에는 `C 수 / nominal occupied / cap32 / selected occupied / discarded occupied /
a_nominal / a_used / a_budget_dyadic / 두 size ratio / nominal 및 coarse bound`를 각각 둔다.
Coarse size의 bound는 그 크기로 같은-cell representative를 택할 경우의 가상 bound다.
현재 nominal+FPS P에 적용되는 보장이 아니다. Coarse grid의 대표점/Final 선택 arm은 만들지 않는다.
이 frame의 occupied count를 일반적인 최적 node budget이라고 선언하지 않는다.

## C. 최소 구현 파일·helper 재사용·계약 테스트 계획

승인 후 신규 구현은 다음 두 evaluation-only 파일로 제한한다.

- `visual/runtime/test/run_voxel_structural_sampling.py`: 순수 grid/대표점/선택 함수,
  count-only resolution 진단, 저장 입력 gate, 평가·산출물 생성 orchestration.
- `visual/runtime/test/test_voxel_structural_sampling.py`: 아래 계약 검증.

Production `sampling.py`, normal/candidate 추출, shared simulator, Vision→Graph 책임과 point-node interface는
변경하지 않는다. 반환값은 평가용 source 배열/metadata이며 새로운 Graph 포장·edge·feature API가 아니다.

| 기존 위치/helper | 재사용과 주의점 |
| --- | --- |
| [run_local_structural_sampling.load_actual](../../visual/runtime/test/run_local_structural_sampling.py) | 저장 C/X, Task01 XYZ, intrinsic/mask/input hash, 원본 feature 및 E λ=0 로딩. 진입점 `run()`은 호출하지 않음 |
| [sampling.fps_indices](../../visual/runtime/sampling.py), [run_priority_fps_trace.fps_seed](../../visual/runtime/test/run_priority_fps_trace.py) | R cap과 전체 X Surface, 원래 object RNG stream. 새 FPS 구현 없음 |
| [run_rim_bottleneck](../../visual/runtime/test/run_rim_bottleneck.py)의 `source_contract`, `validate_saved_selection`, `require_equal`, `occupancy` | 원본 source·budget gate 및 기존 ROI/depth band 정의 |
| [run_local_structural_sampling](../../visual/runtime/test/run_local_structural_sampling.py)의 `summary`, `nearest`, `spatial_metrics`, `evaluate_synthetic` | 동일 UV/XYZ 평가 구역, raw/normalized 거리, 기존 synthetic 평가. 비어 있는 입력/결과는 명시적 N/A wrapper 필요 |
| [run_priority_fps_metric_control](../../visual/runtime/test/run_priority_fps_metric_control.py)의 `feature_rows`, `evaluation_inputs`, `peg_labels`, `rim_labels`, `segment_spacing` | 같은 source exact feature gather와 GT 보조 평가. GT 관련 함수는 pre-GT 선택 저장 후에만 호출 |
| [sampling_evaluation.CoverageMatcher / staged_reference_coverage](../../visual/runtime/test/sampling_evaluation.py) | 기존 same-pair joint matcher, unique reference와 matching node의 분리 |
| [utils.experiment_logger.ExperimentLogger](../../utils/experiment_logger.py) | 승인 후 evaluate run만 새 UUID, `team='visual'`, seed/config 기록, 기존 project/group 규칙 유지 |

현재 `prior.score_case`는 32점을 고정해 P/S slice와 feature role을 만든다.
이를 P<32 경로에 그대로 적용하지 않는다. 먼저 기존 shared evaluation helper의 가변-P 확장 가능성을
검토하고, 이번 최소 범위에서는 기존 하위 matcher/occupancy/feature helper를 조합하는 얇은 평가 adapter에서
`n_priority=len(P)`를 사용한다. GT matcher나 feature gather를 병렬 재구현하지 않는다.
공통 helper 수정이 꼭 필요하면 범위와 문서/계약 변경을 별도 보고한다. 기존 dirty helper를 덮어쓰지 않는다.
`load_actual()`의 고정 current-case gate와 빈 입력 unit contract는 별개로 검증한다.

Feature availability는 현재 S3 조건의 `saved_flat`에 있는 행만 `feature_rows`로 exact gather한다.
R/P/F의 나머지 source는 unavailable mask와 compact positions로 남긴다. 가짜 NaN feature 행,
다른 조건 feature 혼합, normal/score 재계산을 하지 않는다. 원본 pixel, XYZ, source, object ID는
항상 exact 대응한다. Geometry-level 평가이며 완전한 production feature tensor 검증으로 주장하지 않는다.

승인 후 필요한 계약 테스트는 다음과 같다. 이번 계획 단계에서는 테스트를 작성·실행하지 않는다.

| 계약 | 확인할 실패/경계 |
| --- | --- |
| Grid | 양/음수·0·정확 경계·`nextafter` 인접값, floor와 truncation 차이, 고정 origin, int64 overflow/nonfinite 거부 |
| 대표점 | singleton·동일 XYZ·centroid 거리 tie의 최소 source, deterministic 순서, 원본 후보만 출력, 입력 불변 |
| 예산 | R>32/=32/<32, C<32/0, 빈/작은 X, no duplication/padding, mapped-anchor cell 및 P-prefix, Final/합계 cap |
| Scale | fx/fy·X median만 사용, C/GT/ROI/score 변경이 scale 입력에 침투하지 않음, 빈 X=N/A, multiplier 고정 |
| Resolution 진단 | dyadic parent nesting·음수 floor·nonincrease·종료, 최초 ≤32, 정확히 32 미존재 허용, arbitrary-size 단조 가정 없음 |
| 거리 분해 | same-cell bound, `P⊆R`, NN 추가 손실 nonnegative, discarded cell 수/표본 대응, 빈 P의 distance N/A |
| Source/features | C⊆X, 원본 XYZ/pixel/source exact, available-only field exact gather, 누락 feature를 만들지 않음 |
| GT 독립·순서 | 선택 함수가 GT/pose/shape/ROI/annotation/score를 받지 않음; GT evaluator를 차단해도 선택 저장 완료; 평가 후 선택 hash 불변 |
| 재사용/보존 | 저장 λ=1/0 source/order/XYZ/역할/예산 exact gate, 기존 2px AND 3mm 같은 pair 및 dedup, 입력/source/artifact/sidecar hash 불변 |

수학적 경계용 작은 배열은 unit test 입력이지 새 synthetic benchmark fixture가 아니다.
실제 geometry 성능 향상이나 rim coverage 증가는 계약 테스트의 PASS 조건으로 삼지 않는다.
승인 후 root `.venv/bin/python -m unittest`로 신규 테스트와 관련 기존 sampling/geometry/evaluation,
metric-control/local helper 계약을 실행한다. 공통 logger/artifact 구현을 바꾸지 않으면 해당 infrastructure
suite를 불필요하게 확장 실행하지 않는다. 실제 실행 목록·PASS/FAIL/SKIP는 결과에 남긴다.

## D. 승인 후 비교·평가·검증·보고 순서

### D1. Selection-before-GT와 불변성 gate

1. Git 시작 status/diff/HEAD, 기존 dirty 파일, 사용 source/helper snapshot, 입력·기존 artifact·sidecar의
   SHA-256 목록을 보존한다. 출력 경로가 이미 있으면 overwrite를 거부한다.
2. 기존 [tracking](../experiment_tracking.md) 규칙으로 `WANDB_MODE=disabled`의 새 evaluate UUID를 만들고
   규칙·median 집계·grid·tie·FPS metric/anchor·예산·평가 구역·fixture 이름을 config에 사전 고정한다.
   UUID/project/group를 별도 생성·주입하지 않는다. 이번 계획에서 UUID를 미리 만들지 않는다.
3. 기존 input/mask/camera/XYZ/C/X count/order 및 baseline source/역할/feature gate를 통과해야 한다.
   Baseline λ=1은 Task03 원본, λ=0은 E 저장 선택을 그대로 읽고 **새 baseline FPS를 실행하지 않는다.**
   불일치가 있으면 입력/source를 조정하거나 재추출하지 않고 중단한다.
4. Observation-only scale·grid·R·budget 진단 및 단일 P/F 선택을 수행한다. 원본 C/X, R,
   membership/representative, 실제 P/S/F source/XYZ, nominal/used size, anchor mapping 및 config hash를
   `selection_before_gt.npz`와 GT-free JSON에 저장하고 hash/readback을 완료한다.
5. 그 후에만 GT seam/rim·pose/segment·known-negative ROI annotation을 평가에 연결한다.
   보관 NPZ에 GT field가 있어도 selection adapter에는 전달하지 않는다. GT 없이 pre-GT 저장까지
   완료되는 계약을 둔다. 평가 후 선택 hash가 바뀌면 실패다.

### D2. Primary: 일반 geometry preservation와 예산 추가 손실

후보 c의 같은-cell 대표점을 `r(c)`, 전체 nominal 대표점 집합을 R, 최종 Priority를 P라고 두고 구분한다.

- `d_cell(c)=||c-r(c)||`: **자기 cell 압축 오차**, epsilon bound 대상.
- `d_R(c)=min_{r∈R}||c-r||`: 전체 nominal 대표점까지 최근접 거리. 이웃 cell의 대표가 더 가까울 수 있다.
- `d_P(c)=min_{p∈P}||c-p||`: 실제 budget 제한 뒤 거리, 기존 C→P와 같은 정의.
- `delta_budget(c)=d_P(c)-d_R(c)≥0`: **대표점 집합 R에서 cap 때문에 추가된 최근접 손실**.
  `d_cell`과 직접 뺀 차이를 비음수 추가 손실이라고 부르지 않는다.

위 네 배열을 source와 함께 NPZ에 남기고 mean/p50/p95/max, ECDF와 worst-source 위치를 보고한다.
R=P이면 NN 추가 손실은 0이다. `d_cell`과 `d_R`는 다른 기준이므로 서로 대체하지 않는다.
Budget 후에는 `d_P>epsilon_nominal` 후보 수/fraction과 버려진 nominal cell별 거리도 공개한다.
최근접 representative/source distance tie는 source 최소값으로 고정한다.
빈 query는 count0/N/A, nonempty C/빈 P는 거리 undefined 및 missing count를 함께 보고하고
0거리로 변환하지 않는다. JSON은 null/status, NPZ는 unavailable NaN과 mask를 쓴다.

비교는 Peg/Box별 S3 **저장 Global λ=1**, **저장 Global Camera XY λ=0**, 신규 단일 voxel 방식이다.
F/G는 저장 실패 참고군으로만 병기한다. R 전체는 cap 없는 **해상도 진단 집합**이며 32점 sampler와
동등 budget 성능 arm이 아니다. Nominal/used/coarse 진단 scale과 occupied/selected/discarded 수를 병기한다.
R≥32면 신규도 P=32라 exact Priority budget 비교다. R<32면 같은 상한·Final budget 비교임을 명시하고
“동일 P=32”라고 쓰지 않는다. 32점으로 맞추기 위한 후보 refill이나 추가 arm을 만들지 않는다.

기존 독립 UV 4×4, XYZ 최대 2×2×2 구역은 동일 X bbox 정의를 재사용한다.
구역별 C/P count·선택률·C→P p95/max, P=0인 candidate 구역과 동등 구역 평균을 기록한다.
선택 voxel 자체의 all-cell 점유를 독립 local coverage 지표로 대신 쓰지 않는다.
Box outlier 때문에 기존 XYZ coarse 구역에서 C가 한 구역에 몰리는 한계와 bbox 정규화 왜곡을 밝히고,
raw mm·UV·nominal cell 누락/거리 지도도 같이 본다. 이것은 새 GT ROI/quota가 아니다.

Surface 품질은 동일 전체 X→F mean/p95(보조 max)로 비교한다. P가 Surface 초기 집합을 바꾸므로
동일 Surface FPS 함수라도 결과가 달라질 수 있다. X max 30.726m를 포함한 원본으로 계산한다.
X median/p95로 정한 기존 depth bands와 known-negative ROI union의 P/S/F count/fraction,
candidate band 대비 P 선택률을 기록한다. 실제 data의 far/ROI는 noise 정답이 아니므로
“원거리/noise 의심 proxy에 소비된 예산”으로 해석한다. 확정 noise count는 synthetic annotation에서만 쓴다.

### D3. Secondary: GT 구조 평가와 scale 해석

기존 **동일 node/reference pair의 ≤2px AND ≤3mm** matcher를 그대로 사용한다.
Peg seam, Box rim의 C/R/P/S/F matching node, unique reference와 각 분모를 분리한다.
GT-visible/관측 가능 reference 수와 ID를 저장 baseline에 대조한다(Peg observed 462, Box 103).
Box 네 rim segment별 count/coverage, Peg 기존 face-pair/ambiguous 및 기존 ROI union을 보고한다.
`Surface 추가 reference = refs(F) \ refs(P)`와 overlap을 기록하며 S의 전체 hit 수와 혼동하지 않는다.
Box verified line의 gap/누락 run은 기존 helper 정의를 쓰고 Peg의 미검증 물리적 간격은 N/A다.

GT rim에 대한 C→reference 3D 거리 분포는 **선택 저장 후의 해석 자료**다. Nominal support scale과
대조하되 이를 다음 epsilon/대표점/scale 선택으로 피드백하지 않는다. 후보→GT reference 거리는
detector localization뿐 아니라 depth noise, reference 공간 이산화, 가시성·tracker support와 matcher 정의의
영향을 받는다. 한 후보와 하나의 물리 edge의 정확한 대응 오차를 분리 측정했다고 주장하지 않는다.
동일 Box Frame0 GT로 calibration한 뒤 같은 frame의 개선을 주장하지 않는다.
향후 GT 기반 경험적 calibration은 episode 또는 shape가 독립인 calibration/evaluation 분리를 사전 제출한다.

### D4. 기존 7개 synthetic만 사용하는 검증

Task03-F의 `line_uniform`, `line_dense`, `separated_clean`, `separated_noise`, `camera_near`,
`camera_far`, `camera_occluded`를 우선 확인한다. 기존 저장 NPZ의 flat/XYZ/shape와 fixture manifest를
읽어 exact 대응을 검증하며 입력이 누락되면 중단/해당 fixture N/A를 보고한다.
새 fixture를 만들거나 기존 projection/collision/occlusion 규칙을 바꾸지 않는다.

이 voxel 규칙은 normal/pair/patch evidence 없이 **C의 XYZ/source**만으로 대표점·budget 선택을 검증할 수 있다.
Synthetic scale도 같은 공식과 해당 fixture의 관측 C median Z 및 기존 focal=1600px를 사용한다
(dense X가 없으므로 C가 관측 support인 이 stress test의 명시적 예외).
Annotation의 segment ID/길이/reference/noise/가림 label은 선택 후 평가에만 전달한다.
저장 Global λ=0 및 UV 결과를 재사용하고 voxel P만 새로 비교한다. Synthetic Surface는 N/A다.
Reference→P 거리, 구조별 선택 수·거리, 확정 noise 예산, near/far/밀도 변화·부분 가림 보존을 평가한다.

Synthetic에는 실제 PCA5 5×5 normal estimator의 localization evidence가 없다.
따라서 **PCA5 uncertainty/localization 기반 calibration의 검증은 N/A**다.
공식 적용과 voxel bound/거리·예산 stress test가 통과해도 그 calibration이 타당하다는 증거로 쓰지 않는다.

### D5. 실패·중단 및 해석 기준

- Input/source/baseline/feature 계약, deterministic grid/대표점, budget/중복 또는 pre-GT 순서가 깨지면
  구현 검증 실패로 중단한다. 누락 score/normal/후보를 재생성해 통과시키지 않는다.
- `N_occupied(a_nominal)>32`는 해상도와 현재 Priority budget의 충돌이다. 정상적으로 기록할 연구 결과이며
  count를 줄이기 위한 사후 voxel 튜닝/ratio 변경의 근거로 즉시 사용하지 않는다.
- 두 baseline 각각에 대한 Δ를 모두 보고한다. C→P p95/max나 독립 local 누락, X→F mean/p95,
  GT reference가 엇갈리면 혼합 결과다. Rim 하나의 증가로 다른 geometry/Surface 손실을 숨기지 않는다.
- 특히 한 물체라도 geometry tail/local 누락이 악화하거나 Surface trade-off가 있으면 전반적 우월성을
  주장하지 않는다. 계약 PASS와 Shape-First 가설 지지는 별개다. 동일 frame의 작은 개선도 일반화/채택 증거가 아니다.
- Noise/far cell이 예산을 차지하거나 다른 표면이 합쳐지는 실패를 그대로 보고한다.
  Depth filtering, erosion, scale/threshold sweep, score-max, GT quota로 고치지 않는다.

Timing은 scale/grid/grouping/대표점/cap-FPS/Surface-FPS/evaluation/I/O를 분리 기록할 수 있게 설계한다.
Offline CPU wall time과 환경/version을 남기며 Graph/GNN 전체 20Hz 성능을 주장하지 않는다.
Timing을 위해 추가 frame/simulator/GPU/PCL 실험을 하지 않는다.

### D6. 승인 후 산출물과 링크 계약

결과 Markdown 예정 경로는 `docs/visual-graph-integration/03i_voxel_structural_sampling.md`다.
HTML은 새 Logger가 정한 `experiments/<run_id>/voxel_structural_sampling.html`에 둔다.
정적 HTML에는 다음 표/그림을 포함한다.

- 원본 C와 voxel representative R, 실제 P/S/F의 source별 overlay와 camera XYZ view.
- Nominal/used metric size, epsilon/bound, count-only coarse 진단 size/ratio,
  occupied 수와 cap32·selected/discarded 수의 비교.
- Same-cell 압축 오차, C→R/C→P, budget 추가 손실, 독립 local coverage 및 worst 위치.
- 저장 Global λ=1/0의 Peg/Box 비교, R 전체와 fixed-budget arm의 차이, P underfill 여부.
- GT seam/rim과 Box segment 보조 평가, 기존 ROI/depth 및 X→F Surface trade-off.
- 7개 synthetic 결과와 calibration N/A, feature availability, 검증·보존·미실행 범위.

동일 run directory의 `voxel_structural_sampling.json`, `voxel_structural_sampling.npz`,
`selection_before_gt.npz`, PNG 및 verification/hash/config/source 자료를 상대 링크한다.
결과 Markdown→HTML은 `../../experiments/<run_id>/voxel_structural_sampling.html`,
HTML→Markdown은 `../../docs/visual-graph-integration/03i_voxel_structural_sampling.md`로 생성한다.
HTML 내부 JSON/NPZ/PNG는 같은 directory의 filename 상대 링크를 사용한다.
지금은 placeholder 링크 파일·HTML·결과 Markdown을 만들지 않는다.

승인 후 저장 결과 readback, Markdown↔HTML/asset 상대 링크, PNG decode와 정적 HTML 이미지 로딩을 검증한다.
새 파일만 Logger `link_output()`으로 저장 직후 등록한다(HTML/Markdown/PNG는 reports,
JSON은 metrics, NPZ는 기존 허용 artifact kind에 맞춤). 실행 중에 종료 후 전용
`register_existing_output()`을 호출하지 않는다. 기존 baseline artifact에 새 sidecar를 쓰지 않는다.
`utils.experiment_artifacts`의 로컬 inventory와 [기존 publisher](../huggingface_artifacts.md) 계약을 재사용하되
명시적 게시 요청이 없으므로 publisher/외부 업로드는 실행하지 않는다.

완료 시 source/input/기존 artifact·sidecar·dirty 파일 hash와 Git 상태 보존을 확인하고,
테스트·선택·평가·실패/N/A·실행 host/device 범위와 남은 제약을 보고한다.
새 detector/normal/score, threshold sweep, GT calibration/quota, 사후 scale tuning, Adaptive Budget,
depth 전처리, Graph edge/학습, production sampler 변경, 추가 episode/simulator/GPU/PCL 실행,
dependency 설치·외부 업로드·Git commit은 범위 밖이다.

**이 계획 제출 후 중단한다. 후속 runner·테스트·diagnostic·선택·평가·결과 생성은 구현 승인 전까지 실행하지 않는다.**
