# Task 03 — Structural / Surface Sampling 및 Geometry 보존 검증

**Task03 CPU 구현·평가를 완료하고 이 단계에서 중단한다.** 256점 / ratio .25 / 20° / seed 0에서
S3 PCA5는 관측된 Peg seam의 **61.28%**를 보존했고 General FPS는 **27.54%**였다.
Box rim은 FD2가 **15.24%**, PCA5는 **11.43%**, General FPS는 **9.23%**로 여전히 낮다.
S4 depth-only 우선 선택은 이 조건에서 Peg 보존을 낮추고 가림·오염 선택을 늘렸다.
이 수치는 **Tracker가 현재 관측한 reference 대비 Sampling 보존율**이다.
최종 detector, threshold, ratio 또는 feature 표현을 채택한 결과는 아니다.

후속 [Task03-D Box rim 병목 분석 Phase 1](03d_box_rim_bottleneck_phase1.md)은 저장된 frame 0/120의
관측·후보·priority·Surface coverage를 분리한다. Frame 0의 S3는 후보 coverage가 더 높지만
priority와 최종 coverage는 더 낮았다. 오염·원거리 depth·FPS 거리 경쟁의 인과 기여율은 미확인이다.

최종 run: `2551448b0fae4c6bbce3caf48774275f`.
[전체 HTML](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.html), [전체 summary·설정·fidelity JSON](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json),
[검증 기록](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/verification.json), [budget/ratio/false selection/latency 그림](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/tradeoffs.png),
[threshold·seed 그림](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/sensitivity.png), [구현 diff](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/implementation.patch).

## A. 구현 내용과 범위

| 파일 | 내용 |
| --- | --- |
| [sampling.py](../../visual/runtime/sampling.py) | Config, 물체별 allocation, index FPS, priority→Surface FPS, metadata alignment |
| [normal_estimators.py](../../visual/runtime/normal_estimators.py) | Task02-B FD/PCA 원본 구현을 runtime으로 이동 |
| [pca.py](../../visual/runtime/pca.py) | 검증된 CPU optimized PCA5, float64 analytic solver와 fallback |
| [edge_evidence.py](../../visual/runtime/edge_evidence.py) | Task02-C pair evidence를 runtime으로 이동 |
| [run_sampling.py](../../visual/runtime/test/run_sampling.py) | 전체 source/조건 평가, 기존 Logger, 입력/source hash와 실행 source snapshot |
| [sampling_evaluation.py](../../visual/runtime/test/sampling_evaluation.py) | GT 가시성, joint tolerance, 관측/샘플링 분모, 표면·negative 평가 |
| [sampling_plots.py](../../visual/runtime/test/sampling_plots.py) | 동일 crop overlay, 동일 축 3D, 정량 그래프 |
| [test_sampling.py](../../visual/runtime/test/test_sampling.py), [test_sampling_evaluation.py](../../visual/runtime/test/test_sampling_evaluation.py), [test_sampling_geometry.py](../../visual/runtime/test/test_sampling_geometry.py) | 합성 계약·평가·cuboid/공간분산 테스트 23개 |
| [sampling API 문서](../../visual/runtime/sampling.md), runtime README, 이 보고서 | API와 실제 실행 결과 |

기존 `test/normal_estimators.py`, `test/pca_optimized.py`는 호환 import로 남겼고,
`test/structural_edges.py`의 비교 entrypoint는 기존대로 유지했다. 이동한 함수·class
**15개 AST가 원본과 동일**함을 확인했다. Runtime은 test 모듈에 의존하지 않는다.
Task01 `geometry.py`, RGB-D adapter, tracker, simulator, E5/E9, Graph/Recurrent와
기존 결과는 수정하지 않았다.

`extract_sampling_evidence(frame, estimator)` → `sample_points(evidence, config, method)`를
명시적으로 호출한다. 결과는 CV-camera metre 좌표, `(v,u)`/flat source pixel,
object ID, selection-path flags, 모든 원래 geometric field와 독립 candidate evidence다.
최종 EE 좌표 Tensor packaging, feature 차원/onehot/probability/continuous, node type,
physical point ID, cache, intra-object edge, Graph adapter는 만들지 않았다.

**Allocation:** 실험 minimum은 Peg/Box 각각 16점이다. 실제 관측량으로 제한한 minimum을
먼저 예약하고 잔여 budget을 관측점 수 비례로 capped largest remainder 배분한다.
`sqrt_observed_points`도 config로 선택할 수 있다. 정수 동률만 ID 순으로 1점 이내
차이가 날 수 있다. Minimum 합이 전체보다 크면 명시적으로 거부한다.
두 물체 합계 상한은 `num_points`이며 부족·빈 관측에서 가상 point나 padding을 만들지 않는다.

**Structural Ratio:** 전체 `floor(num_points*ratio)`가 priority 경로의 상한이다.
후보 부족분은 일반 FPS로 채운다. S4는 이 상한의 최대 25%만 depth-only에 따로 예약한다.
256/.25에서는 depth-only 상한 16점이다. Orientation와 depth-only는 독립 집합으로 FPS하고,
Surface FPS는 이미 선택한 점까지의 최소 거리를 이어 계산한다. Depth/mask/invalid 원인을
semantic structural 정답으로 바꾸지 않는다. API 기본 S3는 orientation만 우선 선택한다.

**FPS:** 기존 distance-update loop를 index 반환·중복 방지·초기 선택 지원으로 재사용했다.
Random pre-reduction은 없고 squared distance는 float64다. 물체별
`SeedSequence([seed, object_id])`에서 얻은 seed로 해당 raster 순서 집합의 첫 점을
uniform 선택한다. Argmax 동률은 첫 source row다. 물체 누락은 다른 물체 RNG를 바꾸지 않는다.
Seed 0/1/2를 비교했으며, 같은 seed의 최종 54조건 전체 평가를 두 번 실행해 시간 외의
summary가 모두 일치했다. Legacy float32 FPS와 bitwise 동등하다고 주장하지 않는다.

## B. Sampling Contract 검증

- 54조건 × 121프레임 = **6,534개 selection**을 저장 후 독립적으로 재검사했다.
- 전체/물체별 budget·minimum, priority/depth cap, 실제 finite positive 관측점 포함,
  pixel 및 XYZ 중복 0, `(v,u)` 재대조, 모든 원래 feature 및 candidate evidence row alignment를 확인했다.
- 121프레임에서 S1–S4의 ratio 0 결과가 S0와 정확히 일치했다.
- 빈 mask/두 물체 부재/작은 물체/부족한 관측/후보 없음/후보 과잉/잘못된 config를 합성 검사했다.
- 실제 Box 빈 mask는 18–26 및 56–65, 총 **19프레임**이다. 모든 조건에서 현재 Box 결과 0개,
  남은 Peg에 전체 budget을 배분했다. Stale geometry는 사용하지 않았다.
- 합성 cuboid의 가시 face seam 보존, 후보가 한 영역에 몰린 경우 FPS 분산,
  평면의 가상 구조점 부재, mask/invalid hole/depth step 분리, 보이는 hole 입구만 평가를 검증했다.
- 최적화 PCA5는 실제 121프레임의 모든 원래 feature 및 10/20/30/40° 후보 위치를 원본과 비교했다.
  Validity/비부동 field/candidate 위치 차이 **0**, 최대 normal 차이 **3.422×10⁻⁶°**,
  variation 차이 **3.415×10⁻⁶°**였다. 기존 extractor 동작과 이전 결과 파일도 보존했다.

## C. Geometry Preservation

### 입력·평가 분모

같은 `visual/data/step2/ep000.npz`, `visual/results/runtime_tracking/ep000/masks.npz`,
`base_camera` 512×512 source **0–120 전체 121개**다. 현재 `visual/data`에 평가 가능한
독립 RGB-D episode는 없었다. 저장 simulator 관측 한 episode의 진단이며 물리 센서나
독립 episode 검증은 아니다. 모든 방법은 같은 source/support/object allocation/FPS 규칙/GT/tolerance를 사용한다.

- Episode SHA-256: `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`
- Masks SHA-256: `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`

Task02-C의 가시 signed-face seam endpoint와 가시 물리 rim line reference를 그대로 재사용했다.
가려진 전체 형상/입구 투영을 positive GT로 채우지 않았다. GT segmentation/pose/치수는
검출 및 sampling이 끝난 뒤 평가에만 사용했다. 후보를 GT corridor로 제한하지 않았다.

| 구조 | GT 가시 reference | Tracker가 joint tolerance 내에서 관측 | 관측률 |
| --- | ---: | ---: | ---: |
| Peg seam | 32,664 | 32,460 | 99.38% |
| Box 물리 rim line | 1,778 | 1,365 | 76.77% |

기본 matching은 **같은 point/reference pair가 ≤2px AND ≤3mm**를 동시에 만족해야 한다.
GT coverage=`sampled/GT visible`, Sampling retention=`sampled/tracker observed`다.
Tracker 관측 분모는 raw rim-band pixel 수가 아니라 이 joint matching으로 recoverable한
line reference 수다. 이전 보고서의 2,911/3,425 rim-band 관측률과 분모가 다르다.
최근접 2D/3D 거리는 독립 통계이지만 coverage 판정은 같은 pair를 쓴다.
Reference가 없으면 N/A, 선택점이 없으면 0 coverage와 missing distance count를 남긴다.

### 동일 256점 / ratio .25 / 20° / seed 0

표의 `GT / 관측 대비`는 GT coverage / Sampling retention이다. Surface는 **모든 관측점**에서
같은 물체 sampled cloud까지 거리의 mean / p95(mm)다. False selection은 알려진 negative
ROI의 합집합에서 뽑힌 **priority 경로 점 수 / 전체 실제 선택 수**다. 일반 표면 FPS 점을
평면에 있다는 이유로 false selection으로 세지 않는다.

| 조건 | Peg seam GT / 관측 대비 | Box rim GT / 관측 대비 | Peg surface mean / p95 mm | Box surface mean / p95 mm | False priority / selected |
| --- | --- | --- | --- | --- | --- |
| S0 | 27.36% / 27.54% | 7.09% / 9.23% | 3.37 / 6.94 | 6.91 / 16.07 | 0/30,976 (0.00%) |
| S1 | 30.81% / 31.00% | 5.91% / 7.69% | 3.33 / 6.82 | 6.80 / 15.72 | 5,449/30,976 (17.59%) |
| S2 | 59.74% / 60.11% | 11.70% / 15.24% | 3.46 / 7.22 | 7.29 / 17.45 | 1,833/30,976 (5.92%) |
| S3 | 60.90% / 61.28% | 8.77% / 11.43% | 3.42 / 7.02 | 7.17 / 16.28 | 2,430/30,976 (7.84%) |
| S4 | 46.83% / 47.13% | 8.44% / 10.99% | 3.37 / 6.92 | 7.02 / 15.99 | 3,605/30,976 (11.64%) |

S2/S3는 같은 budget에서 Peg 보존을 크게 높였다. S3는 Peg에서 S2보다 약간 높지만,
Box rim에서는 S2가 더 높다. Box의 silhouette/부분 normal/오염 후보와 원거리 유효 depth가
3D FPS budget을 차지하므로, orientation 검출의 높은 rim coverage가 Sampling 뒤 그대로
남지 않는다. 실제 구조 우선 Sampling을 구현한 사실만으로 충분한 geometry 보존을 달성했다고
판단할 수 없다. Hole rim은 이번 구현에서 분명한 실패 영역이다.

### Priority 역할과 후보 포함 수

아래는 121프레임을 합한 Peg+Box count다. Structural ratio는 `priority` 열만 제한한다.
Surface FPS가 후보를 추가로 선택할 수 있어 `최종 후보 포함`은 cap보다 클 수 있다.

| 조건 | 실제 priority | 그중 depth-only | 일반 FPS 경로 | 최종 후보 포함 | 총 selected |
| --- | --- | --- | --- | --- | --- |
| S0 | 0 | 0 | 30976 | 2004 | 30976 |
| S1 | 7744 | 0 | 23232 | 10558 | 30976 |
| S2 | 7744 | 0 | 23232 | 7744 | 30976 |
| S3 | 7744 | 0 | 23232 | 7804 | 30976 |
| S4 | 7373 | 1936 | 23603 | 12744 | 30976 |

### Point budget sweep

Ratio .25 / 20° / seed 0 고정(S0는 priority 없음). 수치는 **관측 대비 Peg / Box 보존율**이다.
같은 budget 안에서 비교해야 하며 기존 1,024점 E9와 혼합하지 않았다.

| 총 budget | S0 | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- | --- |
| 64 | 8.11% / 2.49% | 10.59% / 1.90% | 17.49% / 1.03% | 17.72% / 5.42% | 11.92% / 3.96% |
| 128 | 14.99% / 4.84% | 17.05% / 5.93% | 34.10% / 2.78% | 35.39% / 7.33% | 24.45% / 7.40% |
| 256 | 27.54% / 9.23% | 31.00% / 7.69% | 60.11% / 15.24% | 61.28% / 11.43% | 47.13% / 10.99% |
| 512 | 49.27% / 17.58% | 51.62% / 11.36% | 82.76% / 31.28% | 82.96% / 17.51% | 74.98% / 16.56% |

512점에서도 Box rim 보존은 S2 31.28%, S3 17.51%다. Node budget 확대만으로 해결되지 않는다.
64/128점에서 FD2의 Box rim은 General FPS보다 낮다. 작은 budget에서 특정 후보 집합에
먼저 자리를 쓰는 비용이 실제로 나타난다.

### Structural Ratio sweep

256점 / 20° / seed 0. 각 cell은 **관측 대비 Peg / Box 보존율**이다.

| Ratio | S1 | S2 | S3 | S4 |
| --- | --- | --- | --- | --- |
| 0 | 27.54% / 9.23% | 27.54% / 9.23% | 27.54% / 9.23% | 27.54% / 9.23% |
| 0.125 | 28.56% / 10.04% | 36.30% / 6.37% | 38.77% / 10.92% | 30.84% / 10.33% |
| 0.25 | 31.00% / 7.69% | 60.11% / 15.24% | 61.28% / 11.43% | 47.13% / 10.99% |
| 0.5 | 38.32% / 9.38% | 80.89% / 25.86% | 81.03% / 17.00% | 71.69% / 11.58% |

표면 저하도 함께 확인했다. 256점에서 ratio 0→.5일 때 S2의 surface p95는
Peg **6.942→10.327mm**, Box **16.074→21.930mm**다. S3는 **6.942→8.734mm**,
**16.074→17.876mm**다. 높은 ratio가 seam 보존에는 유리해도 일반 표면 표현은 악화된다.
전체 수치에는 plane-only, signed-face별 표면, ≤64px의 작은 raster component coverage와
frame별/GT reference 사분면별 분포가 있다. 작은 component는 이미지 연결성 진단이며
독립 물체나 물리 edge ID를 뜻하지 않는다.

### Threshold, seed, tolerance 민감도

256점/.25/seed 0의 **관측 대비 Peg / Box 보존율**:

| 조건 | 10° | 20° | 30° | 40° |
| --- | --- | --- | --- | --- |
| S1 | 29.21% / 9.23% | 31.00% / 7.69% | 35.96% / 17.66% | 72.00% / 17.88% |
| S2 | 29.88% / 6.08% | 60.11% / 15.24% | 70.40% / 13.41% | 66.03% / 21.32% |
| S3 | 50.27% / 10.99% | 61.28% / 11.43% | 66.01% / 10.70% | 45.81% / 9.23% |
| S4 | 38.07% / 10.77% | 47.13% / 10.99% | 52.98% / 11.43% | 42.20% / 10.62% |

S1 40°도 Peg 보존에서 강하다. “PCA가 항상 우수하다”거나 ep000 최고 threshold가
독립적으로 검증됐다고 주장하지 않는다. Detector threshold와 candidate pool의 크기·위치가
priority FPS 결과를 함께 바꾸므로, 최종 분류 threshold 선택 문제와 분리해야 한다.

Seed 0/1/2에서의 관측 대비 보존 범위(256/.25/20°):

| 조건 | Peg seed 범위 | Box seed 범위 |
| --- | --- | --- |
| S0 | 26.40%–28.14% | 8.42%–10.04% |
| S1 | 30.43%–31.00% | 7.69%–9.08% |
| S2 | 60.10%–60.81% | 11.14%–15.24% |
| S3 | 60.79%–61.36% | 11.28%–12.45% |
| S4 | 46.57%–47.13% | 10.99%–12.67% |

Box rim은 작은 seed 변화에도 흔들린다. 세 seed는 초기화 민감도 진단이며 독립 episode
일반화 검증이나 통계적 신뢰구간은 아니다. Matching tolerance 민감도 역시 함께 기록했다.
다음은 **관측 대비 Peg / Box 보존율**이며 각 tolerance에서 관측 분모를 다시 계산한다.

| 조건 | 1px AND 2mm | 2px AND 3mm | 3px AND 5mm |
| --- | --- | --- | --- |
| S0 | 13.35% / 3.04% | 27.54% / 9.23% | 51.24% / 19.02% |
| S1 | 15.28% / 1.79% | 31.00% / 7.69% | 54.31% / 21.47% |
| S2 | 28.92% / 5.54% | 60.11% / 15.24% | 85.50% / 34.04% |
| S3 | 30.99% / 3.43% | 61.28% / 11.43% | 82.08% / 22.18% |
| S4 | 22.16% / 3.67% | 47.13% / 10.99% | 74.06% / 21.68% |

### 구조 거리와 false priority

전체 선택점→해당 구조 GT, GT→선택점의 **p95(px / mm)**다. 표면 대표점도 포함한 전체
선택점이므로 selection→GT 거리가 크다는 사실만으로 모두 잘못 선택했다고 해석하지 않는다.
해당 frame에 reference가 없는 경우 거리는 missing이며 그 count를 JSON에 남겼다.

| 조건 | 구조 | sample→GT p95 px / mm | GT→sample p95 px / mm |
| --- | --- | --- | --- |
| S0 | peg | 35.114 / 55.828 | 6.403 / 7.193 |
| S0 | box | 69.924 / 690.805 | 12.999 / 25.193 |
| S1 | peg | 34.886 / 55.828 | 6.403 / 6.902 |
| S1 | box | 69.875 / 690.805 | 13.645 / 28.428 |
| S2 | peg | 34.000 / 51.923 | 4.000 / 5.271 |
| S2 | box | 70.406 / 514.910 | 13.721 / 27.163 |
| S3 | peg | 34.000 / 52.441 | 4.000 / 6.058 |
| S3 | box | 71.395 / 714.983 | 13.003 / 26.462 |
| S4 | peg | 34.176 / 53.068 | 4.472 / 6.484 |
| S4 | box | 70.837 / 690.805 | 13.001 / 25.293 |

Task02-C와 같은 평가 ROI를 썼다. 구조 reference 주변 Manhattan 3px를 제외한 정상 평면,
robot occlusion, same-plane tracker cut, invalid-depth boundary, silhouette, contamination에서
priority를 센다. 아래 원인별 count는 겹치므로 서로 더하지 않는다.

| 조건 | 물체 | 평면 priority | robot 가림 priority | tracker cut priority | 오염 priority | negative 합집합 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | peg | 1776 | 124 | 69 | 115 | 2760 |
| S1 | box | 1047 | 599 | 450 | 31 | 2689 |
| S2 | peg | 44 | 89 | 3 | 202 | 411 |
| S2 | box | 0 | 5 | 77 | 46 | 1422 |
| S3 | peg | 0 | 39 | 24 | 217 | 426 |
| S3 | box | 48 | 177 | 163 | 562 | 2004 |
| S4 | peg | 0 | 199 | 286 | 585 | 1430 |
| S4 | box | 42 | 239 | 186 | 960 | 2175 |

S3의 Peg 평면 false priority는 0이지만 Box의 오염·silhouette 쪽 budget은 남는다.
S4에서 Peg robot 가림 priority는 S3의 39→199개, contamination은 217→585개로 늘었다.
Depth-only가 budget을 차지한 만큼 orientation에 쓸 공간이 줄어 Peg 보존도 낮아졌다.
Near/far 표시는 potential occlusion evidence이며 물리 구조점 확정이 아니다.

## D. 시각화와 실패 사례

- Frame 0: [RGB/후보/선택/GT 확대](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_overlay.png), [Sampling 전후 3D](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_clouds.png).
- Frame 15: [RGB/후보/선택/GT 확대](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0015_overlay.png), [Sampling 전후 3D](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0015_clouds.png).
- Frame 18: [RGB/후보/선택/GT 확대](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0018_overlay.png), [Sampling 전후 3D](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0018_clouds.png).
- Frame 45: [RGB/후보/선택/GT 확대](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0045_overlay.png), [Sampling 전후 3D](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0045_clouds.png).
- Frame 80: [RGB/후보/선택/GT 확대](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0080_overlay.png), [Sampling 전후 3D](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0080_clouds.png).
- Frame 120: [RGB/후보/선택/GT 확대](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0120_overlay.png), [Sampling 전후 3D](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0120_clouds.png).

RGB 그림은 같은 full/Peg/Box crop에서 S0–S4를 비교한다. Red=orientation 후보,
purple hollow square=독립 depth-only 증거, yellow star=priority 선택,
cyan=Surface FPS 선택, magenta +=평가 GT, green=tracker mask다. 색상은 GNN 입력이 아니다.
3D는 원본과 sampled cloud를 같은 카메라/물체별 축으로 표시하며 GT 점은 넣지 않았다.
표시용 .5/99.5 percentile viewport 밖 점 수를 각 panel에 명시했다. 이를 입력 필터로
쓰지 않았고 모든 finite positive depth를 Sampling·평가에 유지했다.

다음 표는 **GT / Tracker 관측 / Sampling 후** reference count다. Main 조건 256/.25/20/seed 0이며
frame별 metric JSON에 S1을 포함한 모든 조건이 있다.

| Frame | 구조 | S0 GT/obs/sample | S2 | S3 | S4 |
| --- | --- | --- | --- | --- | --- |
| 0 | peg | 464 / 462 / 54 | 464 / 462 / 139 | 464 / 462 / 151 | 464 / 462 / 113 |
| 0 | box | 103 / 103 / 6 | 103 / 103 / 16 | 103 / 103 / 3 | 103 / 103 / 6 |
| 15 | peg | 447 / 442 / 72 | 447 / 442 / 234 | 447 / 442 / 247 | 447 / 442 / 185 |
| 15 | box | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 18 | peg | 376 / 376 / 81 | 376 / 376 / 265 | 376 / 376 / 282 | 376 / 376 / 211 |
| 18 | box | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 45 | peg | 287 / 287 / 94 | 287 / 287 / 228 | 287 / 287 / 260 | 287 / 287 / 170 |
| 45 | box | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 80 | peg | 188 / 184 / 53 | 188 / 184 / 98 | 188 / 184 / 76 | 188 / 184 / 54 |
| 80 | box | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 120 | peg | 132 / 129 / 22 | 132 / 129 / 55 | 132 / 129 / 18 | 132 / 129 / 18 |
| 120 | box | 41 / 8 / 3 | 41 / 8 / 0 | 41 / 8 / 2 | 41 / 8 / 2 |

- Frame 0의 rim은 Tracker가 103/103개 관측했지만 S3는 3개만 보존한다. Tracking 실패로
  설명할 수 없는 **Sampling 손실**이다. 공간적 분산만으로 작은 rim에 충분한 node가 남지 않는다.
- Frame 15/45/80은 가시 물리 rim reference가 0이므로 rim coverage는 N/A다. 숨은 입구를 복원하지 않는다.
- Frame 18은 Box tracker mask가 비어 0점을 반환한다. GT 일반 Box 표면과 rim reference 분모는 구분한다.
- Frame 80의 가림·mask 오염에서는 Peg S3가 S2보다 낮다(76 vs 98 reference).
- Frame 120의 rim은 GT 41개 중 Tracker가 joint tolerance 내 8개만 관측한다.
  S3/S4는 그중 2개를 남겨 **GT coverage 4.88%, Sampling retention 25%**다.
  사라진 33개는 관측 손실, 관측된 8개 중 남기지 못한 6개는 Sampling 손실이다.
- 같은 frame의 General FPS는 Peg 22개, rim 3개를 남겨 S3의 18개/2개보다 낫다.
  평균 개선만으로 후반 실패를 숨길 수 없다.

## E. 실제 CPU 성능

Apple M4 Pro, macOS 27.0.1 arm64, root `.venv`, Python 3.11.13, NumPy 2.4.6.
각 estimator 1회 warmup 후 모든 121프레임을 계산하고 estimator 실행 순서를 회전했다.
두 물체를 포함하며 GT·I/O·압축·그림·Logger는 제외했다. CPU 배열만 사용하므로 이 경로에는
H2D/D2H 전송이 없다. Sampling 조건 실행 순서는 고정이고 독립 프로세스로 부하를 격리한
microbenchmark는 아니다. 표는 **p50 / p95 ms**이며 각 단계 percentile의 합이 total은 아니다.

Extraction 단계:

| Estimator | Organized Geometry | Normal | Geometric feature | Pair evidence | Extraction 전체 |
| --- | --- | --- | --- | --- | --- |
| baseline | 1.955 / 2.209 | 3.802 / 5.495 | 1.943 / 3.451 | 1.760 / 3.821 | 9.626 / 14.816 |
| fd2 | 1.914 / 2.183 | 2.724 / 5.398 | 1.880 / 3.320 | 1.730 / 3.860 | 8.329 / 14.657 |
| pca5 | 1.933 / 2.214 | 6.823 / 14.852 | 1.985 / 3.546 | 1.823 / 3.962 | 12.622 / 24.502 |

Sampling 및 전체 단계(256/.25/20/seed 0):

| 조건 | Candidate 선택/정렬 | Budget 배분 | Priority FPS | Surface FPS | Sampling 전체 | Geometry+Sampling |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.548 / 1.177 | 0.041 / 0.047 | 0.052 / 0.073 | 3.241 / 6.198 | 4.000 / 7.359 | 16.537 / 31.940 |
| S1 | 0.572 / 1.262 | 0.047 / 0.055 | 0.593 / 0.725 | 3.095 / 5.803 | 4.450 / 7.852 | 14.151 / 22.668 |
| S2 | 0.558 / 1.196 | 0.049 / 0.055 | 0.244 / 0.296 | 3.085 / 5.817 | 4.034 / 7.336 | 12.522 / 21.959 |
| S3 | 0.553 / 1.163 | 0.047 / 0.053 | 0.235 / 0.292 | 3.068 / 5.788 | 4.025 / 7.300 | 16.653 / 31.876 |
| S4 | 0.550 / 1.225 | 0.053 / 0.060 | 0.236 / 0.302 | 3.147 / 5.832 | 4.085 / 7.291 | 16.566 / 32.025 |

S0는 공정한 feature/candidate 진단을 위해 PCA5 evidence를 같이 계산했고 위 total에도
그 비용을 포함했다. **General FPS의 필수 최소 비용으로 해석하면 안 된다.** S0 selection은
그 evidence를 우선순위로 사용하지 않는다. 현재 serial PCA+pair+sampling p95를 과거의
병렬 CPU geometry 12.97ms나 RTX3090 3.22ms와 같은 구현·측정 범위라고 비교하지 않는다.
이번 run의 CPU 처리 지연은 새로 측정한 값이며 이전 GPU 성능에서 추정하지 않았다.

CUDA/MPS Sampling, 새 simulator backend 실행, tracking 재실행, device transfer,
edge construction, final Vision Output, Graph/GNN/GRU 연결은 **NOT RUN**이다.
**전체 end-to-end 20Hz 검증은 하지 않았다.**

## F. 판단과 한계

1. 같은 256점/20°에서 FD2와 PCA5는 Peg seam에 유용하다. Box rim 개선은 작고 초기/후반의
   실제 Sampling 손실이 크다. General FPS보다 나은 모든-geometry 정책은 아직 확인되지 않았다.
2. Ratio .5는 Peg 구조 보존을 높이지만 일반 표면 p95를 크게 악화시킨다. 작은 물체
   minimum만으로 좁은 hole rim의 coverage까지 보장하지 못한다.
3. Depth-only의 제한적 추가도 이번 조건에서는 false priority를 늘렸다. Raw OR를
   기본 정책으로 삼을 근거가 없다. Depth allocation 25% 자체를 최종값으로 채택하지 않는다.
4. 원거리 positive depth, tracker contamination, 부분 normal, 가림 후보가 3D FPS를 지배할 수 있다.
   GT를 이용한 후보 제거·분배나 임의 전처리로 결과를 개선하지 않았다.
5. Threshold/seed/tolerance에 따라 순위가 달라진다. ep000에서 유망한 값을 같은 ep000의
   일반화 성능으로 보고하지 않는다. 한 episode·연속 pixel-frame micro 집계이며 독립 표본 CI가 아니다.
6. Visible seam은 raster endpoint, rim은 Task02-C visibility gate를 통과한 물리 선분 이산화다.
   정확한 subpixel 전체 CAD edge 검출 성능이나 실제 센서 noise에 대한 검증이 아니다.
7. CPU task만 검증했다. GPU Sampling 최적화나 최종 분류/Graph 변경은 별도 작업이다.

## G. 검증·재현·Git

```bash
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03-mpl \
  .venv/bin/python -m visual.runtime.test.run_sampling
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03-mpl \
  .venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_experiment_logger tests.test_experiment_artifacts -v
git diff --check
```

- Runtime 전체 **100개 실행, 97 PASS / CUDA 3 SKIP / FAIL 0**. 이번 새 테스트 23개 포함.
- 공통 Logger/Artifact 계약 **42/42 PASS**. 인증·원격 업로드·simulator 검증과는 다르다.
- 실제 121프레임의 full PCA fidelity 및 저장된 6,534 selection 재검증 PASS.
- 최종 source hash와 실행 source snapshot, 입력 hash, Logger `status=finished` 확인.
- `git diff --check` PASS. RGB overlay/3D/정량 그림을 직접 열어 표시·빈 mask·실패 사례를 확인했다.
- 공통 `ExperimentLogger`, `experiment_artifacts.digests`, 사후 `register_existing_output`을 재사용했다.
  별도 W&B/HF logger나 publisher는 추가하지 않았다. `WANDB_MODE=disabled` 로컬 기록이며 게시하지 않았다.
- 시작 작업 트리는 clean이었다. 초기 tracked 616파일 중 runtime README와 3개 호환 경로만
  의도적으로 변경했고 나머지 **612파일은 byte 단위 동일**, 삭제 0개다. 기존 dirty/untracked가
  없었으며 원본 episode/mask와 이전 실험 산출물도 보존했다. Git stage/commit/reset은 하지 않았다.
- 첫 실행은 allocation 정수 잔여분 규칙 검토 전 결과다. 두 번째는 수정 코드의 검증 run,
  세 번째(이 보고서의 최종 run)는 3D 원본 cloud 가독성 보정과 테스트 source snapshot 추가 후
  실행했다. UUID별 결과를 별도 보존했고 최종 두 실행의 시간 외 summary는 동일했다.
- 구현 source/보고서는 uncommitted 상태다. Logger의 `git_dirty=true`를 보존하고 source를 함께 저장했다.

최종 artifact 디렉터리에는 `comparison.json/html`, `tradeoffs.png`, `sensitivity.png`,
대표 6프레임의 overlay/cloud PNG, **121개 `fNNNN_samples.npz`와 121개 frame metrics JSON**,
실행 source copy, 검증 기록·테스트 로그·구현 diff가 있다. NPZ는 모든 조건의 선택 좌표,
source pixel, object ID, candidate와 selection role, 정렬된 feature 및 평가 reference를 보존한다.
GT reference는 `evaluation/` prefix이며 실제 Sampling point와 별도다.

완료 범위는 Task03의 현재 관측 Sampling과 Geometry Preservation 검증이다.
최종 Structural Feature 표현, intra-object edge, Vision Output, stale geometry와 Graph 연결은 진행하지 않았다.
