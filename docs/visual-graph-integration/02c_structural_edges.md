# Task 02-C — Structural Edge Candidate Detection 비교

**Task 02-C의 후보 검출 비교를 완료하고 중단한다.** Sampling, 최종 분류, graph,
runtime 기본값은 변경하지 않았다. 후속 Structural Point 검토에는 **C의 PCA 5×5
orientation 후보와 분리된 depth 경계 증거**를 우선 추천한다. CPU 비용이 우선이면
B의 FD 2px도 유력하다. **원시 depth OR variation을 그대로 구조점으로 채택할 만큼
충분한 결과는 어느 방법에서도 얻지 못했다.** A도 높은 threshold에서 경쟁력이 있어
단일 threshold나 단일 지표로 최종 알고리즘을 확정할 수 없다.

최종 run: `b622e76deceb419ab9c5aa80a8f8280c`.
[비교 HTML](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/comparison.html),
[전체 JSON](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/comparison.json),
[threshold / PR 그림](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/tradeoffs.png),
[공간 분포](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/density.png),
[검증·보존 기록](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/verification.json),
[Task 02-C diff](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/implementation.patch).

## 1. 범위와 입력

같은 저장 simulator RGB-D `visual/data/step2/ep000.npz`, 같은
`visual/results/runtime_tracking/ep000/masks.npz`, `base_camera`, 원본 512×512,
source frame **0–120 전체 121개**를 사용했다. 실제 물리 depth 센서 데이터나 독립
episode 검증은 아니다. Tracking / simulator / Grounding DINO를 다시 실행하지 않았다.

- Episode SHA-256: `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`
- Masks SHA-256: `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`
- `geometry.py` SHA-256: `36d20b7873d5ef9e1381e3bbe319da89e16fa5b44455488786dadd250bd814a9`

| 조건 | 후보 검출 |
| --- | --- |
| A | 기존 `estimate_surface_normals` + 기존 variation + depth jump |
| B | Task 02-B `finite_difference(..., 2)` + 같은 variation + depth jump |
| C | Task 02-B `local_plane(..., 5)` + 같은 variation + depth jump |
| D | 공식 PCL 검토 완료, dependency 부재로 **NOT RUN** |

Normal 지원, invalid 처리, patch 연결·rank 조건은 Task 02-B와 동일하다. 최적화
PCA 구현으로 교체하지 않았다. Variation은 1px 4-neighbour의 최대 oriented angle,
depth gate는 기존 `abs(dz) > 0.01m + 0.02*min(z1,z2)`다. GT, segmentation,
pose, 물체 치수는 검출 후 평가에만 사용한다. RGB는 overlay 배경에만 사용했다.
RGB Canny 조건을 추가하지 않았고 RGB node feature도 만들지 않았다.

## 2. 서로 다른 경계 증거와 pixel-pair 보존

새 비교 전용 [`structural_edges.py`](../../visual/runtime/test/structural_edges.py)는
각 물체 raw mask에 닿는 인접 좌우/상하 pair를 한 번씩 저장한다. Pair의 두 endpoint는
원본 `flat_index=v*W+u`다. 매 프레임의 `fNNNN_pairs.npz` 121개에 다음을 보존했다.

| 필드 | 의미 |
| --- | --- |
| `pixels[P,2]`, `shape` | 원본 인접 pixel pair, 원본 raster 크기 |
| `supported[P,2]` | 각 endpoint가 해당 물체의 유효·비중첩 관측점인지 |
| `normal_angle_deg[P]` | 양쪽 normal이 유효할 때의 orientation 차이; 나머지는 NaN |
| `signed_depth_delta_m[P]`, `depth_jump[P]` | 유효 depth 양쪽의 signed 차이와 기존 jump gate 결과 |
| `near_endpoint`, `far_endpoint` | jump의 가까운/먼 endpoint 0 또는 1; jump 없으면 −1 |
| `mask_boundary`, `invalid_depth_boundary`, `overlap[P,2]` | mask 전환, depth validity XOR, tracker 중첩을 독립적으로 기록 |

`orientation`, `depth`, `union`, `occluding`, `occluded`, `invalid`, `mask` map은
pair에서 복원할 수 있다. 표의 주 비교 후보는 요구한 **orientation OR depth**다.
원인 flag는 겹칠 수 있고, 같은 pixel이 서로 다른 pair에서 near와 far 양쪽일 수도 있다.
near/far는 **관측 depth 순서에 따른 잠재적 가림 방향**이며 실제 robot 가림의 확정
라벨이 아니다. Depth jump에는 grazing 표면·silhouette·tracker 오염도 섞인다.
Mask 전환이나 invalid depth 자체를 structural 후보에 OR하지 않는다.

영상 밖에는 가상 pair를 만들지 않는다. Raster shape로 image border를 확인한다.
가림 뒤의 depth를 보간하거나 무효 depth를 0m 점으로 비교하지 않는다. Pair 정보는
향후 intra-object filtering에서 양쪽 소속·jump 방향을 검사하는 데 필요하다.
이 pair는 관측 증거이며 **GNN graph edge를 생성한 것이 아니다.**

## 3. GT와 지표 정의

Task 02-A/B의 `real_truth`, `real_regions`, `rim_region`, analytic cuboid oracle을
변경 없이 재사용했다. Peg plane의 analytic depth 잔차 ≤1.5mm, 다른 면에서 4mm
여유, signed face별 3px erosion도 그대로다.

Peg seam reference는 서로 다른 **GT 가시 signed face의 맞닿는 양쪽 raster endpoint**
32,664 pixel-frame다. 따라서 Peg의 위치 오차는 그 endpoint에 대한 값이며 subpixel
CAD 선분까지의 정확한 오차는 아니다. Box의 다른 가시 face junction도 전체 precision의
positive reference에 포함한다. Silhouette를 구조 GT로 추가하지 않는다.

Hole rim은 기존 **가시 물리 rim band 3,425 pixel-frame**의 coverage를 별도로 유지했다.
그 안에서 보이는 입구 선분을 subpixel로 투영하고, 기존 가시 rim 표면까지 ≤1.5px 및
≤4mm인 선분 sample만 남겼다. 투영된 raster cell당 선분의 가장 가까운 한 점을 선택한
**1,778개 물리 선분 reference**로 위치와 coverage를 평가했다. 이는 평가용 GT 이산화다.
검출 point의 sampling은 아니다. 가려진 전체 rim 투영을 정답으로 채우지 않았다.
GT 가시성 역시 이 raster/metric tolerance를 이용한 근사라는 한계가 있다.

전체 reference는 물체별 seam/rim을 같은 raster cell에서 중복 제거한 **35,869개**다.
같은 물체끼리만 매칭한다. 후보를 GT 주변 corridor로 제한하지 않고 tracker의 모든
관측 point에서 평가한다. 잘못 포함된 robot/background 후보도 precision 분모에 남는다.

- **Precision:** 전체 후보 중 같은 GT reference와 **≤2px AND ≤3mm**를 동시에 만족하는 비율.
- **Recall / coverage:** GT reference 중 같은 조건의 후보가 하나 이상 있는 비율. Tracker가 놓친 reference도 분모에 남는다.
- **일대일 지표:** 같은 허용오차의 최대 bipartite matching으로 후보와 reference를 각 한 번만 인정한다. 넓은 후보 띠의 중복을 벌점화한다.
- **위치 오차:** 후보→GT와 GT→후보의 최근접 2D/3D 거리를 모두 기록한다. 두 거리 통계의 최근접 상대는 다를 수 있지만 P/R 판정은 같은 pair에 두 조건을 적용한다.
- **False candidate:** 알려진 구조 reference 주변 Manhattan 3px를 제외한 plane / silhouette / occlusion / tracker-cut ROI에서 후보 비율. ROI는 서로 겹칠 수 있다.

Box plane은 GT signed face별 3px erosion과 rim 4px 제외를 추가한 평가 control이다.
Robot occlusion은 저장 `segmentation_id_map.json`의 `panda_*` IDs 1–13을 사용해
기존 occlusion ROI의 **물체 가시 측**만 구분했다. 일반 occlusion과 구분해 JSON에 남겼다.
Invalid-depth Peg ROI의 관측점은 0개이므로 그 false rate는 N/A다.

Reference가 없는 물체/frame의 후보도 precision 분모에 남고 위치 오차만 missing이다.
후보가 없으면 recall은 0이다. 일대일 지표는 GT raster 이산화/밀도에 의존하며 최종
sampling 품질의 정답 점수는 아니다. Micro 집계는 연속 pixel-frame이므로 독립 표본에
대한 신뢰구간이나 일반화 성능을 주장하지 않는다.

## 4. 동일 20°에서의 후보 검출

20°는 공통 비교·그림을 위한 조건이며 production threshold가 아니다.

| 방법 | 후보 수 | Precision | Recall | 일대일 F1 | Peg seam coverage | 물리 rim line coverage | 기존 rim band coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 432,182 | 15.88% | 95.03% | 14.36% | 96.44% | 74.92% | 69.75% |
| B | 107,297 | 51.64% | 91.08% | 45.14% | 93.29% | 74.18% | 63.33% |
| C | 101,948 | 49.02% | 91.76% | 47.29% | 94.04% | 75.31% | 63.85% |
| D | NOT RUN | — | — | — | — | — | — |

A의 95.03% coverage는 363,539개의 unmatched 후보를 동반한다. B/C의 unmatched는
51,894/51,973개다. 전체 관측점 1,051,134개에서 후보 밀도는 A 41.12%, B 10.21%,
C 9.70%다. Peg/Box별 밀도는 A 49.76/24.90%, B 11.24/8.28%, C 10.06/9.02%다.
고밀도 A가 coverage에서 유리하다는 사실을 precision과 일대일 점수가 드러낸다.

## 5. Threshold sweep과 trade-off

**2, 5, 10, 15, 20, 25, 30, 40, 60, 90, 180°**의 모든 조건에서 같은 GT와
동일한 tolerance를 썼다. 180°는 depth-only control이다. Depth gate와 normal
support는 기존 비교를 보존하기 위해 고정했다. Depth threshold 최적화 결과는 아니다.
JSON에는 union과 orientation-only의 전체 sweep, 분모·분자, 위치·밀도가 있다.

| 방법·각도 | Union precision / recall | Union 일대일 F1 | Peg seam / 물리 rim coverage |
| --- | ---: | ---: | ---: |
| A 10° | 9.77 / 97.35% | 8.35% | 98.55 / 76.27% |
| A 30° | 24.06 / 92.77% | 24.49% | 94.74 / 74.92% |
| A 40° | 43.48 / 91.65% | 53.22% | 93.79 / 74.92% |
| B 10° | 20.64 / 94.77% | 17.00% | 95.88 / 74.18% |
| B 20° | 51.64 / 91.08% | 45.14% | 93.29 / 74.18% |
| B 30° | 44.04 / 89.36% | 52.19% | 91.89 / 74.18% |
| C 10° | 53.36 / 93.11% | 40.77% | 95.07 / 75.31% |
| C 20° | 49.02 / 91.76% | 47.29% | 94.04 / 75.31% |
| C 30° | 41.45 / 85.93% | 50.56% | 89.17 / 75.20% |

A 40°의 union 일대일 F1은 이 sweep에서 B/C의 최대값보다 높다. Quantization의
큰 반응을 threshold로 잘라내면 A도 유력하므로 “PCA가 모든 지표에서 우세”라고
결론 내릴 수 없다. 이 수치는 같은 episode에서 본 최대값이며 validation으로 선택한
threshold가 아니다. FD/PCA의 지원 window는 진짜 sharp orientation 반응도 퍼뜨려
40° 이후 seam coverage를 크게 잃는다.

Orientation-only 20°의 P/R은 A **17.19/91.86%**, B **86.33/86.39%**,
C **80.29/90.19%**다. 물리 rim coverage는 각각 **40.89/39.82/64.29%**다.
C는 B보다 orientation만으로 rim을 더 보존하지만 부분 patch의 boundary 오반응도 남는다.
Depth-only는 모든 방법에서 후보 **45,697개**, P/R **4.86/7.40%**, rim coverage
37.06%다. Depth를 합치면 rim recall이 늘어도 전체 precision은 크게 낮아진다.
관측 depth jump를 structural이라고 자동 분류하면 안 된다는 직접적인 근거다.

## 6. 위치와 관측 실패

20° union에서 GT→후보 거리의 p50 / p95:

| 대상·방법 | 2D (px) | 3D (mm) |
| --- | ---: | ---: |
| Peg A | 0 / 1 | 0.577 / 1.401 |
| Peg B | 0 / 1 | 0.602 / 3.693 |
| Peg C | 0 / 1 | 0.619 / 3.446 |
| Rim A | 0.345 / 13.465 | 1.483 / 27.064 |
| Rim B | 0.356 / 14.373 | 1.514 / 29.199 |
| Rim C | 0.389 / 14.351 | 1.538 / 28.433 |

전체 후보→GT 거리의 p50 / p95는 A **10.00/37.66px, 9.99/74.69mm**,
B **1.41/47.01px, 1.41/164.69mm**, C **1.41/51.67px, 1.87/176.89mm**다.
물체/frame에 가시 reference가 있을 때의 거리이며 N/A frame 후보 수는 별도 기록한다.
B/C의 p95가 더 큰 것은 평면 후보가 줄어드는 동안 먼 오염·가림 후보의 비중이
커지는 효과도 포함한다. 이를 유효한 seam의 위치가 전부 더 나빠졌다는 뜻으로
해석하지 않는다. Raw positive depth outlier도 제거하지 않아 3D maximum은 매우 크다.

| Frame | Peg seam GT / A·B·C 검출 | 물리 rim GT / A·B·C 검출 | 확인 사항 |
| --- | ---: | ---: | --- |
| 0 | 464 / 461·459·462 | 103 / 103·103·103 | A의 넓은 평면 오반응, B/C의 crease 집중 |
| 15 | 447 / 441·436·441 | 0 / N/A | 손 가림, rim 비가시성을 검출 실패율로 만들지 않음 |
| 45 | 287 / 287·286·287 | 0 / N/A | Box orientation 후보 B/C=0, depth 후보는 각각 159개 남음 |
| 80 | 188 / 176·156·159 | 0 / N/A | 손 가림, tracker 오염·중첩·지원 손실 |
| 120 | 132 / 96·82·83 | 41 / 4·4·4 | Rim 관측 실패가 모든 방법의 coverage를 제한 |

Frame 120은 기존 rim band 51개 중 tracker 관측이 15개이고, union 후보는 모두 6개다.
물리 선분 reference coverage는 4/41=9.76%다. 정상적인 평균 rim 결과만 보고 이런
실패를 숨기면 안 된다. 기존 band 전체 3,425개 중 tracker 관측은 2,911개(84.99%)다.
어떤 estimator도 잃은 관측을 복원하지 않는다.

## 7. 평면·가림·mask 혼동

다음은 알려진 구조 주변 3px를 제외한 **관측 point 중 union>20° 후보 비율**이다.
GT 경계 ROI는 검출 입력이나 후보 삭제에 사용하지 않았다.

| 평가 영역 | A | B | C |
| --- | ---: | ---: | ---: |
| Peg 정상 평면 (342,004개) | 42.133% | 0.116% | 0.0015% |
| Box 정상 평면 (256,187개) | 18.301% | 0% | 0.036% |
| Peg silhouette | 48.130% | 21.014% | 21.235% |
| Box silhouette | 34.105% | 21.630% | 24.567% |
| Peg robot occlusion | 19.683% | 14.535% | 13.935% |
| Box robot occlusion | 44.784% | 28.957% | 29.094% |
| Peg 같은 면 tracker cut | 15.812% | 14.079% | 14.221% |
| Box 같은 면 tracker cut | 35.967% | 20.372% | 22.436% |
| Peg tracker contamination | 48.557% | 42.540% | 48.115% |
| Box tracker contamination | 63.294% | 49.653% | 59.528% |
| Box invalid depth 주변 | 20.915% | 2.577% | 2.577% |

양자화 평면의 대량 후보는 B/C에서 줄었다. 그러나 mask/robot/silhouette 혼동은
해결되지 않았다. Invalid나 mask flag 자체를 union에 넣지 않아도 인접한 실제 depth
차이와 불완전한 normal 지원 때문에 후보가 생긴다. 이 표는 비배타적인 경계별 위험을
보여주며 원인별 FP 수를 서로 더하면 안 된다.

## 8. 시각화와 공간 분포

[0](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_comparison.png),
[15](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0015_comparison.png),
[45](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0045_comparison.png),
[80](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0080_comparison.png),
[120](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0120_comparison.png)은
각 frame의 모든 방법에서 같은 full / Peg / Box crop, source `(u,v)` 좌표계를 쓴다.
GT 패널과 detector 패널은 분리했다. Frame별 crop은 물체 이동을 따라가므로 모든
frame에 똑같은 고정 crop을 강제하지는 않았다.

빨강=orientation 후보, 주황 square=depth jump, 자홍=near/잠재 occluding,
청록=far/잠재 occluded, 흰색=invalid depth, 초록 circle=mask 경계다. 겹친 원인은
marker 형태와 pair flags로 확인한다. 이 색상은 디버깅 전용이며 모델 입력이 아니다.
GT 구조는 별도 패널의 노란색이다.

8×8 source-image tile별 누적 후보 수와 물체별 관측 대비 밀도를 JSON/heatmap에
남겼다. A는 넓은 Peg·Box 평면에 퍼지고 B/C는 seam 주변에 모이지만, 손에 가려진
후반 frame의 외곽·오염 후보는 남는다. 밀도 그림은 point-frame 점유 분포이며
3D 표면 면적당 밀도나 sampling budget을 뜻하지 않는다.

## 9. CPU 비용

macOS arm64, Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1 CPU에서 실행했다.
121프레임 각각 Peg+Box를 계산하고 방법별 1회 warmup을 제외했다. 방법 실행 순서를
frame마다 회전했다. 공유 XYZ 생성 비용을 각 방법 total에 포함했다. I/O, GT,
threshold sweep 전체, 평가·매칭, 압축 저장, 그림, logger는 제외했다.

아래 값은 stage별 **p50 / p95 ms**, total은 단계 percentile의 합이 아닌 실제 합계의
percentile이다. 관측 물체 면적이 줄어들어 특히 PCA의 frame별 비용 차이가 크다.

| 방법 | Normal | 기존 feature | Pair 증거 | 한 threshold map | 전체 (XYZ 포함) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 3.978 / 5.614 | 8.406 / 9.873 | 1.902 / 4.105 | 0.307 / 0.621 | 20.888 / 26.918 |
| B | 2.815 / 5.610 | 8.387 / 9.750 | 1.860 / 3.933 | 0.293 / 0.558 | 19.730 / 26.794 |
| C | 21.909 / 62.898 | 8.539 / 9.698 | 1.911 / 4.136 | 0.294 / 0.587 | 39.364 / 82.433 |
| D | NOT RUN | — | — | — | NOT RUN |

공유 XYZ의 p50/p95는 6.793/7.532ms다. C의 total p50은 B의 약 2배다. 기존 PCA
최적화 산출물은 보존했고 이번 비교에서는 사용하지 않았다. PCL의 추출·array copy·binding
비용과 메모리 사용은 **미측정**이며 0ms 또는 빨라질 것으로 보고하지 않는다.

## 10. PCL 검토·설치 및 binding 제약

현재 root `.venv`에 `pcl`, `pclpy`가 없고 `ctypes.find_library`로 `pcl_common`,
`pcl_features`도 찾지 못했다. `/opt/homebrew`, `/usr/local`, `/usr`의 표준 include
경로에 organized-edge header가 없고 `pkg-config`도 없다. **D=NOT RUN**이다.
시스템 package 설치, C++ build, Python binding 구현은 하지 않았다.

공식 [`OrganizedEdgeFromNormals` header](https://pointclouds.org/documentation/organized__edge__detection_8h_source.html)는
XYZ+normal에서 NAN_BOUNDARY=1, OCCLUDING=2, OCCLUDED=4, HIGH_CURVATURE=8을
반환한다. RGBNormals는 RGB_CANNY=16도 지원한다. Geometry-only 통합 시 FromNormals를
쓰고 `setEdgeType(1|2|4|8)`로 RGB를 제외하는 것이 적합하다. Labels는 bitmask이며
Corner/Edge/Surface 정답으로 변환하면 안 된다.

공식 [구현](https://pointclouds.org/documentation/organized__edge__detection_8hpp_source.html)을 보면
depth는 8-neighbour 및 invalid 구간 검색을 사용하고, high-curvature는 normal x/y에
Canny를 적용한다. A/B/C의 4-neighbour angle threshold와 같은 알고리즘·단위가 아니다.
Depth threshold와 high-curvature low/high threshold를 각각 sweep해야 한다.

추후 허용된 환경에 통합할 정확한 범위:

1. **고정 버전 dependency:** 검토한 PCL 1.15.1 기준 CMake ≥3.16.3,
   C++17 기본 compiler, Eigen ≥3.3, Boost ≥1.71(`iostreams`; C++14 선택 시 `filesystem`도),
   FLANN ≥1.9.1. `pcl_features`의 전이 component는 common/search/kdtree/octree/filters/2d다.
   [루트 CMake](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/CMakeLists.txt),
   [Boost 검사](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/cmake/pcl_find_boost.cmake),
   [features dependency](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/features/CMakeLists.txt) 기준이다.
   이 geometry-only 경로에 visualization/VTK, 센서 OpenNI, CUDA는 필요하지 않다.
2. **작은 독립 adapter:** 전체 PCL Python wrapping 대신 compiled CLI 또는 최소 pybind11
   함수 하나로 `[H,W,3]` XYZ/normal → `[H,W] uint32` label과 원본 index를 반환한다.
   pybind11을 선택하면 해당 Python 3.11 arm64 ABI와 pybind11 build dependency가 추가된다.
3. **Raster/NaN 계약:** width/height를 유지하고 invalid XYZ만 NaN 처리한다. Tracker 밖의
   유효 배경 depth를 mask 때문에 NaN으로 바꾸면 인공 NAN_BOUNDARY가 생기므로 구분한다.
   물체별 normal validity·camera-facing 규약, PCL의 NaN normal 처리도 실행 검증해야 한다.
4. **공정성:** 우선 C의 같은 PCA5 normals를 공급해 edge 알고리즘 차이를 비교하고,
   PCL normal estimator를 추가한다면 독립 조건으로 나눈다. GT는 입력하지 않는다.
5. **Provenance/검증:** 공식 label은 pixel label이라 실제 검색 상대 pair를 주지 않는다.
   이번 4-neighbour pair 증거는 함께 보존하되 PCL의 8-neighbour/invalid-search 상대라고
   주장할 수 없다. 그 상대가 필요하면 C++ instrumentation이 추가 작업이다.
   합성 plane/crease/step/invalid/mask control, 기존 121프레임, 전체 PR, marshal·normal·PCL
   시간을 분리해 실행해야 한다. 이 항목들은 후속 작업 범위이며 이번에 구현하지 않았다.

## 11. 추천과 중단 지점

C의 orientation 후보는 10–20°에서 plane 오반응을 억제하면서 seam coverage
95.07–94.04%, orientation-only rim coverage 65.24–64.29%를 보였다. B의 같은
orientation-only rim은 약 39.82%다. 가시 rim과 부분 normal 지원을 유지해야 하는
후속 Structural Point 검토의 우선 후보로 C를 추천하는 이유다.

CPU 지연이 중요하면 B가 더 현실적이다. B 20°는 C 20°보다 전체 union precision이
높고 비용이 절반 정도다. A 40°의 일대일 F1도 더 높으므로 후속 검토에서 제외하지
않는다. Rim의 잃은 관측, robot/mask 경계, threshold에 따른 구조 약화 때문에
**어떤 raw union도 충분한 최종 구조점 selector로 판단하지 않는다.**

후속 검토에서는 orientation 후보와 depth/near/far 증거를 분리해 해석하는 것이
필요하다. 이번에는 경계 제거 정책, candidate thinning, Structural/Surface sampling,
node budget, onehot/probability feature, intra-object graph, Tensor 출력, stale 처리,
GNN/GRU 변경을 구현하지 않았다. `00_interface.md`와 Point Cloud Representation도
변경하지 않았으며 production 기본값을 확정하지 않았다.

## 12. 테스트·변경 파일·재현성

- 새 unit/integration 테스트 **11/11 통과**: 원본 baseline 전체 field 동일성, 모든
  sweep union 일치, 원본 인접 index/no wrap, mask cut·invalid depth의 비구조성,
  depth 양쪽·외부 이웃, overlap, RGB 비의존성, 2D 근접 occluder의 3D 매칭 실패,
  dense false 후보의 일대일 벌점, missing reference/prediction, UUID와 전 frame 보존.
- 전체 runtime 회귀 **77개 실행: 74 통과, CUDA 3 skip**. Task 02-A/B 및 기존 PCA
  최적화 CPU 테스트도 포함했다. CUDA/GPU, PCL, simulator backend는 이번에 실행하지 않았다.
- 공통 Logger/Artifact **42/42 통과**. Sandbox 안의 실제 W&B offline SDK 테스트는
  `ServicePollForTokenError`로 1개 실패했고, 동일 suite를 허용된 sandbox 밖에서 재실행해
  42개 모두 통과했다. 실제 HF 게시나 W&B online 검증은 하지 않았다.
- 121프레임 × 3방법 × 2물체 × 11 threshold에서 pair union과 기존 feature map의
  full-raster 일치 assertion을 통과했다.

```bash
WANDB_MODE=disabled MPLCONFIGDIR=/private/tmp/task02c-mpl \
  .venv/bin/python -m visual.runtime.test.run_structural_edges

WANDB_MODE=disabled MPLCONFIGDIR=/private/tmp/task02c-mpl \
  .venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v

.venv/bin/python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -v
```

추가한 code/doc는 아래 4개다. 기존 runtime와 Task 02-A/B 구현을 수정하지 않았다.

| 파일 | 역할 |
| --- | --- |
| [`structural_edges.py`](../../visual/runtime/test/structural_edges.py) | 비교 전용 pair evidence, A/B/C 검출·시간 측정 |
| [`run_structural_edges.py`](../../visual/runtime/test/run_structural_edges.py) | GT 평가, PR sweep, 전체 121 pair 저장, 그림·공통 Logger |
| [`test_structural_edges.py`](../../visual/runtime/test/test_structural_edges.py) | 11개 검증 |
| 이 문서 | 결과와 제한, 추천·중단 기록 |

`ExperimentLogger(team='visual', job_type='evaluate')`의 UUID/project/group과 로컬 기록,
`link_output`/기존 사후 등록을 재사용했다. `utils.experiment_artifacts`의 digest와
artifact inventory를 사용했으며 별도 W&B/HF transport를 추가하지 않았다. 원격 게시 없음.
최종 UUID 디렉터리에 JSON, HTML, 그림 7개, pair NPZ 121개, 실행 당시 source 사본을
보존했다. 중간 확인 run `6eed92...`, `935a86...`도 별도 디렉터리에 남겼으며 최종 수치는
위 `b622e76...`만 사용한다.

작업 시작 commit은 `42e30e4...`, 최종 평가가 기록한 commit은 `366e9ce...`, dirty=true다.
작업 도중 별도의 Task 02-B commit과 게시 문서 commit `b1ebe0e...`가 추가되었다.
시작 시 hash한 기존 파일 368개 중 **366개는 동일**하다. 이 작업은 기존 데이터·결과·
runtime·interface를 수정하지 않았다. 달라진 `02b_normal_comparison.md` 1개는 별도
Task 02-B commit `366e9ce`에 포함된 게시 안내 5줄 추가다. 안내를 제외하면 시작 hash와
일치하며 그 원문 사본도 보존했다. 다른 차이는 Finder 메타데이터 `.DS_Store` 1개다.
보존 기록은 이 외부 변경을 구분한다. Task 02-C 파일은 stage/commit하지 않았다.
새 파일을 포함한 diff는 `implementation.patch`로 제공하며, 기존 tracked code diff는 없다.
