# Task 02-B — Quantization-Robust Normal Estimation 비교

전체 HTML·수치·dense NPZ·평가 당시 코드 사본은 [HF 게시 결과 인덱스](../../visual/results/publications/README.md)에서
받을 수 있다. 아래 로컬 결과 링크는 ZIP의 결과 파일을 새 checkout에 복원하면 열린다.
[대표 그림](../../visual/results/diagnostic_previews/README.md)은 Git에서도 바로 볼 수 있다.
이 문서의 실험·테스트·Git 상태는 Task 02-B 완료 시점 기록이며, 이후 게시 내역은 인덱스를 따른다.

**Task 02-B에서 완료·중단한다.** 후속 검토 후보로 **PCA 5×5**를 추천한다.
평면 오반응, 실제 face seam 위치, 관측 normal coverage 사이에서 가장 유용한 절충을
보였다. 다만 hole rim 반응 손실과 CPU 비용이 있어 최종 채택을 뜻하지 않는다.
PCA 7×7이나 FD 4px를 평면 지표만으로 선택하면 실제 crease 반응을 잃는다.
Runtime 기본 estimator와 threshold, [확정 interface](00_interface.md)는 바꾸지 않았다.

Run: `4b1c377ae2044344a4f596b249627d89`.
[비교 HTML](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/comparison.html),
[전체 수치 JSON](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/comparison.json),
[trade-off 그림](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/tradeoffs.png),
[검증 기록](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/verification.json),
[이 Task의 코드·문서 diff](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/implementation.patch).

## 1. 변경 범위와 공통 조건

새 파일만 추가했다.

| 파일 | 역할 |
| --- | --- |
| [`normal_estimators.py`](../../visual/runtime/test/normal_estimators.py) | 독립 FD/PCA 비교 후보; runtime에서 자동 호출하지 않음 |
| [`run_normal_comparison.py`](../../visual/runtime/test/run_normal_comparison.py) | 합성·실제 동일 조건 평가, paired 통계, 경계 위치·coverage, CPU 시간, 시각화 |
| [`test_normal_comparison.py`](../../visual/runtime/test/test_normal_comparison.py) | 수치·지원 제한·baseline 동일성·평가 identity 테스트 13개 |
| 이 문서 및 새 UUID 결과 디렉터리 | 결과·한계·보존 증거 |

Baseline은 Task 01의 `estimate_surface_normals`와 `compute_geometric_features`를 그대로
호출한다. 모든 방법은 **같은 원본 organized XYZ, depth, intrinsic, tracker mask**를
사용하며 512×512 raster를 유지한다. Normal variation은 모두 유효한 **1px 4-neighbour
normal과의 최대 oriented angle**이다. FD 간격을 늘려도 variation 비교 간격은 늘리지
않는다. Task 02-A의 stride downsampling과 다르다.

GT segmentation, pose, 치수, analytic geometry는 estimator 입력이 아니다. 추출 후에만
기존 `real_regions`, `rim_region`, `synthetic_case`, `metrics`, `summarize_rows`를 재사용했다.
RGB는 그림에만 사용한다. NPZ의 GT는 `evaluation/` 아래에 분리했다.

| 방법 | Normal 지원 | 지원 제한·무효 처리 |
| --- | --- | --- |
| Baseline | 중심과 ±1px 십자형 5점 | 기존 동작 그대로 |
| FD 2px | ±2px secant의 cross product, span 4px | 중심부터 네 끝점까지 모든 중간 pixel이 같은 물체 support이고 모든 인접 depth pair가 gate 통과; 전체 경로가 없으면 invalid |
| FD 4px | ±4px secant, span 8px | FD 2px와 같은 규칙; downsampling 없음 |
| PCA 5×5 | patch 안 중심과 연결된 유효 3D 점의 covariance 최소 고유벡터 | 같은 물체·유효 depth·인접 jump gate를 만족하는 4-connected component만 사용; 부분 patch 허용 |
| PCA 7×7 | 위와 같은 PCA, 7×7 patch | 같은 규칙 |

Gate는 기존 `|Δz| > 0.01m + 0.02*min(z1,z2)`를 그대로 사용한다. 후보에서는
**인접 1px pair**에 적용하며 긴 secant의 누적 depth 차이에 다시 적용하지 않는다.
PCA 연결 탐색은 해당 patch 내부에서만 수행하므로 외부 우회 경로로 떨어진 조각을
합치지 않는다. 연결된 연속 sharp crease나 gate 이하의 단차는 여전히 섞일 수 있다.

PCA는 중심 포함 최소 6점, covariance rank ≥2 및 유일한 최소 고유벡터가 필요하다.
두 번째 고유값과 최소 두 고유값의 차이는 각각
`max(1e-16 m², largest_eigenvalue*1e-8)`보다 커야 한다. Float64 중심 상대 covariance,
uniform weight, batch 4096, camera-facing 방향을 사용한다. Residual trimming, GT face
분할, smoothing, 무효점 보간은 없다. Invalid normal/variation은 NaN과 별도 validity다.

Normal 지원 반경은 FD2/FD4가 축 방향 2/4px, PCA5/PCA7이 정사각형 반경 2/3px이며
variation은 여기에 1px 이웃 normal의 지원을 추가한다. z=0.25m, fx=256에서
4/8/6px의 축 방향 span은 횡방향 기준 약 3.91/7.81/5.86mm다. 실제 표면 위 거리와
동일하지 않으며 깊이·기울기에 따라 달라진다. 지원 크기에 맞춰 GT ROI를 재조정하지 않았다.

## 2. 데이터와 평가 영역

실제 데이터는 Task 02-A와 같은 저장 simulator RGB-D `visual/data/step2/ep000.npz`,
`base_camera`, source 0–120의 **121프레임**이다. 물리 센서에서 새로 취득한 데이터가 아니다.
Mask는 `visual/results/runtime_tracking/ep000/masks.npz`를 그대로 읽었다.

- Episode SHA-256: `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`
- Mask SHA-256: `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`
- 보존한 `geometry.py`: `36d20b7873d5ef9e1381e3bbe319da89e16fa5b44455488786dadd250bd814a9`

Peg 평면 ROI는 기존의 GT 가시 signed face, analytic depth 잔차 ≤1.5mm, 다른 면에서
4mm 이상 거리, signed face별 3px erosion이다. Seam은 다른 가시 face가 맞닿는 양쪽
raster endpoint와 기존 2px band다. Hole rim은 실제 입구 선분과의 3D 거리 ≤4mm 및
2px 투영 band를 동시에 만족하는 GT 가시 표면이다. 가려진 rim의 투영은 정답으로
추가하지 않았다. 평가 영역·5/10/20/30° threshold를 결과에 맞춰 변경하지 않았다.

기존 NPZ 94개(Visual 데이터·tracking 및 Graph audit)와 대표 Graph HDF5를 확인했다.
다른 RGB-D·Tracking Mask episode 쌍은 찾지 못했다. `runtime_tracking/ep001`도
`tracking.json`의 입력이 같은 `ep000.npz`여서 독립 검증으로 쓰지 않았다. Graph audit는
state/GT이고 확인한 HDF5도 RGB-D 관측 쌍을 제공하지 않았다.
[탐색 inventory](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/episode_inventory.json).
**실제 결과와 추천은 ep000에 한정된 진단이며 독립 episode 검증 결과가 아니다.**

분포·percentile은 pixel-frame을 합친 micro 통계다. 연속 프레임은 독립 표본이 아니므로
유의성·신뢰구간을 주장하지 않는다. 경계 band는 겹칠 수 있다. Band 반응률과 아래
위치 coverage는 최종 structural detector의 precision/recall이 아니다.

## 3. 실제 Peg 평면: 정확도·variation·false response

고정 GT ROI 342,378 pixel-frame 중 같은 관측 point는 342,257개다.
각 방법의 유효 normal에서 error를, 유효 variation에서 variation과 >threshold 비율을
계산했다. 아래 비율의 분모를 coverage 표와 함께 해석해야 한다.

| 방법 | Error p50 / p95 (°) | Variation p50 / p95 (°) | >5° | >10° | >20° | >30° | Normal / 관측 point |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 5.364 / 23.564 | 14.677 / 33.547 | 66.293% | 60.138% | 42.123% | 21.490% | 99.948% |
| FD 2px | 3.245 / 8.687 | 5.905 / 16.780 | 53.479% | 28.859% | 0.114% | 0% | 99.861% |
| FD 4px | 1.109 / 4.980 | 2.736 / 9.288 | 33.093% | 0.541% | 0% | 0% | 98.257% |
| PCA 5×5 | 1.359 / 6.246 | 2.099 / 7.534 | 17.527% | 0.008% | 0% | 0% | 100% |
| PCA 7×7 | 0.795 / 3.579 | 1.018 / 3.423 | 0.261% | 0.001% | 0.001% | 0.001% | 100% |

이 ROI에서 normal과 variation의 유효 비율은 방법별로 동일하다. PCA7의 >20°는
342,257개 중 3개다. 이 3개의 원인을 별도로 확정하지 않았으며 수치를 제거하지 않았다.
Baseline의 >20°는 144,093개, FD2는 391개다.

모든 방법의 normal이 동시에 유효한 **336,293개의 동일 pixel**에서도 error p95는
Baseline / FD2 / FD4 / PCA5 / PCA7 순서로
**23.564 / 8.696 / 4.980 / 6.256 / 3.579°**다. 단순히 어려운 pixel을 invalid로
제외해서 얻은 개선만은 아니다. JSON에는 방법별 baseline과의 pairwise 공통 지원 및
all-method 공통 variation 비교도 있다.

## 4. Peg의 실제 face seam 보존과 위치

기존 2px band는 95,415 pixel-frame, tracker 관측은 94,052개다. 아래 반응률의
분모는 **고정 관측 point**이며 invalid variation도 분모에 남는다.

| 방법 | Band normal coverage | >5° | >10° | >20° | >30° |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 96.792% | 84.939% | 81.219% | 69.489% | 54.995% |
| FD 2px | 93.330% | 83.994% | 74.949% | 54.571% | 35.621% |
| FD 4px | 85.713% | 80.479% | 69.016% | 24.398% | 2.591% |
| PCA 5×5 | 99.871% | 83.782% | 68.305% | 49.522% | 33.350% |
| PCA 7×7 | 99.968% | 85.868% | 68.957% | 44.086% | 12.015% |

위치 평가는 **원래 seam endpoint**에서 가장 가까운 threshold 반응까지의 1/2/3px
coverage와 반응→seam 거리를 함께 계산했다. 탐색 후보는 GT 가시 영역의 고정 6px
corridor 안의 추출 반응이다. 해당 frame에 반응이 없으면 coverage는 0이고 거리값은
missing이다. 평면 noise가 많은 방법은 가까운 반응을 우연히 찾기 쉬우므로 coverage
하나로 순위를 정하지 않는다.

| 방법 | 2px coverage @5° | @10° | @20° | @30° | @20°의 1 / 3px coverage | 반응→seam 거리 p50 / p95 @20° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 99.023% | 98.981% | 98.895% | 98.665% | 97.891 / 99.519% | 2.000 / 5.831px |
| FD 2px | 97.646% | 97.581% | 97.165% | 95.852% | 93.874 / 98.604% | 1.000 / 2.000px |
| FD 4px | 91.312% | 90.776% | 76.926% | 13.345% | 56.916 / 88.339% | 1.000 / 3.162px |
| PCA 5×5 | 99.611% | 99.467% | 99.011% | 93.975% | 96.874 / 99.489% | 0.000 / 2.000px |
| PCA 7×7 | 99.630% | 99.498% | 97.055% | 40.219% | 87.736 / 98.341% | 1.000 / 2.000px |

PCA5는 평면 오반응을 크게 줄이면서 실제 seam 주변 반응의 위치를 유지했다. PCA7과
FD4의 30° coverage는 크게 떨어져 모든 경계에 큰 window가 좋다고 할 수 없다.

[Frame 0 profile](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0000_profile.png)은
GT seam만으로 고른 같은 수평 25px 선이다. Peak offset은 Baseline/FD2/FD4/PCA5/PCA7
순서로 0/0/2/0/1px, >20° sample 수는 6/3/0/3/2개다. 이는 이 선에서의 threshold
초과 sample 수이며 단일 edge의 연속 FWHM이 아니다. 전체 profile·5/10/20/30° 초과 폭을
JSON에 저장했다. Frame 80의 FD2/PCA5 최대값은 다른 반응에 끌려 reference에서 12px
떨어진다. **단일 profile의 global maximum을 곧바로 seam 검출 위치로 쓸 수 없다.**

## 5. Box의 관측 가능한 물리 hole rim

GT 가시 rim은 **3,425 pixel-frame**, tracker 관측은 **2,911개(84.993%)**로 모든
방법에서 같다. 아래 union은 기존 관측 depth jump와 threshold variation의 OR이며
입구 경계의 최종 semantic 분류가 아니다.

| 방법 | Normal / 관측 rim | Normal / GT 가시 rim | Variation >20° / 관측 rim | Union @5° / GT rim | @10° | @20° | @30° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 74.442% | 63.270% | 58.399% | 83.387% | 83.387% | 69.752% | 63.241% |
| FD 2px | 52.971% | 45.022% | 50.842% | 65.139% | 64.847% | 63.328% | 54.015% |
| FD 4px | 51.941% | 44.146% | 13.294% | 64.000% | 51.912% | 31.416% | 25.985% |
| PCA 5×5 | 99.966% | 84.964% | 65.407% | 68.759% | 67.971% | 63.854% | 56.964% |
| PCA 7×7 | 100% | 84.993% | 43.559% | 69.022% | 64.642% | 45.839% | 42.832% |

PCA5는 거의 모든 **관측된** rim에서 normal을 얻지만, 고정 rim 영역의 union 반응은
Baseline보다 낮다. Normal availability 증가를 구조 검출 coverage 증가로 주장할 수 없다.
Baseline 반응 일부가 평면 quantization인지 물리 구조인지 이 band 통계만으로 완전히
분해할 수 없으므로, 감소한 반응을 모두 noise 제거 또는 모두 구조 손실로 단정하지 않는다.

Frame 0의 가시 rim 223px에서 normal은 174/125/125/223/223개이고 >20°는
139/120/32/151/99개다. Union은 188/169/81/167/117개다. FD4는 hole 인접 지원과
방향 변화 반응을 함께 크게 잃는다.

Frame 15/45/80은 기존 물리 rim ROI가 0개여서 preservation 비율을 만들지 않았다.
Frame 120은 가시 rim 51px 중 tracker가 15px만 남긴다. PCA5/PCA7은 이 15개에서
normal을 모두 얻지만 모든 방법의 union>20° 반응은 **6/51px**다. 관측에서 사라진
rim을 normal estimator가 복원하지 못한다.

## 6. Mask·occlusion 경계와 normal 유효성

아래 각 cell은 `normal coverage / variation>20° 비율`이며 **둘 다 관측 point가 분모**다.
경계 ROI는 서로 또는 실제 구조와 겹칠 수 있어 아래 값 전체를 배타적인 false positive로
간주하지 않는다. 원인별 혼동 위험과 invalid 처리 차이를 드러내는 진단값이다.

| 방법 | Peg occlusion | Peg 동일 면 tracker cut | Box occlusion |
| --- | ---: | ---: | ---: |
| Baseline | 64.408 / 7.988% | 60.614 / 7.514% | 60.976 / 13.634% |
| FD 2px | 28.924 / 1.920% | 26.334 / 2.451% | 24.652 / 0.131% |
| FD 4px | 1.897 / 0.023% | 0.014 / 0% | 0.630 / 0% |
| PCA 5×5 | 99.269 / 3.640% | 99.158 / 7.409% | 99.936 / 4.086% |
| PCA 7×7 | 99.686 / 1.394% | 99.689 / 3.489% | 99.969 / 3.863% |

FD4의 낮은 반응은 거의 모든 normal이 소실된 결과를 포함한다. PCA도 모든 threshold에서
경계 혼동을 줄이지는 않는다. 동일 면 tracker cut의 >5°/관측 point는 Baseline 18.633%,
PCA5 21.074%, PCA7 20.447%로 증가한다. Box occlusion의 >30°는
0.410%→3.231%/3.433%로 증가한다. 부분 지원의 편향·남은 경계 혼동을 최종 분류 전에
검토해야 하며, 이 집계만으로 각각의 물리 원인을 확정하지 않았다.

정확한 정면 평면에 mask cut 또는 무효 depth hole만 만든 synthetic control에서는 모든
방법의 남은 normal variation이 **0°**, jump도 0이었다. Candidate가 관측 경계를
무조건 기하 crease로 만드는 것은 아니다. FD는 normal을 더 잃고 PCA는 연결된 부분
patch로 관측된 점의 normal을 계산한다.

| 방법 | 전체 Peg normal / variation coverage | 전체 Box normal / variation coverage |
| --- | ---: | ---: |
| Baseline | 94.697 / 94.696% | 89.651 / 89.646% |
| FD 2px | 90.065 / 90.065% | 81.060 / 81.058% |
| FD 4px | 81.320 / 81.317% | 65.929 / 65.923% |
| PCA 5×5 | 99.572 / 99.572% | 98.967 / 98.967% |
| PCA 7×7 | 99.628 / 99.628% | 99.152 / 99.152% |

위 분모는 각각 동일한 관측 point 685,762 / 365,372개다. Coverage는 accuracy를 보장하지
않는다. 특히 PCA의 비대칭 부분 지원과 여러 면 혼합을 유효하다는 이유로 정확하다고
해석하지 않았다. 작은 조각·6점 미만·공선점은 invalid로 남는다.

## 7. Depth discontinuity와 variation의 상보성

Depth jump는 normal estimator와 독립적이며 모든 방법에서 기존 map과 동일하다.
관측 rim의 jump endpoint는 모두 **689개**다.

| 방법 | Variation만 >20° | Jump와 >20° 동시 | Jump만 | Union | Jump에 있지만 normal invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1700 | 0 | 689 | 2389 | 689 |
| FD 2px | 1480 | 0 | 689 | 2169 | 689 |
| FD 4px | 387 | 0 | 689 | 1076 | 689 |
| PCA 5×5 | 1498 | 406 | 283 | 2187 | 1 |
| PCA 7×7 | 881 | 387 | 302 | 1570 | 0 |

PCA는 jump 양쪽에서 각각 연결된 patch로 normal을 얻을 수 있다. 이에 따라 jump와
variation의 중복이 늘어, variation 개수가 증가해도 union이 늘지 않을 수 있다.
Normal invalid인 곳의 낮거나 missing인 variation을 표면이 매끈하다는 증거로 쓰면 안 된다.
Jump 자체도 가림·silhouette·grazing 평면을 포함하므로 semantic structural edge를 뜻하지 않는다.
모든 ROI에 대해 5/10/20/30°의 variation-only/both/jump-only/union 수를 JSON에 보존했다.

## 8. Synthetic: 정확도 개선과 sharp geometry 약화

Task 02-A의 정면/기울어진 평면, sharp crease, 50mm depth step을 그대로 재사용했다.
z0=0.2/0.25/0.35/0.6/1.0m × phase=0/0.25/0.5/0.75mm ×
float/floor/nearest × 4종의 **240조건**, 각 조건에서 다섯 방법을 동일하게 비교했다.
추가로 slope magnitude 0.2/0.8, azimuth 0/45/90/135°, z0=0.25/0.6m,
같은 세 depth 표현의 **48조건**을 비교했다. 합계 **288조건 × 5방법**이다.

아래는 z0=0.25m, phase=0.25mm, slope=(0.4,−0.25)의 고정 평면 ROI다.

| 방법 | 1mm floor error p95 | Variation p95 | >20° / 유효 variation |
| --- | ---: | ---: | ---: |
| Baseline | 25.253° | 53.592° | 89.586% |
| FD 2px | 15.357° | 25.173° | 25.768% |
| FD 4px | 6.816° | 12.384° | 0% |
| PCA 5×5 | 3.874° | 6.026° | 0% |
| PCA 7×7 | 1.526° | 1.899° | 0% |

정면 평면은 모든 방법에서 float/floor 모두 error·variation 0°다. 양자화 전 기울어진
평면은 모두 수치 오차 수준이다. 고정 slope의 floor phase sweep에서 PCA5 variation p95는
0.2m에서 7.960–8.018°, 1.0m에서 2.184–2.320°다. Baseline은 각각
61.204–61.326° / 12.561–12.566°다. Nearest의 원본 수치도 JSON에 있다.

방향에 따라 개선 정도가 다르다. 추가 z0=0.25m floor 방향 조건에서 error p95 범위는
Baseline 14.304–38.660°, FD2 10.945–14.322°, FD4 5.460–7.680°,
PCA5 3.988–11.310°, PCA7 2.168–5.685°다. PCA5가 모든 평면 방향에서 FD4보다
정확하다고 주장하지 않는다.

Sharp crease는 연속 두 평면의 실제 경계다. 기존 6열 band에서 다음과 같이 반응이 약해진다.

| 방법 | Float variation p95 | Float >5 / 10 / 20 / 30° 비율 | 1mm floor variation p95 | Floor >20° |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 33.106° | 66.667 / 66.667 / 33.333 / 33.333% | 55.565° | 97.600% |
| FD 2px | 16.952° | 100 / 66.667 / 0 / 0% | 32.896° | 51.067% |
| FD 4px | 8.566° | 99.200 / 0 / 0 / 0% | 16.060° | 0% |
| PCA 5×5 | 21.639° | 87.600 / 66.667 / 33.333 / 0% | 25.182° | 25.733% |
| PCA 7×7 | 15.765° | 100 / 66.667 / 0 / 0% | 16.836° | 0% |

분모는 동일 band의 관측 point다. FD4는 band 끝의 일부 지원도 잃는다. Normal transition이
여러 pixel에 퍼지면 같은 1px 비교에서 angle이 작아지므로, 경계가 기하적으로 모두 사라진
것과 특정 threshold 반응이 사라진 것은 구분해야 한다. Float crease에서 FD2/PCA7도
20° 반응이 0이라는 점은 양자화 제거만으로 설명할 수 없는 구조 약화다. PCA5 역시 30°
반응이 0이다. 20°를 최종 threshold로 고정할 근거는 없다.

50mm step은 모든 방법에서 jump endpoint **500개**가 유지되고 유효 normal의 variation은
0이다. 기존 1,500px band의 normal coverage는 Baseline/FD2/FD4/PCA5/PCA7 순서로
66.667/33.333/0/100/100%다. FD4 variation은 여기서 **0이 아니라 전부 unavailable**이다.
PCA는 단차를 가로지르지 않고 양쪽 평면에서 normal을 구한다. 이 경우 depth jump를 빼면
normal variation만으로 경계를 보존했다고 할 수 없다.

## 9. CPU 처리 시간

같은 macOS arm64, Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1에서 CPU로 실행했다.
전체 121프레임, 매 frame **Peg+Box 둘 다** 계산했다. 방법별 1회 warmup을 제외하고
실행 순서를 frame마다 회전했다. Backprojection은 공통으로 한 번 수행하되 각 total에
그 비용을 포함했다. 입력 로딩·GT·지표·그림·파일 저장·다른 estimator는 시간에서 제외했다.

| 방법 | Normal p50 / p95 (ms) | Feature p50 / p95 (ms) | Total p50 / p95 (ms) |
| --- | ---: | ---: | ---: |
| Baseline | 3.957 / 5.834 | 8.445 / 9.734 | 19.102 / 22.265 |
| FD 2px | 2.802 / 5.589 | 8.354 / 9.439 | 17.792 / 21.816 |
| FD 4px | 3.301 / 6.999 | 8.293 / 9.454 | 18.122 / 22.826 |
| PCA 5×5 | 21.729 / 63.083 | 8.532 / 9.904 | 36.669 / 80.030 |
| PCA 7×7 | 35.783 / 109.460 | 8.577 / 9.855 | 50.552 / 126.870 |

Backprojection p50/p95는 7.011/7.714ms다. 합계 percentile은 각 단계 percentile의 합과
같지 않다. FD 구현은 관측 중심만 순회하므로 Baseline보다 지원이 커도 이 구현에서는
조금 빠를 수 있다. PCA는 patch gathering·연결 탐색·eigendecomposition을 normal 시간에
모두 포함한다. 방법별 고도로 최적화된 구현의 이론적 속도 순위나 end-to-end tracker
FPS를 주장하지 않는다. 관측 point 수·가림에 따른 시간 차이가 섞여 있다.

## 10. 추천과 남은 실패·미검증 사항

**검토 우선 후보는 PCA 5×5**, CPU 비용을 우선하는 비교 대안은 **FD 2px**다.
PCA5는 ep000에서 평면 variation과 오차를 줄이고, 원래 관측을 유지하며, 실제 seam의
위치와 작은 crease 반응을 PCA7/FD4보다 잘 남겼다. FD2는 CPU total p95 약 21.8ms지만
정면 가까운 합성 기울기에서 양자화 오반응이 더 남고 rim normal 지원을 많이 잃는다.

PCA5 채택을 확정하지 않는다. Rim union 반응은 Baseline보다 낮고, CPU total p95는
약 80ms이며, 연속 crease를 혼합하고 일부 mask/occlusion 반응은 남거나 증가한다.
PCA7은 평면 지표가 가장 좋지만 작은 crease와 높은 threshold의 실제 seam·rim 손실이
더 크다. FD4도 low-noise만 보고 고르기에는 지원 소실과 구조 반응 약화가 크다.

미검증: 독립 episode/seed, 물리 depth 센서 noise·multipath·outlier, 다른 camera/K/해상도,
곡면·얇은 구조·gate 이하 작은 단차, temporal 안정성, 부분 지원의 normal uncertainty,
GPU/CUDA/MPS 및 PCL 구현의 성능이다. 기존 3px synthetic ROI는 큰 estimator의 전체
지원에서 항상 순수 평면을 보장하지 않는다. 원래 ROI를 그대로 보존했으므로 해당 수치는
그 고정 ROI에서의 결과이며 안전 ROI로 교체해 유리하게 만들지 않았다. 기존 depth gate의
grazing 평면 오검출 한계도 남는다.

최종 estimator·threshold 채택, PCL 설치/binding/비교는 검토 후 결정할 작업으로 남긴다.
Corner/Edge/Surface 분류, Structural/Surface Sampling, Intra-object Edge, Graph,
Stale Geometry 계약을 구현·수정하지 않았다. Simulator를 새로 실행하지 않았고 이 작업의
실행 backend는 CPU NumPy뿐이다.

## 11. 시각화·테스트·Git·보존

[Frame 0 비교](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0000_comparison.png),
[Frame 0 normal validity](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0000_validity.png),
[Frame 120 실패 예](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0120_comparison.png),
[합성 비교](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/synthetic.png).
0/15/45/80/120의 같은 frame·같은 crop에서 모든 방법을 나란히 배치했다. Plane ROI는 cyan,
실제 seam은 orange, 가시 rim은 yellow다. Variation의 NaN은 gray이며 normal validity의
dark=invalid, gray=unobserved를 별도로 표시했다. 동일한 0–45° 표시 범위를 사용하고
전체 0–180° 수치·histogram은 JSON에 보존했다.

실험은 `ExperimentLogger(team='visual', job_type='evaluate')`, 기존 UUID/project/group,
`experiment_artifacts.digests`, 로컬 manifest/sidecar를 재사용했다. `WANDB_MODE=disabled`로
수행했고 이 Task는 업로드·게시·commit·stage를 실행하지 않았다.

검증 결과:

- 새 비교 Unit Test **13/13 PASS**, 전체 runtime suite **53/53 PASS**.
- Baseline의 원래 `process_geometry` 대비 모든 feature field bitwise 동일성 테스트 PASS.
- 선택 5프레임에서 기존 Task 02-A NPZ와 현재 baseline에 대한 **260개 배열 동일성 assert** PASS.
- 같은 프레임의 기존 GT ROI와 **120개 동일성 assert** PASS. 원래 집계 JSON과 공유 지표 차이 0개.
- 현재 공통 logger/artifact 계약 테스트 **42/42 PASS**. 최초 sandbox 실행의 실제 W&B offline SDK
  1건은 `ServicePollForTokenError`로 실패했고, sandbox 밖의 단독 재실행 및 전체 재실행은 PASS다.
  실제 원격 W&B/HF 게시 검증은 아니다.
- `git diff --check` PASS. Task의 diff는 새 비교 코드·테스트·보고서만 포함한다.

시작 시 기존 파일 1,002개를 hash snapshot으로 보관했다. 이 Task가 기존 파일을 수정하거나
삭제한 것은 없다. Baseline·interface·평가 helper 및 기존 실제 데이터·결과 내용 **692개**는
hash가 그대로다(`.DS_Store` Finder metadata 제외). 작업 중 별도 커밋 `20b906c`가 공통
artifact 관련 8개 파일을 변경했고, 별도 커밋 `42e30e4`가 기존 Task 01/02-A 보고서·
`visual/.gitignore` 등을 변경했다. 이 외부 변경은 되돌리지 않았으며 전체 tree가 시작 시점과 같다고 주장하지 않는다.
세부 차이는 verification JSON과 preservation snapshot에 구분했다.

평가 시작 commit은 `ae5239bb5a0d5ed418ce8e22c434cce5e5893ef4`, dirty 상태였다.
평가 당시 source snapshot은 결과의 `evaluated_source/`에 남겼다. 이후 선택 그림의 NaN을
gray로 명확하게 표시하도록 **plotting만** 수정하고 저장 NPZ로 그림을 다시 그렸다.
Estimator·ROI·지표를 재계산하거나 threshold를 바꾸지 않았다. 평가 당시/최종 source hash를
함께 기록해 dirty tree와 이 시각화 수정의 재현성 한계를 드러냈다.

재현 명령(저장소 root):

```bash
MPLCONFIGDIR=/tmp/task02b_mpl WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_normal_comparison
MPLCONFIGDIR=/tmp/task02b_mpl WANDB_MODE=disabled .venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
WANDB_MODE=disabled .venv/bin/python -m unittest tests.test_experiment_logger tests.test_experiment_artifacts -v
git diff --check
```

재실행은 새 UUID 디렉터리를 만들며 기존 결과를 덮어쓰지 않는다.
