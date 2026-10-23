# Task 02-A — Normal Variation 신뢰성 및 Depth Quantization 영향

**완료 범위: Task 02-A만 수행.** 기존 Vision–Graph 설계와 Task 01 extractor를 보존했다.
넓은 Peg 평면의 높은 variation은 **1mm depth 양자화와 작은 normal stencil의 결합**으로
대부분 설명된다. Normal이 유효하다는 사실은 각도가 정확하다는 뜻이 아니다.

최종 run: `3640b9bbe403409ba1429e0143e5e049`.
[전체 시각화](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/diagnostics.html),
[전체 수치·설정 JSON](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/diagnostics.json),
[분포 그림](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/distributions.png),
[간격 비교 그림](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/scale_probe.png).

## 1. 입력과 변경 범위

기준은 [interface](00_interface.md), [Task 01](01_organized_geometry.md),
[`geometry.py`](../../visual/runtime/geometry.py)다. 추가한 파일은 다음과 같다.

| 파일 | 역할 |
| --- | --- |
| [`normal_diagnostics.py`](../../visual/runtime/test/normal_diagnostics.py) | 합성 관측, 분석 통계, 평가 전용 ray–cuboid GT, 영역 선택, pixel 간격 진단 |
| [`run_normal_diagnostics.py`](../../visual/runtime/test/run_normal_diagnostics.py) | 합성 paired 평가, 실제 121프레임 집계, PNG/NPZ/JSON/HTML, 공통 Logger |
| [`test_normal_diagnostics.py`](../../visual/runtime/test/test_normal_diagnostics.py) | 양자화 분리, 깊이 단차, projection, GT oracle 및 영역 선택 검증 8개 |
| 이 문서 | 방법·결과·원인별 증거·한계·후속 비교 기준 |

실제 관측은 `visual/data/step2/ep000.npz`, mask는
`visual/results/runtime_tracking/ep000/masks.npz`를 읽었다. 저장 ManiSkill 관측이며,
물리 카메라에서 새로 취득한 데이터는 아니다. `base_camera`, source 0–120 전체
**121프레임**, 512×512, `fx=fy=256`, `cx=cy=256`이다. 선택 시각화는 0/15/45/80/120이다.

- Episode SHA-256: `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`
- Mask SHA-256: `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`
- 변경하지 않은 `geometry.py`: `36d20b7873d5ef9e1381e3bbe319da89e16fa5b44455488786dadd250bd814a9`

**실제 feature 입력은 원본 depth, intrinsic, tracker mask뿐이다.** GT segmentation,
pose, 치수 및 ray intersection은 계산 이후의 영역 선택·normal 정답·depth 잔차 평가에만
사용했다. GT depth로 실제 관측을 대체하거나 GT mask로 extractor를 다시 실행하지 않았다.
RGB는 진단 그림에만 사용한다. NPZ의 GT 관련 항목은 `evaluation/` 아래에 구분했다.

`ExperimentLogger(team="visual", job_type="evaluate")`, 기존 UUID/project/group 규칙,
`experiment_artifacts.digests`, 기존 `iter_saved_rgbd`, Task 01 `process_geometry`를 재사용했다.
`WANDB_MODE=disabled`로 로컬 metadata/metrics/manifest/sidecar만 기록했다. 업로드는 없다.

## 2. 지표와 평가 영역

Normal은 완전한 동일 물체의 십자형 5점에서
`cross(X[u+1]-X[u-1], X[v+1]-X[v-1])`을 구한다. 간격은 1px, 양 끝 사이 span은 2px다.
Normal variation은 유효한 4-neighbour normal과의 **최대 oriented angle**이다.
각 normal의 계산과 variation의 전체 depth 의존 범위는 Manhattan radius 2 안에 있다.
기존 discontinuity gate `|Δz| > 0.01m + 0.02*min(z1,z2)`를 그대로 사용했다.

- Angular error: camera-facing analytic normal과의 oriented angle, degree.
- Variation: p50/p95/p99/최대 및 5° histogram, >5/10/20/30° 비율.
- 비율은 유효 variation을 분모로 한 값과 관측 point를 분모로 한 값을 함께 저장했다.
- Normal coverage, variation coverage, 관측 point coverage를 따로 저장했다.
- 경계 반응은 variation과 discontinuity를 별도로 기록했다. `jump OR >20°`는 진단용
  반응 집계이며 Structural Point Classification이나 검출기의 precision/recall이 아니다.
- 전체 분포는 pixel-frame을 합친 micro 통계다. 프레임별 percentile을 평균하지 않았다.
  시간상 반복되는 동일 표면은 독립 표본이 아니며 통계적 신뢰구간을 주장하지 않는다.
- 20°는 Task 01의 표시 기준을 이어받은 진단값이다. 최종 edge threshold로 선택하지 않았다.

### 평면 내부 선택 근거

저장 Peg는 half extents `[0.10695254, 0.02215189, 0.02215189]m`의 cuboid다.
설치된 ManiSkill 3.0.1의
`mani_skill/envs/tasks/tabletop/peg_insertion_side.py`에서 collision cuboid와 두 visual half-box가
같은 평면들을 구성함을 확인했다. 색이 바뀌는 head/tail 접합선은 기하 edge로 취급하지 않는다.

카메라 ray와 저장 pose의 cuboid를 교차시켜 첫 표면의 signed face와 normal을 구했다.
평면 후보는 **GT Peg 관측이며, 관측 z가 analytic z와 1.5mm 이내로 일치하고,
다른 cuboid 면에서 4mm 이상 떨어진 곳**이다. 이후 signed face별로 3px erosion을
적용했다. Normal이나 variation의 값으로 ROI를 선택하지 않았다. Raster erosion은
평가용 ROI에만 적용하며 extractor 입력 mask는 바꾸지 않았다. 면별 ROI와 face ID를 NPZ에 저장했다.

GT segmentation 내부라는 조건만으로 평면을 가정하지 않았다. 실제 전체 GT Peg 관측의
analytic depth 일치율은 프레임마다 최소 99.959%였다. 1.5mm는 1mm 버림 오차와 저장 pose의
수치 오차를 허용하는 평가 tolerance이며, feature를 수정하는 threshold가 아니다.

### 구조·가림·mask 경계 구분

- **면 접합선:** 인접한 두 GT 가시 표면의 analytic face가 다른 곳의 양쪽 endpoint 및 2px band.
- **물체 silhouette:** analytic object hit/miss 경계 중 실제로 GT 가시인 곳의 2px band.
- **Occlusion:** object ray hit는 있으나 다른 actor가 더 가까운 depth를 차지한 곳에 인접한
  가시 물체 pixel의 2px band. 무효 depth는 별도 진단이다.
- **동일 면 tracker cut:** 같은 analytic face의 두 GT 가시 pixel 사이에서 tracker mask만 끝나는 곳의 2px band.
- **오염:** tracker의 유효 point 중 GT 물체 밖인 곳. 해당 geometry의 종류를 추측해 재분류하지 않았다.
- **Hole rim:** 설치 소스의 `_build_box_with_hole`과 동일한 네 cuboid로 Box GT를 구성했다.
  저장 hole pose/radius로 입구의 물리적인 네 선분을 정하고, 첫 표면의 3D 선분 거리 ≤4mm와
  투영 선분의 2px band를 모두 만족하는 GT 가시 표면만 선택했다. Hole 축은 local x,
  aperture는 local y/z의 사각형이다. 가려진 선분의 투영만으로 rim이라고 하지 않는다.

2px band끼리는 겹칠 수 있다. 원인별 수치를 더해 총합이나 배타적인 기여율로 해석하지 않는다.

## 3. 검증 1 — Synthetic depth 양자화 분리

[합성 비교 그림](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/synthetic.png).
모든 paired 비교에서 camera, mask, geometry, feature parameter를 고정하고 **depth만** 바꿨다.
A는 mm 양자화가 없는 float32 depth다. XYZ/normal 저장 정밀도는 baseline의 float32를 유지했다.
B는 A를 float64 산술로 `floor(1000*z)/1000`한 뒤 float32로 저장했다.
`rint` 반올림 방식도 같은 조건으로 비교했다. Noise, smoothing, outlier filter는 없다.

중앙 256×256 mask를 쓰고 3px 내부 ROI를 평가했다. 중심 깊이 0.25025m에서:

- 정면 평면: `z=z0`.
- 기울어진 평면: `z=z0+0.4x−0.25y`; ray와 평면을 정확히 교차시켜 depth 생성.
- Sharp crease: 좌/우 slope `(-0.6,0.15)` / `(0.6,0.15)`인 연속 두 평면.
- 깊이 단차: 정면 평면 두 개의 z 차이 50mm. 동일 물체 mask로 유지했다.

### 평면 내부의 paired 수치

각 표의 비율은 유효 variation 기준이다. Crease/step은 접합선 ±3px를 평면 ROI에서 제외했다.
정면/기울어진 평면은 62,500px, crease/step의 평면 내부는 61,000px다.

| Geometry | 조건 | Normal error p95 (°) | Variation p50 / p95 (°) | >20° | Normal 유효율 |
| --- | --- | ---: | ---: | ---: | ---: |
| 정면 평면 | A float | 0 | 0 / 0 | 0% | 100% |
| 정면 평면 | B 1mm floor | 0 | 0 / 0 | 0% | 100% |
| 기울어진 평면 | A float | 0.000569 | 0.000543 / 0.000967 | 0% | 100% |
| 기울어진 평면 | B 1mm floor | 25.253 | 33.547 / 53.592 | **89.586%** | 100% |
| Crease의 두 평면 내부 | A float | 0.000489 | 0.000475 / 0.000883 | 0% | 100% |
| Crease의 두 평면 내부 | B 1mm floor | 17.244 | 23.127 / 29.176 | **58.016%** | 100% |
| Depth step의 두 평면 내부 | A float | 0 | 0 / 0 | 0% | 100% |
| Depth step의 두 평면 내부 | B 1mm floor | 0 | 0 / 0 | 0% | 100% |

위 평면 ROI의 discontinuity는 모든 조건에서 0개다. 1mm 양자화가 항상 normal variation을
만드는 것은 아니다. 정면 평면에서는 모든 pixel의 z가 함께 이동해 normal이 유지된다.
기울어진 평면에서는 계단형 depth의 2px 차분이 위치에 따라 달라져 normal과 variation이 흔들린다.

### 구조 경계 band의 paired 수치

중앙 선분 길이 250px, 6열 band 총 1,500px다. Normal error는 유효한 normal만 대상으로 한다.

| Geometry / 조건 | Error p95 (°) | Variation p95 (°) | >20° / 유효 variation | Normal 유효율 | Jump endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Crease / float | 14.132 | 33.106 | 33.333% | 100% | 0 |
| Crease / 1mm floor | 31.735 | 55.565 | 97.600% | 100% | 0 |
| 50mm step / float | 0 | 0 | 0% | 66.667% | 500 |
| 50mm step / 1mm floor | 0 | 0 | 0% | 66.667% | 500 |

Float crease도 5점 stencil이 두 면을 가로지르는 위치에서는 혼합 normal을 낸다.
Variation 반응은 남지만 실제 dihedral을 한 pixel에서 그대로 반환하는 estimator가 아니다.
양자화 후에도 crease 반응은 남지만, 평면 내부의 높은 반응 때문에 대비와 위치 해석이 나빠진다.
Step에서는 양쪽 두 열의 normal이 모두 무효이며, jump endpoint는 정확히 보존된다.
이 경우 variation이 0인 유효 주변점만 보고 경계가 사라졌다고 판단하면 안 된다.

### 깊이와 quantization phase sweep

중심 깊이 0.2/0.25/0.35/0.6/1.0m, z offset 0/0.25/0.5/0.75mm,
4종 geometry, A/1mm floor/1mm nearest의 총 **240조건**을 계산했다.
기울어진 평면의 phase별 floor 결과 범위는 다음과 같다.

| 중심 깊이 (m) | Variation p95 범위 (°) | 평면 내부 >20° 범위 |
| --- | ---: | ---: |
| 0.20 | 61.204–61.326 | 93.744–93.904% |
| 0.25 | 53.489–53.592 | 89.323–89.586% |
| 0.35 | 40.426–40.559 | 61.614–61.856% |
| 0.60 | 22.201–22.230 | 10.374–10.475% |
| 1.00 | 12.561–12.566 | 0% |

Nearest 방식도 0.25m의 >20° 비율이 89.120–89.416%였다. 버림 방식 특유의 bias만으로
발생하는 현상은 아니다. 정면 평면은 모든 깊이/phase/양자화 방식에서 variation 0이었다.
이 sweep은 고정 slope/K의 결과이며 모든 방향·곡률·노이즈에 대한 일반적인 깊이 법칙은 아니다.

## 4. 검증 2 — 실제 Peg 평면과 전체 분포

### 선택 프레임의 평면 내부

GT로 확인한 평면 ROI에서 추출한 관측 feature다. Normal error는 analytic face normal 기준이다.
`point / normal`은 tracker support를 통과한 점과 유효 normal 수다.

| Source | Point / normal | Error p95 (°) | Variation p50 / p95 (°) | >20° / 유효 variation | Depth median (m) | z/fx (mm/px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6592 / 6592 | 20.591 | 9.334 / 32.232 | 26.578% | 0.212 | 0.828 |
| 15 | 6452 / 6452 | 20.591 | 9.392 / 32.232 | 27.154% | 0.210 | 0.820 |
| 45 | 2962 / 2962 | 23.934 | 22.479 / 32.405 | 58.440% | 0.186 | 0.727 |
| 80 | 929 / 922 | 18.954 | 24.356 / 31.709 | 70.499% | 0.286 | 1.117 |
| 120 | 359 / 358 | 11.992 | 17.211 / 24.954 | 10.335% | 0.396 | 1.547 |

전체 121프레임의 Peg 평면 내부는 아래와 같다.

- GT 평가 ROI 342,378 pixel-frame, 관측 point 342,257개, 유효 normal/variation 342,079개.
- 관측 point 중 normal 유효율 **99.948%**, normal error p95 **23.564°**.
- Variation p50 **14.677°**, p95 **33.547°**, p99 **36.403°**, 최대 **47.572°**.
- >5/10/20/30° 비율: **66.293 / 60.138 / 42.123 / 21.490%**.
- 전체 Peg 관측의 variation은 p50 19.452°, p95 42.721°, >20° 320,395/649,389개(49.338%).
  확인된 평면 ROI의 >20° 144,093개만으로 전체 Peg의 높은 반응 중 약 45.0%를 차지한다.
- 전체 Box 관측의 variation p95는 22.668°, >20°는 20.178%다.

### 면 방향에 따른 차이

Signed face 번호는 `2*axis + positive_side`다. 0은 local −x 끝면, 2/3은 local −y/+y 측면,
5는 local +z의 긴 면이다. Frame 80에서는 다음처럼 같은 이미지의 평면도 크게 다르다.

| Frame 80의 확인된 평면 | Point / normal | Variation p50 / p95 (°) | >20° |
| --- | ---: | ---: | ---: |
| −x 면 | 654 / 654 | 25.019 / 31.884 | 99.388% |
| +y 면 | 13 / 6 | 2.595 / 2.705 | 0% |
| +z 긴 면 | 262 / 262 | 0.407 / 0.739 | 0% |

Frame 0의 +z 긴 면은 4,161px에서 variation median 0.484°이나 p95 32.356°,
>20° 15.189%다. 넓은 면에서도 terrace 전환 위치에 큰 줄무늬 반응이 집중된다.
[Frame 0 profile](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/f0000_profile.png)과
[Frame 80 profile](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/f0080_profile.png)은
ROI 면적이 가장 큰 signed face를 고른 뒤 더 긴 raster 방향의 선을 선택했다.
선 선택에 normal/variation을 쓰지 않았다. 서로 다른 프레임의 비율 변화에는 깊이뿐 아니라
면 방향, quantization phase, 가림으로 달라진 면 구성도 섞여 있다.

### Depth 양자화와 실제 pixel scale의 연결

평가에 일치한 GT 가시 표면에서 관측 z는 analytic z보다 대체로 0–1mm 낮다.
Analytic z를 1mm로 버림한 뒤 float32로 바꾼 값과 **bitwise 같은 관측 depth**의 비율은
Peg 681,313/681,890개(**99.915%**), Box 359,696/359,893개(**99.945%**)다.
관측값의 1mm 격자 잔차도 JSON에 저장했다. 설치된 `mani_skill/render/shaders.py`의
기본 position transform은 mm 변환 후 int16 cast를 수행한다. Minimal shader도 mm 정수
position을 반환한다. 저장 파일에는 shader 설정이 없어 당시 경로를 단정하지 않는다.
버림 방식 자체는 analytic 관측 잔차와 직접 대조했다.

더 엄격하게, 평면 ROI의 각 variation에 필요한 **radius 2의 모든 depth가** analytic
cuboid의 1mm floor 값과 bitwise 같은지 검사했다. 유효 variation 342,079개 중
339,419개(99.222%), >20° 144,093개 중 **143,242개(99.409%)**가 이 조건을 만족했다.
이들은 실제 sharp edge나 가림을 가로지르지 않는 동일 평면이며, 양자화된 평면 관측만으로
해당 입력 neighbourhood가 재현된다. GT depth를 feature 입력에 넣어 계산한 결과가 아니다.
나머지 불일치 pixel의 원인은 확정하지 않았다. 저장 pose/renderer의 수치 차이가 quantization
bin 경계에서 영향을 줄 가능성은 추정이다.

고정 fx에서 같은 z의 1px 횡방향 간격은 `z/fx`다. 1mm 양자화의 두 endpoint 오차가
2px depth 차분에 미치는 규모는 대략 `q/(2*z/fx)`로 볼 수 있다. Frame 0의 median z에서
이 비는 약 0.604다. 각 방향의 실제 3D secant 길이는 표면 기울기에 따라 더 길다.
원본 NPZ로 별도 확인한 frame 0의 2px secant 길이 median은 u 2.016mm/v 2.337mm,
frame 120은 u 3.117mm/v 3.169mm였다. `z/fx`를 실제 표면 위의 거리와 동일시하지 않는다.

## 5. 검증 3 — 작은 neighbourhood와 원인 분리

원본 raster를 `[::s,::s]`, s=1/2/4로 읽고 intrinsic을
`f'=f/s`, `c'=(c−0.5)/s+0.5`로 바꿨다. 동일 source pixel의 XYZ가 bitwise 같음을 테스트했다.
평균·보간·smoothing 없이 **동일 baseline extractor**의 물리적 지원 간격만 바꾼 진단이다.
최종 pipeline의 sampling이나 새 normal estimator를 구현하지 않았다.

모든 stride에서 radius 8을 확보한 동일한 안전 평면 영역을 사용했다. Stride마다 선택된
pixel 위치가 다르므로, JSON의 `baseline_same_centres`에서 그 위치의 s=1 값과도 직접 비교한다.
간격 확대는 normal stencil span과 variation 비교 거리 둘 다 바꾼다. 이상적인 variation이
0인 평면에서만 해석하며, 곡면·edge 보존 성능 개선으로 일반화하지 않는다.

| Source | s=1/2/4의 point 수 | Normal error p95 s=1→2→4 (°) | Variation p95 s=1→2→4 (°) | s=4 위치의 원래 s=1 variation p95 (°) |
| --- | --- | --- | --- | ---: |
| 0 | 3482 / 875 / 223 | 22.304 → 6.885 → 4.937 | 31.898 → 13.433 → 8.679 | 31.968 |
| 15 | 3394 / 852 / 217 | 22.304 → 7.184 → 4.955 | 31.898 → 13.433 → 8.679 | 32.021 |
| 45 | 1450 / 359 / 90 | 25.483 → 8.694 → 5.850 | 32.356 → 13.443 → 9.089 | 32.356 |
| 80 | 265 / 63 / 16 | 18.954 → 8.683 → 0.625 | 31.525 → 17.150 → 0.222 | 26.021 |
| 120 | 49 / 12 / 4 | 11.841 → 8.982 → 3.824 | 24.932 → 13.042 → 6.258 | 24.047 |

Frame 80/120의 s=4는 16/4점뿐이라 특정 lattice의 aliasing·표본 수에 민감하다.
이를 일반적인 성능으로 주장하지 않는다. 0/15/45 및 별도의 합성 slope/depth sweep에서도
간격 확대 시 error와 variation이 줄어 작은 stencil의 증폭 효과가 확인됐다.
새 estimator의 채택이나 최적 간격 결정은 이번 결과에 포함하지 않는다.

### 원인별 증거 수준

| 원인 | 증거 | 판단 |
| --- | --- | --- |
| Depth quantization | 합성 paired 실험에서 depth만 양자화해 평면 >20° 0→89.586%; 실제 높은 평면 반응의 99.409%는 전체 depth neighbourhood가 quantized GT plane과 일치 | **확인된 주원인** |
| 작은 neighbourhood | 같은 raw depth와 같은 baseline에서 간격 확대, 동일 위치 비교 시 error/variation 감소 | **확인된 증폭 요인**; 양자화와 독립적인 기여율로 나누지는 않음 |
| 실제 sharp edge | analytic face 접합선 band의 별도 높은 반응, float 합성 crease도 반응 | **확인됨**; 평면 내부 반응의 원인은 아님 |
| Tracking mask 오염 | 전체 Peg >20° 중 GT 밖 1,820개(0.568%); frame 120에서는 52/273개(19.048%) | **후반부의 국소적 기여 확인**; 전체 현상의 주원인으로 설명할 수 없음 |
| Mask/occlusion/invalid depth의 불완전 지원 | 평평한 mask-cut 합성 control은 variation/jump를 만들지 않고 normal만 소실; 실제 normal unavailable은 기존 지원·gate 규칙으로 모두 설명 | **지원 소실 확인**; 가림 경계를 물체 구조로 확정할 수 없음 |

전체 Peg의 normal unavailable 36,369개, Box 37,813개에서 설명되지 않은 점은 0개다.
원인은 중복 집계이며 mask/invalid-depth/overlap/depth-jump/image 순서로
Peg `30,000 / 0 / 2,694 / 20,826 / 0`, Box `28,134 / 1,875 / 2,599 / 24,871 / 0`이다.
단순 mask-cut과 invalid-depth-hole 합성 control의 남은 normal은 error/variation 0이었다.
지원이 불완전하면 이 estimator는 불완전 stencil을 계산에 섞지 않고 NaN으로 제외한다.
그러나 남은 normal은 1–3개의 이웃 normal만으로 variation을 낼 수 있어, 경계에서 보이는
낮은 variation을 정상 평면의 증거로 해석하면 안 된다.

## 6. Hole rim과 구조 경계 보존 상태

[Frame 0 경계 확대](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/f0000_boundaries.png),
[Frame 120 경계 확대](../../visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/f0120_boundaries.png).
모든 선택 프레임에는 같은 형식의 전체 맵, 경계 확대, depth/normal/variation profile이 있다.
Variation/error heatmap은 0–45°로 고정했다. Histogram 그림은 0–90°를 확대하며,
JSON에는 0–180° histogram과 최대값을 남겼다. NaN은 유효한 0°로 채우지 않았다.

| 실제 평가 영역, 전체 121프레임 | 관측 point | Normal / 관측 point | Variation p95 (°) | >20° / 유효 variation | Jump endpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peg 확인된 평면 내부 | 342257 | 99.948% | 33.547 | 42.123% | 5 |
| Peg 면 접합선 2px band | 94052 | 96.792% | 76.739 | 71.792% | 1787 |
| Peg occlusion 2px band | 17501 | 64.408% | 42.158 | 12.402% | 2246 |
| Peg 동일 면 tracker cut 2px band | 20893 | 60.614% | 46.200 | 12.397% | 3024 |
| Box 물리적 hole rim | 2911 | 74.442% | 82.292 | 78.449% | 689 |
| Box occlusion 2px band | 39063 | 60.976% | 21.912 | 22.361% | 10634 |

Hole rim의 **GT 가시 평가 영역 전체는 3,425 pixel-frame**이다. Tracker가 2,911개(84.993%)를
관측하고, normal은 2,167개(전체 가시 rim의 63.270%)에 남았다.
`jump OR >20°` 반응은 2,389개로 가시 rim 전체의 69.752%, 관측 rim의 약 82.1%다.
관측 rim에서 normal이 없는 744개 중 689개에는 jump가 남는다.

- **Frame 0:** 가시 rim 223px 전부 tracker에 있고 normal 174개, >20° 139개, jump 49개다.
  Rim의 방향 변화와 깊이 단절이 서로 다른 feature로 남는다.
- **Frame 15/45/80:** 위 기준으로 확인할 수 있는 가시 입구 rim은 0개다. 분모가 없으므로
  preservation 비율을 계산하지 않았다. 손/물체가 가린 입구의 투영선을 feature 정답으로 취급하지 않는다.
- **Frame 120:** 가시 rim 51px 중 tracker가 15개만 남기며 normal 5개, >20° 0개, jump 6개다.
  이 시점의 rim 보존은 불완전하다. Normal estimator만의 손실과 tracker 관측 소실을 구분해야 한다.

Peg 면 접합선은 평면보다 큰 variation을 보이지만 분포가 겹친다. Variation 단독의
높음/낮음으로 구조 경계를 안정적으로 분류할 근거는 부족하다. Gripper occlusion에도
jump/mask 반응이 나타나므로 discontinuity 역시 structural edge의 의미를 보장하지 않는다.

평면 내부의 jump 5개는 grazing 측면의 기존 gate 한계를 보여준다. 예를 들어 frame 120
`(u,v)=(254,216)`, local +y 면의 관측 z=0.478m에서 인접 변화가 21.000mm로
19.56mm gate를 넘는다. 동일한 analytic 평면이고 내부 ROI인데도 normal이 무효다.
연속 표면의 큰 z 기울기와 실제 단차를 이 gate만으로 구별할 수 없다.

## 7. 현재 한계와 후속 estimator의 공통 평가 기준

현재 방식은 **양자화 없는 평면 normal, 일정 깊이의 정면 표면, gate보다 큰 depth 단차의
위치 보존**에는 신뢰할 수 있다. 기울어진 mm 양자화 표면, 접합선을 가로지르는 stencil,
grazing 표면, mask/occlusion/무효 depth 인접 및 작은 조각에서는 normal 정확도나 지원이 불안정하다.
Normal validity, angular accuracy, variation validity를 독립적으로 확인해야 한다.

이 결과는 한 episode의 저장 simulator RGB-D와 명시한 analytic geometry에 한정된다.
곡면, 실제 센서 noise/multipath, 다양한 camera/해상도/seed, 작은 깊이 단차의 정밀도는
이번 비교에서 검증하지 않았다. 양자화 전 실제 renderer depth가 저장되어 있지 않으므로
실제 원본 float depth를 직접 복구해 비교한 실험은 아니다. 실제 근거는 analytic GT와의
관측 일치 및 독립 합성 paired 실험이다.

향후 PCL/다른 normal estimator 비교에서는 다음 기준을 동일하게 적용한다.

1. **입력·좌표 고정:** 같은 episode/source/mask/hash, CV z metre, `(u+.5,v+.5)`, intrinsic,
   invalid/overlap policy와 GT 분리를 유지한다. Point order가 바뀌면 원본 pixel provenance를 보존한다.
2. **동일 위치 + 전체 지원을 함께 보고:** 두 estimator 모두 normal이 있는 pixel의 paired error와
   각 estimator의 전체 관측점 coverage를 함께 제시한다. 어려운 pixel을 제외해 얻은 낮은 error만 비교하지 않는다.
3. **정확도와 평면 오반응:** oriented error p50/p95/p99, variation 분포 및 >5/10/20/30° 비율,
   normal/variation/missing-neighbour coverage를 동일 ROI에서 산출한다. 이 진단의 ROI와 threshold를
   결과에 맞춰 다시 선택하지 않는다.
4. **경계 보존과 혼동:** analytic face seam, 물리 rim, silhouette, occlusion, mask-only cut,
   invalid-depth 경계를 구분한다. 경계 profile의 peak 위치·폭, 1/2/3px tolerance의 반응 coverage,
   평면 false response 및 jump와 normal 소실을 함께 비교한다. 이번 2px band 반응률을 최종 검출 recall로 부르지 않는다.
5. **지원 크기 고정·공개:** pixel/metric radius, k, 최소 이웃 수, 방향 정렬, jump 처리,
   prefilter 유무와 plane/edge를 가로지르는 지원 여부를 기록한다. 큰 neighbourhood의 낮은 noise와
   얇은 rim/crease의 뭉개짐을 동시에 평가한다. 이 보고서의 stride 진단은 채택할 estimator가 아니다.
6. **깊이·방향·phase 제어:** 이번 depth/phase floor/nearest sweep을 반복하고, 추가 slope/azimuth,
   작은 단차 및 곡면은 별도 조건으로 확장한다. Curvature에 의한 정상 variation을 평면 오반응과 구분한다.
7. **조정/검증 분리:** 이번 ep000은 진단용이다. Parameter 선택용 조건과 별도 seed/episode의
   최종 검증을 분리하고, 연속 프레임을 독립 샘플처럼 취급하지 않는다.
8. **실행 비용·실패 보고:** 같은 host/device에서 입력 변환·search 구조·normal·feature 계산과
   전체 시간을 분리한다. 미지원 backend, normal 방향, 빈 mask/좁은 표면의 실패를 숨기지 않는다.

PCL 설치·binding·비교, smoothing, 새 production normal estimator, Structural Point Classification,
Sampling, Intra-object Edge, Vision Output, Graph 변경은 수행하지 않았다.

## 8. 재현·검증·Git 보존

Repository root의 `.venv`, Python 3.11.13 / NumPy 2.4.6, macOS arm64 CPU NumPy로 실행했다.
새 dependency/lock 변경은 없다.

```bash
WANDB_MODE=disabled .venv/bin/python visual/runtime/test/run_normal_diagnostics.py \
  --input visual/data/step2/ep000.npz \
  --masks visual/results/runtime_tracking/ep000/masks.npz \
  --out visual/results/normal_diagnostics

.venv/bin/python -m unittest visual.runtime.test.test_normal_diagnostics -v
.venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
git diff --check
```

8개 새 테스트와 기존 32개를 포함한 **40/40 PASS**, FAIL 0 / SKIP 0이다.
새 테스트는 다음을 확인한다.

- Quantization만으로 tilted plane의 오반응 발생, normal coverage/입력 불변.
- Quantized front plane의 error/variation 0.
- Depth step의 정확한 500 jump endpoint와 66.667% 경계 band normal coverage.
- Stride 2/4의 동일 source pixel CV XYZ bitwise 동일성.
- Parallel ray의 miss 및 signed cuboid face.
- Analytic pose/normal과 평면 ROI가 측정 variation에 의존하지 않음.
- 네 cuboid의 열린 hole, 물리 rim과 가려진 rim의 제외.
- 평면 mask-cut은 variation/jump를 만들지 않으며 지원 소실 이유가 모두 설명됨.

추가 산출물 검증:

- Source row가 중복/누락 없이 정확히 0–120임을 확인했다.
- Task 01의 같은 5개 NPZ와 **130개 feature 배열을 NaN까지 포함해 bitwise 비교해 전부 동일**했다.
- 최종 JSON의 diagnostic source SHA-256을 실제 3개 코드 파일과 대조했다.
- 최종 산출물의 모든 tracking sidecar UUID와 공통 Logger의 `status=finished`를 확인했다.
- 작업 전 스냅샷의 기존 **585개 비무시 파일의 SHA-256이 전부 동일**했다.
  원본 episode/mask의 hash도 Task 01 기록과 동일하다.
- 합성·실제 PNG, 경계 확대, profile, 분포와 scale 그림을 직접 열어 축·범위·NaN 표시를 확인했다.

Simulator rollout, 재렌더링, SAM2/Grounding DINO 재추론, CUDA/MPS/GPU,
Linux/backend 실행, PCL, W&B online/HF 게시: **NOT RUN**.
설치 package의 geometry 구성은 소스로 확인했으며 새 simulator 실행을 주장하지 않는다.

시작/종료 branch는 `visual-graph`, commit은 `ae5239bb5a0d5ed418ce8e22c434cce5e5893ef4`다.
시작부터 있던 tracked 변경은 그대로 보존했다.

```text
 M docs/experiment_tracking.md
 M visual/runtime/README.md
 M visual/runtime/tracking.py
 M visual/scripts/collect.py
```

`git diff --stat`도 시작과 같은 **4 files, 32 insertions, 8 deletions**다. 기존 Task 01과
RGB-D/초기화/tracking의 미추적 파일도 보존했다. 이번 추가는 위 3개 진단 코드, 이 문서,
`visual/results/normal_diagnostics/`다. Code/docs/PNG/JSON/HTML은 아직 untracked이며,
raw NPZ·sidecar·`experiments/<run_id>/`는 기존 ignore 규칙을 따른다. Git add/commit/reset은 하지 않았다.

시각화 확인 과정의 선행 run `202928e0c70146c6aaa3d7fd10d4fee5`,
`30ad334f3a094b91b54cdf89ac276b45`도 별도 UUID에 보존했다. 모든 수치 해석과 위 링크는
최종 run `3640b9bbe403409ba1429e0143e5e049` 기준이다. 기존 결과 파일을 덮어쓰지 않았다.

**Task 02-A에서 중단했다.** 후속 estimator 비교나 구현은 진행하지 않았다.
