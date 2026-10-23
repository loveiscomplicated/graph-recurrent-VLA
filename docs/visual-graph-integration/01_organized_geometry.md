# Task 01 — Organized Geometry 및 Geometric Feature 후보 추출 완료 보고

`00_interface.md`를 기준으로 Task 01 범위만 구현했다. CPU NumPy로 원본 pixel에
정렬된 geometry·normal·관측 경계·normal variation을 계산한다. 최종 feature 분류,
sampling, Graph edge/Tensor, cache, GNN/GRU는 구현하지 않았다.

## A. 변경 코드와 재사용

기존 파일을 수정하지 않고 다음 파일을 추가했다.

| 파일 | 역할 |
| --- | --- |
| [`visual/runtime/geometry.py`](../../visual/runtime/geometry.py) | OrganizedGeometry와 pixel-aligned feature, 계산 함수, 단계별 시간 |
| [`visual/runtime/geometry.md`](../../visual/runtime/geometry.md) | API, shape·단위·좌표·validity, 재현 명령과 한계 |
| [`visual/runtime/test/test_geometry.py`](../../visual/runtime/test/test_geometry.py) | 합성 geometry 15개 수치 테스트 |
| [`visual/runtime/test/run_geometry.py`](../../visual/runtime/test/run_geometry.py) | 저장 관측 전체 평가, 9-panel PNG·dense NPZ·JSON·HTML, 공통 로깅 |
| [`visual/runtime/test/test_geometry_evaluation.py`](../../visual/runtime/test/test_geometry_evaluation.py) | sparse source row·빈 mask·저장 alignment·실행 identity 2개 테스트 |
| 이 문서 | 실제 결과, 실패 사례, 수치 검증과 Git 보존 기록 |

함수별 역할은 [API 표](../../visual/runtime/geometry.md#함수와-결과)에 정리했다.
`build_organized_geometry` → `estimate_surface_normals` → `compute_geometric_features`를
`process_geometry`가 호출한다. 모든 feature는 원본 `[H,W]`이며 XYZ/normal만 마지막
축이 3이다. Flat index `v*W+u`로 함께 선택할 수 있다.

재사용한 것은 `TrackedRGBDFrame`, `iter_saved_rgbd`, `PEG=0`/`BOX=1`, 기존
backprojection의 pixel-centre 식, 시각화의 `pointcloud.to_world`, 공통
`ExperimentLogger`, `experiment_artifacts.digests`다. Legacy backprojection은 compact
point만 반환하고 무효 finite 검사를 하지 않아, 같은 식을 NumPy organized 배열에
적용하고 기존 함수와 일치하는지 테스트했다. Runtime에 legacy plotting 모듈을 import하지 않는다.

`extract()`의 erosion/random pre-selection/SOR/FPS를 호출하지 않는다. Geometry는
모든 finite positive depth를 보존한다. Tracker mask 중첩은 원본을 유지하고 양쪽
물체의 지원 영역에서만 제외한다. 배경 XYZ도 남으므로 물체 point를 선택할 때는
반드시 `valid_points`를 사용한다. Invalid XYZ/normal/variation은 NaN과 validity로
명시한다. 빈 mask는 현재 관측 부재이며 이전 geometry를 가져오지 않는다.

기존 E9, pose fitting, rule-based policy, Graph prototype, RGB-D adapter,
tracking, simulator, 공통 logger/publisher, dependency/lock 파일을 변경하지 않았다.
새 경로는 명시적으로 호출할 때만 실행된다. RGB는 PNG 디버깅용으로만 사용하며
geometry 결과·NPZ·node feature에 포함하지 않는다. 원격 업로드는 수행하지 않았다.

## B. Unit Test — 모두 PASS

실행 환경: repository root `.venv`, Python 3.11.13, NumPy 2.4.6, macOS arm64.

```bash
.venv/bin/python -m unittest visual.runtime.test.test_geometry -v
.venv/bin/python -m unittest visual.runtime.test.test_geometry_evaluation -v
.venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
git diff --check
```

Geometry 15개 PASS, runner 2개 PASS. 최종 전체 runtime suite는 **32/32 PASS**
(기존 초기화/RGB-D/평가 테스트 15개 포함), FAIL 0, SKIP 0이다.
아래 테스트 이름은 `test_` prefix를 생략했다.

| 테스트 | 결과 | 기대값 / 실제 검증 |
| --- | --- | --- |
| `exact_backprojection_and_legacy_equivalence` | PASS | 비대칭 K·z=2m; 최대 XYZ 절대 오차 `2.3841858e-8m` (`atol=1e-7`); legacy point와 bitwise 동일, 기존 world 변환도 검증 |
| `frontoparallel_plane_normals_and_no_spurious_boundaries` | PASS | 9×11 평면 내부 normal 63개 모두 `[0,0,-1]`; jump/mask/무효-depth boundary 0, variation 정확히 0; image boundary 36개 별도 |
| `sloped_plane_analytic_normal` | PASS | `z=1+0.2x−0.1y`; 정규화된 `[0.2,−0.1,−1]` 대비 최대 성분 오차 `6.3027842e-6` (`atol=1e-5`); 최대 variation `1.242751e-5rad` (`<2e-5`); jump 0 |
| `depth_step_marks_exact_pair_endpoints_and_blocks_normal_bridge` | PASS | 9×12, 1→1.2m 단절; 양쪽 endpoint 정확히 18개, 실제 차이 `0.2000000477m` (`atol=6e-8`); 단절을 잇는 normal 0 |
| `jump_threshold_uses_metres_relative_depth_and_strict_comparison` | PASS | 정확히 표현되는 0.25m 차이·0.25m gate에서 0개; gate를 0.1875m로 명시한 별도 case에서 10개 |
| `invalid_depth_and_neighbours_never_produce_normals_or_points` | PASS | NaN/±Inf/0/−1 각각 중심 point 제외, 인접 validity boundary 정확히 4개, normal 49→44; 무효값을 depth jump로 오인하지 않음; NumPy floating error도 raise로 검사 |
| `mask_cut_on_same_plane_is_observation_boundary_only` | PASS | 평면의 5×5 mask는 관측점 25개 보존, mask boundary 16개, normal 9개, jump 0 |
| `two_object_normals_do_not_mix_even_below_jump_gate` | PASS | 작은 depth 차이의 기울어진 Peg/평면 Box에서도 서로 다른 normal을 섞지 않음; 지원 영역 교집합 0 |
| `depth_jump_across_objects_is_reported_separately_from_mask_boundary` | PASS | 1→2m 물체 접점에서 각 물체 쪽 7개 endpoint; normal은 경계를 넘지 않음 |
| `overlap_is_preserved_but_excluded_from_both_object_supports` | PASS | 중첩 9개를 원본 mask에 보존하되 양쪽 지원·인접 normal stencil에서 제외 |
| `empty_mask_all_invalid_and_tiny_images` | PASS | 1×1/1×5/5×1/2×2/7×9, 유효/무효 depth 및 빈 mask; 가짜 point/normal/feature 없음 |
| `isolated_normal_has_no_variation_support` | PASS | normal 1개, 이웃 normal 0개이면 variation NaN/invalid |
| `analytic_curved_surface_has_variation_without_classification` | PASS | 연속 paraboloid의 중앙 기대각 `0.0997628270065rad`, 실제 `0.0997628271580rad`, 오차 `1.5146683e-10rad` (`delta=1e-6`); jump 0; edge class 생성 안 함 |
| `flat_index_alignment_metadata_and_inputs_unchanged` | PASS | 같은 raster index의 XYZ·normal·variation·boundary 정확히 일치; source metadata 일치, 입력 불변, RGB 미포함 |
| `malformed_geometry_inputs_raise` | PASS | 정수 depth, 잘못된 K·ID·mask shape/dtype 및 음수/비유한 threshold 거부 |
| Runner `sparse_sources_and_empty_mask_preserve_dense_features_without_rgb` | PASS | source `[0,2]`/step `[10,12]` 연결, GT 없는 episode 처리, dense 저장의 모든 field 일치, RGB 미저장, PNG decode·sidecar UUID·완료 상태 |
| Runner `invalid_diagnostic_parameters_fail_before_output` | PASS | 잘못된 repeat/각도는 출력 생성 전에 거부 |

첫 곡면 테스트에서 이상적인 paraboloid의 반올림 전 좌표로 기대각을 계산해 약
`5e-6rad` 차이가 발생했다. 실제 float32 관측 depth의 정확한 secant를 기준으로
oracle을 수정했다. 알고리즘 threshold와 기존 assertion tolerance는 완화하지 않았다.

Simulator rollout, SAM2/Grounding DINO 재추론, CUDA/MPS/GPU 실행, PCL 설치·binding·비교는
**NOT RUN**이다. 기존 저장 관측을 사용했으며 이 보고서는 새 simulator backend 검증을 주장하지 않는다.

## C. 실제 RGB-D 검증

- Episode: `visual/data/step2/ep000.npz`, seed 0, `base_camera`, 512×512, source 0–120 전체 121프레임.
- Mask: `visual/results/runtime_tracking/ep000/masks.npz`, runtime SAM2 tracking 출력.
- 입력 SHA-256: `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`.
- Mask SHA-256: `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`.
- 최종 run: `f9b0da8588734a28996a38f3724da16a`. `WANDB_MODE=disabled` 로컬 기록.

실행 명령:

```bash
WANDB_MODE=disabled .venv/bin/python visual/runtime/test/run_geometry.py \
  --input visual/data/step2/ep000.npz \
  --masks visual/results/runtime_tracking/ep000/masks.npz \
  --camera base_camera --out visual/results/runtime_geometry \
  --frames 0 15 18 45 80 120 --repeats 3
```

[시각화 HTML](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/geometry.html)과
[전체 프레임 수치 JSON](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/geometry.json)을 생성했다.
선택한 6프레임에는 각각 dense NPZ와 PNG가 있다. NPZ는 기존 ignore 규칙에 따라 로컬에 남는다.
각 PNG는 RGB, mask, 모든 지원 point의 3D world 표시, CV normal, depth 변화,
경계 원인, normal variation, RGB overlay, normal 지원 여부의 9개 panel로 구성된다.

선택한 6개 실제 NPZ를 원본과 다시 대조했다. Legacy backprojection과 XYZ 최대 차이는
**0m**, 저장 XYZ를 원본 pixel로 재투영한 최대 오차는 **1.4799198e-5px**,
유효 normal의 unit-length 최대 오차는 **4.1945522e-8**이다. 원본 valid depth·물체
지원 mask 일치, 무효 XYZ/normal의 NaN, normal의 camera-facing 방향, 중첩 제외,
discontinuity의 유효 point 포함 관계, artifact UUID sidecar도 확인했다.

| Source / simulator step / 시간 | 시각적 확인과 수치 |
| --- | --- |
| [0 / 0 / 0.00s](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.png) | Peg의 긴 면 접합선과 앞면 경계에 variation이 나타나고 Box hole 내부의 큰 depth 변화가 분리된다. Peg normal 11,357/11,780 points. Box jump 651개 중 mask 경계에서 떨어진 것이 399개다. Hole 주변 관측 구조를 볼 수 있지만 모든 후보가 실제 구조 edge라는 뜻은 아니다. |
| [15 / 15 / 0.75s](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0015.png) | 손이 Box를 대부분 가린다. Box mask 2,038px/normal 1,790개, mask boundary 239개. 잘려 보이는 긴 가림 윤곽도 반응한다. 실제 Box 외곽선으로 확정할 수 없다. |
| [18 / 18 / 0.90s](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0018.png) | Box mask/point/normal/모든 후보 0. GT는 94px가 남아 있어 단순히 GT상 완전 비가시라는 뜻이 아니다. Tracker 관측 부재를 별도로 기록한다. Peg는 계속 계산한다. |
| [45 / 45 / 2.25s](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0045.png) | 손에 의해 Peg·Box가 크게 가려진다. Box mask 2,541px 중 유효 point 2,539개, normal 2,311개. 무효 depth 인접 경계 61개를 mask/depth-jump 경계와 별도로 표시한다. |
| [80 / 80 / 4.00s](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0080.png) | Peg/Box mask 중첩 34px를 양쪽 지원에서 제외한다. Peg 면 내부의 양자화 줄무늬와 가림/오염 반응이 강하다. Peg >20° variation 1,507개 중 GT Peg 밖은 12개이므로 평면 자체의 normal 불안정성도 크다. |
| [120 / 120 / 6.00s](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0120.png) | 중첩 60px 제외. Peg mask IoU 0.8045, 지원 point 1,353개 중 GT 밖 183개. Jump 163개 중 GT 밖 61개, >20° variation 273개 중 GT 밖 52개여서 tracking 오염의 기하 반응도 확인된다. |

Box 빈 mask는 source **18–26, 56–65**, 총 **19프레임**이다. 19프레임 모두 point와
normal이 없으며 이를 현재 관측 부재로 기록한다. Peg의 빈 mask는 없다. 두 tracking
mask의 중첩은 전체 2,068 pixel-frame, 프레임당 최대 75px다. 원본 mask를 바꾸지 않았다.

경계 panel은 red=유효 depth 급변, green=mask 끝, blue=무효 depth 인접을 additive로
표시한다. 예를 들어 같은 위치의 yellow는 depth 변화와 mask 경계가 겹쳤다는 뜻이다.
영상 외곽은 별도 `image_boundary`다. RGB overlay는 jump를 노란색, 표시용 >20°
normal variation을 자홍색으로 그리며 둘이 겹치면 흰색이다. Edge/Corner label은 없다.

실패·혼동은 다음과 같다. Peg의 면 내부에 줄무늬와 체커 모양의 큰 variation이 있다.
Frame 0에서 >20°인 Peg 4,007px는 모두 GT Peg 안에 있어 mask 오염만으로 설명되지
않는다. 저장 depth는 mm 단위 정수값을 m로 바꾼 관측이며, GT Peg/Box 영역의 1mm
격자 최대 반올림 잔차는 각각 `1.48e-8m`/`2.96e-8m`다. 이런 양자화가 1-pixel
미분을 불안정하게 만드는 주요 원인으로 해석된다. 정량적인 원인 분리 실험은 하지 않았다.

또한 Frame 0 Box mask 안에는 finite positive 원거리 관측도 있어 full-extent 3D
축을 크게 늘렸다. 이를 필터링하지 않고 NPZ와 JSON의 전체 CV XYZ 범위에 보존했다.
3D 그림만 1–99 percentile과 여백으로 표시하며, frame 0은 23,445점 중 188점이
표시 범위 밖이다. 모든 점을 계산·저장했으며 그림 범위는 feature에 영향을 주지 않는다.

손에 가린 표면의 윤곽·tracking 오염·실제 hole/면 접합이 같은 후보로 반응할 수 있다.
GT segmentation은 support 오염을 진단할 뿐 구조 edge의 정답이 아니다. 검출 precision/
recall이나 최종 edge classification 성능은 이 결과로 주장하지 않는다.

## D. CPU 성능

Apple M4 Pro, macOS 27.0.1 arm64, root `.venv` Python 3.11.13 / NumPy 2.4.6.
512×512 전체 geometry와 Peg/Box 두 물체를 포함한 **frame당** wall-clock ms다.
Warmup 3회 제외, 121프레임×3회=363개 측정. 별도 프로세스 격리나 GPU 가속은 하지 않았다.

| 단계 | Mean (ms) | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| Geometry 생성·입력 검증·mask 복사/중첩 | 6.560 | 6.660 | 7.213 |
| 두 물체 normal | 4.080 | 3.925 | 5.698 |
| Discontinuity·경계·normal variation | 8.241 | 8.352 | 9.462 |
| 전체 geometry 처리 | **18.881** | **18.651** | **21.873** |

전체는 각 실행의 세 단계 합계이며 stage별 p95의 합은 아니다. I/O, warmup,
시각화, 압축 저장, logger, SAM2 추론을 제외한다. 실시간 전체 pipeline 지연을
의미하지 않는다. GPU/CUDA/MPS: **NOT RUN**, 이 baseline은 CPU NumPy 경로다.

## E. 남은 문제와 PCL 비교 전 확인사항

1. 1-pixel 5-point normal은 depth 양자화와 noise에 민감하다. 측정 해상도·depth 정밀도와
   normal 지원 범위가 결과에 미치는 영향을 구분해야 한다. 이번 단계에서는 smoothing,
   큰 이웃 plane fit 또는 pre-filter로 결과를 바꾸지 않았다.
2. 완전한 stencil을 요구하므로 물체 경계·hole·좁은 조각·단절 양쪽에서는 normal이 없다.
   `valid_points`, `normal_valid`, `normal_variation_valid`의 지원 개수가 다르다.
3. z 기반의 고정 abs+relative gate는 기울어진 연속 표면도 큰 인접 z 변화가 있으면 끊고,
   작은 단절은 놓칠 수 있다. 깊이 변화의 원인이나 near/far occlusion 방향을 분류하지 않는다.
4. 같은 mask 안으로 그리퍼·테이블·먼 depth가 섞이면 실제 관측 geometry로 남는다.
   Mask 소속 오류와 geometry 계산 오류를 별도로 평가해야 한다. 중첩 pixel은 소속을
   추정하지 않아 주변 normal 지원도 감소한다.
5. PCL 비교에서는 같은 원본 depth/mask·pixel centre·좌표/단위·invalid policy·normal
   지원·각도 방향·경계 유형을 맞춰야 한다. 구조 edge GT는 따로 필요하다. 현재 GT mask만으로
   mask/가림 경계를 structural edge로 평가하면 안 된다. 조정용/검증용 frame도 구분해야 한다.

PCL 설치/통합, final class·feature mode, structural/surface sampling, budget,
k-NN·edge filtering, 최종 Tensor, stale cache, Graph adapter, GNN/GRU는 진행하지 않았다.
**Task 01에서 중단했다.**

## F. Git 상태와 기존 작업 보존

시작 branch는 `visual-graph`였다. 시작부터 다음 tracked 변경이 있었다.

```text
 M docs/experiment_tracking.md
 M visual/runtime/README.md
 M visual/runtime/tracking.py
 M visual/scripts/collect.py
```

시작부터 미추적 상태였던 경로:

```text
visual/results/runtime_grounding/
visual/results/runtime_rgbd/
visual/results/runtime_tracking/
visual/runtime/initialization.py
visual/runtime/rgbd.md
visual/runtime/rgbd.py
visual/runtime/test/run_rgbd.py
visual/runtime/test/test_initialization.py
visual/runtime/test/test_rgbd.py
visual/runtime/test/test_rgbd_evaluation.py
```

위 상태를 `git status --porcelain -uall`로 확장한 **17개 파일의 SHA-256을 작업 전후
비교해 전부 동일함을 확인**했다. 기존 ignored 입력 episode·mask도 읽기만 했다.
기존 수정·미추적 파일을 덮어쓰거나 Git add/commit/reset하지 않았다.

이번 추가는 A의 6개 source/doc 파일과
`visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/`의 결과다.
실험 metadata·metrics·manifest는 기존 ignored `experiments/<run_id>/`에 기록된다.
Raw NPZ와 tracking sidecar는 기존 ignore 규칙을 따른다. 코드/문서 파일은 새 untracked 파일이다.

`git diff --stat`의 tracked 변경은 시작과 동일한 **4 files, 32 insertions, 8 deletions**다.
이 diff는 모두 기존 작업이며 이번 추가분은 아직 untracked이므로 그 통계에 포함되지 않는다.
새 코드 4개 파일은 총 948줄이며 새 문서 2개와 결과 HTML/JSON/PNG가 추가됐다.
`git diff --check`는 PASS다. 공통 코드·기존 Vision/Graph 경로의 신규 diff는 없다.
