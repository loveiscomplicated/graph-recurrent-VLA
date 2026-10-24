# Task 03-AB — Adaptive Node Budget Ablation: 최소 구현 계획

**상태: 계획 제출 / 구현·실험 승인 전.** 이번 작업은 이 문서만 추가한다.
코드·저장 JSON·NPZ의 목록과 내용을 읽었으며, sampling, 신규 면적·거리·coverage 진단,
테스트, simulator/device 실행은 하지 않았다. 아래 계산·검증·산출물은 모두 별도 승인 후의 계획이다.

## A. 기존 구현·artifact 재사용 근거

### 계약과 고정 조건

최우선 계약은 [00_interface.md](00_interface.md)다.
[Task03 보고서](03_sampling.md)와 run `2551448b0fae4c6bbce3caf48774275f`를 기준으로 한다.
기본 비교 조건은 S3 / optimized PCA5 orientation-only, 최대 256점, ratio .25,
orientation threshold 20°, seed 0, Peg/Box minimum 각 16점, `observed_points` allocation이다.
Priority와 Surface는 기존 3D Euclidean FPS를 그대로 사용한다.
Task03-E의 λ metric 및 Task03-F의 local grouping·결과는 사용하지 않는다.

| 현재 API / 파일 | 확인한 동작과 재사용 |
| --- | --- |
| [sampling.py](../../visual/runtime/sampling.py) `allocate_budget()` | 먼저 `min(capacity, minimum)` 예약, 잔여량을 관측점 비례로 capacity-capped largest remainder 배분. fractional tie는 물체 ID 순. `sqrt_observed_points`는 이번에 사용하지 않음 |
| `SamplingConfig` | `sum(min_points) > num_points`를 생성 단계에서 거부. Adaptive의 작은 총 budget에는 B의 support-capped minimum tuple을 전달해야 함 |
| `extract_sampling_evidence()` → `sample_points()` | observation만 입력. 같은 evidence에 서로 다른 `num_points`를 전달할 수 있어 production sampler 복제 불필요 |
| `sample_points()` | 전체 요청량의 `floor(.25 * num_points)`를 Priority 상한으로 계산. 후보 capacity와 물체 budget으로 다시 제한. Surface FPS는 Priority 선택을 초기점으로 사용 |
| [run_sampling.py](../../visual/runtime/test/run_sampling.py) | source frame 정합 검사, `validate_result`, 기존 Logger 및 input/source snapshot 기록 방식 재사용. 기존 54조건 sweep entrypoint 자체는 실행하지 않음 |
| [sampling_evaluation.py](../../visual/runtime/test/sampling_evaluation.py) | 기존 `CoverageMatcher`, `evaluation_context`, `score_selection`, `summarize` 재사용. 현재 branch에는 `reference_hits`와 `staged_reference_coverage`도 이미 존재함 |

현재 `sampling.py`는 원본 run의 `evaluated_source`와 파일 diff가 없다.
현재 평가 파일은 원본에 reference ID / 단계별 coverage helper가 추가된 상태다.
이번 계획에서 이를 수정하거나 실행 검증하지 않았다. 구현 시 원래 matcher·reference·scoring
정의가 보존됐는지 확인하고, 현재 source와 원본 snapshot을 각각 provenance로 남긴다.

### 면적과 observed count의 의미

[geometry.py](../../visual/runtime/geometry.py)의 `observed_mask(o)`는 다음과 같다.

\[
X_{o,t}=M_{o,t}\land\operatorname{isfinite}(D_t)\land(D_t>0)
\land\neg(M_{\mathrm{peg},t}\land M_{\mathrm{box},t}).
\]

- `A_raw,o,t = count_nonzero(frame.masks[o])`: tracker 원본 boolean mask 면적.
- `A_valid,o,t = count_nonzero(geometry.observed_mask(o))`: 실제 sampling support 면적.
- `extract_candidates()`가 이 support의 raster index를 만들고, 그 행 수가
  `sample_points().counts[o]['observed']` 및 allocation capacity가 된다.
  따라서 **현재 코드 정의상 Task03 observed count와 A_valid는 동일하다.**
  프레임별 실제 배열의 재계산 대조는 승인 후 검증한다.

Normal validity나 structural eligibility는 `A_valid`의 추가 조건이 아니다.
Normal이 없는 유효 관측점도 Surface FPS support에 포함된다. 거리 기반 outlier 제거도 없다.
두 면적은 pixel 단위 투영 면적이며 실제 3D 물리 표면적이 아니다.
Raw 면적의 합에는 overlap pixel이 중복될 수 있지만 valid support는 두 물체에서 모두 제외된다.
GT segmentation·pose를 이용해 mask나 면적을 보정하지 않는다.

승인 후 `A_raw - A_valid`를 invalid/nonpositive depth와 finite-positive overlap 제외량으로
중복 없이 나누어 기록한다. 이 차이만으로 tracker 오염량을 측정했다고 주장하지 않는다.
잘못 포함된 finite-positive pixel은 raw와 valid 양쪽에 남을 수 있다.

### 실제 저장 범위와 추가 계산

원본 [comparison.json](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json)은
121프레임·54조건을 기록한다. Source 0–120의 `fNNNN_metrics.json` 및 `fNNNN_samples.npz`가
각각 모두 존재함을 파일 목록으로 확인했다. 121개 NPZ 모두에서 아래 S3 후보 index,
Peg/Box reference UV/XYZ 및 baseline 선택 index의 key 존재도 확인했다.
이는 배열 내용의 신규 geometry 검증이나 저장 검증 script 실행과 구분한다.

| 자료 | 이미 저장된 값 | 승인 후 필요한 작업 |
| --- | --- | --- |
| 각 frame metrics의 `S3_n256_r0.25_t20_s0` | 물체별 observed/budget/candidate/Priority/Surface/selected count, joint matcher 수치, 표면 거리 요약, negative ROI, timing | observed를 A_valid로 우선 재사용. Raw 면적, Adaptive request와 선택, 새 control 및 새 집계 추가 |
| 각 frame NPZ | 조건별 선택 XYZ·pixel·flat index·object ID·role·선택 행 metadata | Fixed256 원본 선택을 비교 기준으로 유지. Adaptive 선택은 새 FPS 필요 |
| `evidence/pca5/{0,1}/20/orientation_flat` | 모든 frame의 S3 후보 source index | 원본 depth/intrinsic으로 후보 XYZ 복원 가능. 신규 evidence 후보와 exact index 대조 |
| `evaluation/{peg,box}/reference_{uv,xyz}` | 모든 frame의 평가 reference 배열 | 이 reference를 모든 조건에 공통 적용. 가시성·negative context 재구성 시 원래 reference와 exact 대조 |
| [ep000.npz](../../visual/data/step2/ep000.npz), [masks.npz](../../visual/results/runtime_tracking/ep000/masks.npz) | 원본 RGB-D·intrinsic·camera state·GT 평가 필드, tracker Peg/Box mask와 source mapping | Raw count와 support 검증, dense XYZ 복원, PCA5 evidence 재추출. 둘 다 현재 로컬에 존재 |
| Task01 dense geometry run `f9b0da8588734a28996a38f3724da16a` | source 0,15,18,45,80,120의 6개 NPZ | 전체 121프레임 dense cache로 취급 불가 |
| Task02-B normal comparison run `4b1c377ae2044344a4f596b249627d89` | source 0,15,45,80,120의 5개 NPZ | 전체 optimized PCA5 feature cache로 취급 불가 |
| Task02-C run `b622e76deceb419ab9c5aa80a8f8280c` | source 0–120의 pair NPZ 존재 | Task03 optimized PCA5의 전체 feature map 대체물로 가정하지 않음 |

예를 들어 저장된 frame 0의 observed count는 Peg 11,780 / Box 11,665이며,
frame 18은 Peg 10,913 / Box 0이다. 이는 기존 metrics의 인용이며 새 면적 측정이 아니다.
원본 보고서의 Box 관측 소실 구간은 18–26, 56–65다. 새 구간별 진단은 아직 하지 않았다.

**Task03 NPZ에는 전체 dense XYZ 및 모든 support 행의 feature map이 저장돼 있지 않다.**
선택 행 metadata를 전체 support feature로 확장하거나 보간하지 않는다.
Candidate index와 depth만으로 selection을 별도 구현할 수도 있지만 production sampler를 복제하게 된다.
따라서 최소 구현은 기존 `extract_sampling_evidence(frame, 'pca5')`를 프레임당 한 번 호출하고,
그 evidence를 모든 조건과 반복 timing에서 재사용하는 경로로 한다.

## B. Adaptive Budget 정책 하나

### 추천: 첫 양의 관측 면적 고정 reference + 선형 scaling

이번 control은 **α=1 한 가지**다. Raw mask는 진단용이며 정책 입력은 A_valid만 사용한다.
전체 episode 통계, GT, geometry 평가 결과, 미래 frame을 reference 결정에 쓰지 않는다.

\[
A_t=A_{\mathrm{valid,peg},t}+A_{\mathrm{valid,box},t},\qquad
t_0=\min\{t:A_t>0\},\qquad A_{\mathrm{ref}}=A_{t_0}.
\]

Source 순서로 처리하며 첫 양의 관측에서 reference를 설정한 뒤 episode 끝까지 고정한다.
초기 zero-support frame에서는 reference를 미설정으로 유지하고 요청량 0을 반환한다.
Reference 설정 frame도 평가에 포함한다. 현재까지의 관측만 사용하는 causal 초기화이며,
별도 calibration episode를 요구하지 않는 가장 작은 control이다.
수치 `A_ref`는 승인 후 기존 observed count에서 읽어 기록하며 이번에는 새로 산출하지 않는다.

초기 mask가 가림·원거리·오염 상태이면 기준도 편향된다. 이후 더 좋은 frame을 골라 재설정하지 않는다.
초기 면적이 작아 나중에 256에 자주 포화되거나 절감이 거의 없어도 결과를 그대로 보고한다.
이 한계 때문에 reference를 장면에 보편적인 물리 면적으로 해석하지 않는다.

각 물체에서 실현 가능한 기존 minimum을 다음과 같이 정의한다.

\[
m_{o,t}=\min(16,A_{\mathrm{valid},o,t}),\qquad
N_{\min,t}=m_{\mathrm{peg},t}+m_{\mathrm{box},t}.
\]

관측점이 하나라도 있으면 minimum 16을 목표로 하되 support가 16 미만이면 실제 존재하는 점까지만
예약하는 기존 의미다. Reference 설정 후의 계산 순서는 다음으로 고정한다.

1. `q = 256 * A_t / A_ref`를 계산한다.
2. 비음수 half-up 정수 반올림 `r = floor(q + .5)`를 적용한다. Python의 ties-to-even `round`와 구분한다.
   구현에서는 정수식 `r = (512 * A_t + A_ref) // (2 * A_ref)`로 동률을 명확하게 처리한다.
3. `N_requested = min(256, max(N_min,t, r))`로 clip한다.
4. `SamplingConfig(num_points=N_requested, min_points=(m_peg,t, m_box,t),
   allocation='observed_points', structural_ratio=.25, orientation_threshold_deg=20, seed=0)`를 만들어
   기존 `sample_points(evidence, config, method='S3')`를 호출한다.

`min_points`의 support 제한은 새 minimum 정책이 아니다. 기존 allocator 내부의
`minimum(capacity, (16,16))`와 동일한 예약량을 config 검증 전에 표현한다.
`N_requested=256`일 때도 기존 기본 minimum으로 계산한 allocation과 정확히 같다.
Total만 낮추고 `min_points=(16,16)`를 그대로 전달해 작은 budget에서 예외를 내는 구현은 피한다.
Production config validation을 완화하거나 production 기본값을 바꾸지 않는다.

| 관측 상태 | minimum / requested / actual 처리 |
| --- | --- |
| 양쪽 support ≥16 | `N_min=32`, 요청 범위 32–256 |
| 한쪽만 support ≥16 | 보이는 쪽 16 / 사라진 쪽 0, 요청 범위 16–256 |
| support가 16 미만인 물체 | 그 물체의 모든 실제 support까지 minimum 예약 |
| 양쪽 support 0 | `N_min=0`, 요청 0, 선택 0. Reference가 이미 있으면 보존 |
| 요청량보다 전체 support가 적음 | 실제 선택 합계 `min(N_requested, A_t)`. 없는 점 생성·복제·padding 없음 |
| 관측 복귀 | 보존된 reference와 현재 면적으로 즉시 재계산. smoothing·hysteresis 없음 |

물체별 allocation은 `allocate_budget()`가 그대로 담당한다. 관측량 비례 weight,
capacity redistribution, ID tie, 물체별 `SeedSequence([0, object_id])`, raster 순서를 바꾸지 않는다.

`requested_total`은 Adaptive 함수의 출력이고, `allocated_budget[o]`는 capacity를 반영한
기존 `result.object_budgets[o]`다. HTML/JSON의 물체별 requested budget은 이 **capacity-capped
할당 목표**로 표시한다. 별도의 uncapped 물체별 배분 규칙을 만들지 않는다.
전체 request와 물체별 allocation 합계가 support 부족 때문에 다를 수 있음을 함께 표시한다.
`actual_selected[o]`는 실제 반환 행 수로 독립 기록한다.

Priority 상한은 **실제 선택 합계가 아니라 요청량**으로 `K_t = floor(.25 * N_requested)`를 계산한다.
실제 Priority는 후보·물체 capacity 때문에 더 적을 수 있다. Surface 경로가 추가 후보를 선택할 수도 있다.
Budget 감소는 Priority quota도 함께 줄인다. 결과는 총 budget 정책의 효과이며 Priority 감소만의
독립 인과 효과로 분해하지 않는다. 요청량 256의 모든 frame에서는 원본 S3의 선택 index·순서·좌표·
role·metadata가 정확히 재현돼야 한다. 원본 Fixed256의 support 부족 출력도 그대로 보존한다.

### Reference의 카메라 의존성 기록

Config에 episode/mask 경로와 hash, `base_camera`, source resolution H/W(기존 보고서: 512×512),
reference source/frame_idx/timestamp, 물체별 reference count, reference intrinsic 및 cam2world,
단위 `pixel_count`, 초기화 규칙과 α/반올림/min/max를 저장한다.
Frame별 intrinsic/camera pose 또는 이를 복원할 원본 배열 경로도 기록한다.
다른 resolution·viewpoint로 reference를 자동 이전하거나 resize 보정하지 않는다.
해상도·카메라가 바뀌면 별도 실행의 reference를 초기화해야 하며, 움직이는 viewpoint의 영향도
본 정책의 한계로 남긴다. 이번에는 depth 보정 predictor나 learned predictor를 추가하지 않는다.

## C. 최소 Offline Ablation

### 비교 조건과 frame

| 조건 | 정의 | 재사용 / 새 실행 |
| --- | --- | --- |
| Fixed256 | 기존 S3, 요청 상한 256, 기존 minimum·allocation·FPS | 원본 121개 선택/metrics 재사용. 새 호스트 timing과 exact reproduction 확인을 위해 동일 sampler 재호출 |
| Adaptive | B의 선형 면적 정책 하나, 최대 256, 같은 S3 | 동일 121프레임에 새 budget·FPS·평가 |
| FixedMatched | Adaptive와 평균 실제 node 수를 가능한 가깝게 맞춘 고정 요청량 하나, 같은 S3 | 아래 count-only 규칙으로 정한 한 조건만 추가. 기존 64/128/256이면 저장 선택 재사용 가능 |

**세 번째 조건을 계획에 포함한다.** Fixed256 대비 절감만으로는 시간에 따라 자원을 다르게
배분하는 이점과 단순 저예산 sampling을 구분하기 어렵다. 추가 비용은 최대 121프레임의
단일 S3 조건 및 timing 반복이며 별도의 추출은 필요하지 않는다.

FixedMatched의 요청량 B는 geometry 평가 전에 count만으로 확정한다.
`S_adaptive = sum_t min(N_requested,t, A_t)`와 `S_fixed(b) = sum_t min(b, A_t)`를 사용해,
기존 `(16,16)` minimum을 허용하는 정수 `b ∈ [32,256]` 중
`abs(S_fixed(b)-S_adaptive)`가 최소인 하나를 고른다. 동률은 작은 b를 선택한다.
이는 count 산술이며 여러 b의 FPS·geometry sweep을 의미하지 않는다.
충분한 support에서는 Adaptive 평균 실제 수의 가장 가까운 정수 budget이 된다.
정수·capacity·minimum 제약으로 완전 일치하지 않으면 요청 B와 실제 평균 차이를 공개하고
정확한 동예산 비교라고 부르지 않는다. 목표가 허용 범위 밖이면 endpoint의 불일치를 명시한다.
전체 episode count를 쓰는 이 세 번째 조건은 **사후 효율 비교용 offline control**이다.
그 값은 Adaptive의 A_ref·정책·선택에 피드백하지 않으며 causal 온라인 baseline으로 주장하지 않는다.

평가는 같은 ep000 / source 0–120, 총 121프레임이다. 대표 frame은 0,15,18,45,80,120으로 고정한다.
추가 실패 frame이 발견되면 전 프레임 수치와 함께 표시하고 선택 이유를 남긴다.
다른 episode 실행을 포함하지 않으며 독립 episode 일반화로 해석하지 않는다.

### 승인 후 실행 순서

1. 원본 input/mask/source hash와 source mapping을 확인한다. 기존 run은 read-only 입력으로 다룬다.
   기록된 원본 SHA-256은 episode `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`,
   masks `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`다.
2. 기존 per-frame observed count를 source 순으로 읽어 B의 요청량을 만들고 FixedMatched B를 결정한다.
   GT·retention·surface 결과는 이 단계에 입력하지 않는다. Raw/valid 실제 면적 대조는 다음 단계에서 한다.
3. `iter_saved_rgbd()`로 원본 frame을 읽고 PCA5 evidence를 한 번 추출한다.
   계산된 support count를 저장 observed count와 대조하고, 20° 후보를 저장 S3 index와 exact 대조한다.
   불일치하면 원인을 기록하고 비교를 중단한다. 새 기준으로 원본 artifact를 고쳐 맞추지 않는다.
4. 같은 evidence로 Fixed256/Adaptive/FixedMatched를 호출한다. Fixed256 전체 121개 결과를 원본과 대조한다.
   Adaptive가 256인 frame도 별도 exact 대조한다. 동일 선택이면 저장 선택을 재사용해도 된다.
5. 선택이 확정된 뒤에만 저장 reference와 원래 평가 context를 사용해 geometry를 평가한다.
   원본 reference 정의·joint tolerance·negative ROI를 유지한다. 신규 GT-free 지표도 이 단계에서 계산한다.
6. 새로운 UUID 디렉터리에 frame별 값·선택·timing과 summary/PNG/HTML을 저장한다.
   원본 결과를 `link_output`으로 재소유하거나 sidecar를 덮어쓰지 않고 dependency 경로/hash로 연결한다.

### 변경 파일과 실행 비용

| 승인 후 대상 | 최소 변경 |
| --- | --- |
| `visual/runtime/test/run_adaptive_node_budget.py` 신규 | 평가 전용 area→budget helper, causal reference 상태, 기존 sampler/evaluator 호출, count-only matched control, 집계와 정적 HTML |
| `visual/runtime/test/test_adaptive_node_budget.py` 신규 | D의 누락 assertion. 기존 테스트 fixture·validator·matcher 재사용 |
| `docs/visual-graph-integration/03ab_adaptive_node_budget.md` 신규 | 실제 결과와 한계, 재현 명령, 상대경로 링크 |

기본 계획에서 `sampling.py`, 기존 runner/evaluator/tests, Task03-F 코드를 수정하지 않는다.
공유 API가 부족한 것으로 드러나면 부족한 항목과 최소 공통 확장안을 별도로 제시하고,
현재 helper에서 production sampler를 병렬 구현하지 않는다.

비교 출력은 조건당 121개 selection, 최대 363개다. Latency는 조건별 첫 frame warmup 1회 후
전체 frame에서 조건 순서를 회전하며 3회 반복한다. 첫 측정 선택을 저장하고 나머지는 시간만 남긴다.
동일 조건이 완전히 같으면 중복 FPS 대신 별칭 처리하며 이를 기록한다.
반복 중 evidence는 재사용하고 seed는 고정한다. 추출을 반복 측정하거나 full 54조건 sweep을 돌리지 않는다.

원본 저장 timing의 CPU S3 `sampling_total`은 mean 4.307 / median 4.025 / p95 7.300ms,
PCA5 `extraction_total`은 mean 13.738 / median 12.622 / p95 24.502ms다.
이는 비용 계획을 위한 **과거 기록**이며 이번 호스트의 새 측정·절감 예측이 아니다.
연산량은 121회 PCA5 추출 + 최대 3조건 × 121frame × 3회 selection과 warmup이다.
I/O·GT context·nearest-neighbour 평가·압축·PNG/HTML 생성 시간은 별도이며 현재 wall time은 미측정이다.

## D. 검증·완료 기준·산출물

### Frame별 지표와 실패 분석

| 범주 | 기록·비교할 값 |
| --- | --- |
| 실제 node budget | raw/valid area, support X, reference 상태, 전체 request, 물체별 capacity-capped budget, 실제 선택, Priority/Surface/선택 후보 수, `K_t`, shortfall |
| 절감·밀도 | Fixed256 실제 수 대비 `1-N_adaptive/N_fixed256`, 전체 `1-sum(N_adaptive)/sum(N_fixed256)`; 물체별/전체 selected per valid pixel 및 per raw pixel을 별개 열로 기록 |
| 시간적 변화 | 요청/선택 각각 signed ΔN과 abs(ΔN), source 연속성, mask 축소·확대·소실·복귀와 함께 전체 time series |
| 구조 보존 | Peg seam / Box rim의 GT visible·tracker observable·matching node 수·unique reference 수; X/C/P/S/F 단계, Priority와 Final coverage 및 retention |
| 일반 표면 | 물체별 전체 X→F 최근접 3D 거리 mean/p95/max와 기존 within-3mm; C→P mean/p95/max; 빈 선택의 missing count |
| 공간적 누락 | support의 image 4-neighbour component별 X/C/P/F 수와 거리, 후보가 있으나 P/F가 없는 component, 큰/작은 영역 보존 비교 |
| 오염·거리 | 기존 negative ROI와 중복 제거 union의 X/P/S/F 점 수, 관측/선택 depth 분포, raw-valid 차이 및 finite-positive overlap |
| CPU 비용 | 아래 budget / selection / packaging 잔차 / 전체 sampling 범위의 mean·median·p95와 node 수 관계 |

`shortfall_total = requested_total - selected_total`을 기록한다.
물체별로는 `allocated_budget - selected`와 `max(0,16-A_valid)`의 minimum support 부족을 구분한다.
현재 allocator가 capacity를 반영하므로 물체별 할당-선택 shortfall은 정상적으로 0일 수 있다.
관측점·요청·할당·선택을 하나의 node count로 혼합하지 않는다.
Fixed의 실제 선택이 0이면 절감 비율은 N/A, 면적 0의 밀도도 N/A다.
각 count·면적·밀도·ΔN의 mean/median/p95/min/max와 전체 frame 값을 보존한다.

Primary matcher는 같은 point/reference pair의 **2px AND 3mm**다.
기존 1px/2mm, 3px/5mm는 이미 지원되는 보조 열로만 유지하고 primary를 바꾸지 않는다.
`CoverageMatcher.reference_hits` / `staged_reference_coverage`로 matching node 수와
unique reference 수를 구분한다. 한 node가 여러 reference를 덮거나 여러 node가 같은 reference를
덮을 수 있다. GT visible 집합 V, support에서 observable한 O, Final에서 덮인 R에 대해
`V-O`는 관측 손실, `O-R`는 sampling 이후 손실로 별도 기록한다.
Reference가 없으면 N/A이고, observable reference가 있는데 선택이 없으면 coverage 0이다.

GT-free 일반 보존은 물체별 동일 support에서 계산한다. X가 비어 있으면 N/A,
X/C가 존재하지만 F/P가 비어 있으면 유한 거리를 0으로 채우지 않고 missing 수와 uncovered 상태를 남긴다.
JSON의 비유한 수는 null과 이유로 저장한다. 원본 all-support 거리에는 contamination도 포함한다.
작은 component는 기존 evaluator와 같은 ≤64 support pixel 진단을 우선 재사용하고,
component별 size와 within-3mm 비율·거리도 함께 남긴다. 이는 2D 관측 영역 구분이며 물리적 구조의
정답 분할이 아니다. GT face/plane별 기존 통계는 GT 평가로 표시하고 GT-free 지표와 구분한다.
누락 위치를 대표 frame overlay로 보여준다. 모든 진단은 정책 확정 뒤 수행하며 함수 튜닝에 쓰지 않는다.

Fixed256 대비 frame별 paired retention 차이(pp), surface 거리 차이, node 절감과의 관계,
최악 손실 frame 및 대표 frame을 같이 보고한다. Micro coverage와 frame별 분포를 모두 남기며,
연속 frame을 독립 표본으로 취급한 일반화 CI를 만들지 않는다.

| 실패 가능성 | 확인할 계획과 해석 한계 |
| --- | --- |
| 원거리 | 기존 depth의 median/p95, intrinsic으로 복원한 관측 camera-range와 면적·budget·retention 변화를 함께 표시. z-depth와 실제 range를 구분하며, tracker 오염의 영향도 남김. GT pose로 budget 수정하지 않음 |
| 부분 가림 | 같은 frame의 O를 분모로 Fixed/Adaptive 비교. 관측 reference 소실과 남은 reference의 sampling 손실 분리. 작은 면적만으로 가림 원인을 확정하지 않음 |
| Tracking contamination | raw와 valid가 함께 커질 수 있음을 명시. 기존 negative ROI·depth 분포와 request 증가를 확인하되 GT mask로 수정하지 않음 |
| Budget collapse | minimum 도달/지원 부족/완전 소실을 별개 표시. 소실 18–26, 56–65와 복귀 전후를 우선 확인하고, 양쪽 소실은 기존 합성 fixture로 검증 |
| 초기 reference 편향 | 첫 양의 frame의 camera·물체별 면적을 공개. 포화율과 minimum 도달을 기록하되 새 reference·다른 α로 재실험하지 않음 |

동일 물체의 거리 변화와 면적의 연관을 살피되, 운동·viewpoint·가림·tracking이 함께 변하는
한 episode에서 거리 하나의 인과 효과를 확정하지 않는다. 관련 변화가 실제 episode에 없으면
그 실패 모드는 관측 근거 부족으로 보고한다. 별도 episode나 인위적 mask/depth 변형 실험으로 확대하지 않는다.
Node count 안정성과 node 순서·물리적 point identity는 별개다. Frame 간 point 추적을 도입하지 않는다.

### Timing 범위

원본 evidence를 공유한 상태에서 같은 CPU process의 paired timing을 사용한다.

- `budget_policy_ms`: 이미 확보한 두 support count와 reference 상태에서 요청량을 계산하는 helper 시간.
  Raw/valid mask scan 및 count 확보 비용은 별도 `area_support_count_ms`로 보고해 숨기지 않는다.
- 기존 `budget_allocation`, `structural_sampling`(Priority FPS), `surface_fps`, `candidate_selection`,
  `sampling_total` timer를 재사용한다.
- 기존 timer에는 gather/packaging 전용 필드가 없다. `sampling_total`에서 위 네 세부 시간을 뺀
  잔차를 **결과 gather/packaging 및 나머지 함수 overhead**로 보고한다. 이 값이 순수 gather만의
  정밀 계측이라고 주장하지 않으며, 이 분리만을 위해 production sampler를 수정하지 않는다.
- 비교 범위 합계는 area/count 확보 + budget helper + 기존 sampling 호출이다.
  PCA5 normal·feature·pair extraction은 프레임당 한 번 재실행하지만 조건별 비교 시간에서는 제외한다.
  추출 시간은 별도 context로 기록한다. GT 평가·파일 I/O·보고서 생성도 selection latency에서 제외한다.
- 반복 raw timing을 남기고 먼저 frame별 3회 median을 만든 뒤 전체 frame의 mean/median/p95를 집계한다.
  모든 반복 표본의 분포도 별도로 남겨 frame별 집계와 혼합하지 않는다. 과거 timing은 참고 열로만 둔다.

실제 Graph/GNN 연산·통합 timing이 없으므로 node 절감을 GNN latency 개선으로 확정하지 않는다.
전체 20Hz, CUDA/MPS 성능, simulator backend 성능도 이번 범위에서 주장하지 않는다.

### 테스트 계획과 완료 기준

승인 후 기존 `test_sampling.py`, `test_sampling_geometry.py`, `test_sampling_evaluation.py`의
fixture·assertion을 먼저 재사용하고, 아래 누락된 Adaptive assertion만 새 테스트에 추가한다.

| 검증 대상 | 확인할 assertion |
| --- | --- |
| 면적 정합 | raw boolean count와 valid finite-positive / overlap 제외 count; saved observed = support rows = allocation capacity. Normal invalid는 support에서 제거하지 않음 |
| 수량 분리·범위 | support / request / allocated / selected 독립 기록, `0 ≤ selected ≤ requested ≤ 256`, selected 합계는 `min(requested,A_t)` |
| minimum·capacity | support 충분/16 미만/한쪽 빈 mask/양쪽 빈 mask/invalid-only/overlap-only. 작은 request에서 config 예외 없음, 실제 support 이상 보장 없음 |
| rounding·clip·causality | half-up 동률, 256 포화, minimum clip, 초기 zero 후 첫 양의 reference 고정. 같은 면적·상태는 동일 budget, α=1 단조 비감소, clip·반올림에 따른 plateau 허용 |
| 미래 독립성 | 임의 prefix의 budget/selection이 전체 episode를 처리한 같은 prefix와 동일. 미래 frame 삽입·변경이 이전 결과를 바꾸지 않음 |
| structural cap | `floor(.25*request)` 재계산, 후보 부족·없음, 실제 Priority≤cap, Surface의 추가 후보 포함과 Priority role 구분 |
| source alignment | flat index↔pixel↔XYZ exact, object support 및 모든 metadata 행 정렬, source 중복·padding·가상 point 없음 |
| 기존 S3 재현 | Fixed256 전체 121프레임, Adaptive request256 frame, support 부족 합성 경우의 원본 index·순서·좌표·role·metadata exact reproduction |
| GT 독립성 | GT annotation·pose·reference를 바꿔도 요청량과 선택이 동일. GT 읽기는 선택 후 평가에서만 수행 |
| 평가 정합 | 같은 reference와 joint tolerance, matching node와 unique reference 구분, Priority/Surface/Final 집합 관계, 빈 분모 처리 |
| 집계 정합 | frame 합계·mean/median/p95/min/max, ΔN, reduction/shortfall/density, FixedMatched 실현 평균 오차, timer 잔차와 반복 집계 |
| 원본 보존 | 시작 시 dirty/untracked 목록 보존, input/source/artifact digest 및 원본 sidecar 불변, 새 출력은 새 UUID 하위만 사용 |

관련 기존 sampling 테스트와 새 Adaptive 테스트를 저장소의 기존 Python 환경에서 실행할 계획이다.
공통 Logger API는 변경하지 않으며 신규 연결에서 공통 계약 문제가 생기면 관련 Logger/Artifact 테스트를
추가 검증한다. 현재 문서 작성에서는 테스트·기존 `verify_saved_sampling.py`도 실행하지 않았다.

완료 조건은 정책과 입력의 재현성, 위 계약 검증, 121프레임의 node 절감·geometry 손실·변동·
선택 비용을 비교 가능한 수치로 보고하는 것이다. Geometry 손실이 크거나 절감이 없어도
연구 질문에 수치로 답하면 실험은 완료다. 현재 사전 합의된 허용 손실 threshold는 없으므로
결과를 보고 임의의 채택 기준을 만들지 않는다. Geometry-level 개선 판정, Adaptive 최종 채택,
GNN/BC 개선, end-to-end realtime 성능은 각각 별도의 판단이다.

### Markdown / HTML / 원본 산출물

실제 실행은 기존 `utils.experiment_logger.ExperimentLogger(team='visual', seed=0,
model='task03ab_adaptive_node_budget', job_type='evaluate', config=...)`를 재사용한다.
새 UUID는 Logger가 생성하게 하고 project/group/identity 규칙을 바꾸지 않는다.
`WANDB_MODE=disabled`의 local 기록으로 진행하며 input/source hash와 실행 source snapshot,
dirty 상태·원본 Task03 run dependency를 기록한다. 별도 W&B logger나 업로드는 만들지 않는다.

새 `experiments/<run_id>/`에 다음을 생성할 계획이다.

| 산출물 | 내용 |
| --- | --- |
| `adaptive_node_budget.html` | 외부 CDN 없는 정적 HTML, 로컬 PNG/JSON/NPZ 링크와 frame별 수치 표 |
| `adaptive_node_budget.json` | 확정 정책·reference·FixedMatched B, 입력/source provenance, 전체 summary와 timing 범위 |
| `frame_metrics.json` | 121개 frame의 raw/valid area, 요청/할당/실제 수, geometry·temporal·cost 원시 수치 |
| `frames/fNNNN_samples.npz` | 공통 support/candidate source index 및 XYZ, 조건별 선택·role·정렬 feature, `evaluation/` reference. 정책 입력과 평가 자료를 구분 |
| `latency.json`, `verification.json` | 반복 timing, assertion 및 exact reproduction 결과, 미검증 범위 |
| `plots/*.png`, `frames/fNNNN_overlay.png` | 전체 time series·scatter와 대표 Fixed/Adaptive overlay |
| `evaluated_source/` | 이번 helper와 실제 사용한 sampler/evaluator 및 관련 source snapshot |

HTML에는 raw/valid area, Fixed/Adaptive 실제 node 수, Peg/Box budget time series,
면적-node 관계, geometry preservation-node reduction, Priority/Surface 변화,
occlusion/mask loss 구간, latency-node 관계, 대표 6frame의 동일 crop·축·색상 규칙 overlay를 포함한다.
FixedMatched 결과도 같은 표/plot에 표시한다. 각 plot과 frame 표에서 원본 JSON/NPZ/PNG를
상대경로로 열 수 있게 한다. JSON을 `fetch`해야만 내용이 보이는 구조는 피하고 `file://`로도
기본 그림·표를 볼 수 있는 HTML로 작성한다. 복잡한 dashboard framework는 도입하지 않는다.

최종 Markdown은 [03ab_adaptive_node_budget.md](03ab_adaptive_node_budget.md)에 별도로 작성하고,
실제 UUID 확정 후 `../../experiments/<run_id>/adaptive_node_budget.html` 및 원본 JSON/NPZ/PNG를
상대경로 링크로 연결한다. 이 링크 경로는 미래 산출물 예정이며 현재 결과가 생성됐다는 뜻은 아니다.
Logger의 `link_output()`은 새 파일에만 적용하고 JSON은 metrics, NPZ는 trajectory,
HTML/PNG는 reports로 등록한다. 원본 Task03 문서와 결과는 덮어쓰지 않는다.
Artifact publishing은 승인 범위 밖이며 HF dry-run·외부 업로드·Git commit도 하지 않는다.

Point node budget은 Peg+Box만의 상한이고 실제 N은 가변이다. Robot node 3개는 별도이며
RGB를 기본 GNN node feature에 추가하지 않는다. Vision의 geometric feature/sampling/EE 변환/
intra-object edge 책임과 Graph의 robot node/inter-object edge 책임은 유지한다.
이번에는 현재 관측 sampler만 평가하므로 양쪽 소실의 빈 출력을 측정한다.
이는 마지막 geometry·내부 edge·관측 EE pose를 보존하는 기존 stale 정책 폐기를 의미하지 않는다.
Stale cache와 현재 관측을 합친 총 budget/count는 별도 통합 검증 사항으로 남긴다.
Stale cache·Graph interface·EE Tensor·edge·학습·새 sampling metric·normal estimator·
mask/depth 수정·dependency 설치·GPU 최적화는 구현하지 않는다.

**이 계획 문서를 제출한 뒤 중단한다. 구현·계산·실험·테스트는 별도 승인 이후에만 진행한다.**
