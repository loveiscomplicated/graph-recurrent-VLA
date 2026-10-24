# Task 03-AB — Adaptive Total Node Budget Ablation

전체 HTML·NPZ·실행 소스와 원래 run 기록은 [HF 후속 게시 인덱스](../../visual/results/publications/sampling_followups.md)에 있다. Git에 포함한 핵심 그림·작은 수치는 아래 미리보기 링크로 확인할 수 있다.

**121프레임 구현·검증·실험을 완료했다.** 첫 유효 관측 면적을 고정 reference로 쓰는 선형 정책은
Fixed256 대비 평균 node 수를 **62.93%**, 평균 sampling 시간을 **45.55%** 줄였다.
그 대가로 Peg seam retention은 **61.28% → 28.16%**, Box rim은 **11.43% → 7.77%**로 낮아졌다.
이 결과는 geometry 손실이 작다는 가설을 지지하지 않는다.

평균 95점의 FixedMatched와 비교하면 Adaptive의 누적 reference retention과 전체 표면 거리는
유리하지만, 프레임별 평균 retention과 일부 작은 영역의 보존은 불리하다.
이번 결과만으로 Adaptive를 최종 sampling 정책으로 채택하거나 일반화된 최적 budget으로 판단하지 않는다.

Run: `f68c59f210b5499dadc7e962c7f64492`.
[정적 HTML](../../experiments/f68c59f210b5499dadc7e962c7f64492/adaptive_node_budget.html),
[설정·summary JSON](../../visual/results/diagnostic_previews/03ab/adaptive_node_budget.json),
[121프레임 원시 metrics](../../experiments/f68c59f210b5499dadc7e962c7f64492/frame_metrics.json),
[latency](../../experiments/f68c59f210b5499dadc7e962c7f64492/latency.json),
[검증 기록](../../visual/results/diagnostic_previews/03ab/verification.json),
[추가 집계](../../visual/results/diagnostic_previews/03ab/analysis.json).

## 1. 구현과 실행 전 관문

[승인된 계획](03ab_adaptive_node_budget_plan.md)과 [Vision–Graph 계약](00_interface.md)을 따랐다.
새 repository 파일은 다음 세 개다.

| 파일 | 내용 |
| --- | --- |
| [run_adaptive_node_budget.py](../../visual/runtime/test/run_adaptive_node_budget.py) | 평가 전용 budget helper, 전체 관문, 기존 sampler/evaluator 호출, 세 조건의 평가·timing·NPZ·HTML |
| [test_adaptive_node_budget.py](../../visual/runtime/test/test_adaptive_node_budget.py) | reference·반올림·causality·minimum/capacity·정렬·GT 독립성·집계 테스트 |
| 이 문서 | 실제 수치, 실패 사례, 재현 방법과 범위 |

Production `sampling.py`, normal estimator, 기존 evaluator, tracking, Task03-D/E/F 및 Graph 코드는
변경하지 않았다. 기존 `allocate_budget()`와 `sample_points()`를 그대로 사용한다.
평가에는 기존 `CoverageMatcher`, `score_selection`, `summarize`, 단계별 coverage helper와
기존 source 보존·ROI occupancy 진단 helper를 재사용했다. Sampler를 복제하지 않았다.

Baseline은 Task03 run `2551448b0fae4c6bbce3caf48774275f`의 `S3_n256_r0.25_t20_s0`다.
S3 optimized PCA5 orientation-only / 20° / ratio .25 / seed 0 / 3D Euclidean Priority·Surface FPS /
observed-points allocation / Peg·Box minimum 각 16점을 고정했다.
Task03-E depth metric과 Task03-F local grouping은 사용하지 않았다.

**다음 관문을 전체 121프레임에서 통과한 뒤에만 Adaptive/FixedMatched geometry 평가를 시작했다.**

| 관문 | 결과 |
| --- | --- |
| RGB-D와 tracker mask hash, source 0–120 mapping, camera/input provenance | 일치 |
| 현재 sampler와 원본 evaluated source, 기존 평가 함수·matcher·상수 의미 | 기존 source 계약 검사 통과. 평가 파일의 기존 additive helper는 허용 |
| 재추출 support count와 저장 observed count | Peg/Box 전 프레임 exact |
| optimized PCA5 20° orientation candidate flat index | Peg/Box 전 프레임 exact |
| Fixed256 선택 flat index·순서·XYZ·pixel·object·role·모든 저장 feature | 전 프레임 exact |
| 저장 reference UV/XYZ·순서 및 기존 frame geometry metrics | 전 프레임 exact. 같은 pair의 2px AND 3mm 유지 |

기존 Task03 NPZ에는 dense feature 전체가 없으므로 `extract_sampling_evidence()`를 프레임당
한 번 실행했다. 새 run의 `cache/fNNNN_evidence.npz`에 observation evidence를 손실 없이 저장하고,
전체 관문 통과 후 이를 재사용했다. 54조건 Task03 sweep, simulator, tracker는 재실행하지 않았다.

입력은 ep000 / `base_camera` / 512×512 / source 0–120, 총 121프레임이다.
Episode SHA-256은 `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`,
mask SHA-256은 `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`다.

## 2. 확정한 정책과 실제 node 수

정책 입력은 `A_valid,o = count_nonzero(geometry.observed_mask(o))`다.
Finite-positive depth, tracker 소속, overlap 제외를 만족하는 이 행 수가 기존 allocation의 observed
count와 정확히 일치했다. Normal validity나 GT로 면적을 수정하지 않는다.
Raw tracker 면적은 별도의 진단 값이다. 두 값 모두 pixel 단위 투영 면적이며 물리 표면적이 아니다.

첫 양의 관측인 source 0의 Peg 11,780 + Box 11,665 = **A_ref 23,445px**를 episode 끝까지 고정했다.
Reference camera intrinsic·cam2world·resolution·timestamp와 frame별 camera state는 JSON에 기록했다.
Reference를 성능이나 미래 면적으로 재선택하지 않았다. 다른 camera/resolution에 그대로 적용할
보편적 calibration 값으로 해석하지 않는다.

```python
minimums = (min(16, A_peg), min(16, A_box))
rounded = (512 * (A_peg + A_box) + A_ref) // (2 * A_ref)
requested = min(256, max(sum(minimums), rounded))
```

최초 양의 관측 전에는 요청 0이다. 이후에도 support 0이면 요청·선택 모두 0이며 reference는 보존한다.
Config에는 support-capped minimum tuple을 전달해 작은 요청량에서도 기존 validation을 유지했다.
이는 기존 allocator의 `min(capacity,16)` 예약과 동일하다. 실제 수는 `min(requested, total support)`다.
GT·미래 frame·geometry 평가 수치는 budget/selection에 입력하지 않는다.

FixedMatched는 count-only 절대 총량 차이 최소화 규칙으로 **b=95**를 선택했다.
Adaptive 총 선택 11,482개에 대해 FixedMatched는 11,495개로, frame당 **+0.10744개** 차이다.
정수 제약으로 평균이 완전히 같지는 않다. 이 조건은 전체 episode count를 사용하는 offline
control이며 causal 온라인 정책이 아니다. Adaptive reference와 선택에는 b를 제공하지 않았다.

| 실제 point node 수 | Fixed256 | Adaptive | FixedMatched |
| --- | ---: | ---: | ---: |
| 평균 | 256.00 | 94.89 | 95.00 |
| 중앙값 | 256 | 78 | 95 |
| p95 | 256 | 232 | 95 |
| 최소 / 최대 | 256 / 256 | 45 / 256 | 95 / 95 |
| 전체 합계 | 30,976 | 11,482 | 11,495 |
| 평균·총량 절감률 vs Fixed256 | — | 62.93% | 62.89% |

Adaptive의 중앙값은 Fixed256보다 69.53% 작지만 **최대 node 수는 줄지 않았다**.
요청 256 포화는 source 0,1,2의 3프레임이며 모두 원본 S3와 exact 재현됐다.
총 minimum 포화는 0프레임, 최저 요청 45는 source 69,70이었다.
이 episode에서는 모든 조건의 request=allocated 합계=selected였고 shortfall은 모두 0이다.
원본 계약의 support 부족·빈 관측 출력은 합성 테스트에서 별도로 확인했다.
Adaptive object allocation이 관측 가능한 minimum에 닿은 경우는 Box 1프레임, Peg 0프레임이었다.

| 평균 물체별 selected | Fixed256 | Adaptive | FixedMatched |
| --- | ---: | ---: | ---: |
| Peg | 161.98 | 59.97 | 58.88 |
| Box | 94.02 | 34.93 | 36.12 |
| 전체 Priority | 64.00 | 23.28 | 23.00 |
| 전체 Surface | 192.00 | 71.61 | 72.00 |

Priority cap은 각 frame의 **request**에 대해 `floor(.25 * request)`로 재계산했다.
Adaptive의 실제 Priority는 11–64개다. Node 감소와 Priority quota 감소가 함께 발생하므로,
결과를 Priority만의 독립 인과 효과로 분해하지 않는다. Surface가 선택한 candidate membership은
Priority role과 별도 기록했다.

### Raw / valid 면적 진단

| 121프레임 합계, pixel-frame | Peg | Box |
| --- | ---: | ---: |
| Raw tracker mask | 687,830 | 367,494 |
| Valid observed support | 685,762 | 365,372 |
| Invalid/nonpositive depth로 제외 | 0 | 54 |
| Finite-positive overlap으로 제외 | 2,068 | 2,068 |

Raw-valid 차이는 여기서는 위 두 제외 항목으로 설명된다. 그렇다고 남은 valid support가 GT 물체의
정확한 mask라는 뜻은 아니다. Finite-positive tracker 오염은 두 면적에 모두 남는다.
기존 negative ROI와 depth 분포도 보존했으며 mask·depth를 교정하지 않았다.

## 3. 구조 보존과 FixedMatched 비교

Primary retention은 **같은 point/reference pair가 ≤2px AND ≤3mm**를 만족할 때의
`unique covered reference / tracker-observable reference`다. GT visible과 tracker observable을 분리했다.
분모 0은 N/A다. 아래 micro 집계는 각 frame의 reference count를 합한 것이며 물리 point identity를
episode 전체에서 추적한 수가 아니다.

| 구조 | GT visible / observable | Fixed256 | Adaptive | FixedMatched |
| --- | ---: | ---: | ---: | ---: |
| Peg seam | 32,664 / 32,460 | 19,893 / 32,460 = 61.28% | 9,141 / 32,460 = 28.16% | 8,410 / 32,460 = 25.91% |
| Box rim | 1,778 / 1,365 | 156 / 1,365 = 11.43% | 106 / 1,365 = 7.77% | 85 / 1,365 = 6.23% |

Adaptive는 Fixed256보다 Peg **−33.12pp**, Box **−3.66pp**다.
Fixed256가 덮은 reference 수 대비로는 각각 45.95%, 67.95%를 유지했다.
같은 frame의 observable 집합이 세 조건에서 동일하므로 이 추가 손실을 관측 소실 탓으로 돌리지 않는다.
Visible reference 중 관측부터 잃은 수는 Peg 204 / Box 413으로 모든 조건에서 같았다.
관측 가능하지만 Final에서 보존하지 못한 수는 Fixed256의 12,567 / 1,209에서
Adaptive의 **23,319 / 1,259**로 증가했다.

Candidate가 덮는 unique reference는 Peg 30,355 / Box 1,143으로 모든 조건에서 동일하다.
선택 이후 단계는 다음과 같다. 각 triple은 **selected node / matching node / unique reference**다.

| 물체 / 조건 | Priority triple | Final triple | Surface가 Priority에 추가한 unique reference |
| --- | ---: | ---: | ---: |
| Peg Fixed256 | 5,412 / 4,834 / 19,588 | 19,600 / 4,936 / 19,893 | 305 |
| Peg Adaptive | 2,004 / 1,783 / 8,935 | 7,256 / 1,833 / 9,141 | 206 |
| Peg FixedMatched | 1,942 / 1,683 / 8,226 | 7,125 / 1,734 / 8,410 | 184 |
| Box Fixed256 | 2,332 / 40 / 103 | 11,376 / 67 / 156 | 53 |
| Box Adaptive | 813 / 26 / 75 | 4,226 / 39 / 106 | 31 |
| Box FixedMatched | 841 / 17 / 47 | 4,370 / 36 / 85 | 38 |

Candidate/Priority/Surface/Final의 matching node·reference ID·coverage, 겹침과 집합 차이를
모든 frame JSON에 저장했다. Surface 자체의 reference 수와 위의 추가분도 구분한다.
같은 node가 여러 reference를 덮거나 여러 node가 같은 reference를 덮을 수 있다.

### 평균 node 수가 비슷해도 geometry 결과는 한 방향이 아니다

Adaptive는 FixedMatched보다 누적 retention이 Peg **+2.25pp**, Box **+1.54pp** 높다.
그러나 **프레임별 retention 차이의 평균**은 Peg **−1.16pp**, Box **−16.33pp**다.
Peg 121개 비교는 Adaptive 우세/동률/열세가 44/17/60,
Box observable 분모가 있는 47개 비교는 10/25/12였다.

이는 모순이 아니다. Micro 집계는 observable reference가 많은 frame에 더 큰 가중치를 준다.
Adaptive가 초기 큰 면적에 더 많은 node를 쓰는 방식은 누적 reference를 더 보존했지만,
일부 후반 작은 mask에서는 95점 고정보다 크게 잃었다. 특히 Box의 작은 분모 frame은
frame 평균에서 크게 반영된다. Frame 평균과 누적 비율을 서로 대체해 개선을 주장하지 않는다.

## 4. 일반 geometry 보존과 작은 관측의 실패

아래는 같은 물체의 **전체 observed X → Final F** 최근접 3D 거리다.
Contamination을 포함한 원본 support를 유지했다. Mean/p95는 모든 source point-frame의 pooled
거리이며, frame별 p95의 평균과 구분한다.

| 물체 / 조건 | Mean mm | p95 mm | Max mm | Within 3mm |
| --- | ---: | ---: | ---: | ---: |
| Peg Fixed256 | 3.425 | 7.019 | 12.625 | 45.63% |
| Peg Adaptive | 5.578 | 11.701 | 22.168 | 18.49% |
| Peg FixedMatched | 5.979 | 12.050 | 20.254 | 17.40% |
| Box Fixed256 | 7.175 | 16.277 | 24.840 | 15.63% |
| Box Adaptive | 12.850 | 22.244 | 40.762 | 4.41% |
| Box FixedMatched | 14.928 | 34.515 | 54.285 | 4.82% |

Fixed256 대비 Adaptive의 mean/p95 손실은 Peg **+2.153 / +4.682mm**,
Box **+5.675 / +5.967mm**다. FixedMatched 대비 mean/p95는 두 물체 모두 유리하지만
Peg max와 Box within-3mm는 불리하다. 일반 geometry가 일관되게 개선됐다고 판단하지 않는다.

| Candidate C → Priority P 거리 | Fixed256 mean / p95 / max mm | Adaptive | FixedMatched |
| --- | ---: | ---: | ---: |
| Peg | 1.910 / 5.286 / 14.345 | 4.638 / 16.919 / 33.630 | 4.764 / 14.012 / 26.448 |
| Box | 9.013 / 27.905 / 40.645 | 11.321 / 30.651 / 46.565 | 19.989 / 61.533 / 92.179 |

기존 ≤64px image component 내 within-3mm coverage는 Peg **50% → 0% → 0%**,
Box **38.15% → 17.65% → 21.51%**(Fixed256 / Adaptive / FixedMatched)다.
Peg의 이 분모는 단 2 point-frame, Box는 595 point-frame이므로 일반적인 작은 구조 보존율로
해석하지 않는다. Component-frame 수는 Peg 122 / Box 208이며, 후보가 있는데 P 또는 F가
전혀 없는 support component는 세 조건 모두 0이었다. 이 거친 component 진단은 같은 큰 mask
내부의 국소 누락을 감추므로, 거리·reference ID·overlay를 함께 보아야 한다.

기존 negative union의 Priority 수는 Fixed256 Peg 426 / Box 2,004,
Adaptive 173 / 560, FixedMatched 201 / 739다.
일반 Surface node는 negative ROI에 있다는 이유만으로 잘못 선택된 구조 node로 분류하지 않는다.
Box valid support 365,372 중 기존 negative union 350,610도 그대로 남겼다.
이 ROI 수치는 GT 진단이며 tracker를 수정하는 근거로 sampling에 피드백하지 않았다.

### 작은 면적과 원거리·가림·tracking의 해석

전체 valid 면적의 median 7,134px로 나눈 **진단용** 구간에서 Adaptive−Fixed256 frame 평균
retention 차이는 다음과 같다. 이 분할은 평가용이며 reference나 정책에 입력하지 않았다.

| 전체 mask 면적 구간 | Peg 차이 | Box 차이 |
| --- | ---: | ---: |
| ≤7,134px | −31.25pp (61 frame) | −31.31pp (observable 33 frame) |
| >7,134px | −36.05pp (60 frame) | −1.34pp (observable 14 frame) |

**Box 손실은 작은 면적 구간에 더 컸지만 Peg는 큰 면적에서도 큰 손실을 보였다.**
작은 mask일수록 항상 더 많이 잃는다는 단순 관계는 확인되지 않았다.
Source 73의 Peg는 90개 reference가 모두 관측 가능했지만 Fixed256의 90개 보존에서
Adaptive **21개(23.33%)**로 줄었다. 이때 전체 요청 49, Peg 할당 27점이며
FixedMatched는 57개(63.33%)를 보존했다. 관측 소실로 설명할 수 없는 sampling 손실이다.
Source 75/83/98의 Box에서는 각각 observable reference가 3/1/2개인 상황에서
Fixed256가 모두 보존한 것을 Adaptive가 하나도 남기지 못했다. 작은 분모도 함께 공개한다.

기존 depth/intrinsic으로 계산한 관측 median camera range는 Peg 0.190–0.403m,
Box 0.507–0.532m였다. Median range와 valid 면적의 Spearman 상관은 Peg **−0.665**,
Box **+0.024**다. Camera z-depth와 range는 구분해 JSON에 기록했다.
이는 시간적으로 연속된 한 episode의 기술 통계다. 물체 운동·viewpoint·부분 가림·tracker 오염이
함께 변하므로 원거리 하나의 인과 효과를 입증하지 못한다. 작은 mask가 단순 geometry라는
보장도 없으며, 깊이 보정 정책이나 GT mask 수정은 추가하지 않았다.

## 5. Temporal stability, 소실과 reference 영향

Adaptive의 연속 frame `ΔN`은 **−13…+3**이며, `|ΔN|`의 mean/median/p95/max는
**2.083 / 1 / 9.05 / 13개**였다. Fixed256와 FixedMatched의 전체 node 수 변화는 0이다.
가장 큰 감소는 source 15→16의 148→135점이다. Peg의 최악 retention 차이는 source 73에 있어,
가장 큰 budget 점프와 최악 손실이 같은 frame에 집중되지는 않았다.

| source / 관측 사건 | Adaptive 전체 N | Adaptive Peg / Box allocation |
| --- | ---: | ---: |
| 17, 첫 Box 소실 직전 | 124 | 106 / 18 |
| 18, Box support 0 | 119 | 119 / 0 |
| 26, 소실 마지막 frame | 103 | 103 / 0 |
| 27, Box 재관측 | 101 | 83 / 18 |
| 55, 두 번째 소실 직전 | 78 | 62 / 16 |
| 56, Box support 0 | 79 | 79 / 0 |
| 65, 소실 마지막 frame | 55 | 55 / 0 |
| 66, Box 재관측 | 52 | 35 / 17 |

Box support가 사라진 18–26, 56–65의 총 19프레임에서 Box 선택은 모든 조건에서 0이다.
재관측 즉시 Box node를 다시 할당하지만 전체 N은 Peg를 포함한 총면적에 따라 달라지므로
소실 직전 값이나 256으로 복귀한다는 뜻은 아니다. 이 19프레임의 Peg paired retention 손실은
평균 −50.21pp다. 남아 있는 물체의 geometry에도 예산 감소가 영향을 주었다.

현재 episode에서 양쪽 support가 동시에 0인 경우는 없었다. 이 경우의 빈 출력은 합성 테스트로
확인했다. Smoothing, hysteresis, 물리 point identity tracking은 구현하지 않았다.
Mask 축소나 빈 mask를 모두 실제 occlusion으로 단정하지 않는다.

초기 reference 23,445px는 이 episode의 큰 관측에 가깝다. 이후 총 valid 면적이 약 4천px까지
줄면서 요청 45점이 됐고, 총 minimum 제한은 한 번도 작동하지 않았다.
즉 minimum 위에서도 심한 geometry 손실이 발생했다. 최대 256 포화는 처음 3프레임만 보호했다.
이 결과를 보고 α/reference/minimum을 바꾸거나 추가 sweep하지 않았다.

## 6. Paired CPU latency

동일 macOS CPU process에서 evidence를 공유하고 조건 순서를 회전하며 3회 측정했다.
조건별 동일 선택은 alias로 재사용했다. 첫 frame의 Adaptive=Fixed256은 같은 warmup을 공유하고,
FixedMatched는 따로 warmup했다. 반복 간 모든 selection field·feature·count가 deterministic했다.
아래는 먼저 각 frame의 3회 median을 만든 뒤 121프레임에서 집계한 값이다.

| Sampling total ms | Fixed256 | Adaptive | FixedMatched |
| --- | ---: | ---: | ---: |
| 평균 | 4.453 | 2.425 | 2.190 |
| 중앙값 | 4.172 | 1.928 | 2.044 |
| p95 | 7.605 | 6.907 | 3.825 |

Adaptive의 sampling 절감은 평균 **2.028ms / 45.55%**, median **53.79%**, p95 **9.18%**다.
큰 support의 초기 frame에서는 256점 또는 큰 요청량을 유지해 tail 비용 감소가 작았다.
FixedMatched보다 Adaptive 평균 시간은 **10.72% 더 길었다**.
평균 node 수를 맞춘 것은 FPS의 실제 CPU 연산량까지 맞춘 것은 아니다.

| 평균 세부 시간 ms | Fixed256 | Adaptive | FixedMatched |
| --- | ---: | ---: | ---: |
| Candidate selection | 0.659 | 0.660 | 0.672 |
| Budget allocation | 0.048 | 0.049 | 0.049 |
| Priority FPS | 0.245 | 0.142 | 0.138 |
| Surface FPS | 3.438 | 1.525 | 1.275 |
| Gather/packaging 및 나머지 overhead 잔차 | 0.055 | 0.045 | 0.045 |

각 세부 timer도 frame별 median으로 따로 집계했으므로 그 합이 sampling total median과 정확히
일치할 필요는 없다. 원시 호출별로는 timer 합계와 잔차를 재검증했다.
잔차는 순수 gather 시간이 아니다. 기존 sampler의 결과 생성과 나머지 함수 overhead를 포함한다.

| 별도 비용 ms | Mean | Median | p95 | 범위 |
| --- | ---: | ---: | ---: | --- |
| Adaptive budget helper | 0.00348 | 0.00333 | 0.00463 | 이미 확보한 observed count와 고정 reference의 정수 계산·검증 |
| Valid support count scan | 0.05089 | 0.05158 | 0.05683 | 관문에서 현재 support를 실제로 세는 비용, frame당 한 번 |
| Raw 면적·제외량 진단 scan | 0.08774 | 0.08850 | 0.09608 | 진단 비용. Budget 함수에 사용하지 않음 |
| PCA5 evidence extraction | 14.366 | 12.992 | 26.344 | frame당 한 번. Geometry/normal/feature/pair 포함 |

Sampling total은 budget helper, 별도 support/raw scan, PCA5 extraction, GT, I/O, 그림 생성을
포함하지 않는다. 이미 count가 확보돼 있으면 Adaptive의 추가 직접 비용은 budget helper다.
그 mean을 sampling mean에 더하면 2.428ms이며, raw scan을 온라인 정책 필수 비용으로 더하지 않는다.
Support scan은 검증용 실제 계측을 별도 공개했으며, sampler 내부 allocation의 행 수 확인 비용과
동일한 것으로 다시 계상하지 않았다.

측정 호출은 Fixed256 363회, Adaptive 354회 + alias 9회,
FixedMatched 318회 + alias 45회다. Alias는 `executed=false`, `timings_ms=null`로 저장했다.
모든 frame 비교를 위한 effective 분포는 실제 owner 호출 시간을 참조하며, 실행하지 않은 FPS에
새 시간을 부여하지 않았다. 실행된 호출만의 분포도 latency JSON에 별도로 보존했다.

이 수치는 CPU 현재-observation sampling의 비용이다. Node 감소에 따른 실제 GNN latency,
BC 성능, Vision 전체 속도, end-to-end 20Hz는 검증하지 않았다. GPU/CUDA/MPS와 새로운
simulator backend 실행도 하지 않았다.

## 7. 테스트·재현·원본 보존

[tests.log](../../experiments/f68c59f210b5499dadc7e962c7f64492/tests.log)의 관련 테스트 **42/42 PASS**다.
기존 sampling/geometry/evaluation 테스트와 새 Adaptive 테스트 11개를 포함한다.
반올림 동률, 첫 양의 reference 고정, future/prefix 독립성, 빈/부족/overlap/invalid 관측,
minimum·capacity·structural cap, 256점 exact reproduction, 전체 feature alignment,
GT reference 변경 불변성, FixedMatched 동률, NaN/N/A, timing alias·집계를 확인했다.

저장 결과도 121프레임 전부 다시 읽어 source/pixel/XYZ/모든 feature 정렬, 중복·padding 없음,
capacity/minimum/priority, causal request, matched 총량, ΔN/ratio, reference 집합 관계,
timing 잔차·alias를 재검사했다. HTML의 상대경로 156개가 모두 실제 로컬 파일에 연결됐다.
정량 PNG 3개와 대표 overlay 6개를 만들었고 주요 그림·빈 Box·초기/후반 overlay를 직접 확인했다.
외부 CDN/framework/JS fetch 없이 `file://`로 표와 그림을 읽을 수 있는 정적 HTML이다.

재현 명령은 repository root의 기존 환경에서 다음과 같다. `--preservation-before`에는 실행 전
기존 파일 hash와 Git 상태를 담은 snapshot을 전달한다. 이번 snapshot도 새 run 안에 보존했다.
실제 재실행은 새 Logger UUID를 만들며 원본 run을 덮어쓰지 않는다.

```bash
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03ab-mpl \
  .venv/bin/python -m unittest \
  visual.runtime.test.test_adaptive_node_budget \
  visual.runtime.test.test_sampling \
  visual.runtime.test.test_sampling_geometry \
  visual.runtime.test.test_sampling_evaluation -v

WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03ab-mpl \
  .venv/bin/python -m visual.runtime.test.run_adaptive_node_budget \
  --preservation-before experiments/f68c59f210b5499dadc7e962c7f64492/preservation_before.json \
  --tests-log experiments/f68c59f210b5499dadc7e962c7f64492/tests.log
```

공통 `ExperimentLogger`의 새 UUID, project `graph-recurrent-vla`, group `vision`,
`WANDB_MODE=disabled`의 local 기록을 재사용했다. Logger는 `finished` 상태다.
새 산출물만 기존 `link_output`/`register_existing_output`으로 연결했고 원본 sidecar는 바꾸지 않았다.
실행 Git commit은 `2e10a51736826aa980f2fe79a801eb6c9da8e288`, `git_dirty=true`다.
Commit만으로 복구할 수 없는 현재 source는 evaluated source snapshot과
[implementation diff](../../experiments/f68c59f210b5499dadc7e962c7f64492/implementation.patch)에 보존했다.
공통 logger/publisher 코드는 변경하지 않았다.

시작 시 기존 dirty/untracked 상태와 **1,226개 기존 파일**의 hash를 기록했고,
source·input·원본 Task03 artifact·sidecar의 변경/삭제는 **0개**였다.
Git stage/commit/reset, 외부 업로드, dependency 설치를 하지 않았다.

## 8. 원본 산출물과 판단 범위

[HTML](../../experiments/f68c59f210b5499dadc7e962c7f64492/adaptive_node_budget.html)에서 전체 frame
metrics·NPZ·PNG와 원본 Task03 dependency를 상대경로로 열 수 있다.

| 대표 source | Fixed / Adaptive / Matched 실제 N | 선택점 overlay | 원본 결과 NPZ |
| --- | --- | --- | --- |
| 0 | 256 / 256 / 95 | [PNG](../../visual/results/diagnostic_previews/03ab/frames/f0000_overlay.png) | [NPZ](../../experiments/f68c59f210b5499dadc7e962c7f64492/frames/f0000_samples.npz) |
| 15 | 256 / 148 / 95 | [PNG](../../visual/results/diagnostic_previews/03ab/frames/f0015_overlay.png) | [NPZ](../../experiments/f68c59f210b5499dadc7e962c7f64492/frames/f0015_samples.npz) |
| 18 | 256 / 119 / 95 | [PNG](../../visual/results/diagnostic_previews/03ab/frames/f0018_overlay.png) | [NPZ](../../experiments/f68c59f210b5499dadc7e962c7f64492/frames/f0018_samples.npz) |
| 45 | 256 / 95 / 95 | [PNG](../../visual/results/diagnostic_previews/03ab/frames/f0045_overlay.png) | [NPZ](../../experiments/f68c59f210b5499dadc7e962c7f64492/frames/f0045_samples.npz) |
| 80 | 256 / 57 / 95 | [PNG](../../visual/results/diagnostic_previews/03ab/frames/f0080_overlay.png) | [NPZ](../../experiments/f68c59f210b5499dadc7e962c7f64492/frames/f0080_samples.npz) |
| 120 | 256 / 76 / 95 | [PNG](../../visual/results/diagnostic_previews/03ab/frames/f0120_overlay.png) | [NPZ](../../experiments/f68c59f210b5499dadc7e962c7f64492/frames/f0120_samples.npz) |

[Budget time series](../../visual/results/diagnostic_previews/03ab/plots/budget_timeseries.png),
[geometry·temporal 그림](../../visual/results/diagnostic_previews/03ab/plots/preservation_temporal.png),
[latency·depth 그림](../../visual/results/diagnostic_previews/03ab/plots/latency_depth.png),
[실행 로그](../../experiments/f68c59f210b5499dadc7e962c7f64492/execution.log),
[source/input 보존 snapshot](../../experiments/f68c59f210b5499dadc7e962c7f64492/preservation_before.json)을 함께 보관한다.
NPZ에는 공통 support/candidate, 조건별 source·XYZ·role·정렬 feature와 별도 `evaluation/` reference가 있다.

**완료한 것은 구현·테스트와 ep000 한 episode의 geometry/계산 비용 ablation이다.**
평균 node·sampling 시간 절감은 확인했지만 geometry 손실은 크며 최종 채택은 별도 판단이다.
Graph/GNN/BC 개선과 전체 realtime 성능은 측정하지 않았다.
Pixel-frame 통계와 연속 frame 비교는 독립 episode 일반화 근거가 아니다.

Peg+Box point budget은 상한이고 실제 N은 가변이며 robot node 3개는 별도다.
GNN 기본 feature에 RGB를 추가하지 않았다. Vision/Graph의 책임과 interface는 유지했다.
이번 빈 관측 출력은 현재-observation sampler의 결과이며 기존 마지막 geometry·내부 edge·
관측 EE pose를 유지하는 stale 정책의 폐기를 의미하지 않는다.
Stale와 현재 관측의 합산 budget, EE Tensor/edge/Graph 통합, 03-F 결합은 이번에 구현하지 않았다.

**요청된 실험·Markdown·HTML을 제출하고 여기서 중단한다.**
