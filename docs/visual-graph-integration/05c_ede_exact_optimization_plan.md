# Task 05-C — Exact-Equivalent EDE CPU Optimization Plan

상태: **계획만 작성. 구현·신규 profiling·benchmark·실험은 승인 후 별도 진행한다.**

Task05-B의 최종 EDE를 정확성 기준선으로 삼는다. 권고하는 첫 변경은 **기존 integer SAT supercover의 선분 경로에서 반대 방향의 동일 축 계산을 제거하는 opt-in 최적화 하나**다. 저장 계측상 supercover가 가장 큰 단계이지만, 그 안의 할당·projection·Python 호출별 기여는 아직 측정되지 않았다. 승인 후 세부 profiling으로 이 변경의 타당성을 확인하고, 근거가 부족하면 범위를 늘리지 않고 계획을 재검토한다.

이 문서는 기존 코드·저장 JSON·execution source snapshot을 읽고 기존 timing sample을 집계한 결과다. EDE, 테스트, simulator, GPU, PCL, sampling, PCA5, feature extraction, EE transformation을 새로 실행하지 않았다. 새 UUID·HTML·benchmark artifact도 생성하지 않는다. 기존 dirty/untracked 코드와 실험·입력·sidecar는 보존한다.

## A. 현재 실행 단계와 기존 계측으로 확인되는 병목

### 기준선과 검토 자료

| 작업 | 고정 run ID | 역할 |
| --- | --- | --- |
| Task03-K | `9185355a32b8433abc70f1738f983377` | 저장 selection/source/provenance |
| Task04 | `9f90293b227e4dbfaaa6290afad9cc3b` | 최종 F=5 packet 및 저장 observation evidence |
| Task05 | `91ecd0413a87455cabc5e813cddaea8d` | T0/TMD 및 기존 audit |
| Task05-B | `9f36f7c89dbd4097a8be9358138d1537` | 최종 EDE 정확성 기준선 |

입력은 frame `0, 18, 27, 45, 56, 66, 80, 120` × keep ratio `0.5, 1.0`, phase `0`의 16개 조건으로 고정한다. Point/Camera·EE XYZ/Feature/validity/provenance/component/source ordering은 저장 값을 읽는다. Component 재분할이나 Point·Feature 재생성을 하지 않는다.

검토한 주요 구현은 [EDE helper](../../visual/runtime/extended_delaunay_edges.py), [Task05 raster/validator](../../visual/runtime/triangle_intra_object_edges.py), [평가 runner](../../visual/runtime/test/run_extended_delaunay_edges.py), [component/k-NN helper](../../visual/runtime/intra_object_edges.py), [Graph receiver](../../graph/vision_input.py), [EDE 계약 테스트](../../visual/runtime/test/test_extended_delaunay_edges.py)다. 근거 artifact는 다음과 같다.

- [Task05-B 결과 Markdown](05b_extended_delaunay_edgewise.md)
- [전체 결과·raw timing JSON](../../experiments/9f36f7c89dbd4097a8be9358138d1537/extended_delaunay_edgewise.json)의 `records/*/timings_raw/EDE`, `timing_ms`, `common`, `memory`
- [최대 Node 조건 record](../../experiments/9f36f7c89dbd4097a8be9358138d1537/records/f0000_r1.json)
- [고정 수치 NPZ](../../experiments/9f36f7c89dbd4097a8be9358138d1537/extended_delaunay_edgewise.npz), 조건별 `packets/*_EDE.npz`
- [Source hash inventory](../../experiments/9f36f7c89dbd4097a8be9358138d1537/source_hashes.json), [execution runner snapshot](../../experiments/9f36f7c89dbd4097a8be9358138d1537/source_snapshot/visual/runtime/test/run_extended_delaunay_edges.py), [report runner snapshot](../../experiments/9f36f7c89dbd4097a8be9358138d1537/report_source_snapshot/visual/runtime/test/run_extended_delaunay_edges.py), [저장 verification](../../experiments/9f36f7c89dbd4097a8be9358138d1537/verification.json)

현재 EDE/triangle/intra-object helper와 receiver는 execution snapshot과 SHA-256이 같다. 현재 runner는 report snapshot과 같으며, execution snapshot과 AST 비교 시 변경 함수는 `figures`, `reports`, `verify_outputs`, `main`이다. `execute`, `numerics`, 후보·검증 구현은 같다. 현재 테스트에는 depth gate 경계 fixture 보완이 있으므로 execution 테스트와 현재 테스트를 혼동하지 않는다.

| Execution source | SHA-256 |
| --- | --- |
| `extended_delaunay_edges.py` | `fcc81e0128a6689b7f44b337d3232895930e1e9b240a5a6b7faed911c3db932d` |
| `triangle_intra_object_edges.py` | `e91b759715582e32ce9f5bf9a52a14140424439a422b8a4a498804e9112b6078` |
| `graph/vision_input.py` | `beb482f032318b5e643edfe01463f10cab476cdbf3c2acfc8b3d9248c2642aa4` |
| `run_extended_delaunay_edges.py` | `cc3664526ddfe55f71218dcc2d4d400d527156f2703d524efd5212a55c26ab47` |

고정 aggregate NPZ의 SHA-256은 `8021db882f13c2d8da73457747a8d4de10296895d521c629357c913961be1a1a`다. 후속 실행은 이 inventory와 원본 packet/record/input hash를 시작·종료 시 확인한다. Git commit만으로 현재 untracked 구현을 복원할 수 있다고 가정하지 않는다.

### 실제 호출 경계

1. `load_inputs`는 저장 packet/evidence/T0/TMD와 selection을 읽고 source/hash/좌표/feature 계약을 확인한다. `load_evidence`는 NPZ의 geometry/features/PairEvidence를 역직렬화하며 estimator를 재실행하지 않는다. 관측 배열만 사용하고 GT는 선택에 넣지 않는다.
2. `lookup_maps(packet)`는 저장 `observed_component_id`를 selected-source raster map으로 옮긴다. `observed_components`의 full-support 연결성 계산은 이번 EDE 실행에서 호출되지 않는다. `prepare_validation(evidence)`는 pair/support 일관성과 중복을 확인하고 object별 right/down dense index를 만든다. 기존 benchmark는 이 공통 준비를 Core 밖에서 측정한다.
3. `execute('EDE')`의 Core 시작 → 원본 packet `deepcopy` → `build_extended_delaunay_edges` → metadata/edge packaging → `receive_vision_packet` → Core 종료다.
4. Helper는 source Camera XYZ와 관측 support를 검증하고 `candidate_edges`를 호출한다. Object별 component ID를 `unique`, component별 row를 `flatnonzero`로 구한다. Row는 원래 순서다. Integer UV를 준비하고 N에 따라 후보를 만든다.
5. N=0/1은 후보 없음, N=2는 한 pair, N≥3 exact-collinear는 Python integer cross/projection 및 source/row tie-break의 chain, 나머지는 같은 SciPy Delaunay와 `Qbb Qc Qz Q12`를 사용한다. QhullError/omitted vertices에 새 fallback은 없다.
6. Triangle 변은 `triangles_to_edges`에서 canonical unique 양방향 union으로 만든 뒤 undirected만 취한다. 전체 후보는 row pair lexsort, origin 동반 정렬 및 중복 검사를 거친다. 같은 호출 안에서 후보 Edge는 이미 unique다.
7. 후보별 `supercover(uv[pair])`를 한 번 호출한다. 현재 구현은 primitive bbox의 `mgrid`, centres, keep 및 integer SAT projection 배열을 만들고 row-major cover를 반환한다. 선분에서도 두 반대 방향 변을 순회한다.
8. `validate_cover`의 Mask 단계는 inside/mask/valid/overlap을 NumPy gather로 검사한다. **주요 Mask 검사가 pixel별 Python loop라는 증거는 없다.** Depth 단계의 `adjacent_pairs`는 Python cell set·이웃 순회·pair sort를 수행한다. 이후 support, right/down index, missing, finite delta, 기존 depth-jump bool은 배열 조회로 검사한다. 기존 dense lookup이 있으므로 매 pair마다 전체 PairEvidence를 선형 검색하지 않는다.
9. Validator는 조기 종료 없이 모든 사유와 count, 사유별 최대 4개 witness, cover/pair/support 개수를 만든다. 이 비용은 현재 Core 안이다. Accepted pair는 `directed_edges`의 canonical unique 양방향 배열로 포장된다.
10. Receiver는 feature/provenance/좌표 검증, Edge 대칭·중복·metadata 대수 검증, accepted origin의 whole-component collinearity/chain 재확인을 수행하고 packet을 다시 `deepcopy`한다. 좌표의 참조 계산은 검증이며 새로운 Point 공급 경로가 아니다. 이 계약과 복사 의미를 유지한다.
11. `analyze_condition`, component/cocircular 진단, 전체 edge audit 합치기, normal-angle/degree/length/recovery 집계, plot, NPZ/JSON/HTML 저장은 Core 밖이다. 그림에서 cover를 재생성하는 경우도 이 경계 밖이다.

### 저장된 측정 결과

기존 환경 기록은 Python `3.11.13`, NumPy `2.4.6`, SciPy `1.17.1`, macOS arm64, Qhull `2020.2.r`다. `_qhull` extension SHA-256은 `a4159010f18b05cfb4afaa587ccde36b0d06806fe24cd38eeadf1b295a25650a`이며 OMP/OpenBLAS/VECLIB thread 환경값은 모두 `1`이다. 이는 **과거 실행 기록**이며 이번에 backend나 설치 환경을 실행 검증한 결과가 아니다.

기존 16조건 × 5회 = 80개 저장 sample의 평균은 아래와 같다. 비율은 평균 Core wall `116.208145ms`에 대한 비율이다. 사용자 요청의 약 `116.208ms`는 `perf_counter` wall time이며 별도 process CPU 평균은 `115.984400ms`다.

| 저장 계측 경계 | 평균 ms | Core wall 비율 | 해석 |
| --- | ---: | ---: | --- |
| `candidate_total_ms` | 5.813859 | 5.00% | 입력 확인·grouping·chain·Qhull·candidate 정렬 포함 |
| `supercover_ms` | 48.214620 | 41.49% | 가장 큰 **측정된 단계** |
| `support_validation_ms` | 11.845770 | 10.19% | Mask/valid/overlap/image 검사·사유/count/witness 포함 |
| `depth_validation_ms` | 29.252713 | 25.17% | pair 열거·정렬·lookup·사유/count/witness가 합쳐짐 |
| `edge_packaging_ms` | 0.793769 | 0.68% | accepted 양방향 canonical 배열 |
| `packet_copy_ms` | 0.072380 | 0.06% | 최초 packet copy |
| `policy_metadata_packaging_ms` | 0.068577 | 0.06% | accepted pairs/origins/policy |
| `packet_packaging_ms` | 0.000678 | <0.01% | edge/component 부착 |
| `receiver_ms` | 8.892649 | 7.65% | 재검증·receiver deepcopy·출력 배열 |
| 위 단계 외 잔여 | 11.253132 | 9.68% | 미계측 구간과 loop/bookkeeping/timer overhead 등; 원인 미확정 |
| **전체 Core wall** | **116.208145** | **100%** | 이 행과 위 개별 항목만 합산 가능 |

`delaunay_ms=3.295713`, `collinearity_chain_ms=0.418273`, `two_point_ms=0.000671`은 candidate total 내부다. Delaunay timer에는 Qhull뿐 아니라 입력 변환·simplex 처리·triangle→edge 추출도 들어간다. `helper_wall_ms=107.167426`과 object별 timer도 상위/중첩 구간이므로 위 합계에 다시 더하지 않는다. `validation_lookup_ms≈0.000084`는 미리 준비한 contexts를 받는 비용이며 전체 lookup 구축 시간이 아니다.

공통 준비의 기존 반복 sample 평균은 selected-source map `0.055373ms`, validation context `1.035797ms`다. 최초 correctness 준비 평균 `0.294927/1.717037ms`와 구분한다. 같은 조건 내 EDE 후보/cover/판정 cache는 benchmark 방법 간 공유하지 않았다.

전체 sample의 Core wall median/p95/max는 `98.290250/266.561862/282.706416ms`다. 조건별 값은 아래와 같다. **U는 후보 undirected 수, A는 accepted undirected 수, E는 최종 directed 열 수**로 표기한다. 기존 JSON의 `stages.EDE.U`는 accepted undirected 수이므로 후보 수와 혼동하지 않는다.

| 조건 | N | 후보 U | A | E | mean ms | median ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f0000_r0.5 | 950 | 2587 | 2081 | 4162 | 164.951 | 153.165 | 202.047 | 213.729 |
| f0000_r1 | 1694 | 4711 | 3715 | 7430 | 272.041 | 269.995 | 280.841 | 282.706 |
| f0018_r0.5 | 453 | 1299 | 1041 | 2082 | 81.996 | 81.468 | 87.008 | 88.296 |
| f0018_r1 | 807 | 2344 | 1830 | 3660 | 149.079 | 134.592 | 195.163 | 210.187 |
| f0027_r0.5 | 539 | 1497 | 1253 | 2506 | 87.144 | 87.209 | 88.088 | 88.270 |
| f0027_r1 | 883 | 2500 | 2062 | 4124 | 142.884 | 142.416 | 144.147 | 144.281 |
| f0045_r0.5 | 514 | 1365 | 1181 | 2362 | 80.708 | 80.198 | 83.097 | 83.350 |
| f0045_r1 | 830 | 2255 | 1918 | 3836 | 155.055 | 137.776 | 213.253 | 231.931 |
| f0056_r0.5 | 371 | 995 | 858 | 1716 | 60.002 | 60.083 | 60.612 | 60.701 |
| f0056_r1 | 636 | 1740 | 1451 | 2902 | 121.365 | 101.399 | 180.842 | 200.285 |
| f0066_r0.5 | 424 | 1109 | 995 | 1990 | 65.444 | 64.815 | 67.415 | 67.963 |
| f0066_r1 | 630 | 1660 | 1473 | 2946 | 94.240 | 93.704 | 95.572 | 95.894 |
| f0080_r0.5 | 477 | 1198 | 1051 | 2102 | 71.868 | 70.826 | 75.452 | 76.190 |
| f0080_r1 | 747 | 1922 | 1636 | 3272 | 111.209 | 110.424 | 114.456 | 115.187 |
| f0120_r0.5 | 446 | 1153 | 989 | 1978 | 70.802 | 68.957 | 75.911 | 77.300 |
| f0120_r1 | 694 | 1808 | 1512 | 3024 | 130.541 | 105.424 | 206.075 | 231.134 |

총 30,143개 후보 중 accepted 25,046개, rejected 5,097개다. 최대 N 조건은 f0000_r1이며 평균·최대 latency도 가장 크다. 위 표는 기존 sample의 재집계이며 새 benchmark가 아니다. 기존 5표본 p95는 기술통계이며 안정적인 tail 추정으로 보지 않는다.

기존 `diagnostics_wall_ms` 평균 `85.405ms`는 여러 방법을 함께 분석하는 `analyze_condition` 비용이며 EDE witness 전용 비용이 아니다. 기존 numerical pipeline wall `113473.146ms`는 입력 외 numerical 절차·여러 방법·분석·memory worker 등을 포함하고, render wall `10877.590ms`는 별도다. 모두 EDE Core 시간에 더하거나 EDE 단독 latency로 해석하지 않는다.

기존 EDE fresh-worker Core peak RSS는 `156.34–164.59MiB`로 import/input/common 준비를 포함한다. 최대 N 조건의 Core high-water 증분은 `9,732,096bytes`, cover peak array는 `2,912bytes`, bbox scratch 추정은 `533,888bytes`다. 이는 전체 임시 할당량이나 native allocation별 측정이 아니다.

### 측정된 범위와 추정의 경계

| 단계/후보 | 확인한 사실 | 아직 확정할 수 없는 원인 |
| --- | --- | --- |
| Component grouping | 저장 ID 사용, component별 full-row boolean scan 및 정렬 존재 | grouping만의 비용과 반복 source mapping의 기여 |
| Delaunay | candidate 전체 5.00%, 그중 Delaunay 경계 평균 3.296ms | 순수 Qhull/준비/변 추출 각각의 비중 |
| Collinear/2-point | 정수 chain timer 평균 0.419ms 미만 합계 | 별도 최적화의 실익은 작아 보이나 미검증 |
| Supercover | 평균 48.215ms로 가장 큰 단계 | mgrid·작은 배열 할당·두 축 projection 중 지배 원인 |
| Mask | 현재 gather 기반, 평균 11.846ms | gather와 reason/witness 포장의 각각 비용 |
| Depth/PairEvidence | dense right/down map 이미 사용, Python set 기반 pair 생성 | pair 열거·정렬이 lookup보다 큰지, 반복 pair 검사율 |
| Dedup/packaging | 후보 및 final directed 정렬/unique 반복 존재 | candidate 내부 dedup의 순수 비용 |
| Audit | Core에 모든 사유/count/상위4 witness 포함, 큰 분석·그림·IO는 이미 외부 | Core witness만 분리할 때 실제 절감량 |
| Packet/Receiver | 최초 copy는 작고 receiver 전체는 8.893ms | receiver 재검증·metadata·복사의 각각 비용 |
| 잔여 | 11.253ms가 기존 소계에 배정되지 않음 | per-edge bbox 통계·append·관측 검증·GC/할당·timer 등의 기여 |

따라서 “Qhull이 주병목”, “PairEvidence 선형 검색이 주병목”, “전체 audit를 제거하면 85ms 절감”이라는 결론은 근거가 없다. 작은 배열 생성과 Python pair 열거는 코드로 확인되는 **원인 후보**이며, 세부 측정으로 확인해야 한다.

## B. Edge 결과를 바꾸지 않는 최소 후보와 선택 근거

### 1차 권고: 같은 SAT의 선분 계산 중복 제거

기존 선분 `p=[a,b]`의 loop 방향은 `d=b-a`, `-d` 두 개다. SAT normal도 `axis=(-dv,du)`와 `-axis`다. 동일 doubled-coordinate closed-cell 식

`c + radius >= min(projection)` 및 `c - radius <= max(projection)`

에서 `axis` 부호를 뒤집으면 두 부등식이 서로 대응한다. **산술 overflow가 없는 범위에서는 두 번째 축이 첫 번째와 완전히 같은 제약**이다. 한 축만 계산해도 기존 closed-cell boundary/corner-touch membership을 그대로 얻는다. 판정 근사나 새 raster 알고리즘을 도입하지 않는다.

제안하는 작은 변경 범위는 다음과 같다.

1. EDE 평가 모듈에 segment 전용 private helper를 추가한다. 같은 bbox·integer centres·doubled-coordinate SAT·inclusive 비교를 사용하고 선분의 한 normal만 처리한다. 이와 직접 연결된 `np.roll` 및 두 번째 projection/axis 임시 배열 생성을 줄인다. bbox 전체를 선분 길이 기반 traversal로 바꾸거나 새로운 scanline 알고리즘을 넣지 않는다.
2. 반환 dtype/shape뿐 아니라 **cover의 row-major 순서까지** 기존과 같게 유지한다. 그래야 witness의 첫 4개 선택도 같다. Degenerate/zero direction, 빈 출력, 음수 UV 처리를 기존과 대조한다.
3. Python integer로 안전 범위를 확인한 뒤 적용한다. 실제 packet UV의 image bounds에서 projection/centre/radius 중간값의 int64 안전성을 증명한다. 예를 들어 `M=max(H-1,W-1)`에 대해 `4*M*M+2*M`까지 안전한지 확인하는 보수적 guard를 사용할 수 있다. 검증되지 않은 범위·dtype·primitive는 기존 supercover로 위임하여 기존 오류/overflow 동작을 임의로 바꾸지 않는다. 이는 구현 dispatch이며 후보 생성의 Qhull fallback 조건을 바꾸는 것이 아니다.
4. 기존 `candidate_edges`, `prepare_validation`, `validate_cover`, `directed_edges`, `edge_metadata`, receiver를 그대로 사용한다. 모든 후보의 모든 검증 사유/count/witness를 계속 생성한다. Early exit나 normal veto를 넣지 않는다.
5. 기존 함수의 기본 경로는 baseline으로 유지하고 명시적 opt-in에서만 segment helper를 사용한다. 추가 keyword/내부 dispatch의 명칭은 후속 구현에서 문서화한다. 아래 C의 snapshot baseline은 이 dispatch조차 적용하지 않은 원본으로 실행한다.

선택 근거는 가장 큰 **측정된** 경계에 국한하면서 후보·검증·audit·receiver를 건드리지 않을 수 있다는 점이다. 단, 48.215ms 전체가 중복 축 비용은 아니므로 절감량·speedup을 약속하지 않는다. Supercover 시간을 전부 제거한다는 비현실적 상한도 전체 Core 약 1.71배에 불과하며, 이 변경의 기대 성능으로 사용하지 않는다.

### 나머지 후보의 검토와 보류

| 검토 후보 | 이번 최소 변경에서의 결정 |
| --- | --- |
| 동일 선분 raster memoization | 한 EDE 호출 내 후보는 이미 unique이고 raster도 후보당 1회다. 반복 benchmark·ratio·DE/EDE 간 cache 공유는 공정성을 해친다. 보류한다. |
| 반복 cover 검사/PairEvidence 결과 cache | 관측 전용 결과 재사용은 가능하지만 현재 dense map이 이미 있다. Pair 재사용률·setup·메모리·사유 순서 비용을 먼저 측정한다. 첫 변경에 포함하지 않는다. |
| `adjacent_pairs` vectorization | Python set/loop/sort가 실제 sub-bottleneck인지 아직 미확정이다. membership·sorted pair·witness의 exact 계약이 추가되므로 보류한다. |
| source/component mapping·candidate/final dedup | candidate total 5%, final packaging 0.68%이고 순서 계약에 민감하다. 첫 변경에 포함하지 않는다. |
| Audit/Core 분리 | 큰 진단은 이미 Core 밖이다. Core의 witness·다중 사유를 제거하는 변경은 A/B 작업량을 바꾸므로 권고하지 않는다. |
| receiver 검증/복사 생략 | immutable/deep-copy/metadata 계약에 영향을 주며 공유 API 변경이 된다. 수정하지 않는다. |
| 새 raster·다른 Delaunay library·공통 validator 재설계 | 필요성이 입증되지 않았다. 이번 계획과 자동 구현 대상에서 제외한다. |

### 승인 후 profiling 관문

먼저 원본 snapshot과 저장 16조건의 exact 재현을 통과시킨다. 이어 **같은 16조건만** instrumented 별도 pass로 조사한다. 새로운 episode/Point/feature는 만들지 않는다.

- Candidate: 입력 확인/group row lookup, Qhull 입력 준비, Qhull 호출, triangle→edge unique, 전체 후보 정렬을 분리한다.
- Supercover: 호출 수, bbox/cover cell 수, 할당·centres 구성, 첫/두 번째 축 계산·filter 시간을 분리한다. Python/native 할당 측정과 함수 profile은 최종 speedup 측정과 분리한다.
- Validation: Mask gather와 reason/witness 포장, `adjacent_pairs` set/이웃/pair sort, support filter, right/down gather, delta/jump gather, reason/witness 포장을 분리한다. 동일 observation pair의 중복 검사 횟수는 진단값만 기록한다.
- 잔여: source 관측 검증, per-edge bookkeeping/bbox 통계, append, timer, GC 관련 구간을 확인한다. Receiver도 validate/deepcopy/출력 배열로 나누어 본다.
- 계측 overhead를 별도 보고하고 nested timer를 합산하지 않는다. Main benchmark에서는 추가 profiler와 allocation tracing을 끈다.

중복 SAT 계산이 의미 있는 비용이며 안전 범위가 증명되면 위 한 변경을 구현한다. 해당 근거가 없거나 다른 단계가 지배하면 profiler 결과를 근거로 재계획한다. Pair cache·audit 분리·새 raster까지 자동 확장하지 않는다.

## C. Exact Equivalence·구현 경계·공정한 CPU Benchmark

### 불변 계약과 실패 기준

| 대상 | 비교 방식 |
| --- | --- |
| 후보 | canonical undirected pair 배열·집합, origins dtype/값/순서, triangles, component status/rows/source/chain/omitted/coplanar exact |
| 모든 후보의 판정 | accepted bool 배열, accepted/rejected pair 배열·집합 exact. Rejected는 canonical 후보에서 같은 bool mask로 추출 |
| 검증 의미 | 각 cover의 pixel membership **및 순서**, image 내부의 right/down pair 배열·순서, pair source/index·support·missing/finite/jump 상태 exact |
| 진단 | reason list 순서, 전체 cause count, 사유별 최대4 witness 값·dtype·shape·순서, cover/enumerated/supported pair 수 exact |
| 최종 Edge | `int64 intra_object_edge_index[2,E]` dtype/shape/bytes 및 방향·행·열 순서 exact; 정렬해 비교하는 것으로 대체하지 않음 |
| Node/source/component | Point row→source-index, object/component membership, node ordering/provenance exact |
| Point/Feature | Camera/EE XYZ, F=5, validity, coordinate/transform 원본 값 및 metadata bytewise 또는 기존 equal-NaN 계약. 허용오차 확대 없음 |
| Packet/Receiver | EDE policy와 accepted pairs/origins metadata exact, 원본 입력 불변, receiver packet deep copy·출력 mapping exact, robot 공급 시 point와 Edge +3 |
| 재현/보존 | 저장 selection/input/artifact/sidecar/hash 불변, GT 접근 없는 선택, 실행 source snapshot/hash와 변경 diff 보존 |

Mask support, finite-positive depth, overlap, image boundary, 둘 다 cover 안인 right/down pair, missing/invalid evidence, 기존 strict gate `abs(z_b-z_a) > 0.01 + 0.02*min(z_a,z_b)`를 모두 유지한다. Gate는 기존 PairEvidence 결과를 조회하며 새 estimator나 float 계산으로 대체하지 않는다. Pixel centre, half-pixel closed cell, corner-touch, Edge 독립 판정, normal veto 없음도 고정한다.

사유 순서는 현재 `image_outside`, `mask_outside`, `invalid_depth`, `overlap`, `pair_unsupported`, `pair_missing`, `delta_invalid`, `depth_jump`의 발생 순서를 유지한다. 여러 원인이 동시에 있는 Edge도 모든 원인을 보존한다. Timing·allocation처럼 의도적으로 달라지는 진단 수치는 semantic hash에서 제외하되 제외 field를 명시한다. 정책·output metadata를 구현 식별자로 바꾸지 않고, 구현 식별자는 experiment metadata에만 기록한다.

원본 snapshot ↔ Task05-B 저장 packet/NPZ/JSON, optimized ↔ 원본 실행, optimized ↔ 저장 artifact의 **3방향** 대조를 한다. 저장 JSON에 없는 전체 cover/pair 배열은 원본 snapshot으로 재생성하여 대조한다. Audit 재생성은 저장 입력과 source hash로 재현 가능해야 한다. 하나라도 다르면 correctness FAIL이며 speedup 성공을 선언하지 않는다.

### 계약 테스트 계획

기존 Task05/05-B 테스트, `test_triangle_intra_object_edges`의 Fraction oracle, receiver·feature·k-NN 회귀를 재사용한다. 기존 테스트를 수정해 차이를 허용하지 않고 optimized 경로를 같은 fixture에 추가한다.

- Empty object, N=0/1/2, component/object 분리, horizontal/vertical/양·음 slope collinear, 비균일 chain, source/row permutation과 tie-break.
- Non-collinear Delaunay candidate/triangle exact, cocircular diagonal, 거의 collinear이지만 integer cross가 0이 아닌 경우, QhullError/omitted vertices와 fallback 없음. Python integer 대형 cross 테스트 유지.
- Fraction oracle의 모든 기존 segment 사례와 역방향, 음수 UV, image boundary, half-pixel cell 접촉, diagonal corner-touch 및 zero direction. Bbox 안이지만 cover 밖인 실패는 영향 없음.
- Mask 밖, invalid depth, overlap, image 밖, pair missing/unsupported/delta invalid/depth jump, strict threshold equality/next float, 복수 사유와 witness 상위4 경계.
- Normal variation/normal angle/GT 변경과 무관한 Edge, Edge 독립성, triangle 내부 실패가 유효한 변을 거부하지 않음.
- 원본/optimized candidate·origins·accepted/rejected·reason/count/witness·cover/pair exact, final Edge bytes/order exact, Point/Feature/provenance 불변.
- Receiver deep copy와 metadata validation의 정상·비정상 packet 회귀, robot +3 mapping, robot/inter-object topology 불변.
- **16개 실제 입력 전부** 저장 output metadata/packet/edge_index exact 대조, 원본 input/selection/source/artifact hash 확인. Hash만 비교하지 않고 필드별 exact assertion도 실행한다.

후속 test command는 기존 5개 모듈(`test_extended_delaunay_edges`, `test_triangle_intra_object_edges`, `test_intra_object_edges`, `test_vision_features`, `test_variable_rate_graph_contract`)에 대한 root `.venv/bin/python -m unittest ... -v`를 기반으로 한다. 현재는 실행하지 않는다.

### 최소 변경 파일과 baseline 격리

| 후속 파일 | 예정 변경 |
| --- | --- |
| `visual/runtime/extended_delaunay_edges.py` | private segment SAT helper와 명시적 opt-in dispatch. 기본 baseline 경로·정책 유지 |
| `visual/runtime/test/test_extended_delaunay_edges.py` | Fraction 및 원본/optimized exact 대조·audit/receiver/input 회귀 추가 |
| `visual/runtime/test/run_extended_delaunay_edges.py` | 별도 05-C opt-in mode로 baseline/optimized만 실행. 원본 05-B 기본 실행·보고서는 유지 |
| 필요시 평가 전용 helper 1개 | snapshot loader/05-C 측정이 runner에 과도하게 섞일 때만 분리. 공통 validator 재설계는 없음 |
| `docs/visual-graph-integration/05c_ede_exact_optimization.md` | 승인 후 실제 결과·재현 command·기본 경로 및 opt-in 설명 |

`triangle_intra_object_edges.py`, `intra_object_edges.py`, `graph/vision_input.py`, logger/publisher, sampling/PCA5/feature/EE 변환, voxel/ratio, robot/inter-object Graph, encoder/model은 수정하지 않는다. 기존 snapshot은 어떤 파일도 덮어쓰지 않는다.

Baseline은 Task05-B `source_snapshot`의 원본 helper 및 dependency를 **명시적 별도 module namespace**로 적재하고 import binding을 검증한다. 새 코드를 baseline이라고 부르거나 Python import cache가 optimized helper를 섞지 않게 한다. Snapshot EDE가 참조하는 raster/validator/receiver 함수의 module path와 SHA-256을 기록한다. 변경하지 않은 dependency의 현재 hash가 snapshot과 같으면 그 사실을 검증한 후 공유할 수 있다. 이름 충돌이 해결되지 않으면 격리 worker를 사용하고 process startup/IPC를 Core 시간에서 제외한다. 단순히 `sys.path`만 바꾸고 재현됐다고 간주하지 않는다.

기존 runner 전체 `main/load_inputs/numerics`를 그대로 호출하면 K8/TMD/DE 재실행, 기존 보호 hash 가정, 출력 경로가 함께 따라온다. 05-C mode는 저장 Task05-B 입력/output을 검증하는 gate와 필요한 read/serialize/통계 함수만 재사용하고 **K8/TMD/DE를 재benchmark하지 않는다.** 원본 artifact를 `link_output`의 새 결과로 등록해 sidecar 소유권을 바꾸지 않는다.

### Benchmark의 세 경계

| 경계 | 포함 작업과 비교 원칙 |
| --- | --- |
| **A: 원본 EDE 전체 Core** | 원본 `execute`와 같은 packet deepcopy, fresh 후보 생성, cover, 모든 검증·reason/count/상위4 witness, raster bookkeeping, canonical Edge/metadata/packet packaging, 전체 receiver 검증·deepcopy |
| **B: Optimized EDE 동일 Core** | A와 같은 입력·모든 반환/audit 기능. 선분 SAT 중복 계산만 변경. **A/B가 주 비교** |
| **C: Detailed Audit 추가 비용** | Core 밖의 전체 cover/pair/source 증거 materialization, 분포/normal/degree 집계, 동일 overlay, plot, JSON/NPZ/PNG/HTML 저장을 개별 계측 |

기존 공통 selected-source lookup/`prepare_validation`은 A/B 밖 별도 계측으로 동일하게 준비하고, common+A/common+B 합계도 보조 표로 제공한다. 새 사전 계산이 필요하면 B의 준비 시간·메모리와 총합에 포함한다. 이번 권고에는 그런 새 cache가 없다. 임의의 candidate/cover/판정 결과를 warmup·repeat·ratio·방법 사이에서 재사용하지 않는다.

이번에는 Core audit 기능을 분리하지 않는다. 따라서 `core_audit_mode=full_reasons_counts_top4_witness`, `detailed_audit_in_core=false`와 제외한 외부 집계/plot/IO 목록을 명시한다. 추후 production-like 경로를 제안하려면 원본에도 같은 진단 제외 옵션을 적용한 별도 통제 A′/B′를 설계해야 하며 A/B speedup에 섞지 않는다.

### 고정 계측 절차와 지표

1. 승인 후 실행 시작 전에 config에 **3회 warmup + 30회 측정/조건/구현**을 고정한다. 16조건 모두 동일 반복 수를 쓴다. 결과를 본 뒤 반복 수·입력·제외 기준을 바꾸지 않는다.
2. 같은 CPU host, root interpreter, NumPy/SciPy/Qhull extension을 사용한다. 원본 저장 환경과 현재 환경이 다르면 차이를 기록하고, A/B는 같은 새 환경에서 재계측한다. 동일 thread env `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`을 import 전에 설정하고 기록한다. 가능한 설치 API로 실제 native thread pool 상태도 확인한다. 설치/업그레이드는 하지 않는다.
3. 조건별 warmup은 각 구현 3회, 측정은 AB/BA를 교대로 30쌍 수행해 각 순서 15회씩 유지한다. 조건별 시작 순서도 번갈아 둔다. 실행 순서와 warmup 여부를 저장한다. Candidate/판정은 매번 독립 생성한다. Python GC 설정·instrumentation·반환값 lifetime을 대칭으로 유지한다.
4. `perf_counter` wall과 `process_time` CPU를 모두 기록한다. 원본의 기존 stage timer와 동등한 경계를 A/B에 적용한다. 세부 profile의 추가 timer/cProfile/allocation tracing은 별도 pass로 제한한다. 정확성 hash/assertion은 각 timed call 종료 뒤 수행하고 그 비용은 별도 기록한다. Receiver 자체 검증은 계속 timed Core 안이다.
5. `timings_raw.json`에 sample별 condition/repeat/order/N/U/A/E, wall/CPU, stage 값을 보존한다. 조건별 mean·median·p95·max와 frame별 두 ratio 각각의 `baseline mean / optimized mean` speedup을 표기한다. P95 계산은 기존 `np.percentile(...,95)` 방식과 버전을 고정한다. 30표본 p95의 한계도 명시한다.
6. 전체 16조건의 동일 가중 평균, 총 timed wall/CPU 합계, 비교 pipeline의 wall을 구분한다. Aggregate speedup은 같은 sample 수의 A 총 wall/B 총 wall로 계산하며 조건 speedup의 산술평균으로 대체하지 않는다. 최악 sample, 조건별 max/p95 변화, 최대 N=f0000_r1의 CPU/wall·stage·memory를 별도 강조한다. 이상치를 사후 삭제하지 않는다.
7. Fresh-worker 메모리 pass는 조건/구현별 독립 worker에서 import/input/common/Core high-water와 차이, packet/evidence/lookup/cover/audit 배열 bytes를 동일 경계로 기록한다. Darwin bytes/Linux KiB 변환을 유지한다. Core 증분이 0이어도 할당 0으로 해석하지 않는다.
8. 별도 `tracemalloc` pass에서 traced current/peak·snapshot의 allocation count/bytes를 기록하고 tracing overhead는 timing 표본에서 제외한다. Snapshot은 살아 있는 할당, traced peak는 추적 가능한 peak이며 총 allocation churn/native Qhull 메모리의 완전 측정이 아니다. Supercover 임시 배열의 수·logical bytes·peak live 추정을 별도로 기록하고 measured/estimated를 구분한다. 기본 native 메모리는 RSS에 포함된다.
9. Original/optimized semantic packet·candidate·accepted/rejected·audit hash를 저장한다. 파일 SHA와 dtype/shape/value digest를 구분한다. NaN이 있는 보존 배열은 기존 byte/equal-NaN 계약을 따르며 timed metadata/timing을 hash에 섞지 않는다. 상세 audit 재현 pass의 입력/source hash와 결과도 보존한다.

과거 `116.208ms`는 참고 열로 보존한다. 정확한 speedup은 이번 동시 A/B 재계측으로만 주장한다. 50ms는 end-to-end 예산 참고값이며 Edge 최적화 PASS 기준으로 강제하지 않는다. Correctness PASS와 성능 개선/미개선은 별도로 보고한다. Mean 개선 여부와 함께 최대 N·최악 latency·memory 퇴행을 숨기지 않는다.

## D. 결과 Markdown·정적 HTML·JSON/NPZ/PNG 검증 계획

아래 파일은 **후속 구현 승인 후** 생성한다. 현재 제출물은 이 계획 Markdown 하나다.

| 산출물 | 경로/내용 |
| --- | --- |
| 결과 Markdown | `docs/visual-graph-integration/05c_ede_exact_optimization.md`: 변경·선택 근거, 실제 profile, exact 결과, 공정한 성능/메모리, 실행 한계 |
| 정적 HTML | `experiments/<new_run_id>/ede_exact_optimization.html`: baseline/optimized 표와 그래프, 동일 Edge overlay, 모든 근거 상대 링크 |
| Summary JSON | `ede_exact_optimization.json`: config/environment, baseline source, 조건별 N/U/A/E·exact 판정·stage 통계·speedup·최악 latency |
| Raw timing JSON | `timings_raw.json`: warmup/AB·BA schedule, 모든 wall/CPU/stage sample, 별도 common/C/profile scope |
| Profile/Memory JSON | `profiling.json`, `memory.json`: 하위 비용, profiler overhead, memory measured/estimated/미지원 범위 |
| 수치 NPZ | `ede_exact_optimization.npz`: 양 경로 candidate/origins/accepted/rejected/edge_index·원본 보존 배열, cover/pair는 offset 기반 ragged 배열 |
| Packet/조건 기록 | `packets/*_baseline.npz`, `packets/*_optimized.npz`, `records/*.json`: metadata·receiver readback·모든 reason/count/witness |
| PNG | `plots/`: 대표 동일 overlay, boundary/corner/Mask/Depth witness, CPU 단계, N/U별 latency, frame speedup·worst latency, memory·audit 분리 |
| 검증 기록 | `verification.json`, `tests.log`, `source_hashes.json`, source snapshot, implementation diff, 기존 dirty 상태와 protected hash inventory |

HTML은 baseline EDE와 optimized EDE의 최종 accepted/rejected·edge_index 동일성을 먼저 보여준다. 대표 frame은 사전에 frame0 최대 N/collinear chain, frame45, frame120과 두 ratio로 정하고, 모든 16조건 exact 표를 함께 제공한다. 같은 Point·색·축·viewport에서 동일 Edge overlay를 나란히 표시하고 zero-diff 여부를 수치로 연결한다. 긴 Edge나 f0000의 오염된 plateau 연결을 최적화 과정에서 제거하지 않으며 기존 물리 정확성 한계를 유지한다.

Supercover/Mask/Depth/Fraction oracle 검증, 단계별 병목 분포, frame·Node 수별 latency, 평균 speedup·최악 latency, memory/temp allocation, Core 안 reason/witness와 외부 Detailed Audit의 비용을 모두 표시한다. Plot·저장 비용은 C에 따로 표기하고 A/B에 합쳐 speedup을 만들지 않는다.

결과 Markdown→HTML은 `../../experiments/<new_run_id>/ede_exact_optimization.html`, HTML→결과 Markdown은 `../../docs/visual-graph-integration/05c_ede_exact_optimization.md`로 양방향 연결한다. JSON/NPZ/PNG/tests/verification/source snapshot은 report 기준 상대경로로 연결한다. 계획 단계에는 아직 없는 결과 파일로 실제 Markdown 링크를 만들지 않는다.

검증 순서는 다음과 같다.

1. 새 Logger를 만들기 전에 입력 identity/hash와 snapshot binding, 16조건 completeness를 확인한다. 실행 전후 기존 네 run의 artifact·sidecar·보호 입력 hash 및 dirty 파일을 비교한다. 변경을 허용하는 것은 승인된 구현/test/runner/새 결과 문서뿐이며 과거 artifact 갱신은 허용하지 않는다.
2. 원본 재현→optimized exact test→실제16조건 3방향 비교를 모두 통과시킨 후 benchmark를 진행한다. 실패 시 condition/pair/field별 diff를 기록하고 성공 수치를 발표하지 않는다.
3. NPZ를 `allow_pickle=False`로 다시 읽어 key/dtype/shape/bytes와 ragged offsets를 검사한다. Packet을 다시 읽어 metadata, receiver/deep-copy/+3 mapping, aggregate Edge와 exact 일치를 확인한다.
4. JSON을 parse하여 run identity, 16조건, 반복 수·순서·통계 재계산, exact 항목·audit witness의 serialized 값과 원본 배열을 대조한다. Hash 일치만으로 필드 검증을 대신하지 않는다.
5. PNG 전부 decode/비어 있지 않음을 검사하고 대표 overlay·corner/depth witness·stage/latency/memory plot을 실제 시각 검토한다. 검토 파일·hash를 기록한다. 미검토 그림은 시각 PASS로 묶지 않는다.
6. Markdown/HTML의 모든 상대 `href/src`와 anchor를 확인하고 외부 CDN/JS 의존성 없이 열리게 한다. Browser 검토 실행 여부를 별도로 기록한다. 미실행 검사는 `NOT_EXECUTED`로 남긴다.
7. Immutable/source/input/기존 artifact hash 재검사 후 verification과 결과 문서를 확정한다. Timing·memory·audit·실행 한계를 구분하며 물리 연결 정확성이나 GNN 20Hz를 검증한 것으로 해석하지 않는다.

후속 실행은 기존 `utils.experiment_logger.ExperimentLogger(team='visual', job_type='evaluate', ..., seed=0)`로 **새 UUID를 한 번만** 생성한다. 기존 project/group 우선순위·run ID 규칙을 그대로 쓰고 `WANDB_MODE=disabled`로 로컬 기록만 한다. 같은 평가의 worker/report 단계는 새 Logger를 만들지 않는다. Source run 네 개는 `record()` metadata에 명시하고 identity 필드를 덮어쓰지 않는다. 생성 결과만 `link_output`으로 등록하며 기존 입력·snapshot/sidecar 소유권은 바꾸지 않는다. W&B sync/HF 게시·dependency 설치·Git commit은 실행하지 않는다.

**중단 지점:** 이 계획 제출 후 구현 승인 전까지 코드 변경, 새 profiling/benchmark/test/실험, 결과 Markdown/HTML/UUID 생성으로 진행하지 않는다.
