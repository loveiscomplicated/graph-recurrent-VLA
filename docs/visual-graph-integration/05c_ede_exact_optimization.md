# Task05-C — Exact-Equivalent EDE CPU Optimization

새 평가 UUID: `185bc034567d429daf3f1d09f94dc56c`. **Exact Equivalence PASS (16/16)**. CPU 성능 판정은 계약과 별개다.

[정적 HTML](../../experiments/185bc034567d429daf3f1d09f94dc56c/ede_exact_optimization.html) · [전체 JSON](../../experiments/185bc034567d429daf3f1d09f94dc56c/ede_exact_optimization.json) · [NPZ](../../experiments/185bc034567d429daf3f1d09f94dc56c/ede_exact_optimization.npz) · [검증](../../experiments/185bc034567d429daf3f1d09f94dc56c/verification.json)

## 결과

동일 host/interpreter에서 원본 snapshot baseline mean **117.751ms**, optimized mean **106.007ms**, aggregate wall speedup **1.1108×** (9.97% 감소). Process CPU mean은 **117.418 → 105.696ms**, speedup **1.1109×**다. 과거 Task05-B `116.208ms`는 참고값이며 이번 speedup 분모로 사용하지 않았다.

성능 개선 관측. 최악 sample baseline `f0000_r1` **446.120ms**, optimized `f0000_r1` **410.268ms**. 최대 Node 조건은 `f0000_r1`, N=1694이며 아래 표와 최대 N 그림에 별도 표시한다. 30표본 p95는 기술통계이며 안정적인 tail 추정이 아니다. Outlier 삭제 없음.

| Condition | N | U | A | Baseline mean / p95 / max ms | Optimized mean / p95 / max ms | Wall speedup | Exact |
|---|---:|---:|---:|---:|---:|---:|---|
| f0000_r0.5 | 950 | 2587 | 2081 | 173.478 / 256.580 / 365.054 | 141.700 / 156.366 / 159.325 | 1.2243× | PASS |
| f0000_r1 | 1694 | 4711 | 3715 | 289.958 / 372.254 / 446.120 | 270.139 / 409.712 / 410.268 | 1.0734× | PASS |
| f0018_r0.5 | 453 | 1299 | 1041 | 81.275 / 84.985 / 87.853 | 77.530 / 76.151 / 250.015 | 1.0483× | PASS |
| f0018_r1 | 807 | 2344 | 1830 | 146.750 / 147.245 / 307.920 | 128.524 / 131.562 / 286.761 | 1.1418× | PASS |
| f0027_r0.5 | 539 | 1497 | 1253 | 97.854 / 97.463 / 258.886 | 81.171 / 85.780 / 98.798 | 1.2055× | PASS |
| f0027_r1 | 883 | 2500 | 2062 | 152.773 / 160.560 / 172.641 | 149.477 / 312.378 / 317.843 | 1.0220× | PASS |
| f0045_r0.5 | 514 | 1365 | 1181 | 85.475 / 88.471 / 92.592 | 80.502 / 77.352 / 258.987 | 1.0618× | PASS |
| f0045_r1 | 830 | 2255 | 1918 | 142.261 / 155.220 / 318.875 | 127.946 / 212.172 / 280.655 | 1.1119× | PASS |
| f0056_r0.5 | 371 | 995 | 858 | 62.692 / 66.090 / 68.326 | 54.653 / 57.106 / 73.947 | 1.1471× | PASS |
| f0056_r1 | 636 | 1740 | 1451 | 105.310 / 110.176 / 117.722 | 102.925 / 189.497 / 278.760 | 1.0232× | PASS |
| f0066_r0.5 | 424 | 1109 | 995 | 65.863 / 67.255 / 68.048 | 62.320 / 58.604 / 212.093 | 1.0568× | PASS |
| f0066_r1 | 630 | 1660 | 1473 | 110.539 / 195.815 / 274.879 | 85.810 / 90.319 / 95.478 | 1.2882× | PASS |
| f0080_r0.5 | 477 | 1198 | 1051 | 72.739 / 76.191 / 82.204 | 64.169 / 71.208 / 73.529 | 1.1335× | PASS |
| f0080_r1 | 747 | 1922 | 1636 | 113.686 / 117.610 / 119.413 | 103.716 / 102.423 / 256.427 | 1.0961× | PASS |
| f0120_r0.5 | 446 | 1153 | 989 | 71.792 / 76.274 / 81.583 | 62.219 / 64.680 / 67.897 | 1.1539× | PASS |
| f0120_r1 | 694 | 1808 | 1512 | 111.579 / 115.730 / 117.419 | 103.311 / 101.629 / 279.458 | 1.0800× | PASS |

최대 N 조건 p95는 baseline 372.254ms → optimized 409.712ms로 악화됐다. 전체 평균 개선과 tail 개선은 별개다. Peak RSS도 개선으로 판단하지 않는다.

| Stage | Baseline mean ms | Optimized mean ms | Change ms |
|---|---:|---:|---:|
| candidate_total_ms | 4.673 | 5.009 | +0.337 |
| supercover_ms | 51.722 | 35.512 | -16.210 |
| support_validation_ms | 12.215 | 12.098 | -0.117 |
| depth_validation_ms | 30.216 | 32.390 | +2.174 |
| edge_packaging_ms | 0.829 | 0.821 | -0.008 |
| receiver_ms | 6.292 | 8.449 | +2.157 |

## 기준선·profiling 관문

Task05-B `9f36f7c89dbd4097a8be9358138d1537`의 execution source snapshot을 `_task05c_snapshot_*` 별도 module namespace로 적재했다. ImportFrom을 AST에서 명시적으로 연결하고 raster/validator/receiver/execute의 module·code filename·SHA 및 object binding을 assertion으로 확인했다. Snapshot receiver 내부 import도 같은 namespace에 연결했다. 변경 없는 dependency의 현재 SHA도 확인했다. Optimized runner는 같은 snapshot execute·receiver에 현재 builder의 `optimized=True`만 명시적으로 연결했다. Python cache나 sys.path 우선순위에 의존하지 않는다.

원본 snapshot은 먼저 저장 packet·JSON 후보/origins/accepted/checks·component 상태·NPZ 최종 Edge·원본 배열을 16/16 exact 재현했다. 저장 JSON에 없는 전체 cover/pair/PairEvidence 순서는 동일 입력으로 재생성하여 optimized와 대조했다. Candidate triangles와 component 상태, chain/fallback, accepted/rejected, reason/count/witness 순서, final int64 dtype/shape/bytes/order, receiver metadata/+3/deep-copy, F=5·Camera/EE XYZ·validity·provenance/source alignment를 필드별 assertion으로 검사했다.

선분 normal a와 -a는 `P(-a)=-P(a)`, `c(-a)=-c(a)`, `r(-a)=r(a)`다. 따라서 closed-cell inclusive 두 부등식은 부호 반전 시 서로 대응한다. 계측된 두 번째 SAT normal 합계 **159.500ms**, raster 비중 **17.54%**, 원본 gate Core 대비 **8.57%**로 profiling 관문을 통과했다. Gate 기준은 raster 5% 이상 및 Core 1% 이상이었다. 이는 절감량 예측이나 speedup 수치가 아니다.

별도 raster stage timer, cProfile, intrusive line tracing을 각기 실행했다. [profiling.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/profiling.json)은 bbox allocation·centres·첫/반대 SAT·membership, candidate/Qhull/unique/sort, Mask gather/사유, pair set/이웃/sort, support/right-down/delta/jump/witness, receiver validate/deepcopy/output 및 per-edge bookkeeping source-line 비용을 보존한다. Line time은 child 포함이며 nested 시간을 합산하지 않는다. CProfile overhead 비율 **1.396×**, line tracing **2.353×**; raster timer pass와 uninstrumented gate 차이는 **110.731ms**다. 이 차이에는 실행 변동도 포함된다.

## 구현과 overflow 안전성

`build_extended_delaunay_edges(..., optimized=True)`만 opt-in이다. 기본 `optimized=False`는 기존 raster를 사용한다. 정책 metadata·후보/검증/정렬/threshold는 동일하다. 새 traversal/근사/cache/early exit는 없다.

M을 모든 endpoint 절대 좌표의 최댓값으로 두면 각 normal 성분 ≤2M, doubled vertex/centre 성분 ≤2M, 각 dot product와 그 partial sum 절댓값 ≤8M², radius ≤4M다. **Python integer `8M²+4M <= int64_max`** guard는 차분·부호 반전·doubling·dot·`c±radius`·bbox `hi+1`까지 안전하게 제한한다. Guard 전 signed/unsigned extrema를 Python int로 읽어 wrap을 피한다. 두 normal 모두 안전한 domain에서만 하나를 생략한다. 증명 범위 밖/triangle/unsupported dtype·shape는 기존 `supercover`로 위임한다. 실제 이미지 UV는 이 범위 안이다. Large signed/unsigned fallback과 경계 부근도 테스트했다.

## 공정한 측정·memory·audit

Python `3.11.13`, NumPy `2.4.6`, SciPy `1.17.1`, `macOS-27.0.1-arm64-arm-64bit`. Qhull extension SHA와 interpreter·thread 값은 JSON environment에 보존했다.

16조건·phase 0, 구현별 warmup 3회와 조건별 30쌍, AB/BA 각 15회, 조건 시작 순서 교대. Thread 환경 OMP/OpenBLAS/VECLIB=1을 NumPy import 전 고정했다. 실제 native thread pool 조회는 `NOT_EXECUTED`다. 같은 Python GC 설정과 반환값 lifetime을 사용했다. Warmup 96회 + measured 960회 모두 timed 종료 후 exact assertion을 실행했다. Hash/assertion 비용은 timed Core 밖이며 raw에 별도 기록했다.

양쪽 Core는 packet deepcopy·fresh candidate·동일 cover/validation·모든 reason/count/top4 witness·bookkeeping·canonical Edge/metadata/packet packaging·전체 receiver 검증과 deepcopy를 포함한다. `core_audit_mode=full_reasons_counts_top4_witness`, `detailed_audit_in_core=false`. 공통 selected map/prepare_validation은 별도이며 JSON에 common+core도 기록했다. Detailed cover/pair materialization·plot·IO·tracemalloc·cProfile/line tracer는 timed Core 밖이다. Detailed audit는 전체 cover와 ordered right/down pair 및 evidence index/support/delta/jump를 양쪽 동일하게 생성한다.

RSS high-water baseline 164.86–170.72MiB, optimized 166.30–170.98MiB. 32 fresh workers에서 import/input/common/Core 경계를 동일하게 측정했다. Darwin bytes/Linux KiB 변환 적용. 0 증분은 allocation 0이 아니다. 별도 tracemalloc current/peak·live snapshot count/bytes는 native 전체나 allocation churn 측정이 아니다. Raster logical temporary 추정과 measured RSS/trace를 구분했다. 축 중복 제거로 allocation 반복은 줄지만 bbox/centres와 peak live 배열 크기는 대체로 그대로다.

## 검증·산출물

68 tests PASS: 기존 Task05/05-B·k-NN·feature·receiver 회귀, 전체 EDE fixture의 optimized 재실행, Fraction oracle의 음수·역방향·corner·degenerate, int64 guard, 여러 reason/count/top4 witness를 포함한다. 기존 기대값은 변경하지 않았다. 실제 16조건 three-way exact와 매 benchmark 반환값 exact를 별도 확인했다.

NPZ `allow_pickle=False` readback은 모든 key/dtype/shape/bytes와 ragged offsets를 원본 메모리 값과 대조했다. Packet readback·receiver 및 JSON parse/통계 재계산·기존 입력/artifact/sidecar hash를 검증했다. JSON/NPZ/PNG 상대 링크 및 Markdown↔HTML 양방향 경로도 확인했다. PNG decode `PASS`, 실제 PNG 시각 검토 `PASS`, 실제 browser 검증 `NOT_EXECUTED`. 미실행 항목을 PASS로 표시하지 않는다.

네 source run 및 기존 dirty/untracked 파일 보존 결과: `PASS`. 공통 `ExperimentLogger(team='visual', job_type='evaluate', seed=0)`를 사용했고 새 run만 link_output으로 등록했다. Sampling/PCA5/feature extraction/EE transform·simulator·GPU/PCL·학습·dependency 설치·업로드·Git commit은 실행하지 않았다. 관측 계약 PASS는 물리 연결 정확성 또는 GNN 20Hz를 확정하지 않는다. 기존 frame0 오염 plateau 연결도 그대로다.

측정 후 보강한 NPZ/JSON readback assertion·stage/tail 설명·그림 label은 `report_source_snapshot`과 별도 diff에 보존했다. 측정 source snapshot, 모든 timing sample·packet·record·수치 NPZ·profiling/memory는 보존했고 benchmark Core 함수는 바뀌지 않았다. Packet/record 직렬화의 별도 보고서 IO pass는 `audit_io_refresh.json`에 기록했다.

재현 (같은 저장 입력과 설치 환경; 새 평가는 새 UUID 생성):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task05c-mpl .venv/bin/python -m visual.runtime.test.run_extended_delaunay_edges --exact-optimization
```

[ede_exact_optimization.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/ede_exact_optimization.json)
[ede_exact_optimization.npz](../../experiments/185bc034567d429daf3f1d09f94dc56c/ede_exact_optimization.npz)
[timings_raw.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/timings_raw.json)
[profiling.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/profiling.json)
[memory.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/memory.json)
[verification.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/verification.json)
[tests.log](../../experiments/185bc034567d429daf3f1d09f94dc56c/tests.log)
[source_hashes.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/source_hashes.json)
[implementation.diff](../../experiments/185bc034567d429daf3f1d09f94dc56c/implementation.diff)
[config.yaml](../../experiments/185bc034567d429daf3f1d09f94dc56c/config.yaml)
[preimplementation_gate.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/preimplementation_gate.json)
[saved_npz_exact.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/saved_npz_exact.json)
[audit_io_refresh.json](../../experiments/185bc034567d429daf3f1d09f94dc56c/audit_io_refresh.json)
[report_verification_refresh.diff](../../experiments/185bc034567d429daf3f1d09f94dc56c/report_verification_refresh.diff)
[source_snapshot/visual/runtime/extended_delaunay_edges.py](../../experiments/185bc034567d429daf3f1d09f94dc56c/source_snapshot/visual/runtime/extended_delaunay_edges.py)
[source_snapshot/visual/runtime/test/ede_exact_optimization_eval.py](../../experiments/185bc034567d429daf3f1d09f94dc56c/source_snapshot/visual/runtime/test/ede_exact_optimization_eval.py)
[report_source_snapshot/visual/runtime/test/ede_exact_optimization_eval.py](../../experiments/185bc034567d429daf3f1d09f94dc56c/report_source_snapshot/visual/runtime/test/ede_exact_optimization_eval.py)

![audit_core_boundaries](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/audit_core_boundaries.png)

![cpu_stages](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/cpu_stages.png)

![f0000_r0.5_same_edges](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/f0000_r0.5_same_edges.png)

![f0000_r1_same_edges](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/f0000_r1_same_edges.png)

![f0045_r0.5_same_edges](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/f0045_r0.5_same_edges.png)

![f0045_r1_same_edges](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/f0045_r1_same_edges.png)

![f0120_r0.5_same_edges](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/f0120_r0.5_same_edges.png)

![f0120_r1_same_edges](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/f0120_r1_same_edges.png)

![frame_latency_speedup](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/frame_latency_speedup.png)

![mask_depth_witness](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/mask_depth_witness.png)

![memory](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/memory.png)

![node_candidate_max_latency](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/node_candidate_max_latency.png)

![raster_profile](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/raster_profile.png)

![supercover_fraction](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/supercover_fraction.png)

![tail_latency](../../experiments/185bc034567d429daf3f1d09f94dc56c/plots/tail_latency.png)
