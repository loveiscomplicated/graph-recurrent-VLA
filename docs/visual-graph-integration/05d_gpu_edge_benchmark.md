# Task05-D — Local MPS GPU EDE B1 Benchmark

평가 UUID: `be206b04f7b245c4a9436317b8f87987`. **COMPLETED**. 실제 로컬 MPS 연산을 실행했다.

[정적 HTML](../../experiments/be206b04f7b245c4a9436317b8f87987/gpu_edge_benchmark.html) · [승인 전 계획](05d_gpu_edge_benchmark_plan.md)

## 판정

| 항목 | 판정 |
|---|---|
| BACKEND_SUPPORT | PASS |
| EXACT_EQUIVALENCE | PASS 16/16, full three-way exact |
| GPU_SUPERCOVER_PERFORMANCE | GPU event/kernel latency NOT_EXECUTED; isolated synchronized ordered-cover wall measured |
| HYBRID_PIPELINE_PERFORMANCE | S1 speedup 1.4720x; isolated cover hybrid 6.3676x |
| MEMORY_AND_TIMING_LIMITATIONS | TIMING_UNSUPPORTED; PEAK_MEASUREMENT_UNSUPPORTED |

설치 Torch **2.14.1**, Apple M4 Pro GPU 20코어, device `mps:0`, dtype `torch.int64`. `PYTORCH_ENABLE_MPS_FALLBACK=0`을 Torch import 전에 적용했다. 실제 장치 조회 원문은 environment.json에 보존했다.

B1의 Mask·Depth 검증은 **CPU의 기존 validate_cover**에서 수행한다. GPU는 ragged bbox 전개, single-normal integer SAT, ordered cover/offset compaction만 담당한다. Torch eager의 여러 kernel이며 fused kernel이 아니다.

핵심 연산 probe: PASS. 실패 연산: 없음. Timing: `TIMING_UNSUPPORTED`. Event record·작업 동기화는 완료했으나 elapsed_time 조회가 20초 내 반환되지 않아 자식 프로세스를 종료했다. Event latency는 null이다. 이전 최초 probe의 native stack sample도 보존했다. 측정은 synchronized wall이며 GPU event/kernel latency로 부르지 않는다.

Dispatch 검사에서 probe 결과의 device/dtype를 확인했다. fallback 비활성화와 Python dispatch 검사는 Metal kernel trace를 대체하지 않는다. MPS signpost API는 실행했지만 Instruments capture·실제 kernel launch 수·GPU 실제 peak는 미측정이다.

## 정확성 및 측정 범위

Source UUID: `9185355a32b8433abc70f1738f983377`, `9f90293b227e4dbfaaa6290afad9cc3b`, `91ecd0413a87455cabc5e813cddaea8d`, `9f36f7c89dbd4097a8be9358138d1537`, `185bc034567d429daf3f1d09f94dc56c`.

Frames 0,18,27,45,56,66,80,120 × keep ratio 0.5/1.0, phase 0. 기존 Task05-C snapshot hash와 현재 import binding을 검사했다. Sampling·feature/PCA5·EE 변환·component 재분할·simulator는 실행하지 않았다.

실제 S1 전체 평균 wall: **CPU A 101.941 ms → B1 69.254 ms**, speedup **1.4720×**. 별도 isolated ordered Supercover hybrid speedup **6.3676×**. Kernel speedup은 미측정이다.

| 경로 | mean | median | p95 | max (ms) |
|---|---:|---:|---:|---:|
| A | 101.941 | 89.281 | 239.434 | 289.714 |
| B1 | 69.254 | 59.766 | 157.188 | 214.413 |

S1은 fresh common preparation, packet copy, CPU fresh candidate, cover, CPU Mask/Depth, packaging, 기존 receiver 완료까지 포함한다. 양쪽 조건별 warmup 10회와 측정 30회, AB/BA 각 15회를 변경 없이 수행했다. 결과 exact assertion은 timer 종료 뒤 수행했다. Isolated 단계는 별도 강제 동기화 pass이며 그 합을 실제 S1 wall로 사용하지 않았다.

## 16조건 결과

| Condition | N | U | bbox | cover | pairs | A mean/p95/max ms | B1 mean/p95/max ms | S1 speedup | Exact |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| f0000_r0.5 | 950 | 2587 | 91339 | 26383 | 25191 | 142.161/183.065/184.336 | 99.798/137.386/137.671 | 1.4245× | PASS |
| f0000_r1 | 1694 | 4711 | 117472 | 36742 | 34087 | 256.717/288.138/289.714 | 168.220/205.143/214.413 | 1.5261× | PASS |
| f0018_r0.5 | 453 | 1299 | 43709 | 13081 | 12514 | 70.923/70.802/112.620 | 49.870/54.299/91.531 | 1.4222× | PASS |
| f0018_r1 | 807 | 2344 | 71443 | 18227 | 16928 | 125.129/165.137/173.829 | 84.750/123.471/125.263 | 1.4764× | PASS |
| f0027_r0.5 | 539 | 1497 | 33620 | 11842 | 11049 | 80.818/102.275/121.111 | 54.022/53.453/95.665 | 1.4960× | PASS |
| f0027_r1 | 883 | 2500 | 41966 | 15816 | 14393 | 137.375/174.162/175.018 | 88.037/112.173/137.577 | 1.5604× | PASS |
| f0045_r0.5 | 514 | 1365 | 25940 | 11344 | 10768 | 76.976/118.452/120.555 | 50.057/51.498/52.091 | 1.5378× | PASS |
| f0045_r1 | 830 | 2255 | 29990 | 14558 | 13327 | 119.807/145.719/162.592 | 83.436/122.159/125.199 | 1.4359× | PASS |
| f0056_r0.5 | 371 | 995 | 21207 | 8619 | 8198 | 55.956/77.579/97.616 | 37.574/40.722/42.425 | 1.4892× | PASS |
| f0056_r1 | 636 | 1740 | 26032 | 11602 | 10662 | 92.435/114.545/134.960 | 63.075/85.195/104.807 | 1.4655× | PASS |
| f0066_r0.5 | 424 | 1109 | 14948 | 7022 | 6450 | 57.427/59.037/60.701 | 42.085/64.254/83.081 | 1.3645× | PASS |
| f0066_r1 | 630 | 1660 | 18424 | 9164 | 8206 | 87.039/108.035/128.235 | 58.678/80.400/101.565 | 1.4833× | PASS |
| f0080_r0.5 | 477 | 1198 | 16812 | 8526 | 7908 | 65.225/86.808/106.601 | 43.801/44.043/86.114 | 1.4891× | PASS |
| f0080_r1 | 747 | 1922 | 21323 | 11397 | 10219 | 101.322/124.126/148.055 | 70.336/108.815/111.829 | 1.4405× | PASS |
| f0120_r0.5 | 446 | 1153 | 22902 | 9890 | 9277 | 62.636/64.404/75.144 | 46.540/69.563/88.807 | 1.3459× | PASS |
| f0120_r1 | 694 | 1808 | 28851 | 13213 | 12117 | 99.102/124.807/147.355 | 67.786/91.569/110.019 | 1.4620× | PASS |

최대 Node 조건 `f0000_r1` (N=1694)은 위 표에 포함했다. 최악 sample은 A `f0000_r1` 289.714 ms, B1 `f0000_r1` 214.413 ms다. 30표본 p95는 기술통계이며 outlier를 제거하지 않았다.

3방향 비교는 candidate/origin/source/row, ordered cover/offset, ordered right/down pair와 evidence index/support/missing/delta/jump, 모든 판정/reason/count/top4 witness, final int64 Edge bytes/order, receiver metadata/deep-copy/+3, Point/F=5/validity/provenance 불변을 포함한다. Frame0 8.431m plateau와 기존 Edge를 보존했다. GPU synthetic test에는 Fraction oracle 625개 선분, 역방향·zero-length·음수 UV·closed half-pixel/corner-touch·image boundary, mask hole/overlap, invalid depth, missing/unsupported pair, threshold 경계와 fallback/receiver fixture가 포함된다.

## 메모리 및 한계

작업량은 candidate 30,143, bbox 625,978, cover 227,426, pair occurrence 211,294다. Edge별 원시 수와 min/median/p95/p99/max는 조건 records 및 NPZ에 있다. 1조건=1batch이며 dense padding/chunk 탐색을 사용하지 않았다. Scratch는 사전 선언한 `384*bbox_cells + 512*U + 4096` logical bound를 64MiB와 대조하고 초과 시 중단한다. 내부 backend scratch의 물리 peak를 직접 제한·관측한 것으로 주장하지 않는다.

memory.json의 current/driver allocated는 동기화된 op 경계 관측값이다. 실제 GPU peak는 `null / PEAK_MEASUREMENT_UNSUPPORTED`다. CPU RSS는 입력·audit/report를 포함한 단일 process high water이며 A/B1 peak 비교가 아니다. H2D/D2H는 통합 메모리에서 API copy/materialization·동기화 비용이다.

B1의 성능으로 B2, CUDA, 전체 GPU EDE, 완전한 GPU-resident Observation 또는 GPU Edge→GNN·20Hz 성능을 판정하지 않는다. B2, Grid Traversal, CUDA/RTX 3090, resident endpoint scenario, S3, GNN/BC, 외부 게시·sync·Git commit은 모두 **NOT_EXECUTED**다.

## 단계별 평균 wall

아래 S1 stage는 실제 pipeline 내부 timer이며, selected stage의 합이 전체 wall은 아니다. Isolated stage는 별도 pass의 synchronized wall이다. 이벤트 기반 kernel latency가 아니다.

| S1 stage | CPU A ms | B1 ms |
|---|---:|---:|
| candidate_total_ms | 4.759 | 5.147 |
| supercover_ms | 34.593 | 3.900 |
| support_validation_ms | 11.896 | 11.436 |
| depth_validation_ms | 30.726 | 29.569 |
| receiver_ms | 6.654 | 6.084 |
| edge_packaging_ms | 0.789 | 0.799 |

| Isolated Supercover stage | CPU A ms | B1 ms |
|---|---:|---:|
| CPU raster / B1 device processing | 31.677 | 2.727 |
| CPU ragged output packing / B1 host adapter | 0.215 | 0.742 |
| H2D API materialization | 해당 없음 | 0.821 |
| D2H API materialization | 해당 없음 | 0.717 |
| Complete ordered-cover hybrid | 31.892 | 5.008 |

최대 explicit scratch bound 45.324MiB; MPS 경계 current allocated 최댓값 11.236MiB, driver allocated 최댓값 48.703MiB. Driver 관측값에는 allocator/cache가 포함되며 scratch나 실제 peak와 같지 않다. CPU RSS high water 910.234MiB.

조건당 H2D logical bytes 범위 39,808–188,448; D2H 121,232–625,568. 조건별 값과 분포는 records/JSON/NPZ에 보존했다.

별도 fresh process의 최대 N 조건 첫 B1 cover wall은 54.803ms, Torch import는 361.235ms였다. OS/driver shader cache는 초기화하지 않은 1표본이며 steady-state 통계와 합치지 않았다. 실제 B1 dispatch/CPU profiler 및 MPS signpost를 별도 실행했다. CPU tensor 출력은 명시적 host materialization·transfer·detach뿐이며 CPU 산술 fallback은 관측되지 않았다. CPU profiler time은 GPU kernel time이 아니다.

## 산출물 및 재현

[gpu_edge_benchmark.json](../../experiments/be206b04f7b245c4a9436317b8f87987/gpu_edge_benchmark.json)
[gpu_edge_benchmark.npz](../../experiments/be206b04f7b245c4a9436317b8f87987/gpu_edge_benchmark.npz)
[timings_raw.json](../../experiments/be206b04f7b245c4a9436317b8f87987/timings_raw.json)
[environment.json](../../experiments/be206b04f7b245c4a9436317b8f87987/environment.json)
[backend_support.json](../../experiments/be206b04f7b245c4a9436317b8f87987/backend_support.json)
[cpu_baseline_gate.json](../../experiments/be206b04f7b245c4a9436317b8f87987/cpu_baseline_gate.json)
[exact_equivalence.json](../../experiments/be206b04f7b245c4a9436317b8f87987/exact_equivalence.json)
[profiling.json](../../experiments/be206b04f7b245c4a9436317b8f87987/profiling.json)
[memory.json](../../experiments/be206b04f7b245c4a9436317b8f87987/memory.json)
[verification.json](../../experiments/be206b04f7b245c4a9436317b8f87987/verification.json)
[tests.log](../../experiments/be206b04f7b245c4a9436317b8f87987/tests.log)
[source_hashes.json](../../experiments/be206b04f7b245c4a9436317b8f87987/source_hashes.json)
[implementation.diff](../../experiments/be206b04f7b245c4a9436317b8f87987/implementation.diff)
[config.yaml](../../experiments/be206b04f7b245c4a9436317b8f87987/config.yaml)
[profiles/cold_and_dispatch.json](../../experiments/be206b04f7b245c4a9436317b8f87987/profiles/cold_and_dispatch.json)
[profiles/cold_and_dispatch_probe.py](../../experiments/be206b04f7b245c4a9436317b8f87987/profiles/cold_and_dispatch_probe.py)
[profiles/b1_cpu_dispatch_trace.json](../../experiments/be206b04f7b245c4a9436317b8f87987/profiles/b1_cpu_dispatch_trace.json)
[profiles/event_elapsed_time_sample.txt](../../experiments/be206b04f7b245c4a9436317b8f87987/profiles/event_elapsed_time_sample.txt)
[report_refresh.diff](../../experiments/be206b04f7b245c4a9436317b8f87987/report_refresh.diff)
[report_refresh.json](../../experiments/be206b04f7b245c4a9436317b8f87987/report_refresh.json)

기존 ExperimentLogger(team=visual, job_type=evaluate, seed=0), WANDB_MODE=disabled를 재사용했다. 기존 dirty/untracked 작업, 5개 source run, 입력/artifact/sidecar hash 보존은 verification.json에서 확인한다. 실제 브라우저 검증은 NOT_EXECUTED이며 HTML 파싱·상대 링크 검증과 구분한다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_gpu_edge_benchmark
```

![f0000_r0.5_same_edges](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/f0000_r0.5_same_edges.png)
![f0000_r1_same_edges](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/f0000_r1_same_edges.png)
![f0045_r0.5_same_edges](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/f0045_r0.5_same_edges.png)
![f0045_r1_same_edges](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/f0045_r1_same_edges.png)
![f0120_r0.5_same_edges](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/f0120_r0.5_same_edges.png)
![f0120_r1_same_edges](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/f0120_r1_same_edges.png)
![gpu_hybrid_stages](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/gpu_hybrid_stages.png)
![h2d_d2h](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/h2d_d2h.png)
![memory_workload](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/memory_workload.png)
![pipeline_supercover_latency](../../experiments/be206b04f7b245c4a9436317b8f87987/plots/pipeline_supercover_latency.png)
