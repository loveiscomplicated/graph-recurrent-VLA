# Task05-D — GPU-Aware EDE Benchmark & Feasibility Plan

상태: **계획만 작성, 구현·실험 승인 대기**. 조사일: 2026-10-25 KST. 현재 branch HEAD `63c3f6c`, dirty/untracked 상태를 포함한 작업 트리 기준이다.

권고하는 첫 단계는 **Task05-C optimized CPU EDE를 그대로 둔 채, 저장된 동일 후보의 ragged bbox를 int64 SAT로 batch 처리하는 평가 전용 Torch 경로**다. 로컬 MPS부터 연산 지원을 확인한다. Vast.ai `57585591`은 RTX 3090을 확인했으나 PyTorch 실행 환경이 아직 준비되지 않았다. 두 backend의 지원 여부와 성능은 별도로 판정한다. Grid Traversal은 bbox + SAT 측정 이후의 별도 후보로 남긴다.

이번에 수행한 것은 source/API 열람, 장치·설치 metadata 조회, 저장 JSON/NPZ의 작업량 재집계뿐이다. GPU tensor 생성·정수 연산 probe·kernel 실행·benchmark·profiling·test suite·새 Logger/UUID 생성은 하지 않았다. 결과 Markdown/HTML도 아직 생성하지 않는다.

## A. 실제 환경과 지원 여부

### 1. 확인한 사실

| 항목 | 로컬 MPS 대상 | Vast.ai `57585591` CUDA 대상 |
| --- | --- | --- |
| OS / architecture | macOS 27.0.1 build 26A434, Darwin 27.0.0, arm64 | SSH 실측 Ubuntu 24.04.5 LTS, Linux 5.15.0-191-generic, x86_64 |
| CPU | Apple M4 Pro, 14코어(성능 10 + 효율 4) | AMD EPYC 7452 32-Core Processor. `lscpu` 64 logical CPU; Vast API의 effective CPU는 8. 실제 affinity/cgroup quota는 승인 후 기록 |
| GPU | Apple M4 Pro GPU 20코어, Metal 지원 | `nvidia-smi`: NVIDIA GeForce RTX 3090, compute capability 8.6 |
| 메모리 | 시스템 통합 메모리 24GB. 전부를 GPU 가용 메모리로 간주하지 않음 | 전용 VRAM 24,576MiB, 조회 시 free 24,124MiB. 가용량은 실행 때 다시 확인 |
| Python | repository root `.venv/bin/python`, 3.11.13 | `/venv/main/bin/python` 3.12.15, system `/usr/bin/python3` 3.12.3 |
| 설치 PyTorch | 2.14.1, CUDA build `None` | 기본 venv와 system Python 모두 PyTorch 없음 |
| backend 조회 | `mps.is_built=True`, `mps.is_available=True`; `cuda.is_available=False` | 하드웨어/driver만 확인. PyTorch CUDA 사용 가능 여부는 미확인 |
| 수치 패키지 | NumPy 2.4.6 / SciPy 1.17.1 | 기본 venv에 NumPy·SciPy·Triton 없음. system에는 NumPy만 있고 SciPy·Triton 없음 |
| CUDA / 도구 | CUDA·nvcc·Triton 없음 | driver 580.178.04, `nvcc --version` toolkit 12.8 / V12.8.93. `ncu` 경로 존재, `nsys` 경로 조회는 미발견. 도구 실행·권한·counter 지원 미검증 |
| timing / memory API | 설치 Torch에 `torch.mps.Event`, `synchronize`, `current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory` 존재 | Torch가 없어 event/memory/profiler API의 실제 사용은 미검증 |
| 현재 판정 | **GPU 가용성 확인 / EXACT GPU 지원 미검증** | **GPU 하드웨어 확인 / `ENVIRONMENT_NOT_READY`** |

초기 sandbox 안에서는 MPS 가용성이 False였지만, 권한 환경 변경 후 같은 root interpreter로 True를 확인했다. 이를 backend의 int64 지원 증거로 확대하지 않는다. `PYTORCH_ENABLE_MPS_FALLBACK`은 조회 시 미설정이었다. GPU 메모리 API 호출과 event 생성·실행은 하지 않았다.

Vast 제어 API의 `os_version=22.04`와 실제 컨테이너 `/etc/os-release=24.04.5`는 구분하며 위 표는 SSH 실측값을 사용한다. Driver/toolkit 버전은 설치 Torch CUDA runtime 버전이 아니다. SSH proxy는 접속이 닫혔으나 direct SSH로 읽기 전용 조회에 성공했다. `/workspace`에는 안내 문서만 확인했고 프로젝트/입력은 전송하지 않았다. 인스턴스 운영 안내 `/etc/vast-agents-guide.md`를 열람했으며 서비스·설정·환경을 변경하지 않았다. Instance의 stop/restart/destroy도 하지 않는다.

Vast.ai를 실행 대상으로 지정한 것은 dependency 설치·자료 업로드 승인으로 해석하지 않는다. 기존 준비된 환경이 발견되면 그 환경을 우선 검사한다. 그렇지 않으면 설치 패키지와 버전 및 저장 입력의 원격 복사 범위를 별도 제시하고 승인받아야 CUDA 평가가 가능하다. 이번에는 설치나 복사를 하지 않는다. 과거 PCA CUDA 실험의 다른 instance `57410033` 환경을 현재 `57585591`의 설치 상태로 대체하지 않는다.

### 2. 현재 프로젝트의 실행 방식과 재사용 경계

- Task05-C는 root `.venv`의 NumPy/SciPy CPU 평가다. [`extended_delaunay_edges.py`](../../visual/runtime/extended_delaunay_edges.py)의 `build_extended_delaunay_edges(..., optimized=True)`가 opt-in이며 기본 `optimized=False`는 유지한다.
- [`ede_exact_optimization_eval.py`](../../visual/runtime/test/ede_exact_optimization_eval.py)의 `snapshot()` / `optimized_runner()`는 Task05-B source snapshot의 execute·receiver를 별도 module namespace로 적재하고 optimized builder만 명시적으로 연결한다. CPU 기준선 A도 같은 binding을 재사용하되 새 runner의 helper로 적재하며 기존 `main()`/`benchmark()`/`profiling()`은 호출하지 않는다.
- [`triangle_intra_object_edges.py`](../../visual/runtime/triangle_intra_object_edges.py)의 `prepare_validation`, `validate_cover`, `adjacent_pairs`가 실제 검증 계약이다. EDE는 기존 `PairEvidence`를 조회한다. Depth gate·normal·feature를 다시 계산하는 경로가 아니다.
- [`benchmark_pca_cuda.py`](../../visual/runtime/test/benchmark_pca_cuda.py)에는 기존 CUDA event·전송 포함/resident 비교 사례가 있다. Timing 관례만 참고한다. 저장 입력만 허용하는 이번 작업에서 PCA benchmark/streaming은 호출하지 않는다.
- [`pca_torch.py`](../../visual/runtime/test/pca_torch.py)는 float64 PCA 계약 때문에 MPS를 명시적으로 거부한다. 이는 정수 EDE의 MPS 지원 여부와 다른 문제다. 이번에는 PCA를 재실행하지 않는다. GPU-resident EDE가 현재 MPS Vision 전체 경로의 존재를 증명하지도 않는다.
- [`graph/prototype/v4/train.py`](../../graph/prototype/v4/train.py)에는 MPS 모델 실행과 `deterministic_algorithms(..., warn_only=True)` 사례가 있다. EDE exact 평가에는 warn-only를 그대로 사용하지 않는다. GNN/BC 코드는 수정·실행하지 않는다.
- [`visual/pyproject.toml`](../../visual/pyproject.toml)은 Linux x86_64/Python 3.12/Torch 2.8.0+cu128/Triton 3.4.0 환경을 정의한다. 선언된 dependency가 Vast에 설치됐다는 의미는 아니다. `setup.sh`, `uv sync`, pip 설치를 자동 실행하지 않는다.

### 3. 승인 후 기능 probe: float 대체와 CPU fallback 금지

설치 Torch/backend별로 아래 작은 독립 연산을 먼저 확인하고 `backend_support.json`에 dtype·입력·정답·device·예외·fallback 여부를 저장한다. 이 probe도 현재는 실행하지 않았다. MPS와 CUDA의 PASS는 서로 전파하지 않는다.

| Probe | 확인할 내용 |
| --- | --- |
| int64 저장·왕복 | signed int64의 0, ±1, 2³¹ 경계, 2³² 초과, 2⁵³ 초과, int64 extrema를 그대로 저장·gather·readback. float 경유에 의한 정밀도 손실 검출 |
| 정수 SAT 연산 | add/subtract/negate/abs, 원소별 multiplication, min/max, inclusive `>=`/`<=`, bool AND, division/remainder. SAT dot는 원소별 정수 곱 두 개와 합으로 표현하며 integer matmul 지원을 전제하지 않음 |
| indexing / lookup | int64 index, bool mask, gather/index_select, slice, searchsorted, cumsum, 필요한 scatter. 빈 배열·sentinel `-1`·마지막 pixel·non-contiguous 입력과 index 범위 검사 |
| ragged 작업 | bbox offsets, flatten/unflatten, Edge 및 chunk 경계, variable-length cover/pair, prefix compaction. `.item()`, nonzero 등 host sync 유발 지점 기록 |
| 수집과 순서 | stable ordering 또는 유일한 정수 key에 의한 명시 정렬, segmented count/min/top4, canonical directed Edge. 동일 입력 반복·chunk 경계 변경에서도 byte/order exact |
| 기존 Depth evidence | 저장 float32 delta의 copy/gather/isfinite와 bool jump를 확인. NaN/±Inf/±0/유한 큰 값 포함. gate 산술·threshold를 GPU에서 재계산하지 않음 |
| timing | backend event 생성·record·synchronize·elapsed_time 지원과 빈 event 비용, 동기화 wall 비교. API 존재와 실제 timing 동작을 구분 |
| fallback / determinism | 새 프로세스에서 `PYTORCH_ENABLE_MPS_FALLBACK=0`를 Torch import 전에 적용. 가능한 strict deterministic mode 사용. device assertions와 별도 dispatch trace로 예기치 않은 CPU 산술 검출 |

CPU candidate 생성, 호스트 크기 계산, Python 진단 객체 포장은 명시적으로 남는 host 작업이다. 이를 CPU fallback과 구분한다. Device validation이라고 표기한 구간의 SAT·mask·pair 판정을 CPU로 우회해서는 안 된다. unsupported op를 float32/float16 또는 int32로 조용히 치환하지 않는다. 이 1차 계획의 SAT dtype은 **int64**다.

지원 실패 시 `EXACT_GPU_PATH_UNSUPPORTED`와 실패 op/dtype/backend/범위를 기록하고 해당 경로의 성능 채택을 중단한다. 장치가 없으면 `GPU_UNAVAILABLE`, dependency가 없으면 `ENVIRONMENT_NOT_READY`, 결과 불일치면 `CONTRACT_FAIL`로 구분한다. B1 통과·B2 미지원처럼 단계별 상태도 허용한다. CPU fallback 측정값은 GPU kernel 성능으로 보고하지 않는다. Custom Metal/CUDA/Triton이 있어야만 해결되는 경우에는 별도 재계획하며 자동 설치·구현하지 않는다.

## B. 고정 입력과 저장 결과 확인

### 4. 입력 identity와 CPU reference

| 작업 | 정식 UUID / 역할 |
| --- | --- |
| Task03-K | `9185355a32b8433abc70f1738f983377` / frozen selection·source alignment |
| Task04 | `9f90293b227e4dbfaaa6290afad9cc3b` / 원본 packet·저장 runtime evidence |
| Task05 | **`91ecd0413a87455cabc5e813cddaea8d`** / 기존 삼각형·component 근거 |
| Task05-B | `9f36f7c89dbd4097a8be9358138d1537` / EDE snapshot·기존 후보/출력 |
| Task05-C | `185bc034567d429daf3f1d09f94dc56c` / optimized EDE exact 기준선 |

잘못 제시된 Task05 ID `91ecd0413a87455cabc5e8138d1537`는 입력으로 사용하지 않는다. Frames **0,18,27,45,56,66,80,120**, keep ratio **0.5,1.0**, phase **0**, 총 **16조건**을 그대로 쓴다.

Task04의 `packets/<condition>.npz`, `runtime_evidence/f<frame>.npz`를 read-only로 적재하고, Task03-K의 selection freeze 및 Task05/05-B/05-C의 input hash·sidecar·저장 출력을 대조한다. Point, Camera/EE XYZ, F=5, validity, provenance, observed component, source index와 candidate/final Edge는 고정이다. Sampling/PCA5/feature extraction/EE transform·component 재분할·episode/simulator를 실행하지 않는다.

현재 [`05c 보고서`](05c_ede_exact_optimization.md)와 저장 JSON의 참고값은 original CPU **117.751ms**, optimized CPU **106.007ms**, speedup **1.1108×**, exact **16/16 PASS**다. optimized 단계 평균은 candidate **5.009ms**, supercover **35.512ms**, Mask **12.098ms**, Depth **32.390ms**, receiver **8.449ms**다. 기존 `delaunay_ms=3.706ms`에는 Qhull뿐 아니라 그 분기 내 triangle/edge 처리도 들어가므로 순수 Qhull 시간이라고 부르지 않는다.

CPU candidate 비중의 과거 참고값은 `5.009/106.007 ≈ 4.73%`다. GPU 검증이 짧아지면 이 비중은 변할 수 있다. 새 비교는 **각 backend host에서 CPU A와 GPU B를 함께 재측정**하며 Mac CPU의 106.007ms를 원격 CUDA의 분모로 쓰지 않는다. 원격 SciPy/Qhull이 같은 후보를 재현하지 못하면 그 환경의 pipeline 비교는 중단한다. Frozen candidate kernel 시험만 통과한 것을 전체 pipeline exact로 확대하지 않는다.

소스 확인 결과 optimized builder SHA-256은 `86248cbeed4502e8aca5b2e6120a90fa5349591e4c4b80f555fef6eb4770dd9d`로 Task05-C evaluated source와 일치한다. 현재 eval helper는 evaluated hash와 다르지만 저장된 report refresh hash `6298c301638d1d496f90d06de0321d92a91f4de1ee95f2de66157f312406145b`와 일치하며, 기존 기록에는 benchmark Core 함수 변경 없음이 명시돼 있다. 승인 후 source snapshot·실제 import binding·현재 hash를 다시 검증한다. 이 조사에서 CPU EDE를 재실행한 것은 아니다.

### 5. bbox 작업량의 read-only 재확인

Task05-C [`ede_exact_optimization.npz`](../../experiments/185bc034567d429daf3f1d09f94dc56c/ede_exact_optimization.npz)의 `original/provenance/source_index`, `original/image_shape`, `optimized/audit/candidates`, `optimized/cover_offsets`, `optimized/pair_offsets`만 읽었다. Candidate endpoint의 UV를 `u=source%W, v=source//W`로 읽어 **inclusive bbox `(abs(du)+1)*(abs(dv)+1)`**를 합산했다. Cover/pair는 저장 offsets의 마지막 값이다. Raster·Delaunay·검증 함수는 실행하지 않았다.

| Condition | N | Candidate U | bbox cells | cover cells | right/down pair occurrences |
| --- | ---: | ---: | ---: | ---: | ---: |
| f0000_r0.5 | 950 | 2587 | 91339 | 26383 | 25191 |
| f0000_r1 | 1694 | 4711 | 117472 | 36742 | 34087 |
| f0018_r0.5 | 453 | 1299 | 43709 | 13081 | 12514 |
| f0018_r1 | 807 | 2344 | 71443 | 18227 | 16928 |
| f0027_r0.5 | 539 | 1497 | 33620 | 11842 | 11049 |
| f0027_r1 | 883 | 2500 | 41966 | 15816 | 14393 |
| f0045_r0.5 | 514 | 1365 | 25940 | 11344 | 10768 |
| f0045_r1 | 830 | 2255 | 29990 | 14558 | 13327 |
| f0056_r0.5 | 371 | 995 | 21207 | 8619 | 8198 |
| f0056_r1 | 636 | 1740 | 26032 | 11602 | 10662 |
| f0066_r0.5 | 424 | 1109 | 14948 | 7022 | 6450 |
| f0066_r1 | 630 | 1660 | 18424 | 9164 | 8206 |
| f0080_r0.5 | 477 | 1198 | 16812 | 8526 | 7908 |
| f0080_r1 | 747 | 1922 | 21323 | 11397 | 10219 |
| f0120_r0.5 | 446 | 1153 | 22902 | 9890 | 9277 |
| f0120_r1 | 694 | 1808 | 28851 | 13213 | 12117 |
| **합계** | — | **30143** | **625978** | **227426** | **211294** |

이 값은 16조건의 각 Edge 처리량을 더한 것이며 unique image pixel 수가 아니다. `bbox/cover=2.75244695`, `cover/bbox=36.3313%`다. Pair도 같은 관측 pair를 여러 Edge가 참조하는 occurrence 수다. 공유 pair cache는 도입하지 않는다.

전체 30143 Edge에서 bbox cells의 min/median/p95/p99/max는 **2/8/68/140/8342**, cover cells는 **2/6/18/30/182**다. 최대 N 조건의 bbox 합계는 117472셀이다. 큰 bbox로 인한 padding 낭비와 Edge별 불균형을 조사할 근거이며 GPU speedup 예측은 아니다. NPZ SHA-256은 `2dc0fca8eeac770ca3ed9f77c860c81d14bc47ed147ee2d2a701b5b62bb18836`이다.

## C. 최소 GPU 후보와 단계별 구현 범위

### 6. 비교 경로와 승인 단위

| 경로 | 처리 범위 | 성능 해석 |
| --- | --- | --- |
| **A: CPU optimized reference** | Task05-C와 같은 packet copy, grouping·Delaunay/fallback, supercover, 전체 Mask/Depth 검증, 모든 reason/count/top4 witness, packaging, 기존 receiver | 같은 host에서 재측정하는 기준선 |
| **B1: GPU Supercover만** | A와 같은 CPU 후보. GPU ragged bbox + int64 SAT + ordered cover 생성. Cover를 CPU로 반환해 기존 `validate_cover`, packaging, receiver 수행 | 최소 GPU 평가. Kernel 이득과 cover readback을 포함한 hybrid pipeline 이득을 분리 |
| **B2: GPU Validation까지** | A와 같은 CPU 후보. GPU cover + Mask/valid/overlap + right/down PairEvidence 검증 + counts/top4 witness + accepted/rejected. 필요한 결과를 반환해 같은 packet/receiver 수행 | 별도 구현·exact gate·측정 단계. B1 결과를 B2 성능으로 해석하지 않음 |
| **C: GPU Exact Grid Traversal** | bbox + SAT 측정 결과를 확인한 뒤 비교 필요성 결정 | 첫 GPU 구현에 자동 포함하지 않음 |

**권고하는 첫 승인 단위는 backend/연산 probe와 B1**이다. B1의 전체 16조건 exact·timing·memory를 검토한 뒤 B2 구현을 별도 단계로 판단한다. B2를 실행하지 않았으면 Mask/Depth GPU화의 효과는 미측정이다. B1만으로 후술 Q4에 답했다고 하지 않는다. CUDA 환경 준비와 저장 입력의 원격 복사도 별도 명시 승인이 필요하다.

B1은 평가 adapter, ragged bbox/SAT/compaction, 기존 CPU validator 연결이 필요하다. B2에는 인접 pair 생성, evidence lookup, segmented reduction, ordered witness, GPU Edge packaging과 readback schema의 구현·exact test가 추가된다. B2의 구현 부담이 더 크며 backend별 미지원 op가 있으면 재계획해야 한다. B2를 단순한 B1 실행 옵션이나 비용 없는 성능 상한으로 취급하지 않는다.

### 7. B1의 최소 표현: ragged bbox + SAT

1. CPU `candidate_edges(..., extended=True)`를 재사용한다. Grouping, collinear chain, two-point fallback, Qhull options, candidate/origin 순서를 유지한다. Pipeline 측정에서는 매번 fresh 생성한다. Kernel 단독 시험만 저장 후보를 고정 입력으로 사용하며 이 차이를 표시한다.
2. CPU adapter는 endpoint UV, object/candidate ID, bbox 길이·prefix offsets·전체 셀 수를 정수로 준비한다. 동적 tensor 크기를 결정하는 host 작업이며 `cpu_adapter_ms`와 H2D bytes에 포함한다. CPU에서 SAT나 cover를 미리 계산하지 않는다. Device의 bbox min/max·width/height와 host size metadata의 일치를 exact gate로 확인한다. Device에서 크기를 정하는 다른 구현은 readback/sync 비용까지 포함한 별도 후보로 남긴다.
3. **1조건을 1batch**로 처리한다. `offsets[U+1]`와 flat cell ID를 이용하고, 설치 Torch의 int64 searchsorted 등으로 flat ID→Edge→bbox 내부 row/column을 대응시킨다. `U×H×W`나 `U×max_bbox`의 dense padding은 만들지 않는다. 여러 frame/ratio를 묶어 throughput만 높인 결과를 step latency로 보고하지 않는다.
4. GPU에서 `d=p1-p0`, `a=(-dv,du)`, `P0=2*p0·a`, `P1=2*p1·a`, `c=2*cell·a`, `r=abs(a_u)+abs(a_v)`를 **int64 원소별 곱과 합**으로 계산한다. `c+r>=min(P0,P1)` 및 `c-r<=max(P0,P1)`를 inclusive하게 판정한다. Axis가 모두 0이면 bbox를 유지한다. CPU optimized와 같은 single normal, integer centre, closed half-pixel cell 계약이다.
5. `(candidate_id, local_row_major_id)` 순서에 따라 bool membership을 prefix compaction해 `cover_uv:int64`, `cover_offsets:int64`를 생성한다. 안정적 compaction이 없으면 유일한 정수 key로 순서를 명시 복원한다. Sort/scan/offset과 임시 메모리는 GPU Supercover 비용에 포함한다.
6. B1은 ordered cover 전체를 일괄 readback하고 CPU offset slice로 기존 validator에 전달한다. Edge별 GPU dispatch/readback은 피한다. 모든 reason/count/witness, accepted/rejected, 최종 Edge dtype/bytes/order를 A와 비교한다.

초기 후보는 **Torch eager batch 연산**이다. 여러 kernel이 실행되므로 단일 fused kernel이라고 부르지 않는다. `torch.compile`, CUDA Graph, custom extension, Metal shader, Triton 구현은 첫 범위에 포함하지 않는다. 설치된 `torch.mps.compile_shader` API의 존재가 custom kernel 구현 승인을 뜻하지 않는다.

저장 입력은 최대 117472 bbox셀/조건이므로 먼저 조건 전체를 처리하는 표현을 평가한다. Tensor별 logical bytes를 사전 계산하고 **scratch 상한 64MiB**를 고정한다. 상한 초과 시 고정 65536셀 chunk 후보를 재검토하되 측정 중 batch 크기를 탐색하지 않는다. Chunk 사용은 별도 config/variant로 고정하고 Edge를 가로지르는 membership/pair/witness 합성 및 순서를 exact 재검증한다. OOM 때 조용히 chunk나 CPU로 바꾸지 않는다.

**Bare SAT membership**과 **bbox 전개→SAT→ordered compaction까지의 Supercover**를 별도 측정한다. CPU full supercover 시간을 GPU bare SAT 시간으로 나눈 값을 동일 작업량 speedup으로 쓰지 않는다. 주요 kernel 비교 경계는 같은 후보에서 ordered cover가 완성될 때까지다.

### 8. B2: Mask·Depth까지 이동하는 별도 단계

입력은 raw object Mask, valid-depth bool, overlap bool, support, 기존 PairEvidence의 pixels/supported/float32 delta/bool jump 및 right/down lookup이다. 현재 validator는 Camera XYZ/Depth에서 gate를 다시 계산하지 않고 저장 valid/evidence를 조회한다. 각 배열의 **상주 위치·dtype·bytes·사용/비사용·전송 여부**를 기록한다. Point/XYZ/F=5를 GPU에 보내지 않아도 CPU 원본 확인과 receiver 비용은 유지한다. GNN용 Point/Feature 전송은 이번 EDE 실험의 측정 범위 밖으로 명시한다.

GPU 검증은 다음 계약을 그대로 재현해야 한다.

- Cover의 image 내부 여부를 먼저 판정한 뒤 Mask/valid/overlap을 gather한다. 안전한 gather를 위한 clamp가 image 밖 셀을 유효하게 만들면 안 된다. Bbox 내부라도 cover 밖 셀은 검사하지 않는다.
- 같은 Edge cover 안에 있고 양쪽 모두 image 안에 있는 right/down pair만 생성한다. Row wrap과 다른 Edge로의 연결을 금지한다. CPU `sorted((source_a,source_b))`와 같은 사전순을 복원한다. Right 전체 다음 down 전체인 생성 순서는 그대로 정답이 아니다.
- `support[a]&support[b]`가 거짓이면 pair_unsupported다. Supported pair만 lookup하고 index<0이면 pair_missing이다. Sentinel `-1`을 마지막 evidence로 gather하지 않는다. Present pair만 delta finite와 저장 jump를 조회한다.
- Reason 순서는 **image_outside → mask_outside → invalid_depth → overlap → pair_unsupported → pair_missing → delta_invalid → depth_jump**다. 마지막 두 사유도 독립적이다. Early exit로 복수 사유를 없애지 않는다.
- GPU 고정 schema는 Edge별 accepted bool, 사유별 전체 count, ordered top4 witness·실제 길이, cover/enumerated/supported pair count를 포함한다. Pixel witness는 row-major cover, pair witness는 사전순에서 선택한다. Atomic append/scatter 도착 순서에 의존하지 않고 순서 기반 rank/min으로 결정한다.
- CPU에서 고정 schema를 기존 Python list/dict/ndarray로 복원하는 비용은 `host_result_decode_ms`로 계상한다. 기존 policy metadata·accepted origins·packet packaging·receiver를 유지한다. Count=0인 사유나 padding witness를 기존 결과에 추가하지 않는다.
- GPU directed Edge는 `(src_row,dst_row)` 사전순 canonical unique, 양방향, int64 `[2,E]`로 생성한다. CPU `np.unique(...,axis=0).T.copy()`와 최종 열 순서까지 일치해야 한다.

초기 B2가 CPU `prepare_validation`으로 right/down map을 만들면 그 전처리와 전송을 pipeline에 포함한다. GPU lookup 생성은 별도 variant이며 기존 support 일치·duplicate 검증까지 재현하고 build 비용을 resident 시간에 포함한다. Accepted 결과나 검증 완료 상태를 미리 저장해 GPU validation을 대체하지 않는다.

Depth gate `abs(z_b-z_a) > 0.01 + 0.02*min(z_a,z_b)`는 저장 PairEvidence의 기존 정의로 유지한다. GPU는 저장 jump를 읽는다. Gate 산술 순서 변경, threshold 변경, 원거리 cutoff 추가는 이번 범위가 아니다.

## D. Exact Equivalence와 정수 범위

### 9. Overflow 안전성과 지원 domain

CPU optimized guard는 Python integer로 `8*M*M+4*M <= INT64_MAX`를 검사한다. M은 endpoint 절대좌표의 최댓값이다. Normal 성분≤2M, doubled vertex/centre 성분≤2M, 개별 곱·partial sum·dot≤8M², radius≤4M으로 차분·negation·doubling·`c±r`·bbox `hi+1`까지 안전하게 제한한다.

저장 16조건은 **M=468**, 보수적 상한 **1,754,064**다. 현재 입력이 작다는 사실을 int64 미지원 시 int32로 바꿀 근거로 쓰지 않는다. Flat count/offset/pair key/source index의 곱, prefix 합, 정렬 key에도 별도 Python integer 범위 검사를 적용한다. UV가 작아도 합성 key/cumsum의 안전성이 자동 보장되지 않는다.

승인 후 probe는 2³¹/2³²/2⁵³을 넘는 안전한 중간 곱을 CPU NumPy int64 및 Python arbitrary-precision 정수와 대조한다. 큰 좌표는 짧은 segment나 scalar SAT 중간값으로 검사해 거대한 bbox를 할당하지 않는다. Guard 안/경계/밖, unsigned→signed, INT64_MIN abs/negation, 곱·prefix overflow를 포함한다. Overflow 뒤 wrap 값에 의존하는 SAT는 성능 후보로 채택하지 않는다.

CPU는 guard 밖에서 원래 raster로 위임한다. GPU는 exact 안전성을 보장하는 domain을 별도 명시하고, guard 밖/필수 int64 op 미지원은 `EXACT_GPU_PATH_UNSUPPORTED`로 중단한다. CPU fallback을 호출해 GPU PASS로 보고하지 않는다. Unsupported domain을 빼고 전체 입력 지원이라고 주장하지 않는다.

### 10. 세 방향 exact 비교

승인 후 **저장 Task05-C optimized artifact ↔ 같은 host의 CPU A ↔ GPU B**를 다음 필드별로 대조한다. 전체 16조건 통과 전에는 성능 후보로 채택하지 않는다.

| 대상 | Exact assertion |
| --- | --- |
| 후보 | Candidate undirected pair, origin, triangle, component status/row/source/chain/omitted/coplanar의 dtype/shape/값/순서 |
| Cover | 각 Edge의 pixel membership과 **row-major 순서**, offsets와 cell count |
| Pair | 전체 ordered right/down pair, evidence index, supported/missing, delta finite, jump. B1도 CPU validator 결과와 full audit 대조 |
| 판정 | 전체 candidate accepted bool, accepted/rejected 순서, reason list 순서, 모든 count, 사유별 최대4 witness dtype/shape/bytes/order |
| Edge/receiver | Directed int64 `[2,E]` bytes·열 순서, accepted origin/policy, receiver deep-copy/metadata 및 기존 robot +3 mapping |
| 저장 불변량 | Point/Camera/EE XYZ, F=5/compact feature, validity, provenance, source/row/object/component mapping, transform/metadata |

비교할 때 양쪽 출력을 sort해 순서 차이를 지우지 않는다. 필요한 출력 정렬은 timed 처리 안에서 완료하고 완성 결과를 직접 비교한다. NaN은 기존 bytes/equal-NaN 계약을 필드별로 유지하며 허용오차를 늘리지 않는다. Timing/allocation/backend 식별자는 실험 metadata로 분리한다. Task05-C의 semantic audit/raster_work를 삭제하거나 GPU scratch 값으로 덮어쓰지 않는다. GPU 진단은 별도 필드로 기록한다.

저장 NPZ에 ragged cover/pair가 있으므로 정답 준비를 위해 새 raster 알고리즘을 만들 필요가 없다. CPU 재실행도 source/module binding을 기록한다. 불일치 시 첫 condition/candidate/field/index와 기대값·실제값·hash를 저장하고 `CONTRACT_FAIL`로 중단한다. 최종 Edge 집합만 같으면 PASS가 아니다.

### 11. Synthetic 계약 테스트 계획

기존 [`test_triangle_intra_object_edges.py`](../../visual/runtime/test/test_triangle_intra_object_edges.py)의 Fraction oracle과 [`test_extended_delaunay_edges.py`](../../visual/runtime/test/test_extended_delaunay_edges.py)의 optimized fixture를 재사용한다. 평가 전용 GPU test를 추가하되 기존 기대값은 바꾸지 않는다.

- Horizontal/vertical/diagonal, positive/negative slope, 역방향, zero direction, 음수 UV, closed half-pixel boundary, corner-touch. Small exhaustive Fraction oracle와 row-major 순서 확인.
- Collinear chain의 projection/source/row tie-break, 비균일 간격, row permutation, two-point fallback, empty/singleton, object/component 분리, 거의 collinear지만 cross≠0, Qhull error/omitted, cocircular 후보 순서.
- Image 상하좌우 경계·image 밖 cover, Mask hole, bbox 안/cover 밖 실패, invalid depth, overlap, unsupported/missing PairEvidence, delta NaN/Inf, depth-jump boundary.
- Threshold equality/nextafter의 기존 CPU fixture로 만든 PairEvidence를 GPU가 그대로 조회하는지 확인. Threshold를 GPU에서 재정의하지 않음.
- 복수 reason, count>4, top4의 4/5 경계, pixel/pair 순서, w=1, 빈 accepted/rejected, 최대 bbox, chunk 경계를 가로지르는 Edge.
- **8.431m plateau**와 frame0의 기존 연결을 저장 정답대로 유지. Depth cutoff/outlier 제거로 성능·정확성을 바꾸지 않음.
- 전절 int64 guard/범위/overflow fixture, bool/index/scatter/reduction/compaction/sort의 반복 결정성. 난수 fixture seed=0.
- Point/Feature/provenance와 입력 buffer 불변, receiver deep-copy·metadata 거부·robot offset 회귀. Normal/GT가 후보를 바꾸지 않는 기존 회귀 유지.

후속 실행은 기존 5모듈(EDE, triangle, intra_object, vision_features, variable_rate_graph_contract)과 추가 GPU 계약 test를 사용한다. GPU 불가로 SKIP한 테스트는 PASS로 세지 않는다. Warmup/측정 반환값도 timed 구간 밖에서 exact 대조한다. Resident path 검증용 readback은 timed 종료 뒤 수행하고 실제 경로의 필수 D2H와 구분한다.

## E. Timing·전송·메모리 측정

### 12. 세 가지 상주 조건

| Scenario | 시작→종료와 비용 |
| --- | --- |
| **S1 CPU-resident Observation** | RAM의 저장 packet/evidence→packet copy·CPU grouping/Delaunay/fallback→adapter/lookup→H2D→GPU 처리→필수 D2H→host decode/packaging→**같은 receiver 검증 완료**. 실제 pipeline 주 비교 |
| **S2 GPU-resident Observation, 조건부** | 초기 관측 적재는 별도 계측. 반복 시 같은 frame의 관측만 GPU에 유지하고 fresh CPU 후보·size metadata와 후보 전송은 포함. 고정 device 후보의 device-only 구간과 CPU 후보 포함 resident hybrid wall을 모두 기록 |
| **S3 GPU Edge 유지, 조건부** | 같은 B2에서 canonical GPU edge_index 완료까지. Reason/count/witness도 GPU에서 생성. 기존 receiver 실제 경로와 별도로 D2H·host packet decode·receiver 생략 비용 표시. GNN 실행/통합은 제외 |

B1의 GPU 출력은 ordered cover뿐이며 accepted/rejected와 최종 Edge는 CPU에서 생성한다. 따라서 B1만 실행한 단계에는 S3의 GPU Edge 유지 결과가 없고 `NOT_EXECUTED`로 표시한다. GPU Output 중 accepted/rejected·Edge 계측도 B2에만 해당한다.

S2의 GPU 입력→GPU Edge 출력은 후보까지 device에 놓은 processing 구간이다. CPU Delaunay가 없어진 것이 아니다. CPU 후보에 필요한 source/component가 CPU에도 있다는 조건을 명시한다. 장래 이 배열이 GPU에만 있으면 후보용 D2H가 추가되며 그 비용은 미측정으로 남긴다.

저장 frame 반복으로 초기 전송을 나눠 부담할 수 있다는 사실과 새로운 step의 관측이 계속 GPU에 있다는 가정은 다르다. 상위 Vision이 GPU 배열을 직접 전달할 때만 step별 관측 H2D를 뺄 수 있다. S2/S3를 실제 구현된 Vision→Graph→GNN이라고 하지 않는다. Frame/ratio 간 초기 전송을 임의 공유하지 않으며 CPU/GPU 양쪽 중 어디에 어떤 배열이 상주하는지 기록한다.

MPS는 통합 메모리지만 NumPy→MPS tensor 생성, buffer copy, 소유권·동기화, MPS→CPU 읽기가 무료라고 가정하지 않는다. H2D/D2H는 **API 이동·materialization 비용**으로 측정하며 CUDA PCIe 전송과 구분한다. CUDA S1의 기본은 pageable CPU 메모리다. Pinned/non-blocking/overlap은 별도 후보이며 추가한다면 pin/copy 비용과 실제 overlap을 포함해야 한다.

### 13. 계측 경계

| 구간 | 측정 내용 |
| --- | --- |
| CPU candidate | Grouping/row lookup, exact collinearity/chain, two-point, Qhull call, triangle→edge unique, canonical sort. Candidate_total과 하위 시간을 중복 합산하지 않음 |
| CPU adapter/common | Selected lookup, prepare_validation, size metadata/offset, tensor wrapping, packet copy. 개별 stage와 전체 wall에 포함 |
| Host→Device | 실제 전송한 Mask/valid/overlap/PairEvidence/maps/Edge 입력. Host packing과 device copy event를 분리하고 bytes/복사 횟수 기록 |
| GPU Supercover | Bbox 전개, integer SAT, membership, prefix/ordered cover 출력. Bare SAT는 별도 하위 구간 |
| GPU Validation | Mask/valid/overlap, pair 생성·정렬, support/lookup/finite/jump 판정. B1은 GPU Validation=`NOT_EXECUTED`와 잔존 CPU Validation 명시 |
| GPU Output | Accepted/rejected, 전체 counts/top4 witness, canonical directed Edge, output compaction. Core audit 유지 |
| Device→Host | B1의 전체 cover/offset, B2의 Edge 및 동일 metadata/audit 복원용 결과. CPU 사용 가능할 때까지 동기화. 감사용 full readback은 별도 |
| Host output | Reason/witness decode, origin/policy/packet 포장, 남아 있는 CPU sort |
| Receiver | 기존 receive_vision_packet의 검증·copy·mapping |
| Total wall | RAM 입력부터 receiver 완료까지 하나의 perf_counter_ns 구간. Disk 입력·assertion·그림/저장은 제외. Process_time은 보조 기록 |
| GPU-resident | 고정 device 입력→device Edge와 CPU 후보 포함 resident hybrid wall을 구분. 초기 적재/상주 bytes 별도 기록 |

Task05-C의 common lookup/prepare는 Core 밖이었다. 새 결과는 **Core 및 common+Core 둘 다** 제공하고 S1 주 비교는 양쪽 모두 common을 포함한다. GPU만 추가 준비 비용을 밖으로 빼지 않는다.

주 pipeline pass는 시작 전 device idle을 동기화하고 같은 stream의 의존 순서대로 실행해 필요한 종료점에서 동기화한다. Stage마다 강제 동기화하지 않는다. 별도 isolated stage pass는 경계 동기화를 사용하며 계측에 따른 직렬화 영향을 기록한다. **Isolated stage 합을 Total wall로 대체하지 않는다.** CPU 부모 timer, event 하위 구간, profiler kernel 합도 중복 합산하지 않는다.

### 14. Backend별 시간 측정

**CUDA:** 실제 처리 stream에 `torch.cuda.Event(enable_timing=True)`를 기록하고 end event/device synchronization 뒤 elapsed_time(ms)을 읽는다. Host enqueue 시간과 구분한다. 전송 wall은 tensor materialization과 CPU 대기까지 포함한다. [공식 CUDA Event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html)를 참고하되 채택한 원격 Torch 버전의 API를 다시 확인한다.

**MPS:** 설치 Torch 2.14.1의 `torch.mps.Event(enable_timing=True)`, record, end.synchronize, elapsed_time을 사용할 계획이다. `torch.mps.synchronize()` 전후 wall도 별도 기록한다. 설치 Python source에서 method와 ms 단위를 확인했으나 event 작동·정밀도·짧은 구간 overhead는 승인 후 probe 대상이다. [공식 MPS Event API](https://docs.pytorch.org/docs/2.14/generated/torch.mps.event.Event.html)

Event 간격은 여러 kernel 및 eager dispatch gap을 포함할 수 있는 device 처리 구간이다. 개별 kernel duration 합은 별도 profiler로 구한다. 비동기 호출 직후 CPU wall을 kernel 시간으로 보고하지 않는다. Event가 동작하지 않으면 `device_event_ms=null / TIMING_UNSUPPORTED`로 남기고 synchronized wall만 보고한다.

Cold-start는 fresh worker의 backend/allocator 초기화·첫 shape/op 준비를 포함한 first invocation이다. Process import/input IO, 초기 device 적재, 첫 processing을 분리한다. Probe로 이미 준비된 worker를 cold라고 부르지 않는다. 초기 경로에 compiler가 없어도 lazy backend 준비 비용은 별도 기록한다.

### 15. 사전 고정 절차

1. Backend별 environment→op probe→synthetic→전체16조건 저장/CPU/GPU exact gate를 통과한다. B2는 별도 gate다. 실패 variant를 성능 후보로 채택하지 않는다.
2. Seed=0, frame 오름차순·ratio 0.5→1.0. Backend/variant/scenario별 **10 warmup + 30 paired measurements/조건**. AB/BA 각15회로 교대하며 조건 시작 순서도 교대한다. B1/B2가 모두 있으면 A/B1, A/B2를 별도 paired 비교하고 A 표본을 중복 합산하지 않는다.
3. Backend를 하나씩 실행하고 GPU training과 경쟁시키지 않는다. 타 process를 임의 중단하지 않는다. 외부 GPU 사용률, 온도·clock/power mode, CPU quota/affinity/thread, free memory를 기록한다. 전용 사용이 불가능하면 제약으로 남긴다.
4. 같은 host의 A/B에 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`을 import 전에 적용하고 Torch CPU thread=1, 같은 GC 설정을 기록한다. 설치된 API로 native pool을 확인할 수 있으면 확인하며 미확인 항목은 명시한다.
5. S1은 매번 fresh 후보·adapter/lookup·전송을 수행한다. Warmup 후 allocator cache를 유지하고 반복마다 empty_cache하지 않는다. Cover/accepted 결과를 재사용하지 않는다. S2만 관측 tensor를 유지하고 후보 전송·scratch 생성은 매번 수행한다. 사전 할당 scratch는 별도 variant로 준비 시간/bytes를 기록한다.
6. 모든 warmup/측정 반환값을 timed 밖에서 exact 대조한다. S3 감사용 readback도 시간 밖에 수행하되 다음 표본 전에 완료한다. Hash/assertion/외부 audit IO는 시간 밖, 기존 receiver 자체 검증은 시간 안이다.
7. Cold는 각 조건/variant별 독립 fresh worker first call 1회다. Steady-state와 합치지 않는다. 1회 cold로 분포를 추정할 수 없음을 명시한다.
8. Raw에 backend/scenario/scope/condition/repeat/order/warmup, N/U/cover/pair, wall/event/stage, 전송 bytes, sync점, 상주 상태를 저장한다. 전체16조건 mean/median/p95/max/표본 수를 보고한다. NumPy percentile method를 config에 고정하고 30표본 tail 한계를 쓴다.
9. 전체 평균은 16조건 등가중, aggregate speedup=`sum(A wall)/sum(B wall)`이다. 조건 speedup의 산술평균을 쓰지 않는다. 최속 조건 선택·outlier 삭제 없이 최대N=f0000_r1, 최악 sample/p95/max와 frame/ratio별 결과를 보인다.
10. 추가 profiler/memory trace는 주 측정 종료 후 별도 pass로 전체16조건 각1회 수행한다. 계측 on/off 차이와 overhead를 기록한다. 측정 뒤 반복 수/batch/chunk/audit 범위를 바꾸면 새 config·평가로 취급한다.

### 16. 메모리·launch·불균형

CUDA는 `memory_allocated/reserved`, `reset_peak_memory_stats`, `max_memory_allocated/reserved`, device free/total을 입력 상주 직후와 processing peak에서 기록한다. Output lifetime과 allocator cache 상태를 고정한다. CPU RSS와 packet/evidence/common/adapter bytes도 같은 경계로 남긴다.

MPS 설치 API에는 current/driver allocated, recommended working set이 있지만 `torch.mps.max_memory_allocated`, `reset_peak_memory_stats`, `memory_stats`는 없다. CUDA peak API가 있다고 가정하지 않는다. 별도 memory pass에서 tensor/op 경계 동기화 후 값과 logical live bytes를 수집한다. **경계 관측 최댓값은 실제 peak의 하한**이다. Instruments/Metal trace가 사용 가능하면 peak 보조 측정을 계획하고, 안 되면 `peak_device_bytes=null / PEAK_MEASUREMENT_UNSUPPORTED`로 보고한다. [MPS 메모리 API](https://docs.pytorch.org/docs/2.14/mps.html)

Endpoint/Edge ID, flat bbox, SAT projection, bool membership, prefix/offset, cover/pair, lookup, count/witness, sort scratch의 shape/dtype/logical bytes/최대 동시 live bytes를 기록한다. 추정과 allocator 실측을 분리하고 Torch 내부 scratch 전체를 logical bytes로 측정했다고 하지 않는다.

CUDA는 이용 가능한 `torch.profiler` CPU+CUDA activity로 kernel명·실제 launch 수·device duration·memcpy·implicit sync를 조사한다. CUPTI/권한 부족은 `PROFILER_UNSUPPORTED`이며 자동 설치하지 않는다. `ncu` 경로만으로 counter 사용이 검증된 것은 아니다.

MPS 설치 `ProfilerActivity`에는 MPS가 없으므로 CUDA activity를 대체 사용하지 않는다. `torch.mps.profiler.profile(mode='interval,event', wait_until_completed=False)`의 OS Signpost와 사용 가능한 xctrace/Metal trace를 계획한다. Torch op/command buffer 수와 실제 kernel launch 수를 구분한다. Kernel 단위 trace가 없으면 launch 수는 null/미측정이다. `wait_until_completed=True`는 dispatch를 직렬화하므로 주 측정에 쓰지 않는다. [MPS profiler API](https://docs.pytorch.org/docs/2.14/generated/torch.mps.profiler.profile.html)

Edge별 bbox/cover/pair의 p50/p95/p99/max·최대/중앙값, chunk occupancy, padding이 있으면 낭비율을 보고한다. Warp divergence/occupancy는 counter가 있을 때만 실측으로 제시하고 Torch duration에서 추정하지 않는다. MPS SIMD와 CUDA warp를 같은 구조로 취급하지 않는다. Peak/launch 미측정은 결과에 결측으로 남긴다.

## F. 질문별 판단과 Grid Traversal 관문

| 질문 | 답하기 위한 측정 |
| --- | --- |
| Q1 GPU bbox+SAT 속도 | 같은 후보→ordered cover의 A/B1, bare SAT 보조값, event 구간과 profiler kernel 합. 전체16조건·같은 host로 비교 |
| Q2 kernel/전송 비율 | S1의 `(H2D_wall+D2H_wall)/device_processing_interval`과 각각의 Total wall 비중·실전송 bytes. Host packing/sync 별도 표시 |
| Q3 CPU Delaunay 비중 | Grouping/Qhull/fallback/unique/sort 및 candidate_total을 분리하고 S1/S2 hybrid Total wall 비중 실측. 과거 CPU 비율 외삽 금지 |
| Q4 Mask/Depth까지 GPU가 필요한가 | Exact B1/B2의 validation-stage/S1 wall, cover readback 감소와 compact output 비용, 추가 memory/launch/구현 부담. B2 미실행이면 미답변 |
| Q5 bbox+SAT가 충분한가 | Support/contract, 전체16조건 validation/pipeline mean·p95/max, 최대N, memory/launch와 실제 잔존 병목 |
| Q6 traversal이 필요한가 | 측정 뒤 SAT/bbox 전개가 지배적인지 또는 전송/후보/Mask/Pair/receiver가 지배적인지 판단. 2.75배 셀 차이만으로 구현 결정 금지 |

**Kernel speedup**, **Validation-stage speedup**, **Pipeline speedup**을 별도 열로 기록한다. Validation-stage는 cover부터 전체 reason/count/witness/accepted 판정까지다. B1의 잔존 CPU validation과 필요한 cover D2H를 포함한 hybrid wall도 제공한다. A/B2 device 연산 비교와 이 hybrid 비교를 섞지 않는다. Pipeline 주 열은 동일 receiver를 거치는 S1이다. S3의 receiver 생략분은 `omitted_work`로 별도 측정하며 동일 end-to-end 작업량 speedup이라고 부르지 않는다.

50ms/20Hz는 최종 Vision→Graph→GNN 예산의 참고값이다. 이번에는 Vision/GNN을 실행하지 않으므로 EDE가 50ms 이하여도 통합20Hz PASS가 아니다.

Traversal 후속 제안은 exact bbox+SAT의 실측에서 SAT/bbox 전개가 주요 잔존 비용이며 pipeline에도 개선 여지가 있을 때 검토한다. SAT가 0이 되어도 후보/전송/validation/receiver가 지배한다면 이를 먼저 보고한다. 이 상한 분석은 구현 후 speedup 예측이 아니다. Bbox가 충분히 빠르거나 backend가 미지원이면 traversal로 자동 확대하지 않는다.

후속 traversal은 exact closed-cell boundary, corner-touch, 음수 slope/역방향, Edge별 variable iteration, warp/SIMD divergence, GPU memory access, output compaction, row-major 순서 복원, chunk/launch overhead를 다뤄야 한다. 순서 복원 비용을 뺀 비교는 하지 않는다. 별도 source·contract gate·구현 비용 검토와 승인이 필요하다.

판정은 **Contract**, **backend support**, **kernel 성능**, **validation-stage 성능**, **pipeline 성능**을 독립 필드로 보고한다. Contract FAIL이면 속도가 빨라도 후보에서 제외한다. Contract PASS이면서 pipeline 미개선인 경우도 명시한다.

## G. 최소 파일과 Markdown·정적 HTML 산출물

### 17. 승인 후 파일 계획

| 파일 | 역할/단계 |
| --- | --- |
| `visual/runtime/test/ede_gpu_bbox_sat.py` 신규 | 평가 전용 Torch adapter, backend guard, int64 ragged bbox/SAT/ordered cover, B1 CPU validator 연결. B2 validation/output은 별도 단계 |
| `visual/runtime/test/run_gpu_edge_benchmark.py` 신규 | 고정 artifact loader, CPU snapshot binding, support/exact gate, paired schedule, S1/S2/S3, Logger, 보고서. 기존 runner main은 확장하지 않음 |
| `visual/runtime/test/test_gpu_edge_equivalence.py` 신규 | Fraction/synthetic, int64 op/range, ordered audit/Edge/receiver, 저장16조건 exact, unsupported 중단 |
| `docs/visual-graph-integration/05d_gpu_edge_benchmark.md` 후속 신규 | 실측·재현 command, support/contract/performance 별도 판단, 제약 |
| `experiments/<new_run_id>/...` 후속 신규 | HTML/JSON/NPZ/PNG/raw samples/환경/source/검증 로그 |

초기 정수 연산은 공통 Torch로 시도하되 timing/sync/memory/profiler는 MPS/CUDA별 adapter다. 한 backend의 op가 없으면 float/CPU로 우회하지 않는다. 별도 구현이 필요할 때만 평가 폴더의 `ede_gpu_mps.py`/`ede_gpu_cuda.py`와 대응 test 추가 범위를 재제안한다. Extension/Triton/custom Metal은 별도 승인 대상이다.

`extended_delaunay_edges.py`, CPU 기본 경로, Task05-C opt-in, simulator, Graph receiver/logger/publisher, GNN/BC는 수정하지 않는다. CPU reference helper를 재사용하고 과거 source/artifact를 덮어쓰지 않는다. Production API·모델 architecture를 확정하지 않는다.

### 18. Logger와 결과 구조

승인 후 기존 `utils.experiment_logger.ExperimentLogger(team='visual', job_type='evaluate', seed=0, ...)`로 **실행마다 새 UUID**를 생성한다. 수동 UUID·과거 ID 재사용·project/group 규칙 변경·팀별 W&B logger는 없다. `WANDB_MODE=disabled`로 로컬 기록하며 sync/publishing은 수행하지 않는다. MPS/CUDA 평가는 별도 실행/UUID로 기록하고 결과 문서에 각각 연결한다. Worker는 부모 run 아래 기록하며 새 Logger를 만들지 않는다.

한 backend의 B1/B2/scenario를 같은 승인된 실행에서 측정하면 같은 run 안에 구분한다. 후일 구현/재측정은 새 run으로 source/config 차이를 보존한다. `record()`에 5개 source run/hash를 기록하고 identity를 덮어쓰지 않는다. **새 결과만 `link_output`**으로 등록해 기존 입력/sidecar 소유권을 유지한다. Artifact는 `utils.experiment_artifacts`, 향후 명시적 게시 시 `tools.publish_experiment`를 재사용한다.

| 산출물 | `experiments/<new_run_id>/` 아래 계획 |
| --- | --- |
| 정적 HTML | **`gpu_edge_benchmark.html`**. Hardware/backend, CPU 기준선, support/한계, 전체16조건 exact, kernel/validation/전송/pipeline/resident, launch/memory, failure/미실행 범위 |
| Summary JSON | `gpu_edge_benchmark.json`. Run/source IDs, config, 독립 판정, N/U/cover/pair/bbox비, stage 통계·같은 host speedup·최악 조건·scenario 가정·결측 null과 사유 |
| Raw JSON | `timings_raw.json`. Cold/warmup/전체30측정·순서·wall/event·H2D/D2H bytes·CPU candidate/adapter/validation/decode/receiver·resident/init/sync |
| 환경/지원 | `environment.json`, `backend_support.json`. OS/CPU quota/GPU/driver/toolkit/Torch, dtype/op 검증, fallback 비활성, memory/timing/profiler 제약 |
| 수치 NPZ | `gpu_edge_benchmark.npz`. A/B candidate/origin/accepted/rejected/edge_index, ordered ragged cover/pair/offset/evidence index, 불변 배열, reason/witness |
| 조건 기록 | `records/*.json`, `packets/*.npz`. 모든 reason/count/witness, component metadata, receiver readback/deep-copy/+3 확인 |
| Profile/memory | `profiling.json`, `memory.json`, `profiles/*`. Backend별 trace·실제 launch/copy/sync·tensor bytes·실측 peak/하한/추정/미지원 구분 |
| PNG | `plots/`. 전체16조건 CPU/GPU latency·단계·전송비, N/U/bbox/cover와 latency, 최대N/tail, memory/launch, 동일 Edge/witness overlay |
| 검증/재현 | `verification.json`, `tests.log`, `source_hashes.json`, `source_snapshot/`, `implementation.diff`, protected input/artifact/sidecar inventory. 실패 시에도 support/부분 결과 유지 |

모든 HTML/JSON/NPZ/PNG는 실제 backend/run/scope와 대응시킨다. 미지원/미실행을 0ms로 그리지 않는다. 대표 overlay는 frame0(최대N/plateau),45,120 및 두 ratio로 사전 고정하고 전체16조건 표를 함께 제공한다. Point·축·색을 같게 해 Edge 일치를 비교한다.

결과 Markdown→HTML은 `../../experiments/<new_run_id>/gpu_edge_benchmark.html`, HTML→결과 Markdown은 `../../docs/visual-graph-integration/05d_gpu_edge_benchmark.md`다. 결과→계획은 `05d_gpu_edge_benchmark_plan.md`, HTML→계획은 `../../docs/visual-graph-integration/05d_gpu_edge_benchmark_plan.md`다. 본 계획의 후속 결과 링크는 실제 run 생성 뒤 추가한다. 지금은 미생성 파일의 링크나 가상 UUID를 만들지 않는다. JSON/NPZ/PNG/source/검증도 상대 링크로 연결하고 외부 CDN 없이 열리게 한다.

### 19. 검증과 중단 지점

승인 후 입력5run/selection/source/hash/sidecar와 기존 dirty 파일을 보호 inventory에 기록한다. Source binding→backend/op gate→CPU reference 재현→synthetic/16조건 exact→timing→별도 profile/memory→결과 readback→링크/PNG/HTML→보호 hash 재검사의 순서다.

JSON 재parse/raw 통계 재계산, NPZ `allow_pickle=False` dtype/shape/bytes/order/ragged offsets readback, packet의 동일 receiver 검증을 수행한다. PNG 전체 decode와 대표 그림 시각 검토, 실제 HTML 열람, 상대 링크 검사를 별도로 기록한다. 미실행을 PASS로 처리하거나 부족한 backend/profiler 값을 다른 host 수치로 채우지 않는다.

**이번 제출물은 이 계획 Markdown 1개뿐이다.** 환경 metadata 조사와 저장량 집계는 새 EDE 실험이 아니다. 기존 dirty/untracked 코드와 5개 source run의 artifact/sidecar를 보존한다. GPU kernel/B1/B2/Grid Traversal 구현, GPU benchmark/profiling/연산 probe, EDE 변경/근사, Mask/Depth threshold/cutoff/outlier 추가, Point/Feature 재생성, GNN/BC 변경, simulator/episode, dependency 설치, 외부 upload, Git commit은 수행하지 않는다.

현재 문서 정적 확인에서 로컬 상대 링크 11개가 모두 존재하며, 5개 실험과 기존 변경 파일·sidecar를 포함한 1,601개 보호 파일의 SHA-256이 작업 전후 동일했다. Git 상태의 유일한 추가 파일은 이 계획 문서다. Test suite와 GPU 연산 검증은 실행하지 않았다.

**계획 제출 후 중단한다. 구현·benchmark 승인 전까지 다음 단계로 진행하지 않는다.**
