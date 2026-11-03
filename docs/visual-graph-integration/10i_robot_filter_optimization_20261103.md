# Task 10-I 로봇 필터 계산 최적화 — 측정 후 중단

## 결론과 현재 상태

**512² 로봇 필터 단독 중앙값은 기존 47.04ms → 13.42ms로 33.62ms, 71.47% 감소했다(3.51배).**
동일한 저장 입력 `10i_s1_0921_t080`, actual SAM2 repeat 0의 두 카메라를 처리한 결과다.
128²는 기존 grouped 경로를 그대로 사용하여 개선이 없었다.

변경 범위는 독립 실험 디렉터리의 로봇 필터 계산 및 해당 필터를 연결하는 private adapter다.
나머지 Graph의 gate, 특징, 선택, 3mm 중복 제거, 최종 budget, Graph 연결 로직은 기준 Stage 9를 그대로 재사용했다.
Graph 전체 시간은 영향 확인 및 정확도 회귀 검증용으로 측정했으며, 새 Graph 최적화는 하지 않았다.

사용자의 최신 지시인 **“로봇 필터만 측정하고, 측정이 끝나면 중단”**에 따라 진행 중이던 Graph 측정과 후속 단계를 취소했다.
최종 필터 단독 측정 이후 추가 실험을 실행하지 않았다. 원래 Task 10-I 프로세스는 끝까지 정지 상태로 보존했다.
센서→controller action 50ms 달성은 측정하거나 주장하지 않는다.

## 단독 필터의 측정 경계와 결과

CPU depth/calibration·사전 계산된 로봇 surface·원본 물체 마스크 → 두 카메라의 **CPU clean mask** 반환이다.
로봇 hull 준비, 필요한 ray 계산, clipping, overlap 제거, 내부 CPU↔GPU 전송과 결과 동기화를 포함한다.
Wrist gate, point sampling, 특징 추출, 중복 제거, Graph 생성, action 추론은 호출하지 않았다.
파일 읽기/압축 해제, input join, 별도 FK도 타이머 밖이다.
기존 경로는 전체 화면 diagnostics를 계산하고, 새 private 경로는 Graph에 필요한 mask union 판정만 계산한다.
외부로 비교하는 clean mask 계약은 동일하다.

해상도별·후보별 warmup 3회 후 15회 측정하고 매 반복 기준/후보 순서를 교차했다.
각 해상도는 같은 물리 시점의 저장된 native 입력과 해당 해상도의 실제 mask를 사용했다.

| 입력 | 기존 Stage 9 ms | strict ms | 절감 ms | 감소율 | 속도비 |
|---|---|---|---|---|---|
| 128² | 12.56 | 12.57 | -0.01 | -0.06% | 1.00× |
| 512² | 47.04 | 13.42 | 33.62 | 71.47% | 3.51× |
| 768² | 94.60 | 65.16 | 29.44 | 31.12% | 1.45× |
| 1024² | 153.24 | 75.30 | 77.94 | 50.86% | 2.04× |

128²의 −0.008ms 차이는 동일 구현의 측정 변동이며 개선으로 해석하지 않는다.
이 표는 **1개 Parent의 1개 시점**이다. 여러 해상도나 반복 15회를 독립 장면으로 세지 않는다.
이 사례는 이전 512 진단에서 로봇 제거 성공군에 속했으므로 모집단 성능을 대표하지 않는다.

반복 내 분산은 다음과 같다. p95나 최악 지연 추정이 아니다. 필터의 장면 간 분산은 최종 단독 측정에서 조사하지 않았다.

| 입력 | 기존 Q25–Q75 ms | strict Q25–Q75 ms |
|---|---|---|
| 128² | 12.27–12.59 | 12.37–12.61 |
| 512² | 45.73–48.39 | 13.25–14.40 |
| 768² | 92.95–96.17 | 63.79–66.11 |
| 1024² | 149.61–155.76 | 73.22–78.03 |

[최종 120개 측정 행](../../outputs/task10i/robot_filter_optimization_20261103_v1/filter_only_final_stop_v5/results.json) · [설정·환경·소스 hash](../../outputs/task10i/robot_filter_optimization_20261103_v1/filter_only_final_stop_v5/config.json) · [요약](../../outputs/task10i/robot_filter_optimization_20261103_v1/filter_only_summary.json).

## 기준 버전과 보존한 계약

기준은 `graph_latency_10i_20261103_v2.Candidate(9)`다. `fast_v5_tree`만을 최신 기준으로 삼지 않았다.
Stage 9는 128² 이하에서 기존 grouped clipping, 큰 입력에서 이전 GPU clipping을 사용한다.
고해상도 full/off 비교 역시 이 Stage 9를 사용했다. 기준 `candidates.py` SHA-256은 `6276b5576d24493867c79b6296615cbe0f0cd581cf0cc9a2b65239bebaac1374`이다.

로컬 시작 브랜치는 `archive/visual-graph-wip-20261102`, 시작 commit은 `8a0356f6bdb00bfca6731830c1b8f43b3799ce6e`였고 작업 트리는 깨끗했다.
작업 도중 다른 작업의 변경과 commit 이동을 확인했고 보존했다. 이 작업은 공통 simulator·기존 Graph 파일을 수정하지 않았다.
원격 source 디렉터리에는 Git metadata가 없어 Logger의 commit이 null이다. 이를 clean checkout의 증거로 삼지 않고,
실행 소스 snapshot과 SHA로 고정했다. 종료 시 기존 dependency 소스 36개의 SHA가 그대로임을 확인했다.

보존한 것은 각 링크의 collision convex geometry, camera ray 교차, 가장 앞선 CV-z 깊이,
2.5mm tolerance, 6mm uncertainty band, strict 0.5mm 내부 margin, 링크 순서와 동률 처리다.
물체 overlap, wrist gate, 원본 observation ID와 병합 provenance, 점 선택·3mm 중복 제거·최종 budget·Graph의 이산 출력도 유지한다.
임계값 완화, 필터 생략, FP32 전환이나 fast-math를 사용하지 않았다. 기존 오분류도 보존한다.

## 구현과 후보 비교

[필터 구현](../../graph/experiments/robot_filter_10i_20261103_v1/roi_filter.py)과
[private adapter](../../graph/experiments/robot_filter_10i_20261103_v1/adapter.py)는 팀 소유의 독립 경로에 있다.
기존 전체 화면 API는 scope 밖에서 유지된다. private `SparseDecision`은 원본 두 물체 마스크 union의 pixel ID와
그 범위의 판정만 반환하며, 계산하지 않은 영역을 정상적인 전체 화면 결과로 꾸미지 않는다.
적용 전에 pixel union이 원본과 같은지도 검사한다. patch는 예외 시에도 복구되며 한 process의 동시 builder는 지원하지 않는다.

Gate와 후속 소비자는 정리된 물체 mask를 사용한다. Wrist raw mask의 면적·valid-depth 집계는 원본을 유지한다.
필요한 픽셀에서는 모든 관련 링크의 nearest-depth와 내부 판정을 유지했다.
이미 있던 projected bbox를 새 개선으로 세지 않았고, 추가로 **mask union ∩ 링크 bbox**만 clipping했다.

| 후보 | 결과 및 판단 |
|---|---|
| roi | 원본 mask union 안에서만 clipping하고 sparse flags만 readback. 기존 전체 ray 계산·링크별 연산 배치를 유지. |
| hulls | roi + 두 카메라 사이의 build 내부 hull 재사용. 기존 Stage 3에도 있던 아이디어를 고해상도 ROI에 적용한 것이며 신규 발명으로 세지 않는다. |
| packed | compact ray·plane numerator 일괄 계산·전송 묶기. 실제 입력의 owner 계약 위반으로 기각. |
| rays | content-keyed intrinsic 벡터 cache 후보. packed 계열의 owner 계약 위반으로 기각. cache 자체의 독립 효용이나 일반적 부적합을 주장하지 않는다. |
| atomic | plane 수별 sparse clipping grouping과 정확한 depth-min/interior-OR. 초기 batched arithmetic은 동일한 owner 계약 위반으로 기각. |
| strict | 기준의 전체 ray GEMM과 링크별 numerator 연산 배치를 보존하고 sparse clipping/grouping·전송/readback 축소 적용. 128²는 Stage 9 fallback. 전체 회귀 검증을 완료한 후보. |

초기 후보들은 `0044/t080`, 1024 Main의 union 22,390픽셀 중 **29개 owner**를 11→12로 바꿨다.
깊이 차이는 최대 `1.11e-16m`였고 confirmed/uncertain mask와 Graph는 같았으나 owner의 정확 일치 계약을 위반했다.
이를 tolerance로 덮지 않았다. `strict`는 해당 차이와 후속 전체 회귀 검증을 통과했다.
[실패 진단 원자료](../../outputs/task10i/robot_filter_optimization_20261103_v1/owner_failure_v2.json) · [7조건 재비교](../../outputs/task10i/robot_filter_optimization_20261103_v1/strict_screen_v3/results.json).

고정 local shape는 이미 입력 surface에 있으며, 최종 후보는 world vertices와 hull을 **한 build 안에서만** 재사용한다.
매 build 새 필터 상태를 만들고 현재 robot pose와 wrist extrinsic을 사용한다. 이전 프레임 world geometry나 결과를 memoize하지 않는다.
local plane 변환으로 world Qhull을 대체하는 방식은 vertex rounding·plane 판정의 동일성을 증명하지 못해 채택하지 않았다.
최종 strict는 cross-frame ray cache도 사용하지 않는다. 반복 입력 결과 cache로 얻은 속도가 아니다.

512 최종 입력의 계산량과 전송량은 다음과 같다. link 합계는 여러 링크와 겹친 픽셀을 중복 집계한다.

| 카메라 | 원본 마스크 union / 전체 픽셀 | link 평가 픽셀 합 / 기존 bbox 합 | pixel×plane 계산 / bbox 기준 | CPU decision readback |
|---|---|---|---|---|
| base_camera | 11,001/262,144 (4.20%) | 3,251/73,893 | 124,012/13,228,596 | 22,002 B / 기존 4,194,304 B |
| hand_camera | 4,175/262,144 (1.59%) | 8,350/573,425 | 1,853,700/116,981,580 | 8,350 B / 기존 4,194,304 B |

세부 `array_shapes`, link별 평가 픽셀 수, 실제/padded pixel×plane 수와 선택 buffer 크기는 최종 측정 행의 `metrics`에 있다.
`selected_intermediate_bytes`는 명시된 buffer의 부분 합이며 peak memory로 해석하지 않는다.
전 화면 ray GEMM을 남긴 것은 owner의 연산 배치 보존을 위한 선택이다.

## 정확도 회귀 검증

| 범위 | 조건 수 | 카메라별 ROI 비교 | 결과 |
|---|---:|---:|---|
| 기존 26시점 × 128/512/768/1024 | 104 | 208 | PASS |
| 추가 Parent 2개 × 4해상도 | 8 | 16 | PASS |
| 전체 화면 union stress × 4해상도 | 4 | 8 | PASS |

기존 고해상도 필수 범위인 **26시점 × 768/1024의 52조건**을 모두 포함한다.
이전 표본은 성공 7·실패 4·일반 15시점, 고유 Parent 18개이며 부류 사이 Parent가 겹친다.
추가 입력은 사전에 고정한 hash/availability 규칙으로 선택한 `seed98202/t000`, `10i_s1_0844/t000`이다.
조건을 만족하는 새 Parent가 2개뿐이었다. GT를 읽어 추가 사례를 고르거나 mask를 재생성하지 않았다.
전체 mask stress만 의도적인 합성 mask이며 actual SAM2 평가와 별도로 표시했다.

ROI 안의 confirmed/uncertain/candidate/interior/owner, 전체 clean·condition mask, gate, 선택·병합 source ID,
records의 모든 field와 Graph 이산 tensor·위치·edge 속성은 정확히 일치했다.
ROI predicted depth의 기존 테스트 계약은 `atol=1e-12, rtol=0`; PCA 관련 node field만 기존 `atol=2e-5, rtol=0`를 적용했다.
실제로 관측한 최대 depth/PCA 절대 차이는 모든 위 조건에서 **0**이었다. 모든 field에 PCA tolerance를 일괄 적용하지 않았다.

추가 GPU 테스트 6개에서 tolerance와 uncertainty 경계, 0.5mm 내부 경계, 동률, invalid raw depth,
빈/전체/overlap/이미지 경계 mask, 카메라 앞뒤에 걸친 링크, intrinsic·resolution·pose·geometry cache 갱신과 scope 복구를 확인했다.
기존 optimization/connectivity suite에서 수집된 38 testcase도 통과했다.
Mac에서는 공통 Logger/Artifact 42개와 Graph CPU 7개가 통과했으며 CUDA는 실행하지 않았다.
초기 혼합 suite의 Logger 1개 오류는 imageio lazy import와 import mock 간섭이었다. 공통 suite를 분리한 42개 실행은 모두 통과했다.
실제 simulator 실행과 로봇의 실세계 정확도 향상은 검증하지 않았다.

[전체 정확도 요약·정지 상태](../../outputs/task10i/robot_filter_optimization_20261103_v1/completion.json) · [104조건 검증](../../outputs/task10i/robot_filter_optimization_20261103_v1/regression_v4/validation/) · [확장 검증](../../outputs/task10i/robot_filter_optimization_20261103_v1/extra_v4/validation/) · [GPU 테스트](../../outputs/task10i/robot_filter_optimization_20261103_v1/gpu_tests_v3.log).

## CPU 비교, 메모리와 환경

최종 두 후보만의 측정 전에 같은 저장 시점에서 CPU full/Stage 9/ROI+hulls/strict를 한 process에서 교차 비교했다.
아래 보조 실행도 warmup 3·측정 15회다. 최종 표와 다른 run이므로 서로 섞어 속도비를 계산하지 않는다.
128 strict의 이 보조 helper는 Graph adapter의 fallback과 다른 진입 방식이었으므로 최종 표의 128 재측정을 기준으로 한다.

| 입력 | CPU full ms | Stage 9 GPU ms | ROI+hulls ms | strict ms |
|---|---|---|---|---|
| 128² | 288.70 | 12.91 | 6.89 | 7.70 |
| 512² | 1289.79 | 48.22 | 13.20 | 13.90 |
| 768² | 2983.93 | 93.24 | 69.45 | 61.49 |
| 1024² | 5352.17 | 151.63 | 74.29 | 72.16 |

[CPU 포함 보조 원자료](../../outputs/task10i/robot_filter_optimization_20261103_v1/filter_t4/results.json). hulls와 strict의 작은 차이만으로 전자가 항상 느리다거나
strict가 모든 장면에서 가장 빠르다고 주장하지 않는다. strict는 더 넓은 정확도 검증을 완료한 후보다.

최종 필터 단독 실행에서 측정된 PyTorch peak allocated/reserved는 다음과 같다.

| 입력 | 기존 peak allocated MiB | strict peak allocated MiB | reserved MiB 기존/strict |
|---|---|---|---|
| 128² | 10.04 | 10.04 | 22/22 |
| 512² | 32.71 | 20.55 | 44/44 |
| 768² | 64.54 | 48.31 | 80/80 |
| 1024² | 106.40 | 71.07 | 132/132 |

Reserved는 allocator cache 상태의 영향을 받으며 전체 CUDA context나 정지된 다른 process의 GPU memory가 아니다.
필터 단독 CPU peak RSS는 기록하지 않았고, 별도 Graph 실행의 RSS를 대신 사용하지 않았다.

- GPU: NVIDIA GeForce RTX 3090, driver 595.84.
- CPU: AMD EPYC 7543 32-Core Processor; affinity 128 logical CPU, cgroup quota 6.46736 CPU.
- Python 3.12.15, Torch 2.14.1+cu130, wheel CUDA 13.0, Triton 3.8.0, NumPy 2.5.4, SciPy 1.18.1.
- Torch intra-op 4, OMP/OpenBLAS/MKL 각각 1. 80GiB data limit. GC 활성화, warm JIT/allocator 상태다.
- CPU thread 1/2, 새 cache에서의 JIT/cold 초기화, 상세 profiler와 full-mask **latency** 비교는 사용자 중단 지시로 실행하지 않았다.

## 측정 한계와 중단한 범위

128에는 속도 개선이 없고, 고해상도에서는 여전히 clipping·world hull 준비·전체 ray GEMM·CPU row 준비와 전송이 남아 있다.
단독 필터 시간의 장면별 분산, CPU thread 민감도, cold 시작 비용은 완료하지 못했다.
이번 1시점·15회로 p95, 실시간성, 센서→action 50ms, 모집단 속도 개선이나 실세계 정확도 향상을 주장하지 않는다.

Graph 최종 latency 비교는 64조건 중 **15조건**을 완료한 뒤 사용자 요청으로 종료했다.
해당 결과는 PARTIAL이며 전체 사례의 최종 성능으로 집계하지 않았다. 강제 종료된 Logger metadata도 사용자 요청 취소로 복구했다.
이후 CPU Graph·thread·cold-JIT·profiler 실행은 취소했으며 원래 정지 실험을 재개하지 않았다.
과거 off−full −139.9ms는 정책·후속 작업량 변화가 섞인 차이여서 순수 커널 시간이나 달성 가능한 절감량으로 사용하지 않았다.

## 재현과 실험 identity

모든 benchmark/evaluation은 공통 `ExperimentLogger`를 사용했고 W&B는 disabled, publisher는 호출하지 않았다.
신규 외부 모델 API, SAM2/grounding 실행, 학습, rollout, package 설치, 새 instance 생성이나 게시가 없었다.
최종 filter-only run ID는 `83dafda609fc424fabecaae2ebcc162c`다. 로컬 수신 파일 **1,477개**의 SHA를 확인했다.

```bash
TASK10I_FILTER_ENTRY=filter_benchmark.py \
  bash /workspace/task10i_robot_filter_optimization_20261103_v1/code_final_v4/remote.sh \
  --output /workspace/task10i_robot_filter_optimization_20261103_v1/new_unique_filter_run \
  --cases /workspace/task10i_robot_filter_optimization_20261103_v1/code_final_v4/cpu_threads.json \
  --variants baseline strict --warmups 3 --repeats 15 --threads 4
```

위는 재현 명령의 기록이며, 현재 추가 실행은 하지 않는다. 파일명 `cpu_threads.json`은 입력 목록 이름이고 이 실행의 thread는 4로 고정됐다.
원자료 root는 `/workspace/task10i_robot_filter_optimization_20261103_v1`이며 동일 내용을 로컬 outputs에 수신했다.
[정지·취소 기록](../../outputs/task10i/robot_filter_optimization_20261103_v1/user_scope_stop.json) · [최종 process 상태](../../outputs/task10i/robot_filter_optimization_20261103_v1/completion.json) · [로컬 SHA 검증](../../outputs/task10i/robot_filter_optimization_20261103_v1/local_verification.json) · [실험 README](../../graph/experiments/robot_filter_10i_20261103_v1/README.md).
