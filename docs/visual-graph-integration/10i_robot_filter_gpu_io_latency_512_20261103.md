# Task 10-I 512 로봇 필터 GPU 입출력 변경 효과 실측

## 결론

실제 GPU 직결 경로는 **16.367ms**다. 기록상 흐름은 **47.0 → 13.4 → 16.4ms**이며,
추가 GPU 입출력 변경이 이번 입력의 필터 전체 시간을 개선하지 못했다.
이번 동일 실행의 기존 strict는 13.071ms여서 GPU 직결은 **3.296ms, 25.22% 더 느렸다**.

입력·출력까지 과거와 완전히 동일한 CPU→CPU 경계로 새 구현을 측정하면 **17.118ms**다.
따라서 이 경계로 세 번째 수치를 쓰면 **47.0 → 13.4 → 17.1ms**다.
GPU→GPU 값과 CPU→CPU 값을 같은 입출력 계약이라고 표현하지 않는다.

## 동일하게 고정한 측정 방식

- 같은 원격 RTX 3090/AMD EPYC 7543, 동일 저장 입력 `10i_s1_0921_t080`, native 512², actual SAM2 mask repeat 0.
- Main + Wrist 두 카메라 처리의 총시간. 원래 자료와 두 mask array SHA가 일치함을 로컬에서 확인했다.
- 후보별 warmup 3회, 측정 15회. 이전처럼 forward/reverse 순서를 번갈아 paired 실행.
- `time.perf_counter()` CPU wall timer와 타이머 앞뒤 CUDA synchronize. GPU 완료까지 측정하며 CPU hull 준비도 포함.
- Torch intra-op 4, OMP/OpenBLAS/MKL 각각 1, GC 활성화, 80GiB data limit. JIT/allocator 초기화는 warmup으로 제외.
- CPU 센서 depth·카메라 보정·미리 계산된 FK surface를 공통 입력으로 사용. 필터 내부 hull 준비, ray/numerator/clipping 및 내부 전송 포함.
- 파일 읽기/압축 해제, input join, 별도 FK, SAM2/grounding, gate, Shape-First, Graph와 action은 실행 시간에서 제외.
  모델과 Graph를 실행하지 않고 로봇 필터만 호출했다.

기존 두 후보는 이전 frozen `filter_benchmark.pair()`를 수정 없이 재사용했다.
새 CUDA 후보는 동일 context/cache 초기화·메모리 계측·동기화/타이머 위치에서 호출했다.
GPU 직결은 SAM2가 이미 만든 마스크를 모사해 raw CUDA mask를 타이머 전에 한 번 올려 두며,
clean mask를 GPU로 반환한다. CPU→CPU 보조 조건은 동일 새 필터를 호출하되 raw mask 업로드와 clean mask readback을 타이머 안에 포함한다.
두 새 조건 모두 CPU geometry 준비·보정/센서 업로드를 제외하지 않았다.

## 결과: 각 15회 중앙값

| 단계 | 중앙값 ms | 반복 Q25–Q75 ms | n | 마스크 입출력 |
|---|---:|---:|---:|---|
| 기존 Stage 9 | 46.553 | 45.294–47.166 | 15 | CPU→CPU |
| 기존 strict 계산 최적화 | 13.071 | 13.003–13.269 | 15 | CPU→CPU |
| 새 CUDA 구현: 이전과 동일 경계 | 17.118 | 16.730–18.308 | 15 | CPU→CPU |
| 새 CUDA 구현: 실제 GPU 직결 경계 | 16.367 | 16.133–16.513 | 15 | GPU→GPU |

과거 47.043/13.419ms와 이번 재측정 46.553/13.071ms의 작은 차이는 별도 실행 변동이다.
이전 기록을 덮어쓰거나 새 수치를 과거와 동시 측정한 것으로 표시하지 않는다.
CPU↔GPU mask 입출력을 새 구현 안에서 제외했을 때의 중앙값 차이는 약 **0.751ms**였다.
이는 입력/출력 materialization, 할당·동기화도 함께 달라진 차이로, 순수 PCIe DMA 커널 시간이라고 주장하지 않는다.

## 해석과 한계

마스크 왕복 복사를 줄였어도 필터 전체 시간이 자동으로 줄지는 않았다.
새 CUDA 경로는 기존 strict와 메모리 배치·ROI lane 처리·커널 배치가 같지 않다.
소스상 기존 strict는 두 카메라 사이의 build 내부 hull을 재사용하고 sparse clipping을 plane 수별로 묶는 반면,
새 CUDA 필터는 호출마다 hull을 준비하고 고해상도에서 링크별 masked kernel을 실행한다.
따라서 이 결과는 순수 전송 제거 하나만 분리한 통제 실험이 아니라 **현재 구현된 새 경로의 전체 효과**다.
CPU hull·커널 launch·GPU 연산의 원인별 기여 시간을 이번에 별도 계측하지 않았으므로 회귀 원인을 수치로 단정하지 않는다.

모든 60개 측정 출력은 동일 CPU clean-mask reference와 정확히 일치했다. GPU 직결 출력의 검증 readback은 타이머 밖이다.
필터 정책이나 threshold를 바꾸지 않았고 이번에는 구현을 추가 최적화하지 않았다.
1개 Parent의 선택된 1개 시점이며, 반복 15개는 독립 장면이 아니다. 모집단 성능·p95·실시간성·센서→action 50ms를 주장하지 않는다.

## 재현·실험 기록

공통 ExperimentLogger run ID: `b33728d8e154456d82c2ded709dea36f`. W&B disabled, publisher 미호출.
기존 7개 Task10-I process는 모두 T(stopped)였고 재개하지 않았다.
추가 모델 API, SAM2 추론, 학습, rollout, 설치나 게시를 하지 않았다. 다른 모듈 latency를 측정하지 않았다.
GPU 계산은 기존 원격 환경에서 실행했고 Mac 로컬에서는 CUDA를 실행하지 않았다.

[60개 원자료](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/paired_512_latency_v1/results.json) · [입력 SHA·환경·측정 경계·소스 SHA](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/paired_512_latency_v1/config.json) ·
[실행 소스 snapshot](../../outputs/task10i/robot_filter_gpu_io_20261103_v1/paired_512_latency_v1/source/) · [재현 driver](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/benchmark_512.py).

원격 출력: `/workspace/task10i_robot_filter_gpu_io_20261103_v1/paired_512_latency_v1`.
같은 환경 변수와 기존 Python을 사용해 `benchmark_512.py`를 실행하고 `TASK10I_GPU_IO_BENCH_OUTPUT`만 새로운 경로로 지정하면 된다.
benchmark와 filter 코드는 측정 후 변경하지 않았다. 결과를 보고한 뒤 추가 실험은 진행하지 않는다.
