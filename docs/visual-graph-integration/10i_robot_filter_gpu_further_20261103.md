# Task 10-I 로봇 필터 9.6ms 이후 GPU 픽셀 압축

## 최종 결과

512² 두 카메라 로봇 필터 단독 중앙값은 **8.979ms**다. 동일 실행의 직전 구현은 **9.625ms**였으며,
**0.645ms / 6.71% 추가 단축**됐다. 15개 repeat를 짝지어 비교해도 모두 새 구현이 빨랐고,
paired 차이의 중앙값은 0.611ms, 범위는 0.461–0.906ms였다.

| 구현 | 입출력 | n | 중앙값 ms | Q25–Q75 ms | 최소–최대 ms |
|---|---|---:|---:|---:|---:|
| 직전 9.6ms 구현의 동시 재측정 | GPU→GPU | 15 | 9.625 | 8.648–9.702 | 8.353–9.983 |
| **이번 GPU 픽셀 압축** | **GPU→GPU** | **15** | **8.979** | **8.028–9.034** | **7.847–9.207** |
| 이번 구현, CPU 호환 경계 | CPU→CPU | 15 | 9.539 | 8.922–9.620 | 8.221–9.817 |

기록 흐름은 **47.0 → 13.4 → 16.4 → 10.9 → 9.6 → 9.0ms**다. 과거 수치는 당시 실행 기록이고,
앞의 CPU→CPU 수치들과 GPU→GPU 수치를 동일 입출력 계약으로 표현하지 않는다.
이번 추가 개선율은 위 표의 동일 GPU 경계끼리 계산했다. CPU 경계의 직전 구현은 이번에 함께 재측정하지 않았다.

초기 3회 시험의 약 7.7ms는 최종 15회 결과가 아니다. 여러 실행/설정에서 얻은 개발 수치로 별도 보관했고,
좋은 값만 선택해 최종 결과로 사용하지 않았다. 두 구현 모두 실행 중 약 1ms 수준의 변동을 보여 paired 결과도 보존했다.

## 남긴 변경

고해상도에서 각 링크의 projected bounds 안에 있는 **raw Peg/Box mask union 픽셀 인덱스를 GPU에서 먼저 압축**한다.
이후 clipping이 연속된 유효 row list를 읽어, 빈 마스크 lane에 수행하던 float64 plane 계산을 줄인다.

- row buffer는 CPU에서 알 수 있는 bounding-box 픽셀 수만큼 고정 용량으로 할당한다.
  CUDA counter와 block prefix sum/atomic reservation으로 채워 실제 픽셀 수를 CPU로 가져오지 않는다.
  동적 Torch `nonzero`, `.item()`, mask/카운터 readback이 없다.
- ray GEMM, 링크별 numerator GEMV, 각 픽셀의 plane 곱셈/덧셈/나눗셈/감산 순서와 float64 정밀도를 유지한다.
  확정/불확실 판정, 2.5mm/6mm threshold와 0.5mm interior margin, overlap 제거는 유지한다.
- 픽셀의 실행 순서는 달라질 수 있지만 depth는 atomic min, interior는 OR로 합친다.
  diagnostics owner는 depth가 모두 결정된 다음 원래 링크 인덱스의 최솟값으로 정해 동률 의미를 유지한다.
- row list와 count는 카메라마다 새로 준비한다. pose/depth/mask의 프레임 간 재사용은 없다.
  기존 calibration ray cache와 프레임 내부 hull/geometry 공유는 유지한다.
- 기본 `compact=True, clip_pixels=32`다. `compact=False`는 직전 masked-lane 경로를 남겨 대조할 수 있게 했다.
  8/16/64 tile은 진단 옵션이며 작은 해상도의 기존 grouped classifier는 그대로다.

수정은 기존 private robot-filter 실험 폴더의 필터, benchmark, 관련 테스트와 README뿐이다.
SAM2 모델/추론, gate/Shape-First/Graph 정책, 공유 simulator, 다른 작업의 sampling 코드는 수정하지 않았다.

## 시험 후 제외한 후보

링크 Qhull 병렬화(4개 CPU worker)는 3회 시험에서 기존 8.718 → 후보 9.707ms로 느려 **최종 코드에서 제거**했다.
원인을 별도로 분리 계측하지 않았으므로 GIL/라이브러리 lock/스케줄러 중 무엇 때문이라고 단정하지 않는다.

GPU tile 8/16/32/64도 각각 3회 시험했다. 8은 32와 비슷한 범위였고, 16/64에서 더 좋은 안정적인 결과를 확인하지 못해
32를 유지했다. 이것으로 32가 모든 장면의 최적 tile이라고 주장하지 않는다. 개발 trial의 원본 source/config/results를 보존했다.
사전 cProfile은 직전 로봇 필터 호출 20회만 진단했고, 그 시간을 최종 latency에 섞지 않았다.

## 동일한 측정 방식

기존 RTX 3090 / AMD EPYC 7543에서 저장 입력 `10i_s1_0921_t080`, native 512²,
actual SAM2 mask repeat 0, `base_camera` + `hand_camera`의 합산 시간을 사용했다.
mask·depth·intrinsic/extrinsic·현재 FK world vertices SHA는 직전 실험과 모두 일치한다.
직전 구현은 **9.637ms를 기록한 실행의 원본 source snapshot**을 별도 모듈로 불러 비교했다.

각 후보 warmup 3회 / timed 15회, 정순/역순 교대, 기존 `cuda_pair()`의 `perf_counter()`와 타이머 앞뒤
CUDA synchronize 위치를 유지했다. Torch intra-op 4, OMP/OpenBLAS/MKL 각 1, GC 활성화, 80GiB data limit이다.
Torch 2.14.1+cu130 / CUDA 13.0 / Triton 3.8.0이다. 추가 CPU worker는 최종 구현에 없다.

현재 hull/bounds 준비, 현재 world ray 회전, depth/calibration/geometry 업로드, GPU row 압축과 할당,
clipping/clean mask 생성 및 GPU 완료가 타이머 안이다. calibration cache는 양쪽 모두 warmup에서 채운다.
GPU→GPU 후보의 저장 raw mask 업로드는 SAM2가 이미 GPU에 생성한 출력을 모사해 타이머 밖이다.
CPU 호환 후보는 raw mask 업로드와 clean mask readback도 타이머 안에 포함한다.
파일/decode/join·별도 FK·SAM2/grounding·gate·Shape-First·Graph·action·검증/로깅은 제외했다.
최종 45개 timed 출력은 모두 기준 clean mask와 정확히 일치했다. 검증 readback은 타이머 밖이다.

별도 dense stress는 같은 센서 입력에서 object 0 mask를 전체 화면으로, object 1을 비운 조건이다.
3회 중앙값은 기존 GPU 37.130 → 새 GPU 36.192ms였으며 이번 진단에서는 성능 저하가 관찰되지 않았다.
이 3회 결과는 실제 mask의 15회 결과와 섞거나 모든 밀도의 성능 보장으로 해석하지 않는다.

## 정확도와 연결 검사

| 검사 | 결과 |
|---|---|
| actual 저장 mask 112조건 / 224카메라 | PASS |
| full-mask 128/512/768/1024, 4조건 / 8카메라 | PASS |
| clean mask, confirmed/uncertain/candidate, owner, interior | 평가 영역에서 정확히 일치 |
| predicted robot depth | 총 232카메라 최대 절대 차이 **0.0m** |
| CUDA 계약 검사 6개 | PASS: 기존 경계/동률/invalid/pose/intrinsic/연결 검사 + GPU 압축 순서·용량·tile 검사 |
| GPU mask 경계 단위 검사 1개 | PASS: batch 호출, raw gate 통계와 예외 복원 |
| downstream 연결 128/512/768/1024의 4조건 | PASS: gate/records/sources/packet·위치 정확히 일치, feature 차이 0 |

추가 압축 검사는 empty/sparse/full mask와 tile 8/16/32/64에서 각각 3회 실행해, 순서가 바뀌어도
기존 dense 경로의 모든 diagnostics/clean tensor와 정확히 일치함을 확인했다.
`NoReadback` guard 아래에서도 통과했고 입력 mask는 수정되지 않았다.
고해상도 actual diagnostics는 기존 `evaluated_pixels`에서 비교했고 full-mask stress는 전체 화면을 비교했다.

Mac 로컬은 CPU tracker 검사 2개 통과, CUDA 관련 7개 skip이다. CUDA 계산과 위 검증은 원격에서 실제 실행했다.
simulator backend와 실제 SAM2 추론은 이번에도 실행하지 않았다.

## 비용·범위와 기록

새 row buffer의 용량은 카메라별 `4 × sum(link bbox pixels)` byte이며 counter가 추가된다.
GPU 작업 중 임시 메모리를 더 쓰는 방식이다. dense 입력에서 압축 자체의 비용이 생길 수 있어 위 별도 진단도 수행했다.
선택된 한 저장 시점의 warm latency이고, 장면별 속도 분포/다른 해상도의 속도/cold 비용/전체 perception→action 지연은 미측정이다.
이번 추가 개선 폭은 앞선 단계보다 작다. 9.0ms가 절대 한계라는 주장이나 다음 개선 폭의 보장은 하지 않는다.

공통 `utils.experiment_logger.ExperimentLogger` 재사용, W&B disabled, publisher 미호출이다.
최종 latency run ID: `367cea53f9b0424b92421d82a99b046b`.
원래 정지된 Task10-I 7개 process는 T(stopped)를 유지했다. 원래 실험을 재개하지 않았다.
새 학습·rollout·모델 API·설치·게시를 하지 않았으며 새 검사/측정 process는 모두 종료했다.

로컬/원격 최종 source 일치와 회수한 **155개 파일 SHA256**을 검증했다.
필터 SHA256: `e762aec1a7964b809fd93723d13010b3aa2418fe33e387a6a7cd6cfe00c95667`.
로컬 기준 commit `a6568c61e5b8bf823cd0b3adbaaddd1b002cd5ec`, 미커밋 변경은 source snapshot으로 보존했다.
원격은 기존 Git 없는 frozen source와 격리 overlay를 사용했다.

[최종 45개 원자료](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/final_512/results.json) ·
[설정·입력/소스 SHA](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/final_512/config.json) ·
[dense 3회 진단](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/dense_512/results.json) ·
[actual 정확도](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/validation_v1/results.json) ·
[full-mask 정확도](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/full_mask_stress/results.json) ·
[downstream 검사](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/boundary/results.json) ·
[완료·원래 process 상태](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/completion.json) ·
[SHA·paired 개선 검증](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/local_verification.json) ·
[9.6ms 구현과 차이](../../outputs/task10i/robot_filter_gpu_further_20261103_v1/changes_from_9_6.patch) ·
[재현 driver](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/benchmark_refined.py).

원격 결과: `/workspace/task10i_robot_filter_gpu_further_20261103_v1`.
동일 기존 환경/Python에서 `benchmark_refined.py --reference /workspace/task10i_robot_filter_gpu_refinement_20261103_v1/final_512/source/gpu_filter.py`를
실행하고 `TASK10I_GPU_REFINE_OUTPUT`을 새 디렉터리로 지정하면 재현할 수 있다.
