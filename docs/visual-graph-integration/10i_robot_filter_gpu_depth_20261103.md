# Task 10-I 로봇 필터: signed depth 전송·변환 최적화

## 최종 결과

512² 두 카메라 로봇 필터 단독 중앙값은 **7.159ms**다. 동일 실행의 직전 구현은 **7.957ms**로,
**0.797ms / 10.02% 추가 단축**됐다. GPU raw mask 입력과 GPU clean mask 출력을 유지했다.

| 구현 | 마스크 입출력 | n | 중앙값 ms | Q25–Q75 ms | 최소–최대 ms |
|---|---|---:|---:|---:|---:|
| 직전 8.98ms 구현의 동시 재측정 | GPU→GPU | 15 | 7.957 | 7.944–8.019 | 7.890–8.380 |
| **이번 GPU depth 변환** | **GPU→GPU** | **15** | **7.159** | **7.147–7.188** | **7.118–12.187** |
| 이번 구현, CPU 호환 경계 | CPU→CPU | 15 | 7.507 | 7.487–7.526 | 7.381–7.620 |

15개 paired repeat 중 14개가 빨랐고, paired 차이 중앙값은 0.788ms였다.
새 구현에서 **12.187ms인 1회**가 있었으며 제외하거나 재측정으로 덮어쓰지 않았다.
그 원인을 별도로 분리 계측하지 않았으므로 특정 GC/스케줄러/전송 때문이라고 단정하지 않는다.
median 개선으로 tail latency가 보장된다고 주장하지 않는다.

과거 8.979ms와 이번 직전 구현 7.957ms는 서로 다른 실행 기록이다. 개선율은 이번 동시 비교값으로 계산했다.
기록 흐름은 **47.0 → 13.4 → 16.4 → 10.9 → 9.6 → 8.98 → 7.16ms**다.
초기 CPU→CPU 단계와 GPU→GPU 단계를 동일 입출력 계약으로 표현하거나 그 사이의 누적 개선율을 계산하지 않는다.

## 변경

기존 표준 `CameraObservation.depth`는 signed int16 mm 배열을 CPU에서 float64로 바꾸고 `/ 1000`과
invalid 처리를 수행한 뒤 GPU로 보냈다. 이제 **원본 signed int16 배열을 GPU로 보내 float64 미터로 변환**한다.

- 양의 int16 값 1–32767에 대한 기존 NumPy float64 `/ 1000` 결과를 32768항목의 GPU lookup table에 저장한다.
  table은 filter/device당 **256KiB**이고 단위 변환만 나타낸다. pose·카메라 보정·관측 depth와 무관한 상수다.
- GPU kernel은 표를 읽어 전체 이미지의 float64 metric CV-z를 만든다.
  원본 zero/negative/overflow 값은 NaN으로 유지한다. 근사 division, float32 중간값, 보간은 없다.
- 현재 sensor array는 호출마다 새로 업로드·변환한다. 관측 depth/pose/mask나 hull의 프레임 간 캐시는 없다.
  기존 calibration ray cache와 프레임 내부 geometry 공유는 유지한다.
- 두 카메라 512²의 depth 전송량은 `2×512²×8 = 4MiB`에서 `2×512²×2 = 1MiB`로 줄었다.
  이는 payload byte 계산이며 순수 PCIe 전송 시간만 따로 분리해 측정한 수치는 아니다.
- 정확한 `CameraObservation` 타입과 native signed int16 배열에만 적용한다.
  사용자 정의 camera/subclass는 기존 `.depth` 계약을 유지하고, 이미 제공된 `depth_cuda`도 그대로 사용한다.
  `depth_from_mm=False`로 기존 CPU 변환을 대조할 수 있다.

기존 float64 ray GEMM, 링크별 numerator GEMV, GPU clipping/owner/pixel 압축과 clean 판정,
2.5mm tolerance / 6mm uncertainty band / 0.5mm interior margin, overlap 제거는 그대로다.
수정은 private robot-filter 폴더의 필터·관련 검사·benchmark·README에 한정했다.
SAM2, gate/Shape-First/Graph 정책, 공유 simulator, 다른 작업의 sampling 코드는 수정하지 않았다.

GPU depth는 필터의 임시 내부 버퍼이고 반환 인터페이스는 기존 CUDA clean mask다.
다른 모듈의 depth 복원 경로나 CPU 소비자를 변경하지 않았다. 전체 pipeline 속도 변화는 이번 결과로 주장하지 않는다.

## 수치·정확도 검증

| 검사 | 결과 |
|---|---|
| signed int16 전체 **65,536개 값** | 유효 32,767개는 NumPy 결과와 float64 bit pattern 정확히 일치, 나머지는 NaN invalid 일치 |
| 현재 sensor depth 변경·zero/overflow | GPU/기존 CPU 변환 경로의 모든 반환 판정 일치, 프레임 간 stale 값 없음 |
| CPU `.depth` property를 실패하도록 바꾼 fast-path 검사 | PASS: CPU float64 depth 변환을 호출하지 않음 |
| actual 저장 mask 112조건 / 224카메라 | PASS |
| full-mask stress 128/512/768/1024, 4조건 / 8카메라 | PASS |
| clean mask, confirmed/uncertain/candidate, owner, interior | 평가 영역에서 정확히 일치 |
| predicted robot depth | 232카메라 최대 절대 차이 **0.0m** |
| CUDA 계약 단위 검사 7개 + 경계 단위 검사 1개 | PASS: 기존 경계·동률·invalid·pose/intrinsic·mock 연결·압축/tile 검사 및 signed depth 전수 검사 |
| downstream 연결 128/512/768/1024의 4조건 | PASS: gate/records/sources/packet·위치 정확히 일치, feature 차이 0 |

고해상도 actual diagnostics는 기존 `evaluated_pixels`에서 비교했고 full-mask 조건은 전체 이미지를 비교했다.
`NoReadback` guard 아래 D2H·scalar materialization·동적 `nonzero` 없이 반환했다. 입력 mask도 수정하지 않았다.
Mac 로컬은 CPU tracker 검사 2개 통과, CUDA 관련 8개 skip이다. CUDA 계산과 위 검증은 기존 원격 호스트에서 실제 실행했다.
simulator backend와 실제 SAM2 추론은 이번에 실행하지 않았다.

## 측정 방식

기존 RTX 3090 / AMD EPYC 7543, 동일 저장 입력 `10i_s1_0921_t080`, native 512², actual SAM2 mask repeat 0,
`base_camera` + `hand_camera`의 합산 시간을 사용했다. mask·raw depth·보정·현재 FK surface SHA가 직전 실행과 모두 일치한다.
비교 기준은 **8.979ms를 기록한 실행의 원본 source snapshot**이다.

후보별 warmup 3회 / timed 15회, 정순/역순 교대, 기존 `cuda_pair()`의 `perf_counter()`와 타이머 전후
CUDA synchronize 위치를 유지했다. Torch intra-op 4, OMP/OpenBLAS/MKL 각 1, GC 활성화, 80GiB data limit이다.
Torch 2.14.1+cu130 / CUDA 13.0 / Triton 3.8.0을 사용했다.

lookup table과 calibration cache 최초 생성은 warmup 안이다. 현재 hull/bounds 준비, world ray 회전,
현재 sensor/geometry 업로드, GPU depth 변환, row 압축/clipping/clean 및 GPU 완료는 타이머 안이다.
GPU→GPU 후보의 raw mask는 SAM2가 이미 생성한 CUDA 출력을 모사해 타이머 전에 올린다.
CPU 호환 후보는 raw mask 업로드와 clean mask readback도 타이머 안에 포함한다.
파일/decode/join·별도 FK·SAM2/grounding·gate·Shape-First·Graph·action·검증/로깅은 제외했다.
최종 45개 timed 출력은 모두 기준 clean mask와 정확히 일치했고 검증 readback은 타이머 밖이다.

별도 전체 화면 mask 진단 3회 중앙값은 기존 GPU 35.345 → 새 GPU 34.856ms였다.
실제 mask 15회와 섞지 않았고 모든 밀도/해상도의 성능 보장으로 해석하지 않는다.

## 개발 후보·한계

먼저 GPU에서 정확한 float64 division을 수행하는 후보를 구현해 전수/장면 검증을 통과했다.
그 코드와 검증은 `division_code/`, `validation_v1/` 등에 보존했다.
최종 코드는 lookup 방식이며 `validation_v2/`, `full_mask_stress_v2/`, `boundary_v2/`가 해당 검증이다.
두 방식의 각각 3회 개발 수치를 최종 결과에 섞지 않았고 둘의 속도 우열을 확정하지 않는다.

이 개선은 **CPU에 있는 표준 signed sensor depth가 들어오는 조건**에 해당한다.
이미 float64 `depth_cuda`를 공급하는 경우에는 이번 변환 최적화를 거치지 않는다.
256KiB/device의 상수 표와 일시적인 int16 GPU 입력 버퍼가 추가된다.
선택된 한 저장 시점의 warm latency이며, 장면별 latency 분포·cold 비용·다른 해상도의 속도·전체 perception→action은 미측정이다.
추가 개선 폭이나 최적화의 절대 한계를 주장하지 않는다.

## 기록·재현

공통 `utils.experiment_logger.ExperimentLogger` 재사용, W&B disabled, publisher 미호출이다.
최종 latency run ID: `28ace86809ed47af8282438ce6d2e632`.
원래 정지된 Task10-I 7개 process는 T(stopped)를 유지했고 재개하지 않았다.
학습·rollout·모델 API·설치·게시를 하지 않았고 새 검사/측정 process는 모두 종료했다.

로컬/원격 최종 source 일치와 회수한 **152개 파일 SHA256**을 검증했다.
필터 SHA256: `4c7402e66752473b19e6412982e6b6e353f5c2f1877ea23b472f1b65ef63f8df`.
로컬 기준 commit `a6568c61e5b8bf823cd0b3adbaaddd1b002cd5ec`, 미커밋 변경은 source snapshot으로 보존했다.
원격은 기존 Git 없는 frozen source와 격리 overlay다.

[최종 원자료](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/final_512/results.json) ·
[설정·입력/소스 SHA](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/final_512/config.json) ·
[전수 검사를 포함한 CUDA 검사 로그](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/validation_v2.log) ·
[actual 정확도](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/validation_v2/results.json) ·
[full-mask 정확도](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/full_mask_stress_v2/results.json) ·
[downstream 검사](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/boundary_v2/results.json) ·
[dense 진단](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/dense_512/results.json) ·
[완료·원래 process 상태](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/completion.json) ·
[SHA·paired 검증](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/local_verification.json) ·
[8.98ms 구현과 차이](../../outputs/task10i/robot_filter_gpu_depth_20261103_v1/changes_from_8_98.patch) ·
[재현 driver](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/benchmark_refined.py).

원격 결과: `/workspace/task10i_robot_filter_gpu_depth_20261103_v1`.
동일 환경/Python에서 `benchmark_refined.py --reference /workspace/task10i_robot_filter_gpu_further_20261103_v1/final_512/source/gpu_filter.py`를
실행하고 `TASK10I_GPU_REFINE_OUTPUT`을 새 디렉터리로 지정하면 재현할 수 있다.
