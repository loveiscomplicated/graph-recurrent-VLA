# Task 10-I 로봇 필터 10.9ms 이후 추가 최적화

## 결과

512² 두 카메라 로봇 필터 단독 중앙값은 **9.637ms**다. CUDA raw mask 입력과 CUDA clean mask 출력을 유지했다.
기록 흐름은 **47.0 → 13.4 → 16.4 → 10.9 → 9.6ms**다. 각 과거 수치는 당시 실행 기록이다.

직전 10.861ms 구현은 이번 실행에서 **11.961ms**로 재측정됐다. 따라서 추가 최적화의 개선율은
과거 10.861ms와 섞지 않고 **동일 실행의 11.961 → 9.637ms**, **2.325ms / 19.44% 단축(1.24배)**으로 계산했다.
실행 간 차이의 원인을 별도 계측하지 않았으므로 특정 CPU/GPU 상태 때문이라고 단정하지 않는다.

| 구현 | 입출력 | n | 중앙값 ms | Q25–Q75 ms | 최소–최대 ms |
|---|---|---:|---:|---:|---:|
| 직전 10.9ms 구현의 동시 재측정 | GPU→GPU | 15 | 11.961 | 11.866–12.100 | 10.848–12.460 |
| **이번 추가 최적화** | **GPU→GPU** | **15** | **9.637** | **9.610–9.695** | **8.597–11.008** |
| 이번 추가 최적화, CPU 호환 경계 | CPU→CPU | 15 | 10.258 | 10.206–10.325 | 10.131–11.498 |

CPU 호환 경계는 raw mask 업로드와 clean mask readback을 시간에 포함한다. 이번에는 직전 구현의
CPU→CPU 시간을 함께 재측정하지 않았으므로 그 경계의 추가 개선율은 주장하지 않는다.

## 병목 확인과 변경

사전 cProfile은 직전 구현의 **로봇 필터 호출만** 20회 진단했다. hull 준비, 반복하는 투영 경계 계산과
작은 geometry/metadata GPU 업로드 비용이 드러나 이 부분을 우선 변경했다. 프로파일러가 붙은 시간을
최종 latency에 사용하지 않았고 다른 모듈의 시간은 측정하지 않았다.

변경은 `graph/experiments/robot_filter_gpu_io_10i_20261103_v1/` 안의 필터, 관련 검사, benchmark와 README에 한정했다.

1. 현재 world vertices를 원래 방식으로 카메라에 투영한 뒤, **모든 카메라에서 기존 보수적 bounding box가 빈 링크**는
   Qhull을 생략한다. original link slot은 남겨 owner 인덱스가 바뀌지 않는다.
   이번 512 입력은 전체 18개 shape 중 두 카메라 합집합에서 14개가 보였고 4개의 불필요한 hull 준비를 줄였다.
   카메라 평면을 가로지르는 shape는 기존처럼 전체 화면 bounds를 사용해 생략하지 않는다.
2. 동일 카메라 역행렬의 rigid transform 검증을 링크마다 반복하던 것을 카메라마다 한 번으로 줄였다.
   vertices의 행렬곱·이동 덧셈과 projected bounds의 수치 연산 순서는 유지했다.
3. hull equations, contiguous normals, 두 카메라의 모든 plane-group metadata를 **프레임 안에서 한 번 업로드**한다.
   int64 metadata는 bit pattern을 보존해 packing/view하며, normals는 같은 float64 값을 연속 배치한다.
   프레임이 바뀌면 geometry와 metadata를 전부 새로 준비한다.

직전 구현의 GPU clipping/owner/clean 커널, 전체 float64 ray GEMM과 링크별 numerator GEMV,
2.5mm tolerance / 6mm uncertainty band / 0.5mm interior margin, 객체 겹침 제거를 그대로 사용했다.
pose/depth/mask를 프레임 간 캐시하지 않는다. 기존 calibration ray cache만 유지한다.
화면 밖 링크가 다시 보이거나 intrinsic이 제자리에서 바뀌어도 현재 프레임 값으로 재계산한다.
SAM2, gate, Shape-First, Graph 선택/구성, 공유 simulator나 다른 작업의 sampling 코드는 수정하지 않았다.

별도 CUDA Graph 도입이나 프레임 간 geometry buffer 재사용은 하지 않았다. 프로파일에서 확인한
host 준비와 작은 업로드를 먼저 줄인 결과다. 변경별 기여 시간의 ablation은 수행하지 않았다.

## 고정한 측정 방식

동일 RTX 3090 / AMD EPYC 7543, 저장 입력 `10i_s1_0921_t080`, native 512², actual SAM2 mask repeat 0,
`base_camera` + `hand_camera`의 총 필터 시간을 사용했다. mask·depth·calibration·FK surface의 SHA가 직전 실험과 모두 일치한다.
직전 필터는 **10.861ms가 나온 실행의 원본 source snapshot**을 별도 모듈로 불러 실행했다.

후보마다 warmup 3회, timed 15회, 정순/역순 교대 순서를 유지했다. 공통 기존 `cuda_pair()`를 재사용해
`time.perf_counter()` 및 타이머 전후 CUDA synchronize 위치를 맞췄다. Torch intra-op 4,
OMP/OpenBLAS/MKL 각 1, GC 활성화, 80GiB data limit이다. Torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0을 사용했다.

두 GPU 후보 모두 calibration ray cache를 채운 필터 인스턴스를 유지한다. 현재 FK surface의 hull/bounds 준비,
현재 카메라 world ray 회전, depth/calibration/geometry 업로드, clipping, clean mask 생성과 GPU 완료는 타이머 안이다.
GPU에 이미 존재하는 SAM2 출력을 모사하는 저장 mask 업로드만 GPU→GPU 후보의 타이머 밖이다.
파일/decode/join·별도 FK·SAM2/grounding·gate·Shape-First·Graph·action·검증/로깅은 제외했다.
기준 mask와의 45개 최종 timed 출력 대조는 모두 통과했고, 검증 readback은 타이머 밖이다.

## 정확도와 연결 검증

| 검사 | 결과 |
|---|---|
| actual 저장 mask 112조건 / 224카메라 | PASS |
| full-mask stress 128/512/768/1024, 4조건 / 8카메라 | PASS |
| clean mask, confirmed/uncertain/candidate, owner, interior | 평가 영역에서 모두 정확히 일치 |
| predicted robot depth | 232카메라 최대 절대 차이 **0.0m** |
| CUDA 계약 단위 검사 5개 | PASS: empty/full/overlap/invalid depth, threshold·동률, GPU depth/비연속 mask, mock SAM2 연결, pose/intrinsic 갱신 |
| GPU mask 경계 단위 검사 1개 | PASS: batch 호출, 원래 raw gate 통계, 예외 복원 |
| downstream 연결 검사 128/512/768/1024의 4조건 | PASS: gate/records/sources/packet·위치 정확히 일치, feature 차이 0 |

고해상도 actual mask는 기존 `evaluated_pixels`인 raw mask union에서 diagnostics를 비교했다.
full-mask stress는 전체 이미지를 비교했다. `NoReadback` guard 아래 D2H·scalar materialization·동적 `nonzero` 없이
CUDA clean mask를 반환했고 입력 mask를 수정하지 않았다. 이동한 shape가 화면 밖으로 나갔다가 돌아오는 검사에서는
불필요한 hull이 생략되고, 돌아온 프레임의 hull이 새로 준비되는 것도 확인했다.

Mac 로컬은 CPU tracker 검사 2개 통과, CUDA 관련 6개 skip이다. CUDA 계산과 위 검증은 기존 원격 호스트에서 실제 실행했다.
이번에도 simulator backend나 실제 SAM2 추론을 실행하지 않았다.

## 재현·산출물

공통 `utils.experiment_logger.ExperimentLogger` 재사용, W&B disabled, publisher 미호출이다.
최종 latency run ID: `1dd9d216f68d4240ae77608fa837ea2a`.
원래 정지된 Task10-I 7개 process는 모두 T(stopped)를 유지했고 재개하지 않았다.
학습·rollout·모델 API·설치·게시를 수행하지 않았다. 새 검증/측정 process는 모두 종료했다.

로컬 source와 원격 최종 source의 일치를 확인했고, 회수한 **85개 파일 SHA256**을 검증했다.
필터 SHA256: `cc69113305cb01d651ef51f4d628232db7b64207f196b5fb6af978d996ecb968`.
기준 로컬 commit은 `a6568c61e5b8bf823cd0b3adbaaddd1b002cd5ec`이며 미커밋 변경은 source snapshot으로 보존했다.
원격은 이전과 같은 Git 없는 frozen source/격리 overlay다.

이 결과는 **선택된 한 저장 시점의 warm latency**다. 반복 15개는 독립 장면 15개가 아니며,
장면별 속도 분포·다른 해상도의 성능·cold/JIT 비용·전체 perception→action 지연은 이번에 측정하지 않았다.
개발용 3회 시험은 packing 이전 후보로 최종 수치와 섞지 않았다. 9.6ms가 최적화 한계라는 주장은 하지 않는다.

[45개 최종 원자료](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/final_512/results.json) ·
[측정 설정·입력/소스 hash](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/final_512/config.json) ·
[필터 내부 사전 profile](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/profile_before/profile.txt) ·
[actual 정확도](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/validation_v1/results.json) ·
[full-mask stress](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/full_mask_stress/results.json) ·
[downstream 검증](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/boundary/results.json) ·
[완료·원래 process 상태](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/completion.json) ·
[로컬 SHA 검증](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/local_verification.json) ·
[직전 10.9ms 구현과의 차이](../../outputs/task10i/robot_filter_gpu_refinement_20261103_v1/changes_from_10_9.patch) ·
[재현 driver](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/benchmark_refined.py).

원격 결과: `/workspace/task10i_robot_filter_gpu_refinement_20261103_v1`.
동일 환경 변수와 기존 Python으로 `benchmark_refined.py`를 실행하고 `TASK10I_GPU_REFINE_OUTPUT`을 새 디렉터리로 지정하면 재현할 수 있다.
