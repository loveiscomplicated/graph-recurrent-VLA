# Task 10-I 로봇 필터 GPU 경로 추가 최적화

## 결과

512²의 두 카메라 로봇 필터 단독 시간은 **10.861ms**다. GPU 마스크 입력과 GPU clean mask 출력을 유지했다.
기록 흐름은 **47.0 → 13.4 → 16.4 → 10.9ms**다. 과거 수치는 각각 당시 실행 기록이며,
아래 표에는 비교 후보를 이번에 함께 재측정한 수치를 적었다.

| 구현 | 입출력 | 이번 15회 중앙값 ms | Q25–Q75 ms |
|---|---|---:|---:|
| 기존 Stage 9 | CPU→CPU | 47.469 | 45.400–48.434 |
| 기존 strict 계산 최적화 | CPU→CPU | 13.457 | 13.103–14.346 |
| 추가 최적화 전 GPU 직결 | GPU→GPU | 16.621 | 16.441–17.093 |
| **이번 GPU 직결 최적화** | **GPU→GPU** | **10.861** | **10.714–11.409** |
| 이번 최적화, CPU 호환 경계 | CPU→CPU | 11.070 | 10.985–11.328 |

동일 GPU 입출력끼리 **16.621 → 10.861ms**, **5.761ms / 34.66% 단축(1.53배)**이다.
기존 strict 13.457ms보다도 19.30% 짧지만 이 둘은 마스크 입출력 장치가 다르다.
CPU 입출력을 맞추면 strict 13.457 → 11.070ms다.
원래 Stage 9와 CPU 입출력을 맞춘 비교는 47.469 → 11.070ms, 76.68% 단축(4.29배)이다.

## 구현 범위

수정은 기존 private GPU 로봇 필터 실험 폴더에 한정했다. 공유 simulator, SAM2 모델·추론·prompt,
gate/Shape-First/Graph 구성·선택 정책, 다른 작업의 CUDA sampling 코드는 수정하지 않았다.

- `CudaRobotFilter.filter_cameras()`가 현재 FK surface의 world vertices/Qhull을 한 번 준비하고,
  **같은 호출 안의 카메라들만** 공유한다. GPU hull equation 업로드도 같은 프레임 안에서 공유한다.
  다음 호출에서는 현재 pose로 다시 계산한다. pose/depth/mask의 프레임 간 캐시는 없다.
- 고해상도 링크별 clipping을 plane 수별 커널 묶음으로 변경했다. raw mask union은 GPU에서 직접 읽으며
  nearest depth는 atomic min, interior는 atomic OR로 합친다. diagnostics의 동률 owner는 모든 depth 갱신 후
  원래 링크 인덱스의 최솟값을 선택해 기존 순차 `<` 판정을 보존한다.
- 원래 float64 전체 ray GEMM과 링크별 strided numerator GEMV, 2.5mm tolerance / 6mm uncertainty band /
  0.5mm interior margin, 객체 마스크 겹침 제거를 유지했다. 화면 밖 링크의 numerator 계산만 생략한다.
- 카메라별 역행렬은 한 번 계산한다. CV ray만 해상도·intrinsic 내용·CUDA device로 구분해
  최대 8개 / 64MiB 캐시한다. intrinsic의 제자리 수정은 byte key로 감지한다.
  현재 카메라 world rotation/origin을 적용하는 GEMM은 매번 수행한다.
- `GpuMaskBoundary`가 기본 필터의 `filter_cameras()`를 호출하도록 연결했다. GPU clean mask 반환 이후의
  기존 CPU 소비자 경계와 raw-mask gate 통계는 유지했다. 사용자 정의 단일 카메라 callable도 계속 지원한다.

사용 예:

```python
robot_filter = CudaRobotFilter()  # 프레임 사이에는 calibration ray만 유지
clean_cuda = robot_filter.filter_cameras(cameras, current_fk_surfaces, raw_cuda_by_camera)
```

두 카메라 hull 공유 효과를 얻으려면 위 batch 진입점을 사용한다. 단일 카메라 `__call__`도 지원하며,
그 경우에는 호출마다 자체 hull을 준비한다. 이미지 SAM2 bridge와 tracker callback의 단일 카메라 계약은 유지했다.

## 동일하게 고정한 측정 방식

기존 RTX 3090 / AMD EPYC 7543 환경에서 동일 저장 입력 `10i_s1_0921_t080`, native 512²,
actual SAM2 mask repeat 0, `base_camera` + `hand_camera`의 총시간을 측정했다.
mask·depth·calibration·FK world vertices SHA가 직전 512 실험과 전부 일치한다.

각 후보 warmup 3회, timed 15회, 정순/역순을 교대로 실행했다. `time.perf_counter()`와 타이머 전후
CUDA synchronize로 GPU 완료까지 측정했다. Torch intra-op 4, OMP/OpenBLAS/MKL 각 1, GC 활성화,
80GiB data limit, CPU quota 약 6.47 core를 유지했다. Torch 2.14.1+cu130 / CUDA 13.0 / Triton 3.8.0이다.

CPU 센서 depth·보정 정보·이미 계산된 현재 FK surface가 입력이다. **매번 수행하는 hull 준비,
world ray 계산, clipping, 내부 geometry/depth 업로드와 GPU 완료를 모두 시간에 포함**했다.
새 필터 객체는 후보별로 유지해 warmup에서 calibration ray cache를 채웠으며, 현재 pose hull은 매번 다시 만들었다.
이 캐시 조회·key 확인도 타이머 안이다. JIT/초기 할당과 calibration cache 최초 생성은 warmup에 포함된다.

GPU→GPU 후보는 SAM2가 이미 GPU에서 생성한 마스크를 모사해 저장 마스크를 타이머 전에 올리고,
clean mask를 GPU로 반환한다. CPU 호환 후보는 raw mask 업로드와 clean mask 다운로드도 타이머 안에 포함한다.
파일 읽기·decode/join·별도 FK·SAM2/grounding·gate·Shape-First·Graph·action·검증과 로깅은 제외했다.
다른 모듈의 latency는 측정하지 않았다.

Stage 9 / strict 비교는 frozen `filter_benchmark.pair()`를 수정 없이 재사용했다.
이전 GPU 후보도 직전 측정의 원본 `gpu_filter.py` snapshot을 별도 모듈로 불러 실행했다.
최종 75개 timed 출력은 모두 기준 clean mask와 정확히 일치했다. 검증 readback은 타이머 밖이다.

## 정확도·연결 검증

| 검사 | 결과 |
|---|---|
| 저장 actual mask 112조건 / 224카메라 | PASS |
| 전체 화면 mask stress 128/512/768/1024, 4조건 / 8카메라 | PASS |
| clean mask, confirmed/uncertain/candidate, owner, interior | 평가 영역에서 모두 정확히 일치 |
| predicted robot depth | 검사된 232카메라 최대 절대 차이 **0.0m** |
| CUDA 계약 단위 검사 5개 | PASS: empty/full/overlap/invalid depth, 경계·동률, 이동 pose, intrinsic 제자리 수정, optional GPU depth, 비연속 mask, mock SAM2 연결 |
| GPU mask 경계 단위 검사 1개 | PASS: 필터 batch 호출, raw gate 분모, 예외 복원 |
| 실제 downstream 연결 검사 128/512/768/1024의 4조건 | PASS: gate/records/sources/Graph packet·위치 정확히 일치, feature 차이 0 |

고해상도 diagnostics는 기존처럼 raw mask union에서만 유효하다. 실제 마스크 검사에서는 이 평가 영역을
비교했고, 별도 전체 화면 mask stress에서는 전체 이미지를 비교했다. `NoReadback` dispatch guard가
GPU→CPU 복사, scalar materialization, 동적 `nonzero`를 거부하는 상태에서도 통과했다.
warmup 이후 단위 검사에서는 명시적인 `torch.cuda.synchronize()` 호출도 금지했다.
입력 마스크가 수정되지 않고 반환 결과가 CUDA bool tensor인 것도 확인했다.

Mac 로컬은 CUDA가 없어 관련 6개 검사를 skip했고 CPU tracker 검사 2개는 통과했다.
CUDA 계산과 위 검증은 기존 원격 호스트에서 실제 실행했다. simulator backend나 실제 SAM2 추론은 이번에 실행하지 않았다.

## 기록과 한계

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했다. W&B disabled, publisher 미호출이다.
최종 latency run ID는 `5bbfde74e306489f9f2f4d6cccaa4f9c`이며, 정확도 검사와 개발 실험도 별도 기존 Logger run으로 기록했다.
원래 정지된 Task10-I 7개 process는 모두 T(stopped) 상태를 유지했고 재개하지 않았다.
새 학습·rollout·모델 API·패키지 설치·업로드는 수행하지 않았다.

성능 수치는 **1개 선택된 저장 시점의 warm latency**다. 장면별 latency 분포, cold/JIT 비용,
다른 해상도의 속도 개선, 전체 perception→action 지연을 이 결과로 주장하지 않는다.
full ray cache는 메모리를 추가 사용하며, 512²의 동일 calibration 한 항목은 약 6MiB다.
추가 개선 폭이나 이 구현이 최적화 한계라는 주장은 하지 않는다.

개발용 3회 시험(약 12.0ms)은 캐시/업로드 추가 변경 이전 버전이며 최종 수치에 섞지 않았다.
사전 cProfile도 로봇 필터 호출에만 적용한 진단이며, 그 계측 시간을 최종 latency로 사용하지 않았다.
이번 변경별 기여 시간을 따로 분리한 ablation은 하지 않았다.

로컬 source와 원격 최종 source가 일치하며, 회수한 **82개 파일 SHA256**을 검증했다.
필터 SHA256: `55321a3cd00c8039ac9339514b6e705d8bb1d6107621fed33ca6b6ddabd9b17a`.
원격은 Git 없는 frozen source/격리 overlay로 실행했고 사용한 코드 snapshot과 환경을 결과에 보존했다.
로컬 기준 commit은 `a6568c61e5b8bf823cd0b3adbaaddd1b002cd5ec`이고 작업 내용은 미커밋 상태다.

[75개 최종 원자료](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/final_512/results.json) ·
[측정 설정·입력/소스 hash](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/final_512/config.json) ·
[actual 정확도](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/validation_v1/results.json) ·
[full-mask stress](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/full_mask_stress/results.json) ·
[downstream 일치 검사](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/boundary/results.json) ·
[완료·원래 process 상태](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/completion.json) ·
[로컬 SHA 검증](../../outputs/task10i/robot_filter_gpu_optimization_20261103_v1/local_verification.json) ·
[재현 driver](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/benchmark_optimized.py).

원격 결과: `/workspace/task10i_robot_filter_gpu_optimization_20261103_v1`.
동일 기존 환경 변수와 Python으로 `benchmark_optimized.py`를 실행하고 `TASK10I_GPU_OPT_OUTPUT`을 새 디렉터리로 지정하면 재현할 수 있다.
