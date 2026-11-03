# Task 10-I: Vast58752018 CPU 전체 역투영 vs CUDA 마스크 역투영

2026-11-03 KST. **요청한 인스턴스58752018에서 실제 CPU/CUDA 비교 완료.**

## 결과

이전 실험과 같은 **512×512 카메라1대의 Depth·실제 Mask**를 사용했다. 이번 비교는 Mac 시간을 사용하지 않고, CPU와 GPU 양쪽을 **같은 Vast.ai58752018**에서 새로 실행했다.

| 방식 | 입력 → 출력 경계 | 중앙값 | p95 | CPU 대비 중앙값 속도비 |
| --- | --- | ---: | ---: | ---: |
| 기존 CPU·NumPy 전체 픽셀 역투영 | CPU Depth/Mask/K → 전체 XYZ → CPU 물체 점·출처 | **4.667ms** | 5.361ms | 기준 |
| GPU·Torch 마스크 픽셀만 역투영 | GPU 상주 Depth/Mask/K → GPU 물체 점·출처 | **0.750ms** | 0.936ms | **6.23배 빠름** |
| 같은 GPU 구현, 입력·출력 전송 포함 | CPU 입력 → GPU 역투영 → CPU 물체 점·출처 | **2.034ms** | 3.369ms | **2.29배 빠름** |

각 방식 워밍업100회 후 **1,000회**의 completion-synchronized wall time을 측정했다. 세 방식은 한 프로세스에서 순차 실행했고, 반복마다 실행 순서를 회전·반전했다. GPU 완료를 기다린 시간을 포함하므로 비동기 호출 제출 시간만 비교한 값이 아니다.

GPU 상주 행은 큰 입력/출력 배열 전송을 제외한다. 다만 함수 안의 calibration/finite 검사와 동적 `nonzero`에 따른 host 판단·동기화 비용은 포함한다. 전송 포함 행은 **pageable CPU 입력의 H2D와 세 출력 배열의 D2H**를 모두 포함한다. 두 경계를 섞어 같은 시간으로 취급하지 않는다.

**61,056점의 XYZ·출처 pixel·object ID·순서가 CPU/GPU에서 모두 정확히 일치**했다. XYZ 최대 절대 차이는 **0m**였다. 사전에 정한 좌표 허용치는1µm였지만 이번 실제 입력에서는 bitwise equality까지 확인됐다.

## 인스턴스·입력·환경

| 항목 | 실제 확인값 |
| --- | --- |
| Vast 인스턴스 | **58752018**; 실행 코드가 `CONTAINER_ID`를 검사 |
| CPU | AMD EPYC7543 32-Core Processor |
| GPU | NVIDIA GeForce RTX3090 24GiB |
| Driver |595.84 |
| Python |3.12.15 |
| Torch / CUDA |2.14.1+cu130 /13.0 |
| NumPy |2.5.4 |
| Python 경로 | `/workspace/venvs/task10i-stage4a-cu130/bin/python` |
| CPU thread 설정 | Torch1, OMP/OpenBLAS/MKL 각각1 |
| Container CPU quota | `646736 / 100000`, 약6.47CPU |
| 고정 입력 | `10i_s1_0844_t080`, `hand_camera`, actual SAM2 mask r0 |
| 해상도·Mask |512×512, Peg61,056pixel(**23.29%**), Box0pixel |

이 Mask의61,056pixel은 모두 유효 깊이였다. 새 점을 샘플링하거나 추가로 줄이지 않았다. 이전 CPU 실험의 정규화한 입력 NPZ와 수정 전 원본 NumPy source를 전송했고, 전송·수신 후 SHA를 대조했다. 원본 Depth의 mm→m 변환은 양쪽 timer 밖에서 동일하게 적용된 입력이다.

GPU 측정 직전에 별도 compute process가 없음을 확인했고, 다른 실행을 중단하지 않았다. 이번 프로세스 종료 후에도 GPU compute process가 없고 메모리1MiB/사용률0%인 상태를 확인했다. CPU를 포함한 공유 호스트 전체의 완전한 독점을 보장한 것은 아니다.

이 입력은 앞으로 만들 Peg 완전 가시 학습·평가 데이터셋으로 인증한 표본이 아니다. 이번 결과는 동일 입력의 연산 경로 비교이며 새 데이터 분포의 평균 성능이나 모델 성공률 평가가 아니다.

## 이번 GPU 구현

독립 실험의 [masked_torch.py](../../graph/experiments/backprojection_cuda_58752018_20261103_v1/masked_torch.py)에 일반 eager Torch 연산으로 구현했다.

1. GPU Depth에서 finite/positive 여부를 검사한다.
2. 두 물체 mask의 overlap을 제외하고, 각 물체의 유효 source pixel을 현재 입력에서 찾는다.
3. 해당 pixel의 Depth만 읽어 camera-frame XYZ를 계산한다.
4. **`[N,3]` XYZ, `[N]` pixel index, `[N]` object ID**를 GPU tensor로 반환한다.

**`[512,512,3]` 전체 XYZ 배열은 만들지 않는다.** 전체2D Depth·Mask는 입력으로 유지하며 mask 판정을 위해 읽는다. 전체 mask를 검사하는 것과 전체 pixel의 XYZ를 역투영하는 것은 구별한다. Pixel 목록도 timer 밖에서 미리 만들거나 캐시하지 않는다.

기존 NumPy 수식의 float64 계산→float32 XYZ 저장 규칙을 유지했다. Torch compile, custom CUDA kernel, Triton, 혼합 정밀도 변경 또는 좌표 근사 최적화는 사용하지 않았다. Calibration 유효성, source 순서와 overlap 처리 의미를 유지했다.

NumPy 비교군은 이전 수정 전에 보존한 **원본 `pca.build_geometry`**를 실제 import했다. 원래 전체 이미지 XYZ 계산·검증·mask 복사와 물체 점 추출을 수행하고, GPU와 같은 의미의 좌표·pixel·object ID 출력으로 구성했다. 이전 Mac 측정값을 새 GPU 시간과 나누어 속도비를 만든 것이 아니다.

이 비교에서는 backend와 출력 저장 방식이 함께 달라진다. 전체 개선율을 GPU 장치 효과 또는 mask 제한 효과 하나의 독립 기여도로 분해한 실험은 아니다.

## 측정에 포함한 것과 남은 경계

주 시간은 **역투영·유효 마스크 점 추출·출처 배열 생성**까지다. 파일 읽기/디코딩, mm 변환, Normal/PCA, 기하 특징, sampling, Graph 조립, 로봇 필터와 정책 추론은 포함하지 않았다. **로봇 필터링 로직은 수정·실행하지 않았다.**

CUDA event span100회 중앙값은 약0.703ms였다. 함수 중간의 host 판단과 launch 간격을 포함할 수 있으므로 순수 CUDA kernel 시간의 합계로 보고하지 않는다. 주 표는 CPU와 비교할 수 있는 완료 대기 wall time이다.

입력 배열 크기는1,572,900bytes, 압축하지 않은 세 출력의 합계는1,709,568bytes다. 전송 포함 측정은 이 입력과 출력의 이동을 실제 수행했다. 단순히 transfer bandwidth로 나누어 추정한 시간이 아니다.

워밍업 이후 별도1회 측정의 Torch peak allocated 증가량은5,655,040bytes였다. CUDA context, 입력 tensor 및 다른 기존 할당 전체가 포함된 GPU 총 사용량은 아니며 CPU 전체 XYZ 메모리와 같은 지표로 비교하지 않는다.

이번 함수의 compact 출력을 기존 dense XYZ 기반 Normal/Feature 코드에 연결하는 작업은 하지 않았다. **전체 Graph의6.23배 개선 또는 전체 시스템 실시간 처리를 검증한 결과가 아니다.** 이번 단계에서 선택점·표본·코드 경로를 넓혀 추가 최적화하지 않았다.

## 검증·보존

- 로컬 Torch CPU 계약2개 PASS, CUDA2개 SKIP.
- **58752018에서 CPU2개와 실제 CUDA2개, 총4개 PASS.** 빈/전체/겹침 Mask, 무효 깊이, 작은 영상, 비연속 tensor, calibration 오류와 source 순서를 포함한다.
- 실제512 입력에서 CPU/GPU point 수·좌표·pixel·object ID와 순서를 비교했고, 저장된 양쪽 출력 NPZ를 로컬에서 다시 읽어 **전부 array-equal**인지 독립 검사했다.
- 각 방식 원시1,000개 sample에서 median/p95를 로컬에서 다시 계산했다.
- 원본 입력·수정 전 NumPy source의 local/remote SHA가 일치했다. 실행 중 사용한 source/dependency SHA의 불변 검사도 통과했다.
- 원격 산출물20개를 다운로드하고 모든 파일의 SHA를 대조했다. 실행 source, input, metadata, config, raw timings와 출력 arrays를 포함한다.

공통 `utils.experiment_logger.ExperimentLogger`를 재사용한 완료 run ID는 **edf5f1f9e50d4635856542880824b8ff**다. W&B는disabled이며 기존 project/group/run ID 규칙과 Publisher를 수정하지 않았다. 게시·패키지 설치·driver 변경·Git commit은 수행하지 않았다.

CUDA 실행은 **원격58752018**에서 수행했다. Mac에서는 CPU 계약과 저장 결과 readback만 실행했다. Simulator/renderer, MPS, GPU PhysX, Terra/SAM2/GNN forward, 학습·rollout은 이번에 실행하지 않았다. 기존 중단 실험과 production Graph 경로는 변경·재개하지 않았다.

## 코드·원자료

- [실험 코드·재현 안내](../../graph/experiments/backprojection_cuda_58752018_20261103_v1/README.md)
- [원시 측정·비교 결과](../../outputs/task10i/backprojection_cuda_58752018_20261103_v1/remote_snapshot/paired_v1/results.json)
- [실제 환경·경계·source/input SHA](../../outputs/task10i/backprojection_cuda_58752018_20261103_v1/remote_snapshot/paired_v1/config.json)
- [CPU/GPU 출력 배열](../../outputs/task10i/backprojection_cuda_58752018_20261103_v1/remote_snapshot/paired_v1/outputs.npz)
- [전송 identity](../../outputs/task10i/backprojection_cuda_58752018_20261103_v1/upload_manifest.json)
- [수신20개 파일 SHA](../../outputs/task10i/backprojection_cuda_58752018_20261103_v1/remote_snapshot/download_receipt.json)
- [프로세스 종료 후 GPU 상태](../../outputs/task10i/backprojection_cuda_58752018_20261103_v1/gpu_after.txt)

원격 독립 root는 `/workspace/task10i_masked_backprojection_cuda_58752018_20261103_v1`이다. 이 보고서는 최종 Markdown 보관 위치인 `docs/visual-graph-integration/`에 저장했다.
