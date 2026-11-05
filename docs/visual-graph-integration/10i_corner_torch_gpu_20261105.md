# Task10-I — 독립 꼭짓점 검출기의 Torch/GPU 전환

작업일: 2026-11-05 KST. 상태: **기본 경로 전환·MPS 실행·동일 입력 정확도/시간 검증 완료**.

이 문서는 최초 전환 snapshot의 기록이다. 이후 개선은
[정확도 유지 MPS 최적화:75.3→38.9ms](10i_corner_torch_mps_optimization_20261105.md)를 따른다.

독립 검출기의 주요 기하 계산을 Torch로 옮기고 기본 `detector.detect`에 연결했다.
기존 NumPy V4는 `detector_v4.py`에 그대로 보존했다. 현재 Apple M4 Pro에서는
**MPS GPU**로 실행했고, 28상태×2카메라=56장에서 V4와 꼭짓점 수 및 3mm 정확도 지표가
모두 같았다. 대응 좌표의 최대 차이는 **0.000304mm**였다.

이번 전환은 속도 개선을 달성하지 않았다. 같은 실험의 Main512 평균은 NumPy CPU
**60.39ms**, Torch/MPS의 호환 API **75.70ms**였다. CUDA 구현 경로는 제공하지만
이 호스트에서 CUDA/NVIDIA GPU를 실행하지 않았다.

## 변경 범위

소스와 원자료는 기존 ignored 디렉터리
`local_corner_probe_10i_20261104/`에만 두었다. 현재 checkout을 유지했고 `recurrent`로
전환하지 않았다. 주 live의 Graph/GNN/controller, simulator, robot filter, wrist gate,
World/EE 선택, 영상 저장, viewer에는 연결하거나 변경한 부분이 없다.

| 파일 | 역할 |
| --- | --- |
| [`detector.py`](../../local_corner_probe_10i_20261104/detector.py) | 기본 Torch 진입점, 기존 NumPy helper/API 호환 |
| [`detector_torch.py`](../../local_corner_probe_10i_20261104/detector_torch.py) | 프레임 준비·후보 구성·Torch 검증·결과 tensor |
| [`torch_kernels.py`](../../local_corner_probe_10i_20261104/torch_kernels.py) | device 선택, 역투영·평면·batch 직선·가시성·edge 연산 |
| [`detector_v4.py`](../../local_corner_probe_10i_20261104/detector_v4.py) | 변경 없이 보존한 NumPy CPU 기준 구현 |
| [`validate_torch.py`](../../local_corner_probe_10i_20261104/validate_torch.py) | 예측 동결 → GT 평가 → 동기화 시간 비교 |
| [`test_torch_detector.py`](../../local_corner_probe_10i_20261104/test_torch_detector.py) | device·출력 좌표·현재 관측·선/평면 기하 계약 검사 |
| [`evaluate.py`](../../local_corner_probe_10i_20261104/evaluate.py) | `--device` 선택과 실제 backend 기록 지원 |

별도 패키지 설치나 공유 API 확장은 필요하지 않았다. 기존 shared Torch projector도
확인했지만 그 API는 픽셀에 `+.5`를 더하고 float64를 사용한다. 이 독립 검출기의
기존 정수 픽셀 중심·MPS float32 계약과 달라 여기서 그대로 재사용하지 않았다.

## GPU로 옮긴 계산과 CPU에 남은 처리

완전히 GPU에 상주하는 파이프라인은 아니다. 보존된 OpenCV topology와 작은 host
descriptor 처리가 있는 **Torch/GPU 기하 + CPU OpenCV 구성**이다.

| 실행 위치 | 실제 계산 |
| --- | --- |
| Torch, 선택 device | depth 변환·valid·전체 XYZ 역투영 |
| Torch, 선택 device | RANSAC 가설의 잔차/득표, 평면 재적합, plane label/mask와 morphology |
| Torch, 선택 device | 프레임 전체 contour 직선의 batch robust fitting |
| Torch, 선택 device | 다수 후보의 ray/plane 교차, plane snapping, 현재 depth 지지·가림 판정 |
| Torch, 선택 device | 양쪽 physical edge와 Box quadrant 검증, cube 관측 좌표 통계 |
| Torch, 선택 device | 최종 camera→world 변환, 반환 좌표/mask tensor |
| CPU OpenCV | HSV 색 분리, 연결 성분, contour/hierarchy/근사 polygon, RGB subpixel feature |
| CPU 소량 데이터 | seed를 유지하는 RANSAC index schedule, contour descriptor 조립, 작은 plane 교차/face envelope, 최종 greedy merge |

512×512 XYZ나 전체 3D 점군을 NumPy로 내려 기존 검출기에 다시 넣는 방식은 쓰지 않는다.
CPU topology를 위해 RGB·2D plane mask가 왕복하고, 작은 plane/line descriptor와 최종
후보 좌표도 host에서 읽는다. Native API의 최종 결과 tensor는 요청한 device에 남는다.
명시적 GPU 요청이 불가능하거나 연산이 실패하면 오류를 내며 NumPy 검출기로 숨겨서
재시도하지 않는다. `PYTORCH_ENABLE_MPS_FALLBACK`는 미설정인 상태로 검증했다.

## 함께 정리한 중복과 수치 정책

- 현재 프레임의 depth·valid·XYZ를 한 번 만든다.
- HSV/연결 성분 결과를 공유하되 Box의 component cutoff35/8 결과는 보존한다.
  56장 모두 두 mask가 원래 함수 결과와 pixel 단위로 같음을 확인했다.
- 같은 mask이면 V2 기본 검출과 V3/V4 회복이 같은 평면 결과를 공유한다.
- 직선마다 작은 SVD를 호출하던 부분을 프레임 단위 batch와 2×2 covariance의
  주방향 계산으로 바꿨다. 기존 outlier 제거와 재적합 조건은 유지했다.
- 평면은 중심을 뺀3×3 covariance의 최소 고유벡터를 사용한다. 재적합 중에는 고정 크기
  weighted reduction을 사용해 매번 점을 추려 내며 발생하던 동적 크기 동기화를 줄였다.
- MPS를 포함해 기하 tensor는 float32다. NumPy float64와 bit-identical 결과를 주장하지
  않으며 실제 좌표 차이와 기존 3mm 판정을 비교했다.

보이는 현재 depth 지지, 가림, physical edge, contact, crop, 중복 억제는 계속 검사한다.
V2 기본 점을 우선 보존하고 V3 회복, V4 Box 회복을 순서대로 merge한다. GT pose/크기,
숨은 꼭짓점, 이전 프레임, 다른 카메라는 inference 입력이 아니다. SAM2도 추가하지 않았다.

## 사용법

```python
from local_corner_probe_10i_20261104.detector import detect_tensors, detect, detect_cpu

# CUDA가 있으면 CUDA, 아니면 MPS, 둘 다 없으면 Torch CPU.
result = detect_tensors(rgb, raw_depth_mm, intrinsic_cv, extrinsic_cv, device="auto")
xyz = result.xyz                 # [N,3], float32, world metres, 선택 device
uv = result.uv                   # [N,2], float32, 선택 device
object_id = result.object_id     # [N], int64, peg=0 / box=1
mask = result.masks              # [2,512,512], bool, 선택 device

# 명시적 device="cuda:0" 또는 device="mps"도 지원한다.
# 기존 dictionary/list API가 필요할 때:
legacy = result.as_legacy()      # CPU export를 명시적으로 수행
legacy2 = detect(rgb, raw_depth_mm, intrinsic_cv, extrinsic_cv, device="auto")
baseline = detect_cpu(rgb, raw_depth_mm, intrinsic_cv, extrinsic_cv)
```

RGB uint8 native512, signed/raw millimetre depth, CV intrinsic[3,3]와 world-to-camera
extrinsic[3,4]를 받는다. Optional bool mask[2,512,512]와 명시적 빈 mask도 유지한다.
Native `elapsed_ms`는 GPU 완료 동기화까지의 호출 시간이고, 이후 `as_legacy()` export는
포함하지 않는다. 전체 호환 호출의 비용은 아래 외부 wall timer 결과를 사용한다.
인스턴스가 프레임별 working state를 가지므로 동시 호출에는 별도 인스턴스를 사용한다.

## 정확도와 회귀 검증

기존 `capture_v4_final`의 같은28상태를 사용했다. Main과 wrist는 **각각 따로 한 장씩**
검출했다. 모든56장 예측 JSON과 hash를 동결한 뒤에만 GT/segmentation을 열어 기존의
3mm one-to-one score를 계산했다. 새로운 held-out 일반화 추정이나 연속 live 평가는 아니다.

| 카메라 | 물체 | 정답 출력 / 전체 출력 | 관측 GT | Precision3mm | Recall3mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Main | Peg | 135/136 | 144 | 99.26% | 93.75% |
| Main | Box | **121/129** | **148** | **93.80%** | **81.76%** |
| Wrist | Peg | 51/51 | 61 | 100.00% | 83.61% |
| Wrist | Box | 47/48 | 51 | 97.92% | 92.16% |
| 두 카메라 집계 | Peg | 186/187 | 205 | 99.47% | 90.73% |
| 두 카메라 집계 | Box | 168/177 | 199 | 94.92% | 84.42% |

모든 지표가 보존한 V4와 같았다. 두 카메라 집계는 camera-corner 관측 수의 합이며
multi-view fusion이나 중복 제거한 실제 꼭짓점 수가 아니다.

동일 object의 출력들을 one-to-one 대응시켰을 때 56장 모두 개수가 같고 모든 대응이
0.01mm 이내였다. 실제 최대 차이는 **0.000304234mm**다. 출력 순서의 bit-identical
일치는 요구하지 않았다. 반복 시간 측정에서도 Torch 출력의 개수/순서/좌표가 최초 Torch
출력과 absolute tolerance1e-6m 이내로 같음을 매 호출 확인했다.

기존14개 + 새 device/기하7개 = **21개 테스트 통과**. GPU tensor 반환, float64 host
calibration의 MPS 전송 전 cast, world 좌표, invalid depth/empty mask, hidden corner,
near aperture, robust line, metric plane, NumPy detector fallback 차단 등을 검사했다.
Torch CPU 경로도 저장 Main 한 장에서13개 출력으로 smoke 실행했다.

## Main512 처리 시간

최종 run은 모드별28장 전체를2회 warmup한 뒤 10회씩 섞어 처리했다.
**모드별280회, 합계840회**다. 외부 `perf_counter_ns()` 앞뒤로 GPU 완료를 동기화했다.
정확도 확인·파일 읽기·renderer·GT·로그·영상 저장은 timer 밖이다.

| 모드 | 입력/반환 | 평균 | 중앙값 | p95 | 평균 역수 |
| --- | --- | ---: | ---: | ---: | ---: |
| NumPy CPU V4 | host arrays → 기존 CPU dictionary | **60.39ms** | 64.05ms | 85.31ms | 16.56FPS |
| Torch/MPS native | GPU resident 입력 → GPU tensor | **76.00ms** | 78.33ms | 109.00ms | 13.16FPS |
| Torch/MPS 호환 | host arrays → CPU dictionary, 전송/export 포함 | **75.70ms** | 77.82ms | 109.84ms | 13.21FPS |

**MPS에서는 현재 전환본이 CPU 기준보다 약25% 느리다.** Native와 호환 모드의0.30ms 차이는
실행 변동도 포함하므로 전송/export 비용이 음수라고 해석하지 않는다. 두 모드의 입력과
반환 경계도 다르다. 이 FPS는 검출기 계산만의 환산이며 전체 live 제어 주파수가 아니다.
20Hz/50ms 목표를 달성하지 않았다.

MPS 경로에는 작은 GPU kernel 호출, RANSAC의 데이터별 반복/compaction, CPU topology와
작은 descriptor를 위한 동기화가 남아 있다. 별도4입력 stage 진단에서도 평면 준비가 큰
구간이었다. 이 비용들이 코드상 남는 것은 확인했지만, 최종 run의15.3ms 차이를 원인별로
정확히 분해한 수치는 아니다. CUDA 속도로 외삽하지 않는다.

최초 전환 run도 보존했다. 그 run은 NumPy62.41ms, MPS native79.14ms였고 source snapshot이
별도로 있다. 이후 weighted plane refinement를 적용했다. 두 실행의 CPU 시간도 바뀌었으므로
run 사이의 감소를 모두 알고리즘 효과라고 주장하지 않는다. 초기 GPU shape 준비 시간은
warm timing에서 제외했으며, 새로운 형상의 첫 실행 지연 가능성은 남는다.

Host: Apple M4 Pro, macOS27.0.1 arm64, Torch2.14.1, Python3.11.13, NumPy2.4.6/Accelerate,
OpenCV5.0.0. OpenCV14 threads, Torch CPU threads1, float32 matmul precision `highest`.
OPENBLAS/OMP/MKL thread 환경값1, VECLIB_MAXIMUM_THREADS 미지정.
OpenCV OpenCL flag=True, 명시적 UMat은 사용하지 않았다. 일반 OS/앱 background 부하는
있는 단일 benchmark process이며 exclusive host/thermal-controlled 시험은 아니다.

## 실험 식별·원자료·재현

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했다. `WANDB_MODE=disabled`의
로컬 기록이며 원격 게시·학습·새 simulator 실행은 없다. CUDA/NVIDIA 서버는 미실행이다.

- 최종 run: **`dd3a7d61a7f249af8f6f4284203b19de`**.
- 최초 전환 run: `f99d9de8642b43de8a2aa5c99fc69b0d`.
- [최종 summary](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2/summary.json)
- [56장 좌표 회귀 비교](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2/regression.json)
- [동결 inference receipt](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2/inference_complete.json)
- [기존 정의로 계산한 GT score](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2/detailed_metrics.json)
- [840회 latency CSV](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2/latency.csv)
- [최종 소스 snapshot](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2/sources/detector_torch.py)
- [최종 실행 로그](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v2.log)
- [21개 테스트 로그](../../local_corner_probe_10i_20261104/torch_final_tests_20261105.log)
- [최초 전환 summary](../../local_corner_probe_10i_20261104/torch_mps_validation_20261105_v1/summary.json)
- [MPS fallback 환경 확인](../../local_corner_probe_10i_20261104/torch_backend_environment_20261105.json)

입력 hash와 모든 Python source hash는 run config에 기록했고 실행 종료 시 다시 확인했다.
원래 V4 freeze에서 `detector.py`로 지칭한 소스는 이제 `detector_v4.py`와 비교하면 같으며,
V2/V3와 Box helper hash도 변하지 않았다. 이전 CPU 프로파일 driver는 의도적으로 당시
default source freeze를 검사하므로, 새 기본 경로의 측정에는 다음 driver를 사용한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector local_corner_probe_10i_20261104.test_torch_detector -v
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.validate_torch --capture local_corner_probe_10i_20261104/capture_v4_final --reference local_corner_probe_10i_20261104/v4_final --output local_corner_probe_10i_20261104/new_torch_validation --device mps --rounds 10 --warmups 2
```

Ignored 원자료 링크는 현재 로컬 폴더 보존을 전제로 한다.
