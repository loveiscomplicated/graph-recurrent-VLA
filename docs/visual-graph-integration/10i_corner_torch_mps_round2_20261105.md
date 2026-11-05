# Task10-I — Torch/MPS 꼭짓점 검출 추가 최적화

작업일: 2026-11-05 KST. 상태: **기본 경로 반영·정확도/시간 대조 완료**.

Main512 검출 시간이 **평균29.17ms, p95 38.62ms**로 줄었다. 앞서38.93ms였던 동일
snapshot을 이번 실행에서 다시 측정한 값은40.98ms였고, 같은 process의 비교에서
**28.83% 시간 감소, 약1.41배 처리량**이었다. 모드별280회 측정했다.

28상태×2카메라=56장 모두 기존 꼭짓점 수와 3mm 정밀도·검출률이 같았다. 보존한 V4와
대응 좌표의 최대 차이는 **0.00024754mm**다. 기존 테스트와 새 단계별 대조를 합해
**30개 통과**했다. 입력 해상도·검출 조건·RANSAC trial·가시성 판정을 줄이지 않았다.

## 추가로 바꾼 부분

지난 [1차 MPS 최적화](10i_corner_torch_mps_optimization_20261105.md)는 작은 기하 helper를
묶었다. 이번에는 helper 사이에 남아 있던 Torch tensor 생성·indexing·patch 연산까지
후보 검증 단위로 묶었다.

| 변경 | 유지한 의미 |
| --- | --- |
| 경계 후보 검증을 한 kernel로 통합 | ray/plane, 각도,7×7 지지/가림/contact, snap, 재투영, physical edge, 현재 가시성 |
| 세 면 교차점 검증을 통합 | 세 plane의 현재9×9 지지,13×13 core, 앞쪽3×3 가림, 보충 후보의 footprint |
| face envelope 검증을 통합 | physical edge, 현재 depth 지지,5×5 contact, snap·crop·가시성 |
| Box RGB 후보 검증을 통합 | 네 quadrant의4개 표본, 외곽/구멍 판별, physical edge, contact, 앞쪽 해석 우선 |
| Box cube 검증을 통합 | 관측 크기, snap, 가까운 RGB feature로 보정, 현재 관측·가림·edge 지지 |
| plane basis와 관측 점군 재사용 | 같은 프레임에서 같은 값을 재계산하던 부분만 공유 |

Box의 plane basis는 feature마다 만들지 않고 plane마다 한 번 만든다. 준비 단계에서 이미
추린 관측 점군도 현재 프레임의 `PlaneData`에 보관해 cube 좌표 통계에서 재사용한다.
프레임 간 detection cache나 hold는 추가하지 않았다.

Cube physical-edge 검증은512×512 전체의 plane-distance mask를 만들던 부분을,
실제로 검사하는 edge 표본 픽셀에서 같은 거리식을 계산하도록 바꿨다. 거리 threshold와
표본 위치·수는 같다. GPU가 많은 임시 배열을 만들고 다시 읽을 필요가 줄었다.

직선 적합은 이전과 같은 Torch batch 계산을 유지했다. 잔차의 미세한 수치 차이가
근접 후보 순위를 바꾸는 문제를 다시 도입하지 않았다. Plane RANSAC은 seed42/43,
최대 sample1300, trial160, tolerance1mm, 최소35점과 기존 재적합 횟수를 유지한다.

## 변경 위치와 경계

모든 소스/원자료는 기존 ignored `local_corner_probe_10i_20261104/`에 있다.
현재 checkout을 유지했고 `recurrent`로 전환하지 않았다. 주 live·공유 simulator·
Graph/GNN/controller·robot filter·wrist gate·World/EE·영상/viewer에는 변경을 연결하지 않았다.

- [`metal_kernels.py`](../../local_corner_probe_10i_20261104/metal_kernels.py): MPS 후보 단위 통합 kernel과 wrapper.
- [`detector_torch.py`](../../local_corner_probe_10i_20261104/detector_torch.py): MPS 통합 경로 선택, 현재 관측 점군·plane basis 재사용.
- [`test_metal_kernels.py`](../../local_corner_probe_10i_20261104/test_metal_kernels.py): 다섯 단계의 통합 검증을 portable Torch CPU와 대조.
- [`validate_torch.py`](../../local_corner_probe_10i_20261104/validate_torch.py): `--previous-snapshot`으로 비교할 소스 snapshot 선택.

Torch tensor를 사용하고 주요 기하 계산은 MPS에서 실행한다. OpenCV의 HSV/연결 성분,
contour, RGB subpixel feature와 작은 CPU descriptor/merge는 유지한다. Native 출력
`xyz/uv/camera_xyz/object_id/masks`는 GPU에 남는다. 전체 XYZ/3D 점군을 NumPy detector로
내리지 않으며, 숨겨진 CPU detector fallback도 없다. SAM2는 포함하지 않는다.

CPU/CUDA에는 portable Torch 검증 경로를 유지한다. 점군과 basis의 재사용은 이 경로에도
적용되지만 CUDA/NVIDIA 서버는 이번에 실행하지 않았다.

## 정확도 유지 확인

기존 `capture_v4_final`의 같은56장을 각각 한 카메라 입력으로 처리했다. 모든 예측 JSON을
저장하고 hash를 동결한 뒤에만 GT/actor segmentation을 열어 기존 정의로 평가했다.
새 held-out 일반화 추정이나 multi-view fusion, 연속 live 주행 실험은 아니다.

| 카메라 | 물체 | 정답/출력 수 | 보이는 GT | Precision3mm | Recall3mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Main | Peg | 135/136 | 144 | 99.26% | 93.75% |
| Main | Box | **121/129** | **148** | **93.80%** | **81.76%** |
| Wrist | Peg | 51/51 | 61 | 100.00% | 83.61% |
| Wrist | Box | 47/48 | 51 | 97.92% | 92.16% |

위 지표와 두 카메라 단순 집계 지표가 V4와 모두 같았다. 56장 모두 object별 출력 개수가
같고, one-to-one 대응 좌표가0.01mm 이내였다. 실제 최대 차이는0.000247537mm다.
이는 **기존 출력과의 차이**이며 GT에 대한 절대 위치 오차가 이 값이라는 뜻은 아니다.

반복 측정에서는 최초 최적화 출력과 좌표·순서가 absolute tolerance1e-6m 이내로 같음을
매 호출 확인했다. Native GPU 출력 device도 확인했다.

기존25개 + boundary/junction/envelope/Box RGB/Box cube 단계별 대조5개 = **30개 통과**.
새 단계 대조는 같은 합성 buffer를 CPU portable Torch와 MPS에 넣고 선택 flag, 선택한
좌표·영상 위치, supporting-plane 수를 비교한다. 각 검사는 실제 선택된 양성 후보도
포함한다. 추가 수동 확인에서 RGB feature가 빈 cube 입력도 GPU에서 정상 처리했다.

## Main512 처리 시간

각 모드에 전체28개 Main 입력을2회 준비 실행한 뒤10회 반복했다. 모드와 입력 순서를
seed42로 섞은 **모드별280회, 합계1120회**다. 이전 버전은 보존한1차 최적화 소스를
별도 namespace에 불러와 같은 process에서 실행했다. 외부 wall timer 앞뒤로 GPU 완료를
동기화했으며 파일 읽기·renderer·GT·로그·영상 저장은 제외했다.

| 모드 | 평균 | 중앙값 | p95 | 평균 역수 |
| --- | ---: | ---: | ---: | ---: |
| 이전 MPS, GPU 입력→tensor | **40.98ms** | 43.29ms | 56.89ms | 24.40FPS |
| 추가 최적화 MPS, GPU 입력→tensor | **29.17ms** | 29.02ms | **38.62ms** | **34.29FPS** |
| 추가 최적화 MPS, host 입력→CPU dictionary | **30.85ms** | 30.50ms | 40.71ms | 32.42FPS |
| NumPy CPU V4, host 입력→CPU dictionary | 63.15ms | 66.45ms | 89.01ms | 15.84FPS |

같은 GPU 경계에서11.82ms,28.83% 줄었다. 같은 host/CPU dictionary 경계로 NumPy와
비교하면51.15% 줄었다. 이전 보고서의38.93ms를 이번29.17ms에서 단순 차감한 값을
동일 조건의 개선율로 사용하지 않는다. Host 부하·시점에 따른 실행 간 변동이 있으므로
이번의40.98→29.17ms 대조를 기준으로 삼는다.

이번 warm 본 측정의 Native 최대는44.41ms였고, host 호환 API 최대는56.08ms였다.
측정한 Native280회에서는50ms를 넘지 않았지만, 새로운 연속 live 입력이나 최초 Metal
컴파일·새 Torch shape의 지연을 포함한 deadline 보장은 아니다. FPS도 전체 live/policy
주파수가 아닌 검출 계산만의 환산이다.

Host는 Apple M4 Pro, macOS27.0.1 arm64, Torch2.14.1, Python3.11.13,
NumPy2.4.6/Accelerate, OpenCV5.0.0이다. OpenCV14 threads, Torch CPU threads1,
float32 matmul precision `highest`; OPENBLAS/OMP/MKL 환경값1이다.
OpenCV OpenCL flag=True이나 명시적 UMat은 쓰지 않는다. 일반 OS/앱 부하는 남아 있고
exclusive host/thermal-controlled 시험은 아니다.

## 부가 profiler 확인

별도의 reset400 Main 한 장 CPU-activity profiler에서 PyTorch operator 호출은 다음처럼
줄었다. 직접 실행하는 Metal kernel 내부는 이 operator 수에 포함되지 않는다. 따라서
이 표를 GPU hardware dispatch나 물리적 CPU↔GPU 전송 횟수로 해석하지 않는다.

| Operator | 1차 최적화 | 추가 최적화 |
| --- | ---: | ---: |
| `aten::copy_` | 265 | 99 |
| `aten::_to_copy` | 211 | 96 |
| `aten::index` | 81 | 10 |
| `aten::nonzero` | 11 | 10 |
| `aten::_local_scalar_dense` | 17 | 16 |

별도4입력 stage probe도 보존했다. 함수마다 동기화를 추가한 진단이며 다른 시점의
실행이므로, 그 단계 시간을 위280회 latency의 기여율이나 원인별 단독 개선량으로
합산하지 않는다.

## 보존·실험 식별·재현

공통 `utils.experiment_logger.ExperimentLogger`와 기존 run ID 규칙을 재사용했다.
`WANDB_MODE=disabled`로 로컬 기록했다. 학습·새 simulator 실행·원격 게시는 없다.
기존 NumPy V4/V2/V3/Box helper는 원래 source hash 그대로 보존했다.

- 최종 run: **`874e2379fb404623b40deebdff13a6b3`**.
- [최종 summary·source/이전 snapshot hash](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final/summary.json)
- [56장 좌표 회귀 비교](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final/regression.json)
- [동결 inference receipt](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final/inference_complete.json)
- [GT 상세 score](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final/detailed_metrics.json)
- [1120회 latency CSV](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final/latency.csv)
- [실행 로그](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final.log)
- [최종 source snapshot](../../local_corner_probe_10i_20261104/torch_mps_round2_20261105_final/sources/metal_kernels.py)
- [30개 테스트 로그](../../local_corner_probe_10i_20261104/torch_mps_round2_tests_20261105.log)
- [이전 profiler](../../local_corner_probe_10i_20261104/torch_optimization_after_probe_20261105.json)
- [추가 최적화 profiler/단계 진단](../../local_corner_probe_10i_20261104/torch_mps_round2_probe_20261105.json)

새 output 경로로 재현한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector local_corner_probe_10i_20261104.test_torch_detector local_corner_probe_10i_20261104.test_metal_kernels -v
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.validate_torch --capture local_corner_probe_10i_20261104/capture_v4_final --reference local_corner_probe_10i_20261104/v4_final --output local_corner_probe_10i_20261104/new_round2_validation --device mps --rounds 10 --warmups 2 --compare-previous --previous-snapshot torch_mps_optimized_20261105_final
```

比較用には `torch_mps_optimized_20261105_final/sources/` snapshotが必要。
Ignored 原資料リンクは現在のローカルフォルダ保存を前提とする。
