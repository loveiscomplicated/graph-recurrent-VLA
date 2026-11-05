# Task10-I — 정확도를 유지한 Torch/MPS 꼭짓점 검출 최적화

작업일: 2026-11-05 KST. 상태: **기본 경로 반영·동일 입력 정확도/속도 검증 완료**.

후속 개선은 [추가 MPS 최적화:29.2ms](10i_corner_torch_mps_round2_20261105.md)를 따른다.
이 문서의38.9ms는1차 최적화 snapshot의 기록이다.

이전 Torch 전환은 작은 연산과 중간 결과 읽기를 여러 번 반복해 GPU 대기 비용이 컸다.
같은 계산을 Metal 커널로 묶고 후보 선택 결과를 마지막에 모아서 읽도록 바꿨다.
최종 Main512 실측에서 **기존 MPS75.33 → 최적화 MPS38.93ms/frame**, 시간48.31% 감소,
약1.93배 처리량이었다. 같은 실행의 NumPy CPU는59.85ms였다.

56장 모두 출력 꼭짓점 수와 기존 3mm 정밀도·검출률이 같았다. 보존한 V4와 대응 좌표의
최대 차이는 **0.00024754mm**였고, 모든 대응이0.01mm 이내였다. 새 테스트를 포함해
25개가 통과했다. 시야 조건·threshold·RANSAC trial 수를 줄여 얻은 속도가 아니다.

## GPU로 바꾸고도 더 느렸던 이유

기존 경로는 평면 재적합마다 평균·covariance·고유분해·inlier mask를 여러 Torch 연산으로
실행했다. 후보마다 patch·edge·snapping을 위한 작은 tensor 연산도 많았다. 또한 각 검증
단계에서 `nonzero()`로 출력 크기를 확정하고 index와 support count를 CPU에서 바로
읽었기 때문에, 다음 GPU 계산을 보내기 전에 기다리는 구간이 반복됐다.

별도 profiler에서 reset400 Main 한 장의 호출 수는 다음과 같았다. 아래는 **PyTorch
operator 호출 수**이며 GPU hardware dispatch 수나 PCIe 전송 수로 해석하지 않는다.
직접 실행한 Metal 커널 내부 계산은 이 PyTorch operator 표에 포함되지 않는다.

| 호출 | 이전 Torch/MPS | 최적화 후 |
| --- | ---: | ---: |
| `aten::copy_` | 849 | 265 |
| `aten::_to_copy` | 557 | 211 |
| `aten::nonzero` | 24 | 11 |
| `aten::_local_scalar_dense` | 94 | 17 |
| `aten::linalg_eigh` | 24 | 0 |
| `aten::linalg_solve` | 4 | 0 |

작은3×3 문제에 범용 linear algebra와 tensor 변환을 반복하던 비용을, 현재 문제 크기에
맞는 GPU 계산으로 묶었다. 단순히 GPU의 산술 처리 성능이 낮아서 느렸다는 결론이 아니다.

## 반영한 변경

변경은 기존 ignored `local_corner_probe_10i_20261104/` 안의 독립 검출기에 한정한다.
현재 checkout을 유지했고 `recurrent`로 전환하지 않았다. 주 live·Graph/GNN/controller·
simulator·robot filter·wrist gate·World/EE·video/viewer는 변경하지 않았다.

- [`metal_kernels.py`](../../local_corner_probe_10i_20261104/metal_kernels.py):
  `torch.mps.compile_shader`로 컴파일하는 GPU 커널을 추가했다. Torch tensor의 MPS
  저장소를 직접 받아 역투영, 평면 label/morphology, 평면 RANSAC 득표와 재적합,
  현재 depth 지지·가림, plane snapping, physical edge 검증을 수행한다.
- [`torch_kernels.py`](../../local_corner_probe_10i_20261104/torch_kernels.py):
  MPS에서는 위 커널을 선택한다. CPU/CUDA에는 기존 portable Torch 계산을 유지한다.
  명시적 GPU 요청을 NumPy 검출기로 숨겨서 재시도하는 경로는 없다.
- [`detector_torch.py`](../../local_corner_probe_10i_20261104/detector_torch.py):
  `add()`에서 매번 후보를 압축·다운로드하지 않고 GPU batch를 보관한다. 전체 검증을
  큐에 넣은 뒤 `merge()`에서 선택 flag·support count·작은 후보 좌표를 모아 읽는다.
  V2 → V3 → V4 우선순위와 greedy merge 의미는 보존한다.
- [`validate_torch.py`](../../local_corner_probe_10i_20261104/validate_torch.py):
  `--compare-previous`로 이전 Torch 소스 snapshot을 별도 namespace에 불러와, 같은
  process·입력·반환 경계로 이전 MPS와 최적화 MPS를 비교할 수 있게 했다.

평면의 RANSAC seed42/43, 최대 sample1300, trial160, tolerance1mm, 최소 지지35점을
유지했다. 재적합은256-thread 협력 reduction으로 중심을 뺀 covariance를 계산하고,
대칭3×3 Jacobi 고유분해로 법선을 구한다. 기존3회 inlier 재선택과 마지막 적합을 유지한다.
픽셀 해상도512, 관측 가시성, 가림·접촉·physical edge 조건, 현재 프레임 입력을 유지했다.

CPU OpenCV의 HSV·연결 성분·contour·RGB subpixel feature와 작은 descriptor/merge는
계속 사용한다. 전체 XYZ/3D 점군을 CPU로 내려 NumPy 검출기로 처리하지 않는다.
Native 결과의 `xyz/uv/camera_xyz/object_id/masks`는 요청한 GPU에 남는다.
CPU export를 포함하는 `detect()` 호환 API도 그대로 사용할 수 있다.

### 직선 적합의 수치 순서를 보존한 이유

직선 적합도 한 커널로 합친 pilot는 3mm 지표를 유지했지만, float 합산 순서에 따른
잔차 차이로 근접 후보 순위가 바뀌어 두 Peg 출력이 최대0.626mm 이동했다.
커널별 대조 실험에서 직선 적합만 원래 batch로 되돌리면 이 차이가 사라졌다.
따라서 **최종 경로는 원래 Torch batch 직선 적합을 유지**한다. 아래 최종38.93ms는
이 수치 보존을 포함한 결과다. Pilot의 순차 직선 적합 커널은 기본 경로에서 호출하지 않는다.

## 동일 입력 정확도

기존 `capture_v4_final`의28상태, Main/Wrist 각각 한 장씩 총56장을 사용했다.
GT/actor segmentation 없이 모든 예측을 먼저 저장하고 hash를 동결한 뒤, 별도 단계에서
기존 GT 가시성 정의와 3mm one-to-one matching으로 평가했다. 새 held-out 일반화 실험,
두 카메라 fusion, 연속 live 주행 검증은 아니다.

| 카메라 | 물체 | 정답/출력 수 | 보이는 GT | Precision3mm | Recall3mm |
| --- | --- | ---: | ---: | ---: | ---: |
| Main | Peg | 135/136 | 144 | 99.26% | 93.75% |
| Main | Box | **121/129** | **148** | **93.80%** | **81.76%** |
| Wrist | Peg | 51/51 | 61 | 100.00% | 83.61% |
| Wrist | Box | 47/48 | 51 | 97.92% | 92.16% |

위 지표와 두 카메라 단순 집계 지표가 보존한 V4와 모두 같았다. 모든56장에서 개수가
같고, object별 one-to-one 대응 좌표가0.01mm 이내였다. 최대 차이는0.000247537mm다.
반복 시간 측정에서는 최초 최적화 출력과 좌표/순서가 absolute tolerance1e-6m 이내로
같음을 매 호출 확인했다. Native 출력 device도 검사했다.

## 메인 카메라 프레임당 시간

최종 run은 각 모드별28개 Main RGB-D를 모두2회 준비 실행한 뒤, 모드와 프레임 순서를
seed42로 섞어10회씩 측정했다. **모드별280회, 총1120회**다. GPU 완료를 외부 wall timer
앞뒤로 동기화했다. Renderer, 파일 읽기, GT scoring, 영상 저장, 로그는 timer 밖이다.
이전 MPS는 이전 소스 snapshot을 불러와 같은 process에서 다시 측정했다.

| 모드 | 평균 | 중앙값 | p95 | 평균 역수 |
| --- | ---: | ---: | ---: | ---: |
| 이전 Torch/MPS, GPU 입력·tensor 반환 | **75.33ms** | 77.89ms | 110.60ms | 13.28FPS |
| 최적화 Torch/MPS, GPU 입력·tensor 반환 | **38.93ms** | 41.44ms | **53.44ms** | **25.69FPS** |
| 최적화 Torch/MPS, host 입력·CPU dictionary 반환 | **39.96ms** | 41.92ms | 55.04ms | 25.03FPS |
| 보존한 NumPy CPU V4, host 입력·CPU dictionary 반환 | **59.85ms** | 64.28ms | 84.17ms | 16.71FPS |

같은 반환 경계의 GPU 비교에서 평균 시간48.31% 감소, 처리량1.93배다. 같은 host/CPU
dictionary 경계로 NumPy CPU와 비교하면 평균 시간33.24% 감소다. GPU native와 CPU의
반환 경계를 섞어 전송/export의 단독 비용으로 계산하지 않는다.

평균은50ms보다 짧지만 p95가53.44ms이므로 모든 프레임의20Hz deadline을 만족했다는
결과는 아니다. 이 FPS도 검출기 계산만의 환산이다. 새 shape의 첫 Torch 연산과 최초
Metal 컴파일은 warm 통계에서 제외되며, 새 연속 live 입력의 지연 분포는 별도 검증 대상이다.

Host: Apple M4 Pro, macOS27.0.1 arm64, Torch2.14.1, Python3.11.13,
NumPy2.4.6/Accelerate, OpenCV5.0.0. OpenCV14 threads, Torch CPU threads1,
float32 matmul precision `highest`; OPENBLAS/OMP/MKL 환경값1,
VECLIB_MAXIMUM_THREADS 미지정. OpenCV OpenCL flag=True이나 명시적 UMat은 쓰지 않았다.
MPS fallback 환경변수는 미설정이다. 일반 OS 부하는 남아 있으며 exclusive host나
thermal-controlled 시험은 아니다. CUDA/NVIDIA GPU 성능은 이번에 실행하지 않았다.

## 부가 단계 진단

reset400/reset403/Parent0121 t80/Parent0589 t40의4입력을 준비 실행한 뒤3회 반복한
별도 stage probe다. 함수 전후를 동기화하므로 위 최종 latency와 다르며, 최종 전체28입력의
단계별 기여율로 사용하지 않는다. 입력·상태별 변동과 계측 영향도 포함한다.

| 단계 | 이전 ms/frame | 최적화 후 ms/frame |
| --- | ---: | ---: |
| 평면·mask 준비 | 29.30 | **8.81** |
| contour 직선 적합/descriptor | 4.55 | 2.83 |
| boundary 검증 | 10.33 | **3.16** |
| trihedral 검증 | 3.57 | 2.73 |
| face envelope 검증 | 3.86 | 1.37 |
| V4 Box cube | 7.50 | 5.02 |
| V4 Box RGB | 6.48 | 4.30 |
| merge | 0.62 | 1.02 |

merge는 앞 단계에서 미룬 readback을 담당하므로 그 구간 시간은 늘었다.
직선 batch 구현은 유지했으므로 그 행의 감소를 직선 커널 변경 효과로 주장하지 않는다.
작은3×3 적합/검증을 묶고 중간 host readback을 줄인 부분이 핵심 변경이다.

## 테스트·보존·실험 식별

기존14개 + Torch/device7개 + Metal 대조4개 = **25개 통과**.
새 대조는 plane mask/morphology, plane snapping, physical edge/가림, outlier를 포함한
두 평면 RANSAC을 portable Torch CPU 계산과 비교한다. 기존 invalid depth, explicit empty
mask, 숨은 점 억제, crop, world transform과 NumPy detector fallback 차단도 유지했다.
기존 NumPy V4/V2/V3/Box helper는 변경하지 않았다.

공통 `utils.experiment_logger.ExperimentLogger`와 기존 run ID 규칙을 재사용했다.
`WANDB_MODE=disabled`로 로컬 기록했고 학습·새 simulator 실행·원격 게시를 하지 않았다.

- 최종 run: **`27554fc68dae4ab9a79f8b1aa650d7a8`**.
- [최종 summary·source hash·이전 GPU source hash](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final/summary.json)
- [56장 좌표 회귀 비교](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final/regression.json)
- [동결 inference receipt](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final/inference_complete.json)
- [GT 상세 score](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final/detailed_metrics.json)
- [1120회 latency CSV](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final/latency.csv)
- [최종 실행 로그](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final.log)
- [최종 소스 snapshot](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_final/sources/metal_kernels.py)
- [25개 테스트 로그](../../local_corner_probe_10i_20261104/torch_optimized_tests_20261105.log)
- [이전 profiler/단계 진단](../../local_corner_probe_10i_20261104/torch_optimization_baseline_probe_20261105.json)
- [최적화 후 profiler/단계 진단](../../local_corner_probe_10i_20261104/torch_optimization_after_probe_20261105.json)
- [pilot](../../local_corner_probe_10i_20261104/torch_mps_optimized_20261105_pilot/summary.json)
- [직선 적합 수치 차이의 커널별 대조](../../local_corner_probe_10i_20261104/torch_metal_numerical_ablation_20261105.json)

재현 명령은 새 output 경로를 사용한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest local_corner_probe_10i_20261104.test_detector local_corner_probe_10i_20261104.test_torch_detector local_corner_probe_10i_20261104.test_metal_kernels -v
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.validate_torch --capture local_corner_probe_10i_20261104/capture_v4_final --reference local_corner_probe_10i_20261104/v4_final --output local_corner_probe_10i_20261104/new_mps_optimized_validation --device mps --rounds 10 --warmups 2 --compare-previous
```

`--compare-previous`는 보존한 `torch_mps_validation_20261105_v2/sources/` snapshot이 필요하다.
Ignored 자료 링크는 현재 로컬 폴더 보존을 전제로 한다.
