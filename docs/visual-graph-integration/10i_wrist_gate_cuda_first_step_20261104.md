# Task10-I: 비어 있지 않은 손목 Peg gate의 첫 CUDA 비교

측정일: 2026-11-04 KST. 실행 ID: `46f0b7e894df4c5cb32c725ba83e6810`.

**512×512, 유효 Peg 29,775점 고정 입력에서 GPU gate 중앙값은 5.790 ms였다.**
같은 실행에서 번갈아 측정한 기존 CPU gate는 2,831.505 ms였다. 채택 mask와
전체 판정 trace는 CPU·GPU 사이 및 15회 반복 모두 정확히 일치했다.

이번 단계는 저장소에 이미 있던 `wrist_gate_tensor`를 같은 입력으로 실행·검증하는
독립 비교다. 새 gate 구현을 추가하거나 production/live 기본 경로를 변경하지 않았다.

## 결과

입력: `10i_s1_0531_track_t011`, 두 카메라 모두 512×512. 원래 raw·clean mask,
signed-mm depth와 calibration을 사용했다. 손목 Peg의 유효 clean 점은 29,775개이며
downsampling·점 수 제한·GT 대체를 하지 않았다. 빈 입력은 성능 표본에 넣지 않았다.

| 경로 | 이번 15회 중앙값 | 최솟값–최댓값 | 결과 |
| --- | ---: | ---: | --- |
| 기존 CPU `core.wrist_gate` | 2,831.504599 ms | 2,768.961509–2,972.882005 ms | 통과, 29,269픽셀 |
| 기존 CUDA `wrist_gate_tensor` | **5.790390 ms** | 5.058029–8.276419 ms | 통과, 29,269픽셀 |

이번 두 중앙값의 비는 **489.00배**다. 이는 이 고정 입력과 아래 측정 경계에 대한
결과이며, 전체 모델의 개선율이나 장치 변경만의 효과가 아니다.

[이전 CPU 기록](10i_wrist_gate_cpu_timing_20261104.md)의 2,322.198434 ms도
동일한 입력 hash에 대한 유효한 기록이다. 이번 CPU 재측정은 별도 실행에서 얻은
2,831.504599 ms이므로, 개선율은 이전 수치 대신 이번 짝지은 측정으로 계산했다.
기존 기록을 덮어쓰거나 두 실행의 CPU 시간 차이를 특정 원인으로 단정하지 않았다.

## 일치 검증

- NPZ 파일 SHA-256은 원래 CPU 측정과 같은
  `a1d6a86482a4c9a3bc2dcf82bf35544e0340f031df65663d60fdd8c9168748a5`다.
  카메라별 raw·clean mask hash도 원래 기록과 비교했다.
- 현재 CPU gate의 untimed 결과가 원래 저장된 accepted-mask hash 및 전체 trace와
  정확히 일치함을 먼저 확인했다.
- GPU의 accepted mask는 CPU와 배열 전체가 정확히 같았다. 양쪽 hash는
  `93bf7132bbd324a5a6d994e5937bb1f72ddb2de0e0ba0b839a9c7172b3bd4540`다.
- 통과 여부, 거부 이유, 점 수, 연결성·국소 지지 비율, 카메라 충돌 수를 포함한
  **전체 trace dictionary가 정확히 일치**했다. 비교 코드에는 extent의 절대 오차
  허용치 1e-12 m를 사전에 두었지만, 이번 실제 extent 차이는 **0.0 m**였다.
- 15회 모두 같은 검사를 타이머 밖에서 수행했다. 각 backend는 자기 reference의
  전체 trace도 매번 정확히 재현했다.
- 첫 GPU 호출 중 CPU `components`와 `CameraObservation.world_points`를 호출하면
  실패하도록 막은 상태에서도 통과했다. CPU 연결성·전체 XYZ 경로로 우회하지 않았다.
- 원격 CUDA 계약 테스트 2개가 통과했다. gate의 정상·다른 객체 충돌·겹침·유효하지
  않은 depth·희소 지지와, 연결 성분의 거리 경계·같은 크기 성분·중복점·5개 무작위
  입력을 CPU reference와 비교했다. 이는 모든 실제 장면의 동등성 증명은 아니다.

## 측정 경계

CPU는 원래처럼 host 입력이 준비된 뒤 `core.wrist_gate()` 호출만 측정했다.
전체 depth에서 XYZ 생성, 반경 연결성, 국소 이웃 지지, 타 카메라 depth/semantic
모순 검사, trace 및 accepted mask 구성이 포함된다.

GPU는 raw·clean mask와 float64 depth를 CUDA에 올린 뒤 아래 호출만 측정했다.

```python
wrist_gate_tensor(cameras, raw_cuda, clean_cuda, gate_cfg, views=views)
```

선택점 XYZ 생성, 연결성·국소 지지·카메라 일치 검사, accepted CUDA mask와 Python
trace 반환을 포함한다. 내부의 host identity/calibration 검사, 작은 calibration
upload 및 scalar 동기화도 포함한다. 따라서 CPU 작업이나 동기화가 전혀 없는
kernel-only 시간은 아니다.

초기 image/mask/depth upload와 검증용 accepted-mask readback은 제외했다.
센서 파일 I/O, FK·로봇 필터, SAM2/detection, Graph, 정책 및 simulator도 제외했다.
따라서 **CPU 입력부터 CPU 출력까지 전송을 모두 포함한 전체 처리 시간으로
5.79 ms를 사용할 수는 없다.**

Untimed reference/최초 compile 후 backend별 warmup 3회를 수행했다. 이후 15쌍을
CPU→GPU, GPU→CPU 순서로 번갈아 측정했다. 시작 전에 CUDA를 동기화하고 GPU 호출
완료 후에도 동기화한 `perf_counter_ns()` wall time이다. 검증, 로그와 저장은
타이머 밖에서 수행했다. 초기 입력 준비와 compile을 매 프레임 반복한 값이 아니다.

## GPU에서 재사용한 계산 방식

`tensor_gate.py`와 `tensor_gate_kernels.py`의 기존 구현을 변경 없이 재사용했다.
CPU 구현은 `cKDTree.query_pairs`로 모든 반경 이웃 쌍을 생성해 sparse 연결 성분을
구한다. CUDA 구현은 반경/2 크기의 셀과 실제 점 사이의 반경 연결 witness를 사용해
연결 성분을 구하며, 같은 셀 내 충분한 점으로 국소 지지가 보장되지 않는 경우에는
실제 점 거리로 지지를 검사한다. XYZ는 필요한 마스크 픽셀만 복원한다.

원래 점·반경·판정 기준은 유지하지만 계산 구조와 장치가 함께 달라진다. 이번
489배를 단순 NumPy→Torch 이동이나 GPU 하드웨어만의 효과로 해석하지 않는다.
GPU 경로의 Python 분기·동적 크기 처리·host scalar 반환에 남은 비용은 다음
최적화 대상으로 검토할 수 있지만, 이번 단계에서는 추가 최적화를 하지 않았다.

## 실행 환경 및 변경 범위

- Vast 58752018, AMD EPYC 7543 / NVIDIA GeForce RTX 3090에서 실제 실행했다.
  Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1, Torch 2.14.1+cu130, Triton 3.8.0.
- Torch threads 4, OMP 4, OpenBLAS/MKL 1, seed 42, TF32 matmul disabled.
  CPU cgroup quota는 `646736 100000`이며 affinity pinning은 하지 않았다.
  공유 호스트의 한 고정 프레임 측정이다. p95·최악 시간·장면 평균을 주장하지 않는다.
- 독립 source snapshot 1,073개 파일의 hash를 검증했다. Snapshot에는 `.git`이
  없으므로 Logger의 null commit/false dirty는 clean checkout 증명이 아니다.
  로컬 commit·dirty 목록과 실제 파일 hash는 별도 identity에 보존했다.
- 공통 `ExperimentLogger`를 재사용했고 원래 CPU 실행을 `source_run_id`로
  기록했다. W&B disabled, 종료 상태 `finished`다. 별도 logger/publisher는 만들지 않았다.
- 추가한 것은 독립 benchmark·계약 테스트·README, 원자료와 이 보고서다.
  기존 GPU 구현과 production 파일, 기존 실험 입력·결과는 수정하지 않았다.
- 기존 설치 환경을 사용했다. 새 설치, 모델/API 호출, 학습, 게시나 기존 작업
  재개는 없었다. 종료 점검에서 GPU는 0%, 1 MiB였다.
- 로컬 Mac은 CUDA가 없어 계약 테스트 2개가 skip됐다. 위 GPU 결과는 원격에서
  직접 실행한 것이다. 이번에 simulator·렌더링·SAM2·전체 live rollout 또는
  production 통합 경로를 실행한 것으로 해석하지 않는다.

## 코드와 원자료

- [실행 방법과 경계](../../graph/experiments/wrist_gate_cuda_10i_20261104_v1/README.md)
- [비교 runner](../../graph/experiments/wrist_gate_cuda_10i_20261104_v1/benchmark.py)
- [계약 테스트](../../graph/experiments/wrist_gate_cuda_10i_20261104_v1/test_contracts.py)
- [재사용한 GPU gate](../../graph/persistent_geometry/tensor_gate.py)
- [재사용한 Triton kernel](../../graph/persistent_geometry/tensor_gate_kernels.py)
- [15쌍 원자료·전체 trace·통계](../../outputs/task10i/wrist_gate_cuda_20261104_v1/remote/results.json)
- [측정 환경·경계·실행 소스 hash](../../outputs/task10i/wrist_gate_cuda_20261104_v1/remote/config.json)
- [CPU·CUDA accepted mask](../../outputs/task10i/wrist_gate_cuda_20261104_v1/remote/accepted_masks.npz)
- [원자료 재계산·소스 검증](../../outputs/task10i/wrist_gate_cuda_20261104_v1/remote/verification.json)
- [원본 실행 로그](../../outputs/task10i/wrist_gate_cuda_20261104_v1/measured.log)
- [계약 테스트 결과 기록](../../outputs/task10i/wrist_gate_cuda_20261104_v1/test_result.json)
- [소스 snapshot identity](../../outputs/task10i/wrist_gate_cuda_20261104_v1/source_identity.json)
- [Logger metadata](../../outputs/task10i/wrist_gate_cuda_20261104_v1/remote/experiments/46f0b7e894df4c5cb32c725ba83e6810/metadata.json)
- [종료 상태 점검](../../outputs/task10i/wrist_gate_cuda_20261104_v1/postcheck.txt)
