# Task10-I — V4 꼭짓점 검출기 Main 카메라 프레임당 시간

측정일: 2026-11-05 KST. 상태: **실측 완료**.

현재 V4 검출기의 메인 카메라512 RGB-D 한 장 처리시간은 CPU Apple M4 Pro에서
**평균57.82ms, 중앙값61.11ms, p95 81.53ms**였다. 준비 실행을 제외한560회 실측이다.
평균의 역수는 검출기 계산만 기준으로 약17.29FPS다.

## 무엇을 측정했는가

대상은 [`detector.detect(...)`](../../local_corner_probe_10i_20261104/detector.py) 호출
전체다. 이 함수는 **메인 영상 한 장에서 Peg와 Box를 함께 검출**한다. Box만을 따로
검출하는 API의 시간이 아니다. 현재 인터페이스와 검출 정확도는 변경하지 않았다.

| 포함 | 제외 |
| --- | --- |
| RGB 색상 분리 | 센서/시뮬레이터 렌더링 |
| V2 Peg/Box 기본 검출 | NPZ 읽기·압축 해제와 module import |
| V3 국소 경계·관측 면 회복 | GT 접근·오차 계산 |
| V4 Box 형상·RGB corner 회복 | 영상 합성·파일 export |
| 결과 dictionary/array 생성과 함수 반환 | SAM2·Terra·GNN·recurrent·action 실행 |

SAM2는 이 독립 검출기에 포함되어 있지 않다. RGB mask를 이미 준 가상 조건이 아니라
현재 함수의 기본 색상 분리도 포함했다. 손목 영상은 읽거나 사용하지 않았다.

## 입력·실행 조건

- 입력은 `capture_v4_final`의 기존28상태 중 `base_camera` native512 RGB-D다.
  reset seed400–407 및 Parent0121/0349/0589/0770의 t0/20/40/60/80이다.
- 모든 센서 배열을 미리 메모리에 올린 뒤 read-only로 고정했다. Inference에서는
  RGB/depth/intrinsic/extrinsic만 읽었고 actor segmentation/GT/Robot은 읽지 않았다.
- 전체28입력을2회, 총56회 준비 실행했다. 쉬운 한 프레임만 warmup하지 않았다.
- 본 측정은28입력×20회=**560 calls**다. 각 pass의 입력 순서는 seed42로 섞었고
  영상별 반복 횟수는 같다. CPU 입력이므로 CUDA synchronize는 사용하지 않았다.
- Detector 호출 바로 앞뒤의 외부 `time.perf_counter_ns()`로 wall-time을 쟀다.
  출력 확인·통계·로그·CSV 저장은 timer 밖이다. 본 측정에는 stage wrapper/profiler가 없다.
- 단일 benchmark process로 실행했다. 기존 정확도 평가처럼 두 평가 process를 병행하지 않았다.
  OS/앱의 일반 background 부하는 남아 있으며 CPU affinity나 exclusive host는 사용하지 않았다.
- GC는 기본 enabled다. Detected output의 좌표·개수·순서가 동결된 V4 output과
  절대오차1e-10 이내로 같음을 모든 warmup/본 측정/부가 probe에서 검사했다.

| 항목 | 실제 조건 |
| --- | --- |
| CPU | Apple M4 Pro, logical CPU14 |
| OS/architecture | macOS27.0.1, arm64 |
| Python / NumPy / OpenCV | 3.11.13 / 2.4.6 / 5.0.0 |
| NumPy BLAS | Accelerate |
| OpenCV threads | 기존 기본값14, optimized enabled, OpenCL 사용 flag=True |
| Native thread 환경 | OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 |
| VECLIB_MAXIMUM_THREADS | 미지정 |

OpenCV의 기본 thread14를 유지했으므로 **단일 thread CPU 시간으로 해석하지 않는다.**
또한 BLAS가 Accelerate이므로 OPENBLAS 환경값만으로 Accelerate 내부 thread 수가1이라고
주장하지 않는다. 측정 전후1분 load average는 약2.40→2.43이었다.
OpenCV에는 NumPy 배열을 전달했고 명시적 `UMat` 입력은 사용하지 않았다.
OpenCL flag가 True인 기존 설정은 유지했으며, 내부 OpenCL kernel dispatch를 별도로
추적한 측정은 아니다. CUDA/MPS 성능으로 해석하지 않는다.

## 본 측정 결과

| 지표 | 프레임당 시간 |
| --- | ---: |
| 평균 | **57.822ms** |
| 중앙값 | **61.106ms** |
| p90 | 79.034ms |
| p95 | **81.535ms** |
| p99 | 87.707ms |
| 최소 / 최대 | 21.624 / 108.446ms |
| 표준편차 | 17.403ms |

평균의 역수 `1000 / 57.822`는 약17.29FPS다. 이 수치는 위28개 정적 상태를 동일 가중한
검출기 계산량에 대한 환산이며 전체 live FPS나 실시간 제어 주파수가 아니다.
현재 검출기 계산만으로도20Hz의50ms budget을 평균상 초과한다.

입력별 점·면·가림 정도 차이가 커 최소/최대 차이가 난다. 영상별20회 평균이 가장
짧은 입력은 Parent0121 t80 약23.13ms, 긴 입력은 reset400 약85.25ms였다.
이 상태별 범위를 누적560프레임 시간과 혼동하지 않는다.

앞280회와 뒤280회 평균은57.11ms와58.53ms로 약2.48% 차이다. Pass별 평균은
약55.44–61.89ms였다. 별도 host reboot/thermal-controlled 재현성 시험은 수행하지 않았다.
첫 호출은98.77ms였지만 import는 이미 끝난 상태였고 본 통계에서 제외했다.

## 부가 stage probe

본 측정이 끝난 뒤 process-local wrapper로 별도의28입력×5회=140 calls를 쟀다.
Detector 파일을 바꾸지 않았고 wrapper는 종료 전에 복원했다. 아래 값은 **별도
계측 pass의 평균**이며 headline57.82ms에 더하거나 중앙값을 합산하지 않는다.

| 구간 | 평균 |
| --- | ---: |
| V2 기본 Peg/Box 검출 | 12.72ms |
| 나머지 V3 회복·준비·포장 | 31.48ms |
| V4에서 추가한 Box 회복 부분 | 15.34ms |
| 부가 probe 전체 | 59.54ms |

**15.34ms는 Box 전체 시간이 아니다.** V2와 V3 구간에도 Box 처리가 포함된다.
본 API는 Peg/Box를 함께 처리하므로 이번 작업에서 Box-only 시간을 임의로 만들지 않았다.
Probe와 본 측정의2.97% 평균 차이는 실행 시점·계측 overhead 등이 함께 존재하며
그 원인별 기여를 분리하지 않았다. 더 깊은 병목 profiler나 최적화는 이번 범위가 아니다.

## 보존·실험 인프라·검증 범위

UUID: `5ebfb3068a3044aa99cc5d5ea0af2df1`.
공통 `ExperimentLogger`를 재사용하고 `WANDB_MODE=disabled`로 로컬 기록했다.
Benchmark code는 기존 ignored 폴더에 추가했으며 shared/live 검출 코드는 수정하지 않았다.
시작/종료 시 V4 detector/helper/원본 dependency hash와 입력 manifest의 동결 일치를
검증했다. 기존 V4 출력 일치도 확인했다. GT 오차 평가·학습·새 simulator 실행은 없다.
CUDA/MPS, NVIDIA 서버의 성능은 이번에 실행하지 않았다. HF/W&B 게시도 수행하지 않았다.

## 재현과 원자료

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m local_corner_probe_10i_20261104.benchmark_latency --capture local_corner_probe_10i_20261104/capture_v4_final --reference local_corner_probe_10i_20261104/v4_final --output local_corner_probe_10i_20261104/new_main_latency --rounds 20 --warmups 2 --profile-rounds 5
```

- [독립 benchmark 코드](../../local_corner_probe_10i_20261104/benchmark_latency.py)
- [측정 protocol](../../local_corner_probe_10i_20261104/main_latency_20261105_v1/protocol.json)
- [전체 통계](../../local_corner_probe_10i_20261104/main_latency_20261105_v1/summary.json)
- [560회 원자료](../../local_corner_probe_10i_20261104/main_latency_20261105_v1/samples.csv)
- [입력별 통계](../../local_corner_probe_10i_20261104/main_latency_20261105_v1/per_frame.csv)
- [140회 stage probe](../../local_corner_probe_10i_20261104/main_latency_20261105_v1/stage_probe.csv)
- [실행 로그](../../local_corner_probe_10i_20261104/main_latency_20261105_v1.log)

Ignored 자료 링크는 현재 로컬 폴더 보존을 전제로 한다.
