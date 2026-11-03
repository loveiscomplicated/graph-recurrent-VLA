# Task10-I: 실제 SAM2와 로봇 필터 준비 병렬 실행 검증

측정일: 2026-11-03 (KST). 최종 실행 ID: `e652726cea5448b6b9c31a5c3a0d5157`.

실제 SAM2로 CPU 로봇 형상 준비를 겹쳐 실행했다. 실행을 중단시키는 예외나
병렬 처리 오류는 없었고, 기존 직렬 경로와 SAM2 raw mask 및 robot-filter clean
mask가 모두 픽셀 단위로 일치했다. 이미지 경로는 중앙값이 9.0% 줄었지만,
기본 video tracking 경로에서는 속도 개선을 확인하지 못했다.

## 주 결과

두 카메라의 **512×512 입력 → 실제 SAM2 → CUDA clean mask**까지의 wall time이다.
Graph/액션을 포함한 통합 모델 시간이나 로봇 필터 단독 시간이 아니다.

| 경로 | 기존 직렬 (`legacy_serial`) | 병렬 (`parallel`) | 중앙값 변화 | 표본 |
| --- | ---: | ---: | ---: | --- |
| SAM2 image predictor + 로봇 필터 | 92.071 ms | 83.790 ms | −8.281 ms (−9.0%) | 동일 입력 15회 |
| 기본 SAM2 video tracking + 로봇 필터 | 147.573 ms | 148.396 ms | +0.824 ms (+0.6%) | t6–t20의 대응하는 15프레임 |

이미지는 15쌍 중 13쌍에서 병렬 경로가 빨랐고, 같은 쌍의 시간 차이
`parallel − legacy_serial` 중앙값은 −7.390 ms였다. Tracking에서는 15쌍 중
5쌍만 빨랐고, 대응 차이 중앙값은 +0.914 ms였다. 작은 단일 시퀀스 측정이므로
tracking의 0.6% 차이를 확정적인 성능 저하로 일반화하지 않는다. 이번 조건에서
개선이 확인되지 않았다는 결론이다.

| 경로 | 기존 직렬 범위 | 병렬 범위 |
| --- | ---: | ---: |
| Image | 83.679–104.273 ms | 80.269–92.038 ms |
| Tracking t6–t20 | 140.937–152.458 ms | 141.129–153.684 ms |

기존 필터 단독 7.16 ms는 다른 측정 경계다. 그 값을 위 표에 더하거나 빼서
SAM2 단독 시간이나 이번 필터 잔여 시간을 추정하지 않는다.

## 비교 경로와 타이머

동일 모델, 입력, recorded box prompt, 필터 설정 및 CUDA clean-mask 계약을
공유하는 세 경로를 비교했다.

1. `legacy_serial`: SAM2 실행 후 caller thread에서 기존
   `CudaRobotFilter.filter_cameras`를 직접 호출한다. 새 snapshot/worker 비용이
   없는 **주 비교 기준**이다.
2. `serial`: SAM2 실행 후 새 `RobotFilterPreparation.submit`과 `filter`를
   호출한다. 새 snapshot/worker를 유지하면서 실행 순서만 직렬로 둔 대조군이다.
3. `parallel`: 현재 프레임의 준비를 submit한 뒤 SAM2를 실행하고,
   준비 결과를 받아 GPU 필터를 실행한다.

| 경로 | 기존 직접 직렬 | 새 worker 직렬 대조군 | 병렬 |
| --- | ---: | ---: | ---: |
| Image 중앙값 | 92.071 ms | 90.308 ms | 83.790 ms |
| Tracking 중앙값 | 147.573 ms | 153.207 ms | 148.396 ms |

Tracking의 새 worker 직렬 대조군에 비하면 병렬이 4.810 ms 짧지만, 이를 기존
직접 직렬 대비 개선으로 해석하면 안 된다. 대조군의 199.326 ms 표본도 제거하지
않았다. 앞서 채팅에서 언급한 예비 실행의 `150.9 → 147.3 ms` 역시 새 worker
직렬 대조군과 병렬을 비교한 값이었다. 최종 표는 기존 직접 직렬을 추가한
`measured_v2` 실행을 사용하며 예비 실행 수치와 섞지 않는다.

타이머는 메모리에 미리 읽은 CPU RGB-D, 현재 카메라 calibration, 미리 계산한
FK surface 및 고정 prompt가 준비된 시점에 시작한다. CPU snapshot/submit,
실제 SAM2 전처리·H2D·encoder·decoder/메모리 propagation, 로봇 필터와
CUDA clean mask 완료를 포함한다. CUDA synchronize는 전체 타이머 시작/종료에만
두며, SAM2와 필터 사이에 인위적인 synchronize를 넣지 않았다. 출력 검증을 위한
CPU 복사 및 배열 비교는 타이머 밖에서 수행했다.

NPZ 로딩, 별도 FK 계산, 모델 생성/로드, DINO/새 API 호출, Gate, PCA,
Shape-First, Graph, controller/action, simulator step/render는 제외한다.
Tracking은 실제 `OnlineVisionAdapter`의 JPEG 쓰기/읽기 및 전처리를 포함한다.
그 다음 CurrentGeometry Gate/Graph의 CPU 소비 경계까지 잰 값은 아니다.

## 입력과 실행 조건

- Image: `10i_s1_0921_t080`, 두 카메라 각 512×512, 실제 image predictor를
  매번 실행한다. 각 경로 warmup 3회 후 15회 측정하고 실행 순서를 회전/반전했다.
  Base Peg/Box와 wrist Box의 총 3개 유효 prompt를 사용하며, 원래 없었던 wrist
  Peg prompt는 그대로 없다. 두 카메라 모두 image encoding은 실행한다.
- Video: `10i_s1_0531`의 t0–t20 전부, 두 카메라 각 Peg/Box로 총 4개 prompt다.
  각 경로에 독립적인 두 camera state를 두고 같은 eval 모델 가중치를 공유했다.
  실제 `SAM2Tracker`와 `OnlineVisionAdapter`를 실행하되 초기 detection만 기존
  RGB 기반 recorded box로 대체했다. 매 프레임 알려진 길이는 `t+1`로 제한해
  미래 프레임을 노출하지 않았다. 경로 순서는 회전/반전했다.
- Video t0 initialization은 별도이며 t1–t5는 warm progression이다.
  t6–t20의 15개 대응 프레임 중앙값을 보고했다. 동일 상태 15회 반복 실험과는 다르다.
  t0은 기존/worker 직렬/병렬 각각 205.756 / 193.765 / 189.055 ms의 한 번 관측이며,
  초기화 속도 개선을 주장할 반복 표본은 없다.
- 모든 입력은 기존 저장 RGB-D/robot state이며 object GT로 마스크를 만들지 않았다.
  Robot76/FK는 기존 schema/assets로 검증한 뒤 타이머 전에 준비했다.
- 모델: 설치된 SAM2.1 Hiera Small,
  `configs/sam2.1/sam2.1_hiera_s.yaml`, 내부 image size 1024.
  Checkpoint SHA-256:
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.
- 기존 eager 모델 그대로 FP32 weights + BF16 autocast, TF32 허용.
  Torch threads 4, OpenBLAS threads 1, seed 42. 새 compile/모델 최적화는 적용하지 않았다.
- 실행 장치: 원격 NVIDIA RTX 3090, AMD EPYC 7543. Python 3.12.15,
  Torch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA runtime 13.0,
  driver 595.84, NumPy 2.5.4, SciPy 1.18.1.
- CPU cgroup quota는 `646736 100000`, affinity는 0–127이며 특정 CPU에 pin하지
  않았다. 공유 호스트 부하에 대한 통계적 보정이나 신뢰구간 추정은 하지 않았다.
- 모델 로드 시간은 이미지 943.196 ms, video 552.379 ms로 별도 기록하고 위 표에서
  제외했다. CUDA peak allocated는 이미지 573,087,232 bytes, video 전체 최대
  4,363,692,544 bytes였다. Video는 세 경로의 state가 동시에 살아 있으므로
  경로별 단독 메모리 사용량 비교로 해석하지 않는다.

## 실제로 준비 작업이 겹쳤는가

기본 시간 측정과 별도로, 새 worker/filter와 cold ray cache에서 CPU timestamp를
찍는 이미지 진단 3회를 실행했다. 아래 시간은 별도 실행 결과이며 주 표의
중앙값과 더하거나 빼지 않는다.

| 진단 | snapshot/submit | CPU 준비 | SAM2 host 호출 | host 호출과 준비의 겹친 구간 | SAM2 반환 후 future 대기 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.417 ms | 4.812 ms | 79.790 ms | 2.423 ms | 0.016 ms |
| 2 | 3.512 ms | 5.296 ms | 93.914 ms | 2.799 ms | 0.013 ms |
| 3 | 3.539 ms | 5.352 ms | 80.985 ms | 2.912 ms | 0.009 ms |

세 번 모두 CPU 준비가 SAM2 host 호출 반환 전에 끝났다. 따라서 실제 모델을
실행하는 동안 준비 worker가 진행되고, 뒤의 join 대기는 거의 없는 것을 확인했다.
이는 CPU wall 기준 겹침이며 GPU kernel occupancy를 직접 profiling한 결과는 아니다.

준비가 겹친다고 항상 전체 시간이 줄어들지는 않는다. Snapshot/submit 비용,
SAM2 CPU 작업과의 경합, 기존 CUDA 비동기 실행이 이미 허용하던 겹침 등이 전체
이득에 영향을 줄 수 있다. 이번에는 원인별 profiler 분석까지 하지 않았으므로
tracking에서 이득이 사라진 원인을 하나로 확정하지 않는다.

## 정확도와 오류 검증

- Image: warmup을 포함해 동일 reference와 raw/clean 배열을 비교했고,
  측정 15회 모두 세 경로가 bitwise 일치했다.
- Video: 초기화를 포함한 21프레임 모두, 두 카메라의 raw/clean 배열이 세 경로에서
  bitwise 일치했다. 단순 픽셀 수 비교가 아니라 전체 배열을 비교했다.
- 실제 `CurrentGeometry`의 기본 생성자를 사용한 smoke도 수행했다.
  실제 SAM2 producer → 준비 → CUDA 필터의 handoff에서 마스크를 검증했으며,
  backend `cuda`, schedule `before_vision`을 확인했다. 이 smoke는 이미지 입력을
  사용하고 후속 `_build_prepared` Gate/Graph 부분을 가로채 범위 밖 실행을 막았다.
  전체 cache 생성/live rollout를 실행한 검증은 아니다.
- 최종 `complete.json`은 `PASS`, ExperimentLogger 상태는 `finished`다.
  병렬 실행 중단, future/thread 예외, 출력 불일치는 없었다.

설치된 SAM2의 선택적 C++ 확장 `_C`를 import할 수 없다는 기존 경고가 video
후처리에서 발생했다. SAM2가 해당 후처리(작은 구멍 채우기 등)를 건너뛰는 fallback으로
계속 실행했다. 세 경로 모두 동일 환경/fallback이므로 이번 동등성 비교에는 같은
조건이지만, 확장이 정상 설치된 환경까지 검증한 것은 아니다. 패키지 설치나 수정은
하지 않았다. SAPIEN import 시 Vulkan ICD 경고와 Torch JIT deprecation 경고도
기록됐다. 이번 실험은 simulator를 실행하지 않았으므로 이 경고를 simulator 정상
동작 검증으로 해석하지 않는다.

이 검증은 스케줄 변경 전후 마스크의 동등성 및 실행 안정성 검증이다. Object GT
정확도나 downstream task 성공률을 새로 평가한 결과는 아니다. 원격 CUDA에서
실제 실행했으며 로컬 Mac의 CUDA/MPS, 다른 GPU, 전체 물리 시뮬레이터는 실행하지
않았다. 이미지 1장면과 tracking 1시퀀스 결과로 전체 분포, p95, 실시간 보장을
주장하지 않는다.

## 코드, 기록 및 완료 상태

이번 요청에서는 실험 runner/README와 이 보고서를 추가했다. Production 필터,
SAM2, 기본 스케줄 구현은 이 측정을 위해 변경하지 않았다. 이전 요청대로 병렬
기본 경로를 유지했으며 측정 결과를 이유로 자동 rollback하지 않았다.
공통 `utils.experiment_logger.ExperimentLogger`를 재사용하고 W&B를 disabled로
기록했다. HF 게시, 학습, 새 DINO/API 호출, 패키지 설치는 수행하지 않았다.
기존 중단 프로세스 7개는 종료 점검에서도 모두 `T` 상태였고 재개하지 않았다.
종료 시 GPU utilization은 0%, 기존 resident allocation은 636 MiB였다.

로컬 작업 트리에는 다른 작업의 변경이 함께 있으므로 Git commit만으로 실행
소스를 특정하지 않는다. 원격은 `.git` 없는 고정 소스 복사본이라 Logger의
`git_commit: null`, `git_dirty: false`는 clean checkout 증명이 아니다.
초기 작업 트리 identity, 실행 파일 SHA-256, 설치 SAM2 소스 SHA-256 및 overlay를
함께 보존했다. 초기 source snapshot의 기준 commit은
`7981b979ca8a211835ec0e9327354035c8f79c0b`이며 당시 dirty 상태도 기록했다.

- [실험 runner/방법](../../graph/experiments/robot_sam_overlap_live_10i_20261103_v1/README.md)
- [최종 수치 요약](../../outputs/task10i/real_sam_overlap_20261103_v1/summary.json)
- [최종 image 원자료](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/measured_v2/static.json)
- [최종 video 원자료](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/measured_v2/video.json)
- [환경·측정 경계·소스 hash](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/measured_v2/config.json)
- [입력 hash](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/measured_v2/input_hashes.json)
- [설치 환경·SAM2 소스 hash](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/runtime_identity.json)
- [원본 실행 로그](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/measured_v2.log)
- [공통 Logger metadata](../../outputs/task10i/real_sam_overlap_20261103_v1/remote/measured_v2/experiments/e652726cea5448b6b9c31a5c3a0d5157/metadata.json)
- [초기 소스 archive](../../outputs/task10i/real_sam_overlap_20261103_v1/source.tar.gz) 및
  [초기 identity](../../outputs/task10i/real_sam_overlap_20261103_v1/source_identity.json)
- [최종 실행 소스 overlay](../../outputs/task10i/real_sam_overlap_20261103_v1/final_overlay.tar.gz) 및
  [최종 identity/검증](../../outputs/task10i/real_sam_overlap_20261103_v1/final_identity.json)

재현 시 초기 source archive 위에 final overlay를 덮어쓴다. 최종 runner는 기존
직접 직렬 대조군을 추가한 버전이며, 초기 archive의 runner만 실행하면 최종
세 경로 비교가 재현되지 않는다. 관련 runtime 파일들과 최종 runner를 포함한 실행
hash는 최종 config와 대조했고 모두 일치했다. 원자료의 중앙값/대응 차이 및
모든 exact 판정을 로컬에서 재계산해 확인했다. 관련 구현 테스트는 이전
[기본 경로 전환 보고서](10i_robot_overlap_default_20261103.md)에 있으며, 이번 검증의
추가 근거는 위의 실제 SAM2 image/video 실행과 기본 생성자 smoke다.
