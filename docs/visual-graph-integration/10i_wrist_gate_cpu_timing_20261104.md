# Task10-I: 기존 CPU 손목 gate 단독 시간

측정일: 2026-11-04 KST. 실행 ID: `46439da34962499a989535b737b71488`.

**512×512 입력에서 손목 Peg 유효점 29,775개를 처리하는 기존 CPU gate의 단독
시간은 중앙값 2,322.20 ms(약 2.32초)였다.** 손목 Peg가 없는 입력에서는 5.44 ms였다.
두 값은 서로 다른 고정 입력에 대한 결과이며 전체 장면의 평균이나 상한이 아니다.

## 결과

두 카메라의 depth/calibration/raw·clean mask를 공급하고, 손목 Peg gate를 한 번
호출한 시간이다. 두 카메라 각각에 gate를 실행한 합계가 아니다.

| 저장 입력 | 손목 Peg 유효 clean 점 | 기존 gate 중앙값 | 15회 최솟값–최댓값 | 판정 |
| --- | ---: | ---: | ---: | --- |
| `10i_s1_0531_track_t011` | 29,775 | **2,322.198 ms** | 2,286.004–2,371.904 ms | 통과; 29,269픽셀 채택 |
| `10i_s1_0921_t080` | 0 | **5.440 ms** | 5.368–6.682 ms | 빈 관측 거부 |

손목 Peg가 비어 있던 두 번째 장면은 앞선 실제 SAM2 이미지 측정과 같은 장면이다.
이 입력만 측정하면 연결성·이웃 검사의 실제 부하가 드러나지 않으므로, 앞서 사용한
추적 시퀀스의 t11 프레임도 선택했다. 추가 downsampling이나 점 수 제한을 하지 않았다.
CPU gate 자체는 빈 입력에서도 전체 camera world-point 생성 등 고정 작업을 수행한다.

## 측정 경계

기존 `graph.plan_a_geometry.core.wrist_gate`를 직접 호출했다. 새 GPU gate,
실험용 `radius_components` 대체 구현 및 Graph 조립 경로는 사용하지 않았다.
원래의 SciPy `cKDTree.query_pairs` → sparse adjacency → connected components
구현을 그대로 실행했다.

각 입력은 untimed reference 1회 및 warmup 3회를 거친 뒤 **동일 입력을 15회**
측정했다. `time.perf_counter_ns()`로 함수 호출 시작부터 반환까지 wall time을
기록했다. 여기에는 다음이 포함된다.

- 기존 전체 depth의 camera/world XYZ 생성과 support 선택
- 3D 반경 연결 성분, 최대 성분 선택 및 local-neighbor 지지 검사
- 다른 카메라의 depth/semantic mask와의 일치 검사
- 비율·거부 이유·trace 및 최종 accepted mask 구성

센서/마스크 파일 로딩, FK, 로봇 필터, 입력 snapshot, SAM2/detection,
GPU↔CPU 전송, Shape-First, Graph, 정책, 물리 시뮬레이터는 제외했다.
Raw·clean mask와 CPU depth가 준비된 후 gate 함수만 잰 것이다. 함수 내부의
XYZ 생성 비용은 포함하되 별도 단계별 분해 계측은 하지 않았다.

모든 출력 mask와 trace dictionary를 untimed reference와 비교했다. 이 검사는
각 타이머 종료 후 수행했으며 두 입력 모두 15회 전부 정확히 일치했다.
이는 반복 실행의 결정성 검증이며 새 GT 정확도 평가나 GPU 구현과의 동등성
비교 결과는 아니다.

## 입력과 실행 환경

- 기존 `/workspace/task10i_resolution_perception_graph_20261103_v1`의 실제
  SAM2 예측 `r0` 마스크를 디스크에서 읽었다. 이번 측정에서 모델/API를 새로
  호출하거나 object GT를 사용하지 않았다.
- 두 카메라 모두 원래 512×512 RGB-D 및 calibration을 사용했다.
  원래 CPU 로봇 필터로 clean mask를 타이머 전에 계산했다. 저장한 gate 입력
  NPZ에는 raw·clean mask와 원래 카메라 배열을 함께 보존했다.
- Tracking 마스크는 원래 저장된 `sam_worker` 출력이다. 앞선 별도 online adapter
  측정과 같은 시퀀스/프레임이지만 마스크가 완전히 같다고 가정하지 않는다.
  이번 손목 Peg는 29,775픽셀이고, 앞선 online adapter t11은 29,593픽셀이었다.
  Static 입력의 raw·clean mask hash는 앞선 실제 SAM2 측정과 일치했다.
- 원격 CPU: AMD EPYC 7543. Python 3.12.15, NumPy 2.5.4, SciPy 1.18.1,
  Torch 2.14.1+cu130. Torch threads 4, OpenBLAS/MKL threads 1,
  `OMP_NUM_THREADS=4`, seed 42.
- CPU cgroup quota는 `646736 100000`, affinity는 0–127이며 pinning하지 않았다.
  공유 호스트에서의 단일 입력 반복 측정이다. 신뢰구간, 장면 분포, p95 또는
  최악 시간 보장을 주장하지 않는다.
- 원격 Linux CPU에서 gate를 실행했다. 로컬 Mac CPU, GPU gate, 실제 simulator,
  전체 live rollout는 이번 작업에서 실행하지 않았다.

이전 실제 SAM2 측정에 사용한 고정 소스 복사본을 재사용해, 다른 Codex가 수정 중인
GPU gate/live 경로와 분리했다. 고정 소스와 현재 checkout의 전체 core 파일 hash는
다른 Graph 변경 때문에 다르지만 `wrist_gate`/`components`의 AST는 동일하다.
Gate가 사용하는 camera depth/world-point 함수, SurfaceObservation 검증·subset,
`consistent_views`/`filtered_masks`도 AST가 같음을 확인했다. 실행에 사용한 실제
파일과 SHA-256은 원자료에 보존했다.

## 완료 및 제한

두 입력 모두 정상 종료했고 측정/검증 실패는 없었다. SAPIEN import의 기존
Vulkan ICD 경고는 있었지만 simulator를 생성하거나 렌더링하지는 않았다.
공통 `ExperimentLogger`를 사용했고 W&B는 disabled이며 상태는 `finished`다.
원격 소스는 `.git` 없는 고정 복사본이므로 Logger의 `git_commit: null`,
`git_dirty: false`를 clean Git checkout 증명으로 해석하지 않는다.

Production 구현은 수정하지 않고 단독 측정 runner, 입력/결과 및 이 보고서만 추가했다.
새 설치, 게시, 학습, API 호출, 기존 프로세스 재개는 하지 않았다. 기존 중단 PID 7개는
종료 점검에서도 모두 `T` 상태였다. GPU는 종료 시 0%였으며 이번 gate 측정은 CPU만 사용했다.

기존 로봇 필터 7ms대나 앞선 SAM2+필터 147ms대는 손목 gate를 제외한 다른 측정
경계다. 이번 고정 입력 측정과 그 수치를 합쳐 현재 전체 모델 지연시간으로 발표하지 않는다.
다음 GPU 비교가 필요하면 같은 저장 raw·clean mask/카메라 입력에서 별도로 검증해야 한다.

## 코드와 원자료

- [실험 방법/실행 runner](../../graph/experiments/wrist_gate_cpu_timing_10i_20261104_v1/README.md)
- [15회 원자료·판정 결과](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/results.json)
- [환경·경계·실행 소스 hash](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/config.json)
- [원본 센서 hash](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/input_hashes.json)
- [비어 있지 않은 gate 고정 입력](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/10i_s1_0531_track_t011_gate_input.npz)
- [빈 gate 고정 입력](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/10i_s1_0921_t080_gate_input.npz)
- [원본 실행 로그](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/measured.log)
- [Logger metadata](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote/experiments/46439da34962499a989535b737b71488/metadata.json)
- [로컬 시작 identity](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/local_identity.json)
- [결과 재계산·소스 비교 검증](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/verification.json)
- [종료 상태 점검](../../outputs/task10i/wrist_gate_cpu_timing_20261104_v1/remote_postcheck.txt)
