# Task10-I: 현재 GPU 손목 gate 단독 시간

2026-11-04. **512×512, 손목 clean 유효점 29,775개인 고정 입력에서 현재 손목 gate
단독 중앙값은 4.0870555ms(약 4.09ms)**였다. World 판정 한 가지만 측정했다.

| 입력 | 유효점 | 측정 횟수 | 중앙값 | 최소 | 최대 |
|---|---:|---:|---:|---:|---:|
| `10i_s1_0531_track_t011` | 29,775 | 30 | **4.0870555ms** | 3.794615ms | 4.630906ms |

## 측정 경계

계측 함수는 현재 live 경로에서 사용하는
`wrist_gate_tensor(cameras, raw_cuda, clean_cuda, gate_cfg, views=world_views)`다.
production gate·live 코드는 이번 측정을 위해 수정하지 않았다.

계측 전에 두 카메라의 입력, raw/clean mask, float64 depth, intrinsic 및 World
calibration tensor를 GPU에 준비했다. `coordinate_frame='world'` view만 준비하며
EE 행렬과 Graph용 view 준비는 실행하지 않는다.

포함한 작업:

- 현재 snapshot identity 확인 및 gate 내부 view/mask 참조 구성
- 유효 wrist pixel 선택과 **World 3D 복원**
- 반경 기반 연결성, 최대 component 및 주변 지지 검사
- 카메라 간 depth·object identity 일치 검사
- diameter·fraction·count 및 전체 Python trace 생성에 필요한 CPU 동기화
- accepted CUDA mask 생성과 호출 직후 CUDA 완료 대기

제외한 작업:

- 입력 업로드, float64 depth 변환 및 World view/calibration 사전 준비
- `prepare_tensor_gate`, raw/clean subset 검사 및 C3 mask 복제·교체
- Graph frame/EE 행렬 준비, Graph 구성, policy
- SAM2, robot filter, simulator 및 FK
- 정확성 확인용 accepted mask 다운로드, hash 계산 및 파일 기록

각 호출 직전 CUDA를 동기화한 뒤 `perf_counter_ns`를 시작하고, 함수 반환 직후
CUDA를 다시 동기화한 시점까지 wall time을 기록했다. 따라서 GPU kernel 시간만을
더한 값이 아니라 이 함수의 CPU 실행·동기화까지 포함한 완료 시간이다.

기존 [live 준비 구간 측정](10i_wrist_gate_live_refinement_20261104.md)의
EE 5.4232ms / World 5.1374ms는 `prepare_tensor_gate` 전체 시간이었다.
**이번 4.0871ms가 요청한 gate 함수 단독 시간**이다. 서로 다른 경계의 수치를
최적화 전후 성능비나 정확한 준비 비용 차이로 해석하지 않는다.

## 조건과 정확성

- 원격: Vast instance 58752018, NVIDIA RTX 3090, AMD EPYC 7543.
- Python 3.12.15, Torch 2.14.1+cu130, NumPy 2.5.4, Triton 3.8.0.
- Torch CPU threads 4, seed 42, inference mode, TF32 off.
- 별도 untimed guard 검사 후 워밍업 5회, 본 측정 30회.
- 매 호출의 accepted mask와 **전체 trace를 타이머 밖에서** 저장된 CPU 기준과
  정확히 비교했다. diameter도 허용오차 비교로 대체하지 않았다.
- accepted points는 모두 **29,269개**, 판정은 accepted, reasons는 빈 목록이었다.
- CPU gate/CPU components 호출 및 gate 내부 view 준비가 발생하면 실패하도록
  guard를 걸고 결과를 검증했다. 두 guard 모두 통과했다.
- 입력 SHA-256:
  `a1d6a86482a4c9a3bc2dcf82bf35544e0340f031df65663d60fdd8c9168748a5`.
- accepted mask SHA-256:
  `93bf7132bbd324a5a6d994e5937bb1f72ddb2de0e0ba0b839a9c7172b3bd4540`.

빈 입력은 측정하지 않았다. 한 고정 입력의 결과이므로 다른 점 수·분포에 동일한
시간을 보장하지 않는다. 로컬 macOS에서는 CUDA를 실행하지 않았고 원격 GPU에서
측정했다. 실제 simulator+SAM2 전체 rollout이나 전체 파이프라인 FPS 측정은 아니다.

## 재현성과 산출물

기존 `ExperimentLogger`를 `WANDB_MODE=disabled`로 재사용했다.
run ID는 **`bedd7daa578342678fcfc7154b17bcba`**다. artifact 게시는 수행하지 않았다.
source SHA는 실행 전에 확인했다. 기준 commit은
`d42f777bd8a2a23eef4f814f5922783383c17948`이며 미커밋 변경이 있어 commit만으로
현재 코드를 재현할 수 없다. 이번 작업은 측정 runner·문서만 추가/갱신했다.

- [runner](../../graph/experiments/wrist_gate_live_refinement_10i_20261104_v1/benchmark_gate_only.py)
- [실행 명령·측정 범위 설명](../../graph/experiments/wrist_gate_live_refinement_10i_20261104_v1/README.md)
- [30회 개별 시간·통계·trace](../../outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/remote/results.json)
- [config·환경·source identity](../../outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/remote/config.json)
- [로컬 source identity](../../outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/source_identity.json)
- [accepted mask](../../outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/remote/accepted_mask.npz)
- [실행 로그](../../outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/benchmark.log)

runner의 Python compile 검사와 결과·입력·source hash 확인을 수행했다.
production 코드 변경이 없어 이전 회귀 suite는 재실행하지 않았다.
