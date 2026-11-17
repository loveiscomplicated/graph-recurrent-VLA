# Task06-C — FP32 Vision 속도 개선 검증

2026-10-25. **개선 코드를 구현했지만 실행 설정 전환은 보류했다.** 기존 CUDA
전처리 조건에서 컴파일 후 마스크 1픽셀 차이가 생겨, 출력 일치 Gate를 통과하지
못했다. 학습 성능 하락을 측정한 결과는 아니다. 요청한 기존 Discord 경로로
검증 결과를 한 번 전송했고 HTTP 전송 성공을 확인했다.

기존 Pipeline `2be5e62357a44b14a5651a1a346edc7a`는 원래 소스/Config로 계속
실행한다. 확인 시점 2026-10-25 12:15 UTC(21:15 KST)에 Vision 111/120 Episode,
Supervisor PID8984, Vision PID8986이었다. 새 설정을 활성화하는
`execution_revision.json`은 생성하지 않았다. 학습 완료까지 기다리지 않았다.

## 구현 범위

- 동일 RGB 바이트·dtype·shape·Prompt의 DINO 원본 탐지 결과 캐시. Prefix별
  객체 선택과 최초/추가 초기화 규칙은 기존 코드를 그대로 호출한다.
- SAM2에 새 Prompt/객체/Conditioning Row가 없고 임시 Prompt Output도 없을 때
  완료한 Preflight를 재사용한다. 변경이나 Reset이 있으면 다시 수행한다.
- 기존 SAM2.1 Small 이미지 인코더의 `torch.compile`을 검증했다. Inductor와
  기존 CUDA 커널을 유지하는 AOT Eager를 확인했지만 모두 출력 일치에 실패했다.
  AOT Eager는 그래프 캡처/AOT 컴파일이며 GPU 커널 생성 가속을 의미하지 않는다.
- 향후 통과한 변경만 활성화하도록 Source/Config/검증 결과 SHA가 연결된
  실행 Revision Gate와, 기존 Discord 창을 유지하는 tmux Resume 경로를 추가했다.
  현재 Run에는 활성화하지 않았다.

Expert, 탐지 Prompt/Threshold, 재탐지 정책, Tracking Horizon, 1024 내부 해상도,
FP32 가중치, Shape-First/PCA5/F5/EE/Raw EDE, Point Budget, GNN/GRU/BC 구조,
학습 Hyperparameter와 Split은 변경하지 않았다. BF16/FP16과 해상도 축소는
실행하지 않았다. 관측 결손을 GT로 보완하지 않았다.

## 측정 조건

RTX3090, Torch2.8.0+cu128. 기존 Train Episode30002/30047의 실제 연속 218프레임,
별도 16프레임 Warmup, B=1/T=1, Episode 내 GRU Hidden Carry를 사용했다.
시간 요약은 Episode 첫 프레임 2개를 제외한 216개다. 모델은 동일한 무작위
초기화 가중치이며 Optimizer Update와 정확도/Closed-loop 평가는 수행하지 않았다.

입력은 메모리에 준비한 원본 RGB-D/Calibration/Robot76이다. DINO/SAM2, Vision
Packet, Graph 구성/전송, GNN/GRU/Head와 CPU Action7 반환까지 GPU 동기화해서
측정했다. 센서 촬영/전송, Simulator/Controller, HDF5 Preload, 모델 Load와
Artifact 쓰기는 제외했다. 기존 SAM2 영상 디렉터리 인터페이스의 준비 비용과
초기 프레임 비용은 별도로 남겼다. 실제 온라인 카메라 어댑터의 측정은 아니다.

## 실행별 결과와 채택 판단

모든 실행은 한 최적화 검증 작업의 기록이다. 새 Dataset이나 학습 Run을
만들거나 HPO를 수행하지 않았다. 각각 기존 ExperimentLogger로 기록했다.

| 검증 실행 | 설정 | 기존 평균 ms | 개선 평균 ms | 기존 p95 ms | 개선 p95 ms | 판단 |
|---|---|---:|---:|---:|---:|---|
| `3eb1763d…` | Inductor, 학습용 결정론/TF32 설정 | 394.35 | 228.25 | 1848.27 | 296.35 | 두 경로는 일치하나 기존 캐시와 불일치 |
| `ea14d771…` | Inductor, 기존 cuDNN TF32 설정 복원 | 422.25 | 223.01 | 2358.44 | 293.56 | 같은 캐시 불일치 잔존 |
| `0cccc110…` | Inductor, 기존 Vision 결정론 설정까지 복원 | 629.18 | 344.00 | 3239.23 | 483.38 | 기존 경로는 캐시와 일치, 개선 경로는 1픽셀 차이 |
| `1a6bd581…` | AOT Eager, 기존 Vision 조건 유지 | 629.18 | 232.59 | 3239.23 | 298.85 | 동일한 1픽셀 차이, 채택 보류 |

마지막 AOT Eager 비교의 Baseline은 바로 앞 실행에서 저장한 동일 조건의
Baseline을 재사용했다. 따라서 마지막 행은 같은 프로세스 안에서 새로 양쪽을
재측정한 결과가 아니다. Run 설정/원본/가중치와 프레임별 출력을 연결해 비교했다.
앞의 두 실행은 Vision에 학습용 수치 설정을 적용했으므로 원래 전처리 설정의
채택 근거로 사용하지 않았다. 설정별 수치를 섞어 단일 FPS로 주장하지 않는다.

최종 후보의 차이는 Episode30047, timestep49의 마스크 1픽셀이다. Episode30002는
모든 마스크가 일치했다. Graph 입력이 달라졌고, GRU 연속 출력까지 비교한
Action7 최대 절대 차이는 **2.6557594537734985e-05**였다. 차이가 작다는 이유로
검증 허용치를 완화하거나 성능 저하가 없다고 판정하지 않았다.

최종 후보는 계산상 평균 약4.3Hz로, 20Hz(50ms) 목표에는 미달한다. 전체 시간
개선은 DINO 중복 추론 제거의 기여가 크며, 컴파일 자체의 단독 이득으로
해석하지 않는다. 기존 캐시 파일과 학습 결과/Checkpoint는 덮어쓰지 않았다.

## 검증 및 운영 기록

- 최종 로컬 통합 계약 테스트: **60 PASS**.
- 원격 통합 계약 테스트: 60 PASS. 최종 Revision/Compiler Gate 보강 후 원격
  `test_vision_speed`: **5 PASS**. 로컬 CUDA/MPS는 실행하지 않았으며 CUDA 측정은
  Vast3090에서 실제 수행했다.
- 캐시 결과의 외부 변경 격리, Prefix/Suffix 인과성, Preflight 무효화,
  Source/Config 변조 거부, 기존 tmux Session Resume를 테스트했다.
- 각 GPU 벤치마크는 독립 tmux의 제한 시간 Guardian으로 원래 작업을 잠시
  정지했다가 정상/실패 모두 재개했다. 학습 프로세스를 Codex 터미널에서
  유지하거나 완료까지 기다리지 않았다.
- 기존 notifier 프로세스가 사용하는 목적지와 기존 `DiscordTransport`를
  메모리에서 재사용했다. 별도 Webhook/Notification Framework를 만들지 않았고
  Secret은 Source/Config/Manifest/Console에 저장하지 않았다.
- 알림은 **2026-10-25 12:15:01 UTC, 1회, DELIVERED**. 실패 판정과 기존 실행
  유지 내용을 포함한다. `NOT_ADOPTED` 상태를 남겼다.
- 원격 원래 실행의 Source Guard PASS, 기존 Supervisor/Vision PID가 실행 중인
  것을 확인했다. 실제 학습 성능, 최종 Test, Closed-loop 성공률은 이 검증의
  결과로 보고하지 않는다. Git Commit/Push, 의존성 설치와 외부 Artifact
  Publishing은 수행하지 않았다.

## 산출물

- [최종 후보 요약](../../outputs/graph/integration_baseline/speed_optimization/remote/experiments/1a6bd5815ca24209a705f61b753d477b/latency_summary.json)
- [프레임별 수치](../../outputs/graph/integration_baseline/speed_optimization/remote/experiments/1a6bd5815ca24209a705f61b753d477b/latency_frames.json)
- [출력 차이 진단](../../outputs/graph/integration_baseline/speed_optimization/remote/experiments/1a6bd5815ca24209a705f61b753d477b/output_difference_diagnostic.json)
- [Discord 전송 확인](../../outputs/graph/integration_baseline/speed_optimization/remote/active_run/discord_delivery.json)
- [전환 보류 상태](../../outputs/graph/integration_baseline/speed_optimization/remote/active_run/cutover_status.json)
- [최종 로컬 Source SHA](../../outputs/graph/integration_baseline/speed_optimization/final_local_sources.json)
- [로컬 계약 테스트](../../outputs/graph/integration_baseline/speed_optimization/local_contracts.log)

구현은 [`vision_speed.py`](../../graph/integration_baseline/vision_speed.py),
[`direct_dataset.py`](../../graph/integration_baseline/direct_dataset.py),
[`run_control.py`](../../graph/integration_baseline/run_control.py),
[`test_vision_speed.py`](../../graph/integration_baseline/tests/test_vision_speed.py) 및
[`README`](../../graph/integration_baseline/README.md)에 있다. 현재 원격 실행은 이
구현으로 교체하지 않았으며 원격 검증 코드는 `/workspace/task06c-speed-code`에
격리돼 있다.
