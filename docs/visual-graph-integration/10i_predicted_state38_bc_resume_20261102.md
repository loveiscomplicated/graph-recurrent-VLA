# Task 10-I — 평가 재개 인계

2026-11-02. 사용자 요청으로 중단된 평가를 재개한다. 학습11,487 updates와 완료된 Development23개를 재사용한다.
본 문서는 재개 준비·시작 기록이며, 남은 평가 완료나 성능 개선을 의미하지 않는다.

## 보존과 변경 범위

기존 Source Snapshot, 실행 Source Identity, F1/G/H 자산, 학습된 S 가중치, Normalizer,
ActionCodec, Sampling Schedule, Seed Manifest, 선택 규칙을 수정하지 않았다.
재개 전 기존 보호 자산과 Source Hash를 다시 확인했고, Task10-I 보존 파일11,668개의 SHA도 기록·재확인했다.
기존 실패 로그와 FAILED/Status는 재개 작업의 `prior_attempt/`에 보존한다.
기존23개 Episode의 본문·Trace·MP4·기존 Tracking Sidecar를 덮어쓰지 않는다.

운영 변경은 **각 Episode를 새 Python/CUDA/Vulkan/Vision 프로세스에서 실행**하는 것이다.
기존 Renderer 오류의 근본 원인은 확정하지 않았으며, 이 변경은 장시간 프로세스의 자원 누적을 피하기 위한 조치다.
Policy, GRU Hidden, Causal Vision Memory는 원래 계약대로 각 Episode 시작에서 초기화하고 Prefix 전체를 처리한다.
새 프로세스가 Handoff 중간에 Hidden/Tracker를 초기화하는 구조가 아니다.

동결 Source를 수정하면 학습 Checkpoint의 Identity가 달라지므로, 재개 Runner는 별도 경로에 설치했다.

- Instance: **58752018**.
- 실험 Root: `/workspace/task10i_predicted_state_gru_20261102`.
- 운영 Root: `<실험 Root>/operations/resume_eval_20261102_v1`.
- 운영 Script: `<운영 Root>/runner.py`.
- tmux: `task10i_predstate38_bc`.

## 실행할 순서

1. 유효한 Development23개를 SHA 확인 후 재사용한다.
2. 미완료 `step11487 / seed30042 / S0`만 처음부터 원래250-action 조건으로 평가한다.
3. 원래 Development CSV를 완성하고 기존 선택 함수로 Checkpoint를 동결한다.
4. 사전 등록한118200–118211, E/S0/S1/S2/R0/R1/R2/J0/J1/J2의120 Episode를 평가한다.
5. 기존 Paired 검증, 분석, Markdown/HTML 보고서와 Artifact/원본 보호 검증을 실행한다.
6. 실제 감사 완료 후에만 PIPELINE_COMPLETE를 기록한다.

학습을 다시 실행하지 않는다. 실패한 Episode 자동 재시도, Seed 교체, Budget/조건/모델 변경은 없다.
실패 시 비정상 Child Exit Code를 전파하고 FAILED/Traceback을 남겨 중단한다.

## 재개 전 검증

- Checkpoint와 기존 완료 Episode를 검증하고 Source 동결을 확인했다.
- 실제 Child exit7 처리, 중복 실행 Lock 거부, 변조된 완료 Trace 거부를 검증했다.
- 서로 다른 새 프로세스2개에서 학습 완료 S step11487로 Development Parent30007의 실RGB-D2 frame씩 실행했다.
- RGB-D/Graph/물리 상태, Hidden 전후 Hash, Predicted State38, Policy Action과 실행 Action이 정확히 일치했다.
- 두 실행은 각각 공통 ExperimentLogger UUID `108ec6e33a2e48e0b29bf80ffc2bb64d`,
  `6dbb8e7acb764b5aa2c5210b69e6c5a7`을 사용했다. 새 평가도 공통 Logger를 사용한다.
- Bash 문법과 pipefail의 실제 exit7 전파를 검증했다.

[재개 계획](../../outputs/task10i/predicted_state38_bc_20261102/resume_eval_v1/resume_plan.json) ·
[Runner 검사](../../outputs/task10i/predicted_state38_bc_20261102/resume_eval_v1/fixture_verification.json) ·
[새 프로세스 실추론 동등성](../../outputs/task10i/predicted_state38_bc_20261102/resume_eval_v1/fresh_process_equivalence.json) ·
[운영 Source/Launcher Identity](../../outputs/task10i/predicted_state38_bc_20261102/resume_eval_v1/execution_manifest.json).

새 프로세스의 실추론 검사와 CPU PhysX/CUDA Renderer는58752018에서 실행했다.
GPU PhysX와 로컬macOS CUDA는 실행하지 않았다. 드라이버·CUDA Wheel·의존성은 바꾸지 않았다.

## 상태 확인

```bash
ssh -p 25559 root@151.237.25.16
# 접속 후:
tmux attach -t task10i_predstate38_bc
tmux capture-pane -pt task10i_predstate38_bc -S -60
cat /workspace/task10i_predicted_state_gru_20261102/pipeline_status.json
tail -n 100 /workspace/task10i_predicted_state_gru_20261102/pipeline.log
```

새 Job별 상세 로그는 `<운영 Root>/logs/`에 보존한다. 기존 실패 로그는 원래 위치에 그대로 남긴다.
최종 보고서는 `<실험 Root>/docs/visual-graph-integration/10i_predicted_state38_graph_recurrent_bc.md`에 생성된다.
Codex는 Startup Health Check 한 번 이후 반복 감시/재호출을 하지 않는다.
tmux는 SSH와 독립적이나 인스턴스 중단·삭제·recycle 뒤 프로세스 지속을 보장하지 않는다.
Git Commit과 외부 Publishing은 수행하지 않았다.

## 실제 시작 확인

2026-11-02 15:24:38 KST의 단 한 번의 점검에서 **STARTUP_PASS**를 확인했다.
tmux `task10i_predstate38_bc`, Runner PID6556, 남은 Development Job `dev_11487_30042`의 Child PID6565가 실행 중이었다.
Status/Heartbeat/로그와 과거 실패 기록 보존을 확인했다. 평가 완료를 뜻하지 않으며 이후 반복 감시하지 않았다.
[시작 점검 원문](../../outputs/task10i/predicted_state38_bc_20261102/resume_eval_v1/startup_health_check.json).
