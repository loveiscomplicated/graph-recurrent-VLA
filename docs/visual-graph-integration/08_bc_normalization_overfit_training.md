# Task08 — BC normalization, Small5 overfit, closed-loop evaluation

**기존 Vast57640710에서 Small A/C를 완료했고, 후속 평가 및 Full A/B/C 병렬 학습을 수행하는 독립 tmux Pipeline을 시작했다. 이 문서는 실행 현황 snapshot이며, 최종 학습·조작 성능 보고서가 아니다.**

- 현재 Comparison UUID: `3ed77d8b64884f879b9a628e34a9426a`
- 이전 Comparison UUID: `90229dff2be04681b6abba2a2cb3f71a` — Small A/C 완료 후 GPU 프로세스 종료 판정 오류로 중단
- 첫 복구 UUID: `ed33c842b5454b6ebc654a33dc7d661a` — 학습 전 GPU probe의 host/container PID 검증 오류; 학습 실행 없음
- Small A UUID: `2bc5701a80494a72910af0044219ce43`, container PID41605
- Small C UUID: `9a8cec829aaa4c2aaccc10ed91770c30`, container PID41611
- 실행 위치: `/workspace/task08`; 기존 평가 Python: `/workspace/task07-stop-clock-work/eval-venv/bin/python`
- 현재 Pipeline tmux: `grvla08-pipeline-3ed77d8b`
- 최초 Small tmux: `grvla08-90229dff-overfit_A`, `grvla08-90229dff-overfit_C`

[현재 Plan](../../experiments/3ed77d8b64884f879b9a628e34a9426a/plan.json) ·
[3프로세스 GPU 점검](../../experiments/3ed77d8b64884f879b9a628e34a9426a/triple_resource_gate.json) ·
[실행 상태](../../experiments/3ed77d8b64884f879b9a628e34a9426a/status.json) ·
[복구 감사](../../experiments/3ed77d8b64884f879b9a628e34a9426a/recovery_audit.json)

사용자 변경 지시에 따라 새 인스턴스57676962의 준비·전송 프로세스를 종료하고, 삭제 없이 인스턴스 중지를 요청했다. API의 `success=true`를 확인했다. 새 서버에서는 연구 학습을 시작하지 않았다. 두 서버의 HF 토큰 파일은 값을 출력하지 않고 인증·write 권한을 확인했으며, 이후 기존 서버의 자료를 직접 사용했다. Dataset/checkpoint의 외부 게시나 Git commit은 수행하지 않았다.

## 구현과 통제 조건

기존 Graph/GNN/GRU, Robot76, Vision, sampling, edge 정책, Controller 및 기존 Source/Artifact를 수정하지 않았다. 신규 `bc08_codec`, `bc08_train`, `bc08_prepare`, `bc08_eval`, `bc08_pipeline`, `bc08_entry`, `bc08_report`와 계약 테스트를 추가했다. 기존 `ExperimentLogger`, 로컬 artifact 등록, `Events`/Discord transport, `DirectDataset`, Stop-the-clock rollout loop를 재사용한다.

| 방식 | Optimization loss | Controller 전달 전 decode |
| --- | --- | --- |
| A | 기존 Raw Action7 MSE와 exact | Native continuous Action7 |
| B | `(6 × motion z-MSE + gripper MSE) / 7` | Motion6 역정규화; continuous gripper 유지 |
| C | `(6 × motion z-MSE + gripper BCEWithLogits) / 7` | Motion6 역정규화; logit>0이면 +1, 그 외 -1 |

Motion 통계는 실제 Training Subset만 사용한다. Recurrent의 action 규약인 float32 population std + `1e-6`을 사용하며, observation std floor와 혼동하지 않는다. 원시 std/실제 std/작은 std 성분/floor 미사용을 Manifest에 기록했다. Full Train96의 action target은11,975개이며, std<1e-6인 성분은 없었다. Robot76 Normalizer는 원래 Train96 기준 값을 그대로 재사용했다.

Optimization loss를 A/B/C의 공통 점수로 비교하지 않는다. 모든 공통 offline metric은 decode 후 Native Action7과 설치 Controller 전처리 명령으로 계산한다. Translation clipping, rotation norm clipping과 음의 rotation scale, intrinsic XYZ Euler convention을 유지한다. Controller 명령 오차와 실제 달성 TCP motion은 구분한다.

Small은 AdamW/lr3e-4/B2/T32/clip1/seed42, 정확히2,000 updates와 update 기준 cosine schedule을 사용한다. 200/500/1000/1500/2000에서 전체 train episode를 평가한다. Full은 같은 AdamW 설정으로50 epochs를 실행하고, validation Native Action7 MSE로 선택한 checkpoint에 대해 frozen test를 한 번 평가한다. 모든 모델은 한 initial state dict를 사용하며 optimizer/RNG는 별도다.

`OFFLINE_FIT_PASS`의 운영 기준은 학습 전 Plan에 고정했다: Native Action7 MSE≤0.001, Motion6 MSE≤0.0001, 초기 Action7 오차 대비95% 이상 감소, gripper accuracy≥99%. 이것을 manipulation success로 해석하지 않는다. Loss weight, architecture, controller, 데이터 또는 seed를 성능에 따라 자동 변경하지 않는다.

## Small5와 reset identity

정책 출력을 보기 전에 Train ID 오름차순·서로 다른 reset group·완전한 단계/첫 성공까지의 episode라는 규칙을 고정했다. 전체 Raw/Vision/split hash와 join을 검증했다. 원본 best epoch39/global step8959, SHA256 `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`은 읽기 전용으로 유지한다.

| Episode/seed | Actions | Observations | 현재 runtime reset이 원래 collection과 일치 |
| --- | ---: | ---: | --- |
| 30000 | 134 | 135 | 아니오 — `BLOCKED_RESET_IDENTITY` |
| 30001 | 134 | 135 | 아니오 — `BLOCKED_RESET_IDENTITY` |
| 30002 | 105 | 106 | 아니오 — `BLOCKED_RESET_IDENTITY` |
| 30003 | 137 | 138 | 예 |
| 30004 | 101 | 102 | 예 |

총611 targets의 gripper는 정확히 -1/+1이며, Close391/Open220개다. 다섯 episode 모두 offline fitting에 사용한다. 동일 초기 상태의 train-seed rollout은 재현이 확인된 두 seed만 실행하며, 나머지는 성공/실패로 집계하지 않는다. 다른 초기 상태나 snapshot teleport로 대체하지 않는다. 원본/현재 reset hash는 위 Manifest에 보존했다.

Development는42000..42009, Final은43000..43039로 학습 전에 고정했다. 별도 historical diagnostic41000..41006에서 기존 checkpoint와 새 Full A/B/C를 같은 현재 runtime 조건으로 비교한다. 41005는 새로운 held-out 성능에 포함하지 않는다. Frozen test12의 reset audit와 원본 offline test는 별도로 남긴다.

## 실제 사전 검증과 최초 실행

- 로컬/서버 관련 계약: 복구 전 **59 PASS**, PID namespace 수정 후 관련 계약 **20 PASS**. Loss exactness, train-only 통계, decode/threshold, padding, hidden carry/detach, 전체 모델 gradient, 동일 초기 가중치/sequence, save/reload/resume, test 재소비 거부, 정적 HTML/readback을 포함한다.
- 별도 실제 CUDA probe: 최대 관측 point-count의 real B2×T32 batch에서 forward/backward/optimizer, sequence/step 일치, nonzero hidden 및 model/optimizer/RNG의 exact resume를 확인했다. A/C 각각 단독과 동시 resident probe가 통과했다. Probe 가중치는 연구 학습에 사용하지 않았다.
- Probe별 peak allocated 약0.71GiB, peak reserved 약0.90GiB. 두 프로세스 동시 상주 시 free VRAM21,552MiB. 실제 GPU process 정보, CPU/RAM/thread/storage IO는 `resource_gate.json`과 `resources.jsonl`에 있다. NVIDIA가 표시한 host PID와 container PID를 구분한다.
- Full A/B/C 실제 B2×T32 resident probe 3개 동시 실행을 통과했다. 각 Torch peak reserved 962,592,768 bytes, 동시 실행 중 RTX3090 free VRAM 20,263MiB였다. 컨테이너의 probe PID 42727/42734/42739는 NVML의 host PID와 달라 namespace 차이를 명시적으로 감사했다. Probe 체크포인트는 본 학습에 사용하지 않는다.
- 설치된 simulator 예제에서 CPU PhysX / `pd_ee_delta_pose` reset/step을 실행했다. 별도 실제250-step clock probe에서20Hz/100Hz/Action당5 physics calls를 검사했다. CUDA PhysX, MPS 및 real-time deadline 성공은 검증하지 않았다.
- Small A/C 각각 정확히2,000 optimizer updates를 완료했다. A Native Action7 MSE는0.004992238, C는0.007974608로, 사전에 정한 `OFFLINE_FIT_PASS` 기준에 모두 미달했다. Train-seed 조작 결과는 아직 측정 중이다.
- 기존 서버 Source/Artifact72,835개와 복구 원본 Run을 합쳐 74,158개를 사전 hash로 보호했다. 로컬 기존 Source의 내용 변경은0개였다. 공통 인프라 테스트는42개 중 기존의 외부 HF organization 거부 테스트1개가 실패했다. 이번 작업은 게시 기능을 호출하지 않는다.

## 독립 실행과 결과 위치

Pipeline 순서: 완료한 Small A/C 체크포인트 연결 → train-seed/기능 smoke → Full A/B/C **3개 병렬** → Development/historical/Final closed-loop → 최종 비교 Markdown/HTML. Rollout은 training 종료 후에만 실행한다. Training OOM이면 유효 checkpoint를 유지하고 세 job 종료 후 순차 재개한다. 다른 작업을 종료해 자원을 확보하지 않는다.

서버의 기준 디렉터리:

```text
/workspace/task08/experiments/3ed77d8b64884f879b9a628e34a9426a/
  status.json / pipeline.log / jobs/ / logs/
  plan.json / seeds_manifest.json / resource_gate.json / resources.jsonl
  recovery_audit.json / triple_resource_gate.json / pipeline_error_bundle.json (오류 시)
  comparison.json / verification.json
  bc_training_comparison.html
  08_bc_normalization_overfit_training.md
```

각 실제 train/evaluate는 별도 Logger UUID 디렉터리에 checkpoint/receipt, codec/statistics, normalizer identity, optimizer/scheduler/RNG, source snapshot/hash/diff, native prediction/robot/graph/hidden trace 및 평가 PNG/MP4를 보존한다. 완료 후 최종 문서는 서버 `/workspace/task08/docs/visual-graph-integration/08_bc_normalization_overfit_training.md`에도 생성된다. 현재 로컬 HTML은 초기 실행 snapshot이며 서버의 이후 결과를 자동으로 반영하지 않는다.

Discord webhook은 현재 설정되지 않았다. 알림은 로컬 Events/상태/로그로 남으며, 자동 Codex 재호출은 구성하지 않았다. 오류 시 마지막 stage와 error bundle을 보존하고 후속 실험을 중단한다. 동일 Plan으로의 명시적 재개 예:

```bash
cd /workspace/task08
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/workspace/graph-recurrent-VLA/visual/cache/hf
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export VK_ICD_FILENAMES=/workspace/graph-recurrent-VLA/visual/vulkan/nvidia_icd_egl.json
export MPLCONFIGDIR=/tmp/task08-mpl PYTHONUNBUFFERED=1
/workspace/task07-stop-clock-work/eval-venv/bin/python -m graph.integration_baseline.bc08_pipeline \
  --plan /workspace/task08/experiments/3ed77d8b64884f879b9a628e34a9426a/plan.json \
  --launch --resume
```

진행 중인 Pipeline에 중복 실행하지 않는다. 이미 소비한 offline test의 무결과 재실행, 다른 Codec/Plan의 checkpoint resume는 거부한다. Q1–Q8의 최종 답은 Pipeline이 생성하는 실측 보고서를 사용해야 한다. **계약 PASS, Small offline fitting, 같은 초기 상태의 rollout, 새로운 held-out insertion success는 각각 다른 증거다.**
