# Task07 Vast.ai57640710 — 학습된 Best Checkpoint의 Stop-the-clock 실행

2026-10-26 KST. **학습된 Checkpoint로 GPU smoke를 통과하고, held-out40 본평가를 시작했다.**
이 문서는 최초 실행 확인 시점의 기록이다. 본평가는 독립 Supervisor 작업으로 계속 실행되며,
최종40-episode 성공률/Confidence Interval은 아직 확정되지 않았다.

공식 이름은 `Stop-the-clock / Zero-simulation-time-inference-delay / 20Hz simulation-control Closed-loop Evaluation`이다.
Offline BC (A), 이번 평가 (B), wall-clock deadline을 포함한 real-time 성능 (C)을 구분한다.

## 1. 실행 Identity와 현재 상태

| 항목 | 값 |
| --- | --- |
| Vast instance / GPU | 57640710 / RTX3090 24GB |
| 원본 학습 pipeline UUID | `2be5e62357a44b14a5651a1a346edc7a` |
| 원본 training execution UUID | `002abd72baf84c4da185fd7134c3146f` |
| Best Checkpoint | `epoch039-s8959-495e0806.pt`, epoch39 /global_step8959 |
| Checkpoint SHA256 | `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776` |
| 학습 종료 | early stopping,49 epochs, pipeline COMPLETE /exit0 |
| 실제 trained-checkpoint smoke UUID | `e94093f571f64e889fcf37cf52aed8ca` — SMOKE_COMPLETE |
| 본평가 UUID | `f1f98e87a3bb424db7632f59381b7cb4` — RUNNING |
| Supervisor | `task07-stop-clock-57640710`, PID31084, autorestart=false |
| 고정한 본평가 집합 | held-out40:41000..41039 |
| Frozen test12 | NOT_EXECUTED — 원본 reset hash를 일부 재현하지 못해 전체 집합을 제외 |

최초 확인한 본평가 Seed41000/41001/41002는 각각250-step timeout으로 종료됐다.
이는 **진행 중인 실행의 episode별 원시 결과**다. 최종 성공률이나 학습 모델의 일반 성능으로
확대하지 않는다. 이후 episode 결과는 서버의 동일 UUID 디렉터리에 계속 기록된다.

최신 실행 확인: **2026-10-26 01:18:57 KST,7/40 완료, 성공0개,7개 모두250-step timeout**.
Supervisor는 RUNNING이며 원본 artifact 변경0개다.
[확인 시점의 상태 JSON](../../outputs/graph/integration_baseline/task07_vast57640710/latest_launch_status.json)을 보존했다.

- [로컬 실행 상태 snapshot HTML](../../experiments/f1f98e87a3bb424db7632f59381b7cb4/stop_the_clock_rollout.html)
- [첫 실제250-step 영상](../../experiments/f1f98e87a3bb424db7632f59381b7cb4/episodes/heldout_41000/rollout.mp4)
- [첫 episode 결과](../../experiments/f1f98e87a3bb424db7632f59381b7cb4/episodes/heldout_41000/episode.json)
- [로컬 snapshot decode/link 검증](../../experiments/f1f98e87a3bb424db7632f59381b7cb4/local_snapshot_verification.json)

로컬 HTML은 진행 중인 서버 실행의 명시적인 snapshot이다. 첫 두 episode 결과가 기록된 시점의
JSON과 첫 episode 미디어를 가져왔다. 서버에서 생성될 최종40-episode 보고서와 구분한다.

## 2. 원본 보존과 코드 배치

원본 `/workspace/graph-recurrent-VLA`에는 이전 작업의 미커밋 변경이 있었다.
그 작업 트리에 pull하거나 덮어쓰지 않았다. 별도 `/workspace/task07-stop-clock` checkout을 만들고
GitHub `visual-graph`를 fetch해 HEAD `2d701a370d7280fbf32f68970f3a97397a30d453`을 배치했다.

완료된 원본 run의 Best pointer, immutable receipt, checkpoint, config, robot/action schema,
dataset/split/Vision manifests, source inventory와 simulator conditions를 독립
`/workspace/task07-stop-clock-work/bundle`에 복사했다. 읽기 전후 SHA를 확인했다.
원본 `.tracking.json`에서 training execution UUID도 확인했다.

실제 checkpoint의 `weights_only=True` 제한 로드, 모든 모델 key/shape/FP32 dtype의 strict load,
Robot76/Action7, 정확한 train96 episode의 mean/std normalizer, config/dataset/split/Vision identity를
검증했다. 추론 관련 원본 Python source hash는 모두 최신 평가 checkout과 일치했다.
SAM2 checkpoint hash는 원본120개 Vision episode provenance 모두와 일치했다.
DINO는 원본 cache의 동일 revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`을 사용한다.

[Checkpoint 검증](../../outputs/graph/integration_baseline/task07_vast57640710/checkpoint_validation.json),
[이관 provenance](../../outputs/graph/integration_baseline/task07_vast57640710/transfer_provenance.json),
[원본 보존 검증](../../outputs/graph/integration_baseline/task07_vast57640710/training_protection_verified.json)을 보존했다.
보존 검증 시 원본 파일 변경0개, 원본 git status 변경없음이었다.

기존 tmux/GPU/학습 process, error handler/callback, 원본 source/data/checkpoint는 변경하지 않았다.
Git commit/push, HF/W&B 게시, 외부 메시지 전송도 하지 않았다.

## 3. 실행 환경과 배포 보완

| 항목 | 평가 환경 |
| --- | --- |
| Python / Torch | 3.12.3 /2.8.0+cu128 |
| NumPy / ManiSkill / SAPIEN | 2.1.2 /3.0.1 /3.0.3 |
| Gymnasium | 독립 eval-venv의1.3.0 |
| Simulation / Rendering readback | physx_cpu /sapien_cpu |
| Policy / DINO / SAM2 | CUDA, FP32; speed revision 미활성화 |
| Robot / Controller | panda_wristcam /pd_ee_delta_pose |
| Control / Physics / Horizon | 20Hz /100Hz /250 |
| Camera | base_camera128×128 RGB-D |

원본 수집의 Gymnasium1.3.0과 서버1.4.0 차이는 사용자 추가 승인 후 해결했다.
`/workspace/task07-stop-clock-work/eval-venv`에 Gymnasium1.3.0만 `--no-deps`로 설치했다.
나머지 패키지는 기존 학습 환경의 site-packages를 읽기 전용으로 재사용한다.
기존 환경의 Gymnasium1.4.0과 Torch/CUDA 패키지는 바꾸지 않았다.

최초 CUDA 초기화에서 error304가 발생했다. GPU tensor와 RGB-D 동시 사용을 별도로 검사하고,
CUDA context를 먼저 초기화하는 작은 bootstrap으로 실제 smoke와 본평가를 시작했다.
드라이버·시스템 라이브러리는 설치하거나 바꾸지 않았다. Vulkan은 이미 존재하던 EGL ICD 설정을
해당 평가 process에만 지정했다. 초기 실패 로그도 보존했다.

평가 전용 코드의 보완:

- 최신 Hugging Face Hub가 safetensors cache의 미사용 README/duplicate `.bin`까지 요구하던 문제를
  수정했다. 오프라인 inventory 요청을 실제8개 inference 파일로 제한하며 가중치 hash 검증은 유지한다.
- `--mode smoke --checkpoint-manifest ...`로 실제 학습 weight/normalizer를 검증하는 경로를 추가했다.
  성능 집계에서는 계속 제외한다.
- `evaluation_groups`로 실행 전에 고정한 평가 집합을 명시한다. 원래 전체 seed manifest 검증은 유지한다.
- 실행 metadata에 실제 CUDA 사용, model/vision device, 독립 dependency 설치와 simulator conditions를
  정확히 기록한다.

Graph/GNN/GRU, Shape-First/PCA5/F5/Raw EDE, k4/M16, Robot76/Action7, loss, frozen dataset split,
원본 normalizer와 학습 hyperparameter는 변경하지 않았다. 배포 보완은 commit하지 않았으며
실제 run의 source snapshot/hash와 implementation diff에 보존된다.

## 4. Frozen test와 held-out 집합

Gymnasium 버전을 맞춘 뒤 원본 frozen test12를 `reset(seed)`만으로 검사했다.
4개 reset hash는 같고8개는 달랐다. 어느 test seed에도 정책 action을 적용하거나 hyperparameter를
조정하지 않았다. 검사 결과를 통과시키기 위한 hash/허용치 변경이나 GT 상태 복원도 하지 않았다.

[Reset identity 검사](../../outputs/graph/integration_baseline/task07_vast57640710/reset_preflight.json)와
[수치/렌더 진단](../../outputs/graph/integration_baseline/task07_vast57640710/reset_numeric_diagnostics.json)을 보존했다.
초기 관측 두 샘플의 Robot76 최대 절대 차이는 약2.63e-7/5.35e-7이었다. RGB는2519/2752개 channel
값이, raw depth는18/20 pixel이 달랐다. 한 샘플은 reset hash가 같아도 sensor/kinematics의 byte
동일성이 성립하지 않았다. 원본 macOS와 현재 Linux의 수치·렌더 차이를 구분해서 기록한다.

따라서 일부4개를 선별하지 않고 frozen test12 전체를 본평가에서 제외했다.
실행 전부터 후보였던 새 held-out41000..41039 **40개 모두**를 고정해 평가한다.
전체 원본 manifest와 split은 그대로 유지하고 config에 `evaluation_groups=["heldout"]`을 명시했다.
제외된 frozen test12는 결과의 `excluded_by_evaluation_scope`에 NOT_EXECUTED로 남는다.
Held-out seed/reset group은 원본 dataset/train/validation/test/pilot/smoke와 중복되지 않는지 검사한다.

이 결과는 현재 Linux simulator의 새 held-out seed 집합에 대한 평가다.
원본 frozen reset과 byte가 같은 paired evaluation이라고 보고하지 않는다.

## 5. 실제 GPU smoke와 첫 전체 Episode 검증

Smoke는 원본 Best와 normalizer를 사용해39000/39001 각각3-step을 실행했다.
실제 새 RGB-D → 기존 causal DINO/SAM2 → Graph → GNN/GRU → Controller → 다음 관측이 연결됐다.

| 검사 | 결과 |
| --- | --- |
| 로컬/원격 평가·baseline 계약 | 각각40 tests PASS |
| 실제 trained weight/normalizer strict load | PASS |
| 별도250-step 시간 검사 | PASS, 실제 physics1250회 /12.5s simulation |
| 추론 동안 physics /control elapsed | 0회 /불변 |
| Action당 실제 physics | 5회 |
| 실제 smoke hidden carry/reset, finite Action7/controller scale | PASS |
| 실제 RGB4프레임 online/offline mask 비교 | 픽셀/Presence 차이0 |
| Changed suffix의 이전 prefix | 픽셀 차이0 |
| 실제 SAM2 + t5 late-object prompt fixture | 이전 memory 유지, offline 픽셀/Presence 차이0 |
| 첫 본평가 Seed41000 | 250 steps /1250 physics calls /timeout /성공없음 |

Smoke seed39000/39001의 평균 mask wall latency는 각각422.56ms/218.45ms였다.
이는3-frame의 짧은 기능 검사 값이며 첫 프레임 초기화 비용이 섞여 있다.
50ms deadline을 달성한 수치가 아니다. Policy/vision은 CUDA지만 simulation은 CPU다.

본평가 첫 두 Seed의 episode wall time은120.14s/108.04s, simulation time은 각각12.5s였다.
세 번째 Seed는 wall265.79s에250-step timeout이었다. wall 시간이 길어도 physics는 추론 중 진행하지 않는다.
영상은 simulation-control 순서의20FPS다.

Presence1.0과 missing mask0은 관측/point 집합이 비어 있지 않다는 통계다. 객체 위치를 정확히
추적했다는 정확도 지표는 아니다. 영상과 mask/graph trace를 함께 확인해야 한다.
미계측 collision/task failure는 null로 유지한다.

## 6. 실제 명령과 작업 관리

독립 checkout: `/workspace/task07-stop-clock`.
운영 파일/venv/bundle: `/workspace/task07-stop-clock-work`.
원본 학습 저장소를 수정하는 명령은 사용하지 않는다.

```bash
# 현재 일회성 본평가 상태/로그 확인
supervisorctl status task07-stop-clock-57640710
tail -n 30 /workspace/task07-stop-clock-work/evaluation.log
tail -n 30 /workspace/task07-stop-clock-work/evaluation_stderr.log
```

이미 실행한 명령은 아래와 같다. 환경변수와 초기 CUDA bootstrap은 보존된
[`run_evaluation.sh`](../../outputs/graph/integration_baseline/task07_vast57640710/run_evaluation.sh)에 있다.

```bash
cd /workspace/task07-stop-clock
/workspace/task07-stop-clock-work/eval-venv/bin/python \
  /workspace/task07-stop-clock-work/bootstrap_cuda.py \
  --mode evaluate \
  --config /workspace/task07-stop-clock-work/rollout_cuda_heldout40_config.json \
  --checkpoint-manifest /workspace/task07-stop-clock-work/bundle/checkpoint_manifest.json \
  --seeds-manifest /workspace/task07-stop-clock-work/bundle/seeds_manifest.json
```

실행 직전 gate가 실제 trained smoke 완료, checkpoint SHA, 고정 config/seed manifest SHA,
Python source inventory와 원본 보존 SHA를 검사했다.
Supervisor는 `autostart=false`, `autorestart=false`인 별도 일회성 process다.
기존 서비스나 tmux를 중지/재시작하지 않았으며 웹 포트도 열지 않았다.

서버 최종 출력 경로:

```text
/workspace/task07-stop-clock/experiments/f1f98e87a3bb424db7632f59381b7cb4/
  config.yaml /metadata.json /source_hashes.json /source_snapshot/ /implementation.diff
  checkpoint_provenance.json /seeds_manifest.json /executed_conditions.json
  episodes/heldout_<seed>/
    observations.npz /masks.npz /hidden.npz /actions.npz
    packet_*.npz /graph_*.npz /trace.json /action_attempts.json
    episode.json /representative.png /rollout.mp4
  rollout_results.json
  verification.json                  # 전체 실행 종료 후 생성
  stop_the_clock_rollout.html         # 전체 실행 종료 후 생성
```

에러가 발생하면 해당 episode와 나머지 계획 seed의 미실행 상태를 남기고 중단하며 자동 재시작하지 않는다.
전체40개가 끝나면 동일 실행이 집계와 JSON/NPZ/PNG/MP4 decode·상대 링크 검증 및 정적 HTML을 생성한다.
최종 성공률/Confidence Interval을 현재 snapshot에 미리 채우지 않는다.

## 7. 현재 완료한 것과 남아 있는 것

**완료:** 실제 서버 접근, 학습 종료 확인, 원본 보존, 최신 코드 별도 fetch, immutable Best 이관,
실제 checkpoint/normalizer strict load, 승인된 독립 Gymnasium 환경, 계약40개, 실제 GPU trained smoke,
250-frame 전체 policy closed-loop 실행, held-out40 고정 및 별도 Supervisor 본평가 시작,
첫 실제 영상과 수치 자료의 로컬 snapshot/검증.

**진행 중:** 나머지 held-out episode 실행과 전체40개 최종 성공률/Wilson95% CI·최종 HTML.
**미실행:** 원본과 동일 reset을 요구하는 frozen test12, 실시간 제어 성능(C), HPO/재학습,
외부 artifact 게시. 학습 완료를 기다리거나 원격 학습 상태를 주기적으로 polling하지 않았다.
