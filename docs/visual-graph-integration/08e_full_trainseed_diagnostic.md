# Task 08-E — Full A/B/C train-seed diagnostic

ExperimentLogger UUID `1cbed275766549fda6945456503019aa`. Vast.ai 57640710 RTX 3090 inference; CPU PhysX 20 Hz control / 100 Hz physics. No training.

## Full 작업 중지와 보존

원본 pipeline `3ed77d8b64884f879b9a628e34a9426a`은 user-request interrupt 후 `ERROR` 상태다. Full A/B/C 50 Epoch와 Development10/Method는 완료됐다. Historical7 A/B는 완료, C는 1/7 완료 뒤 interrupt. Final40/Method는 0/40 완료, 미실행은 실패로 계산하지 않는다. 중지 사유 `INTERRUPTED_BY_USER_REQUEST`.
원본 Checkpoint, Metrics, Plan, Split의 23개 고정 파일은 중지 전후 SHA256 일치. 원본 status/log/job 파일 3개는 중지 기록으로 변경됐다. 세부 내역은 pre/post stop audit JSON.

## Train seed 30003: Offline 및 실제 Closed-loop

| Method | Single MSE | Full Best MSE | 배율 | Full Epoch50 MSE | Best Epoch/Update | Controller translation mm | Rotation rad | Grasp | Lift4cm | Insert | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| A | 1.131014e-07 | 0.002275702 | 20121× | 0.001534355 | 35/8036 | 2.527 | 0.00363 | False | False | False | 250 |
| B | 2.774612e-07 | 0.003438041 | 12391× | 0.003433573 | 48/11026 | 1.525 | 0.00309 | False | False | False | 250 |
| C | 9.131975e-08 | 0.002448319 | 26810× | 0.002061261 | 38/8725 | 1.599 | 0.00405 | False | False | False | 250 |

Single A/B/C는 각각 20,000 update와 약 4,000 Episode 노출로 학습했다. Full Best는 A35/B48/C38 Epoch 및 A8036/B11026/C8725 update다. Full은 96개 서로 다른 학습 Episode를 동시에 Fit해야 한다. 20,000 대 11,487 update 차이만으로 원인을 단정할 수 없다.

Epoch50은 세 방법 모두 Train Seed Action7 MSE가 Best보다 낮다. 이는 특정 학습 Episode에서 추가 노출이 효과를 낼 수 있음을 지지하지만 Validation 및 실제 성공 개선을 보여주지 않는다. Epoch50 Closed-loop는 실행하지 않았다.

## 최초 오류와 상태 이탈

Threshold: 정상 Expert observation의 controller translation 오차 >1 mm, Native Action7 L2 >0.1, Gripper sign 오류; 실제 rollout의 같은 시점 TCP–Peg 상대 벡터와 Expert 차이 >1 cm. 마지막 지표는 서로 다른 자율 history의 기술적 비교이며 인과 위치를 확정하지 않는다.

- A: offline translation >1 mm t=0; Action7 L2 >0.1 t=0; gripper sign t=None; same-time relative state >1 cm t=5.
- B: offline translation >1 mm t=0; Action7 L2 >0.1 t=0; gripper sign t=49; same-time relative state >1 cm t=10.
- C: offline translation >1 mm t=0; Action7 L2 >0.1 t=0; gripper sign t=None; same-time relative state >1 cm t=8.

## 연구 판단

Full Best의 정상 Expert sequence 오차는 Single보다 수천~수만 배 크므로 Training Fit 한계가 확인된다. 이는 학습 예산, 96개 상태의 다양성, capacity 또는 optimizer 문제 중 어느 하나를 단독으로 확정하지 않는다.
실제 rollout은 새로운 RGB-D를 매 step 처리하고 이전 Student 행동이 만든 상태에서 다음 action을 예측했다. 상대 geometry의 Expert 기준은 기존 저장 Action 재생 GT sidecar이며 원본 Robot76과 byte-identical하지 않다. Offline 오차와 실제 상태 이탈은 함께 관측되므로 Distribution Shift만의 독립 기여를 분리할 수 없다.
다음 우선순위는 같은 학습 seed의 stage별 action/controller 오차와 rollout 이탈 시점을 연결하고, 별도 사전등록 실험에서 Epoch 연장 시 Train/Validation/Closed-loop를 함께 평가하는 것이다. 필요하면 실패 state에 대한 복구 자료와 representation 진단을 후속 실험으로 분리한다.

## Q1–Q10

- **Q1 CONFIRMED:** 대상 Full pipeline/worker 종료, 선택 checkpoint와 학습 지표 hash 일치. Historical7 C 부분 결과 보존; Final 0/40.
- **Q2 CONFIRMED:** 137-frame Expert observation offline MSE와 controller 오차를 위 표와 JSON에 기록.
- **Q3 CONFIRMED:** Single 대비 Full MSE 배율과 controller 오차를 comparison.json에 기록.
- **Q4 CONFIRMED:** A/B/C 각 1회 실제 CPU PhysX closed-loop의 grasp/lift/insert 결과를 기록.
- **Q5 SUPPORTED:** 분석용 수치 threshold의 최초 frame과 상대 geometry 이탈을 기록. 인과적 첫 오류 시점은 단일 rollout으로 확정 불가.
- **Q6 CONFIRMED:** Epoch50의 동일 seed offline MSE를 Best와 읽기 전용 비교.
- **Q7 SUPPORTED:** Single보다 큰 학습 seed 오차는 현재 Full 설정의 부족한 fit을 지지. 원인 분리는 미완료.
- **Q8 SUPPORTED:** Epoch50 train-seed 개선과 원래 validation 선택 시점은 구분 가능. Epoch50 closed-loop 효과는 NOT_TESTED.
- **Q9 INCONCLUSIVE:** Development 0/10에는 train fit과 새로운 상태 일반화가 모두 기여할 수 있음.
- **Q10 SUPPORTED:** 즉시 학습 연장 단정 대신 검증 악화와 폐루프 이탈을 분리할 후속 실험 근거 확보.

## 시각화 및 Artifact

- [single_vs_full_action7](../../experiments/1cbed275766549fda6945456503019aa/figures/single_vs_full_action7.png)
- [controller_errors](../../experiments/1cbed275766549fda6945456503019aa/figures/controller_errors.png)
- [action_timeline](../../experiments/1cbed275766549fda6945456503019aa/figures/action_timeline.png)
- [stage_heatmap](../../experiments/1cbed275766549fda6945456503019aa/figures/stage_heatmap.png)
- [tcp_3d](../../experiments/1cbed275766549fda6945456503019aa/figures/tcp_3d.png)
- [grasp_release_lift](../../experiments/1cbed275766549fda6945456503019aa/figures/grasp_release_lift.png)
- [relative_geometry](../../experiments/1cbed275766549fda6945456503019aa/figures/relative_geometry.png)
- [epoch_trend](../../experiments/1cbed275766549fda6945456503019aa/figures/epoch_trend.png)
- [Expert vs Full A/B/C after-action MP4](../../experiments/1cbed275766549fda6945456503019aa/full_abc_vs_expert_after_action.mp4)
- [Full A after-action MP4](../../experiments/1cbed275766549fda6945456503019aa/rollout/A/after_action_rgb.mp4)
- [Full B after-action MP4](../../experiments/1cbed275766549fda6945456503019aa/rollout/B/after_action_rgb.mp4)
- [Full C after-action MP4](../../experiments/1cbed275766549fda6945456503019aa/rollout/C/after_action_rgb.mp4)

원본 Checkpoint와 학습 산출물은 수정하지 않았다. 원본 운영 상태·로그·작업 기록 3개는 중지에 따라 갱신됐다. CUDA PhysX, real-time deadline, Epoch50 closed-loop 및 새로운 학습은 실행하지 않았다. 외부 업로드·dependency 설치·Git commit 없음.

## 추가 실행 감사와 지표

- 원본 Full Pipeline UUID: `3ed77d8b64884f879b9a628e34a9426a`. 신규 평가 UUID: `1cbed275766549fda6945456503019aa`. User interrupt 시 확인한 Coordinator PID 42812, Historical7_C worker PID 57061 모두 종료됐고 해당 tmux Session이 없다. Worker shell exit code 1은 사용자 중단으로 기록했다.
- Frozen split: Train96/Validation12/Test12; Seed30003은 Train에만 포함. Expert HDF5 SHA256 `ba578d3e17e804c48c5490dc9970406a2b560d9ea214519094965fa815a09122`. Dataset Split SHA256 `2471629fbc55073acc3b3fbce3cc7b1d36f209c3d318e7c1ac5c4e6f690edae7`.
- 원본 고정 파일 23개, Plan source 290개, Plan input 152개 hash 일치. 운영 상태·로그·job 상태 3개만 사용자 중단 기록으로 변경됐다. 원본 Final 평가 A/B/C 각각 0/40 완료, 40/40 미실행이다.
- Robot76 Train96 Normalizer SHA256 `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`. A/B/C 선택 Checkpoint와 Epoch50 Checkpoint는 각각 원본 receipt의 SHA256을 로드할 때 검증했다.

| Method | Full Run UUID | Best epoch/update | Best SHA256 | Epoch50 SHA256 | Codec SHA256 | Validation Best MSE | Validation Epoch50 MSE |
|---|---|---:|---|---|---|---:|---:|
| A | `7e5d89177f9942a79d5343bd415fafce` | 35/8036 | `409a9456740193c3a46337c4e172dac7d64dc22c45250c365d3879d4fb337b5e` | `46632da5b1071317cf543b2eee4edb4d8d8cfad74d67400ae27fcb0d186ba3d7` | `2fdc60c1b60c4ab62b1ce6e4ac653caf4edbd5badc38da0d7ae9323245c1200b` | 0.00475772 | 0.00494965 |
| B | `7952d1c812f142c784159bda3825a245` | 48/11026 | `f094f3cc85f8013c37e70d6043a7035071f5733b7db04371207c31fbce7b4e02` | `224c199e0f985b558ae44537c295f2bb137dad43da0190366dc3e5617f086dfa` | `9a475b4a51779a913db563b84495ff6f2189f09013c3e2193863702b352b782c` | 0.00613824 | 0.00614156 |
| C | `84b64bac5ca141aea1152c3b6fac2ca9` | 38/8725 | `e615ec3ded96b0242004e08ee13273b782ba4af1032c8a84f205144f4ab8d177` | `7e8c9b4a14d96f30c78136e29f0c89d21d41181ede8ee596614663f435c08b0f` | `b1e7a9dc175b09c47f0def8d4062a8c4c94ea10c981fd9089e8d2f97db7b369b` | 0.00450394 | 0.00533039 |

Validation 최소값이 실제 Best receipt의 Epoch와 일치했다. Epoch50에서 Train Seed30003 MSE는 A 32.6%, B 0.13%, C 15.8% 낮아졌지만 Validation MSE는 모두 높아졌다. Epoch50을 사후 선택하거나 폐루프 실행하지 않았다.

### Native와 Controller 오류

| Method | Motion6 MSE | Translation3 MSE / MAE | Rotation3 MSE / MAE | Gripper1 MSE | Sign accuracy | Translation mean/p95 mm | Rotation mean/p95 rad | Gripper target mean mm | Offline close / Rollout close |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.00172554 | 0.000407116 / 0.0121168 | 0.00304397 / 0.0185175 | 0.00557666 | 100.000% | 2.527/6.845 | 0.00363/0.00822 | 0.482 | 49/53 |
| B | 0.00170694 | 0.000216104 / 0.00695282 | 0.00319777 / 0.0146432 | 0.0138247 | 99.270% | 1.525/5.945 | 0.00309/0.00745 | 0.794 | 50/23 |
| C | 0.00285637 | 0.000207008 / 0.00713628 | 0.00550574 / 0.0183765 | 0 | 100.000% | 1.599/5.437 | 0.00405/0.00809 | 0.000 | 49/39 |

C의 Expert 상태 Gripper sign은 137/137으로 맞았지만 실제 폐루프에서는 t=39에 첫 Close를 했고 파지하지 못했다. B는 Expert 상태에서 t=50이지만 폐루프 t=23이었다. A는 49/53이다. 세 방법 모두 정상 Expert 상태의 t=0부터 Controller translation 오차가 1 mm를 넘었고 실제 상대 geometry가 초기 몇 step에서 이탈했다. 같은 시점의 replay GT geometry 비교는 원본 Robot76과 byte-identical하지 않아 기술적 참고 지표로만 사용한다.

### Reset identity와 단계별 실패

| Method | qpos/qvel 최대 차이 | TCP/손가락 Pose 최대 차이 | RGB/Depth 최대 정수 차이 | Grasp | 양쪽 접촉 | 4 cm Lift | Insert | Peg 최고 Lift |
|---|---:|---:|---:|---|---|---|---|---:|
| A | 0/0 | 4.87e-07 | 1/1 | False | False | False | False | 0.0000 mm |
| B | 0/0 | 4.87e-07 | 1/1 | False | False | False | False | -0.0001 mm |
| C | 0/0 | 4.87e-07 | 1/1 | False | True | False | False | 0.0015 mm |

세 방법의 reset hash는 원본과 일치하고 RGB 및 Depth는 최대 1 정수 단위의 렌더 양자화 차이만 있었다. 모두 최초 파지 이전에 실패했으므로 Lift/Align/Insert의 폐루프 제어 능력은 이번 Full 세 Rollout에서 직접 관측되지 않았다.

### 판단 및 다음 실험 우선순위

1. **Training Fit 우선 진단:** 학습 Seed에서 1.2만~2.7만 배 큰 Action7 MSE와 파지 실패가 확인됐다. Development 0/10을 일반화만의 문제로 해석하지 않는다. 특히 Reach/Descend와 Close 구간별 Controller 오차가 실제 접근·Close 상태에 어떤 영향을 주는지 확인한다.
2. **Epoch 연장은 별도 사전등록 실험:** Epoch50의 Train Seed 개선은 후보 근거지만 Validation이 모두 악화했다. Epoch 연장을 시험한다면 Train96/Validation12와 사전에 정한 새 폐루프 Seed를 함께 평가해 overfit 위험을 분리한다. 이번 작업은 학습을 추가하지 않았다.
3. **상태 이탈과 표현 분석:** Expert 상태의 오차를 줄인 뒤에도 Rollout이 실패하면 자기 행동으로 방문한 RGB-D/Graph 상태, GRU history, 물체 presence 및 복구 능력을 조사한다. 이 결과만으로 representation이나 DAgger의 필요성을 확정하지 않는다.

### 감사와 재현 파일

- [중지 감사](../../experiments/1cbed275766549fda6945456503019aa/full_stop_audit.json), [프로세스 종료](../../experiments/1cbed275766549fda6945456503019aa/process_session_exit_status.json), [자원 감사](../../experiments/1cbed275766549fda6945456503019aa/resource_audit.json), [보호 감사](../../experiments/1cbed275766549fda6945456503019aa/pre_post_protection_audit.json)
- [원본 참조](../../experiments/1cbed275766549fda6945456503019aa/original_references.json), [Reset identity](../../experiments/1cbed275766549fda6945456503019aa/reset_identity_audit.json), [Reset 세부](../../experiments/1cbed275766549fda6945456503019aa/reset_identity_addendum.json), [Best/Final 오차](../../experiments/1cbed275766549fda6945456503019aa/best_final_comparison.json), [Single/Full 비교](../../experiments/1cbed275766549fda6945456503019aa/comparison.json)
- [최초 오류](../../experiments/1cbed275766549fda6945456503019aa/first_error_diagnostics.json), [계약 테스트 로그](../../experiments/1cbed275766549fda6945456503019aa/contract_test.log), [검증 결과](../../experiments/1cbed275766549fda6945456503019aa/verification.json), [보고서 문구 수정 내역](../../experiments/1cbed275766549fda6945456503019aa/report_correction.json)
- [진단 소스 Snapshot](../../experiments/1cbed275766549fda6945456503019aa/source_snapshot_diagnostic.py), [보고서 소스 Snapshot](../../experiments/1cbed275766549fda6945456503019aa/source_snapshot_report.py), [Implementation diff](../../experiments/1cbed275766549fda6945456503019aa/implementation.diff)
- Offline NPZ/CSV는 `offline/{A,B,C}/{best,final}/`, 실제 Trace/Action/Hidden/RGB-D/Graph/Contact는 `rollout/{A,B,C}/`에 보존했다.
