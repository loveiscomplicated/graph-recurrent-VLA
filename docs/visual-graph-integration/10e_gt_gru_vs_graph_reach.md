# Task 10-E — GT-state GRU vs Graph Reach

Experiment UUID: `771c27f66f6d4f27850a41a87478dc16`. 실제 GT closed-loop 54회 완료. Reach 13/54, Insertion 35/54. 새 학습·가중치 변경·Git commit·외부 publishing 없음.

**판정: COMPLETED_WITH_PAIRING_LIMITATION.** 기존 A/B와 기록된 초기 물리 필드 15개 및 CPU PhysX·controller config·주파수·Reach predicate를 대조했다. 그러나 Task10D에는 초기 controller/full simulator snapshot이 없어 그 전체 상태의 직접 동일성은 NOT_TESTED다. 엄격한 full-state paired 성공률로 합산할 수 있는 분모는 0이며, 아래 수치는 기록된 물리 상태를 맞춘 비교다. 완전한 Paired Evaluation PASS라고 보고하지 않는다.

## 복원과 실행

지정 HF 재평가 백업은 결과·소스·초기상태 fixture를 포함하지만 best.pt는 없다. 원본 main 학습 백업의 Clean seed0/1/2를 사용했으며 각 SHA256이 재평가 identity와 일치했다. Hidden144, encoder128, GRU1 layer, 142,823 parameters다. 원본 encoder·Corruptor·normalize_obs·LearnedPolicy를 isolated source에 복원해 실제 로드했다. 원본 소스 백업 자체는 원래 dirty 실행과의 완전 동일성이 보증되지 않는다고 manifest에 명시돼 있다.

Clean은 fresh=1, age=0이다. 36D GT+freshness2D 순서·world displacement·wxyz canonical quaternion을 그대로 사용한다. Checkpoint의 obs_mean/std/keep와 act_mean/std를 재사용했다. Output6 denormalize/clip, gripper logit>0만 +1이다. Hidden은 episode마다 None, step마다 carry한다. 미래 GT·expert action·target·reward는 입력에 없다.

원본 NumPy encoder와 공식 tensor CPU encoder, action과 streaming hidden을 세 checkpoint에서 각각12step 검증했다. CUDA inference도 비교했으나 본 실행은 공식 CPU LearnedPolicy(평균 약0.5ms/step)를 사용했다. 현재 host는 RTX3090 Linux이며 정확한 버전과 controller config는 environment contract에 있다. CPU/GPU renderer 혼용 실패 첫 smoke는 보존했고 native 검사를 subprocess로 분리했다. macOS/CUDA PhysX/MPS는 본 평가에서 실행하지 않았다.

## 평가 계약

Final seeds61000–61005 × frozen nominal/seen_type/unseen_diagonal 18개를 각 학습 seed에 적용했다. Offset vector는 benchmark_protocol.json에 그대로 저장했다. 기존 reset guard를 수정하지 않았고 pose 강제 보정·seed 교체도 없다. 실제 현재 GT38D→GRU→원래 controller→env.step을 실행했다.

Task10D의 실제 평가 horizon은130, simulator TimeLimit은250이다. Primary는 동일하게130actions 이내 Reach이며 endpoint는 최초 성공 또는130번째 action이다. GT full-task는 최대250actions 계속 실행한다. Reach 이후 자연스러운 close·운반은 이미 성취한 Reach를 실패로 되돌리지 않는다. 130step의 고정 target 오차는 manipulation 중 이동을 포함하므로 별도 fixed130 컬럼으로 분리했다. Target은 초기 expert pose로 고정하고 metric 계산에만 쓴다.

Reach는 approach10mm/0.05rad 이후 pregrasp4mm/0.03rad, opening≥65mm, 조기close없음, pegdrift<2mm 및 collision audit를 그대로 사용한다. 본문 endpoint error는 성공하면 일찍 종료한 Graph와 같은 stopping convention이다. 전체 trajectory의 최소 위치/자세 오차가 서로 다른 step에서 발생할 수 있으므로 각각의 최소값만으로 동시 target 도달이라고 판정하지 않는다.

## Seed별 결과

|Train seed|Group|Reach|Grasp|Lift|Insertion|Final pos mm|Min pos mm|Final rot rad|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|0|all|6/18|18/18|18/18|18/18|202.089|7.516|0.2680|
|0|nominal|2/6|6/6|6/6|6/6|202.834|7.667|0.2687|
|0|seen_type|2/6|6/6|6/6|6/6|202.155|7.415|0.2687|
|0|unseen_diagonal|2/6|6/6|6/6|6/6|201.278|7.467|0.2667|
|1|all|3/18|18/18|18/18|9/18|229.819|10.304|0.4758|
|1|nominal|1/6|6/6|6/6|3/6|228.833|10.097|0.4680|
|1|seen_type|1/6|6/6|6/6|3/6|228.705|10.618|0.4833|
|1|unseen_diagonal|1/6|6/6|6/6|3/6|231.920|10.196|0.4759|
|2|all|4/18|18/18|18/18|8/18|235.358|5.412|0.4004|
|2|nominal|1/6|6/6|6/6|3/6|255.192|5.127|0.4309|
|2|seen_type|2/6|6/6|6/6|3/6|194.413|5.468|0.3427|
|2|unseen_diagonal|1/6|6/6|6/6|2/6|256.470|5.642|0.4276|

A/B 원본 Final Reach는 각각0/18이며 수정하지 않았다. 이 수치는 원본 결과이고 이번에 재실행하지 않았다. 54회는18개 초기조건을 학습 seed3개에서 반복한 상관된 관측이며 새 blind test가 아니다.

## Q1–Q8

**Q1:** 세 원본 체크포인트 로드 및 SHA·38D·normalizer·decoder·streaming 검증 PASS. 원본 GPU physics 전체 rollout 재현과 source dirty provenance는 별도 한계다.
**Q2:** 기록된 물리 초기상태·offset·controller config·CPU backend·주파수·primary horizon·predicate를 대조했다. 새 full-state/controller 반복 reset은 검사했지만 역사적 full-state snapshot 부재로 엄격한 paired PASS는 아니다.
**Q3:** Reach 13/54. 학습 seed별·offset별 분모는 위 표에 있다. Full-task insertion 35/54와 분리한다.
**Q4:** 정확한 조건은 아래 목록과 paired_reach_comparison.csv에 모두 공개한다. A/B는 같은 모든 조건에서 실패했다. 다만 strict full-state pairing 제한이 적용된다.
**Q5:** 아래 model별 error 평균과 그림을 본다. p-value나 독립표본 우위를 주장하지 않으며 endpoint의 stopping convention과 full-task 이후 오차를 구분한다.
**Q6:** 원래 GPU Clean 성공은 seed0=25/30, seed1=15/30, seed2=14/30, 합계54/90=60%다. 이번 native CPU smoke는 고정2000·2001 각 seed2회만 실행했다. 이 소표본은 GPU30seed 성공률 재현 검사가 아니다. 이번 Final18은 seed·offset·physics·평가목표가 다르다.
**Q7:** 기록된 물리 상태에서 oracle GRU가 실제 grasp/lift/insertion을 수행하고 일부 strict Reach도 통과했다는 사실은 물리적으로 접근 가능한 환경임을 지지한다. 현재 live Vision→Geometry→Graph→GNN→GRU 시스템의 입력·학습·제어 연결을 더 살펴볼 근거다. Vision 또는 GNN 하나를 원인으로 확정할 수 없다. Full-task/Reach-only, demonstration, feature 정보량, capacity, runtime 차이가 함께 남는다. 특히 scripted grasp target과 다른 유효 grasp 경로는 이 Reach 기준에서 실패하므로 strict Reach가 기능적 접근 가능성의 완전한 대리 지표가 아니다.
**Q8:** 다음 한 가지 통제는 frozen A/B에 현재 GT pose에서 만든 정확한 geometry를 기존 graph schema 그대로 공급하는 oracle-geometry inference ablation이다. 같은 full simulator/controller 초기 snapshot을 세 정책에 저장·복원하고, predicted geometry와 GT geometry를 바꾼 frozen A/B를 비교한다. 새 학습 없이 perception/geometry의 영향을 먼저 좁힐 수 있으며 graph construction 의미가 같다는 계약 검사가 선행돼야 한다. 이번 작업에서는 실행하거나 feature를 개발하지 않았다.

## GT Reach 성공 사례 (eval seed / offset 유형)

- train0: 61000/nominal, 61000/seen_type, 61000/unseen_diagonal, 61005/nominal, 61005/seen_type, 61005/unseen_diagonal
- train1: 61001/nominal, 61001/seen_type, 61001/unseen_diagonal
- train2: 61002/seen_type, 61003/nominal, 61003/seen_type, 61003/unseen_diagonal

Reach 실패지만 Insertion 성공한 rollout은 25개다. 엄격한 approach/자세/close timing 계약의 실패와 실제 물체 조작 실패를 동일시하지 않는다. failure_diagnostics.csv에서 구분한다.


대표적인 metric 불일치: GT train0/eval61001 nominal은 close 직전 위치오차 약17.3mm여서 strict Reach에는 실패하지만 Grasp/Lift/Insertion은 성공했다. 실패 후130step endpoint는 이미 운반·삽입 중이므로 고정 초기 pregrasp target에서 멀어질 수 있다. Endpoint가 크다는 것만으로 실제 접근이 나빴다고 결론내리지 않는다. 최소오차와 close 시점·실제 조작 결과를 함께 본다.


## 공통 target error (18조건 평균; GT는 학습 seed별)

|Model|Final pos mm|Min pos mm|Final rot rad|
|---|---:|---:|---:|
|A|121.254|87.263|0.3978|
|B|93.151|65.153|0.6590|
|GT s0|202.089|7.516|0.2680|
|GT s1|229.819|10.304|0.4758|
|GT s2|235.358|5.412|0.4004|

## Native CPU smoke

|Train seed|Eval seed|Insertion|Actions|
|---|---|---|---|
|0|2000|True|151|
|0|2001|True|98|
|1|2000|True|152|
|1|2001|True|103|
|2|2000|False|250|
|2|2001|True|94|

## Contract status

|항목|상태|
|---|---|
|checkpoint_identity_3seeds|PASS|
|checkpoint_original_eval_SHA_match|PASS|
|feature_order_units_clean_freshness|PASS|
|normalizer_action_decoder|PASS|
|streaming_hidden_native_tensor_CPU|PASS|
|eval_no_grad|PASS|
|unchanged_weights|PASS|
|current_GT_only_no_future_or_expert_input|PASS|
|Graph_input_unchanged|PASS|
|Reach_threshold_sequence_and_latch|PASS|
|action_controller_scaled_alignment|PASS|
|hidden_episode_reset|PASS|
|trace_timestep_alignment|PASS|
|fixed_seed_native_CPU_rollouts|PASS|
|CPU_PhysX_common_backend|PASS|
|exact_offset|PASS|
|action_frequency20_physics100|PASS|
|primary130_secondary250|PASS|
|no_training_no_publishing|PASS|
|rollout_metric_artifact_tests|PASS|
|remote_SHA_readback|PASS|
|video_first_last_frame_decode|PASS|
|inherited_nonrecurrent_source_preserved|PASS|
|controller_reset_source_reconstruction|PASS|
|original_download_manifest_SHA|PASS|
|recorded_physical_state_AB_exact|PASS|
|full_state_repeat_determinism|PASS|
|historical_full_simulator_controller_identity|NOT_TESTED|
|GPU_PhysX|NOT_TESTED|
|original_GPU_30seed_success_reproduction|NOT_TESTED|
|MPS|NOT_TESTED|
|CUDA_inference|PASS|
|CUDA_inference_adopted|NOT_TESTED|
|historical_contact_settings_runtime_snapshot|NOT_TESTED|

Historical controller state와 contact runtime snapshot은 소스에서만 복원 가능하며 실제 역사적 snapshot과 필드 비교를 수행한 것으로 표현하지 않는다. Task10D 원본은 read-only로 유지했다. 기존 CPU source/controller 설정을 재사용했으며 CUDA PhysX로 기존60%와 직접 합산하지 않았다.

## 산출물·검증

[Checkpoint inventory](../../outputs/task10e/recurrent_checkpoint_inventory.json) · [38D contract](../../outputs/task10e/gt38_input_contract.json) · [Environment](../../outputs/task10e/evaluation_environment_contract.json) · [Reset audit](../../outputs/task10e/reset_identity_audit.json) · [Protocol](../../outputs/task10e/benchmark_protocol.json) · [Metrics](../../outputs/task10e/gt_gru_rollout_metrics.csv) · [Paired cases](../../outputs/task10e/paired_reach_comparison.csv) · [Diagnostics](../../outputs/task10e/failure_diagnostics.csv) · [Verification](../../outputs/task10e/verification.json) · [Artifact SHA index](../../outputs/task10e/artifact_index.json) · [Static HTML](../../outputs/task10e/report.html)

원래 ExperimentLogger를 사용했고 새 UUID·로컬 scalar·sidecar를 보존했다. 공통 logger/artifact 42 tests PASS. rollout metric alignment 검사는 test_metrics.log에 기록한다. 원본 Source/체크포인트/관련 A/B artifact SHA 및 신규 trace/video SHA는 artifact_index.json에 있다. 실행 로그와 실패 smoke는 outputs/task10e에 보존했다.

![Reach](../../outputs/task10e/figures/reach_success.png)

![Target errors](../../outputs/task10e/figures/target_errors.png)

![Trajectories](../../outputs/task10e/figures/distance_orientation_steps.png)

대표 영상:

- [GT train0, eval61000 nominal — Reach True, Insertion True](../../outputs/task10e/benchmark/rollouts/s0/61000_nominal/rollout.mp4)
- [GT train1, eval61000 nominal — Reach False, Insertion True](../../outputs/task10e/benchmark/rollouts/s1/61000_nominal/rollout.mp4)
- [GT train2, eval61000 nominal — Reach False, Insertion False](../../outputs/task10e/benchmark/rollouts/s2/61000_nominal/rollout.mp4)
- [GT train0, eval61001 nominal — Reach 실패 / Insertion 성공](../../outputs/task10e/benchmark/rollouts/s0/61001_nominal/rollout.mp4)
