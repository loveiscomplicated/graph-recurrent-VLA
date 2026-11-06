# Graph Prototype v4 — History Stacking

이번 실험은 고정된 과거 window가 v2 BC controller의 제어 성능을 개선하는지 확인한다. RNN/GRU/LSTM/Transformer, hidden state, RL, DAgger를 구현하지 않았다. Graph 팀 경로에만 코드를 추가했고 기존 Surface graph/BC objective/action convention을 유지했다.

## 구현과 데이터 정렬

v2 geometry encoder를 그대로 공유한다. Surface는 131 nodes × 14 features(type one-hot 5, TCP-relative position 3, local position 3, world normal 3), context 81, edge features 8, 3 message layers다. Flat은 동일 node features와 context를 1915-vector로 받아 기존 MLP로 encoding한다. Context에는 gripper2, qpos9, EE/root world position 및 rotation matrix, peg/box/hole의 TCP-relative position 및 world rotation, half sizes, hole radius/offset이 들어간다.

두 encoder 모두 embedding128을 출력하며 공통 bypass31(qvel9 + goal22)을 붙여 timestep당 h159를 만든다. Goal22는 EE-frame grasp displacement3, grasp-relative rotation6, hole-frame peg-head3, peg-relative rotation6, clearance1, insertion axis3다. 기존 qpos/gripper/geometry를 중복 추가하지 않는다. Action head는 기존 ReLU MLP, hidden128, output7이다.

| History mode | Head input (K=4) | 구성 |
|---|---:|---|
| Current | 159 | h_t |
| Action | 191 | h_t + past actions28 + validity4 |
| State | 799 | past/current h795 + validity4 |
| Both | 827 | past/current h795 + past actions28 + validity4 |

State/action은 오래된 순서에서 최신 순서로 concatenation한다. `length=4`는 현재를 제외한 과거 개수다. 부족한 state는 s_0를 반복하고 action은 zero padding한다. Validity4는 두 modality에서 같은 timestep 유효성을 나타내므로 Both에서도 한 번만 제공한다. Current는 mask/추가 parameter가 없어 v2와 완전히 동일하다.

전체 원본 HDF5 audit은 91 episodes/13,958 targets의 action 일치, 현재 obs의 qvel, 경계/누출 검사를 통과했다. Train/validation/test samples는 11,274/1,325/1,359다. `outputs/graph/v4/data_audit.json`에 차원과 parameter count도 저장했다.

Dataset의 `timestep=t` geometry와 qvel은 `obs[t]`, target은 `actions[t]`다. HDF5는 T actions/T+1 observations/env states이며 terminal s_T는 action target row가 없다. Window는 episode별 contiguous timestep을 검사한 후 생성한다. Input actions는 `a[t-K:t]`로 a_t를 포함하지 않는다. Train/validation/test는 기존 91 episodes의 73/9/9 split이다. Geometry/state normalizers는 train의 실제 state rows에서만 계산하며 padding/window 중복은 통계에 포함하지 않는다.

Actions는 기존 normalized `pd_ee_delta_pose` 7-vector를 그대로 사용하며 별도 action z-score나 displacement target을 추가하지 않는다. 원본 converted demonstration은 [-1,1] 범위이므로 component clamp로 바뀌지 않는다. Train history는 expert command, learner history는 `env.step(applied)`에 실제 전달한 component-clipped command다. Controller 내부 translation scaling ±0.1m 및 rotation norm clipping/scaling은 유지된다. Command, controller 내부 변환, 실제 TCP displacement는 서로 다른 값이다.

## Training과 runtime

모든 timestep에 하나의 encoder parameter 집합을 사용한다. Training은 batch 내 동일 state index만 deduplicate한 뒤 encode/indexing하며 gradient를 끊지 않는다. Evaluation은 고정된 `model.eval()` encoder의 h159를 cache해 매 step 현재 state만 encode한다. Runtime cache는 학습에 사용하지 않는다.

Predict → action clamp → env.step → pre-step representation/applied command commit 순서로 갱신한다. Episode reset은 buffer를 비운다. Warm-start prefix replay는 매 expert step 직전 live state를 encode하고 실제 적용 command를 commit한다. Prefix 끝에서 reset하지 않고 learner history로 이어진다. Prefix observation reconstruction error >.001이면 평가를 중단한다. Full start는 과거가 없어 padding으로 시작한다.

Current baseline은 기존 v2 completed 30-epoch checkpoint를 별도 v4 path로 복사했다. Validation/test 전체의 예측을 신규 pipeline과 비교한 maximum absolute difference가 Flat/Surface 모두 정확히 0이었다. Original checkpoint/normalizers는 보존했다. Closed-loop에서도 Current baseline의 4개 held-out 조건 모두 기존 v2와 initial geometry, episode별 success 및 전체 applied-action sequence가 정확히 일치했다. 나머지 6개 설정은 seed0, AdamW lr=.001/weight_decay=1e-5, cosine 30 epochs, batch32, threads4, 7-component mean squared BC loss, validation minimum checkpoint로 실행한다. 테스트 결과에 따른 설정 변경은 없다. 독립된 Surface State/Both 학습은 CPU 프로세스별 threads4를 유지한 채 병렬 실행했다. 순차 driver만 종료하고 진행 중인 Surface Action 학습 subprocess는 끝까지 유지했으며, orchestration 변경은 `outputs/graph/v4/orchestration.json`에 기록했다.

Evaluator는 v2 original reset snapshot/converted prefix/thresholds를 재사용한다. Acquired와 transport prefix는 `v2/diagnostics/expert.json`의 starts, alignment는 기존 phase3(없으면 phase4) 최초 index다. Horizon은 prefix 뒤 learner250 steps이며 TimeLimitWrapper는 prefix+250으로 설정한다. 등록 task horizon100과 구분한다. Success는 rollout 중 `info['success']`가 한 번이라도 true인 경우다. Stage counts/failure 분류는 v2와 같은 `stage_summary`를 사용한다.

## 검증과 실제 측정

측정표는 [results.md](../outputs/graph/v4/comparison/results.md), 전체 수치는 [results.json](../outputs/graph/v4/comparison/results.json)에 생성된다. 생성되지 않았거나 실행되지 않은 조건은 N/A다. Full held-out9 / fresh30 / acquired9 / transport9 / alignment9는 각각 별도의 denominator를 가진다. Warm-start success를 full success나 full rollout에서 alignment에 도달한 비율로 해석하지 않는다.

- Window alignment, current/future action leakage, episode boundary, initial state/zero action padding, mask, K=0/1/4/8/16 shape.
- 8개 설정 forward/backward/finite gradients/checkpoint round-trip, 단일 encoder parameter 집합, 모든 timestep representation의 gradient.
- 실제 prefix77 history와 offline window 일치, full start padding, 제어 전환 뒤 정확한 learner applied command 사용.
- 실제 train demonstration 16 samples/8개 설정 200-step overfit. 이는 main training과 별도의 Adam .003 최적화 smoke이며 held-out 결과가 아니다.
- 기존 simulator/GT/v1/v2/v3 regression42 및 v4 테스트. 로그는 `outputs/graph/v4/{tests,regression,overfit}.log`.
- 모든 model/condition의 paired initial geometry fingerprint와 history transition 자동 검증.
- 이전 output artifact hash 보존 검사. 기존 v3 untracked 파일도 수정하지 않는다.

Runtime trace에는 full pre/post state, qpos/qvel, gripper, contact, raw/applied action, history action/mask, representation norm, realized TCP translation/rotation, step flags와 stage failure가 있다. Summary는 action delta RMS, gripper sign switches, clipping fraction과 failed learner state/action 분포를 기록한다. `distribution.py`는 train expert marginal distribution과 failed learner qvel/gripper/hole-head/action을 비교한다. 이는 기술통계이며 distribution shift의 인과 증거로 해석하지 않는다.

History diagnostic은 학습된 Both 모델의 동일 held-out conditions에서 h_t 반복 + zero past actions로 교체한다. Padding validity는 유지한다. Offline에서도 모든 history model의 ablated error를 함께 기록한다. 이 입력은 OOD일 수 있어 성능 하락만으로 temporal reasoning을 입증하지 않는다.

Latency는 feature/graph construction+normalization, 현재 encoder, history processing, action head, cache commit으로 나눠 측정한다. `latency.py`는 실험 종료 뒤 CPU idle 시 재측정하고 uncached whole-window 비용 및 cache/offline prediction agreement도 기록한다. GT extraction/physics stepping은 제외된다. 실험 중 concurrent process가 있는 rollout report의 초기 latency 대신 최종 idle latency를 비교표에 사용한다.

## 재현

[README](../graph/prototype/v4/README.md)에 fresh-root full reproduction 명령이 있다. 기본 실행:

```bash
.venv/bin/python -m unittest graph.prototype.v4.tests.test_history -v
.venv/bin/python -m graph.prototype.v4.overfit --output outputs/graph/v4/overfit_new.json
.venv/bin/python -m graph.prototype.v4.run_matrix --phase train --root outputs/graph/v4_new
.venv/bin/python -m graph.prototype.v4.run_matrix --phase evaluate
.venv/bin/python -m graph.prototype.v4.run_matrix --phase ablate
.venv/bin/python -m graph.prototype.v4.latency
.venv/bin/python -m graph.prototype.v4.distribution
.venv/bin/python -m graph.prototype.v4.compare
.venv/bin/python -m graph.prototype.v4.figures
.venv/bin/python -m graph.prototype.v4.report
```

`run_matrix --phase train`은 Current import가 이미 있는 root에서 6개 history 모델을 학습한다. 완전히 새 root에서는 README loop로 Current도 학습하거나 원본 v2 checkpoint를 별도로 검증/복사해야 한다. Checkpoint와 평가 파일을 덮어쓰지 않으므로 기존 root에 명령을 반복하지 않는다.

Live host 확인은 `examples/check_env.py --control-mode pd_ee_delta_pose`로 실행했다: ManiSkill3.0.1, SAPIEN3.0.3, Gymnasium1.3.0, Torch2.14.1, CPU `physx_cpu`, renderer `sapien_cpu`/MoltenVK, obs[1,43], action7[-1,1], reset/step 성공. CUDA는 unavailable이다. 사용자 요청으로 Surface State/Both의 나머지 학습을 MPS로 전환했다. CPU 기록은 `surface_{state,both}_cpu_interrupted/`에 보존했다. State는 saved best epoch11, Both는 epoch14에서 optimizer/scheduler를 복원해 재개하므로 live partial batch에서의 exact continuation은 아니다. Shuffle generator는 기존 seed/per-epoch permutation으로 복원했다. MPS의 index_add는 strict deterministic 모드를 지원하지 않아 warn_only를 사용한다. 학습 backend가 섞인 비교이며 CPU와 bitwise 동일한 결과를 보장하지 않는다. Simulator 및 최종 offline/latency inference는 CPU로 유지한다.

## 해석 범위와 후속 실험

아래 완료된 실험 결과는 실제 측정값을 기준으로 정리했다. 단일 seed와 9-episode warm-start 비교는 구조의 일반적 우열을 확정하지 못한다. History head parameter 증가가 동반되므로 개선이 관찰되면 parameter-matched Current control을 후속으로 고려한다. K8/K16 성능, v3 graph+history, 복수 seed, parameter-matched control과 GRU 비교는 이번 최소 matrix의 필수 작업이 아니며 별도 후속 실험이다. 이번 구현에는 새로운 temporal architecture가 없다.

## MPS tmux 재개

`tmux attach -t graph-v4-mps`로 확인한다. `state`, `both` 창에서 batch32/K4 두 MPS 학습을 동시에 실행하고 `evaluate` 창은 완료 후 기존 평가/ablation/비교/report를 실행한다. 모델별 MPS memory fraction=.4다. 실제 병렬 smoke에서 forward/backward/AdamW 복원/weight update를 확인했고 CPU-MPS prediction max difference는 약3.6e-7이었다. `mps_smoke.json`, `mps_parallel_{state,both}.json`, `mps_resume_smoke/`, `mps_migration.json`에 증거를 기록했다. 이후 trainer는 매 epoch `last.pt`에 optimizer/scheduler/generator 상태도 저장한다.

<!-- MEASURED_RESULTS -->
## 완료된 실험 결과

| Encoder | History | Action MSE | Full SR held-out | Full SR fresh | Alignment-start SR | Params | CPU cached ms |
|---|---|---:|---|---|---|---:|---:|
| flat | current | 0.00181967 | 0/9 | 0/30 | 0/9 | 299655 | 0.416 |
| flat | action | 0.00131254 | 0/9 | 0/30 | 1/9 | 303751 | 0.422 |
| flat | state | 0.00117103 | 0/9 | 0/30 | 0/9 | 381575 | 0.424 |
| flat | both | 0.00118009 | 0/9 | 0/30 | 1/9 | 385159 | 0.423 |
| surface | current | 0.00156587 | 0/9 | 0/30 | 1/9 | 449031 | 2.434 |
| surface | action | 0.00101473 | 0/9 | 0/30 | 0/9 | 453127 | 2.434 |
| surface | state | 0.00118573 | 0/9 | 0/30 | 1/9 | 530951 | 2.713 |
| surface | both | 0.00107910 | 0/9 | 0/30 | 0/9 | 534535 | 2.706 |

Full held-out 9개와 fresh 30개는 별도 조건이다. Alignment-start는 expert prefix 이후 9개 episode의 task success이며 full-task alignment reached와 구분한다. Latency는 idle host, CPU4 threads, batch1, 100 repeats, cache commit 포함/GT extraction 및 simulator step 제외다.

### 질문에 대한 답변

1. **Offline error:** Current 대비 MSE 변화는 flat: action -27.9%, state -35.6%, both -35.1%; surface: action -35.2%, state -24.3%, both -31.1%. 음수는 감소다. Expert-history 조건의 held-out error다.
2. **Closed-loop:** Full-task success 개선은 확인되지 않았다. Held-out/fresh 조건에서 모든 모델이 각각 0/9, 0/30이다. Warm-start 결과는 위 표와 비교표에 별도로 기록했다.
3. **State vs Action:** flat의 최저 offline MSE는 state (0.00117103); surface의 최저 offline MSE는 action (0.00101473). 이것만으로 closed-loop에서 더 효과적이라고 결론 내리지 않는다. Head input과 parameter count도 함께 변했다.
4. **Surface vs Flat:** 같은 history 조건에서 Surface의 offline MSE 변화(Flat 대비)는 current -13.9%, action -22.7%, state +1.3%, both -8.6%. Closed-loop 우열은 별도의 success/단계별 실패 결과로 판단해야 하며 이 소규모 실험으로 일반적인 encoder 우위를 확정하지 않는다.
5. **남은 실패 단계:** Full held-out 실패는 flat/current: acquisition 7, retention 2; flat/action: acquisition 4, retention 3, transport 2; flat/state: acquisition 3, retention 6; flat/both: acquisition 3, retention 6; surface/current: acquisition 7, retention 1, transport 1; surface/action: acquisition 6, retention 1, transport 1, alignment 1; surface/state: acquisition 6, retention 3; surface/both: acquisition 5, retention 3, alignment 1. 전체 warm-start/fresh failure counts는 comparison에 있다. Trace는 gripper command, contact, applied action 변화와 state 분포를 남겼지만 어느 문제가 실패의 원인인지 인과적으로 분리한 실험은 아니다.
6. **GRU 근거:** History가 offline 예측에 도움이 되는 결과는 temporal 입력을 후속 연구할 근거다. 그러나 고정 window만으로 full-task 제어가 해결됐다는 근거가 없으므로 GRU 도입이 다음 병목을 해결한다고 결론 내릴 수 없다. BC의 expert/learner history 분포 차이, phase ambiguity, grasp retention 및 gripper/controller 문제를 분리해 확인하고 parameter-matched Current와 복수 seed를 비교하는 것이 후속 후보다.

### 진단과 검증

- flat/Both ablation: offline MSE 0.00118009 → 0.02504234; full 0/9 → 0/9; alignment-start 1/9 → 0/9. 현재 representation 반복/zero action 입력은 OOD일 수 있다.
- surface/Both ablation: offline MSE 0.00107910 → 0.02163112; full 0/9 → 0/9; alignment-start 0/9 → 0/9. 현재 representation 반복/zero action 입력은 OOD일 수 있다.
- v4 tests7 + 기존 regression42 = 49 tests 통과. 별도 실제 train demonstration 8개 설정의 200-step small-batch overfit도 모두 통과했다.
- Paired initializations: True; learner history = 실제 applied command: True; 이전 output artifacts SHA256 보존: 185/185.
- Current baseline은 offline prediction과 held-out 4조건의 전체 applied command sequence까지 기존 v2와 정확히 일치했다.
- 성공/실패 Flat Both alignment video는 기록된 applied commands를 다시 실행해 EE/peg pose maximum error 0으로 검증했다.

### 수정 파일과 미실행 범위

`graph/prototype/v4/`에 history dataset/buffer/models, config, train/evaluate, baseline import, matrix driver, compare, distribution/latency, report/figures/video, overfit 및 tests를 추가했다. 이 보고서 외 기존 shared/Visual/Recurrent/v1/v2/v3 코드는 수정하지 않았다. Checkpoints, traces, metrics, timing, figures/videos는 `outputs/graph/v4/`에 있다.

필수 K4 matrix의 학습/offline/full/warm-start/fresh/Both ablation은 완료했다. K8/K16 성능, v3 graph+history, parameter-matched Current, 복수 seed 및 GRU 비교는 후속 실험으로 남긴다. 이번 최소 matrix는 K4로 고정했으며 window 길이나 test-informed 설정을 확장하지 않았다. CUDA는 이 호스트에서 unavailable이며 CUDA backend는 실행하지 않았다. Surface State/Both의 일부 학습은 사용자 요청으로 MPS에서 재개했으며 CPU/MPS backend 차이와 deterministic warn_only 제약을 함께 고려해야 한다. 새로운 초기화에서의 history ablation은 필수 진단 범위에 포함하지 않았고 held-out full/alignment 조건에서만 실행했다.
