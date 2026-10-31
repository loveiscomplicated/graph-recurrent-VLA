# Task10-I Stage3 Expert Dataset Collection

Status: **COMPLETED**, subject to the separately generated completion audit. ExperimentLogger UUID `a66ec3d698e249b588f392e76038b54f`. Executed root: `/workspace/task10i_stage3_20261031_v5_finalization/run`. Stage2 selection UUID `34d10bf652b344a38d54b7672dbfe0c0` remains immutable.

## Dataset result — CONFIRMED

| Dataset | Parents | Logical | Full | Corrective | Action frames | BC frames | Disturbance frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 96 | 272 | 96 | 176 | 17201 | 16168 | 1033 |
| D | 96 | 272 | 96 | 176 | 17215 | 16182 | 1033 |
| validation | 16 | 48 | 12 | 36 | 3037 | 2625 | 412 |

R/D contain exactly the preserved Train33/82 episodes plus verified new templates; shared Validation preserves Parent7/23 and adds the frozen nine slots. Existing visibility-failed215 logical episodes, original raw/GT/checkpoints and Stage1 candidate data were not modified. Full-task validation trajectories without a planned logical entry remain diagnostic sources, excluded from counts. R/D share **78 parents / 236 identical logical references**, with exact raw/GT paths and SHA equality. Train/Validation parent leakage, Diagnostic Holdout use and historical Final/Test inclusion are zero.

## Actual expert success and attrition — CONFIRMED

Q1: **87/90** original selected parents have an approved official full-task success. Q2: per-parent role/membership, reasons and outcomes are fully retained in [selection_attrition_analysis.csv](../../outputs/task10i/stage3/run/selection_attrition_analysis.csv). Runtime errors are infrastructure failures and are never counted as expert task failures. Q3/Q4: **3 reserve assignments** were made using frozen queues in validation/common/R/D order; skips and failures are preserved in [reserve history](../../outputs/task10i/stage3/run/reserve_replacement_history.csv). Original, expert-success-conditional and final distributions are retained in [distribution analysis](../../outputs/task10i/stage3/run/selection_distribution_analysis.json), with [fixed-distance coverage](../../outputs/task10i/stage3/run/replacement_coverage_comparison.csv). We do not re-run greedy or alter weights/seeds. Expert conditioning is selection bias and limits generalization beyond this eligible, expert-solvable distribution.

## Sensor, action and GT contract — CONFIRMED

Q5: [shared reference audit](../../outputs/task10i/stage3/run/shared_episode_reference_audit.json) checks every shared physical parent/branch by raw and GT SHA. Q6: original uint8 RGB/int16-mm depth at Main `base_camera` and Wrist `hand_camera`, measured Robot76 and unchanged native Action7 are retained. Every T+1 observation/T action sequence is reopened; timestamps, measured robot/GT, quaternion order/world coordinates/metres, GT38 construction, controller preprocessing and masks are audited. Terminal observations have no invented action. New controller-state readback and physical diagnostics are additional audit fields; historical full-task recordings did not include the newer full substep contact audit. The base sensor/action/GT schema is unchanged. No GT sidecar is passed to student inference.

Q7/Q8: Corrective history contains exact approved nominal prefix, all real TCP disturbance observations and fresh current-state Expert recovery. Disturbance BC is always OFF; prefix BC is ON only after exact nominal comparison; recovery BC is ON. Frozen bounded feedback, offsets, anchors and SafetyV2 tolerances remain unchanged. Disk snapshots and 2×5-step observed replay are tested; PhysX hidden solver caches are not claimed serialized. No learned hidden state is stored. Corrective success is pre-grasp pose recovery under the inherited <4mm/<0.03rad and open-gripper contract, rather than an invented full insertion label.

## Physics and execution limits

Existing F1 full-task evidence covers executed expert commands, finite robot/object states, GT alignment and official insertion; historical full-task SafetyV2 substep contacts remain **NOT_TESTED**. New full-task records retain all five physics-substep contacts and joint/finite checks, with finger/peg grasp compliance recorded separately from abnormal non-grasp penetration at the inherited200µm tolerance. Grasp contact forces are recorded without inventing a rejection cap. Corrective uses the original no-contact robot/object and enabled self-collision SAT safety audit. No continuous swept-volume collision freedom is claimed.

Linux CPU PhysX / CUDA SAPIEN rendering on RTX3090 instance57987772 executed at20Hz control/100Hz physics and128×128 cameras. GPU PhysX, macOS simulation, perception, graph inference and policy learning/evaluation were not executed. Task10-H and unrelated processes were not interrupted. The measured worker benchmark selected **3 workers**; [engineering gates](../../outputs/task10i/stage3/run/engineering_gates.json) preserve actual one/two/three-worker throughput and exact sensor/action/GT equivalence, simulator crash/resume, verified skip, distinct-role full-task smoke and a separate miniature end-to-end dataset. Engineering data are excluded from research manifests.

## Readiness — CONFIRMED / NOT_TESTED

Q9: [frame statistics](../../outputs/task10i/stage3/run/dataset_frame_statistics.csv) provide every episode's Full-task, Prefix, Disturbance, Recovery, BC and geometry-eligible frame count plus phase distribution and parent length. Equal episode counts do not imply equal exposure; no frames were truncated to equalize datasets. Future Stage5 sampling is outside this task. Q10/Q11: [completion summary](../../outputs/task10i/stage3/run/dataset_completion_summary.json) and [split audit](../../outputs/task10i/stage3/run/final_split_and_leakage_audit.json) prove counts and initial visibility/split preservation. Q12: [Stage4 receipt](../../outputs/task10i/stage3/run/stage4_raw_dataset_receipt.json), [raw manifest](../../outputs/task10i/stage3/run/stage4_recache_input_manifest.json) and [schema contract](../../outputs/task10i/stage3/run/stage4_source_and_schema_contract.json) are ready for a later explicitly requested recache. Q13 policy improvement: **NOT_TESTED**. No Stage4 perception decision, graph recache, GNN/GRU training, student rollout, commit or external publication occurred.

## Artifacts

- [Raw/GT/action audit](../../outputs/task10i/stage3/run/action_gt_alignment_audit.json), [reset audit](../../outputs/task10i/stage3/run/initial_reset_replay_audit.json), [parent/branch audit](../../outputs/task10i/stage3/run/parent_branch_identity_audit.json)
- [Full-task video](../../outputs/task10i/stage3/run/media/expert_full_task.mp4), [recovery video](../../outputs/task10i/stage3/run/media/corrective_recovery.mp4), [Main/Wrist gallery](../../outputs/task10i/stage3/run/media/main_wrist_gallery.png)
- [Action/GT timeline](../../outputs/task10i/stage3/final_media/expert_full_task_timeline.png), [recovery timeline](../../outputs/task10i/stage3/final_media/corrective_recovery_timeline.png), [collection outcomes](../../outputs/task10i/stage3/run/media/collection_outcomes.png)
- [R/D initial/final distribution](../../outputs/task10i/stage3/run/media/R_D_initial_final_distribution.png), [replacement coverage](../../outputs/task10i/stage3/final_media/replacement_coverage.png)
- [Verification](../../outputs/task10i/stage3/run/verification.json), [artifact index](../../outputs/task10i/stage3/run/artifact_index.json), [completion audit](../../outputs/task10i/stage3/run/completion_audit.json)

## Engineering finalization history

Collection UUID remains `a66ec3d698e249b588f392e76038b54f`. Finalization UUID is `79b2d995c5694e479ad6667e562226c0`. Original collection root `/workspace/task10i_stage3_20261031_v3/run` and its failed reporting source/state are preserved. All raw episodes, GT, successful branches, failure traces, parent mappings and dataset references are reused without simulator rollout or collection retry. The correction separates disturbance and recovery orientation-error CSV fields. Python-Markdown is vendored under the isolated finalizer source; the existing simulator/training environment was not modified. Collector and finalizer source identities are independently pinned.

## 전달 시 독립 집계 — 원래90 Parent의 실제 역할별 결과

| 원래 역할 | Parent | Full-task 성공 | 전체 요구 Branch PASS |
|---|---:|---:|---:|
| Common44 | 44 | 42 | 42 |
| R-exclusive | 18 | 17 | 17 |
| D-exclusive | 18 | 18 | 18 |
| Shared tail | 1 | 1 | 1 |
| Validation | 9 | 9 | 9 |

이 집계는 동결 Stage2 Selection Hash와 실제 `expert_attempts.csv`에서 독립 계산했다. Full-task 성공과 모든 필수 Corrective Branch 승인 여부를 구분한다. 원래 원격 보고서는 출력 root에 그대로 보존했다.

| Dataset | Full-task frame | Prefix | Disturbance | Recovery | BC | Geometry eligible |
|---|---:|---:|---:|---:|---:|---:|
| R | 9659 | 4766 | 1033 | 1743 | 16168 | 17201 |
| D | 9677 | 4762 | 1033 | 1743 | 16182 | 17215 |
| validation | 1194 | 709 | 412 | 722 | 2625 | 3037 |

실제 Frame 수를 맞추기 위한 절단이나 재선정은 수행하지 않았다. Geometry eligible은 정합된 원자료의 이용 가능성을 뜻하며, 이번 단계에서 Geometry Loss를 계산한 결과가 아니다.

| 평가 모집단 | Dataset / 선정 상태 | Mean nearest distance | p90 | p95 | Maximum |
|---|---|---:|---:|---:|---:|
| training | R_initial | 0.266372 | 0.446726 | 0.476141 | 0.602347 |
| training | R_expert_success_conditional | 0.272185 | 0.453462 | 0.481985 | 0.602347 |
| training | R_final | 0.266402 | 0.446726 | 0.476141 | 0.602347 |
| training | D_initial | 0.249332 | 0.411577 | 0.440338 | 0.500637 |
| training | D_expert_success_conditional | 0.253428 | 0.414378 | 0.443881 | 0.558957 |
| training | D_final | 0.250553 | 0.414378 | 0.443881 | 0.558957 |
| holdout | R_initial | 0.361772 | 0.453996 | 0.473598 | 0.508257 |
| holdout | R_expert_success_conditional | 0.361772 | 0.453996 | 0.473598 | 0.508257 |
| holdout | R_final | 0.361772 | 0.453996 | 0.473598 | 0.508257 |
| holdout | D_initial | 0.367467 | 0.457905 | 0.475047 | 0.567599 |
| holdout | D_expert_success_conditional | 0.367735 | 0.457905 | 0.475047 | 0.567599 |
| holdout | D_final | 0.367735 | 0.457905 | 0.475047 | 0.567599 |

위 수치는 동결 거리·Scale을 그대로 사용한다. 최초 선정, Full-task 성공 조건부 선정, Reserve 적용 이후 최종 선정을 모두 보존했다. Selection/Weight/Greedy는 다시 계산하지 않았다.

실패 원본의 실제 Trace에서 확인한 내용:

- `10i_s1_0813`: `EXPERT_INSERTION_FAILURE`; 마지막 Phase `insert`, 기록된 Action 250개, truncated=True, official success=False. Phase별 기록 `{'reach': 14, 'descend': 13, 'close': 8, 'lift': 12, 'align': 15, 'insert': 188}`. 실패 원본은 보존하고 최종 학습 Manifest에서 제외했다.
- `10i_s1_0690`: `EXPERT_INSERTION_FAILURE`; 마지막 Phase `insert`, 기록된 Action 250개, truncated=True, official success=False. Phase별 기록 `{'reach': 20, 'descend': 13, 'close': 8, 'lift': 11, 'align': 23, 'insert': 175}`. 실패 원본은 보존하고 최종 학습 Manifest에서 제외했다.
- `10i_s1_0393`: `EXPERT_INSERTION_FAILURE`; 마지막 Phase `insert`, 기록된 Action 250개, truncated=True, official success=False. Phase별 기록 `{'reach': 19, 'descend': 13, 'close': 8, 'lift': 12, 'align': 36, 'insert': 162}`. 실패 원본은 보존하고 최종 학습 Manifest에서 제외했다.

[별도 구현의 독립 인계 감사](../../outputs/task10i/stage3/run/../independent_handoff_audit.json)는 원래 Template·Common44·Split·모든 원자료와 GT38 재조립·Controller Trace·실제 교란 변위·안정화·Frame 수 및 SHA를 다시 확인했다. 이 감사는 Collector/Assembler 구현을 import하지 않는다.

[원래 Stage1 환경 생성 코드의 독립 초기 재생](../../outputs/task10i/stage3/run/../native_initial_replay_audit.json)은 최종 신규 Parent 전부에서 Main/Wrist RGB·Depth·Calibration을 저장된 Stage3 T0와 exact 비교했다. Stage1의 역사적 NPZ에는 Main만 저장되어 있으므로, Wrist는 역사적 파일 비교가 아니라 동일 Native Runtime의 독립 재생 검증이다. 이 검사에서는 Action·Perception·Graph·Policy를 실행하지 않았다.
