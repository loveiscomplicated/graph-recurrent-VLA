# Task10-G — Pipeline engineering log

Status: engineering validation and full A/B/C training completed; primary
Development evaluation and checkpoint selection completed; primary Final evaluation
is running. This is an execution history, not the final
research report. Final evaluation and overall completion remain unverified.

The requested experiment is F1-initialized A Frozen / B Joint BC / C Joint BC +
Geometry, followed by Development selection, new paired Final12 per condition,
geometry and action diagnostics, Markdown/HTML reports, and artifact readback.
The complete request is preserved in the user-supplied attachment. Full training
is gated on the same orchestrator completing a separate miniature pipeline.

## Existing assets verified

- Designated instance57987772 was running, with an NVIDIA RTX3090/24GiB.
- Initial device memory was1MiB, no CUDA jobs were listed; scratch free was5.8GiB.
- Existing F1 root: `/workspace/task10f1_gt_pretrain_20261030`.
- Actual selected checkpoint SHA256 matched
  `781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6`.
- Actual runtime: Torch2.14.1+cu130, NumPy2.5.4. F1 release includes272 Train,
  48 Validation, and previously viewed12 Final episodes. Main Final will use new
  seeds98200–98211; engineering fixtures98180/98190 are separate.
- Original F1 files, datasets, and reports have not been edited. Other Vast.ai
  instance57640710 has not been accessed or modified in this task. No new paid
  instance, Git commit, external publishing, callback, or agent invocation.

## Implementation

New opt-in package: `graph/graph_recurrent10g/`.

- Existing Plan A E0 GraphEncoder, parameter-free five-slot readout,
  Projection128/GRU144/Action7, Method B ActionCodec, robot normalizer and
  controller APIs are reused. C restores F1's head for loss-only geometry
  supervision; the head is absent from inference calls and action inputs.
- Data preparation verifies raw T+1 observation/action/GT/cache alignment,
  approved source SHA, parent splits, and prefix/disturbance/recovery masks.
  Copied expert prefixes are supervised only after exact comparison with their
  approved nominal expert action, robot and timestep source.
- A single saved sampling schedule and common train-only ActionCodec feed all
  conditions. Host mmap arrays share read-only data pages; only bounded graph
  minibatches enter each worker's independent CUDA context.
- Lambda uses train-only gradient medians at a preregistered25% ratio, finite
  clamp and explicit fallback. C uses F1 absolute/rotation/shape/displacement
  objectives. Extra pair graph exposures are recorded separately.
- The tmux orchestrator runs subprocess stages with atomic manifests, hashes,
  actual counts, exit codes, errors and restart evidence. Flock rejects duplicate
  orchestrators/workers. Checkpoints contain optimizer/scheduler/RNG/hidden/cursor
  and frozen config/dataset/schedule/normalizer identities.
- Single A/B/C and actual concurrent triple CUDA warm-ups precede concurrency
  selection. OOM recovery lowers concurrency with unchanged batch/objective and
  resumes only a fully saved checkpoint. Invariant failures are not retried.
- Development and Final use current causal RGB-D, robot observations, the
  preserved graph/GRU path and actual env.step. Paired initial simulator,
  controller, RNG and graph hashes are checked. Geometry and action analyses
  remain separate from success metrics and from GT-state references.
- Final report code writes the full report into docs/visual-graph-integration
  in the isolated execution source, plus artifact-root Markdown/HTML/figures.
  No automatic off-host backup is configured or claimed.

## Executed checks

Local repository environment:

| Check | Result |
|---|---|
| Python compilation of new package | PASS |
| Stage evidence negative controls |3 tests PASS |
| Existing ExperimentLogger/Artifact contracts |42 tests PASS |
| Existing F1 and Plan A semantic tests |15 tests PASS |

These local tests do not execute CUDA or the simulator. Actual CUDA/renderer/
vision/physics and full pipeline gates run on designated57987772. GPU PhysX has
not been executed. Shared simulator/logger/publisher code has not been changed.

## Engineering failures preserved

The first independent miniature root is
`/workspace/task10g_f1_abc_20261031/mini_v1`. Initial import failure exposed the
missing archived `legacy_fps` dependency in the transfer bundle; the existing
F1 copy was copied into the isolated source without modifying the original.

Data verification then rejected an overrestrictive newly added component-wise
Action7 ±1 assumption. Inspection showed F1 episode2000000 frame0 rotation
component1.096476. Its raw expert action exactly matched recorded controller
input and the stored arm command showed the installed rotation-norm clipping.
This is the inherited valid collection convention. Labels were not altered.
Verification now computes installed controller commands and checks them against
stored scaled-command traces; finite raw commands/gripper checks remain.

A fresh miniature root `mini_v2` was started in tmux session`task10g-mini-v2`.
Both failed attempt logs and source archives are preserved. Full training remains
blocked until all engineering gates pass; low policy success is not a gate.

## Execution locations

- Isolated source: `/workspace/task10g_f1_abc_20261031/source`.
- Current miniature: `/workspace/task10g_f1_abc_20261031/mini_v2`.
- Local engineering evidence: `outputs/task10g/`.
- Final automatic report name: `10g_f1_initialized_graph_recurrent_abc.md`.

This log will be updated with actual miniature/launch evidence. It is not proof
that pending CUDA, training, evaluation or report stages have completed.

## CUDA miniature evidence (mini_v3)

The verified host mmap preparation from mini_v2 is reused only after all mapped
array and raw/cache source hashes are rechecked. This avoids another geometry
cache build or a second large host-data copy. The same mechanism will permit
Full to share these immutable host pages while using its own sampling schedule.

On mini_v3, actual CUDA gates passed: initial control weights/predictions were
identical; B/C lambda-zero BC gradients were identical; C geometry gradients
reached the encoder and state head; the GT-read and auxiliary-inference denial
positive controls passed; whole-sequence/hidden-carry agreement passed; saved
optimizer/scheduler/RNG/hidden/cursor state reproduced the next actual update
exactly. A's frozen encoder remained unchanged.

Actual RGB-D→causal DINO/SAM2→C3/A-SF graph→GNN→GRU→ActionCodec→controller→CPU
PhysX rollouts ran for all three conditions. Paired initial simulator/controller/
RNG snapshots and first graph inputs matched exactly across conditions.

Single-worker and triple-worker20-update CUDA benchmarks passed. The measured
triple aggregate rate was17.1325 updates/s with at least21.9053GiB device memory
free. Three-worker concurrency was selected. These short engineering rates are
not a prediction of the complete training or vision evaluation wall time.

Each miniature condition completed4 optimizer updates. The orchestrator then
intentionally stopped at the completed training barrier. A concurrent second
orchestrator CLI was rejected with nonzero exit and `another orchestrator owns
this task`; no second set of training workers was started. A separate tmux-owned
engineering harness is testing interruption/restart of a running Development
stage and subsequent completion of every remaining stage.

## Final operational audit and autonomous driver

The actual F1 checkpoint also reported the requested training UUID
`a803354bff8f414492456af4c0b43bce`.

A250-action Development-parent engineering rollout completed successfully, with
250 video/trace steps. This is runtime validation, not a new Blind Final result.
The miniature v3 full report is locally preserved at
[the complete integration report](10g_miniature_v3_integration_report.md).

Final audit added preservation of uncommitted resume ledgers, recovery of a
missing scheduled checkpoint copy from a fully committed identical last.pt,
completed-fixture reconstruction of Action Correction CSVs, full cursor/step
checks, and a source/restart/full-horizon gate before primary training. Live
policy restoration no longer opens any dataset/GT mmap. Strict Reach is measured
with the inherited Task10D ordered pose/open-gripper/early-close/motion/collision
contract and stays auxiliary to Insertion. CUDA encoder+readout, complete policy,
and end-to-end inference times have distinct columns.

An intermediate mini_v4 engineering run was intentionally stopped before any
Full launch to include these audit items. Final candidate mini_v5 uses the same
immutable host data and a new isolated configuration/output root. It has passed
its actual CUDA, gradient, checkpoint-resume, triple-worker benchmark, runtime
rollout, training barrier, Development and separate Final fixture gates. Its
remaining analysis/report verification and250-action check are driven by Python.

The standalone `autonomous_driver.py` runs inside tmux session`task10g`. It owns
the miniature tests, intentional running-stage interruption/restart, full-horizon
smoke, and then the complete primary orchestrator. The Full root is
`/workspace/task10g_f1_abc_20261031/full`. Primary training is gated on the final
miniature's complete artifact verification, restart proof, full-horizon proof,
and unchanged source hashes. No normal stage requires another Codex invocation.
The complete primary result remains pending until actual full execution ends.

## Primary launch verified — 2026-10-30 18:55:59 UTC

Final candidate mini_v5 completed all19 stages, intentional running-stage
interruption/restart, artifact verification, and a separate250-action runtime
rollout. The live backend reported`physx_cpu`, `sapien_cuda`, CPU simulation,
20Hz control and100Hz physics. GPU PhysX and local macOS CUDA remain unexecuted.
The final miniature report is preserved in full at
[the integration report](10g_miniature_integration_report.md).

The Python driver then created and ran the independent Full pipeline without
another stage instruction. The Full preflight/source/GT/initialization/gradient/
resume/runtime/concurrency gates passed. Full and the tested miniature had
identical research-source hash dictionaries. All8 live DINO files and the SAM
checkpoint matched inherited F1 model hashes. The train-only8-batch median
calibration froze lambda0.03136492161957183; fallback was false.

The primary stage is`PARALLEL_TRAIN_A_B_C`, owned by orchestrator PID134988.
Worker parent coordinator PID135457 owns three independent CUDA processes:

| Condition | Worker PID | Observed update | Verified checkpoint update | Run UUID |
|---|---:|---:|---:|---|
| A Frozen |135467|1630|1500|fd1a73f1aeda4aab8d98615395cc4aac|
| B Joint BC |135468|1542|1500|c7102150d47640dfb83a7d476ac8818d|
| C Joint BC + Geometry |135469|1275|1250|40e6ccfd42c14c13923913dff1a8014c|

All three saved checkpoints passed SHA/readback checks. A's checkpoint encoder
hash remained identical to F1 initialization. B/C had positive encoder gradients;
C had positive state-head gradient and geometry loss. Different instantaneous
steps reflect worker throughput, not different frozen budgets or sampling
schedules. Each condition must finish11487 optimizer updates before the barrier
allows Development.

Frozen Development parents:30007,30012,30020,30023,30036,30042. Fixed candidate
steps:3000,6000,9000,11487. New primary Final parents:98200–98211, three policies,
36 paired full-task episodes, up to250 actions each. Model selection, Final,
geometry/action diagnostics and Markdown/HTML/figures/hash verification remain
owned by the autonomous Python pipeline. Their actual results are pending;
primary completion and hypothesis success are not claimed.

Tmux session`task10g` contains`pipeline`, `train_A`, `train_B`, `train_C`, and
`stages` windows. The extra windows only display logs. The main execution is
independent of the Codex terminal/SSH session. No continuous Codex watch or
callback is needed. The executing research source remains frozen; identical
operational helper copies are also preserved locally under
`graph/graph_recurrent10g/ops/` for reproduction.

Primary artifacts: `/workspace/task10g_f1_abc_20261031/full`.
Expected final full report in the isolated source:
`source/docs/visual-graph-integration/10g_f1_initialized_graph_recurrent_abc.md`.
The whole research goal remains active until actual Full results and every
required deliverable have been verified. Selected engineering/launch evidence
is copied locally; a complete off-host backup of all raw data/videos is not
configured or claimed.

## Independent collection configured after launch

The primary process tree was revalidated on the next goal continuation: the
same orchestrator PID134988 and CUDA worker PIDs135467/135468/135469 were live,
with the same coordinator parent. No research worker or source was restarted or
modified. This continuation is progress from adding independent delivery/audit,
not a claim that the full research run has finished.

A local read-only SSH collector now runs in tmux session`task10g-delivery`.
It waits for the exact existing remote orchestrator to become terminal and then
copies the Task10-G output tree, excluding only the shared host-data mmap link.
It never invokes an agent, restarts a research process, changes the protocol,
selects another checkpoint or publishes externally. SSH observation failures
remain delivery observations and do not terminate/restart the research job.

Its real transfer/readback path was executed on the completed mini_v5 root:
305 indexed files and18 Development/Final video frames passed local checks.
The remote frozen-source/original-F1/raw-graph SHA checks and an independent
recomputation of the actual common BC exposure also passed. Four negative-control
tests reject missing/duplicate episodes, invalid boolean values and escaped
artifact paths. Full collection will additionally check the11487-update ledgers,
Best/Last checkpoint hashes/resume state, Development selection tie-breaks,
paired initial states, trace-derived functional metrics, complete Final36 and
MP4 first/last-frame decoding. It will save the complete report with quantitative
CSV summaries and resolved artifact/video links in the requested docs directory.

Collector status: `outputs/task10g/collector_final_results_status.json`.
Future collection target: `outputs/task10g/final_results/`.
Future independent receipt: `outputs/task10g/independent_completion_audit.json`.
The local collection is not a backup of original F1 RGB-D/Graph/GT datasets or
host mmap files. Delivery success and complete research results remain unclaimed
until the receipts and the actual full artifacts are present and verified.


## Training barrier completed

At 2026-10-30T19:18:03.756553+00:00, all three primary worker result files independently reported COMPLETE with11487 updates. The orchestrator recorded both PARALLEL_TRAIN_A_B_C and CHECKPOINT_VERIFICATION as COMPLETED. All three final sampler cursors and schedule SHA values matched; A’s encoder stayed identical to initialization, while B/C changed. The pipeline had automatically advanced to DEVELOPMENT_CLOSED_LOOP. The full result remains pending until Development, frozen selection, Final36, analyses, reports and final artifact audits finish. Local evidence: `outputs/task10g/training_completion_verified.json`.

## Development completed and Final started

At 2026-10-30T23:54:21.226169+00:00, the same live orchestrator PID134988 had automatically advanced to `FINAL_CLOSED_LOOP`. All72 Development episode records reported COMPLETE. Development-only selection chose A step9000, B step9000 and C step11487. Each selected candidate and Best copy independently matched its frozen SHA. The selection manifest recorded Final access false at freeze. Local evidence is [the selection readback](../../outputs/task10g/selection_freeze_verified.json).

The new Final protocol remains36 episodes over seeds98200–98211. At this observation no Final episode had completed. Final results, geometry/action analyses, reports and final independent completion audit remain pending; no hypothesis conclusion is drawn from Development.

## Local audit field correction

The local collector now distinguishes `video_frame_count_total` from `decoded_endpoint_frames`. It verifies declared frame counts and decodes the first/last frames; it does not claim every video frame was decoded. A repeat readback of the305-file miniature fixture and all4 negative-control tests passed. Its9 videos contain18 total frames, with18 endpoint decode operations. The previous fixture receipt is retained; the new [fixture re-audit](../../outputs/task10g/collector_fixture_reaudit_v2.json) records the corrected field names. The quantitative summary also correctly states that the final BC minibatch is common across A/B/C.

Only the local read-only collector was intentionally restarted to load this correction (PID53949→35789, tmux`task10g-delivery`). Its new process confirmed the original remote orchestrator PID134988 live in `FINAL_CLOSED_LOOP`. Remote source, models, settings and workers were untouched. [Upgrade receipt](../../outputs/task10g/collector_upgrade_v2_receipt.json).
