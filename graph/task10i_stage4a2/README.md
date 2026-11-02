> **PAUSED_SCOPE_REVIEW**: User requested a scope review. All Task10I Stage4A2
> model/coordinator/waiter processes are stopped and operational launchers are
> blocked. Preserve the frozen implementation and completed evidence. The current
> candidate is a partial improvement that fails mandatory Development safety gates.
> Internal candidate38/60 complete;0858 full_task21-frame prefix is PARTIAL and
> requires t0 replay if explicitly reauthorized. No new inference/benchmark/raw
> rerun/Oracle/recache/training is authorized. Earlier status paragraphs below are
> historical. See [pause review](../../docs/visual-graph-integration/10i_stage4a2_paused_scope_review.md).

# Task10-I Stage4A-2

Independent research implementation. No production modules, original datasets,
checkpoints, previous Stage4A decisions, publishing or policy training are changed.

The independent workflow now includes baseline/raw-tensor equivalence, actual
three-model traces, proposal/prompt attribution, DINOv2 feature/reference/hard
negative controls, SAM prompt/memory controls, lossless preprocessing, causal
correction ablations, registered Development metrics, a one-candidate freeze,
untouched Internal30 and synchronous new Normal32/Stress12 evaluation, live
latency, paired current Graph checks, decision generation and failure galleries.
Implementation availability is not evidence that the whole study has completed.
Read the atomic remote receipts and the full report; old stage-local
`whole_goal_complete=false` fields deliberately remain unchanged.

Current remote roots on verified instance58576054:

- `/workspace/task10i_stage4a2_20261101_v1`: baseline trace, finished.
- `/workspace/task10i_stage4a2_dinov2_20261101_v1`: preserved first diagnostic;
  pair labels incorrectly used semantic slots, so do not use its pair metrics.
- `/workspace/task10i_stage4a2_dinov2_20261101_v2`: physical-actor-labelled
  diagnostic, separately frozen source/output; inspect receipt before reuse.
- `/workspace/task10i_stage4a2_models/dinov2-small`: exact requested model,
  pinned public revision and per-file SHA256; original checkpoints untouched.

`run_pipeline` validates source and artifact hashes before skipping a completed
stage. A source change needs a new root. Failed temporal inference must restart
from the episode prefix; there is no serialized-memory resume implementation.
The original fixture verifier intentionally creates its own ExperimentLogger
UUID nested under the coordinator; neither UUID is overwritten.

`diagnostic_runner` waits for the actual baseline PASS before running crop
contracts and DINOv2 in a separate process. Its final state only denotes a
historical diagnostic. No Internal/Final access follows automatically.

`dinov2_matching_audit` uses one source per historical Development parent,
selected deterministically before mask inspection, and t0/midpoint/terminal.
The legacy full-image448 masked-patch feature and bbox/padded CLS crops are
compared without training. Oracle-mask branches are explicitly diagnostic.
Pairs use the same physical parent and GT-supported actor identity; uncertain
mixed/background crops are excluded from labelled ROC pairs and remain in
provenance. Cross-parent same-category objects are not negatives. This limited
pair set does not yet cover all requested hard negatives, corrective temporal
phases, reference updates or causal re-identification. No threshold is selected.

Local checks:

```bash
.venv/bin/python -m unittest graph.task10i_stage4a2.test_contracts -v
```

The latest pre-independent miniature exercised26 real captured RGB-D frames per
condition, strict original-baseline memory isolation, separate GT conversion,
all five live/posthoc Graph tensors, registered statistics, actual Re-ID output
labelling and complete runtime readback. Real branch-prefix controls cover188
correlated branch pairs/3,521 common input-prefix frames and18 Graph controls.

`frozen_pipeline_runner` waits for completed Development and engineering, freezes
one `BC_WB_HANDOFF` challenger, then invokes `independent_pipeline` without tuning.
The live source, evaluator source and original registered collector source are
separate immutable roots. `reporting_runner` waits for those completed results
and runs `final_decision` and the gain/damage/stress/control gallery. None of
these entrypoints starts Stage4B, training, publishing or production replacement.

Remaining: full raw-proposal attribution, prompt/selection controls, practical
reference contamination and hard negatives, causal state/correction wrappers,
SAM2 memory/oracle controls, exact synchronous new benchmark capture and collision
audit, development ablations, numerical adoption preregistration, Internal30,
candidate freeze, untouched independent32+12 benchmark, downstream paired Graph
and safety/latency analysis, galleries and final evidence/readback/handoff.

## Additional controls and active Development work

`prompt_audit` completed336 fresh controlled forwards on24 historical parents.
With all old consumer rules retained, Wrist `red and white peg` localized14/24
with0 wrong accepted boxes versus10/24 and1 for the old phrase. Main deteriorated
to7/24 with11 wrong accepted boxes, so the A challenger changes Wrist only.
This box-localization finding is not a mask/Graph or independent-test result.

`sam2_controls` compares original masks, correct/wrong oracle initialization,
official correction of existing memory and separate fresh-object memory. v1
exhausted24GB with simultaneous conditions; preserved v2 executes each causal
prefix sequentially without changing model math or pruning memory. Its timing
is diagnostic, not observer-free latency. Original B0 masks must equal caches.

`gt_reference` passed actual six-frame/two-camera synchronous readback engineering
on CPU PhysX/CUDA SAPIEN. `benchmark_protocol` preregistered2048 collision-free
Normal/Stress attempt seeds, while explicitly leaving power review and stress
geometry pending. No Final RGB/mask/performance has been opened.

`development_inference` produces new Development36 B0/A outputs without importing
GT evaluators. `mask_forward_runner` waits for an actual live SAM-control process
and its completion receipt, then runs contracts, a3-frame engineering barrier and
all registered Development sequences. OOM has one bounded fresh-process retry;
partial episode outputs are preserved and the prefix is reprocessed from t0.
`development_reference` separately replays current Development raw actions and
collection snapshot/paired discontinuities, retaining full exact comparisons
and unsupported flags. It is not an inference dependency.

`reanchor_state_machine` is experimental support hysteresis plus an explicit
candidate transaction; it has not yet been integrated with model corrections or
validated as an improved pipeline. New local contracts plus old Stage4A contracts
total20 passing tests. The11 new tests are also callable separately.

Current additional remote roots:

- `/workspace/task10i_stage4a2_controls_20261101_v1`: completed prompt control.
- `/workspace/task10i_stage4a2_sam_controls_20261101_v1`: preserved OOM failure.
- `/workspace/task10i_stage4a2_sam_controls_20261101_v2`: sequential oracle controls.
- `/workspace/task10i_stage4a2_benchmark_protocol_20261101_v1`: frozen seed queues.
- `/workspace/task10i_stage4a2_gt_smoke_20261101_v1`: completed synchronous GT smoke.
- `/workspace/task10i_stage4a2_newdev_A_20261101_v1`: autonomous B0/A mask runner.
- `/workspace/task10i_stage4a2_newdev_reference_20261101_v1`: separate GT replay.

Inspect live processes and terminal receipts before interpreting these roots.
Whole-goal completion and candidate adoption remain unproven.

## Completed SAM controls and active paired evaluation

SAM oracle controls v2 completed3 windows/3980 object-camera-condition rows.
Episode0121 Main Box was already contaminated at t0 (TP720/FP1215; selected box
IoU.329), so its t95 failure is not evidence of pristine initialization followed
by a pure tracking failure. Correct GT-box initialization held IoU.888 at t95.

The installed predictor defaults corrected tracked frames to non-conditioning
outputs. Immediately propagating that same frame overwrites the correction with
a prompt-free inference. `sam2_correction` instead returns the immediate prompt
mask and preflights memory once, with a separately forked mutable state.
`api_controls` completed1596 rows on2 windows, proving original-state full tensor
byte isolation and rejected-transaction baseline equality. Correct API order
alone still did not fix0121's contaminated memory (t95–106 mean IoU.162 versus
B0.133); fresh memory reached.876. All interventions used oracle timing/boxes,
not a practical detection trigger. Source and outputs are preserved separately.

`sam2_gallery` generated/decoded full-prefix MP4s and frame strips for0924/0121/
2000003;0121 t94/t95 was visually inspected. It is a mask-control gallery; final
DINOv2-rank/Graph panels remain pending.

New Development reference replay completed36 parents/88 episodes/5845 frames:
5234 EXACT_GT,611 unsupported. New B0/A mask forwards remain active. The first
OOM invoked the registered one-retry supervisor. An evaluator v1 stopped on the
transient child failure; its completed10 full episodes passed all SHA checks.
`resume_evaluation` verifies frozen source identities and evaluator AST
compatibility before reusing those complete episodes. Only engineering-prefix
reference counting changed; no model/Graph computation changed. It distinguishes
a live bounded retry from terminal failure. The corrected engineering proof
counts3/3 frames rather than the entire underlying reference's79 frames.

`evaluation_metrics`, `paired_evaluation` and `aggregate_evaluation` preserve
undefined empty precision, correct-actor-only coverage, end/rim support, separate
raw versus depth visibility, physical-parent macro/paired bootstrap and recorded
phase labels read only after inference. Same masks must yield identical5 Graph
tensors. There are23 combined passing local contracts (14 new,9 previous).

Current authoritative additional roots:

- `/workspace/task10i_stage4a2_api_controls_20261101_v1`: completed API/rollback controls.
- `/workspace/task10i_stage4a2_sam_gallery_20261101_v1`: completed mask gallery.
- `/workspace/task10i_stage4a2_graph_engineering_20261101_v2`: corrected3-frame proof.
- `/workspace/task10i_stage4a2_paired_eval_20261101_v1`: preserved transient-failure run.
- `/workspace/task10i_stage4a2_paired_eval_20261101_v2`: active verified Graph resume.
- `/workspace/task10i_stage4a2_newdev_summary_20261101_v1`: preserved stopped dependency waiter.
- `/workspace/task10i_stage4a2_newdev_summary_20261101_v2`: active autonomous summary waiter.

At that progress checkpoint, still required: practical DINO/DINOv2/correction integration and hard-negative/
reference-contamination/crop controls; preprocessing controls; numeric adoption
preregistration before Internal/Final; new blind synchronized benchmark; final
paired safety/latency; complete galleries,20 answers,artifact/readback/completion
audit and Stage4B receipt. No scientific adoption or whole-goal completion yet.

## Latest completed evidence and queued controls

New Development36 B0/A completed88 episodes/5845 frames, with5234 exact-reference
frames. A changes only Wrist Peg phrase. Wrist recall improved14.95 percentage
points; Peg correct-source coverage improved1.71 points, but end coverage lost
1.45 points (parent paired95% interval excludes zero). A is not adopted.
`development_diagnostics` verifies unchanged Main masks and examines actual
selected source pixels;751/768 exact changed frames with end loss also removed
old Main source pixels. It reports FP events separately from paired-defined
precision so empty baseline predictions do not conceal new false predictions.

`integration_ablation` completed a live21-frame engineering barrier and three
unmodified historical windows. Engineering exercised5 actual DINOv2 forwards
and3 actual past-reference comparisons. Historical BC executed26 forwards but
zero past-reference comparisons; C and BC results were identical. Do not call
these26 forwards successful Re-ID. C improved0121 Box masks but harmed2000003
Box Graphs through uncertainty suppression, without accepting a reanchor.
All11 lost previously-correct Box frames failed only the temporal-center gate.
Suppressing ghosts during true Box occlusion was beneficial in the same sequence.
Neither blanket suppression nor blanket retention is a verified solution.

`adoption_criteria.json` is numerically frozen before Internal/Final access;
`register_criteria` records Development headroom and sample-power limits.
Primary correct-mask availability has33.65% baseline error headroom. Formal
validation requires primary improvement, downstream coverage benefit, surface/
contamination/false-positive safety and bounded actual latency. The registration
does not select or freeze a candidate. Internal30 and new Final remain unopened.

`stress_capture` implements actual collision-free visual slabs positioned from
camera calibration, fixed steps8–19, then current physical-step reappearance.
`final_capture` reuses Stage1 initial visibility/physical/replay gates and writes
RGB-D/proprioception HDF5 separately from synchronous GT sidecars. It refuses
Final reset without verified candidate/config/criteria/design freezes. Both have
queued simulator engineering; Final collection has not run. Capture engineering
v1 was stopped before execution to add a required `raw_hole_radius` sidecar key;
v2 uses a separate source/output root. This did not change any captured sequence.

`keep_control_runner` queues a separate uncertainty-retention control after
capture engineering. It preserves the current frame's SAM result pending an
accepted replacement, while keeping the uncertainty state. It is a diagnostic
control, not an adopted solution. Default `suppress` behavior remains available
and original frozen C output/source remains intact.

Authoritative additional roots on58576054:

- `/workspace/task10i_stage4a2_causal_ablation_20261101_v1/run`: completed engineering/historical.
- `/workspace/task10i_stage4a2_newdev_diagnostics_20261101_v1/run`: completed safety/endpoints.
- `/workspace/task10i_stage4a2_criteria_20261101_v1/run`: registered numerical adoption rules.
- `/workspace/task10i_stage4a2_component_controls_20261101_v1/run`: JPEG/matched-crop controls.
- `/workspace/task10i_stage4a2_stress_engineering_20261101_v1/run`: queued simulator stress proof.
- `/workspace/task10i_stage4a2_capture_engineering_20261101_v2/run`: queued collector proof.
- `/workspace/task10i_stage4a2_keep_control_20261101_v1/run`: queued separate retention control.

Current local checks:28 passing contracts (19 new,9 original Stage4A), all new
modules compile. New Development practical correction evaluation, final candidate
selection/freeze, Internal30, untouched independent32+12 capture/evaluation,
observer-free latency, full galleries and final audit/handoff remain outstanding.
Full progress report: `docs/visual-graph-integration/10i_stage4a2_dino_dinov2_sam2_integration_optimization.md`.

Latest receipt update: component controls,78-frame stress capture and104-frame
collector engineering have completed. `capture_gallery` renders the actual
non-Final pre/occlusion/reappearance RGB and aligned GT boundaries; Wrist slab
partially occludes Box and another camera, so intended camera is not equivalent
to all-actor full occlusion. Final seeds remain unused. Matched-patch crop audit
completed4889 embeddings/14887 instance pairs/4416 category pairs. Raw rectangular
DINO crops have zero positives under the existing80%-of-entire-crop actor purity
labelling rule; that branch is inconclusive and needs a separate label audit
that distinguishes crop background from object identity. Do not interpret it as
an embedding failure. JPEG removal helps some normal trackers but harms the
already-misinitialized0121 Box; full new Development mask/Graph evaluation is
required before selecting any preprocessing challenger. Keep-control engineering
has passed and its unmodified historical control is running in its frozen root.

## Full Development expansion and reference controls

`development_lossless` and `lossless_pipeline` run named challenger P on all
36 new Development parents/88 sequences, followed automatically by the original
paired Graph builder and exact-only summaries. The same-image JPEG condition
already reproduced baseline masks; P's real three-frame Graph engineering passed.
P model inference completed5845 frames. Graph evaluation is active. Source/config,
checkpoint/processor/package/asset hashes and the unchanged adoption criteria are
frozen per root. OOM has one prefix-restart retry; completed stage reuse checks
full artifact SHA. This is not observer-free latency.

`development_correction`, `model_trace` and `correction_pipeline` run C_KEEP and
BC_KEEP sequentially with fresh per-episode/camera/object memory. Engineering and
both same-builder three-frame Graph controls passed; full inference is active.
The evaluator now accepts explicit challenger names; original B0/A folders are
not relabelled. The new evaluator additionally supports locked source-checked
resume and validates inference/reference receipt hashes before episode reuse.
Existing frozen P code is not edited during its run.

`reference_pair_audit` relabels rectangular context crops using actor localization
and nonbackground dominance, retaining the original strict-purity audit. It adds
strictly past-reference pairs and separates simultaneous cross-camera diagnostics.
It uses7934 stored-vector pairs and zero new forwards. Hypothetical B0-t0 banks
are explicit, because B0 never used DINOv2. `paired_reference_summary` compares
ideal and practical references on the same candidates, rather than attributing
unpaired denominator differences to contamination. These are diagnostic raw-mask
vectors, not the actual runtime's clean-mask/candidate gates.

`background_matching_audit` completed fixed4x4-tile extraction after P released
the GPU:4608 embeddings/2410 actual forwards/8930 pairs,5476 background pairs.
GT labels were first opened after inference output was saved. An LRU hit now
refreshes cache order in this separately frozen diagnostic source; no baseline
model checkpoint or pipeline module is changed.

`pixel_alignment_audit` compared native sensor Position to both integer and
half-pixel backprojection in3 non-Final engineering frames. The half-pixel rule
matched the installed signed-int16-mm OpenGL Position texture. All actual frozen
camera, sampling and pointcloud paths already use(u+.5,v+.5), so this rejected
an alignment-bug hypothesis and caused no geometry/calibration changes. Initial
concern about integer rays was corrected after checking all actual source paths.

Current additional roots:

- `/workspace/task10i_stage4a2_newdev_P_20261101_v1/run`
- `/workspace/task10i_stage4a2_newdev_C_KEEP_20261101_v1/run`
- `/workspace/task10i_stage4a2_reference_pairs_20261101_v1/run`
- `/workspace/task10i_stage4a2_paired_refs_20261101_v1/run`
- `/workspace/task10i_stage4a2_background_pairs_20261101_v1/run`
- `/workspace/task10i_stage4a2_pixel_alignment_20261101_v1/run`

Local contracts:32 passed (23 new,9 original). Internal30 and fresh Final remain
unopened; candidate selection/freeze, beneficial-track integration, independent
validation, fair latency and final handoff/completion remain outstanding. DINO's
original funnel still labels localization, not proven semantic identity; finish
semantic/mixed-prompt attribution before claiming all failures are classified.


## Latest registered-gate and operational checks

New Development P completed88/5845 frames,5234 exact and611 unsupported.
The frozen10,000-parent-bootstrap primary improved+.10628 and downstream
coverage+.02738, but healthy-mask damage and contamination uncertainty failed
registered safety gates. Do not adopt P from its average gain.

Full C_KEEP/BC_KEEP mask inference also completed; all5845 masks are byte equal.
Actual BC forwards636, actual past-reference comparisons1, accepted corrections44.
The sole compared candidate was Box requested as Peg and was rejected by negative
margin and independently lacked temporal confirmation.247 exact-supported stored
references had0 observed contamination;16 references remain GT-unsupported.
No incremental accuracy benefit from DINOv2 was measured on this Development set.

`registered_evaluation` applies frozen numeric criteria, retains missing-support
constraints, separates damage from undefined empty precision and uses10000
physical-parent bootstrap draws. `correction_diagnostics` labels saved crops and
reanchor outcomes only after inference. `temporal_mask_diagnostics` preserves real
frame clocks, interrupted references, ghost/empty runs and posthoc recovery delay.

`frozen_inference` runs one live condition per process, records synchronized
module/end-to-end timing and current Graph fingerprints with no GT reads.
`live_engineering_runner` passed199 mask frames/condition for B0/P/C/BC;
B0/C/BC also matched all five original Graph tensors on every frame. A separate actual-Re-ID control passed111 frames:
lazy comparison embeddings cut17 forwards to3 while every mask, all five Graph
tensors and the actual positive/negative similarity values stayed exactly equal.
This is operational cost reduction, not new matching accuracy.

`final_path_engineering` passed actual collector HDF5 -> live B0/BC -> paired masks
-> synchronous sidecar conversion -> identical current Graph -> exact GT -> full
readback on26 frames/condition. `evaluation_inputs` refuses GT conversion before
prediction completion; Internal reference/manifest entrypoints require a frozen
candidate barrier. Final collector design32 normal/12 stress is separately
registered. No registered Final seed has been reset or inspected.

Full C/BC has a new failure: parent0250's t6 Wrist anchor tracks only part of Peg,
then stays support-confident;100 frames that B0 later handles correctly lose mask
availability. Primary gain+.06180 does not establish safety. A separately frozen
`BC_MAIN_BOX` Development study limits correction to Main Box; it retains all
three models and baseline Peg/Wrist masks, and does not repair or overwrite the
failed full-scope result. Main Box source/config froze before its run. This is
not yet a selected challenger or a CONDITIONAL_GO.

Local checks now52 pass. Entire-goal completion remains false. Still required:
scoped Development results and Graph/safety gates, one frozen challenger,
Internal30 once, untouched synchronous Final32+12, fair latency, final20-question
report, static HTML, verification/index/completion and handoff receipts. No
production replacement, full Stage4B recache or training follows automatically.


## Current state after the one-challenger freeze

Earlier progress paragraphs above are historical. Full53 local contract checks
pass. P, full C/BC and Main-Box-only Development gates failed and stay preserved.
`BC_WB_HANDOFF` finished36 parents/88 episodes/5845 frames: primary+.0578661
CI[.0274031,.0923996], downstream+.0140699 CI[.0051873,.0244537],29/35 gate
passes,3 failures (Box background/other-object upper limits and Peg-end lower
limit),3 unsupported negative constraints. It is frozen for independent testing,
not adopted. No criteria changed.

Freeze UUID3d30805423554b1e9df9649223117204 precedes Internal30 opening.
The autonomous independent root is
`/workspace/task10i_stage4a2_independent_20261101_v1/run` (30 parents,60 episodes,
4066 frames). It executes Internal once, then the registered new32 normal/12
stress benchmark with the same candidate and no retuning. The reporter is
`/workspace/task10i_stage4a2_reporting_20261101_v2/run`; named-artifact collection
is queued at `/workspace/task10i_stage4a2_finalization_20261101_v2/run`.
Finalization v1 was superseded before collecting data solely to distinguish the
downloaded HF default processor config from actual manual448 live preprocessing.
Frozen inference/evaluation/collector sources were not changed.

Additional evidence bundles9/10/11 passed local full SHA/readback. The0250
handoff gallery keeps94 exact and18 unsupported frames distinct. Offline ROC/PR
figures and HTML rendering are implemented; full final Markdown,20 answers,
independent decision, final static HTML, final local readback and completion audit
still require the independent results. Current task completion remains false.


Post-test supplements are queued independently at
`/workspace/task10i_stage4a2_supplemental_20261102_v2/run`. They wait until all
frozen live inference/latency work finishes, then check miniature and full new
Final detector recomputation (exact original postprocessing; explicitly not
original subthreshold-logit equality) and GT-t0 native SAM tracking Oracle.
CPU additions cover observed-surface geometry drift, native-grasp source clocks,
phase/occlusion feature strata, actual Re-ID recovery and missed trigger windows.
Existing Internal corrective traces are audited too. No original evaluation
outputs or adoption criteria change. Eight new metric contracts pass; three CPU
postprocessors passed actual26-frame engineering. Supplemental GPU execution
is still pending. Final reporting/completion must wait for both independent and
supplemental terminal receipts, not just code availability.


`completion_contract.json` inventories39 named artifacts plus static HTML,
20 questions,12 scenarios and8 controls; it is not a completion certificate.
`stratified_evaluation` adds explicit frame/episode/parent aggregation and
visibility/presence/t0/workflow/phase slices without changing adoption gates.
Two aggregation tests and actual26-frame engineering pass (63 local contracts
in total). `/workspace/task10i_stage4a2_strata_20261102_v1/run` waits for all
supplemental diagnostics before full Internal/Final descriptive slices.

`independent_evidence_readback` is a separate final integrity check. It requires
all26 independent,2 reporting,15 supplemental and3 stratification stage receipts,
then reads every referenced file and checks SHA256, expected frame/parent counts,
baseline-memory isolation, live/posthoc Graph equality receipts and unchanged
registered metrics. Three negative controls reject missing stages, conflicting
shared-file hashes and external-root substitution (66 local contracts total).
The actual full readback is still pending. A passing receipt certifies this
evidence inventory only; the full report, gallery QA and requirement-by-requirement
completion audit remain separate. No frozen source or running process changed.
