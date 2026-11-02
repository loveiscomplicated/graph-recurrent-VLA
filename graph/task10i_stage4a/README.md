# Task10-I Stage4A perception reliability investigation

Task-owned research instrumentation. Production detection, tracking, geometry,
checkpoints, dataset split and main-visible eligibility code are unchanged.
The investigation is complete with **NO_GO** for production adoption. All208
registered evaluation sources/13,150 observations, independent validation, Oracle
controls and actual artifact review finished. The full research report is
[10i_stage4a_final_report.md](../../docs/visual-graph-integration/10i_stage4a_final_report.md).

## Completed evaluation on instance58576054

The current frozen evaluation root is
`/workspace/task10i_stage4a_finalization_20261101_v5/run`, with its immutable source
copy at the sibling `source/`. Its tmux session is
`task10i-stage4a-finalization-v5`. `resume_finalization.py` recovers a Phase
metadata aggregation failure after the previous Internal Test completed; it
asserts unchanged inference hashes and reuses those completed model outputs.
It is a specific recovery entrypoint, not a general way to bypass Development.
Partial legacy traces without phase labels are reported as
`EXPERT_PHASE_UNAVAILABLE`; they are never shifted onto raw frames by guesswork.

Use `/workspace/task10i_stage4a_environment_20261101/activate.sh` for the isolated
cu130 environment, preserving the original dependency and Torch math settings.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` before the Python
process starts. The base cu132 environment is unchanged.

`posthoc_runner.py` produced matched Parent statistics,
an actual failure gallery, correct-source coverage and data/causal-Graph checks
under `/workspace/task10i_stage4a_posthoc_20261101_v3/run`. Separate completed audits
checked native grouped GT-box Oracle semantics, Oracle graph structure, camera
contracts, explicitly approximate references, raw-to-refined mask retention and
Development global Graph-missing Oracle cases. No candidate was retuned.
Inspect their live processes and output receipts before resuming them; a state
file alone is not evidence that a worker is live or terminal.

The local read-only handoff commands are:

```sh
.venv/bin/python -B -m graph.task10i_stage4a.collect_evidence outputs/task10i/stage4a/final_evidence_58576054
.venv/bin/python -B -m graph.task10i_stage4a.report_digest EVIDENCE NEW_DIGEST_OUTPUT
.venv/bin/python -B -m graph.task10i_stage4a.complete_handoff EVIDENCE REPORT ACTUAL_REVIEW_JSON NEW_HANDOFF_OUTPUT
```

The digest and completion gates require finished scientific receipts. The review
must inspect the actual gallery/plot images and report; generating a review JSON
alone does not satisfy that work. `NO_GO` is a valid scientific conclusion and
never authorizes production adoption, policy training or external publishing.

## Original workflow and preserved history

`run_pipeline.py` verifies all356 unique raw/GT sources and the three Stage3
handoff documents, preregisters24 development and24 separate internal-test
parents by a deterministic hash, and leaves every validation image unopened.
It pins dependency source bytes. `baseline.py` re-forwards one entire original
development episode and requires exact masks and all five graph tensors against
the preserved F1 cache before continuing. Its Torch flags follow the actual F1
cache entrypoint rather than a training helper with different math settings.

`instrumentation.py` observes the original DINO forward with a hook, storing
all query boxes/token logits, input token IDs, preprocessing shape, postprocessed
boxes/scores, actual thresholds and source-frame hash. Scores are uncalibrated.
`replay_reference.py` compares native action replay RGB, raw depth and all three
calibration fields for both cameras at every frame. Only full array equality
permits EXACT_GT; mismatching frames remain GT_UNSUPPORTED. `run_pipeline` also
contains a separate render-only pose-restoration feasibility probe. These paths
are diagnostic and cannot feed GT to student inference.

`stagewise.py` keeps actual mask/source-pixel support through depth, confirmed
robot, uncertain robot, overlap, wrist gate, shape-first sampling, identity,
deduplication, budget and graph. This is the source's actual order. It stores
mask pixel indices, filter ownership/depth, representative3D points and all
source associations, F5 validity, gate reasons and graph tensors. Pixel-source
support is not spatial surface coverage. `diagnose_cached.py` checks instrumented
forward equality against historical graph tensors. New sources without a cache
and terminal frames without historical graphs are explicitly pending.

Current execution roots on instance57987772:

- `/workspace/task10i_stage4a_20261031_v1`: failed seed-schema feasibility probe.
- `/workspace/task10i_stage4a_20261031_v2`: strict baseline mismatch preserved.
- `/workspace/task10i_stage4a_20261031_v3`: original runtime-math baseline retry,
  then reference probes and preliminary stagewise analysis if its gate passes.

Runner commands use `/venv/main/bin/python -B -u -m
graph.task10i_stage4a.run_pipeline NEW_ROOT` inside `task10i-stage4a` tmux.
Never restart from a lock/state file alone: inspect its live PID and tmux handle.
A changed source gets a new source/output root. Existing roots are preserved.
A PENDING_IMPLEMENTATION state is not completion and does not select a candidate.

Local checks:

```sh
.venv/bin/python -m unittest graph.task10i_stage4a.test_contracts -v
```

The initial pending checklist (subsequently completed, with unsupported GT
explicitly retained) included dataset-wide stage traces, exact reference support,
failure galleries, macro/phase/temporal/spatial metrics, causal oracle controls,
evidence-based practical candidates, latency without observer I/O overhead,
freeze/internal test/full independent validation and the final handoff audit.

The broader autonomous audit now uses `development_runner.py`. It runs contracts,
a separate engineering3-frame smoke, all64 preregistered Development sequences'
reference replay, one full causal mask pass with a bounded3-worker geometry
pipeline, and macro aggregation. Terminal observations are included. Reference
replay and model inference run in different processes to preserve baseline Torch
math settings. DINO memoization is limited to identical RGB bytes and prompt;
it does not cache SAM2 memory or cross episode tracking states.

`repair_reference.py` reproduces original collection discontinuities in a new
reference root: saved anchor state restore for corrective branches and original
paired teleport/readback/restore for historical paired full-task sequences.
Previously unsupported frames become EXACT_GT only after full two-camera
RGB/depth/calibration equality. Existing masks and references are not overwritten.
`reevaluate_references.py` automatically waits for baseline and repaired receipts,
then measures saved masks/points against the corrected references with zero
DINO/SAM2/graph forwards. It asserts previously exact stage metrics are unchanged.

The original pre-migration broader roots were v5 (baseline audit), v6 (reference repair), and v7
(corrected reference evaluation queued after v5). Their stage processes and tmux
sessions must be inspected before resuming. No result here selects a practical
candidate or opens Independent Validation.

`forward_equivalence.py` is a separate new-host verification entrypoint. Run it
from an isolated copy of the frozen v5 Source using the restored cu130 venv.
It requires all672 frozen source hashes, selected raw/model/receipt hashes and
the original Torch settings to match, then invokes the unchanged `baseline.run`.
A post-build observer preserves actual Mask/Graph arrays and also compares with
the old Stage4A cache. It checks101 action frames of the original Development
fixture2000003; the original F1 cache has no terminal graph. It creates a new
ExperimentLogger UUID and never resumes the remaining dataset episodes.

```sh
source /workspace/task10i_stage4a_environment_20261101/activate.sh
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -B -u \
  -m graph.task10i_stage4a.forward_equivalence NEW_OUTPUT_ROOT \
  /workspace/task10i_stage4a_20261031_v5/run \
  /workspace/task10i_stage4a_20261031_v3/run
```

The resumed experiment on58576054 is stage-ordered by `finish_runner.py`, then
`candidate_development_runner.py` / `paired_candidate.py`, `finalize_runner.py`,
and `posthoc_runner.py`. These create fresh roots/ExperimentLogger identities.
The final coordinator waits for all registered Development challengers before
O0--O4, observer-free latency, source/config freeze, Internal Test, final freeze,
and the complete Shared Validation partition. Test/Validation cannot retune the
candidate. A challenger may be independently rejected with NO_GO. A completed
scientific run still requires final report, source/data checks and evidence review.

Current candidates isolate Wrist Peg reanchoring with shared-depth refinement,
Main Peg dominant depth component selection, and extension of that same rule to
Main Box. They do not accept object GT, future frames, expert phase or hole pose.
Oracle conditions use those references only in explicitly separate diagnostics.
`perframe_cache.py` preserves native prepare bytecode and caches identical current
inputs only; it never stores a surface map. Its source/config/depth/pose/mask keys
and per-frame invalidation have negative controls and real Graph-byte tests.

`selection.py` keeps the existing numeric gates. Graph object presence recall,
all-node geometric surface coverage, source-pixel retention and correct-source
surface coverage are separate metrics. `report_analysis.py` also separates
matched-parent means, undefined precision, frame-weighted means and pooled pixel
counts. `valid_coverage.py` never gives false-positive nodes valid-surface credit.
`attribution.py` uses only raw-query evidence available by the actual decision
time and keeps intentional Wrist Box exclusion separate from recognition errors.

Final research reports belong in `docs/visual-graph-integration/`. Raw artifacts
remain in isolated experiment roots; publishing and policy training are separate
user-authorized workflows. Do not infer full completion from a queued runner or
a `REPORT_REVIEW_PENDING`/`PENDING_IMPLEMENTATION` state.

## Final handoff

The final report review used the common ExperimentLogger and produced
[completion_audit.json](../../outputs/task10i/stage4a/final_handoff_58576054_v1/completion_audit.json).
It binds the full report SHA, actual image review and finished scientific receipts.
[final_report_review.json](../../outputs/task10i/stage4a/completion_58576054/final_report_review.json)
records scope, numeric checks and the reviewed source/image hashes. EXACT_GT is
available for11,155 observations;1,995 remain unsupported in primary metrics.
Do not turn the separate890 approximate references into exact validation.
Wrist mask recovery did not establish a final Graph improvement on Validation.
No policy training, blind-test evaluation or external publishing was performed.
