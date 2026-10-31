# Task10-I Stage3 raw expert dataset collection

This task-owned workflow reuses the frozen Stage2 selection and reserve queues,
the F1 raw capture/GT38 encoder, unchanged ScriptedExpert, Task10-F bounded TCP
disturbance/replanning/SafetyV2, Task10-G prefix BC contract and the shared
ExperimentLogger. It never runs perception, graph recaching, policy training,
student evaluation or publishing. Existing raw/GT/checkpoints are read-only.

On instance57987772, prepare an isolated copy of the Stage1 source dependencies,
this package and `graph/task10i_stage2` (distance diagnostics only). Do not change
the installed runtime or another task's source. Launch the complete coordinator:

```sh
cd /workspace/task10i_stage3_NEW/source
tmux new-session -d -s task10i-stage3 \
  'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python -B -u -m graph.task10i_stage3.run_pipeline /workspace/task10i_stage3_NEW/run > /workspace/task10i_stage3_NEW/pipeline.log 2>&1'
```

The coordinator verifies every frozen input SHA and all105 preserved episode
contracts, runs unit tests, four distinct-role full-task smoke tests and the
largest18-entry corrective template, performs an actual interrupted simulator
episode/resume/verified-skip fixture and a separate miniature R/D/Validation
assembly. It benchmarks three workers first, then one and two; every sensor,
action and GT array must agree exactly across concurrency settings. These
engineering episodes are excluded from research manifests.

Only after those gates does it freeze the collection protocol, collect the
initial90 physical parents, apply frozen reserves in role order and assemble
R96/272, D96/272 and shared Validation16/48. Full-task approvals without a
planned logical entry remain diagnostic. Shared branches reference a single
raw source. Prefix BC requires exact nominal comparison, disturbance BC stays
off and recovery reads current physical state. Physics audits distinguish the
inherited no-contact corrective protocol from intended full-task grasp contact.
The original tolerance remains200µm; grasp forces are recorded without an
invented force cap. Hidden PhysX solver cache and continuous swept-volume
collision freedom are not claimed.

`receipt.json` seals an episode only after HDF5/GT readback and SHA verification,
then an atomic directory rename exposes it. Parent locks prevent concurrent
owners. Partial data are archived rather than approved or deleted. A committed
full task is never rerun to recover later branches; stored controller/RNG/clock
snapshots and its actual causal prefix are reused. An approved parent can add a
new dataset-specific branch while preserving its earlier result record. Task
failures are not retried. Runtime/alignment/reset failures stop the coordinator
for investigation; they are never silently treated as expert attrition.

Resume the identical command only after verifying the coordinator and its worker
handles are terminal. The launcher refuses to duplicate surviving workers. Source
hash changes require a separate execution root, with earlier evidence retained.
Completed episodes and assignment journals are revalidated and reused; successful
and failed branches remain preserved. Finalization independently reopens all
dataset sources, checks original hashes again, produces media/reports and emits
a Stage4 raw-data receipt. A failed gate yields INCOMPLETE plus a non-ready
receipt, never a fabricated completed dataset.

```sh
.venv/bin/python -B -m unittest graph.task10i_stage3.test_contracts -v
```

The full Markdown/HTML report is delivered to `docs/visual-graph-integration/`.
Raw scientific files remain under the execution output root with source/hash
references. Static report links are rewritten during local delivery.

## Executed collection and finalization

Collection UUID `a66ec3d698e249b588f392e76038b54f` wrote the immutable raw store
under `/workspace/task10i_stage3_20261031_v3/run`. Original90 parents yielded87
official successes; two common slots and one R-tail slot used the frozen reserves.
The first full raw alignment readback passed before a reporting-only CSV field
collision. That failed root and source are retained. Disturbance and recovery
orientation metrics now have distinct CSV keys.

Final artifacts are under
`/workspace/task10i_stage3_20261031_v5_finalization/run`, with finalization UUID
`79b2d995c5694e479ad6667e562226c0`. `finalize_existing` reuses the original data,
successful alignment proof and exact manifest references; it rehashes raw/GT/masks
and preserves separate collector/finalizer identities. No expert rollout is repeated.
An intermediate renderer preflight failure remains separately preserved in v4.
Python-Markdown3.11.2 is vendored in the isolated source, using fully qualified
extension imports. Nothing was installed into the existing training environment.
For a fresh full pipeline using that isolated source, include `PYTHONPATH=vendor`.

`native_initial_replay.py` uses the original Stage1 factory with zero actions and
verified93 initial states, including every final new parent. Main/Wrist RGB-D and
calibration match Stage3 T0 exactly; Stage1 historical NPZ stores Main only, so the
Wrist comparison is a native-factory replay, not a nonexistent historical image.
`audit_handoff.py` independently reconstructs GT38, checks all356 unique logical
sources/22,705 action frames, recomputes dataset frame totals and verifies the640
historical raw/GT sources plus all8,514 canonical pool files. It imports neither
collector nor assembler. The delivered report, HTML, media and receipts are under
`outputs/task10i/stage3/` and `docs/visual-graph-integration/`.
