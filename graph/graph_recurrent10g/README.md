# Task10-G — F1-initialized recurrent A/B/C

Isolated experiment package. Existing F1 files are read-only. `Policy` reuses
Plan A E0 and GRU, and restores the selected F1 encoder. A freezes it, B updates
with Method B BC, C adds the restored F1 state head for loss-only supervision.
All branches share byte-identical initial control weights and a saved BC schedule.

`data.prepare` verifies graph/action/Robot76/GT alignment against the approved
F1 release and raw T+1 observations, checks copied expert prefixes exactly, and
writes read-only host mmap arrays. Workers stream bounded minibatches to CUDA.
C adds the F1 absolute/rotation/shape and actual-displacement pair objective;
extra graph exposures are logged separately. Train-only median gradient
calibration freezes one lambda before primary updates. No GT enters the policy.

The tmux-owned orchestrator runs every stage through report and artifact
verification in subprocesses. It checks output content and hashes as well as exit
codes. `flock` rejects duplicate orchestrators. Worker checkpoints retain the
optimizer, scheduler, RNG, detached episode hidden states, cursor and frozen
identities. A failed partial update is never saved over a valid checkpoint.
OOM recovery only lowers concurrency; batch size/objective stay fixed. Stage
failure records BLOCKED and returns nonzero; invariant violations are not retried.

Run CPU negative controls with the repository environment:

```bash
.venv/bin/python -m unittest graph.graph_recurrent10g.tests -v
```

On designated instance57987772, the complete driver and engineering fixtures
are preserved in `ops/`. The executed copies live at the independent task base,
outside the immutable research source. The current driver uses
`/workspace/task10g_f1_abc_20261031`, with source in its `source/` directory:

```bash
cp graph/graph_recurrent10g/ops/{autonomous_driver,miniature_restart_test,full_horizon_smoke}.py /workspace/task10g_f1_abc_20261031/
# Run inside tmux, with the inherited CUDA/offline runtime environment.
python -u /workspace/task10g_f1_abc_20261031/autonomous_driver.py
```

The driver runs the same miniature orchestrator, intentional stop/restart and
running-stage interruption fixture, full artifact readback, and a250-action
Development-parent rollout before creating/running Full. Its fixture/launch
manifests record the helper SHA values. Full also checks
`miniature_restart_verification.json`, `full_horizon_smoke_verification.json`,
and exact research-source hash equality with the tested miniature. Low policy
success is never a gate. Direct `orchestrator --root` invocation remains the
resume entrypoint for a prepared, frozen root.

A full launch requires all miniature stages, actual runtime rollouts, CUDA
save/resume and real single/triple worker benchmarks to pass. The parent process
owns all workers. Training subprocesses exit before sequential vision evaluation.
The main Final uses new98200–98211 parents after Development selection freeze;
engineering fixtures98180/98190 are separate. Full Final always runs36 episodes,
including when all policies fail. Optional recovery is explicitly NOT_TESTED.

Geometry analysis reuses previously viewed F1 Final pairs, with equal-budget
train-only probes for A/B/C and separately labeled inherited/joint heads. Actual
geometry-to-action intervention analysis reuses the verified Task10D code.
Markdown/HTML/figures/CSV/videos and atomic manifests remain in the independent
root; the full Markdown report is also copied into docs/visual-graph-integration.
No external publishing, callback, extra experiment or instance creation occurs.
Scratch is not a backup; tmux survives SSH exit but not instance destruction.

`ops/collect_results.py` is a separate local, read-only SSH collector. It waits
for the exact existing orchestrator process to terminate, copies Task10-G outputs
without deleting remote/local evidence, checks the remote frozen-source and BC
exposure manifests, then independently verifies local checkpoints, selection,
paired rollouts, functional metrics, videos and SHA indices. It does not start or
restart any research process. The shared `host_data` mmap and original F1 dataset
are excluded from this result delivery. Run it in a local tmux session:

```bash
.venv/bin/python -m graph.graph_recurrent10g.ops.collect_results \
  --output outputs/task10g/final_results \
  --docs docs/visual-graph-integration
```

The collector writes delivery/audit receipts under `outputs/task10g/` and copies
the complete report, with quantitative CSV summaries and resolved artifact links,
into the requested documentation directory. An SSH observation failure only
updates the local delivery status; it never restarts the remote pipeline. No
agent callback or external publishing is used. Its real end-to-end transfer and
independent readback were tested against the completed mini_v5 fixture.
The audit separately records the total video frame count and the number of
first/last endpoint frames actually decoded; it does not claim full-video decoding.
