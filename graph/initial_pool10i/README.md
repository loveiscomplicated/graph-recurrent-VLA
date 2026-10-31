# Task10-I Stage1 initial-state candidate pool

This task-owned package performs no simulator actions, model inference, expert
collection, diversity selection, or split assignment. `pipeline.py` reuses the
unchanged Task10-G visibility audit, native Task10-G environment wrapper, and
shared `ExperimentLogger` with W&B disabled.

The executed runner, frozen source, audit dependencies and artifacts live at
`/workspace/task10i_stage1_20261031/` on designated instance 57987772. The first
engineering failures are preserved in `run/` (BatchedRNG serialization) and
`run_v2/` (null projection in gallery); the corrected execution
uses `run_v3/`. Never resume a root with different source or runtime. Source hash
mismatches are intentional hard errors; use a new root after implementation fixes.

```sh
# On the original designated host, with frozen source/audit/code already present:
tmux new-session -d -s task10i-stage1 \
  '/venv/main/bin/python -u /workspace/task10i_stage1_20261031/code/pipeline.py run /workspace/task10i_stage1_20261031/run_v3 >> /workspace/task10i_stage1_20261031/pipeline_v3.log 2>&1'
```

The runner performs visibility regression and live renderer reset validation,
freezes independent seeds, tests real interrupted transactions/resume/replay and
report generation, collects until 300 eligible unique states or 1,200 attempts,
replays every eligible state in a separate process, generates diagnostics, and
verifies artifacts. Candidate `record.json` is the atomic commit; files without
that record are uncommitted and recomputed on resume. No automatic retry of
invariant failures occurs. Seed ordering is immutable and results do not change
proposal distribution. Each resume has a new logger UUID linked to the original.

Use `eligible_candidate_pool.json` for Stage2 input; every split remains
`UNASSIGNED`. `snapshot.pkl` is trusted local experiment data, not a complete
PhysX serialization. The tested reproduction route rebuilds shapes using the
original seed and compares raw state/controller/observation readback exactly.
Hidden solver caches and future trajectory bit identity are not guaranteed.

`independent.py` uses GL calibration and OBB slab rays, independently of the
primary convex-depth implementation. Pixel-perfect segmentation is diagnostic.
Physical checks use actual collision shapes, joint limits, native reset support,
and state/calibration validity; they do not test expert feasibility.

Local contract tests:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest graph.initial_pool10i.test_pipeline -v
```

Full experiment report belongs in
`docs/visual-graph-integration/10i_stage1_main_visible_candidate_pool.md`.

Final handoff: `outputs/task10i/stage1/verified_pool_v2/`. The original
`run_v3` remains BLOCKED on a separately serialized GL camera boundary pixel.
`calibration_diagnostic.py` independently verifies all 926 states using the
frozen primary CV calibration and preserves the GL difference. `reverify.py`
creates a separate logger UUID and verification root; it never changes any
candidate state, label, seed, physical verdict, or replay proof. Final Export
explicitly distinguishes raw RGB-D NPZ keys from annotated gallery previews.
The first local verifier was superseded to cache aggregate NPZ arrays instead
of repeatedly decompressing them; its status/code remain preserved.
