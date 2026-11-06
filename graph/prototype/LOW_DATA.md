# Small Flat / Surface comparison

This Graph-team experiment asks whether Surface GNN predicts expert actions more
accurately than a similarly sized Flat MLP with a small, fixed training set.
It reuses v2 `CommonPolicy`, normalizers, optimizer/loss, training CLI and offline
evaluator. There is no simulator, new expert, architecture or rollout in this test.

## Run

From the repository root, using the existing Python 3.11 environment and extracted
`outputs/graph/dataset.pt` plus its source converted H5:

```bash
WANDB_MODE=disabled .venv/bin/python -m graph.prototype.low_data
```

Defaults: 10 complete training episodes, subset seed 0, initialization/shuffle
seeds 0/1/2, 10 full epochs, batch 32, AdamW 0.001, weight decay 0.00001, cosine
schedule, four CPU threads. Existing validation/test IDs and all their samples
are unchanged. Whole reset groups are retained; a source with repeated resets
can retain more than the requested episode count, which the plan reports.

Flat width 192 gives 451,015 parameters; Surface has 449,031 (0.44% difference).
Both receive the same 131 nodes' information, context and shared goal31 state,
and the same 159→128→7 action head. Selection uses minimum validation raw 7D
action MSE. The common input goal31 includes qvel9 and geometric residual22.
Normalizers are refit using only the selected training rows. Dataset selection
uses a separate seed; all six runs share the exact saved subset and source hash.

`outputs/graph/low_data/` must initially be empty. Use `--root NEW_PATH` for another
experiment. No checkpoint is overwritten. Settings are frozen in `plan.json`
before training/test evaluation. Other flags include `--dataset`, `--config`,
`--train-episodes`, `--subset-seed`, `--seeds`, `--epochs`, and `--flat-width`.
Preparation rejects parameter differences above 1%. `--trajectory PATH.h5`
supports relocating the source H5 without changing its original content hash.

A small execution check (not the default six-run measurement):

```bash
WANDB_MODE=disabled .venv/bin/python -m graph.prototype.low_data \
  --root outputs/graph/low_data_smoke --train-episodes 2 --epochs 1 --seeds 0
```

Individual phases read the saved plan. Dataset/budget flags apply to `all` or
`prepare`; existing plans are authoritative for the other phases:

```bash
.venv/bin/python -m graph.prototype.low_data --phase prepare --root outputs/graph/low_data_new
.venv/bin/python -m graph.prototype.low_data --phase train --root outputs/graph/low_data_new
.venv/bin/python -m graph.prototype.low_data --phase evaluate --root outputs/graph/low_data_new
.venv/bin/python -m graph.prototype.low_data --phase report --root outputs/graph/low_data_new
```

Training skips only compatible, fully completed runs. Interrupted runs are not
resumed from a best checkpoint; use a new root for those. Evaluation checks all
completed checkpoints and their exact selected-train normalizers before looking
at test error, then skips existing evaluations only when identities still match.
Evaluation and training are sequential, so their latency measurements do not
overlap with this matrix's training processes. Other host load can affect timing.

## Outputs and interpretation

- `dataset.pt`, `dataset.json`, `config.toml`, `plan.json`: unchanged held-out rows,
  selected train groups/IDs, source/subset hashes, actual counts and fixed budget.
- `training/{flat,surface}_seed*/`: existing v2 checkpoints, history and loss plot;
  stdout/stderr are under `logs/`.
- `evaluation/*.json`: global/phase/episode errors, selected epoch, identity and
  CPU batch-1 latency (feature/graph + shared state + network; excludes GT/physics).
- `comparison/results.{md,json}`: all seeds, mean/sample standard deviation,
  matched seed differences, Surface wins, sample-weighted and episode-mean MSE.

Positive relative reduction means lower Surface MSE. Report all runs, including
negative results. A consistent reduction is a small offline observation under
one fixed subset/budget; it establishes neither closed-loop success, statistical
significance, nor a data-efficiency curve. Phase labels are diagnostics only.
Translation/rotation errors are normalized commands, not realized TCP motion.

The input is the historical extracted dataset. This experiment does not resolve
the currently observed live replay drift or prove arbitrary-state expert recovery.
It does not collect Recurrent demonstrations or run Visual perception. CPU
training/evaluation is the only implemented device path here.

Existing `ExperimentLogger` run IDs, checkpoint sidecars and local manifests are
reused. The command above disables W&B transport while preserving local logs;
omit `WANDB_MODE=disabled` for the repository's existing logging policy. No HF
upload is performed. See [tracking](../../docs/experiment_tracking.md).

## Checks

```bash
.venv/bin/python -m unittest graph.prototype.tests.test_low_data -v
```

Tests cover selected-only statistics, complete/reset-group selection, unchanged
held-out values and sample/action alignment, seed pairing and incomplete-matrix
rejection, custom-dataset hash validation, episode aggregation and logger seed.
They do not invoke a simulator or network service.
