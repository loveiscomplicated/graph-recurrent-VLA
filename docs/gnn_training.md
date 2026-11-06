# GNN training

The Graph-team trainer now supports optimization, validation and checkpoint
resume. It is independent of prediction layout: a task supplies the loss and
scalar metrics. The first supported training recipe is gated MPNN Object21
supervision from the Visual corner detector's output and raw saved GT.

## Run on the training host

Use Python 3.11 and install `requirements.txt`,
`visual/corner_detection/requirements.txt`, and optionally
`requirements-tracking.txt`. Extract the approved dataset TAR shards so
`dataset_manifest.json` and its `shard_*/episode_*` paths resolve from one root.
Set `data.root` in `graph/configs/train_gated_mpnn.yaml` to that root, then run
from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_gated_mpnn.yaml
```

For a single collector directory instead, set `data.manifest: manifest.json`.
The directory must have a COMPLETE manifest, committed episode receipts and
per-frame GT (`save_step_gt: true` during collection). This workflow does not
start the simulator, collect data, extract archives or publish artifacts.
W&B disabled mode retains the shared logger's local records.

`device` selects the GNN device. `detector.device` independently selects corner
extraction (CPU by default; select CUDA on the server if supported). GPU detector
execution requires `data.num_workers: 0`; CPU detection can use worker processes.
The loader constructs graphs on CPU and the batch moves to the selected GNN
device. There is no local training or cache preparation command run automatically
when importing modules, validating config or constructing a dataset.

## Modules

| Module | Responsibility |
| --- | --- |
| `training/gnn/trainer.py` | Shared optimization, validation, logger and checkpoint selection |
| `training/gnn/tasks.py` | Task contract, StepResult and Object21 loss/metrics |
| `training/gnn/data.py` | Splits, optional caches, loaders and graph collation |
| `training/gnn/checkpoint.py` | Atomic epoch snapshots and RNG/state restoration |
| `training/gnn/loss.py` | Existing Object21 and paired-delta objective functions |
| `data_collection/episode_dataset.py` | Approved raw frame/GT reading, no GNN dependency |
| `graph/utils/object21_targets.py` | Raw world GT to Graph-team Object21 targets |

The old `training.gnn_trainer` and `training.gnn_loss` modules moved to the paths
above. Update external imports accordingly. The common entrypoint remains
`python main.py --config ...`, with `trainer: training.gnn.trainer:train`.

## Data, splits and cache

Only manifest-listed episodes with valid receipts are read. Partial/rejected
paths and repeated reset seeds are rejected. Splitting is deterministic by seed
and occurs by episode before frame selection. The default reserves 20% validation
and 10% test, each at least one episode; at least one training episode must remain.
Test episodes are reserved, never passed through detection or validation.

An optional `data.split_file` JSON can specify exact `train`, `validation` and
`test` episode-ID lists. Every approved episode must occur exactly once. Keep
this split file for future evaluation. `frame_stride` and optional
`max_frames_per_episode` are explicit sampling controls, not automatic subsampling.

With `data.use_cache: true`, each requested frame is detected and graphed on its
first access, then saved as a compact NPZ below `data.cache_dir`. Subsequent
epochs reuse it. Dataset construction only indexes/hashes source files and does
not write caches. Validation frames are cached when validation first reads them.
`data.use_cache: false` runs detection/graph construction for every access.

Keys include raw frame SHA256, sample ID, camera, complete detector parameters,
backend, graph/confidence settings, library versions and preprocessing code
hashes. Corrupt or mismatched entries raise an error. Entries use no pickle and
are published by atomic rename. Changing settings/code creates a different
namespace. Raw data changing after indexing is rejected on cached access.
No GT corner identities, object sizes or hole pose enter detection or graph
construction. Saved TCP position is treated as proprioception; remaining GT
is used only to construct supervision.

`data.empty_inputs: keep` retains empty detections with valid GT and reports
`empty_input_fraction`. `skip` filters empty samples during collation; metrics
then describe the retained subset. If a split has no usable batches, training
fails instead of reporting a fabricated loss. Padding is excluded from graph
operations. Custom node-level tasks must also mask padding in their losses.

## Objective and extension contract

Object21 uses TCP-relative positions with world-aligned axes, world wxyz
quaternions, and the peg +X tip relative to the hole position. It reuses
`object21_loss` without changing its weights. Loss normalization scales and
endpoint consistency weight are recorded under `task.settings`. Paired-delta
loss remains available but is not used in this recipe.

`Task.compute(model, batch)` returns:

```python
StepResult(
    loss=differentiable_scalar,
    metrics={"loss": python_float, "another_metric": python_float},
    metric_weights={"loss": sample_count, "another_metric": its_denominator},
)
```

Metrics are means with explicit positive denominators, so uneven final batches
are aggregated correctly. Object21 also records position-error norms in metres.
The loop handles backward/optimizer only in training and disables gradients in
validation; it never accesses Object21-specific output keys.

Another GNN can use the built-in model selector or `model.factory:
package.module:Constructor`, with the remaining model mapping passed as keyword
arguments (excluding `name`/`factory`). Select `task.factory` and task constructor
arguments in `task.settings`. If targets/data differ, a `data.factory` callable
accepts `(effective_config, shuffle_generator)` and returns `DataBundle` with
train/validation iterables and `identity.dataset_sha256`; batches provide
`.to(device)`. All factories use `module:callable` notation. No trainer rewrite
is needed. BasicGNN currently returns node tensors, so Object21Task is unsuitable
without a compatible readout. Its example config retains `mode: initialize`.

## Outputs and resume

During training, the terminal reports the current/total epoch, train or
validation phase, completed/total phase batches, elapsed time within the epoch,
estimated time remaining within the epoch, and estimated time remaining across
all epochs. `training.log_interval_seconds` defaults to 10 seconds; reports are
emitted at batch boundaries and phase transitions. A slow batch (including
first-access detection/cache creation) can therefore delay the next report.

The same numeric measurements are saved through `ExperimentLogger` under
`progress/*` in `experiments/<run_id>/metrics.json` and W&B when enabled.
`time/epoch_seconds` and each history row's `seconds` record epoch wall time
through checkpoint writing/registration, before writing that history row.
Epoch time includes both training and validation. Timing starts after dataset
indexing, model construction and resume loading; it does not include preparation.

ETAs use measured seconds per loader batch, separately for training and
validation once each has been observed. Initially the observed rate also
estimates the unobserved phase; until a batch has completed, estimates display
`estimating` and ETA scalars are omitted. Future epochs use the mean duration
of completed epochs in this invocation, including checkpoint overhead. Resume
uses the restored epoch number and measures rates afresh. Cache warmup and
variable batch costs can make initial estimates inaccurate. Custom unsized
iterables retain elapsed-time logging but show unknown batch totals and omit
ETAs. Measurements use the host wall clock without adding accelerator
synchronization; CUDA/MPS timing has not been verified on this host.

Each invocation creates a fresh ExperimentLogger UUID and a separate
`output.root/<run_id>/` directory containing effective config, data identity and
splits, epoch history, `last.pt` and `best.pt`. The logger records scalar metrics
and registers outputs through `link_output`; it does not upload checkpoints.

`last.pt` contains model, optimizer, scheduler, completed epoch, earlier best
weights, configuration, dataset hash and Python/NumPy/Torch/accelerator/shuffle
RNG state. Resume sets top-level `resume` to its path and `training.epochs` to a
larger **total** epoch count. The new run has its own history and source sidecar
identity. Data/split/preprocessing hash, objective, optimization settings, seed,
batch settings and device type must match. Best weights survive even if no
resumed epoch improves validation loss.

`best.pt` is a weights-only validation-loss selection snapshot, for later model
loading, not training resume. Exact continuation additionally depends on the
same runtime and deterministic backend behavior; CUDA reproducibility has not
been established here. Best validation loss is not simulator success rate.

## Verification limits

Local verification deliberately does not run training or generate frame caches.
Contract tests check targets, splits, collation, both model output contracts in
validation, settings and checkpoint/RNG round trips using untrained weights:

```bash
python -m unittest tests.test_gnn_trainer tests.test_gnn_training \
  tests.test_graph_builder.ConfidenceTests tests.test_graph_builder.GraphBuilderTests -v
```

The optimizer/resume integration test is disabled by default. On the training
host it can be explicitly enabled with `RUN_GNN_TRAINING_TESTS=1`; it uses synthetic
graph fixtures and does not establish dataset quality. Real-dataset training,
cache materialization, CUDA/MPS training and end-to-end resume were not run on
this local host. The supplied data path is a placeholder, not a bundled dataset.
