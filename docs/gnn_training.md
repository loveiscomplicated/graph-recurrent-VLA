# GNN training

The Graph-team trainer now supports optimization, validation and checkpoint
resume. It is independent of prediction layout: a task supplies the loss and
scalar metrics. The first supported training recipe is gated MPNN Object21
supervision from the Visual corner detector's output and raw saved GT.

## Run on the training host

### Train from a downloaded cache

After extracting the published `graph_cache.tar` into `/workspace/gnn_cache`, use:

```bash
source /venv/main/bin/activate
cd /workspace/graph-recurrent-VLA
python main.py --config graph/configs/train_gated_mpnn_cached.yaml
```

The dedicated config selects `training.gnn.cached_data:build_data`. Set
`data.cache_dir` to the extraction directory, whose contents must include the
preprocessing namespace and `metadata/data_identity.json` plus
`metadata/cache_inventory.json`. Raw episodes, simulator packages, OpenCV and the
original detector runtime are not required by this data factory. Torch, NumPy
and PyYAML are needed to deserialize batches and execute the trainer.

The loader retains the published episode assignments, never reads reserved test
frames and never regenerates a missing cache entry. Indexing checks metadata
consistency, split overlap, frame counts, file existence and sizes. Each sample
read checks the file's SHA256 against the published inventory, its embedded
source identity, graph schema and target shape. It does not hash original RGB-D
or execute current preprocessing to choose a new namespace. The original
preprocessing versions/hashes are preserved as provenance. Graph schema version
1 is supported; incompatible entries fail validation. Camera, graph and input-
confidence settings must match the archive; replay cannot change cached features.

The existing data factory remains available for raw data and on-demand cache
creation. Cache-only replay uses a distinct dataset identity combining the
original dataset hash, inventory bytes and empty-input policy. Checkpoints from
raw mode cannot silently resume into this mode. Same-cache resume across paths
uses the same identity. DataLoader seed controls shuffling and model RNG, while
the archive's saved split is authoritative; validation/test fraction defaults
are not used to resplit it. `split_file`, frame stride other than 1, and frame
limits are rejected to prevent silently training on a different subset.

Cache-only loader contracts can be checked without training or real cache
generation using `python -m unittest tests.test_gnn_cached_data -v`.

### Train from raw episodes

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

The gated recipe enables CUDA model compilation:

```yaml
training:
  compile:
    enabled: true
    dynamic: true
```

Other configs default to `enabled: false`. Requested compilation runs only on
CUDA; CPU/MPS keep eager execution. The trainer records the request and actual
activation under `model_compilation` in the shared experiment metadata. It uses
`model.compile(dynamic=True, mode="default", fullgraph=False)` after checkpoint
restoration. This calls `torch.compile` in place, preserving parameter identities
and plain checkpoint keys without an `_orig_mod.` prefix. Both training and
validation use compiled calls; initial calls or new graph shapes can trigger
compilation. Compiler failures propagate rather than silently changing execution.

Gated node encoding, embeddings, attention scores and deterministic message
updates now compute dense candidates and mask their results. Edge-type selection
and empty-edge early returns are retained, including `None` gradients for absent
edge groups. With nonzero dropout during training, message updates retain the
selected-node path to preserve the original dropout draw count/order. Saved
weights remain compatible; exact eager/compiled resume trajectories are not
guaranteed, and compiler settings are recorded but are not locked by the existing
resume signature.

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

Graph batches are validated during collation; the gated MPNN does not repeat
validation in `forward()`. Direct callers and custom data factories must validate
their graphs before model execution. When semantic IDs are enabled, pass the
model's `num_semantic_ids` to `GraphBatch.validate` to check vocabulary bounds.

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
    metrics={"loss": detached_scalar_tensor, "another_metric": python_float},
    metric_weights={"loss": sample_count, "another_metric": its_denominator},
)
```

Metrics are means with explicit positive denominators, so uneven final batches
are aggregated correctly. Scalar tensors and Python numbers are accepted;
Object21 returns detached tensors. The loop accumulates weighted metrics on the
model device and transfers the epoch means to CPU together for Python-number
logging. CPU/CUDA accumulation uses float64; MPS uses float32. Loss and metric
finiteness are checked together each batch, retaining one host decision before
an optimizer step. Object21 also records position-error norms in metres.
The loop handles backward/optimizer only in training and disables gradients in
validation; it never accesses Object21-specific output keys.

With gradient clipping enabled, `clip_grad_norm_(error_if_nonfinite=True)`
checks gradient finiteness before the optimizer step. Without clipping, the
loop reduces per-parameter finiteness flags on the device and checks the combined
result once. These checks retain host synchronization; they do not make the
entire training loop suitable for full-graph compilation.

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

### Loss and geometry metrics

Every epoch records sample-weighted train/validation means through the existing
`ExperimentLogger` and `history.json`. In addition to total loss, its raw
components, position/endpoint mean distance errors (metres), and empty-input
fraction, Object21 records:

| Metric suffix | Meaning |
| --- | --- |
| `peg_rotation_error_deg`, `hole_rotation_error_deg` | Mean shortest quaternion rotation angle in degrees; q and -q are equivalent |
| `peg_half_size_x_error_mm`, `peg_half_size_y_error_mm`, `peg_half_size_z_error_mm` | Mean absolute peg half-size error on each local axis, in millimetres |
| `hole_radius_error_mm` | Mean absolute hole radius error in millimetres |
| `weighted/<loss_component>` | Contribution after the existing loss weight; these sum to total loss |
| `empty_input/<geometry_error>`, `nonempty_input/<geometry_error>` | Validation-only geometry errors, separately for samples with no valid nodes and samples with at least one valid node |

Subset means use each group's own sample count across batches. A group with no
samples is omitted, rather than assigned zero error. With `empty_inputs: skip`,
the empty group is filtered before the task and therefore has no error metrics.
Validation grouping transfers one group count to the host per batch; other
geometry metric calculations remain on the model device until reporting.

Training also records `gradient_norm` (mean global L2 norm before clipping) and,
when clipping is enabled, `gradient_clip_fraction` (fraction of optimizer steps
whose gradients were scaled). These are averaged by optimizer step, independently
of sample-weighted loss metrics. Norm measurement does not modify gradients;
the clipping path reuses the norm returned by the existing clipping operation.

At the existing progress cadence, the logger records cumulative means for the
current phase under `progress/train/*` or `progress/validation/*`, and the
terminal includes cumulative mean `loss`. Means reset at each phase and epoch;
they describe all processed samples in that phase, rather than just the latest
batch. Device scalars are copied together only when a progress report is due.
The objective, loss weights and best-checkpoint selection are unchanged.

### Time and progress

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

### CUDA masking and compilation measurement

On 2026-11-06, an isolated source snapshot was measured on Vast.ai instance
58752018 (RTX 3090, PyTorch 2.14.1+cu132). The local config at capture used
hidden 64, two layers, dropout 0, batch size 32 and AdamW. The active cache
generator on instance 59288396 (PID 1545) was left running. Only 96 completed
cache files were read from that instance; benchmark code and compiler caches
were placed in a separate directory on the idle instance. Existing remote
repositories and Python environments were not modified.

The samples formed three batches with padded node counts 15/16/12 and edge
counts 1664/1586/1374. Timings cover the copied local training loop with
GPU-resident batches: model, loss, metrics, backward, gradient clipping and
AdamW. Each case used five 30-step windows, repeated in reverse case order
in a second process execution; the table averages both executions. Disk reads,
preprocessing, transfers, checkpointing, logger/progress output and initial
compilation are excluded. These are compute-loop measurements, not end-to-end
epoch times or convergence results.

| Benchmark-only model variant | Eager ms/step | Compiled ms/step |
| --- | ---: | ---: |
| Current masked selection | 28.59 | 27.07 |
| Dense node encoding, pooling and updates; existing edge selection | 24.97 | 21.40 |
| Dense nodes and all edges, with masked message weights | 21.65 | 19.51 |

Compilation used `torch.compile(model, dynamic=True, fullgraph=False)` in default
mode. The first pass over all three shapes, including compilation/backward,
took about 8–14 seconds; variants shared a fresh, isolated compiler cache, so
these are not independent cold-cache comparisons. Current-code execution means
ranged from 27.14–30.04 ms eager and 25.88–28.26 ms compiled, so the small
compile-only benefit overlaps execution variability. Dense nodes plus compilation
reduced mean step time by about 25% (1.34x throughput) versus current eager code.
The current compiled model recorded `aten.nonzero` graph breaks.

All six cases matched reference outputs and gradient values within the recorded
tolerances on these batches. CPU checks additionally covered empty, isolated,
single-object and padded graphs. Dense nodes preserved gradient presence in
these checks. The all-edge candidate changed some absent-edge parameter
gradients from `None` to zero: AdamW can then update parameters it previously
skipped. It is therefore an experimental speed comparison, not a replacement
with equivalent optimizer behavior. Nonzero-dropout RNG equivalence and broader
graph-shape coverage remain unverified by that benchmark. The measured dense-node
candidate was subsequently adopted in the production model, with the original
selected-node update retained for nonzero-dropout training. The all-edge
candidate remains benchmark-only.

Local evidence: [summary](../outputs/gnn_compile_benchmark/summary.json),
[first execution](../outputs/gnn_compile_benchmark/initial_results/),
[reverse-order execution](../outputs/gnn_compile_benchmark/results/), and
[edge cases](../outputs/gnn_compile_benchmark/source/edge_cases.json).
The ignored [source snapshot](../outputs/gnn_compile_benchmark/source/) contains
sample identities/hashes, exact local loop functions, variant generation and
benchmark scripts. From that directory on an idle CUDA host, rerun with
`OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python benchmark.py --variant dense_nodes --compile`.
The benchmark refuses to start if another CUDA compute process is present.

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
