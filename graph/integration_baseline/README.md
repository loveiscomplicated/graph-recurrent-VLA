# Task06 opt-in integration baseline

This package implements the approved feasibility baseline. Existing simulator,
GT prototypes, Recurrent policies, Task03–05 sources/artifacts and defaults are
unchanged. No dependency installation or publishing is needed.

From the repository root, with the existing Python environment:

```bash
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/task06-mpl

# New output directories are required; existing datasets are never overwritten.
.venv/bin/python -m graph.integration_baseline.replay \
  --output outputs/graph/integration_baseline/replay_new
.venv/bin/python -m graph.integration_baseline.diagnose_replay \
  --output /tmp/task06_diagnostic.json
.venv/bin/python -m graph.integration_baseline.vision \
  --replay outputs/graph/integration_baseline/replay_new \
  --output outputs/graph/integration_baseline/vision_new --device cpu
.venv/bin/python -m graph.integration_baseline.data \
  --replay outputs/graph/integration_baseline/replay_new \
  --vision outputs/graph/integration_baseline/vision_new \
  --output /tmp/task06_dataset_validation.json
.venv/bin/python -m unittest discover -s graph/integration_baseline/tests -t . -v
```

The replay pilot repeats the first three existing expert episodes twice. It
uses the original initial-state repair mapping and existing tolerances (initial
2e-5, sequential observation43 1e-3). If any complete pilot fails, the remaining
88 episodes are **not executed**. Verified pre-failure prefixes may be saved,
with failed frames/suffixes excluded and the final valid observation unlabelled.
Their existence never changes the full pilot status to PASS. `--count` limits
post-pilot collection only; the three-episode pilot remains mandatory.

The observed Task06 run has 3 partial prefixes, 0 complete episodes, 473 raw
observations and 470 targets. See the [result report](../../docs/visual-graph-integration/06_graph_gnn_gru_baseline.md).
Data directories are ignored outputs, and every builder writes an independent
manifest, source hashes, coverage and failures. The manifest fixes the existing
73/9/9 episode/reset-group split. No frame-random split is created.

`vision` uses locally cached Grounding DINO and SAM2 weights. It requires offline
mode and does not install dependencies or substitute GT segmentation. The
existing initializer sees only RGB prefixes, and only current-frame decisions
are accepted. SAM2 propagates one current frame at a time with `num_frames=t+1`.
The installed SAM2 uses `min(num_frames,max_obj_ptrs_in_encoder)` for temporal
pointer encoding; full-video length therefore changed a prefix pixel in the
rejected `vision_v1` attempt. Its audit/log/source snapshot remain preserved,
and `ReplayDataset` rejects it. The accepted output is `vision_v2`.
JPEG video loading
may read all RGB arrays into memory, but future-frame features/prompts do not
enter earlier outputs; no backward tracking or backfill is run. Existing Shape-
First/Surface-Second helpers, PCA5/F5, EE transform, observed components and Raw
EDE are called directly. Research helper imports remain explicit opt-in
dependencies. No point cap/filter is introduced.

`ReplayDataset` validates source/input hashes, frozen split/reset groups, exact
episode/timestep joins, target/flags, TCP calibration, and mask provenance.
`sequence()` rejects missing frames, final unlabelled observations and any
terminated/truncated crossing. The original 100-step TimeLimit flags are kept
even though source expert trajectories continued; their semantics must be
settled before full BC training.

Model interfaces:

```python
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.data import collate_sequences, ReplayDataset
from graph.integration_baseline.model import IntegrationPolicy, EpisodeStream

dataset = ReplayDataset(replay_directory, vision_directory)
normalizer = dataset.fit_normalizer()  # captured train episodes only; frozen after fit
model = IntegrationPolicy(dataset.schema)
sequences = [dataset.sequence(episode_id, 0, 4)]
graphs, state, target, valid = collate_sequences(sequences)
action, hidden = model(graphs, state, normalizer)  # [B,T,7], [1,B,144]

# For streaming inference, read robot_observation(env) and use the exact saved
# schema. Build the Vision packet from that same observation. EpisodeStream
# preserves hidden across consecutive observations and resets on identity,
# explicit reset or previous terminated/truncated flags.
stream = EpisodeStream(model, normalizer)
```

The actual state schema is derived from field sizes and stored in `schema.json`:
all `agent.get_proprioception()` qpos/qvel; duplicate gripper positions,
velocities and summed opening/opening velocity; TCP, left/right finger and base
world pose/linear/angular velocity. It is 76 dimensions on this installed Panda.
A changed/nonempty Controller State is rejected for explicit schema review,
never silently dropped. Only current robot kinematics enter this adapter.
Effort/acceleration/contact absent from the observation API are not invented.

Graph input is position3 + finite F5 + availability3 + validity3, plus a learned
five-type embedding. The original packet is a separate deep copy. Raw intra
edges preserve supplied directed order with +3. Robot k-NN ties are ordered by
distance²/source/row. Exact Peg–Box block scan ties are
`(distance², peg_source, box_source, peg_row, box_row)`; each block retains the
global first M. It computes all P×Q distances, uses O(block_rows×Q+M) scratch
(plus sorting arrays), and sorting costs O(PQ log(block_rows×Q+M)) in this
implementation. M=0 and empty objects are valid. No Delaunay runs in the graph
consumer. Edge geometry is destination−source, with finite zero-distance output.

Readout is exactly 5d without an extra projection. Presence2 and normalized robot
state enter after GNN. The existing `RNNPolicy` encoder/GRU/head and Graph v2 7D
regression MSE are reused. Raw normalized controller targets are unchanged:
translation ±1→±0.1m, installed rotation norm clipping then multiplication by
`rot_lower=-0.1`, root-aligned-body rotation, gripper -1→-0.01m / +1→0.04m.
Loss does not clip predictions, z-score actions or use gripper BCE.

To reproduce integration evaluation **after** finished dataset manifests:

```bash
.venv/bin/python -m graph.integration_baseline.smoke \
  --replay outputs/graph/integration_baseline/replay_v2 \
  --vision outputs/graph/integration_baseline/vision_v2 \
  --start-snapshot /tmp/task06_start
```

The start snapshot is an audit input specific to this work session. For another
workspace, create a directory with `protected.json` mapping preexisting relative
paths to SHA-256 before beginning, and retain the replay diagnostic JSON/NPZ
there. Its purpose is protecting existing dirty/untracked work, not selecting
data or model hyperparameters. `smoke` creates a new ExperimentLogger UUID,
tests all contracts and saved Task05-E inputs, then performs exactly one Adam
step on two verified actual train-prefix sequences. It does not run a training
loop or policy rollout. It records the executed collection-source snapshots
separately from the final integration-source snapshot when supplied.

Artifacts include config/metadata, manifests, JSON/NPZ/PNG/HTML, test logs,
gradients and parameter values before/after, source hashes/snapshots/diff and
readback/link verification. CPU packet→action inference timing excludes replay,
upstream Vision and IO. Per-frame offline costs and per-episode replay costs
remain in their manifests. CUDA/MPS are NOT_EXECUTED on this task.

## Task06-C: direct dataset and independent BC pipeline

This is a separately authorized opt-in path. It does not use converted replay,
change the shared 100-step default, or modify ScriptedExpert, Vision algorithms,
Graph/GNN/GRU, action schema or BC loss. The frozen first-run settings are in
[`direct_config.json`](direct_config.json). Discord was explicitly waived by the
user for this run: no webhook is required or used. Full logs, summaries, error
bundles and recovery events remain local.

The pipeline runs collection → causal Vision → full integrity/learning gate →
training → single final offline test → final HTML/JSON/PNG/Markdown in one
detached tmux job. A nonzero stage exit prevents subsequent stages. Codex does
not watch progress, poll tmux, or wait for training to finish after initial
launch verification. There are no timers that invoke Codex.

From the repository root, using the same existing Python environment:

```bash
# Read-only local status (does not attach to or poll tmux).
.venv/bin/python -m graph.integration_baseline.run_control status --run <run_id>
tmux ls
tmux attach -t grvla-06c-<first_8_uuid_characters>
# Detach from tmux without stopping the job: Ctrl-b, then d.

# Cooperative stop. Collection finishes the current episode. Vision may stop
# after its current frame and preserve a partial graph attempt. Training saves
# model/optimizer/RNG/hidden/sequence cursor at the next chunk boundary.
.venv/bin/python -m graph.integration_baseline.run_control stop --run <run_id>

# Explicit resume: checks checksums, frozen config/split, completed data, and
# last valid checkpoint. It never starts a different backend or a fresh run.
.venv/bin/python -m graph.integration_baseline.run_control resume --run <run_id>
```

`stop` is a normal event, not an error callback. A stopped Vision episode is
reprocessed causally in a **new attempt directory**, preserving its previous
partial files. A completed episode is skipped only after checksum validation.
A collection intent without a completed receipt is audited as interrupted and
its seed is never recollected. A finalized HDF5 without its receipt requires
explicit integrity recovery; it is not silently assumed valid. Collection
cannot reduce the target/split when fewer than 120 successes are available.

Canonical raw storage is one gzip-compressed HDF5 per episode, with complete
T actions/T+1 observations and both cameras' RGB, original int16-mm depth,
floating metre views and calibration. Every reset installs a fresh controller
audit hook. Native expert Action7, env.step input and actual controller entry
must match exactly. Diagnostic expert phase/reset identity is outside model
inputs. Success-only training selection is distinct from all-attempt success
rate; failed and interrupted attempts have their own audit records.

Files become valid via readback and atomic no-overwrite finalization. Immutable
episode receipts, final dataset/graph manifests, source hashes and split hashes
are retained. Append-only events and mutable status/pointer files are explicitly
separate from those immutable records. Partial files are never valid episodes
or checkpoints. An optional explicit compatibility export is available:

```bash
.venv/bin/python -m graph.integration_baseline.run_control export \
  --episode experiments/<run_id>/collection/episodes/ep30000.h5 \
  --output /tmp/ep30000-export.npz
```

The direct loader joins dataset/episode/timestep/source SHA, RGB, calibration,
EE, robot state and target. Full validation consumes every observation including
the final unlabelled frame; only t<T is used for BC. Split is episode/reset-group
96/12/12, frozen with seed42. Train-only RobotNormalizer is the existing mean/std
adapter with .001 std floor. Existing graph finite-feature/mask preprocessing is
unchanged and has no fitted feature statistics. Validation/test never update
statistics. The initial collection seeds30000..30159 exclude prior pilot2000..2004,
evaluation2000..2029, and smoke29000..29003.

Existing causal DINO/SAM2, Shape-First keep_ratio0.5, optimized PCA5/F5, EE and Raw
EDE are reused with no point budget. Empty objects remain valid. Per-frame mask,
overlap, depth-invalid, observed support, selected points, phase presence and
recovery patterns are recorded. Processing exceptions, corruption and schema
violations are fatal, never converted into valid missing observations. SAM2's
optional extension warning is preserved; no dependency is installed to hide it.

CPU is the explicitly tested/frozen device. Candidate B2×T32 is tested on real
packets before launch and again on the full dataset. Episode pairs advance in
consecutive TBPTT chunks, preserving each stream's detached hidden state across
chunks, shrinking the active batch when one episode ends, and resetting at the
next episode. Padding is masked. Chunk selection is not random; only episode
order is shuffled each epoch. No hidden is shared across episodes. TimeLimit250
ends this collection episode and the next sequence starts after a physical reset;
there are no artificial 100-step segment boundaries.

Optimizer follows Graph v2: AdamW, weight_decay1e-5, betas(.9,.999), eps1e-8,
amsgrad=false; the first approved lr is3e-4. Scheduler follows Graph v2's
CosineAnnealingLR with T_max50/eta_min0. Max50 epochs, clip1, patience10,
validation each epoch, native continuous7 regression MSE; no target/prediction
clipping. Model initializes randomly. Checkpoint selection uses validation only;
test is evaluated once after selection. A test-start marker without a finalized
result blocks automatic repetition. BC loss does not establish manipulation
success. Student closed-loop rollout is not included.

All checkpoints have unique immutable paths; `best.json` and `last_valid.json`
are atomic pointers with checksums. They include model/optimizer/scheduler,
Python/NumPy/Torch RNG, epoch/global step, training config, robot/action/model
schemas, train normalizer, dataset/split/vision/config hashes, and mid-epoch
cursor/hidden/aggregates when cooperatively stopped. Normal epoch checkpoints
resume at an epoch boundary; failed work since that checkpoint is replayed.

The primary UUID identifies the prepared pipeline/dataset. To preserve existing
ExperimentLogger identity rules, each stage invocation and each resumed
invocation gets a new **execution Logger UUID**, linked via `source_run_id` and
`pipeline_run_id`; see `execution_runs.jsonl`. Logger IDs are never overridden or
reopened. W&B is disabled and only existing local artifact registration is used.
No HF/W&B upload or git commit occurs.

Fatal stage exits create a sanitized local error bundle and may dispatch one
separate `grvla-06c-repair-<uuid8>` tmux session. Dispatch is event-driven after
`child.wait()`, not periodic monitoring. `codex exec --sandbox workspace-write`
with `approval_policy=never` is used, without bypass options. Missing verified
CLI support yields `BLOCKED_AUTO_CODEX_CALLBACK`. The actual CLI sandbox fixture
and mocked dispatch/HTTP tests are reported separately. Discord delivery is
`USER_DISABLED` for this run, not falsely reported as verified.

Only confirmed direct collector/loader/trainer/runner bugs may be repaired.
The repair prompt forbids changes to Expert/Vision/Graph/GNN/GRU/loss/schema,
split/hyperparameters/thresholds or existing data/checkpoints. The independent
handler checks a source allowlist, a regression test, all integration contracts,
and dataset/checkpoint integrity before a single safe resume. A persistent
one-use dispatch marker and repair lock prevent duplicates, recursion and
repeated restarts. An uncertain fix or second failure requires user intervention.
Authentication/sandbox failures are never bypassed. A killed OS/tmux server or
host shutdown cannot itself deliver a local error callback while the host is off.

tmux survives detaching and normally terminal/SSH disconnects on this host. It
does not survive reboot, power loss, tmux server termination, or all macOS logout
policies. Sleep suspends execution. No sleep settings or host policy are changed.
On restart use explicit `resume`; neither datasets nor checkpoints are replaced.

The final pipeline creates `graph_gru_training.html` with relative PNG/JSON links,
readback/link verification, and updates the run-owned
[`06c_direct_expert_bc_training.md`](../../docs/visual-graph-integration/06c_direct_expert_bc_training.md).
Before completion, that document and HTML explicitly show in-progress status.
Browser layout validation, CUDA/MPS, HPO, architecture search and closed-loop
success evaluation are not claimed.

### Existing-data CUDA execution

`gpu_run` is an explicit opt-in path for migrating the collected CPU dataset
to an NVIDIA host. It preserves the original raw dataset UUID, byte-exact raw
config and frozen 96/12/12 split. A new ExperimentLogger UUID owns GPU graphs,
training and reports; `data_source.json` links the immutable original directory.
No simulator or Expert is run on this path. The original CPU default remains.

Use the repository's existing locked Linux environment (`visual/setup.sh`),
which pins Torch2.8.0+cu128 and the existing SAM2 revision. Populate the same
DINO/SAM2 assets before offline execution and record their revisions/hashes.
No global Python environment, NVIDIA driver or model architecture is changed.

```bash
source visual/env.sh
source /opt/nvm/nvm.sh  # this Vast image installs Node/Codex outside default PATH

# Server preflight must contain actual callback verification, not a copied
# success claim from another host. --fixture prepares a separate smoke UUID.
python -m graph.integration_baseline.gpu_run prepare \
  --data /workspace/datasets/<original_dataset_uuid> \
  --preflight /workspace/task06c-preflight --fixture

# Run this smoke command in its own tmux session. It validates all raw files,
# processes two whole real TRAIN episodes, and executes B2×T32 forward/backward,
# optimizer, sequence-vs-step and exact model/optimizer/CPU+CUDA RNG resume gates.
python -m graph.integration_baseline.gpu_run smoke --run <smoke_uuid>

python -m graph.integration_baseline.gpu_run prepare \
  --data /workspace/datasets/<original_dataset_uuid> \
  --preflight /workspace/task06c-preflight
python -m graph.integration_baseline.gpu_run launch \
  --run <new_cuda_uuid> --smoke-run <smoke_uuid>
```

The launcher verifies matching smoke/full configs, reuses the immutable smoke
graph episodes and starts the existing independent tmux supervisor from Vision.
Status/stop/resume still use `run_control` and the new CUDA UUID. CUDA is mandatory
for both Vision and model computation; failures never select CPU automatically.
The BC model retains FP32, with AMP/TF32 off and deterministic operations required.
Vision uses the existing upstream CUDA inference defaults; no autocast is added.
Mid-chunk hidden state is saved on CPU and explicitly restored onto the selected
device. Checkpoints preserve CUDA RNG alongside existing state. Memory reports
use correct Linux RSS units and include CUDA allocated/reserved/peak bytes.
The tmux launch script persists the selected model-cache and Codex executable
directory so it does not depend on the SSH client's transient shell environment.

Each CUDA run owns `06c_direct_expert_bc_training_cuda_<uuid8>.md`; the original CPU report
and raw data are retained. GPU Vision timing and numeric/memory verification must
come from the target host's smoke, not from the earlier CPU result.

If the container forbids the namespaces required by Codex's workspace sandbox,
the actual callback fixture fails even after login and training stays
`BLOCKED_AUTO_CODEX_CALLBACK`. No sandbox bypass or kernel restriction change is
used. **Only an explicit user waiver** permits `gpu_run prepare --without-callback`.
That run preserves fatal error bundles and exits for manual intervention without
dispatching Codex. Smoke/full configs may differ in this operational requirement
the fixture marker and the report destination; numerical/model/data/runtime settings must still match.

### Validated FP32 Vision execution revision

`vision_speed` is opt-in through a checksummed `execution_revision.json` pointer.
It memoizes raw image/prompt DINO detections before prefix-dependent assignment,
reuses SAM2 preflight only while objects/conditioning inputs are unchanged, and
compiles the existing SAM2 image encoder forward using AOT Eager, retaining eager
CUDA kernels. Inductor is excluded when paired tests change mask pixels. Precision,
1024 internal image size, prompts, causal horizon, per-frame propagation, masks,
Shape-First/F5/Raw EDE and policy architecture remain unchanged. No BF16 or VOS
predictor replacement is enabled. A temporary prompt, object change or reset
invalidates the preflight fast path.

Adoption requires real paired RGB-D-to-CPU-Action7 timing, exact masks/packets/
actions, agreement with existing graph-cache masks and contract tests. The
immutable original config/source snapshot remains; `execution_revisions/`
stores the verification, new source inventory and source/config checksums.
Source guard rejects unvalidated revisions or any later unrecorded source edit.
Completed graph receipts are reused. Unfinished episode attempts remain partial.
Checkpoint/config/split identities and optimizer/RNG state are retained.

Normal `run_control resume --run <uuid>` can create a new pipeline window in an
existing STOPPED run's tmux session, retaining its independent Discord window.
First compile/setup and warm frame timings must be reported separately. The
source/config/verification checks run again before each subsequent stage.

## Opt-in reuse of Visual's GPU streaming path

Implementation, isolation and measured validation are recorded in
[the streaming reuse report](../../docs/visual-graph-integration/06d_visual_stream_reuse.md).

`vision.backend="visual_stream"` selects the existing
`visual/scripts/stream_v1.py::StreamTracker`: SAM2.1 Small, 1024 internal input,
native image-encoder compilation, BF16 autocast with FP32 weights, GPU resize,
two retained frames and sixteen non-conditioning memory frames. A missing
backend preserves `legacy_fp32`. CUDA is required; no CPU fallback is selected.
The new path does not use the FP32 `execution_revision.json` mechanism.

Grounding DINO still performs detection using the existing prompts, thresholds,
five-frame scans and two-scan confirmation. `MemoizedDetector` caches only raw
image/prompt results; prefix-dependent initialization is evaluated as before.
New current-frame DINO boxes enter the existing SAM2 state before propagation.
Existing objects keep their conditioning memory. Missing objects produce empty
masks, and the adapter returns the existing `(2,H,W)` boolean CPU mask boundary.
JPEG staging and independent late-object feature caches are removed.

Visual's opt-in `StreamTracker(..., observed_horizon=True)` exposes only `t+1`
frames, accepts an empty `start(rgb,{})`, and supports `step(rgb,boxes=...)`.
The original fixed-horizon default and existing callers are preserved. The
preprocessing and propagation algorithms remain in the Visual module.

For a **new** existing-data run, prepare both smoke and full configs with
`gpu_run prepare --vision-backend visual_stream` and the normal data/preflight
arguments. This only prepares the run; normal smoke/launch requirements remain.
Cached legacy Vision receipts cannot be relabelled or reused as streaming output.
For online evaluation, add `"backend":"visual_stream"` inside the new config's
`vision` object and set `vision_device="cuda"`. Checkpoint/source/config checks
remain strict: an existing FP32 training config does not silently become a
streaming checkpoint. No training, default switch, or existing-run migration is
implied by selecting this internal experimental backend.

The standalone recorded-input verification is:

```bash
source visual/env.sh
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_COMPILE_THREADS=4
export TQDM_DISABLE=1

# Run once where all original TRAIN files are available. New outputs required.
python -m graph.integration_baseline.streaming_vision_benchmark \
  --prepare-robot-from /path/to/original_dataset_uuid \
  --write-robot /path/to/new_inputs/train_robot.npz

# Requires original TRAIN RGB-D episodes 30002/30047 and the frozen metadata.
python -m graph.integration_baseline.streaming_vision_benchmark \
  --dataset /path/to/original_dataset_uuid \
  --train-robot /path/to/new_inputs/train_robot.npz \
  --repeats 3 --threads 8 --output-root experiments
```

Each condition owns a fresh worker process. The comparison measures legacy
offline/online, direct Visual, and the new adapter. Controlled ablations use
frozen RGB-keyed DINO outputs from the direct run, with unchanged prefix rules;
they compare to `tracking_reference`, not to DINO-inclusive latency. Precision,
PIL/GPU preprocessing, native compile, preflight reuse, separate states, pruning
and inherited deterministic/cuDNN settings are tested separately. Their deltas
must not be added together. A separate profiling pass reports inclusive CUDA
intervals and call counts; model load, warmup and compile are excluded from
steady timing and reported separately.

The benchmark requires an initially idle GPU and monitors other NVML processes
once per second. If another GPU process appears, it terminates only its own new
worker process group and preserves partial outputs. Host NVML PIDs are never
used as container kill PIDs. The default Torch allocator limit is 35% of device
memory, and each condition requires at least 2GiB free disk. Override these
explicitly with `--gpu-memory-fraction` / `--minimum-free-gb` when appropriate.
Use `--instance-id` to record the actual host instance in the new run metadata.

Native max-autotune can select different floating-point kernels in independent
processes. `direct_control` repeats the direct path in another process to audit
that variability. Adoption compares direct/adapter on the **same compiled
predictor**, preserving exact pixel equality, repeatability, initialization,
real SAM2 causality/memory checks and matching packets/graphs/actions. Timed
paired repeats alternate their order and require <=10% p50/p95 adapter overhead.
Independent-process pixel/action differences remain in the report and are not
silently treated as identical. Tracking p95 <=50ms is a separate gate. The synchronous
RGB-D-to-CPU-Action7 pass retains the existing FP32 policy with random weights
and the exact frozen TRAIN96 normalizer. It is a latency/output diagnostic,
not task success, a live-camera test, or a full-path 20Hz claim. All artifacts
are local `ExperimentLogger` outputs; the command never publishes them.

## Task07: opt-in stop-the-clock closed-loop evaluation

This evaluates **B: Stop-the-clock / Zero-simulation-time-inference-delay /
20Hz simulation-control Closed-loop Evaluation**. It establishes neither offline
BC error (A) nor real-time closed-loop performance (C). Each current RGB-D frame
is processed once before one synchronous policy-controlled `env.step`. Wall-time
inference has no 50ms deadline. Simulation advances 50ms per action: 20Hz control,
100Hz physics, five actual `scene.step` calls. The run-local TimeLimit is250;
the shared simulator's default100 remains unchanged.

The existing DINO Tiny/SAM2.1 Small, prefix initialization, independent late-object
tracking groups, Shape-First/PCA5/F5/Raw EDE, graph k4/M16, GNN and GRU are reused.
`online_vision_adapter` registers observed JPEGs into a bounds-checked SAM2 image
sequence and extends `num_frames=t+1`; appending files alone is insufficient.
It retains existing predictor feature/object memory. No backward refinement,
GT object input, expert/rescue action, `vision_speed` revision, or training change
is enabled. `EpisodeStream` carries hidden across consecutive observations,
including empty objects, and resets at episode boundaries. Hidden artifacts
identify the episode and consumed timestep; `restore_stream_hidden` restores
GRU state only and does not pretend to restore simulator/tracker state.

Use the existing Python environment, with locally cached model assets:

```bash
# Two random-weight contract episodes × 3 live steps, a separate250-step timing
# probe, fixture tests, and recorded-prefix/offline SAM2 comparisons.
MPLCONFIGDIR=/tmp/task07-mpl .venv/bin/python -m graph.integration_baseline.rollout_stop_clock \
  --mode smoke --config graph/integration_baseline/rollout_config.json
```

Each invocation gets a new ExperimentLogger UUID and writes local metadata,
JSON/NPZ, graph/hidden/action/controller traces,20FPS MP4/PNG, source snapshot/diff,
verification JSON and `experiments/<uuid>/stop_the_clock_rollout.html` without a
CDN. W&B is disabled; nothing is uploaded. Random/identity-normalizer smoke
results have `ROLLOUT_NOT_EXECUTED` and no learned-policy success rate.
Failed initialization/runtime contracts stop execution and retain error status;
remaining planned seeds are explicitly `NOT_EXECUTED`. Metrics unavailable from
the current instrumentation (collision/task-failure) stay null.

### Receive a completed immutable checkpoint bundle

Do not run these commands against an active training directory. After training
completion or receipt of an explicitly safe immutable snapshot, place these
**original files**, preserving relative paths, in an independent transfer bundle:

- `training/best.json`, the selected `training/checkpoints/<name>.pt`, and its
  immutable `<name>.json` checkpoint receipt.
- Original `config.yaml`, `robot_state_schema.json`, `expert_action_schema.json`,
  `dataset_manifest.json`, `split_manifest.json`, `vision_manifest.json`,
  `source_hashes.json`, and `executed_conditions.json`.
- `vision_runtime.json` describing the Vision weights and SAM2 source actually
  used by training. The helper below inventories the local cache/source. Run it
  in the original completed training environment with an **external bundle**
  destination, and verify its identity against that run's recorded source and
  weight revisions. A later inventory alone cannot prove historical provenance.
- The original training execution UUID. GPU dataset UUID and training UUID may
  differ. Preserve the effective training source inventory when a recorded
  execution revision was used; an unverified revision/missing source blocks
  evaluation rather than enabling an optimization.

```bash
# Completed training environment only; writes to an independent bundle.
HF_HUB_OFFLINE=1 .venv/bin/python -m graph.integration_baseline.rollout_checkpoint \
  --write-runtime-identity --output /absolute/transfer/bundle/vision_runtime.json

# Evaluation host, after receiving the entire immutable bundle.
.venv/bin/python -m graph.integration_baseline.rollout_checkpoint \
  --bundle /absolute/transfer/bundle --source-run-id ORIGINAL_TRAINING_UUID \
  --output /absolute/transfer/bundle/checkpoint_manifest.json

# Fix the exact original frozen test12 reset groups and fresh held-out40 seeds.
.venv/bin/python -m graph.integration_baseline.rollout_stop_clock \
  --mode prepare-seeds --config graph/integration_baseline/rollout_config.json \
  --checkpoint-manifest /absolute/transfer/bundle/checkpoint_manifest.json \
  --seeds-manifest /absolute/transfer/bundle/seeds_manifest.json

# Explicit trained-policy evaluation; never started by smoke or a training poll.
.venv/bin/python -m graph.integration_baseline.rollout_stop_clock \
  --mode evaluate --config graph/integration_baseline/rollout_config.json \
  --checkpoint-manifest /absolute/transfer/bundle/checkpoint_manifest.json \
  --seeds-manifest /absolute/transfer/bundle/seeds_manifest.json
```

The loader hashes every bundle member, checks Best pointer/receipt/progress and
the checkpoint's config/dataset/split/Vision identities, validates exact model,
robot/action schemas and train-only normalizer statistics, then strictly loads
FP32 weights. It loads verified bytes with `weights_only=True` and a fixed
allowlist for the existing NumPy RNG types, never unrestricted pickle. No
optimizer, scheduler or RNG restoration is needed. Partial/corrupt snapshots,
missing metadata, schema/source mismatch or different Vision weights are fatal.
Evaluation also compares installed simulator versions/controller semantics to
the original collection conditions. Only the host-specific URDF installation
directory is canonicalized to its filename; controller numeric values stay exact.

`rollout_seeds.template.json` is intentionally not a valid evaluation manifest
before the original split arrives. Default smoke seeds39000/39001 and timing
seed39002 are excluded. Held-out candidates41000..41039 exclude the original
collection range30000..30159, prior evaluation/pilot2000..2029, smoke29000..29003,
and every supplied dataset seed/reset group. The frozen-test12 and held-out40
groups get separate success rates and two-sided Wilson95% intervals; errors or
unexecuted seeds prevent a complete-group success rate. Do not tune on these
test seeds. Any seed allocation change must be frozen before the first evaluation.

See [Task07 implementation and measured verification](../../docs/visual-graph-integration/07_stop_the_clock_rollout.md).

### 3D graph viewer for saved rollouts

Task07 saves the exact per-observation graph tensors passed to the model as
`episodes/<episode>/graph_XXXX.npz`, and same-time TCP world poses in
`observations.npz`. The read-only frontend in [`visualization/`](../../visualization/)
browses saved episodes without rerunning Vision, the simulator, or the policy:

```bash
.venv/bin/python -m visualization.server
```

Open `http://127.0.0.1:8765/`. The separate list page shows compatible saved
episodes as large cards with search and success/failure/unknown filters. Select
one to open its graph viewer; use the back link to return to the list. Use the
EE relative / world absolute toggle to switch displayed node positions. World
positions are computed from the matching observation's TCP world pose; the
reader checks the first three transformed robot nodes
against the saved world robot geometry. Drag to rotate, Shift-drag/right-drag
to pan, scroll to zoom, and use the timeline or play button to inspect time.
Node and edge type filters reduce clutter; clicking a node shows both positions,
the original `node_input` F5/availability/validity, source index and directed
degree. The original F5 and `edge_attr` remain EE-frame model inputs in the
inspector. The viewer draws bidirectional pairs as one line for clarity while
reporting the saved directed edge counts. This is visualization of saved inputs
only; it does not recalculate graph construction or establish model correctness.
Frames load on demand. See [frontend usage and status semantics](../../visualization/README.md).

### Checkpoint smoke and explicitly selected evaluation groups

`--mode smoke --checkpoint-manifest <manifest>` runs the same small, excluded
smoke protocol with the strictly loaded trained weights and train normalizer.
It checks the original simulator conditions and still reports no performance
success rate. Without that optional manifest, smoke uses random weights.

An evaluation config may set `"evaluation_groups": ["heldout"]` to run only the
already frozen held-out group. The complete original seed manifest is still
validated; excluded frozen-test seeds are listed separately as `NOT_EXECUTED`.
This supports a held-out run when a different host cannot reproduce the original
frozen reset hashes. It does not waive reset checks, reinterpret excluded seeds
as failures, or select seeds after observing policy outcomes. Omitting this key
retains the default frozen-test12 plus held-out40 protocol.

The offline checkpoint inventory requests only the eight DINO inference assets.
New Hub versions therefore do not require an unused README or duplicate
`pytorch_model.bin` in a safetensors-only cache. Weights/source checks remain strict.

For the actual Vast.ai57640710 execution, independent Gymnasium1.3.0 environment,
CUDA/Vulkan startup, selected-group rationale and results/status, see
[Task07 Vast execution](../../docs/visual-graph-integration/07b_stop_clock_vast57640710.md).

### Fixed-checkpoint Expert grasp diagnostic

`grasp_diagnostic.py` compares the actual `direct_train.evaluate()` T32 path with
the existing Rollout `infer`/`EpisodeStream.step` path on four preselected stored
Train/Validation episodes. It checks exact Graph/Robot76 inputs, identical past
hidden and independent hidden carry, then compares Expert Action7 around
reach/descend/close. It also replays the first64 stored inputs from41005 against
the original recorded action/hidden. No new Vision, simulator steps, training,
normalizer fitting or model changes are performed. The fixed checkpoint,
selection, FP32 tolerance and GPU memory budget are in `grasp_diagnostic_config.json`.
See [the measured diagnostic and CLI](../../docs/visual-graph-integration/07c_fixed_checkpoint_grasp_diagnostic.md).

### Live close-window intervention diagnostic

`grasp_intervention.py` is a separate, opt-in mixed-control experiment. For each
fixed Train/Validation seed, four fresh real Expert approaches build live Vision
and GRU history. Only the eight close actions differ: Expert zero-arm/full-close,
model/model, model/full-close, or zero-arm/model. It checks identical paired
prefixes including the first close inference, and never restores another trial's
state or resets hidden at takeover. All conditions then use the same robot-only
8cm lift target and full-close gripper for20 lift/20 hold steps. Later full closing
can recover a weak initial close, so close-end contact is recorded separately.
GT pose/contact are outcome diagnostics; they do not enter model inputs.

```bash
python -m graph.integration_baseline.grasp_intervention \
  --config graph/integration_baseline/grasp_intervention_config.json \
  --checkpoint-manifest <immutable_bundle>/checkpoint_manifest.json \
  --output-root <independent_diagnostic_experiments>
python -m graph.integration_baseline.grasp_intervention_report <diagnostic_run>
```

Run on the existing CUDA evaluation environment; this diagnostic does not install
dependencies. For CUDA/Vulkan bootstrap on Vast, see the recorded invocation in
[the intervention report](../../docs/visual-graph-integration/07d_live_grasp_intervention.md).
These hybrid trials are excluded from policy evaluation and do not change the
closed-loop rollout policy, training, normalizer or production defaults.
