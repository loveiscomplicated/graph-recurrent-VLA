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
