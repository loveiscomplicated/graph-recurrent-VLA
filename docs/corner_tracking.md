# Tracked corner completion

This workflow manages track birth, confirmation, missing states and retirement.
The separate [detector-only association workflow](detected_corner_gnn.md) retains
every observed track and has its own evaluation protocol. Data preparation is
described in [data collection](data_collection.md#full-episodes-for-tracked-corner-completion).

## Tracked corner completion before GNN

This Graph-team feasibility experiment predicts the displacement of a previously
seen individual corner from its **last accepted observed world position**, then
passes observed and admitted predicted corners into a source-aware GNN to produce
the existing Object21D. It does not generate never-seen corners, establish a
fixed cuboid-corner numbering online, change the simulator, or define a cross-team
model contract. Mamba and causal Transformer are alternative backbones.

### Run and prerequisites

Python 3.11, Torch, NumPy, SciPy and PyYAML are required. Run from the repository root:

```bash
# A tiny CPU fixture checks plumbing; it is not an accuracy benchmark.
WANDB_MODE=disabled python main.py --config graph/configs/temporal/corner_tracking/smoke.yaml

# Set source.family_dataset to a contiguous full-frame vertex-family dataset first.
WANDB_MODE=disabled python main.py --config graph/configs/temporal/corner_tracking/prepare.yaml

# Set data.root to the emitted prepared dataset directory first.
WANDB_MODE=disabled python main.py --config graph/configs/temporal/corner_tracking/compare.yaml

# Set checkpoint and output.root first; saved model/calibration/config are restored.
WANDB_MODE=disabled python main.py --config graph/configs/temporal/corner_tracking/evaluate.yaml
```

The comparison runs seeds 0/1/2 sequentially. Use a copied recipe with
`trainer: training.gnn.temporal.corner_tracking:train`, one `architectures` entry
and one `seeds` entry to launch an individual run. All effective defaults and
supplied overrides are saved. Optional `base_config` is a repository-root-relative
YAML path with one inheritance level, used by the smoke recipe.

The real adapter is documented in [data collection](data_collection.md#full-episodes-for-tracked-corner-completion).
Existing frozen GNN/temporal checkpoints cannot initialize this different input
representation. Fresh GNN weights are used. No full real-data training or CUDA
execution is implied by the synthetic workflow.

### Memory and predict-before-update order

Each runtime track stores its internal ID, object type, observed anchor XYZ,
anchor frame, observed robot state at the anchor, observation quality, lifecycle
status, and its own temporal cache/last hidden representation. The ID indexes
memory; it is never a learned numerical input or a GraphBatch semantic ID.

At frame t:

1. Read history through t-1 and current observed robot state. Predict total world
   displacement from the fixed anchor and diagonal position variance.
2. Associate predicted positions with current detections, allowing no-match.
3. Accept reliable observations and update their anchors. Do not reset their
   temporal memories. Unmatched anchors stay fixed; predictions never become
   observed anchors.
4. Construct the completed graph, rebuild directed edges/distance from its actual
   coordinates, and compute Object21D.
5. Commit exactly one observed/missing token for each remaining track. The current
   measurement is unavailable to its own frame's prior prediction.

The record encoder consumes 42 features: TCP-relative observed XYZ (3), observed
flag (1), quality/quality-valid (2), object-type one-hot (3), observation age (1),
robot features (16), same-object context (16). Missing XYZ is zero with the flag
false, while robot state/age/context still advance. Robot features are observed
TCP XYZ/wxyz (7), finger positions (2), frame-to-frame TCP translation/relative
rotation-vector changes (6), and a saved-frame interval of one (1). Timestamp-free
data does not provide physical velocities or elapsed seconds.

Context is a shared 5→32→16 MLP over each currently retained point's TCP-relative
XYZ/quality/quality-valid, averaged separately by object type. Empty context is
zero. It is computed after masking and recorded for the next prediction, so a
hidden point cannot survive in another corner's current context.

The motion head combines the last 128D hidden representation with a 24D current
query: anchor-relative-to-current-TCP XYZ (3), current observed TCP/fingers (9),
anchor-to-current TCP translation/relative rotation (6), anchor age (1), object
type (3), anchor quality/valid (2). Both backbones output displacement XYZ (3)
and log diagonal variance (3). World position = observed anchor + predicted
displacement; it is not an accumulation of predicted frame-to-frame displacements.
Log variance is bounded in normalized units [-8,4]. Coordinates use a 0.05m scale,
finger positions 0.04m, and age 32 saved frames.

Logical sequence shapes are records [B,K,T,42], queries [B,K,T,24], displacement
and variance [B,K,T,3]. The online implementation steps ragged tracks individually
with shared weights; it does not assume a fixed K or detection ordering. Mamba
stores per-layer convolution/SSM state; Transformer stores past normalized
attention states, with causal positions starting at track birth. Full episode
history is retained; birth/episode boundaries reset state, anchor updates do not.

### Initial lifecycle and association rules

These are experimental starting settings, not measured optimal thresholds.
All ages below count frames since the last **accepted** observation.

| State | Rule | Graph use |
| --- | --- | --- |
| tentative | Create from an unmatched point with no competing existing hypothesis | Real observation only |
| observed | Confirm with 2 accepted hits, including birth, within 3 frames | Real observation |
| missing | Confirmed track lacks an accepted observation; age 1–8 | Conditional predicted support |
| dormant | Age 9–32 | No predicted graph support; continue missing-token memory updates |
| retired | Age 33, unconfirmed after the 3-frame birth window, or episode end | Release memory; IDs are not reused |

Object types must agree. A calibration-only constant-velocity replay determines
age-specific distance gates: Q99 anchor-motion prediction error plus Q99 observed
detector coordinate error, with a 1mm numerical floor. A complete rectangular
assignment includes no-match dummy columns at cost 1. Distances are normalized
by their gates. A selected match must be best in both row and column and at
least 20% cheaper than every competing match/no-match. Exact competing ties are
held rather than broken using array order. These gates remain fixed within a run.

Unaccepted points inside an existing hypothesis gate cannot immediately create
duplicate tracks. They are anonymous observed graph evidence; competing predicted
nodes are suppressed for that frame. Held measurements do not refresh anchors,
confirmation counts or last-observation times. Missing tracks need one accepted
match to recover; dormant tracks need two consecutive accepted hypotheses, with
the first kept pending and the latest confirmed point becoming the new anchor.
Reidentification is a hypothesis, not a guarantee under symmetric/fully hidden
motion. Observation quality being unknown is not itself grounds for deletion.

A missing track can emit a predicted graph point only at age ≤8 with a calibrated
95% Euclidean error radius ≤5mm and no competing ambiguous observation. Calibration
fits Q95(error / predicted-std-norm), separately by object type and age bins 1–2,
3–8, 9–16, 17–32. Each bin needs 32 samples; unsupported bins emit no predictions.
The resulting scaled XYZ uncertainty, observed flag and age are the five extra
GNN features. Report actual coverage: the 95% interpretation is empirical and
must not be assumed to hold on a new population.

`CompletedGraph` keeps provenance separate from raw observed graph semantics.
Its internal GNN adapter counts included support, including predictions; those
counts/anchors are not observations or physical visibility labels. See
[graph input](graph_input.md#optional-features-for-completed-corner-graphs).

### Paired masks and supervision

Each training episode gets one mask mode: unchanged 25%, single corner 40%,
same-object corner group 25%, whole object's detections 10%. Conditional gap
lengths are 1–2 frames 20%, 3–8 frames 40%, 9–16 frames 30%, 17–32 frames 10%.
Whole-object removal can hide other points, but predictions/supervision remain
limited to previously established tracks. Partial group masks choose 2..all
eligible same-object corners.

Eligibility is determined before the mask begins: a shared constant-velocity
tracker must already have confirmed the corner, and its physical certificate
must occur in at least two of the preceding three frames. This baseline replay
uses no GT for association; certificates only select a labelled augmentation.
An individual learned tracker can still fail to confirm that corner; no oracle
track/anchor is injected to repair it. Requested length is sampled first; a
start with sufficient episode length is preferred. Short-episode truncation and
unavailable eligible masks are recorded explicitly.

Masks remove detections before association, context pooling and graph formation.
All certified duplicates and same-object detections within 3mm of the selected
GT reference are removed. Object masks remove all points of that object kind.
No mask reason, original count, certificate, removed-point quality or old edge
enters model inputs. Raw RGB-D and source graphs are not modified. This is
point-removal augmentation, not a claim to render physically realistic occlusion.

Offline references/certificates supervise positions and audit identity only.
The node target is GT current position minus the **same noisy observed anchor**
used at inference; the loss is a diagonal Gaussian NLL in the configured position
unit (10mm by default).
UNKNOWN identity is excluded and counted. Certified identity changes remain
tracking failures; they do not repair online IDs or provide false coordinate
targets for a mismatched anchor. Existing `object21_loss` supervises the GNN.

### Matched training, outputs and limitations

Both candidates use width 128, one temporal layer, dropout 0; Transformer has
4 heads and feedforward width 256, Mamba d_state=16/d_conv=4/expand=2. Common
context/record encoders, motion head and fresh GNN start with identical weights,
verified by SHA256. Backbone parameter counts differ and are recorded. GNN defaults
are hidden 64, two layers, existing geometry aggregation and multiseed readout.

The initial budget is five node-NLL-only pretraining epochs with GNN parameters
frozen, followed by twenty joint epochs of Object21 loss + 0.1 node NLL. AdamW
uses 0.001 for both model groups, weight decay 0.0001, gradient clipping 1. Each
complete episode is one optimizer step; state and gradients carry through the
episode, not across episodes. Pretraining episodes with no certified node loss
do not update weights. Uncertainty is recalibrated after each epoch exclusively
on calibration parents. The best **joint** checkpoint minimizes development
validation Object21 loss. Equal epoch budgets do not imply equal compute or
parameter counts.

Masks/shuffle are derived from seed, episode and epoch using a stable hash.
Validation/calibration masks remain fixed. Training/validation-mask hashes and
common-initialization hashes must agree between candidates in each seed.
The comparison does not select a backbone on reserved test data. Existing source
development validation has already been used in earlier studies; results here
are development evidence, not an independent final test.

Each tracked run writes effective config, dataset identity, epoch history, exact
mask ledger, atomic best/last checkpoints, per-frame NPZ predictions and JSON
track decisions/node errors. Evaluation includes natural inputs, fixed synthetic
masks and masks extended to episode end. Metrics reuse existing physical Object21
scoring, including relative/lateral/axial errors and size/rotation components.
Reports include age-binned node error, physical-reference ID switches,
within-track identity changes, duplicate confirmed tracks, false reconnections,
recovery delay/unrecovered returns, adoption fraction and certified-missing-corner
coverage. Certification-based coverage is not a physical visibility metric.

`comparison.json` verifies aligned frame IDs/GT, averages seed **errors** per frame,
and reports Mamba-minus-Transformer differences with a paired parent bootstrap
when there are at least two parents. It does not average poses before scoring.
Optional `evaluation.ablations=true` also evaluates current-only, last-observation
hold and constant-velocity completion on each candidate's same trained GNN.
These are input/predictor interventions, not separately trained baseline GNNs.

For epoch-boundary resume, use the single-run entrypoint with `resume_checkpoint`
pointing to `last.pt`, retain the architecture/seed/data/model/tracking/loss and
optimizer settings and the implementation fingerprint, and increase
`training.joint_epochs`. Optimizer, selected
best weights, calibration, RNG and mask/history ledger are restored into a new
ExperimentLogger run, retaining the source checkpoint sidecar identity.

For episode-level progress on a new run or checkpoint resume, invoke the same
config through the separate observer:

```bash
WANDB_MODE=disabled python -m tools.corner_tracking_progress --config path/to/config.yaml
```

The observer logs mask preparation, training forward-completed episode counts,
calibration and evaluation counts, and stage-local elapsed time. It reports after
every ten completed episodes or after 30 seconds at the next episode completion;
the current episode's backward/optimizer step may still be pending. Observer
version and intervals are recorded in the effective config. The observer leaves
the model implementation fingerprint unchanged so a compatible checkpoint can
be resumed; it cannot be injected into an already running Python process.

`tools.corner_tracking_telemetry` can separately record process CPU/RAM, GPU
utilization and the latest completed checkpoint epoch for existing processes.
It does not expose their current episode or infer progress from utilization.

The default Mamba backend is `torch_reference`. The streaming training recurrence
is functional Torch for autograd, even when parameters come from an optional
`mamba_ssm` module; it does not claim fused official-kernel training throughput.
Transformer streaming reprojects stored past features and is not an optimized
KV-cache implementation. Models can run on a supported selected Torch device,
but association is CPU NumPy/SciPy with device synchronization. No online latency
or memory-efficiency advantage has been established. Shared ExperimentLogger and
artifact sidecars are reused; training never uploads artifacts.

For independently launched parallel runs, set the six completed run directories
in `graph/configs/temporal/corner_tracking/aggregate.yaml` and invoke it through
`main.py --config`. The aggregator rejects incomplete/failed runs, mismatched
data/config/source identity, checkpoint ownership, initial weights, seed-paired
mask manifests, validation frames or targets before computing the same paired
comparison. It reads validation episodes only. See
`tests.test_corner_tracking_aggregate` for matched and rejected examples.

### Verification

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest tests.test_corner_tracking \
  tests.test_gated_mpnn tests.test_graph_builder -v
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_experiment_logger tests.test_experiment_artifacts -v
```

Tests cover reordered detections, confirmation/expiration, ambiguity/duplicate
suppression, age boundaries, pending dormant reconnection, calibrated admission,
anchor stability, shared initialization, current-observation causality, GT input
separation, full-episode backward, gradient propagation through generated graph
coordinates, paired masking, checksummed datasets and train/save/re-evaluate.
The synthetic checks establish plumbing only; real-corpus tracking accuracy,
uncertainty calibration, full training and CUDA execution remain unmeasured.

On 2026-11-08 KST, macOS/Apple M4 Pro, Python 3.11/Torch 2.14.1:
108 model/shared regression tests ran (106 passed, two skipped: the opt-in legacy
server optimizer test and official CUDA Mamba parity), and all 36 logger/artifact
contract tests passed in a separate process. The final corner suite's 22 tests
passed after the data validation changes. The real cache/raw adapter was exercised
with native-shaped synthetic files; no real corpus was prepared. Resume matched
uninterrupted weights exactly. Both backbones also completed a five-frame
synthetic episode forward/backward on MPS. The existing three-frame CPU simulator
RGB-D streaming regression passed; the new tracked model has not been evaluated
in live simulator rollouts.

The real root-CLI smoke completed one seed, one pretraining epoch and one joint
epoch per backbone, including admitted predicted nodes, best/last checkpoints,
three evaluation scenarios, counterfactual baselines and paired aggregation.
See [verification metadata](../outputs/corner_tracking_smoke/verification.json),
[paired synthetic comparison](../outputs/corner_tracking_smoke/comparisons/3f858a64211040efa3c4728877d23625/comparison.json),
[model regression log](../outputs/corner_tracking_smoke/model_regression_tests.log),
[infrastructure log](../outputs/corner_tracking_smoke/infrastructure_tests.log),
and [final corner tests](../outputs/corner_tracking_smoke/corner_tests.log).
Run simulator/model and infrastructure suites in separate processes: loading
ImageIO through simulator tests can interact with the existing logger test's
global `importlib.import_module` mock during unittest warning-registry inspection.
The initial combined-process attempt encountered this unrelated test-isolation
error; both isolated suites passed without modifying operational infrastructure.

## Corner tracking profiling and execution efficiency

The three diagnostic tools share `tools.corner_tracking_inputs.load_training_episode`
to select the first complete training episode, reject paths outside the prepared
dataset and verify its NPZ checksum before loading. They reuse `utils.timer.Timer`;
the GPU tools retain synchronization around their measured optimizer step.

Set `data.root` and an available CUDA device in a complete corner-tracking recipe:

```bash
python -m tools.benchmark_corner_tracking --config /path/to/cuda_recipe.yaml \
  --phase joint --frames 0 --repeats 2 \
  --output outputs/corner_tracking_profile/benchmark.json
python -m tools.profile_corner_tracking --config /path/to/cuda_recipe.yaml \
  --architecture transformer --frames 20 --node-only --no-diagnostics \
  --output outputs/corner_tracking_profile/transformer
```

Both tools require CUDA and reject CPU/MPS before reading episodes or building
models. The benchmark uses the first configured architecture and seed; zero
frames means the complete episode. The profiler takes an explicit architecture,
uses seed 0 and limits the episode to `--frames` (at least three). Existing
calibration, loss, gradient and optimizer settings remain unchanged. Each
benchmark repetition restores the same initial weights and creates a fresh
optimizer. Measurements describe these diagnostic calls under their actual
GPU load, rather than full training throughput.

Eligibility profiling runs on CPU and needs only the prepared dataset:

```bash
python -m tools.profile_corner_eligibility --data-root /path/to/prepared_dataset \
  --output outputs/corner_tracking_profile/eligibility.json
```

The native-shaped source fixture is in `tests/fixtures/corner_tracking.py`.
Serial/parallel preparation tests build their own inputs from it and check
array equality, immutable source bytes, incomplete-run manifests and rejection
of noncontiguous episodes. They no longer call another test method to arrange
their inputs. Run these checks and the profiling input contracts with:

```bash
.venv/bin/python -m unittest tests.test_corner_tracks_parallel \
  tests.test_corner_tracking_tools tests.test_corner_tracking_diagnostics -v
```

On the cleanup host, profiling tests use CPU reference models and mock CUDA
timing/memory calls for orchestration checks; actual CUDA execution and GPU
timing were not performed. The historical CUDA measurements below remain the
original evidence.

The 2026-11-09 [tool and fixture comparison](../outputs/workflow_tool_verification/equivalence.json)
matched 348 arrays from 36 native-shaped fixture archives exactly against the
preceding dirty snapshot. CPU reference checks also matched benchmark losses
and gradient norms, profiler predictions and tracking, and all 93 saved updated
weight tensors. CUDA calls were mocked; elapsed-time values were not compared.
The combined launcher, fixture, tracking, profiling and Timer suite ran 64 tests:
61 passed and three existing opt-in tests were skipped. See the local
[test log](../outputs/workflow_tool_verification/tests.log).

The corner tracker batches motion predictions and memory updates across live
tracks while retaining separate state per track. Transformer tracks with equal
history lengths share a batch; Mamba states are stacked by shape. Association
still runs on the CPU, with predictions and uncertainty copied once per frame.
`CornerTracker(..., batch_tracks=False)` retains the scalar path for validation.
During training, diagnostic exports are disabled; evaluation retains them. The
node-only pretraining phase skips graph construction and the frozen GNN because
its loss uses only the motion model. Eligibility preparation likewise skips the
unused graph while preserving the same causal assignments.

On 2026-11-08, one real 105-frame training episode was profiled on an RTX 3090
while six existing jobs shared the GPU. These are single-run wall times with
cProfile overhead and contention, including forward, backward and one optimizer
step, not full-epoch throughput or evidence of final model quality:
The saved summaries call this a partial episode because of an old profiler
label; the source episode contains exactly 105 frames.

| Phase | Backbone | Previous | Optimized |
| --- | --- | ---: | ---: |
| Node-only pretraining | Mamba | 25.50 s | 4.49 s |
| Node-only pretraining | Transformer | 25.90 s | 5.10 s |
| Joint training | Mamba | 28.77 s | 12.42 s |
| Joint training | Transformer | 28.75 s | 13.05 s |

All compared runs used the same initial weights, episode and node supervision
count. The measured losses matched; the CUDA tests also compare tracking,
predictions and gradients for both backbones. Floating-point reduction order and
CUDA nondeterminism can change individual optimizer updates, so identical
training trajectories are not guaranteed. On this episode, two CPU eligibility
runs took 0.246–0.266 s before and 0.115–0.116 s after the change, producing the
same eligibility hash. The original implementation made thousands of tensor
conversions and hundreds of per-track CPU transfers in a 20-frame sample;
profile times overlap and should not be summed. Full evidence is retained in
[profiling outputs](../outputs/corner_tracking_profile/profiles/), including
[original Mamba profile](../outputs/corner_tracking_profile/profiles/baseline_mamba_1/profile.txt),
[joint Mamba](../outputs/corner_tracking_profile/profiles/full_optimized_mamba/summary.json),
[joint Transformer](../outputs/corner_tracking_profile/profiles/full_optimized_transformer/summary.json),
[pretraining Mamba](../outputs/corner_tracking_profile/profiles/pretraining_optimized_mamba/summary.json),
[pretraining Transformer](../outputs/corner_tracking_profile/profiles/pretraining_optimized_transformer/summary.json),
and [eligibility comparison](../outputs/corner_tracking_profile/profiles/eligibility_optimized.json).
The server's original six runs continue using their original source snapshot;
the optimized snapshot is separate. Checkpoint implementation fingerprints are
strict, and migration between these source versions has not been implemented.

### One-frame association recovery

When a one-frame-old track's predicted position has no same-type detection
inside its calibrated gate, the tracker tries its last observed position if a
same-type detection is within that same gate. Assignment and ambiguity checks
are still shared across tracks. The model prediction and gradient are retained
for motion supervision; the fallback does not reconnect tracks across longer
gaps or replace the position supplied to graph completion.

During the November 2026 server run, Transformer seeds 0 and 1 had median
predicted displacements of about 8.5 cm and 7.5 cm at their first checkpoints,
while the one-frame gate was about 0.8 cm. On two natural real episodes, neither
checkpoint produced any supervised nodes with the old association. The fallback
produced 888/1397 and 954/1592 supervised nodes, respectively. It also exposed
substantial duplicate-track counts, so this result is recovery evidence, not a
claim of good tracking quality. A small CPU optimizer-step check on both
checkpoints produced finite losses and updates. The resumed recovery runs use a
separate source snapshot and record the original checkpoint hash and the changed
implementation hash; the four ongoing original-source runs are not a matched
training comparison with these recovered runs. A fair Mamba/Transformer result
requires running both architectures under the same revised association settings.

### Compiled corner tracking execution

The CUDA preset is
`graph/configs/temporal/corner_tracking/compare_compiled.yaml`. Set the prepared
`data.root` in its base comparison recipe and run
`python main.py --config graph/configs/temporal/corner_tracking/compare_compiled.yaml`.
PyTorch CUDA, Inductor/Triton and a working C++ compiler are required. The
`execution` settings are recorded in effective configs and included in matched
comparison and resume checks. Default execution remains eager with single-frame
GNN calls; the CUDA preset enables compilation, fused AdamW and graph batches of
32, with `float32_matmul_precision: highest`.

Observed robot/query features are prepared together on the CPU before transfer.
Learned context pooling uses dense membership weights instead of CUDA boolean
indexing. Supervised node NLL is evaluated in batches per frame with the same
mean over all supervised nodes. Complete-edge indices are cached by node count.
Graphs from independent frames are padded and evaluated together; edge indices
are offset per frame and cannot connect separate samples. Graph validation runs
on the resulting batch, and chunk losses are weighted by their frame counts.
The tracker still predicts and updates in temporal order, and generated positions
retain their gradient to the motion model. GNN output does not feed back into
tracking. This grouping relies on the matched recipe's zero dropout.

Tensor-only encoders, heads, context pooling, Mamba steps and GNN message layers
use `torch.compile` with dynamic shapes. Transformer attention stays eager:
compiling its streaming attention with changing history lengths failed a backward
stride assertion on the installed PyTorch 2.14.1 CUDA build. Transformer layer
normalization and FFN regions are compiled. CUDA graphs are excluded because
multiple recurrent forwards retain tensors until the episode-level backward.
Compiler cache reuse reduces subsequent startup costs, but cold compile time is
reported separately from warm step time. The installed compiler's error is
preserved in the server benchmark log.

On the RTX 3090 with six other training jobs sharing the GPU, the same real
105-frame episode was repeated from identical initial weights. Median warm
forward/backward/optimizer-step times over two repetitions were:

| Phase | Backbone | Corrected eager baseline | Optimized compiled execution |
| --- | --- | ---: | ---: |
| Pretraining | Mamba | 4.06 s | 1.55 s |
| Pretraining | Transformer | 4.57 s | 2.32 s |
| Joint | Mamba | 11.55 s | 2.73 s |
| Joint | Transformer | 12.20 s | 3.51 s |

The baseline already includes the one-frame association recovery, and fused
AdamW is used in both benchmark variants. Supervised-node count was 797 for
all runs; losses and gradient norms agree within floating-point tolerance.
CUDA tests additionally compare nonzero motion/GNN gradients, track assignments,
predictions and evaluation with compilation and frame batching enabled. The
first call of each final benchmark took 8.7–15.0 seconds with a partially warmed
disk compiler cache. These are single-episode measurements under contention,
not measured full-training completion times or a model-quality comparison.
Evidence: [benchmark reports](../outputs/corner_tracking_execution/profiles/)
and [CUDA verification log](../outputs/corner_tracking_execution/logs/compiled_verified.log).

The fresh matched server cohort uses
`/workspace/corner_tracking_compiled_experiment`, with six jobs in tmux session
`corner_tracking_train`. All architecture/seed combinations start from fresh
weights with the same source fingerprint, corrected association, execution
settings, complete episodes, paired masks and 5+20 epoch budget. The previous
mixed-source cohort is stopped and retained under
`/workspace/corner_tracking_experiment`. The prepared dataset is reused through
a data-directory link. A launch manifest records source hashes and common
execution settings; all six outputs and their experiment logger records are
separate from the previous cohort. Progress lines include `episode_loss` and
`supervised_nodes`, and the common aggregation runs only after all six succeed.

The episode progress observer reports the mean Euclidean position error in
millimetres over the same identity-certified prior nodes used by the node loss,
together with their count. If there are no supervised nodes, the error is N/A.
The stage ETA extrapolates the current stage's completed episodes and excludes
later stages such as calibration and validation. These detached measurements do
not alter the loss or gradients. For an existing run, the logging-only rollout
change needs an explicitly recorded checkpoint migration because implementation
fingerprints include the rollout source; the original checkpoint is preserved.

### Tracking failure audit

Run `python main.py --config graph/configs/temporal/corner_tracking/diagnose_tracking.yaml`
after setting a frozen checkpoint path. The CPU diagnostic keeps weights and
validation masks fixed, uses GT only after online decisions, and records source
and checkpoint hashes through `ExperimentLogger`. `episodes: 40` selects the
entire development validation split. It separates ID changes, track retirement,
association competition and graph-admission failures. It also recomputes graph
admission with different age/radius limits without changing association or
memory updates. This isolates admission from the current coupling between
`graph_max_missing_frames` and dormant reconnection behavior.

A frozen Mamba seed 0 epoch 3 and Transformer seed 0 epoch 2 were each replayed
on all 40 validation episodes / 4102 frames. Their baseline ID, duplication,
reobservation and false-reconnection counts and missing coverage exactly
reproduce the saved validation summaries. These are within-checkpoint policy
probes on development validation, not a final architecture comparison.
Evidence: [Mamba report](../outputs/corner_tracking_diagnostics/analysis_bundle/mamba_diagnostics.json),
[Transformer report](../outputs/corner_tracking_diagnostics/analysis_bundle/transformer_diagnostics.json).

The current `id_switches` counter includes fragmentation after retirement. Of
Mamba's 242 switches, 103 followed retirement of the previous track, 95 moved to
another track while the previous track remained alive, and 44 involved an
existing track changing its physical corner. The corresponding Transformer
counts are 152, 53 and 69, totaling 274. The separate
`track_identity_changes` metric is 46 / 71; it need not match the third category
because not every physical identity change causes the ID-switch counter to fire.

The missing-reference denominator contains 24301 corner-frame opportunities:
4604 were within eight frames of the last certified detection, 9196 were 9–32
frames old, and 10501 were older than 32 frames. It measures missing certified
IDs, so uncertified observations are included alongside true occlusion. Of
Mamba's 20573 uncovered opportunities, 9960 followed retirement of the previous
confirmed track and 8857 exceeded the eight-frame admission limit; together these
explain 91.5% of exclusions. The five-millimetre uncertainty condition explains
497, or 2.4%, of exclusions. The analogous Transformer counts are 10206, 8821 and
798 out of 21027 exclusions. Retirement at age 33 deletes all model memory for
that track; no dormant identity archive currently exists.

| Fixed-weight graph admission | Mamba missing coverage | Transformer missing coverage |
| --- | ---: | ---: |
| Age <= 8, radius95 <= 5 mm | 15.34% | 13.47% |
| Age <= 16, radius95 <= 5 mm | 27.67% | 24.48% |
| Age <= 32, radius95 <= 5 mm | 45.73% | 38.80% |
| Age <= 8, radius95 <= 10 mm | 15.64% | 15.57% |

At age 32 / radius 5 mm, correct-ID predictions within 5 mm are 98.22% / 98.59%
of auditable adopted predictions. Known ID inconsistencies are counted as
incorrect. Predictions with no certified binding are excluded from that accuracy
number and reported separately: even at the current policy they comprise
1076/4844 and 1006/4300 of all generated graph nodes. These probes do not measure
the downstream GNN's accuracy after changing its support distribution.

Association currently ranks distance divided by the age-specific gate. Large
old-track gates can discount a distant hypothesis relative to a recent, accurate
track. Two tested remedies improve ID persistence but have tradeoffs:

| Association | Mamba ID switches / duplicate track-frames | Transformer ID switches / duplicate track-frames |
| --- | ---: | ---: |
| Current | 242 / 551 | 274 / 745 |
| Confirmed tracks observed <= 2 frames ago first | 155 / 432 | 186 / 1136 |
| Age-specific feasibility gate, common absolute-distance cost | 149 / 356 | 172 / 1269 |

Absolute-distance ranking also reduces physical identity changes from 46 to 29
and 71 to 35, and false reconnections from 71/642 to 33/389 and 89/610 to 66/655.
Transformer duplication increases, so neither ranking probe should replace the
production policy without explicit duplicate-hypothesis handling. Duplicate
track-frames are not the same as duplicate nodes admitted to the GNN.

The preferred next steps are to separate prediction-admission age from dormancy,
retain the calibrated 5 mm bound while testing longer admission, and jointly
validate association ranking plus suppression/management of duplicate hypotheses.
Longer identity preservation additionally needs a dormant identity archive,
geometry-based reidentification and matching training gaps; simply increasing
retirement age is insufficient. Geometric descriptors or an observed rigid-object
motion estimate can help distinguish same-type corners. The current coordinate
NLL does not directly train the discrete assignment to maintain physical IDs.
All six training runs remain on their original common settings during this audit.
