# Temporal GRU state estimation

The Graph team's temporal estimators consume frozen GNN outputs and observed
robot metadata. Run from the repository root with Torch, NumPy, PyYAML and
Matplotlib installed. Set checkpoint, cache and parent-manifest paths before use;
Pose recipes document optional raw robot observations.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/train.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/evaluate.yaml
```

Models and observation features live in `graph/temporal/`; data preparation in
`data_collection/temporal/`; training and evaluation in `training/gnn/temporal/`
and `evaluation/gnn/temporal/`. Geometry utilities remain in
`graph/pose_geometry.py`. Settings validation and frozen perception preparation
are explicit helpers in `training/gnn/temporal/config.py` and `frozen_source.py`.
The root entrypoint and each workflow's existing config schema remain supported.

- [Hole GRU](#gru-temporal-object-state-estimator)
- [Pose GRU](#joint-peghole-temporal-pose-estimator)
- [GRU insertion objectives](#gru-insertion-precision-objectives)
- [Base GNN and learned Gate](gnn_training.md)

## GRU temporal object-state estimator

Prerequisites: Torch, NumPy, PyYAML, Matplotlib, the exact published gated-MPNN
`best_weights` checkpoint, its extracted graph cache, and the explicit source
parent map/manifest described above. Run from the repository root after setting
`checkpoint`, `data.cache_dir`, and `split_manifest` in both configs:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/train.yaml
# Set gru_checkpoint in the evaluation config to outputs/temporal_gru/<run_id>/best.pt.
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/evaluate.yaml
```

Full CUDA training and large evaluation require separate user approval. Existing
GNN/rule results are preserved; each GRU invocation writes a new UUID directory
under `outputs/temporal_gru/` or `outputs/temporal_gru_evaluation/`. The root launcher
accepts a config path, with no additional workflow CLI overrides. Effective config,
seed, source code hashes, checkpoint/cache/manifest identity and library versions
are recorded through the existing `ExperimentLogger`. Publishing remains an
explicit `tools.publish_experiment` operation; these workflows never upload.

`graph/temporal/hole_gru.py` defines an independent state estimator:
observation23 → one causal GRU layer → Linear(hidden,3) → Hole world XYZ.
The default hidden size 64 has **17,283 trainable parameters**; hidden size 128
has **59,139**. Both include GRU biases and the XYZ head. There is no residual
addition of the current GNN position, forced cold-start output, manual update gate,
or Action7 prediction. The original vertex GNN and all state heads load strictly,
remain in eval mode with `requires_grad=False`, and run without gradients.
Only the new GRU/head is optimized. The original checkpoint file is not rewritten.

`graph/temporal/observations.py` owns the external feature contract
`hole_world_tcp_quality_v1`, in this exact order:

| Indices | Observation | Units/convention |
| --- | --- | --- |
| 0:3 | GNN Object21[10:13] + observed TCP | Hole world XYZ, metres |
| 3:6 | observed TCP world XYZ | metres |
| 6:14 | BOX count, present, missing, known-confidence fraction, confidence-present, mean, min, max | existing valid-node/confidence masks |
| 14:22 | the same eight HOLE statistics | unknown confidence is masked |
| 22 | all vertices missing | includes PEG presence in the existing graph contract |

Coordinates are divided by the fixed configured `position_scale_m` inside the
model; the XYZ head is scaled back to metres. Counts/confidence are kept as in
the learned gate. The gate's historical 30D MLP input additionally contains its
own previous estimate, innovation and initialization flag; these are internal
recurrence values, not new external sensors. Its external observations and their
quality extractor are shared with GRU; a contract test compares them directly.
At the original Hole-only GRU inspection, no temporal state-estimator Mamba path
was present; `recurrent/` contained a separate Action7 Mamba controller. The
observation23 contract remains available to estimators that use the same sensors.
Separately added temporal Mamba work is outside the GRU verification reported here.
GT position, GT visibility, previous GT and future frames never enter GRU inputs.

`data_collection/temporal/frozen_episodes.py` constructs complete episodes using the
same frozen inference as the learned gate (old imports remain valid). The existing
cache reader verifies sample identity/checksum and graph/target pairing. Frozen
inference sorts every array with one sample-ID permutation; the sequence dataset
checks shapes, finite values, unique IDs, consistent episode/frame metadata and
contiguous frames beginning at zero. It does not infer parent identity from names:
reuse `data_collection.temporal_gate_data` to produce an explicit source-parent
manifest. GRU train/validation use disjoint original GNN-train parents; the complete
original GNN validation split is reserved as temporal test. Original GNN test
parents and their siblings cannot enter temporal development. The original GNN
has already seen temporal development data, so this measures a downstream
estimator with a frozen pretrained perception model, not end-to-end generalization.

Training performs full-episode BPTT with one episode per optimizer step and
shuffles episode order only. Every episode starts with zero hidden state. GT Hole
world XYZ supervises every step, including frame zero and missing-vertex frames,
using coordinate-mean Huber (delta in metres, loss in m²) or L1 (loss in metres).
Reported epoch loss is frame-weighted. Validation disables gradients and selects
`best.pt`; the temporal test is not consulted during training. Gradient clipping
is configured. No teacher forcing, truncated history, padding or hidden-state
sharing is used. Entire frozen-output arrays remain in host memory, and each
complete episode's BPTT graph resides on the selected device; long episodes may
need more memory. Resume/optimizer restoration is not implemented in this first
experiment. Compare hidden 64/128 using development validation before final test.

Streaming inference accepts only current observation tensors and explicit hidden
state. `TemporalHoleGRU.forward()` accepts `[B,T,23]` and returns `[B,T,3]` plus
`[1,B,H]` hidden state; `.step()` accepts `[B,23]`. Call with hidden=None at every
episode boundary. `StreamingHoleEstimator` owns that hidden state and provides
`reset()` and observation-to-Object21/State38 assembly:

```python
from graph.temporal.hole_gru import StreamingHoleEstimator, TemporalHoleGRU
from graph.temporal.observations import observation_features

# Load the new checkpoint's model weights and exact gru config before inference.
estimator = StreamingHoleEstimator(model)
estimator.reset()  # mandatory for every new episode, including unchanged batch size
# frozen_object21[B,21], observed_tcp[B,3], robot15[B,15], freshness2[B,2]
result = estimator.step(frozen_object21, observed_tcp, observation_features(graph),
                        robot15, freshness2)
refined_object21 = result['object21']
state38 = result['state38']
```

`object21` changes only Hole [10:13] to predicted_world − current_TCP. All other
fields are copied from the current frozen GNN. `consistent_object21` separately
updates the Peg Endpoint−Hole vector [7:10] by old_hole − new_hole, preserving the
original estimated peg endpoint. `state38` concatenates caller-provided Robot15,
this consistent Object21, and caller-provided Freshness2. Robot/freshness are
neither fabricated nor derived from labels; existing caches lack these inputs,
so saved-data evaluation reports Object21/geometry ablations and tests State38
assembly using explicit supplied tensors. Freshness semantics remain caller-owned.

Evaluation fixes all three predictions before loading optional GT-derived
visibility diagnostics. GNN alone, unchanged rule memory (the existing five-node,
0.82 confidence, three-frame recipe), and GRU see identical saved frames. It saves
frame/episode mean, median/p90, first-frame failures, fixed bad-GNN-initialization
cohort recovery, harmful updates, per-episode position drift/error slopes and
quarter-episode error summaries. Missing HOLE and missing BOX+HOLE are observation
proxies, not GT occlusion. Long-run groups score complete missing runs of at least
`long_occlusion_frames`; long-age groups score only elapsed missing ages at least
that threshold. Optional visibility labels add the existing strong-occlusion proxy.
No diagnostic threshold controls hidden updates. Reobservation events report
before/return/end errors and first recovery within the following available run;
unrecovered cases have null latency. `events.json` retains individual run/recovery
records. Per-episode PNGs align all three errors, GRU/GT world XYZ and vertex counts.
The assembly ablation separately scores endpoint-vector error, endpoint preservation
and the inconsistent endpoint shift from changing Hole only.

Every evaluation checks full-sequence versus streaming predictions and final
hidden states for every episode (position atol=1e-6 m, rtol=1e-5; hidden
atol=1e-5, rtol=1e-4), reporting the maximum position difference. It rejects changed
source checkpoint, split manifest, feature contract, model settings or dataset
identity. Parameter counts and measured batch-one inference mean/median/p90 latency
are saved for GNN, rule-only CPU memory, GRU-only and GNN→quality→GRU pipeline.
Latency uses repeated resident input, warmup and accelerator synchronization,
includes Python/validation overhead and excludes cache IO, detection and robot
assembly. It is not a replay-throughput measurement. On CUDA it records post-warmup
allocator baseline, peak allocated/reserved bytes and peak incremental allocation
with both models resident. On CPU/MPS CUDA-memory fields are null, never zero.
Prediction NPZ is registered as trajectory, JSON as metrics and PNG as a local
output, respecting the existing publisher's artifact kinds.

CPU correctness checks:

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_temporal_hole_gru tests.test_temporal_gate tests.test_hole_memory \
  tests.test_gnn_temporal_memory tests.test_gnn_cached_data -v
```

On 2026-11-07 KST, the 41 tests passed, including 12 new GRU tests. They cover
sequence/streaming equivalence, causal prefixes, episode resets, missing-frame
backpropagation, trainable initialization, label isolation, shared gate features,
frame sorting/alignment rejection, parameter counts, geometry assembly, unchanged
frozen weights and a root-CLI one-epoch train/evaluate smoke using a tiny actual
checksummed graph cache and reduced VertexEdgeGatedMPNN. This is correctness and
plumbing evidence, not an accuracy result. The specified production best.pt/cache
are not available at the configured local paths. Full training, production-scale
comparison, CUDA, MPS and simulator execution have not been run for this estimator.
The historical 26.59 mm and 7.35 mm figures are retained above; no new GRU accuracy
claim is made. Local smoke outputs use a separate `outputs/temporal_gru_smoke/` root.

The expanded CPU suite (the command above plus `tests.test_gnn_training`,
`tests.test_gnn_trainer`, `tests.test_experiment_logger` and
`tests.test_experiment_artifacts`) ran 98 tests: 97 passed and one existing GNN
optimizer/resume test was skipped because `RUN_GNN_TRAINING_TESTS` was not enabled.
Separately, `tests.test_gated_mpnn` has three passing
tests and two pre-existing failures: invalid semantic-ID and wrong-edge-type
validation are expected in forward by those tests, while the current optimized
GNN validates inputs at collation instead. Both failures were reproduced using
the unmodified `HEAD:graph/gated_mpnn.py` loaded in memory; GNN code and those
tests were left unchanged as requested.

Retained local fixture evidence (CPU, one epoch, five comparison frames):
[verification](../outputs/temporal_gru_smoke/144f6be5e9714593b4615d1a4bbb1a4f/verification.json),
[summary](../outputs/temporal_gru_smoke/144f6be5e9714593b4615d1a4bbb1a4f/evaluation/f0bf5c1d478a461c96e45a676e5114ca/summary.json),
[trace](../outputs/temporal_gru_smoke/144f6be5e9714593b4615d1a4bbb1a4f/evaluation/f0bf5c1d478a461c96e45a676e5114ca/episode_trace_000.png).
The maximum sequence/streaming position difference was 1.86e-9 m. Batch-one
GRU-only mean latency was 0.0344 ms over 100 timed steps after ten warmup steps;
the reduced-GNN pipeline mean was 0.2897 ms. These timings describe this CPU
fixture and include validation overhead, not the production model or GPU.
GPU memory is unmeasured (null). Frozen checkpoint SHA256 was unchanged.


## Joint Peg/Hole temporal pose estimator

This extends the frozen-output dataset and GRU infrastructure above. The original
`TemporalHoleGRU`, its 23D schema/checkpoint kind, configs and streaming API remain
available. `TemporalPoseGRU` is a separate pose variant with `hole_only`, `peg_only`
and `joint_peg_hole` modes. The three new modes use identical frozen GNN output,
quality extraction, causal preprocessing and parent-disjoint splits. They change
which pose heads are trained. The Action7 controller is untouched.

With the existing Torch/NumPy/PyYAML/Matplotlib dependencies, original gated-MPNN
checkpoint, extracted cache and explicit temporal parent manifest, set paths in:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/pose_gru/train_hole.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/pose_gru/train_peg.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/pose_gru/train_joint.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/pose_gru/evaluate.yaml
```

The default configs use `data.raw_root: null`: the published graph cache contains
TCP XYZ but no TCP quaternion or gripper qpos. No missing robot feature is filled
with zero or GT. To enable the additional observed robot features, set `raw_root`
to the matching extracted raw episodes in **all three** configs. The installed
ManiSkill source exposes TCP pose in `_get_obs_extra()['tcp_pose']` and joint
positions via `get_proprioception()['qpos']`. The adapter reads exactly
`observation/extra/tcp_pose[1,7]` and `observation/agent/qpos[1,9]`; the last two Panda
joints are finger positions. Every raw frame must match the source SHA256 already
recorded in the published cache, and its observed TCP XYZ must match cached TCP.
Missing fields, malformed values or mismatches stop the run. `ground_truth/`
fields are never an inference fallback. This robot path was exercised with a
synthetic raw-schema fixture; no simulator rollout was run for this change.

The named input contract is `object21_world_vertex_quality_tcp_motion_wxyz_v1`:

| Indices | Input | Convention |
| --- | --- | --- |
| 0:21 | complete Frozen GNN Object21 | positions at 0:3 and 10:13 restored to world using current observed TCP; endpoint remains a world-axis relative vector |
| 21:46 | PEG, BOX, HOLE quality (8 values each), then all-vertices-missing | count/present/missing/known-confidence fraction/confidence-present/mean/min/max; actual validity masks |
| 46:49 | TCP world XYZ | metres |
| 49:52 | current minus previous TCP XYZ | metres per saved frame |
| 52 | previous-frame-present flag | zero at reset, one thereafter |
| 53:57, optional | observed TCP quaternion | unit world-frame wxyz |
| 57:60, optional | observed TCP rotation change | world-axis shortest rotation vector, radians per saved frame |
| 60:62, optional | observed finger joint positions | metres, left/right active-joint order |

Input width is 53 without raw proprioception and 62 with it. Zero motion on frame
zero is an explicit episode-boundary value, distinguished by the previous-frame
flag. No Peg-follow-TCP relation, grasp offset or phase-dependent update rule is
encoded. Quaternion signs are canonicalized to w >= 0 for input stability. Object
and TCP quaternions are world-frame wxyz, verified against `object21_from_gt()`
and the raw pose accessor. GT visibility, GT phase, future frames and previous GT
poses are excluded from model inputs.

`data_collection/temporal/pose_sequences.py` extends the existing aligned episode
dataset. Raw proprioception is joined by the same sample IDs/checksums. Input
normalization uses population mean/std over valid **training observations only**,
with unit scale for constant columns. No targets or validation/test observations
fit these statistics. Mean/std are saved as model buffers and in effective config;
evaluation checks checkpoint buffers against the train-normalization record.
Padded batches contain complete episodes with a boolean right-padding mask.
The GRU uses packed sequences so final hidden state excludes padding, and the loss
selects valid rows before quaternion operations, including when padded labels are
NaN. Every new episode starts at zero hidden state. Initial and occluded frames
are supervised. The earlier no-padding Hole-only workflow is unchanged.

The initial pose model is:
Linear(D,64) + SiLU → one shared GRU(64,128) → separate active pose heads,
each Linear(128,64) + SiLU + Linear(64,6). A head outputs XYZ and a 3D rotation
vector. Single-object modes create only that object's head and retain the other
Frozen GNN pose. Parameter counts include the projection, GRU and active heads:

| Input contract | Hole-only / Peg-only | Joint |
| --- | ---: | ---: |
| 53D cache | 86,598 | 95,244 |
| 62D observed robot | 87,174 | 95,820 |

Position defaults to **direct world regression**, scaled by 0.1 m: this lets the
hidden state output a remembered position even when the current GNN is wrong.
`gru.position_mode: residual` instead adds the scaled XYZ head output to the
current GNN world position. This choice is stored in config/checkpoint; it does
not alter the Frozen GNN. Rotation always uses a unit quaternion produced by the
exponential map of a world-axis rotation-vector residual, left-composed with the
current GNN quaternion and renormalized. It never averages quaternions linearly.

`training/gnn/temporal/pose_loss.py` computes four configurable dimensionless
terms: Peg position/rotation and Hole position/rotation. Position is coordinate-
mean Huber with unit delta on `(predicted_world - target_world) / position_unit_m`
(default 0.01 m). Rotation is `4*(1-dot(q,q_gt)^2)/rotation_unit_rad^2` (default
0.1 rad), equal to normalized `4*sin(angle/2)^2`; this smooth surrogate is locally
squared angle and respects q/-q equivalence without acos singularities. Rotation
metrics use exact geodesic angle in degrees. `loss.weights` controls the four
terms; a single-object mode includes only its active terms. Training reports each
term and the weighted sum, frame-weighted across valid rows, selects best using
held-out temporal-validation parents, and never evaluates temporal test for
selection. Full episode BPTT, episode-batch shuffling, Adam and gradient clipping
reuse the earlier workflow conventions. No optimizer resume is implemented.

GT defines Endpoint−Hole as Peg's **local +X half-length tip**, rotated by world
Peg quaternion, minus Hole world position. The refined estimator therefore
recomputes Object21[7:10] using its refined Peg/Hole poses and **Frozen GNN size**.
It discards the GNN's learned endpoint residual in this assembly; evaluation saves
a separate GNN geometric-endpoint ablation to expose that difference. No GT
endpoint or size is used at inference. Frozen Size4D is copied unchanged.
`StreamingPoseEstimator.reset()` clears hidden/TCP history; `.step()` accepts
current Object21, quality25, TCP and optional observed quaternion/finger values.
It returns world Peg/Hole poses and consistent TCP-relative Object21. When the
caller supplies Robot15/Freshness2, `assemble_state38` returns:
Robot15[0:15], PegPose7[15:22], Endpoint3[22:25], HolePose7[25:32], Size4[32:36],
Freshness2[36:38]. Robot/freshness semantics remain caller-owned; existing caches
do not provide complete Robot15/Freshness2, so these are not fabricated in replay.

```python
from graph.temporal.observations import observation_features
from graph.temporal.pose_gru import StreamingPoseEstimator

# Load TemporalPoseGRU with its saved config, weights and normalization first.
estimator = StreamingPoseEstimator(model)
estimator.reset()  # every episode boundary
result = estimator.step(frozen_object21, observation_features(graph, include_peg=True),
                        observed_tcp_world, robot15=robot15, freshness2=freshness2)
# With the 62D contract, also supply tcp_quaternion=... and gripper_qpos=....
predicted_state38 = result['state38']
```

Evaluation accepts `pose_checkpoint` plus `comparison.learned_gate` and
`comparison.gru_checkpoints` entries for the other pose modes and optionally
`legacy_hole_only`. Every supplied checkpoint must match the exact source GNN,
dataset and temporal parent manifest. Pose modes must also share the observation
contract; each loads its saved train normalization. Comparison methods are:
Frozen GNN, unchanged Hole Rule Memory, First Observation Hold, Learned Gate,
legacy Hole XYZ GRU and the three new pose GRU modes. Unconfigured checkpoint
comparisons are explicitly listed as unavailable rather than assigned historical
scores. First Observation Hold seeds the initial GNN Hole world pose without GT
and retains it for that episode; endpoint is adjusted to preserve the original
estimated Peg tip. Other Hole baselines retain their historical position-only
behavior, leaving current GNN rotations unchanged.

The evaluation recipe uses the previously measured **count-only** Rule Memory
(minimum five target nodes, no confidence gate, one-frame update), which produced
7.35 mm in the recorded comparison above; the older quality-gated recipe produced
8.19 mm there. Historical user-reported values (GNN Peg 4.34 mm/2.94°, Hole
26.59 mm/5.50°, endpoint 27.18 mm; first hold 5.12 mm; learned gate 4.78 mm) are
context, not automatically valid same-split GRU results. All available baselines
are recomputed on the exact same frames, and source hashes are saved. Historical
checkpoints that lack required provenance cannot be silently accepted.

Metrics include mean/median/p90 for both position errors (mm), both angular errors
(degrees), endpoint vector error (mm), and per-episode breakdowns. GT is consumed
only after every prediction is fixed, to define Peg stationary/moving slices
using configured translation/angular-change thresholds. These diagnostic thresholds
never control learned updates. An optional `phase_labels` NPZ with `sample_ids`
and `phase` records enables actual grasp-phase groups; missing phases remain
unavailable, with no inferred grasp labels. Existing visibility diagnostics add
the strong-hole-occlusion proxy. Missing PEG, missing HOLE, missing BOX+HOLE,
long missing runs and elapsed missing ages are scored explicitly; vertex absence
is an observation proxy rather than proof of physical occlusion.

Per-episode Peg metrics include translational and angular lag (positive means
tracking earlier GT, negative means lead), position drift and candidate-lag errors.
Candidate lags use the same interior evaluation window; static/too-short signals
have null lag. This is trajectory-matching diagnostic lag and can be confounded
by bias; it is not measured controller delay. Hole metrics include position drift,
orientation-change error and reconstructed GT stationarity. Recovery records cover
Peg, Hole and combined BOX/HOLE support, with missing-run duration, errors before,
at return and after the visible run, and latency until both position and angular
errors meet the diagnostic bounds. Long-run recovery events are flagged. PNG
traces show paired position/rotation errors and actual observed support.

Every new pose model is replayed with full sequences and streaming, checking all
world Object21 fields (atol 1e-6, rtol 1e-5) and final hidden state (atol 1e-5,
rtol 1e-4). Summaries record parameters and warmed-up batch-one latency for every
available method, with explicit component/device scope. Pose timing includes
feature construction, GRU and Object21 assembly, excluding GNN, disk IO and robot
assembly. Rule/hold timing is CPU position-memory work; GNN/gate/legacy timings
state their separate scope. CUDA allocator memory counters are available through
the existing timing helper and are null on this CPU host. These are not comparable
end-to-end sensor-to-control throughput numbers.

Training outputs use separate `outputs/temporal_pose_<mode>/<run_id>/` roots;
evaluation uses `outputs/temporal_pose_evaluation/<run_id>/`. Effective configs,
normalization, source hashes, histories/checkpoints, prediction NPZ, metrics and
traces reuse `ExperimentLogger` and existing artifact-kind registration. Publishing
remains explicit via `tools.publish_experiment`; no remote write is part of training
or evaluation. Existing GNN, learned gate, Rule Memory and legacy GRU files are
preserved. Full CUDA training, large evaluation and commit/push require approval.

Verification on 2026-11-07 KST: the 12 new pose tests and 98 earlier expanded
checks ran as 110 tests, with 109 passing and the existing opt-in GNN optimizer
execution test skipped. The two previously reported GNN validation-test failures
remain outside this change; the GNN implementation is untouched. The new tests
cover quaternion composition/signs/identity gradients, exact GT endpoint geometry,
53D/62D features, observation-only raw joins, train normalization, all modes,
causal prefixes, masks/hidden state, missing-frame gradients, State38 assembly,
GT motion and phase diagnostics, provenance rejection and tiny real-cache root
CLI train/evaluation. A retained smoke uses matching synthetic raw proprioception,
five one-epoch fits (gate, legacy GRU, three pose modes) and all eight comparisons
on five reserved fixture frames. This is plumbing evidence, not research accuracy.
Production checkpoint/cache evaluation, real grasp/slip phases, CUDA/MPS, GPU
memory, simulator rollouts and large runs were not executed.

Retained local pose fixture evidence:
[verification and test-log location](../outputs/temporal_pose_smoke/fb3f91e49522469582b23273c8c681b9/verification.json),
[eight-method summary](../outputs/temporal_pose_smoke/fb3f91e49522469582b23273c8c681b9/evaluation/4db2eeb2c1204081973cd9aa46b81ac4/summary.json),
[pose traces](../outputs/temporal_pose_smoke/fb3f91e49522469582b23273c8c681b9/evaluation/4db2eeb2c1204081973cd9aa46b81ac4/pose_trace_000.png).
The largest full/streaming Object21 component difference was 2.98e-8 across the
three pose modes. The 62D Joint model had 95,820 parameters and a CPU mean of
0.286 ms/step over 100 timed steps after ten warmup steps, including feature/pose
assembly and excluding GNN/IO. The source fixture GNN SHA256 was unchanged.
The retained JSON metrics also passed local publisher-plan validation without
network access; predictions remain an explicit trajectory artifact selection.


## GRU insertion-precision objectives

This extends the Joint GRU above through loss/evaluation/configuration changes;
its projection, shared GRU, separate pose heads and parameter counts remain
unchanged. Legacy Hole XYZ GRU, all three pose modes, earlier pose-only configs,
Frozen GNN/State Head and learned/rule baselines remain available. New experiment
outputs use separate `outputs/temporal_precision_<ablation>/<run_id>/` roots.
No existing checkpoint/result is overwritten and the Action7 controller is untouched.

After setting the same frozen checkpoint/cache/parent manifest and optional raw
proprioception paths as the Joint recipe, run from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/precision/train_pose.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/precision/train_pose_relative.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/precision/train_pose_relative_lateral.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/precision/train_pose_relative_full.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/precision/evaluate.yaml
```

Set `pose_checkpoint` and optionally `comparison.precision_checkpoints` in the
last config to compare the selected ablation checkpoints on identical frames.
The existing gate, original Hole XYZ GRU and the other pose-mode checkpoints can
still be supplied. These comparison executions never select weights or tune
hyperparameters. Full CUDA training, large evaluation and commit/push remain
subject to separate approval.

### Verified simulator geometry and the meaning of 3mm

The installed ManiSkill 3.0.1 `PegInsertionSideEnv` defines `_clearance=0.003` m.
Peg square half-width is randomized between 15 and 25mm; the Hole inner half-width
is that half-width plus 3mm. This is a geometric half-width difference, not a
3D position-estimation threshold or Euclidean lateral success radius.
`has_peg_inserted()` actually tests the **head centre** in Hole coordinates:
`x >= -0.015m`, `abs(y) <= hole_radius`, `abs(z) <= hole_radius`. It has no explicit
orientation flag and no upper-X bound. Collision geometry still affects physical
motion, but this head-position predicate alone is not a shaft-clearance or
orientation test. The task uses local **+X** as the insertion axis and Hole YZ as
the perpendicular plane. The success predicate uses `peg_head_pose`, composed
from Peg pose and the local `[half_length,0,0]` offset; it does not use the
simulator's separate `peg_head_pos` convenience property with an unrotated offset.

A live CPU run on 2026-11-07 KST checked reset plus two zero-action steps against
that installed implementation (`physx_cpu` / `sapien_cpu`). Reconstructed endpoint
and Hole-frame vectors matched `peg_head_pose` and `has_peg_inserted()` within
4.85e-8m, with the same bool predicate. This was not an insertion rollout or a
success-rate experiment. Evidence:
[installed predicate and three-frame checks](../outputs/temporal_precision_smoke/simulator_geometry.json).
CUDA simulation/rendering was not executed.

### Differentiable relative objectives

The world-axis relative vector is reconstructed as
`r_pred = peg_position + R(peg_quaternion)*[frozen_half_length,0,0] - hole_position`.
It is translation-invariant, so world or TCP-relative positions give the same
vector. The training objective **recomputes** this from refined poses and Frozen
size even if Object21[7:10] contains another value; it never copies the original
GNN endpoint into the relative loss. GT uses the verified target endpoint minus
Hole definition. Gradients reach both refined positions and the Peg orientation.
No endpoint auxiliary head or GT inference input is needed.

`training/gnn/temporal/pose_loss.py` retains the earlier four pose losses and adds:

- `relative_position`: coordinate-mean Huber on `(r_pred-r_GT)/position_unit_m`.
- `lateral_position`: coordinate-mean Huber on Hole-GT-frame YZ components of that
  **prediction error**, divided by `position_unit_m`. It does not force a currently
  unaligned Peg to appear physically aligned. GT Hole quaternion is loss-only.
- `relative_orientation`: Huber on the geodesic angle between predicted and GT
  `q_hole^-1 * q_peg`, divided by `rotation_unit_rad`.

Default scales remain 0.01m and 0.1rad, with unit Huber delta after normalization.
Default weights are 1 each for Peg/Hole position/rotation, 0.25 for relative vector,
0.25 for lateral and 0.1 for relative orientation. The extras start as modest
auxiliary contributions. `loss.weights` controls all seven terms;
`loss.effective_weights` records zero for inactive terms. The four supported
ablations activate extras cumulatively: `pose_only`, `pose_relative`,
`pose_relative_lateral`, `pose_relative_full`. For `rotation_objective:
pose_surrogate` (default), pose terms preserve the earlier mathematical objective,
including q/-q invariance. Its squared-angle surrogate now uses an algebraically
equivalent sign-invariant chord calculation to give exact zero for identical
float32 quaternions. `geodesic_huber` is an explicit optional pose-rotation recipe
shared with the separate insertion Mamba/GRU workflow; it must be matched for a
fair comparison and changes the pose-only objective. Relative orientation is
always geodesic Huber. Padding is masked before all geometry and losses.

The directed +X head matters: reversing the shaft changes which endpoint is
scored. The square cross-section has physical roll symmetries, but labeled pose
quaternions retain roll; full SO(3) relative orientation loss does not quotient
out quarter-turn roll symmetry. Evaluation additionally reports directed relative
**axis-direction** error, which ignores roll. Neither angular metric is called
an explicit simulator success condition.

### Precision, motion and checkpoint selection

`evaluation/gnn/temporal/precision.py` is a reusable metric contract:
`peg_head_plus_x_hole_x_axial_yz_lateral_v1`. Metrics distinguish:

- Actual Object21 relative 3D error, preserving historical GNN/learned-gate
  endpoint residuals; pose-reconstructed relative error is a separate ablation.
- Signed X/Y/Z estimation errors in the **GT** Hole frame, axial absolute error,
  lateral YZ norm, relative SO(3)/axis angular errors, and local-vector error when
  the predicted vector is instead transformed by the **predicted** Hole rotation.
- Strict frame fractions `lateral <1/<2/<3mm` and `relative_3d <3mm`. These are
  estimation-accuracy fractions, never insertion success rates.
- Mean/median/P90/P95/max, per-episode tails/worst cases, top failed frame IDs,
  corresponding absolute pose errors, and changes/ratios against Frozen GNN.
  Predicted/GT traces and all per-frame errors are saved for follow-up inspection.
- Separate GT physical head-centre lateral alignment and the documented task
  head-position predicate count; these are not model estimation errors or measured
  closed-loop rollout success rates.

Existing stationary/moving Peg, missing PEG/HOLE/target support, long absence,
reobservation, temporal lag and Hole drift metrics remain. Offline-only additions
slice Peg/TCP co-motion, abrupt translation direction reversal, and a near-axis
insertion-alignment **geometry proxy**. Thresholds are configurable and recorded;
co-motion does not assert a grasp, and the direction-change slice uses displacement
per saved frame because timestamps are unavailable. Actual phase groups require
supplied `sample_ids`/`phase` labels; unavailable grasp/slip labels are never
fabricated. None of these GT-derived diagnostic slices enters inference.

New ablation recipes select checkpoints on **temporal validation only**, using
normalized weighted geometry/motion/absolute metrics, rather than total loss alone:
relative P90/3mm + lateral P90/3mm + moving-Peg position mean/3mm +
0.25*moving-Peg rotation mean/5deg + 0.25*(Peg+Hole absolute-position means)/3mm +
0.1*(Peg+Hole angular means)/5deg. Component scales/weights are configurable under
`selection.components` and recorded individually. If no Peg-motion frames exist,
that motion component uses the corresponding overall Peg error; frame counts
and null motion metrics make this fallback auditable. Earlier recipes without
`loss.ablation` retain validation-loss selection. Each new checkpoint records
both `validation_loss` and the independent `selection_score`.

Input features remain the existing 53D/62D observation schema; train-only population
normalization, complete-episode zero-hidden/right-padding context, frozen source,
parent manifest and precision-coordinate convention are recorded as named
contracts. Loss-ablation comparisons require equal feature contracts, matching
train mean/std and the same pose mode. The separate Mamba insertion workflow
currently adds observed-support ages (55D/64D) and uses truncated reset windows;
its results cannot be pooled with this full-episode GRU recipe as a same-input,
same-context comparison. For that pair use its explicit `architecture: gru`
comparator or first match feature/context/normalization/rotation-loss contracts.
The shared geometry/loss utilities and precision definitions are available to both;
no fairness claim is made from mismatched settings.

### Independent final-test preparation and limits

The repeatedly inspected original GNN validation is retained as a **reused
reference**, not final independent generalization. A manifest with
`test_role: independent`, explicit source/external `parent_by_episode` and
`data.independent_cache_dir` can select a separate final-test graph cache through
`data_collection.temporal.insertion_sequences`. All external parent identities must be
disjoint from every original GNN train/validation/test parent, not just temporal
train. The original train/validation episode/parent assignments remain fixed.
During training only external split metadata is used for leakage checks; external
frame tensors are opened only for an explicit test evaluation. Observed-robot
independent data additionally requires `data.independent_raw_root`. Source/cache
checksums and independent dataset identity are recorded. Legacy baselines with
the same real development parents can be evaluated there even if only their
old reference-test manifest differed. No independent production data was collected
or evaluated in this implementation.

Tests cover vector sign, actual local axes, GT endpoint agreement, q/-q and finite
rotation gradients, relative-only gradient to both GRU heads, poisoned copied
endpoint slots, lateral-error/physical-alignment separation, strict thresholds,
padding, validation selection, independent parents/cache CLI, streaming/reset/GT
isolation and all four tiny one-epoch ablation workflows. Existing checkpoints
and source hashes are preserved. Production insertion precision and absolute-pose
accuracy improvements remain unmeasured; full retraining, large evaluation and
remote changes were not performed.

The final GRU suite ran 123 tests: 122 passed and one pre-existing opt-in GNN
optimizer test was skipped. Three additional shared Mamba geometry/loss/causal-cache
tests passed after the common loss extension. The 13 new precision tests include
four one-epoch ablation fits and independent-cache train/test CLI coverage. The
retained CPU smoke reused the previous tiny 62D cache/Frozen GNN, compared 11
available baselines/modes/ablations on five reference frames, preserved its Frozen
checkpoint SHA256, and checked full/streaming component differences <=2.98e-8.
These synthetic figures are plumbing evidence, not 3mm research-performance claims.
Evidence:
[verification/test log](../outputs/temporal_precision_smoke/bb02b4bc5a84451aba70af8ff47b3eba/verification.json),
[paired precision summary](../outputs/temporal_precision_smoke/bb02b4bc5a84451aba70af8ff47b3eba/evaluation/d5e0de588d694ab6b13866bc67eb1075/summary.json),
[relative/lateral/axial traces](../outputs/temporal_precision_smoke/bb02b4bc5a84451aba70af8ff47b3eba/evaluation/d5e0de588d694ab6b13866bc67eb1075/precision_trace_000.png).
Local JSON publisher-plan validation passed without network access. The source
GNN's two previously documented forward-validation test failures were not changed.

### Vast CUDA validation and detached training launch

On 2026-11-07 KST, instance 59441593 (RTX 3090 24GB, Torch 2.14.1+cu132) passed
66 related unit tests. A real-cache CUDA smoke used one temporal-train episode
(105 frames) and one temporal-validation episode (93 frames), performed one
optimizer update, checked nonzero gradients in both heads and verified unchanged
Frozen GNN weights/checkpoint SHA256. Full/streaming maximum component difference
was 1.19e-7; no final-test frames were read.

The user-authorized production training was launched in tmux session
`gru_precision_48e96aab`: `joint_peg_hole`, `pose_relative_full`, hidden 128,
20 epochs, batch of four episodes, lr 0.001 and 53D cache-only observations.
The published frozen checkpoint/cache and explicit parent manifest were verified
against immutable SHA256 values. Raw quaternion/gripper fields were unavailable
in this cache and were omitted. The run uses an isolated venv, existing
`ExperimentLogger`, local-only W&B-disabled logging, and no automatic publishing.

Remote root:
`/workspace/gru_temporal_precision/48e96aabdb54427391d43cc6ed3784c8/`.
The config is `train.yaml`, output root is `training/`, and log is `training.log`.
Only process startup was confirmed; the user requested no watching or completion
monitoring. This records launch, not final training results or accuracy.
Local evidence:
[launch receipt](../outputs/temporal_precision_launch/48e96aabdb54427391d43cc6ed3784c8/launch_receipt.json),
[real CUDA smoke](../outputs/temporal_precision_launch/48e96aabdb54427391d43cc6ed3784c8/cuda_smoke.json),
[unit test log](../outputs/temporal_precision_launch/48e96aabdb54427391d43cc6ed3784c8/unit_tests.log).

## GRU commit verification

On 2026-11-07, the GRU commit candidate was exported independently of the pending
Mamba and Hybrid implementations. All 50 Hole/Pose/precision GRU, learned Gate
and temporal-memory tests passed against that candidate. The 11 GRU YAML recipes
also resolved to callable root-entrypoint workflows without Mamba/Hybrid sources.
The existing Logger/artifact suites ran separately from the repository with Git
metadata available: all 36 tests passed. These checks used CPU fixtures and tiny
cache-based training/evaluation; CUDA/MPS, live simulator execution, full-dataset
training and external publishing were not run for this commit.
