# Temporal state estimation

The Graph team's temporal estimators consume frozen GNN outputs and observed
robot metadata. Run from the repository root with Torch, NumPy, PyYAML and
Matplotlib installed. Set checkpoint, cache and parent-manifest paths before use;
Pose and insertion recipes document optional or required raw robot observations.

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
- [Insertion Mamba/GRU](#mamba-insertion-precision-estimator)
- [Base GNN and learned Gate](gnn_training.md)
- [Hybrid composition and diagnostics](hybrid_estimation.md)

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

## Mamba insertion-precision estimator

`graph/temporal/insertion.py` contains a separate temporal **state** estimator. The
Frozen GNN and recurrent Action controller retain their weights and behavior.
Models stay in `graph/`, orchestration in `training/gnn/temporal/insertion.py` and
`evaluation/gnn/temporal/insertion.py`, and selection/windowing in
`data_collection/temporal/insertion_sequences.py`. Existing geometry, pose feature
construction, checksum loading, memory/gate replay and `ExperimentLogger` are
reused. No new simulator interface, controller, W&B client or publisher is added.

### Prerequisites and run commands

From the repository root, install the existing Torch/NumPy/PyYAML/Matplotlib
requirements. Set the original gated MPNN `best.pt`, published graph cache and
matching raw episode root in `graph/configs/temporal/insertion/train_mamba.yaml`. The raw
root is needed for observed TCP wxyz and finger qpos; missing observations fail
instead of falling back to GT. Frozen model coordinate/size scales come from the
original checkpoint config. Object21 outputs already use metres and world wxyz;
there is no extra inverse target-normalization step. The original published checkpoint is not present
at the recipe's default local path on this Mac.

Create a manifest from an **actual collection parent map**, never infer parent
independence from a filename:

```bash
python -m data_collection.temporal.insertion_sequences \
  --cache-dir /workspace/gnn_cache \
  --parent-map /workspace/parent_by_episode.json \
  --output /workspace/gnn_cache/metadata/temporal_insertion_splits.json
WANDB_MODE=disabled python main.py --config graph/configs/temporal/insertion/train_mamba.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/insertion/evaluate.yaml
```

The first command, without an independent cache, labels `test` as
`gnn_validation_reference`. This is the previously reused 180-episode GNN
validation set, not an independent generalization test. Development train and
validation remain subsets of GNN-train parents. Siblings of all reserved GNN
validation/test parents are excluded from development, including unused siblings.
The manifest is created exclusively; an existing file is never overwritten.

For final generalization, separately obtain new episodes with new parent IDs and
prepare their checksummed cache using the **same preprocessing**. The current
published cache reader materializes train/validation entries, so independent
cache episodes are selected from those entries; its own reserved `test` entries
remain unread. Distinct source/external episode IDs and an explicit combined
parent map are required. Create the manifest with the extra option:

```bash
python -m data_collection.temporal.insertion_sequences \
  --cache-dir /workspace/gnn_cache \
  --independent-cache-dir /workspace/independent_cache \
  --parent-map /workspace/combined_parent_by_episode.json \
  --output /workspace/temporal_insertion_independent_splits.json
```

Set `data.independent_cache_dir` and `data.independent_raw_root` in both recipes.
All independent-cache parents must be absent from **every** original GNN split.
Training reads independent metadata to check parents, but no independent cache
inventory, frame or target. Evaluation defaults to `evaluation.split: validation`.
Set `temporal_checkpoint` to the selected best checkpoint and add comparison
checkpoints only after fixing the experiment plan. Switch to `split: test` once
hyperparameters and thresholds are frozen; test statistics must not tune models.
Neither collection nor large evaluation is launched by these preparation steps.

### Mamba implementation and environment audit

The implementation reference is the official
[Mamba v2.2.6 mixer](https://github.com/state-spaces/mamba/blob/v2.2.6/mamba_ssm/modules/mamba_simple.py)
and [selective scan reference](https://github.com/state-spaces/mamba/blob/v2.2.6/mamba_ssm/ops/selective_scan_interface.py).
Two **explicit** backends are supported; failure never silently selects another:

- `torch_reference`: local FP32 Mamba-1 selective SSM. Input-dependent Delta/B/C,
  stable negative diagonal A, causal depthwise convolution, D skip and SiLU gate
  are present. Projection/convolution are batched; selective scan uses a readable
  Python loop. It maintains real convolution and SSM caches, with no history
  window recomputation. This backend is not an optimized CUDA performance claim.
- `mamba_ssm`: official `Mamba` mixer, parallel fused sequence forward and actual
  `allocate_inference_cache`/`step` incremental state. The state dictionary layout
  of the reference mixer matches the official mixer for an optional parity test.
  This path was validated on Vast.ai 59441593 as recorded below; it has not run on this Mac.

The [pinned upstream requirements](https://github.com/state-spaces/mamba/blob/v2.2.6/README.md#installation)
include Linux/NVIDIA, Torch and CUDA. On an authorized Linux CUDA host, check the
actual Torch/CUDA/compiler ABI first; a candidate installation command is
`python -m pip install --no-build-isolation 'mamba-ssm==2.2.6.post3' 'causal-conv1d>=1.4.0,<2'`.
Prebuilt wheel compatibility is unverified; the approved source build against
Torch 2.14 is documented below.
Optional convolution kernels are not required for the reference backend.
Mamba-2 was reviewed as an upstream option; this small first prototype uses
Mamba-1, whose state size 16 and explicit incremental interface match the recipe.
The local Mac environment was retained; the subsequent approved remote installation
and its compatibility pins are recorded below.

On 2026-11-07 KST the host reports macOS arm64, Torch 2.14.1, no Torch CUDA,
MPS available, ManiSkill 3.0.1 and SAPIEN 3.0.3. `mamba_ssm`, `causal_conv1d` and
Triton are absent. Local execution uses CPU; the separate remote CUDA reference smoke is recorded
below. MPS temporal execution remains unverified; official CUDA kernels were subsequently
validated on the approved Vast.ai host as recorded below. Existing `recurrent.models.MambaPolicy` is an Action policy and
was inspected but is not imported or modified by this estimator.

### Observation and output contracts

Full input is **64D**, identically ordered for Mamba and the new GRU comparator:

| Slice | Input |
| --- | --- |
| `0:21` | Frozen Object21 with Peg/Hole XYZ translated from TCP-relative to world; world wxyz normalized |
| `21:46` | PEG/BOX/HOLE count, presence, missingness, known-confidence fraction/presence/mean/min/max, plus all-missing flag |
| `46:49` | Observed TCP world XYZ |
| `49:52`, `52:53` | Consecutive observed TCP delta XYZ, previous-frame flag |
| `53:57`, `57:60` | Observed TCP wxyz, shortest world-axis rotation delta |
| `60:62` | Observed left/right finger joint positions |
| `62:64` | Consecutive missing-observation ages for PEG and BOX-or-HOLE support |

The first frame has zero TCP motion and no previous-frame flag. An initially
missing object has age one; observed support resets its age to zero. These ages
are observation metadata, not GT visibility or predicted-state reliability.
`data.require_robot: false` plus `raw_root: null` is an explicit **55D cache-only
ablation**, not the full robot experiment. Both compared models must use the same
choice. Raw proprioception is joined by exact saved-frame checksum and observed
TCP agreement. GT poses/endpoints/visibility/phases never enter features.
Train-only population mean/std are fitted before windowing, stored as buffers and
checked against checkpoint metadata. Constant training features use unit scale.

World Object21 output has Peg pose `0:7`, endpoint-minus-Hole vector `7:10`, Hole
pose `10:17`, and unchanged predicted Size4D `17:21`. Positions are metres;
quaternions are unit wxyz. Streaming also returns TCP-relative Object21 and
predicted-Hole-frame relative geometry. Passing caller `robot15` and `freshness2`
adds State38 as `[Robot15, Object21, Freshness2]`; observation ages are separately
named and do not silently replace the caller's established freshness semantics.

Default model: d_model 128, one layer, d_state 16, expand 2, convolution width 4,
shared temporal context and separate 6D Peg/Hole residual heads. Positions default
to residual correction with scale 0.01m because the Frozen GNN supplies a useful
spatial anchor, especially for Peg. `position_mode: direct` supports a controlled
comparison. Rotation uses an exponential-map residual left-composed with the GNN
rotation, never linear quaternion interpolation. Inactive poses remain Frozen GNN
outputs; all modes use the same differentiable Object21 assembler.

Full-feature parameter counts, including projection, normalization layers and
heads (normalization buffers are excluded):

| Backbone | Hole-only / Peg-only | Joint |
| --- | ---: | ---: |
| Mamba-1 reference | 133,702 | 142,348 |
| Same-feature GRU | 116,038 | 124,684 |

The comparison holds feature/data/protocol constant; these defaults do not claim
matched parameter budgets. Record architecture/backend and actual counts per run.

### Simulator geometry and losses

Installed `mani_skill/envs/tasks/tabletop/peg_insertion_side.py` defines the head
pose as `peg.pose * Pose([peg_half_size_x, 0, 0])`. Thus
`endpoint_world = peg_position + R_peg @ [predicted_half_size_x, 0, 0]` and
`relative = endpoint_world - hole_position`. Inference uses only predicted size;
GT size is used only for labels and independent geometry checks. The installed
`peg_head_pos` convenience property omits the rotation of its offset; the
rotation-aware `peg_head_pose` is the geometry reference for this task.

Hole local X is axial; Y/Z are lateral. The task uses `_clearance=0.003m` when
constructing hole radius as peg half-width plus clearance. Its actual
`has_peg_inserted()` predicate is head Hole-frame X >= -0.015m and absolute Y/Z
<= hole radius, with no orientation flag or upper axial bound. Neither the 3mm
clearance nor a 3mm estimation threshold is an insertion success criterion.

`insertion_loss` extends the existing pose-loss module without changing the
original GRU objective. It masks padding **before** quaternion/geometry operations.
Each position/relative error is divided by `position_unit_m` (default 0.01m), then
coordinate-mean Huber with delta 1. Each quaternion-equivalent geodesic angular
error is divided by `rotation_unit_rad` (default 0.1rad), then scalar Huber.
The seven individually weighted terms are:

- Peg/Hole position and geodesic rotation (four terms).
- World endpoint-minus-Hole prediction error.
- Y/Z components of that **error** rotated into the GT Hole frame. It supervises
  actual GT arrangement, not forced zero lateral displacement.
- Relative SO(3): compare `inverse(q_hole) * q_peg` against its GT counterpart.

Full directed SO(3) is retained to match the labeled pose contract. Square-section
rotation symmetries are not quotiented out, so these orientation diagnostics can
penalize physically equivalent cross-section orientations. The signed +X head
axis is retained; orientation losses do not claim to reproduce simulator success.
Relative gradients flow through predicted endpoint geometry into both active
pose heads. Frozen Size4D limits attainable endpoint accuracy and remains a
separately inspectable source of error.

Use `loss.ablation` = `pose_only`, `pose_relative`, `pose_relative_lateral`, or
`pose_relative_full`. Use `mode` = `hole_only`, `peg_only`, or `joint_peg_hole`.
All seven unweighted terms and the weighted total are saved for train/validation
in `history.json`, local/W&B scalar curves, and `loss_curves.png`. Inactive-object
pose errors remain diagnostics but are excluded from that mode's pose objective.
Set weights and units in YAML; position and angular units are explicit.

### Causal training, streaming and fair comparisons

`training.sequence_length` accepts 32/64/128 or another positive integer. Complete
episodes first supply actual consecutive TCP deltas/ages; windows never cross
an episode. `context_policy: reset_windows` is the only implemented training
policy: each nonoverlapping window starts zero recurrent state, without cache
carry or gradients across windows. This truncates learned temporal context.
Validation checkpoint selection uses the same window policy. Full streaming
inference retains episode history, so evaluation context can be longer than
training context; this limitation must be considered for long occlusions.
Right padding never contributes supervision or future influence to real frames.

`StreamingInsertionEstimator` exposes `reset(episode_id)`, `step(observation,
episode_id=None)` and `forward_sequence(prebuilt_features, mask=None)`. Multiple
interleaved episode IDs own independent TCP history, support ages and caches;
`discard(id)` releases completed state. Sequence forward starts zero state and
leaves live caches untouched. Streaming consumes only a whitelist of current
observation fields, rejects label keys, and uses actual recurrent caches.
Evaluation asserts sequence/step agreement within absolute 2e-5 and relative 1e-4
FP32 tolerance. CPU reference and GRU agree more closely in the smoke checks.
The optional CUDA test also compares official/reference forward and input
gradients with identical mixer weights; it is skipped when CUDA/package is absent.

Set `model.architecture: gru` with backend `torch_reference` to train the comparator
through the **same** dataset, features, normalization, heads, geometry, loss,
window length, seed, batch size, epoch budget and minimum-validation-loss selection.
Add its checkpoint to `comparison.checkpoints`, e.g. `gru_joint: /path/best.pt`.
Named checkpoints also support pose-mode/loss ablations. Evaluation verifies
Frozen GNN, source cache, actual development assignments/parents and feature
contracts. It records per-comparison seed/training/loss/normalization equality
rather than silently treating mismatched runs as fair. Deliberately different
loss ablations are identified explicitly. Select best within each run; weighted
loss values from different objectives are not directly comparable.

Frozen GNN, count-based Hole memory, first-observation Hole hold and a geometric
endpoint-only GNN ablation always run on the same frames. Configure
`comparison.learned_gate` for a preserved learned gate with matching development
parents; it retains its original restricted feature interface. Absent learned
checkpoints are explicitly unmeasured. Historical 4.34/26.59/4.78/11.77mm numbers
are not copied into new results. Baseline stored endpoint predictions remain
intact; the separately named GNN geometric ablation isolates endpoint assembly.

### Evaluation artifacts and verification

`summary.json` includes absolute mean/median/P90/P95/worst pose errors, relative
3D error, GT-Hole-frame lateral/axial error, relative orientation, directed
relative axis direction error and axis-angle alignment estimation error. It also
reports control-frame relative-vector error, using predicted Hole rotation for
the predicted vector, distinct from the GT-reference-frame estimation error.
Strict lateral <1/<2/<3mm and relative 3D <3mm frame percentages are saved overall,
by diagnostic group and by episode. Per-episode worst relative errors and all
paired episode mean differences are saved. Relative-geometry gains accompanied by
worse moving-Peg position error are flagged explicitly for each paired method. Motion slices and translational and
rotational lag reuse the existing post-inference evaluator. Optional exact-ID
phase labels supply before-grasp, after-grasp and insertion phases; unavailable
labels are reported, not inferred as grasp truth. Missing vertices are an
observation-absence proxy, not GT occlusion labels. Raw predictions, predicted
control geometry and source arrays are stored in `predictions.npz`.

Performance reports actual parameter counts, backend/device/FP32, synchronized
streaming and sequence latency, reset/reallocation overhead, actual cache bytes,
and CUDA allocation peaks when run on CUDA. `benchmark_training: true` adds
resident forward+loss+backward throughput with no optimizer update. Use identical
batch/length/warmup/steps on the same GPU for compute comparisons. Python reference
latency does not characterize fused Mamba. Wall-clock times are not simulator
control frequency. Artifacts use new UUID output directories and shared Logger
sidecars. Nothing is uploaded automatically and existing checkpoints are untouched.

Run the new contract/integration suite and existing regressions:

```bash
.venv/bin/python -m unittest tests.test_temporal_insertion -v
.venv/bin/python -m unittest tests.test_temporal_pose_gru tests.test_temporal_hole_gru \
  tests.test_temporal_gate tests.test_hole_memory tests.test_gnn_temporal_memory \
  tests.test_experiment_logger tests.test_experiment_artifacts -v
```

Synthetic checksummed-cache training is a workflow smoke, not production accuracy.
The live CPU test separately runs actual RGB-D/detection/GNN/streaming frames and
checks rotated task endpoint/Hole-axis consistency. Its GNN/temporal weights are
untrained because the production `best.pt`/published dataset are absent locally.
The approved 20-epoch training and requested development validation subsequently
completed, as recorded below. Independent test generalization, Mamba superiority,
stable 3mm insertion precision, MPS execution and closed-loop success are not established.

Local verification on 2026-11-07 KST: model/geometry suites ran 67 tests (66 passed,
one official-CUDA test skipped), and the Logger/Artifact suites ran 36 tests in a
separate process (all passed). Two targeted scoring/workflow tests also passed
after the motion/relative tradeoff report was added. The live smoke used three
actual 512×512 RGB-D frames on `physx_cpu`/`sapien_cpu`, including native corner
detection, an untrained frozen GNN and Mamba reference streaming. Endpoint and
Hole-frame reconstruction matched the installed task geometry. A single combined
process exposed an existing logger import-mock interaction with lazily loaded
ImageIO modules after simulator import; the documented separate invocations pass.
Raw local evidence is the ignored
[verification JSON](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/verification.json),
[model tests](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/model_tests.log),
[infrastructure tests](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/infrastructure_tests.log),
and [live CPU smoke](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/live_cpu_smoke.log).


On the user-selected Vast.ai instance **59441593** (2026-11-07 KST), CUDA reference
smoke also passed: RTX 3090 24GiB, driver 595.91.07, Torch 2.14.1+cu132,
CUDA toolkit 13.2 and Triton 3.8.0. The new instance has no production repository,
GNN checkpoint, dataset, simulator, Matplotlib, mamba_ssm or causal_conv1d. Only an
isolated source snapshot was copied into
`/workspace/temporal_insertion_validation/8e066415885e4395ae3a824472bc94b1/`;
no packages or system configuration were changed. GPU reference and same-feature
GRU forward/backward, both-head gradient checks and episode cache replay passed.
Maximum sequence/stream difference was 1.49e-8 for Mamba reference and 1.09e-8 for
GRU. Batch-one cache bytes were 20,480 and 512 respectively. The tiny probe used
batch 1, sequence length 5, FP32, two warmups and five measured repetitions;
streaming means including Python/validation/features/geometry were 7.04ms and
6.51ms. These are reference smoke timings, not optimized Mamba or production
throughput conclusions. No optimizer update or production evaluation occurred.
The initial reference smoke preceded the user's subsequent approval of an isolated
official-library installation and full training launch, recorded below. Raw result:
[CUDA reference smoke](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/cuda_reference_smoke.json)
and [probe source](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/remote_cuda_smoke.py).


The user subsequently authorized testing and a detached tmux training launch on
Vast.ai 59441593, with GRU sharing the same GPU and no ongoing training watch.
The production deployment uses `training.cuda_memory_fraction: 0.4`, batch 4,
sequence length 64, 20 epochs and the official Mamba backend. CPU math threads
are limited in the launcher; the source-GNN inference batch is 32. The allocator
limit applies to this process's Torch allocations, not driver/context memory or
another process's allocations. Frozen episode features are materialized once
when building sequence windows, avoiding repeated raw-feature reconstruction in
every epoch/window. The reusable default CPU recipe still leaves the optional
CUDA limit unset. Training uses the preserved 504/126 development parent split;
the previously reused 180-episode GNN validation remains reference-only.

The initially reviewed PyPI spelling `mamba-ssm==2.2.6` is not distributed;
`2.2.6.post3` is the available release in that line. Its default C++17 build
fails against the installed Torch 2.14 ATen headers, which require C++20. The
isolated installation uses the verified upstream sdist, changes setup build flags
to C++20 and sm_86 only, and records the patch and sdist SHA256. Mixer and CUDA
kernel source are unchanged. The build is labeled
`2.2.6.post3+cu132.torch214.cxx20.sm86` for reproducibility; default Torch and
system CUDA are retained. See the raw
[build provenance](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/mamba_build_provenance.json)
and [build flag patch](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/mamba_build_flags.patch).


### Approved CUDA training launch

On 2026-11-07 at 02:35 KST, after the user authorized installation, testing and
training, the official source build passed forward/input-gradient parity against
the reference mixer and real cache-based streaming equivalence. The runtime is
`mamba-ssm 2.2.6.post3+cu132.torch214.cxx20.sm86`, Torch 2.14.1+cu132,
Transformers 4.44.2, huggingface-hub 0.36.2 and tokenizers 0.19.1. Mamba's package
initializer requires generation aliases removed in Transformers 5; the isolated
venv pins the compatible
[Transformers 4.44.2 generation API](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/generation/__init__.py).
The optional causal-conv1d package is absent: this run uses the official CUDA
selective scan and Torch convolution, plus the library's incremental state path.

A production-data smoke used the original GNN checkpoint with one 105-frame train
episode and one 93-frame validation episode. Both temporal pose heads updated,
Frozen GNN gradients stayed disabled, and sequence/streaming output difference on
validation was 1.19e-7. Its normalization/model were discarded after the smoke;
the full run independently fits statistics on all 504 train parents. No test
frames were used by this smoke.

The published graph cache (83,453 frames) and exactly 64,702 raw development
frames were restored with archive and original-frame SHA256 checks. Raw GNN
validation/test episodes were not extracted. Shared input locations are
`/workspace/temporal_insertion_data/gnn_cache`,
`/workspace/temporal_insertion_data/episodes`, and
`/workspace/temporal_insertion_data/published_inputs/best.pt`. The original
checkpoint tracking sidecar was restored unchanged, preserving source run
`4186cdeb0fdb449ea443f28e3bddc936`. Existing default-path consumers can use the
shared `/workspace/gnn_cache` and `/workspace/episodes` aliases; no existing paths
were overwritten.

The detached training process was launched with:

```bash
cd /workspace/temporal_insertion_mamba/source
/workspace/temporal_insertion_validation/8e066415885e4395ae3a824472bc94b1/venv/bin/python -u \
  main.py --config graph/configs/train_temporal_mamba_cuda.yaml
# Attach after connecting to instance 59441593:
tmux attach -t mamba_insertion_joint
```

- tmux session: `mamba_insertion_joint`.
- Shared Logger run ID: `94c59d23eeba486a81adfa221fa829c3`.
- Config: `/workspace/temporal_insertion_mamba/source/graph/configs/train_temporal_mamba_cuda.yaml`.
- Log: `/workspace/temporal_insertion_mamba/logs/train.log`.
- Checkpoints/curves: `/workspace/temporal_insertion_mamba/training/94c59d23eeba486a81adfa221fa829c3/`.
- Local tracking: `/workspace/temporal_insertion_mamba/experiments/94c59d23eeba486a81adfa221fa829c3/`.
- Settings: Joint Peg/Hole, full relative loss, d_model 128, one layer, sequence 64,
  batch 4, 20 epochs, seed 0, GPU allocator limit 40%, math threads 4 and resident
  frozen inference batch 32. W&B is disabled and local shared tracking is active.

One startup confirmation found the training PID alive and the train-side Frozen
GNN/proprioception preparation in progress. No periodic watcher, completion
monitor, large evaluation or automatic publishing was launched. The user will
inspect the training directly. GRU can reuse these inputs; a scientifically
matched comparison must also use the same 64D feature schema, loss and protocol.

Raw evidence:
[official CUDA test](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/official_cuda_test.log),
[production-data smoke](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/production_cuda_smoke.json),
[data preparation](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/preparation.json),
[deployed config](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/train_temporal_mamba_cuda.yaml),
and [launch receipt](../outputs/temporal_insertion_smoke/8e066415885e4395ae3a824472bc94b1/launch_receipt.json).

### Matched GRU training and automatic comparison

The user requested retraining GRU on instance **59441593** with Mamba's input,
loss, training context and checkpoint selection on 2026-11-07 KST. Run the
team-owned recipe through the existing root entrypoint:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/insertion/train_gru_matched.yaml
```

The recipe requires the prepared production data and checkpoints at its explicit
`/workspace/` paths and the existing compatible CUDA environment. It reuses
`TemporalInsertionEstimator(architecture='gru')`, the 64D observed-robot/age
features, residual pose heads and the same insertion trainer. The selected Mamba
checkpoint `94c59d23eeba486a81adfa221fa829c3/best.pt` is pinned by SHA256
`ca6c6931e421d0bba776b8787038b0d16b87bc203bd82e1c757ac3053ed83069`.

Before training, `train_and_compare` rejects differences in source/cache/raw
paths, Frozen GNN hash, parent-manifest hash, feature contract, pose mode/heads,
seed, loss or training settings. Shared settings are seed 0, hidden width 128,
one layer, batch 4, 64-frame reset windows, Adam 0.001, gradient clip 1,
20 epochs, and 0.4 CUDA allocator fraction. Position/rotation units remain
0.01m/0.1rad; all seven geodesic-Huber loss weights are 1. Both runs select the
minimum frame-weighted validation objective. The 53D legacy GRU is preserved.

Train normalization is independently fitted and checked against Mamba's saved
train-only statistics, allowing only floating-point reduction roundoff; the
exact Mamba statistics are then reused. Validation never fits these values.
The existing train/epoch implementation performs all optimizer updates; no
second trainer or team-specific logger was added. GRU and Mamba parameter counts
may differ; this comparison matches the requested training protocol and shared
input/output components, rather than parameter count or training throughput.

After successful training, the same process verifies actual dataset identity,
parent manifest and exact normalization, then evaluates the selected GRU,
unchanged Mamba, Learned Gate and Frozen GNN together on full validation episodes.
The summary's `primary` method is the matched GRU and `mamba` is the preserved
reference. `comparison_protocols.mamba.matches_primary` records seed, training,
loss and normalization agreement. CuDNN TF32 is disabled for this inference
stage to check full-sequence/one-step parity. Training and evaluation use their
own shared ExperimentLogger run IDs and save effective configs, checkpoints,
predictions, metrics and provenance. No final-test data or publishing is included.

The tmux job runs training followed directly by evaluation; it has no polling
loop, watch process or automatic restart. Results are written under
`/workspace/temporal_insertion_gru_matched/training/<train_run_id>/` and
`/workspace/temporal_insertion_gru_matched/validation/<evaluation_run_id>/`.
The previously measured table below still refers to the legacy 53D GRU until
the new training and evaluation complete.

Launch verified on 2026-11-07 at 01:40:51 UTC (10:40:51 KST): tmux session
`gru_insertion_matched`, with a live pane at launch. The isolated source workspace
is `/workspace/temporal_insertion_gru_matched/source`; the combined log is
`/workspace/temporal_insertion_gru_matched/logs/training_and_comparison.log`.
The launcher limits CPU math threads to 4 and preserves the existing environment
and previous runs. No training watch or completion monitor was started.

Preflight verified the original feature/model/loss source hashes and identical
epoch/update/checkpoint-selection code. The actual-data CUDA smoke used one
105-frame train episode and one 93-frame validation episode, updated both pose
heads, kept Frozen GNN parameters frozen, and passed streaming parity with
maximum component difference **1.19e-7**. Smoke normalization used only that train
episode and was discarded. Related local suites ran **39 tests: 38 passed,
one official-CUDA test skipped**; the remote native-GRU CUDA path was exercised
by the actual-data smoke. Full training/comparison completion was not observed.

Raw launch evidence:
[protocol preflight](../outputs/temporal_insertion_gru_matched/protocol_preflight.json),
[actual-data CUDA smoke](../outputs/temporal_insertion_gru_matched/production_cuda_smoke.json),
[launch receipt](../outputs/temporal_insertion_gru_matched/launch_receipt.json),
[deployed config](../outputs/temporal_insertion_gru_matched/deployed_config.yaml),
and [local tests](../outputs/temporal_insertion_gru_matched/model_tests.log).

### Measured temporal validation

On 2026-11-07 KST, the requested post-training evaluation scored the unchanged
epoch-19 `best.pt` from training run `94c59d23eeba486a81adfa221fa829c3` on **126
validation episodes / 12,841 frames**. The checkpoint SHA256 is
`ca6c6931e421d0bba776b8787038b0d16b87bc203bd82e1c757ac3053ed83069`.
This is the temporal development validation split drawn from original GNN-train
parents, not the previously reused 180-episode GNN validation or an independent
final test. All baselines below were recomputed on these same frames.

Training completed all 20 epochs. Best validation objective was **0.923652738**
at epoch 19 (train objective 0.78526); epoch 20 validation was 1.055772535.
The objective fell 75.37% from epoch 1 to the selected epoch. These are normalized
weighted Huber/geodesic losses, not millimetres. Re-evaluation under the original
64-frame reset-window protocol reproduced **0.923652707**, within 3.1e-8 of the
saved selection objective. The metrics below primarily use uninterrupted episode
streaming, with a fresh cache at each episode boundary.

Frame-weighted mean errors and strict precision percentages:

| Metric | Frozen GNN | Learned Gate | GRU | Mamba |
| --- | ---: | ---: | ---: | ---: |
| Peg position mm | 3.93 | 3.93 | 5.52 | 3.56 |
| Peg rotation degrees | 2.87 | 2.87 | 1.61 | 1.43 |
| Hole position mm | 24.62 | 3.67 | 7.44 | 9.01 |
| Hole rotation degrees | 4.95 | 4.95 | 1.63 | 1.80 |
| Endpoint-minus-Hole vector mm | 23.24 | 11.05 | 10.18 | 8.65 |
| GT-Hole-frame lateral mm | 17.42 | 7.46 | 6.58 | 6.51 |
| GT-Hole-frame axial mm | 12.14 | 6.84 | 6.67 | 4.64 |
| Relative orientation degrees | 5.83 | 5.83 | 2.29 | 2.19 |
| Lateral <3mm % | 16.34 | 25.82 | 19.73 | 26.79 |
| Relative 3D <3mm % | 9.16 | 11.94 | 4.37 | 13.92 |

The preserved Rule-memory and first-observation-hold baselines remain in the
original streaming summary: relative 3D / lateral errors are 12.24 / 8.60mm and
11.53 / 7.88mm, respectively.

#### Existing GRU comparison audit

Comparison run `7113412787dd4fbd8e0883e338bd0705` remeasured the unchanged epoch-19
GRU from `b506bc44dc9a484094048d4fac9b903c` on instance 59441593. Its checkpoint
SHA256 is `521364a326bf195ad3f0e16a8a07ef229b472ba156169841d92cccae0a18993f`.
The audit verified the same Frozen GNN, dataset identity, train/validation episode
sets and actual parent mappings. It reused the original Mamba evaluation's exact
ordered sample IDs, GNN outputs, observed TCP, robot observations and targets;
the cache-only GRU receives only its trained observation fields. Each model keeps
its saved train-only normalization. No validation fitting or retraining occurred.

All four systems use the same complete-episode causal inference, episode reset,
world/WXYZ conventions, endpoint-minus-Hole sign, GT-Hole axes, scoring function
and frame weighting. Recomputed Frozen GNN, Gate and Mamba means matched their
original results. GRU model/feature/geometry functions were also checked against
its preserved training source. CuDNN TF32 was disabled for GRU replay; strict
full-sequence versus one-step inference checks passed with maximum component
difference **1.79e-7**. Its old training-validation history already covered 12,841
frames; the five position/vector mean errors reproduce within **0.00055mm**.
The table uses the new streaming measurements, including newly scored strict
thresholds. GRU loss 0.56678 and Mamba loss 0.92365 have different definitions
and must not be ranked against each other.

The table compares the completed trained systems on identical evaluation data.
It does not isolate the recurrent backbone, because their training differs:

| Setting | Existing GRU | Mamba |
| --- | --- | --- |
| Input | 53D cache-only | 64D observed-robot/age features |
| Position output | Direct, scale 0.1m | Residual, scale 0.01m |
| Rotation loss | Pose surrogate | Geodesic Huber |
| Relative/lateral/orientation loss weights | 0.25 / 0.25 / 0.1 | 1 / 1 / 1 |
| Training context | Complete episode | Reset windows of 64 frames |
| Checkpoint selection | Geometry-tail/motion composite | Minimum weighted validation loss |
| Epochs / selected epoch | 20 / 19 | 20 / 19 |

Against this GRU, Mamba reduces mean relative 3D error by **14.98%**, Peg position
by **35.49%** and axial error by **30.48%**. Mean lateral improves only **1.16%**
(0.076mm). GRU has lower Hole position/rotation errors and a lower lateral tail:

| Tail metric mm | GRU | Mamba |
| --- | ---: | ---: |
| Relative 3D P95 | 22.75 | 22.14 |
| Lateral P95 | 15.48 | 17.08 |

Mamba improves episode-mean Peg position in 117/126 episodes, relative 3D in
90/126 and lateral in 69/126; Hole position improves in 40/126. This is mixed
performance across metrics, not a uniform Mamba-over-GRU result. A controlled
backbone claim requires matching the training settings above.

Mamba distributions (position/vector errors in mm; rotations in degrees):

| Metric | Mean | Median | P90 | P95 | Worst frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peg position | 3.56 | 2.62 | 7.25 | 9.53 | 37.76 |
| Peg rotation | 1.43 | 1.02 | 2.89 | 3.94 | 18.95 |
| Hole position | 9.01 | 6.76 | 17.60 | 22.35 | 88.82 |
| Hole rotation | 1.80 | 1.33 | 3.87 | 5.20 | 17.89 |
| Endpoint-minus-Hole vector | 8.65 | 6.63 | 17.45 | 22.14 | 74.02 |
| GT-Hole-frame lateral error | 6.51 | 4.94 | 13.13 | 17.08 | 44.85 |
| GT-Hole-frame axial error | 4.64 | 2.89 | 11.15 | 15.44 | 63.50 |
| Relative orientation | 2.19 | 1.58 | 4.69 | 6.06 | 19.84 |

Strict lateral <1/<2/<3mm frame percentages were **3.59% / 13.62% / 26.79%**.
Relative 3D <3mm was **13.92%**. These measure estimation error,
not physical alignment or insertion success. Mamba does not yet provide stable
3mm precision on this validation set.

The same checkpoint under reset-window versus full-episode context:

| Context | Peg mm | Hole mm | Relative 3D mm | Lateral mm | Lateral <3mm % | Relative <3mm % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reset every 64 frames | 3.50 | 9.55 | 9.26 | 7.04 | 27.98 | 15.15 |
| Full episode streaming | 3.56 | 9.01 | 8.65 | 6.51 | 26.79 | 13.92 |

Full context improves mean and tail relative errors, while the reset-window
condition has slightly more frames below the strict 3mm thresholds. Both remain
above the required precision; the shortfall is not solely a streaming-context
change. Parallel full-sequence and actual cache-step outputs agreed within
4.17e-7 maximum component difference across all validation episodes.

Paired evidence and limitations:

- Against Frozen GNN, Mamba improves mean Hole position, relative 3D and lateral
  error in **126/126 episodes**; Peg position improves in **74/126**. Mean relative
  3D error falls 62.77%, and lateral error falls 62.63%.
- Against Learned Gate, Mamba improves relative 3D in **89/126 episodes** and
  lateral error in **76/126**, but Hole position in only **3/126**. Relative-vector
  improvement therefore does not establish better stationary-Hole memory.
- Moving-Peg position error is 3.70mm versus GNN 4.09mm. In the 125-frame fast
  direction-change proxy slice, it is 3.03mm versus 2.88mm, a small degradation.
  Median translational lag is zero frames; 4/126 episodes have positive fitted
  lag. These are offline GT-based diagnostics, not measured controller latency.
- With no observed BOX-or-HOLE support (1,849 frames), Mamba relative error is
  16.75mm and lateral error 11.56mm; relative 3D <3mm is only 0.81%.
- The GT alignment-geometry proxy covers 782 frames: relative error 6.01mm,
  lateral error 4.34mm, lateral <3mm 45.01%, relative <3mm 35.68%. It is not a
  grasp-phase label or a closed-loop success rate.
- PEG node count never reaches zero, and typed HOLE node count is zero in all
  validation frames. Target support is represented by BOX nodes. These counts
  cannot establish physical Peg/Hole visibility or absence of aperture geometry;
  no GT visibility or before/after-grasp phase labels were supplied.
- GRU is now included with matching evaluation data/protocol. Its unequal
  training features, objective, context and selection remain explicit in the
  audit above; these measurements do not isolate a backbone effect.

Evaluation run: `d159dcc6d8464253a76ab9095be5a221`; context-analysis run:
`56bb89f31aa1414495e0f89a2a619ae9`. Both use the shared ExperimentLogger and
new output directories, preserve the selected checkpoint, and read no final-test
frames. No further training or publishing was started.

Raw results:
[four-system comparison](../outputs/temporal_insertion_validation/7113412787dd4fbd8e0883e338bd0705/summary.json),
[comparison config and provenance](../outputs/temporal_insertion_validation/7113412787dd4fbd8e0883e338bd0705/effective_config.json),
[GRU source compatibility](../outputs/temporal_insertion_validation/7113412787dd4fbd8e0883e338bd0705/gru_code_compatibility.json),
[GRU training history](../outputs/temporal_insertion_validation/training/gru_history.json),
[streaming summary](../outputs/temporal_insertion_validation/d159dcc6d8464253a76ab9095be5a221/summary.json),
[context comparison](../outputs/temporal_insertion_validation/56bb89f31aa1414495e0f89a2a619ae9/summary.json),
[evaluation provenance](../outputs/temporal_insertion_validation/d159dcc6d8464253a76ab9095be5a221/provenance.json),
[training loss history](../outputs/temporal_insertion_validation/training/history.json),
and [loss curves](../outputs/temporal_insertion_validation/training/loss_curves.png).
Full prediction arrays remain on instance 59441593 under
`/workspace/temporal_insertion_mamba/validation/d159dcc6d8464253a76ab9095be5a221/predictions.npz`
and the matching `validation_context` output directory.

## GRU commit verification

On 2026-11-07, the GRU commit candidate was exported independently of the pending
Mamba and Hybrid implementations. All 50 Hole/Pose/precision GRU, learned Gate
and temporal-memory tests passed against that candidate. The 11 GRU YAML recipes
also resolved to callable root-entrypoint workflows without Mamba/Hybrid sources.
The existing Logger/artifact suites ran separately from the repository with Git
metadata available: all 36 tests passed. These checks used CPU fixtures and tiny
cache-based training/evaluation; CUDA/MPS, live simulator execution, full-dataset
training and external publishing were not run for this commit.

## Mamba commit verification

On 2026-11-07, the Mamba commit candidate was exported independently of the
pending Hybrid implementation. The combined Mamba, GRU, learned Gate and temporal
memory suite ran 64 tests: 63 passed and the official CUDA-kernel parity test
was skipped. The suite includes CPU reference forward/streaming/gradient checks,
matched-GRU training/comparison, parent-disjoint cache validation and a live
ManiSkill CPU RGB-D smoke. The existing Logger/artifact suites ran separately
from the repository: all 36 tests passed.

All 15 temporal YAML recipes resolved to callable workflows without Hybrid
sources; local feature links and prior raw artifact links were checked. No
CUDA/MPS execution, full-dataset training, remote launch or publishing was
performed for this commit.

## Repository layout verification

On 2026-11-07, the temporal additions were grouped under `temporal/` in the
model, data, training, evaluation and config directories. Models retain their
weights/state keys and observation contracts. AST comparison confirmed unchanged
model, loss and episode-data definitions; all 19 YAML recipes retained their
values apart from relocated entrypoints/config references. Fourteen temporal
recipes also produced identical effective settings and Torch RNG states before
and after configuration extraction.

The related workflow tests ran 88 cases: 86 passed and two CUDA-only cases were
skipped. Logger/artifact tests ran separately in a fresh process: 36 passed.
The suites include tiny synthetic CPU training/evaluation, a live ManiSkill CPU
RGB-D check, and a regression rejecting calculation changes during historical
source-path verification. No CUDA/MPS execution, official CUDA Mamba kernels,
full-dataset training or remote publishing was performed during this cleanup.

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_experiment_logger tests.test_experiment_artifacts -v
```

The workflow command is listed in [the Graph README](../graph/README.md).
The existing 12,841-frame Hybrid reference was validated and its frozen branches
restored on CPU; the saved Hybrid diagnostic inputs also passed validation.
This was input/checkpoint validation, without full reference inference. All 83
raw artifact links from the original guides were preserved, and local feature
links and recipe dispatch targets were checked.
