# Graph representation feasibility prototype

## Scope and runtime

This implementation is owned by the Graph team under `graph/prototype/`.
It reuses the shared environment factory and raw GT accessor. It does not modify
Visual/Recurrent implementations or impose a shared final architecture.
The exact runnable commands are in [the prototype README](../graph/prototype/README.md).

Files created under `graph/prototype/` (28 total):

| Purpose | Files |
|---|---|
| Geometry/adapter | `geometry.py`, `surface_sampling.py`, `maniskill_gt.py` |
| Representation/models | `graph_builder.py`, `models.py`, `normalization.py`, `metrics.py` |
| Data/training | `data.py`, `replay_demos.py`, `prepare_dataset.py`, `train.py` |
| Inspection/evaluation | `inspect_env.py`, `render_diagnostics.py`, `visualize_graph.py`, `evaluate.py`, `validate_expert.py`, `compare.py` |
| Support | `__init__.py`, `common.py`, `visualization.py`, `config.toml`, `requirements.txt`, `README.md` |
| Tests | `tests/__init__.py`, `tests/fixtures.py`, `tests/test_geometry.py`, `tests/test_models.py`, `tests/test_data.py` |

This report is also new; root `README.md` was updated with the team prototype
links. Pre-existing uncommitted `graph/gt_nodes.py`, its tests, camera example
changes and `.vscode/` were preserved.

Executed on macOS / Apple M4 Pro with Python 3.11.13, PyTorch 2.14.1,
ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0, h5py 3.16.0 and
Matplotlib 3.11.2. Simulation and training ran on CPU (`physx_cpu`).
Rendering uses SAPIEN's macOS Vulkan/MoltenVK path; this does not mean CUDA or
GPU training ran. PyTorch reported both CUDA and MPS unavailable.
No CUDA/vectorized simulator path was executed. No PyTorch Geometric was added.

## Installed API verification

`inspect_env` executes reset and a zero action step in `pd_ee_delta_pose`,
prints the robot/backend, recursively records state-dict observation shapes,
all raw GT poses/dimensions, controller configs and action bounds.
The robot is `panda_wristcam`; state observation has shape `[1,43]`.
Action space is float32 `Box(-1,1,(7,))`.
Inspection artifacts for reset seeds 0, 1 and 2 include PNG, geometry PT and JSON.

The installed controller uses robot-root axes for translation and root-aligned
body rotation. Translation commands scale normalized values by 0.1 m.
Rotation components are norm-clipped to 1, multiplied by **rot_lower = -0.1**,
converted with XYZ Euler angles, then left-multiplied onto the current TCP
orientation. These are not joint actions, quaternions or axis-angle targets.
Gripper -1/+1 corresponds to mimic-joint targets -0.01/0.04 m.
Training regresses the recorded normalized commands directly. Rollout clips all
seven outputs to action-space bounds and lets ManiSkill apply rotation norm
clipping and its own IK/controller.

The API sources examined were the installed task, Panda robot, `pd_ee_pose.py`,
collision mesh utility, replay utility and Panda v3 URDF. Official references:
[task source](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/peg_insertion_side.py),
[Panda](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/robots/panda/panda.py),
[replay](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html),
[trajectory format](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html),
[BC examples](https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/bc).
Installed package behavior and runnable checks determine this prototype's API.

## Geometry and sampling

`GeometryState` contains float32 tensors with a leading batch dimension B.
Positions/local surface coordinates/half extents are metres; normals are unit
vectors; poses are `[xyz,qw,qx,qy,qz]`. World axes are used for positions/normals.
The raw quaternion is normalized before applying a rigid transform.

| Fields | Shape | Frame/meaning |
|---|---|---|
| ee_pose, root_pose, peg_pose, box_pose, hole_pose | B × 7 | World pose; hole frame includes randomized box-local offset |
| left_fingertip, right_fingertip | B × 3 | Rubber-pad collision centres in world coordinates |
| gripper, qpos | B × 2, B × 9 | Finger joints in m; arm joints in rad; active joint order |
| peg_half_size, box_half_size, hole_offset | B × 3 | Peg/box local extents and box-local aperture offset |
| hole_radius | B × 1 | Square aperture half width |
| peg_local, hole_local | B × P × 3 | Peg-local points and **box-local** tube surface points |
| peg_points, hole_points | B × P × 3 | Rigidly transformed world points |
| peg_normals, hole_normals | B × P × 3 | Rotated world normals |

Finger link origins are different from pad centres. Installed Panda v3 URDF
rubber boxes have local centres `[0,+0.00758,0.04525]` and
`[0,-0.00758,0.04525]`. The adapter reuses the shared accessor and verifies
these centres against actual SAPIEN finger collision meshes, then applies each
finger link pose. These nodes approximate the rubber pad location; they are
not dynamically inferred contact points or the most distal geometric point.
Camera overlays distinguish pad centres from white finger-origin markers.

The peg uses six exposed rectangle faces, sampled deterministically in fixed
round-robin patch order. The default 64 samples cover the +x insertion head,
tail and four side faces. Hole geometry is the **union of four collision boxes**.
Its exposed boundary is decomposed into 16 patches: four outer walls, four inner
walls and four rim rectangles on each x end. Default 64 samples allocate four
to each patch. Overlaps/internal primitive faces are excluded, and the aperture
interior is never filled. This prioritizes inner walls/rims rather than strictly
area-uniform sampling. UV coordinates use seed 17 with one stratified dimension.
This sparse discretization does not explicitly add every edge/corner.

Peg dimensions, outer tube dimensions, hole radius and offsets come from each
live reconfigured episode. Box half extents are all the current peg half length,
as in the installed task. Samples are cached only while these dimensions and
offsets are identical; current poses are applied every call. Peg count must be
at least 6, hole count at least 16; both are configurable.

Actual primitive bounds and reconstructed bounds agree within `5.59e-9 m` over
the three inspection seeds. Pad world centres agree within `2.98e-8 m`.
Tests check that a tiny inward displacement of each hole sample enters the
collision union and an outward displacement leaves it, including inner-wall
normal direction. Seeded shape changes, transform/inverse transform, stable
ordering and empty aperture are also tested. Diagnostic PNG are in
`outputs/graph/inspection/`, `camera_diagnostics/`, and `demonstration_graphs/`.

## Features and graph definition

Default surface graph has 131 nodes: TCP, left/right pad, 64 peg and 64 hole
surface nodes. `NodeType` and `EdgeType` are explicit enums. Per-node 14-vector:

```text
5 node-type indicators
3 position coordinates: world position minus current TCP position, WORLD axes
3 object-local surface coordinates (robot nodes: zero)
3 world surface normals (robot nodes: zero)
```

Shared 81-vector context contains gripper joints, robot qpos, TCP and root world
position/rotation matrix, peg/box/hole TCP-relative position and world rotation
matrix, peg/box extents, aperture radius and box-local aperture offset.
Orientation/shape information is preserved even in the centre baseline.
Rotations are represented by nine matrix entries to avoid quaternion sign
ambiguity in learned features. This prototype deliberately includes robot
proprioception in all three models.

Edges are directed and coalesced; reverse edges exist with negated xyz vectors:

1. TCP ↔ each pad (kinematic).
2. Within each object's surface: local-coordinate kNN, k=8, plus a nearest
   earlier-index tree. The tree ensures connectivity when kNN alone separates
   opposite faces. Some added tree edges can span faces.
3. Each pad ↔ its four nearest peg samples.
4. Each peg sample ↔ its four nearest hole samples.

Cross-object kNN has no radius cutoff, so distant objects remain connected.
The centre graph contains five nodes and ten directed edges: TCP/pads,
pads/peg, peg/hole. Edge features have eight components:
`destination_xyz - source_xyz`, Euclidean distance, four relation indicators.
They contain information computable from the current node geometry.
`GraphBatch` uses flattened global edge indices; batches never cross-connect.
Its `stats` reports node/edge counts and construction latency.
The seed-0 initial surface graph had 1,800 directed edges after coalescing;
the edge count changes with current interactions.

Node continuous features and context are standardized using only train
episodes; categorical node indicators retain 0/1 values. All three models share
these statistics. Edge xyz/distance remain in metres. A and C receive exactly
the same normalized node features/context, tested by exact tensor equality.
A flattens them in fixed node/sample order; C supplies them to the graph.
B uses the same context and five centre/robot nodes, with zero local coordinates
and normals. It receives all shape/orientation/task-goal information but has no
surface representation.

## Encoders and BC

Flat encoder: input width 1,915 → 128 → 128 → 128. GNNs: node MLP → three
edge-aware message-passing layers, hidden 128, source/destination node and edge
MLP, mean incoming aggregation, residual node updates and LayerNorm/ReLU.
Readout concatenates the three robot embeddings, peg mean, hole mean and context
before an MLP maps to 128. The action head is identical for all models:
128 → 128 → 7. There is no Recurrent or Visual model in this implementation.

Flat has 295,687 trainable parameters; both GNNs have 445,063. This difference
is a capacity confound. `flat_width=192` supports a near-matched follow-up;
that extra experiment was not run for the reported pilot.

Official download contains 1,000 motion-planning demonstrations plus RL data.
The first 100 motion-planning demonstrations were replay-converted from
`pd_joint_pos` to `pd_ee_delta_pose` using the installed official utility on two
CPU worker processes. 91 successful replays were retained. This subset is not a
random sample of the whole dataset, and successful-conversion filtering can bias
the data. Source/conversion hashes are recorded in the dataset manifest.

The 91 episodes contain 13,958 samples, with episode/reset-group split 73/9/9
under seed 0 (approximately 80/10/10). Every reset explicitly reconfigures
randomized geometry. All T+1 snapshot observations are validated after restore;
only states 0 through T-1 are paired with actions 0 through T-1.
No future action/state, reward, success flag or phase enters policy features.
Three independent full action replays succeeded and exactly matched all stored
observations. Maximum restored observation difference across the dataset was
`1.93e-7`.

Training: seed 0, CPU, four Torch threads, batch 32, AdamW lr 0.001,
weight decay 1e-5, cosine schedule, equal unweighted seven-component MSE,
10 full epochs / 3,530 optimizer updates for each model. A separate fixed seed
generator controls batch order so model parameter count cannot change ordering.
Each model selects the lowest validation MSE checkpoint. Targets keep the
controller's normalized scale. Held-out test data is accessed only in evaluation.
Eight-batch real-demo smoke runs and small analytic overfit unit tests preceded
full training; analytic fixtures are never used as experimental performance data.

## Evaluation and diagnostics

Offline test evaluation reports raw action MSE, translation/rotation component
MSE/RMSE and gripper MSE/MAE. It also reports bounded-action MSE and physical
controller-command errors after clipping/scaling. Command errors are not
realized TCP motion errors. CPU batch-1 latency separates geometry/graph feature
construction from inference, sampled across 16 test states after warm-up.

Rollouts pair all nine test reset seeds and initial geometry hashes across
models. The explicit horizon is 250 policy steps, because expert lengths are
105–219 while the registered task limit is 100. Results at this extended horizon
must not be presented as the default 100-step ManiSkill benchmark.
The policy ends on success, termination or its common horizon. Records include
episode length, head transverse/axis error, insertion progress, contact-based
grasp reports after steps, grasp losses, heuristic phase and failure category.
Insertion progress is clamped penetration of peg head between the near/far tube
faces, not a reward or the task's exact success condition.

Phase labels are diagnostic heuristics from current geometry: near/opening
grasp proxy, peg elevation, head transverse position and insertion depth.
Counts are approach 3,082, grasp 4,530, transport 514, alignment 4,082 and
insertion 1,750. They are not oracle controller inputs or definitive task phases.
Alignment evaluation starts at the first alignment sample (insertion fallback)
in each test episode. Past expert actions reconstruct that state before policy
control begins, preserving controller/contact state. No expert future action
is fed to the policy. Warm rollouts receive 250 additional policy steps.

## Problems discovered and handled

- macOS renderer can change process cwd. The replay wrapper resolves absolute
  trajectory paths and uses the shared MoltenVK setup.
- `CombinedController` exposes `configs`, not a single `config`. Finger `_objs`
  contain PhysX articulation link components rather than actor entities.
  Inspection was corrected against the live objects.
- Installed CPU replay's first-state replacement uses original state **1**
  while saving observation **0**. Initial observation drift reached 0.00352.
  Extraction obtains source state 0 from original H5, matches initial observation
  (including duplicate source seeds), and validates all restored observations.
  Installed package code is not patched.
- Snapshot state does not restore contact caches/controller/physics history.
  Phase extraction therefore uses geometry only. Isolated snapshot warm resume
  also made expert actions fail: 4/9 successes, versus full expert 9/9.
  The final warm protocol uses expert prefix replay, and the expert positive
  control is rerun with that same initialization. Snapshot-only control results
  are saved separately in `outputs/graph/expert_snapshot_control.json`.
- Point sparsity, normalization outside the demo distribution and compounding
  BC error remain feasibility limitations rather than solved problems.

## Actual results

The single-seed pilot used 11,274 train, 1,325 validation and 1,359 test samples.
All three selected epoch 10. Measured CPU results:

| Model | Action MSE | Translation RMSE | Rotation RMSE | Full rollout SR | Params | Total latency ms |
|---|---:|---:|---:|---:|---:|---:|
| Flat MLP | 0.004505 | 0.039098 | 0.065927 | 0/9 | 295,687 | 0.200 |
| Center GNN | 0.003691 | 0.037569 | 0.067663 | 0/9 | 445,063 | 0.744 |
| Surface GNN | 0.004475 | 0.039421 | 0.069151 | 0/9 | 445,063 | 2.210 |

Translation and rotation RMSE in this table are normalized per-component action
errors. Full rollout uses 250 steps; all also have zero successes by 100 steps.
Latency is batch 1, 16 representative test states, 100 measurements, four Torch
threads, sampled sequentially after all training/rollout processes finished.

| Model | Feature/graph ms | Network ms | Best val MSE | BC time s | Gripper MSE |
|---|---:|---:|---:|---:|---:|
| Flat MLP | 0.174 | 0.025 | 0.005623 | 6.12 | 0.013907 |
| Center GNN | 0.210 | 0.534 | 0.005347 | 21.34 | 0.007870 |
| Surface GNN | 0.612 | 1.598 | 0.005571 | 332.13 | 0.012319 |

Training timings include overlapping CPU workload during the initial runs.
Feature timing excludes simulator GT access and includes normalization; GNN
timing includes edge construction/coalescing. Detailed geometry construction
stats are in evaluation JSON. Bounded translation-command RMSE was 3.910,
3.757, 3.942 mm for A/B/C; rotation-command Euler RMSE was 0.006570,
0.006766, 0.006907 rad. These are command errors, not actual tip deviations.

| Model | Alignment start SR | Mean final transverse head error m |
|---|---:|---:|
| Flat MLP | 0/9 | 0.12659 |
| Center GNN | 0/9 | 0.08888 |
| Surface GNN | 2/9 (22.2%) | 0.04482 |
| Expert positive control | 9/9 | N/A |

The expert positive control succeeded 9/9 from both full and alignment starts,
with zero prefix observation error. Expert final transverse error was not
recorded by the positive-control CLI, hence N/A. Surface alignment successes
were episode 41 (seed 63, 44 policy steps) and episode 59 (seed 86, 215 steps).
The test seeds were 50, 54, 63, 81, 83, 86, 102, 114 and 116; all model starts
passed exact initial-geometry hash comparison. An alignment run lasting 215
steps would fail at a 100-policy-step limit; this is explicitly an extended
horizon evaluation. Warm-start and full-task SR are different tasks.

| Phase | Test samples | Flat MSE | Center MSE | Surface MSE |
|---|---:|---:|---:|---:|
| Approach | 285 | 0.005262 | 0.005054 | 0.004585 |
| Grasp | 439 | 0.008053 | 0.005770 | 0.007937 |
| Transport | 71 | 0.008985 | 0.006720 | 0.009664 |
| Alignment | 397 | 0.000923 | 0.001221 | 0.001301 |
| Insertion | 167 | 0.000494 | 0.000485 | 0.000527 |

Center GNN achieved the lowest overall offline error, especially in the
grasp/transport samples. Surface GNN and Flat MLP had nearly equal overall
action MSE. Flat's lower alignment prediction MSE did not translate into warm
rollout success. Surface's two alignment successes provide a limited observation
worth checking with more seeds/episodes, not evidence of general superiority.
All full-task BC policies failed. Common failure categories were no detected
grasp (A/B: 4 each, C: 1), grasp lost (A/C: 1 each) and remaining
alignment/insertion failure. These coarse categories come from rollout traces,
not manual certainty about causation.

The prototype establishes a runnable comparison and exposes compounded BC
errors. It does **not** establish that surface graphs improve full manipulation
control. Capacity differs from Flat MLP; only one training seed, nine paired
episodes, 91 selected demonstrations and 10 epochs were measured. Multiple-seed,
parameter-matched, 30-epoch and GPU experiment metrics are N/A because those
experiments were not run. No result is extrapolated from smoke tests.

Full artifacts: `outputs/graph/comparison/results.md`, `results.json`,
`errors.png`, `phase_errors.png`; per-model `evaluation/*.json`;
`runs/*/{best.pt,history.json,training.json,loss.png}`; failure MP4 under
`evaluation/videos/` and a selected Surface alignment success MP4 under
`success_example/videos/`. The saved report records exact values rather than
only rounded table entries. Twelve Graph-team tests and eight existing shared
GT/environment tests executed successfully (20 total); unavailable CUDA was
not treated as a passing test. The replay wrapper also ran a separate copied
source episode and retained one successful conversion.

## Provisional embedding interface and next experiments

```python
from graph.prototype.maniskill_gt import GTAdapter
from graph.prototype.graph_builder import GraphBuilder
from graph.prototype.models import GraphEncoder

geometry_state = GTAdapter()(env)
graph = GraphBuilder()(geometry_state)
embedding = GraphEncoder()(graph)  # [B,128], untrained example
```

For a trained checkpoint, load its `FeatureNormalizer` and encoder weights.
`normalizer.apply("surface", geometry_state, graph_builder)` constructs and
normalizes the graph; `policy.encoder(graph)` returns `[B,128]` without invoking
the action head. These Graph-team tensors do not declare a final shared interface.

Visual integration will need explicit world frame/calibration, quaternion
convention, object identity/sample order, normals/confidence, aperture offset and
visibility semantics. Robot qpos/gripper/root pose should come from proprioception
when GT geometry is replaced. Recurrent integration will need timestamps,
episode/reset handling, missing-object behavior and the agreed 128-vector
normalization/encoder version. Neither integration is implemented here.
Next experiments should retain held-out episode splits, match encoder capacity,
repeat seeds, expand expert coverage, examine phase errors and compounding
rollout failure, and assess surface density/cross-edge sparsity on training or
validation data before any further test comparison.
