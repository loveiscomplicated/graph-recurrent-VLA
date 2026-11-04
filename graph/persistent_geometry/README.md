# Task09-C independent geometry diagnostic

The historical persistent-map contract remains World-only. Current-only callers
can now explicitly select a common `world` or current-`ee` frame through
`CurrentGeometry(..., coordinate_frame=...)`; see the
[current adapter](../plan_a_bc/README.md). `SurfaceObservation.coordinate_frame`
defaults to `world`; EE observations carry `reference_ee_pose`. Use
`current_records(..., fuse_views=False, coordinate_frame=...)` for these current
observations. Frame-aware records expose `xyz`/`normal` with explicit metadata;
they do not place EE coordinates under `world_xyz`. Mixing frames or EE reference
poses is rejected, and the persistent map rejects EE observations. Legacy
World observations/records and standalone diagnostics retain their old defaults.

The default current Graph adapter uses the Torch implementation in
`tensor_fusion.py` for selected-observation view checks. Prepare fresh
`TensorViews` with `prepare_tensor_views(cameras, masks, coordinate_frame=...)`,
then call `consistent_views_tensor(observations, views)`. Observation arrays,
keep decisions and conflict-count tensors remain on the chosen device. A
validated CUDA packet can enter via `observations_from_tensor_packet`;
`tensor_observations` is an explicit upload for NumPy compatibility.
`numpy_observations` explicitly downloads compact selected results, optionally
returning aligned structural flags for the existing Python record builder.
Only static identity/calibration metadata is validated on CPU; depth conversion
and point-dependent math use Torch. The NumPy `consistent_views` remains the
reference/CPU diagnostic and existing pre-sampling gate dependency.
Tensor observation XYZ/normals/F5/origin use float64, source pixels int64 and
validity/structural flags bool. These match the existing packet-derived NumPy
observations. Compatibility upload canonicalizes other accepted NumPy dtypes.
Audit `count` is shape metadata; `unknown_conflict` remains an int64 tensor until
the caller explicitly materializes it for logging.

The live `CurrentGeometry._build_prepared` path calls `prepare_tensor_gate`
with resident raw/clean masks and optional float64 depths from robot preparation.
It validates/prepares one current `TensorViews` snapshot. Each prepared camera
also holds World calibration; `views.world_views()` shares depth, masks and
intrinsics while selecting those World matrices without uploads. The wrist gate
always judges World geometry, including its axis-aligned component extent,
regardless of the selected Graph frame. Graph observations still use the requested
EE (default) or World frame after gating. Reuse is confined to the same snapshot;
foreign episode/time/pose identities are rejected by the gate.

The CUDA gate packs all trace statistics into one host readback and checks label
convergence after up to four integer propagation steps. This preserves the
original radius predicates, canonical component order, iteration limit and full
trace. CUDA remains the default whenever available; GPU failures propagate and
only CUDA absence selects the existing CPU gate. The NumPy compatibility adapter
uploads each clean mask once and explicitly downloads the accepted mask.

The World gate shares each cell-pair compaction index between both endpoint
arrays. A stable cell-code sort feeds `unique_consecutive`, counts and the
original-point inverse mapping; canonical component labels retain their original
first-point order. CUDA World-point reconstruction uses one FP64 Triton kernel
with round-to-nearest division and FMA disabled. CUDA label propagation reads
immutable input labels, atomically reduces into a separate copy, then pointer
jumps into a separate output buffer. Its per-block change flags check the last
step of each batch, preserving complete convergence and the original iteration
limit. The existing CPU gate and explicit CPU test helpers remain available.

Opt-in feasibility implementation for RGB-D geometry, object-local surfels,
multi-view association, surface graphs and diagnostic geometry encoders. It does
not change the shared simulator, existing vision defaults, BC datasets, policy
checkpoints or the production Graph/GNN interface. No GRU or action head is
constructed by these runners.

Surface-edge free-space checks query the original signed-int16 `depth_raw_mm`
only at projected pixels. They apply the same float64 mm→m/nonpositive→NaN rule
without constructing an image-wide depth array for `.shape` or indexing. No
cross-frame depth cache is kept. Projection, strict6mm threshold, edge order and
audit remain unchanged. This CPU fix is measured independently of a GPU Graph
port; see the [implementation and benchmark](../experiments/graph_depth_lookup_58752018_20261103_v1/README.md).

The practical `pipeline.GeometrySystem` accepts synchronized camera observations,
predicted masks, current robot observations and known robot link geometry. Its
input whitelist rejects actor poses and segmentation sidecars. `oracle.py`
separately implements evaluation-only GT-motion variants. Masks and poses from
the simulator are stored in separate capture archives and used only by metrics,
diagnostic targets and explicitly labelled oracle conditions.

## Representations

| Variant | Robot filter | Memory | Camera input |
|---|---|---|---|
| G0 | none | none | original Main Shape-First/F5/Raw EDE |
| G1 | per-link collision projection/interior | none | Main |
| G2 | same | object-local surfels | Main |
| G3 | same | none | Main + Wrist |
| G4 | same | object-local surfels | Main + Wrist |
| O1 | same | estimated registration | both, GT visible masks |
| O2 | same | GT relative rigid motion | both, predicted masks |
| O3 | same | GT relative rigid motion | both, GT visible masks |

Oracle motion transports only previously observed surfaces. It never creates
full-mesh nodes. Known robot collision shapes are handled individually; the
filter does not remove a large radius or a convex hull around the whole robot.

The original G0 packet and 14-feature graph are saved unchanged. The separately
labelled common schema has 21 features: original XYZ/F5/availability/validity14
plus status3, confidence, age, positional uncertainty and pose confidence.
Historical depth-jump and normal-variation fields are unavailable and invalid.
Current measured normals and positional uncertainty remain distinct from
historical pose-dependent estimates.

Persistent edges use a versioned 3D radius/k-neighbour rule with signed normals,
tangency, surface witnesses and camera free-space checks. They are never called
Raw EDE. Robot–object k4 and Peg–Box M16 reuse the existing helpers. Conservative
edge rejection can fragment the graph; `geometry_audit` reports this cost.

The primary GNN input has a maximum of128 nodes per object. It is a common cap,
not a claim that every condition has the same number of supported nodes.
Native variable-N maps and an additional geometry-only matched-count subset are
saved separately. Matched counts use predicted G0–G4 counts, never GT. An object
missing from any representation can make its matched count zero.

## Runners

Run from the repository root with existing dependencies and offline perception
checkpoints. Each runner creates its own `ExperimentLogger` UUID. Output roots
are parents of UUID directories; finalized inputs are required by downstream
runners.

```bash
python -m graph.persistent_geometry.collect --pilot --output /workspace/task09c/pilot
python -m graph.persistent_geometry.preflight --capture CAPTURE_UUID_DIR --output PREFLIGHT_ROOT
python -m graph.persistent_geometry.evaluate --capture CAPTURE_UUID_DIR --workers 4 --output GEOMETRY_ROOT
python -m graph.persistent_geometry.branch_eval --capture CAPTURE_UUID_DIR --geometry GEOMETRY_UUID_DIR --output BRANCH_ROOT
python -m graph.persistent_geometry.geometry_audit --geometry BRANCH_UUID_DIR --output AUDIT_ROOT
python -m graph.persistent_geometry.train --geometry BRANCH_UUID_DIR --updates 2000 --output TRAIN_ROOT
python -m graph.persistent_geometry.probes --training TRAIN_UUID_DIR --output PROBE_ROOT
```

`collect --branch-smoke` captures one development episode with same-pose repeats
and actual actor-pose interventions. All branches share the parent robot,
calibration and causal history through t−1. Invalid collision conditions remain
in the artifacts and are excluded from fitting/primary paired results. These
are live kinematic render interventions without a physics step, not dynamic
contact-force tests.

`collect --development` uses the fixed8 Train/4 Validation episodes and defers
reserved final seeds62091/62092. Only after `probes` writes
`frozen_selection.json`, run `collect --evaluation-only`, evaluate that capture
and its branches, then apply the frozen models:

```bash
python -m graph.persistent_geometry.final_apply --training TRAIN_UUID_DIR --probes PROBE_UUID_DIR --geometry FINAL_BRANCH_UUID_DIR --output FINAL_ROOT
python -m graph.persistent_geometry.cases --capture CAPTURE_UUID_DIR --geometry BRANCH_UUID_DIR --output NEGATIVE_ROOT
python -m graph.persistent_geometry.verify --runs UUID_DIR_1 UUID_DIR_2 --output READBACK_ROOT
python -m unittest graph.persistent_geometry.tests.test_contracts -v
```

The final runner has no optimizer or selection call. It checks freeze timestamps,
reserved seeds, actual reset-group separation and all weight hashes, and loads
development-only normalizers. A causal perturbation family is a parent physical
episode/timestep plus all its branches; those rows stay in the parent's split.

E0/E1/E2 use the same21-feature input, seed42, AdamW budget and validation rule.
E1 removes message passing; E2 adds a deterministic30-value summary of the same
input XYZ. Their diagnostic decoder predicts relative actor-position6 and
current visible-surface-summary30. No grasp/hole goal, phase, reward, action or
success target enters the loss. These are newly trained diagnostic encoders,
not rankings based on applying old BC checkpoints to a changed schema.

The opt-in encoder identifies TCP/left/right slots by node type when an input
permutation moves the robot nodes. It remaps packed edges before using the
existing structured readout. Canonical builder output uses the unchanged
forward path; the shared production encoder is not modified. The permutation
contract covers all nodes, including the three named robot nodes.

`report --state PIPELINE_STATE_JSON --output COMPARISON_ROOT` aggregates completed
runs, renders the final videos, audits real negatives, reads back source files
and creates a portable `sources/<uuid>` bundle with a path-remapping manifest.
It creates comparison facts; the reviewed Markdown/HTML interpretation is a
separate artifact and must be verified before delivery.

## CPU stage handoff

For the executed final evaluation, CPU array geometry ran on the user's M4 Pro
with one worker, while frozen CUDA inference stayed on Vast. The opt-in
`handoff.maybe_delegate` CLI hook checks
`/workspace/task09c/local_cpu_handoff.json`. Only enabled, explicitly listed
module/output pairs are delegated. Request/receipt files preserve the original
experiment UUID; every returned file is SHA-256 checked before the remote
runner continues. Disable the marker after the three stages finish. Without an
enabled matching marker these CLIs execute normally on the current host.

Capture/geometry references can be relative to matching sibling run roots.
Branch frame references are relative to their original run directory;
`verify.frame_folder` resolves that recorded origin before applying a bundle
path map. Transport does not change map math or frozen model selection.

Cross-host numerical identity is not guaranteed. The executed 20-frame audit
found identical semantic node/edge sets for G1–G4/O1–O3 within float tolerances,
but G0 Raw EDE edge sets differed across installed numerical environments.
The final report records this train/evaluation-host limitation. The measured
8.7x ratio describes that CPU prefix's per-frame wall time, not the complete
pipeline or transfer time. It is not a claim of bitwise graph equivalence.

## Executed environment and limits

The research execution uses Vast instance57640710, RTX3090, existing CUDA
perception/GNN dependencies, CPU PhysX and SAPIEN rendering. CUDA PhysX was not
executed. Local macOS work includes synthetic/array contracts and the final CPU
geometry evaluation, not a simulator/GPU execution claim. No dependencies are
installed or artifacts published by these
runners.

Partial surfaces leave rigid pose directions ambiguous. Registration explicitly
reports partial observability, failures and increasing uncertainty. A wrong
predicted object mask can still contaminate a map, and persistence can worsen
geometry or create ghosts. Current observations, historical estimates and GT
evaluation overlays must remain visibly distinct. The completed benchmark
report and its artifacts, rather than node count or probe error alone, determine
which representations have supporting evidence.

## Current-only CUDA Graph

The default `CurrentGeometry` triton_fused path uses `tensor_graph.build_graph_tensor`
after `budget_tensor(...,128)`. It reuses the current `TensorViews` depth/calibration
snapshot. Surface neighbor selection, distance/normal/tangency/witness/free-space
gates, interaction selection, Node21/Edge4 packing and outputs stay on CUDA.
Small host identity/calibration and three FK robot positions retain the validated
reference convention; scalar validation/cardinality checks can synchronize.

`Graph.packet['node_ids_encoded']` contains common episode/time/camera-name metadata
and CUDA camera/pixel tensors in node order. Surface audits are CUDA scalar
tensors. `tensor_graph.numpy_graph` explicitly returns the legacy CPU tensor and
string-ID metadata view; `tensor_records.numpy_records` separately exports record
dictionaries/provenance. These adapters are used for file writing, never by the
default Graph/model path. Explicit CPU perception and the historical persistent
map `build_graph` API keep their existing outputs and policy.

The bounded KD implementation targets the executed SciPy1.18.1/Linux compact
median-tree neighbor policy, including strict cutoff ties, for at most128 points
per object. It does not import or run CPU KD-tree code. Policy references:
[SciPy build](https://github.com/scipy/scipy/blob/v1.18.1/scipy/spatial/ckdtree/src/build.cxx),
[SciPy query](https://github.com/scipy/scipy/blob/v1.18.1/scipy/spatial/ckdtree/src/query.cxx).
Equal-distance choices can depend on the reference CPU standard library/version.
See the [executed Graph CUDA report](../../docs/visual-graph-integration/10i_graph_cuda_58752018_20261103.md).
