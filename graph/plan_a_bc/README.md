# Plan A cached graph BC with periodic live rollout

Choose the common geometry frame explicitly:

```python
geometry = CurrentGeometry(assets, schema, coordinate_frame='world')  # absolute World
geometry = CurrentGeometry(assets, schema, coordinate_frame='ee')     # current TCP, default
```

After camera-local projection/PCA/Shape-First, points and normals are transformed
directly into the selected frame once. Packet, same-time observations, 3mm
deduplication, final FPS and Graph stay in that frame. The three robot nodes and
edge directions use it too. View/free-space tests compose frame-to-camera
matrices for projection without moving the stored cloud back through World.
Raw camera calibration, source camera points and robot sensor inputs remain
explicit audit inputs; scalar source-camera depth-jump retains its original
measurement definition. Both `triton_fused` and explicit CPU perception support
the choice. `ee` uses the current observation's TCP pose;
mixed frames and stale EE reference poses are rejected.

Set `geometry.coordinate_frame` to `"world"` or `"ee"` in a new experiment's
`config.json` for matching cache creation and live rollout. Cache identity now
records the effective frame and uses `graphs_common_coordinate_frame_v2`;
old caches are not reused. New diagnostic arrays are `point_xyz`, `point_f5`
and a `coordinate_frame` scalar, rather than labelling EE points `point_world`.
Current records expose `xyz`, `normal`, `coordinate_frame`, and for EE,
`reference_ee_pose`. Packet/Graph labels are `World_metres` / `EE_metres`.
World and EE graphs cannot be packed into one batch.

Standalone `PlanA` without a frame retains the historical diagnostic adapter;
pass `coordinate_frame` with a matching packet builder to use the unified path.
The shared persistent-map prototype remains World-only. The implementation and
CPU/CUDA verification are in the
[coordinate-frame report](../../docs/visual-graph-integration/10i_coordinate_frames_20261103.md).

Current `CurrentGeometry(assets, schema)` defaults to
`perception_backend='triton_fused'`: Torch CUDA projects only valid,
non-overlapping object-mask pixels, then compact Triton PCA5 normals/features
consume those CUDA points directly. CUDA and Triton are required; there is no
implicit CPU fallback. `perception_backend='cpu'` explicitly selects the NumPy
diagnostic route. Historical frozen source copies/caches retain their recorded
implementation. Standalone `PlanA` retains its historical packet builder unless
one is passed explicitly.

Sampling evidence and Shape-First now also run on CUDA: ordered external
contours, centroid-nearest voxel representatives, thinning and seeded FP64 FPS.
Only selected points/features are downloaded for CPU packet assembly. Python
still controls launches and reads convergence/cardinality scalars. The robot
filter, gate, sampling policy, point cap and graph-edge policy are unchanged.
No dense XYZ or CPU feature raster is rebuilt in this default path.
See the [GPU sampling report](../../docs/visual-graph-integration/10i_shape_first_cuda_58752018_20261103.md)
and [preceding geometry integration](../../docs/visual-graph-integration/10i_graph_perception_default_20261103.md).

The sampling API `sample_cuda(...)` itself now returns CUDA tensors, including
selected features and provenance, and downloads only when `device='cpu'` is
requested. This NumPy Graph adapter explicitly requests CPU output; final Graph
assembly has not been ported as part of the sampling-output change. See the
[output-device comparison](../../docs/visual-graph-integration/10i_shape_first_device_output_58752018_20261103.md).

Independent opt-in experiment. This package consumes the **frozen09-D v2**
`graph.plan_a_geometry` and its preserved dependencies from an isolated source
directory. A clean HEAD checkout alone does not provide that opt-in source.
It never imports `graph.box_persistence` or instantiates a persistent surface map.

The geometry remains C3 + A-SF, cap128/object, current Main+gated Wrist Peg,
current Main Box only, original robot filter/3mm representative dedup/3D edges,
Node21 and Edge4. Geometry thresholds are not policy-training hyperparameters.
`geometry.py` only supplies current calibrated sensor/robot inputs. Its URDF FK
is checked against recorded09-D SAPIEN link transforms and every original robot
observation. It preserves individual collision shapes, including finger pads.

`prepare.py` audits the existing120 successful HDF5 episodes and original96/12/12
split. It reproduces Task08 Full B action statistics exactly and reuses the
preserved Train96 Robot76 normalizer from `initial.pt`. No original files change.
`cache.py` reveals RGB frames sequentially through the original causal streaming
DINO/SAM2 adapter. It stores all pre-GNN graph tensors, Robot76, native Action7,
node/edge pointers, pixel/F5 provenance, current masks and gate traces. Object
absence remains empty. T+1 terminal observations have no action label. Dataset,
geometry and source identities are required for cache reuse. Test graph creation
is locked until both policies and validation checkpoint selections are frozen.

`model.py` keeps the existing Readout320 + Presence2 + Robot76 -> encoder128 ->
GRU144 -> head7 path. E0 has two edge-aware64D layers; E1 has no message layers.
Their common initial weights match, but parameter counts differ. Both are fresh
BC runs. The unused E1 edge embedding remains in the original encoder schema;
it does not influence the forward result or receive a gradient.

`train.py` uses Task08 AdamW3e-4/weight_decay1e-5/batch2/sequence32/clip1 and the
original epoch cosine50 schedule, seed42 and exactly11,487 updates. It carries
hidden state only within each episode, detaches at TBPTT boundaries, and masks
padding. Validation Method B loss is evaluated every250 updates plus registered
rollout steps; first minimum selects validation best. This criterion/cadence
differs from Task08's native-MSE epoch selection and is disclosed in comparison.

At500 (3-step integration smoke), every1000 updates, and11487, the trainer saves
an immutable checkpoint and blocks on a separate live rollout process. Normal
evaluations use Train30003 + Dev42000/42001; even2000-step checkpoints and the
last use Train30003 + Dev42000..42004. Each Dev5 evaluation includes Dev2 without
duplication. Training weight/optimizer/scheduler/RNG/hidden/cursor are compared
before and after evaluation. Smoke weights are discarded before Full training.

`rollout.py` uses new simulator observations, never expert graph caches. It
resets GRU/tracker state, checks native controller entry and actual preprocessing,
and checks zero physics advancement during perception/inference. GT exists only
in the evaluation branch. Reach is diagnostic TCP–peg proximity<=30mm. Grasp uses
the installed predicate. Lift requires grasp and >5mm for3 consecutive steps;
it is not a claim of stable4cm lift. Alignment uses official hole-frame transverse
head coordinates plus axis angle<=10deg. Insertion uses official `info.success`.
Full rollouts run to horizon250 or environment terminal. Videos show observation
and action timestep t, with predicates explicitly labelled post-action t+1.

`pipeline.py` executes original-data hash gate, contracts, two-episode causal
pilot, all Train96/Validation12 caching, real-data gradient/save-resume/live smoke,
E0 training, E1 training, selection freeze, Test12 cache/evaluation, and reporting.
Any contract/runtime failure preserves partial artifacts and blocks later stages.
Manipulation failure alone does not alter settings or stop training.

Example from the isolated source, with existing dependencies and model weights:

```bash
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B -m unittest graph.plan_a_bc.tests graph.plan_a_geometry.tests -v
python -u -B -m graph.plan_a_bc.pipeline --root /path/to/new/comparison
```

The prepared directory must contain the audited configs/manifests, original
robot shape/URDF assets, runtime model/data paths, and verified original data.
The machine-specific deployment is supervised under its own service names. No
other process/service is stopped or altered. All Logger invocations receive new
UUIDs; comparison UUID is linkage metadata. W&B is disabled and no HF publishing
is called. `STOP` in the comparison directory cooperatively preserves training
at the next update. Explicit checkpoint resume creates a new training UUID.

`final.py` produces metrics, plots, Markdown/static HTML, Q1–Q10 answers and a
SHA256 artifact index from actual records, including incomplete/failed runs.
Read `pipeline_state.json` and `verification.json` for executed scope; the
presence of this implementation does not mean Full cache/training/rollout ran.

The training consumer is independent of geometry creation. A future separately
authorized geometry-memory comparison would need a new provider and identity,
with the same frozen training/evaluation procedure. That integration is not
implemented or validated here.
