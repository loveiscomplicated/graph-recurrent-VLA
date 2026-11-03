# Plan A cached graph BC with periodic live rollout

The current cache/live frontends use **Terra Standard as the primary grounder**
and Grounding DINO when `OPENAI_API_KEY` is missing/blank or Terra returns
HTTP401/403. Environment values take precedence over the repository-root `.env`.
Terra jointly discovers Peg/Box in the current RGB and supplies SAM2 box prompts;
it runs only at initialization/missing-object scans. DINO is loaded lazily on
fallback. Startup, requests/results, authentication fallback and fallback model
loading are printed through the `[live-grounding]` logger. Other API/response
errors stop with sanitized diagnostics. See the [runtime contract](../../visual/runtime/README.md)
and [verification report](../../docs/visual-graph-integration/10i_live_terra_primary_20261103.md).
Per-camera `vision_trace.json` and episode/cache receipts record actual backend,
fallback reason and request counts. Existing ExperimentLogger identity rules
remain in place. Historical frozen deployments are not modified by this change.

The normal cache and live-rollout paths now start robot CPU preparation before
their current-frame SAM2 calls. `CurrentGeometry` defaults to CUDA robot filtering
for `triton_fused`, with a persistent `RobotFilterPreparation` worker. No opt-in
flag or experiment helper context is required. The callers supply a zero-argument
mask producer to `build`; the builder submits the current snapshot, invokes the
producer, then consumes the preparation and runs the GPU filter once.

```python
geometry = CurrentGeometry(assets, schema)
try:
    def masks_for_current_frame():
        return {c.camera: adapters[c.camera].process(
            c.rgb, episode, timestep, mask_device='cuda')[0] for c in cameras}
    graph, records, gate = geometry.build(cameras, data, surfaces, masks_for_current_frame)
finally:
    geometry.close()
```

SAM2 raw masks stay on CUDA until filtering. The unchanged CPU wrist gate still
receives both original raw masks and clean masks, so its raw-pixel denominator
is preserved. `geometry.last_raw_masks` provides that CPU raw-mask view for cache
artifacts/video; it is not an already-clean mask. `last_robot_schedule` reports
`before_vision` in these default frontends. Passing a ready mask dictionary is
still supported and reports `masks_already_available`; SAM2 has already finished
in that case. `robot_filter_backend='cpu'` explicitly selects the historical
filter for diagnostics; `perception_backend='cpu'` selects it automatically.
Full-screen filter diagnostics remain available through standalone `PlanA.prepare`;
the prepared-clean fast path explicitly has no full filter diagnostic dictionary.

Cache identity is now `graphs_common_coordinate_frame_v4_terra_primary` and records
the grounding policy plus geometry backend/schedule. Existing v2/v3 caches are rejected rather than silently reused
or overwritten. Original frozen experiment sources/runs are not resumed or migrated.
See the [default-path implementation report](../../docs/visual-graph-integration/10i_robot_overlap_default_20261103.md).

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
records the effective frame and uses `graphs_common_coordinate_frame_v4_terra_primary`;
old caches are not reused. New diagnostic arrays are `point_xyz`, `point_f5`
and a `coordinate_frame` scalar, rather than labelling EE points `point_world`.
Current records expose `xyz`, `normal`, `coordinate_frame`, and for EE,
`reference_ee_pose`. Packet/Graph labels are `World_metres` / `EE_metres`.
World and EE graphs cannot be packed into one batch.

The default `triton_fused` adapter now keeps selected packets and observations
on CUDA through the same-time view check. It uses
`graph.persistent_geometry.tensor_fusion`: Torch projection, signed-mm depth
lookup, opposite-mask/depth predicates and compact result tensors. Frame-local
depth/mask/calibration tensors are prepared once and shared across camera calls.
World/EE, strict4mm depth support and source/structural flags are preserved.
The result API stays on its input device. Default SF now continues through
`current_records_tensor` and `deduplicate_tensor`: measured XYZ/F5, structural
priority, object IDs and merged source provenance stay on GPU.
`budget_tensor(records,128)` performs the final per-object FPS on GPU, preserving
persistent-ID string seed/ties, short-input order, features and merged sources.
Final FPS now feeds `build_graph_tensor` directly: CUDA neighbor queries, surface
gates, free-space depth gathers, robot/Peg–Box interactions and Graph tensors.
The same `TensorViews` snapshot is reused through Graph construction.
`CurrentGeometry.build` returns `(CUDA Graph, TensorRecords, gate)` on this default
path. Graph IDs are `packet['node_ids_encoded']` camera/pixel tensors and common
identity metadata; audit counts are device scalars. `numpy_graph(graph)` and
`numpy_records(records)` are explicit CPU archival/inspection adapters. Cache
writing and diagnostic rollout dumps call them after construction; model input
packing retains CUDA tensors. Explicit CPU perception keeps CPU Graph/list output.
Explicit CPU perception and historical standalone PlanA retain the NumPy path.
See the [paired GPU measurement and verification](../../docs/visual-graph-integration/10i_view_consistency_cuda_20261103.md).

`PlanA.represent(..., ('SF',), tensor_records=True)` returns a `TensorRecords`
table with tensor-valued measurements and ragged source IDs. It requires the
tensor SF-only path. Optional `views=TensorViews` shares the current depth and
calibration snapshot; `PlanA.graph(tensor_records, data, cameras, views=views)`
requires that snapshot. Existing calls without the flag retain Python-list output
through the explicit post-dedup adapter. The kernel preserves the original
strict-radius, nontransitive greedy representative rule and stable source order;
it never averages XYZ/F5. See the
[records/dedup API and benchmark](../experiments/records_dedup_cuda_58752018_20261103_v1/README.md).

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

Sampling evidence and Shape-First also run on CUDA: ordered external
contours, centroid-nearest voxel representatives, thinning and seeded FP64 FPS.
The default adapter keeps view-checked observations,3mm-deduplicated records and
final128-point FPS, edges and Graph tensors on GPU. Python still controls
launches and reads convergence/cardinality scalars. The robot
filter and gate are outside this Graph change; sampling policy, point cap and
graph thresholds are unchanged. The bounded CUDA KD query preserves the tested
SciPy1.18.1/Linux neighbor policy, including cutoff ties. Other CPU standard
libraries/versions may choose different equal-distance neighbors; the new CUDA
path has no CPU KD-tree fallback. See the
[Graph CUDA report](../../docs/visual-graph-integration/10i_graph_cuda_58752018_20261103.md).
No dense XYZ or CPU feature raster is rebuilt in this default path.
See the [GPU sampling report](../../docs/visual-graph-integration/10i_shape_first_cuda_58752018_20261103.md)
and [preceding geometry integration](../../docs/visual-graph-integration/10i_graph_perception_default_20261103.md).

The sampling API `sample_cuda(...)` itself now returns CUDA tensors, including
selected features and provenance, and downloads only when `device='cpu'` is
requested. Standalone `make_packet` retains NumPy compatibility output; default
`CurrentGeometry` uses CUDA packets and keeps records resident through dedup.
Final FPS and Graph assembly stay on GPU. See the
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
Terra-primary/SAM2 adapter with DINO authentication fallback. It stores all pre-GNN graph tensors, Robot76, native Action7,
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
