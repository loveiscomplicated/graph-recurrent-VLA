# Current-observation sampling (Task 03)

`sampling.py` is an opt-in CPU NumPy sampling API inside Vision. It consumes only
`TrackedRGBDFrame` observations or extracted observation evidence. It does not
change the agreed Vision–Graph interface. Coordinates are still CV-camera metres;
this internal result is **not** the final EE-relative Graph tensor output.

```python
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.sampling import (
    SamplingConfig, extract_sampling_evidence, sample_points,
)

config = SamplingConfig(
    num_points=256, structural_ratio=0.25, min_points=(16, 16),
    allocation='observed_points', seed=0, orientation_threshold_deg=20,
    depth_priority_fraction=0.25,
)
for frame in iter_saved_rgbd('visual/data/step2/ep000.npz',
                            'visual/results/runtime_tracking/ep000/masks.npz'):
    evidence = extract_sampling_evidence(frame, estimator='pca5')
    selected = sample_points(evidence, config, method='S3')
    assert len(selected.points) <= config.num_points
```

These numeric values are experimental settings, not an adopted detector,
threshold, object allocation or node-feature design. `sampling='fps'` is the only
supported sampler. `num_points` and `seed` must be nonnegative integers, ratios
finite in `[0,1]`, and orientation threshold finite in `[0,180]` degrees. Object
minimums must be two nonnegative integers with sum no greater than the total,
even if one object is absent. A zero budget needs `min_points=(0,0)`.

## Observation and extraction

The Graph CUDA adapter `graph_perception.extract(frame)` produces the same
`SamplingEvidence` with compact XYZ. Consumers must call
`evidence.frame.geometry.points_at(source_pixels)` and `camera_depth()` instead
of directly indexing `geometry.xyz`. Mask/depth and feature rasters keep their
original image alignment. This adapter hands off to CPU only after masked CUDA
projection and compact `triton_fused` normals/features have both completed.

The current Graph default bypasses that CPU-evidence adapter. CUDA entrypoints
in `cuda_sampling.py` are `build_sampling_info_cuda`, `shape_first_cuda` and
their composition `select_cuda`. They preserve the current S3 strict >20deg,
external canonical contour order, nominal voxel scale and centroid-nearest
representative, ratio .5/phase0, and additional surface 97/95 policy. FP64 FPS
retains structural row order and first-source distance ties. Other historical
sampling arms/ratios remain available through their original NumPy APIs.

The current CUDA path limits external-component/walk processing to the raw
mask's bounding rectangle plus one background pad and restores original raster
indices. A GPU scan reads the full mask once to obtain four shape scalars.
Voxel centroid arithmetic is preserved; nearest representative selection and
tie handling are fused, and already sorted PCA sources skip redundant sorting.
Selection provenance reuses the projector's raster-to-compact row map.

FPS seeding uses exact FP64 seed-tile AABB lower bounds to skip distance tiles
that cannot improve a point block. For at least4096 points and at least16 new
surface rows, seed/update/argmax work replays through a CUDA Graph. Plans use
power-of-two point/seed capacity buckets and a device/dtype/stream/count key;
the process keeps at most8 plans. Each call copies current points and seeds
and materializes independent outputs. Work buffers are reused; observations
and selected results are not cached across frames. Smaller calls remain eager.

The first plan/JIT construction costs more than replay and must be separated
from warm timings. Bucket padding and spatial seed layout affect performance.
Full depth sorting was retained because the tested dual-kthvalue alternative
was slower. Component convergence still uses host scalar checks.
See the [refinement report](../../docs/visual-graph-integration/10i_shape_first_refinement_58752018_20261103.md)
for before/after timings, rejected variants and exact-output checks.

The next refinement vectorizes canonical-walk minimum searches/copies and skips
identity comparisons. Component compression writes both roots and flattened
parents and resets the union flag in one kernel. A CPU convergence check runs
after each pair of completed union rounds. It still verifies convergence;
there is no fixed-iteration approximation. FPS plans recompute seed-tile bounds
once per invocation and share them across point blocks. Entirely inactive
point blocks skip seed scanning. The prior small eager FPS path is preserved.
See the [further optimization report](../../docs/visual-graph-integration/10i_shape_first_further_58752018_20261103.md).

`sample_cuda(projected, features, depth, masks, intrinsic)` returns selected
points, metadata, source IDs and selection provenance as CUDA tensors on the
sampler's device. Its optional `device='cpu'` explicitly requests CPU tensors;
this selects the output device, while sampling computation still uses CUDA.
`selection_output` applies the same terminal contract to a raw `select_cuda`
result. `diagnostic=True` additionally includes full candidate/representative
source lists on the chosen device. No default output array is downloaded.

`numpy_selection` only converts tensors already on CPU and rejects CUDA tensors.
`to_cpu_selection` is the compatibility wrapper explicitly requesting CPU output.
The current NumPy Graph consumer in `graph_perception.make_packet` explicitly
calls `sample_cuda(..., device='cpu')`; porting final Graph assembly is separate
from this sampling-output API. See the
[output-device measurement](../../docs/visual-graph-integration/10i_shape_first_device_output_58752018_20261103.md).

`vision_packet.complete_selected_vision_packet`
validates selected source identities and reuses the same EE conversion/v2
packaging as `complete_vision_packet`, without building dense feature rasters.
Raw depth/masks remain available for source checks. Convergence/cardinality
scalars still synchronize with Python. See the
[execution/timing report](../../docs/visual-graph-integration/10i_shape_first_cuda_58752018_20261103.md).

`extract_sampling_evidence(frame, estimator, parameters)` accepts `baseline`,
`fd2` or `pca5`. The original Task01 estimator is unchanged. Task02 finite
difference, PCA5 and pair-evidence functions were promoted to runtime modules;
old `test/` import paths re-export those implementations. Runtime imports no
`test/` module. PCA5 uses the verified float64 analytic eigensolver with guarded
library fallback, serial object processing. Cropped computation preserves the
full original raster and boundary policies; it does not crop candidate support.

Only positive finite depth inside the object's tracker mask and outside tracker
overlap is eligible. Mask/invalid/image boundaries and near/far depth ordering
remain independent evidence. RGB, GT segmentation, poses, dimensions and hidden
surfaces are not candidate/sampling inputs. Missing current masks return no
points; no previous observation is reused.

`visual.runtime.pca.build_geometry(frame, mask_only=True)` is an opt-in
backprojection variant. It calculates X/Y only at valid pixels in the union of
the two raw masks, retaining original source indices and float32 coordinates.
The original full-image behaviour remains the default. The organized output
still retains all valid Z values for depth comparisons with background
neighbours; outside the masks, X/Y are NaN. Use `observed_mask(object_id)` to
gather object XYZ. This option does not reduce the raster allocation or sample
points before normal estimation.

## Allocation and selection

1. Count the unambiguous observed pixels of Peg and Box. Reserve each configured
   minimum, capped by actual availability.
2. Distribute remaining nodes in proportion to observed counts. The alternative
   `sqrt_observed_points` uses their square roots. Capped largest-remainder
   integer apportionment redistributes exhausted capacity. Exact indivisible
   ties use ascending object ID (at most a one-slot rounding distinction).
3. Reserve at most `floor(num_points * structural_ratio)` priority slots globally.
   Distribute these by object node allocations, capped by actual eligible
   candidates and each object's node budget.
4. S1/S2/S3 use only orientation candidates. S4 independently reserves at most
   `floor(global_priority_cap * depth_priority_fraction)` depth-only slots,
   distributed by the same object weights, capped by depth-only availability and
   each object's priority allocation. Remaining slots select orientation
   candidates. Unused depth reservations may select orientation; missing
   orientation cannot silently expand the depth-only cap.
5. FPS disperses each candidate group in observed 3D coordinates. Surface FPS
   continues over all object observations with distances seeded by every
   priority point. Unused priority slots become Surface FPS slots.

The total result has `min(num_points, total_observed_points)` rows, without
padding, reconstruction or repeated source rows. Structural candidates can also
be selected by Surface FPS: the ratio caps **priority selections**, not all
candidate-flagged rows in the final cloud. Counts report `priority`,
`orientation_priority`, `depth_priority`, `surface_fps`, `selected_candidates`,
`candidates`, `observed`, `budget`, and `selected` separately for both objects.

| Method | Required evidence | Priority policy |
| --- | --- | --- |
| S0 | any estimator | General object-wise FPS, no priority |
| S1 | baseline | Original 1px normal orientation |
| S2 | fd2 | FD 2px orientation |
| S3 | pca5 | Optimized PCA 5×5 orientation (API default) |
| S4 | pca5 | Orientation and separately capped depth-only priority |

At ratio zero, every method selects the exact same pixels as S0 for the same
observations/config/seed. No-candidate input also equals general FPS. S4 is an
experimental depth ablation, not a final structural-label policy.

FPS adapts the legacy `visual/scripts/pointcloud.py::fps` distance-update loop,
returns source indices, uses float64 squared distances, and explicitly excludes
selected indices even when coordinates coincide. No legacy erosion, random
pre-reduction, outlier removal or padding is applied. For each object the seed
is `SeedSequence([config.seed, object_id]).generate_state(1)[0]`. The first
candidate is uniform in its raster-ordered group; distance ties use the first
remaining source row. Surface FPS without priority uses the same initial rule.
Streams are independent of missing objects. This is reproducible for the same
inputs, NumPy RNG and arithmetic, not a promise of bitwise parity with legacy
float32 FPS or arbitrary devices.

## Returned aligned metadata

`SampledPoints` contains:

| Field | Shape / meaning |
| --- | --- |
| `points` | `[N,3]` observed CV-camera XYZ, metres |
| `pixels_vu`, `flat_index` | `[N,2]`, `[N]`; original raster `(v,u)` and `v*W+u` |
| `object_id` | `[N]`, Peg=0 / Box=1 |
| `priority_selected` | `[N]` selection through the priority path |
| `depth_priority_selected` | `[N]` subset selected through S4 depth-only path |
| `metadata` | Dictionary of arrays with the same N-row order |
| `object_budgets`, `counts` | Allocation and actual path/candidate counts |
| `timings_ms` | Candidate selection, allocation, priority FPS, Surface FPS, total |

Metadata retains all original `GeometricFeatures` fields, normal and variation
validity, `orientation_score_deg`, `orientation_candidate`,
`depth_only_candidate`, `structural_candidate`, plus independent
`orientation/depth/occluding/occluded/invalid/mask_evidence` flags. The original
`depth_jump_m`, `depth_pair_valid`, `depth_discontinuity`, normal-neighbour count
and boundary flags remain available. `structural_candidate` is eligibility for
this sampling condition, not a Corner/Edge/Surface answer. `priority_selected`
records what the sampler actually did, independently of candidate flags.

`extract_candidates()` returns only observed rows before selection, preserving
the same evidence and raster order. `SamplingEvidence.pairs` preserves the
original signed depth pair and near/far endpoint provenance. No new node type,
final feature dimension, onehot/probability/continuous representation, graph
edge, cache or physical point identity is defined.

## Evaluation and reproducibility

```bash
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03-mpl \
  .venv/bin/python -m visual.runtime.test.run_sampling
WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03-mpl \
  .venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
```

The runner evaluates all 121 source frames under staged sweeps (54 conditions),
checks optimized PCA fidelity against the original at every source frame, saves
per-frame selections/evidence/metrics, and renders sources 0/15/18/45/80/120.
Geometry reference definitions are unchanged from Task02-C. Same-pair matching
uses ≤2px AND ≤3mm, with 1px/2mm and 3px/5mm sensitivity conditions. Reports
separate visible-GT coverage, tracker-observed coverage and sampling retention.
Per-frame distributions and four reference-image quadrants expose uneven
preservation; quadrants are display-axis spatial bins, not physical edge IDs.
Surface distances include full support, with separate GT plane, signed-face and
small raster-component (≤64 observed pixels) summaries. Negative ROIs overlap;
`known_negative_union` avoids double counting their combined budget occupancy.

`ExperimentLogger` creates the existing UUID/local manifest. No team logger,
new identity rules or automatic upload is added. JSON/NPZ/HTML/PNG and evaluated
source copies live under `visual/results/sampling/<run_id>/`; local tracking is
in `experiments/<run_id>/`. Input and source hashes are in `comparison.json`.
The source copy captures dirty implementation files; Git commit alone cannot
recreate an uncommitted run. S0's reported diagnostic total includes the shared
PCA evidence, even though general FPS does not need that evidence.

CPU NumPy only. CUDA/MPS sampling and end-to-end 20Hz are **NOT RUN**. Saved
RGB-D replay excludes tracking, rendering, device transfers, final packaging,
edge construction and GNN/Recurrent work. See the [Task03 measured report](../../docs/visual-graph-integration/03_sampling.md).
