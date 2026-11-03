# Task 10-I: SAM2 CUDA masks → robot filter → CUDA clean masks

Only the mask device boundary and its wiring change. The filter keeps current
collision hull/ray geometry, per-link order/ties, 2.5mm tolerance, 6mm band,
0.5mm interior margin and object overlap removal. Full ray/per-link numerator
arithmetic is retained above 128²; the existing grouped arithmetic is retained
at small sizes. No Graph features, point selection, gate rules, training,
model weights, prompts or thresholds are changed.

## Direct GPU output

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
clean_cuda = CudaRobotFilter()(camera, current_fk_surfaces, raw_cuda)
```

For a synchronized frame with both cameras, keep one filter instance and use:

```python
robot_filter = CudaRobotFilter()
clean_by_camera = robot_filter.filter_cameras(
    cameras, current_fk_surfaces, raw_cuda_by_camera
)
```

This entrypoint prepares the current world collision hulls once and shares them
between cameras only within that call. The next call recomputes them from current
FK, including in-place pose edits. `GpuMaskBoundary` uses this entrypoint by
default; existing custom single-camera filter callables still work.

Above 128², clipping kernels group links by plane count, load the raw mask union
on GPU and atomically reduce nearest depth/interior. Diagnostic owners resolve
equal depths to the first original link in a second pass. Original float64 ray
GEMM, per-link numerator GEMV and thresholds remain intact. Equation uploads are
shared only within a frame. A bounded cache (8 entries / 64 MiB) retains only CV
camera rays keyed by resolution, intrinsic bytes and CUDA device. Current world
rotation/origin, sensor depth, hulls and masks are always recomputed or reread.

Frame preparation first projects every current world collision shape into each
camera. A shape whose existing conservative bounds are empty in every camera
needs no Qhull; its original link slot remains present so owner IDs retain their
meaning. Camera inverse transforms are validated once per camera. Equations,
contiguous normals and all camera/group int64 metadata share one frame-local
upload. These preparations refresh on every call, including when an invisible
link or moving wrist becomes visible again.

High-resolution filtering now compacts the mask-union pixels inside each link
bounds on GPU before clipping. Fixed-capacity row lists and CUDA counters avoid
CPU count readback or dynamic `nonzero`. Only pixel scheduling changes: every
plane calculation stays float64 and diagnostic owners still use deterministic
minimum original link IDs after nearest-depth reduction. Row-list order may vary
between launches; returned decisions do not. Temporary row capacity is the sum
of projected bounding-box pixels (4 bytes per slot), recreated for each camera.
`CudaRobotFilter(compact=False)` retains the prior masked-lane path for direct
comparison. `clip_pixels=32` remains the default; 8/16/64 are diagnostic tuning
options. The small-resolution grouped path is unchanged.

For the standard `CameraObservation` with signed int16 millimetre depth, the
filter uploads the original sensor array and converts to float64 metres on GPU.
One 256KiB table per used CUDA device stores NumPy's exact float64 `/ 1000`
results for all positive int16 values; zero/negative/overflow values stay invalid.
The table is a universal unit conversion, independent of pose, calibration and
frame depth. GPU conversion covers the full image. This temporary depth buffer
is private to the filter. Custom camera objects retain their `.depth`
contract, and optional `depth_cuda` remains unchanged. `depth_from_mm=False`
selects the prior CPU conversion path for comparison. No converted depth is
cached between frames. At 512², two-camera sensor upload drops from 4 MiB of
float64 data to 1 MiB of signed int16 data.

`raw_cuda` is bool `[2,H,W]`, Peg then Box. The result is a new bool tensor on the
same CUDA device. Inputs are not modified. There is no mask `.cpu()`, `.numpy()`,
`.item()` or dynamic GPU `nonzero` in this filter. Collision hull preparation
remains CPU and small calibration/geometry/sensor data may upload. Optional
`depth_cuda` accepts exactly float64 metric CV-z, including NaN invalid values.
This change does not move FK/Qhull to GPU or claim that all CPU work disappeared.
Diagnostics carry `evaluated_pixels`; high-resolution robot classifications
outside the raw mask union are not computed or presented as full-screen results.

## SAM2 connections

CPU world-shape/ConvexHull/projection-bound preparation now starts **before**
SAM2 inference. The worker performs only NumPy/SciPy preparation; the predictor
and all CUDA filtering remain on the caller's existing thread/stream. The join
occurs after raw masks are produced, before the unchanged GPU filter:

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.preparation import RobotFilterPreparation

with RobotFilterPreparation(robot_filter) as preparation:
    pending = preparation.submit(cameras, current_fk_surfaces)
    raw_cuda_by_camera = run_existing_sam2()  # existing inference, main thread
    clean_cuda_by_camera = preparation.filter(pending, raw_cuda_by_camera)
```

`image_clean_masks_cuda(..., preparation=preparation)` uses this ordering
automatically for one camera. `images_clean_masks_cuda` does so for a dictionary
of predictors/RGB/boxes and synchronized cameras, sharing one prepared geometry
snapshot across both views. Keep one preparation context to reuse its CPU worker;
the existing single-call image helper also works with a temporary worker.

Exactly one frame may be outstanding. Raw signed depth, calibration and FK
geometry are snapshotted before submission to protect against subsequent buffer
reuse. Hull equations and original link slots/order are unchanged. This async
path requires standard CPU `CameraObservation`/`LinkSurface` inputs; direct
`CudaRobotFilter` calls keep their previous custom-camera behavior. Frame join,
mask identities and source-index mappings remain the caller's responsibility.
Inference/preparation errors propagate, and cancelled/closed inference drains
the CPU job instead of reusing it on the next frame. No frame-ahead pose cache
or GPU-wide synchronization is introduced.

The existing tracker already yields device tensors. Its optional callback now
runs before the existing CPU RGB-D adapter:

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.sam2_bridge import SAM2RobotFilter
with SAM2RobotFilter(observation_for_frame, robot_filter) as callback:
    for index, clean_masks_cuda in tracker.track(state, mask_filter=callback):
        ...
```

`observation_for_frame(index)` returns the exactly joined current camera and FK
surfaces. It must respect the caller's original source-frame/subsampling mapping.
Tracker IDs remain 1/2; missing objects remain missing. A clean mask must not be
passed downstream as if it were the original raw mask for gate statistics.

For the normal SAM2 `init_state` state containing `num_frames`, the tracker calls
`prepare_frame(index)` **before** advancing each forward propagation frame. It
checks consecutive output indices and cancels pending work on inference error,
iterator close or early termination. Prompt initialization in `start` is
unchanged. Custom states without `num_frames` retain post-output filtering;
they cannot schedule the next current geometry in advance. Other callbacks and
tracking without a callback retain their existing mask behavior.

Image SAM2's public `predict()` returns NumPy, so `image_masks_cuda()` follows
its installed `_prep_prompts`/`_predict` calls and skips only final CPU
materialization. `image_clean_masks_cuda()` connects those device masks directly
to `CudaRobotFilter` and returns CUDA clean masks plus CUDA score scalars (None
for missing prompts). Use the same model/autocast settings as the existing caller.
The installed private SAM2 signatures were inspected; incompatible output shapes
raise. This is an opt-in live device path; archived NPZ workers remain unchanged.

## Existing CPU Graph consumers

To retain the original raw-mask gate denominator and avoid filtering twice:

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.graph_boundary import GpuMaskBoundary
with GpuMaskBoundary() as boundary:
    graph, records, gate = geometry.build(cameras, data, surfaces, raw_cuda_by_camera)
    clean_cuda_by_camera = boundary.clean_cuda
```

This scope first completes GPU filtering. Only then it explicitly copies raw and
clean masks for the unchanged CPU gate/Shape-First/Graph consumers. No original
mask is downloaded and then uploaded for filtering. The filter's own return stays
CUDA; this compatibility boundary does not claim an all-GPU Graph pipeline.
Do not chain already-cleaned tracker masks into this raw-mask entrypoint. Use
raw tracker output or `image_masks_cuda()` for this route. Patches restore on
exceptions; concurrent builders in a single process are unsupported.

## Verification scope

Saved actual masks are uploaded once outside verification to stand in for SAM2
CUDA outputs. No SAM2 inference, grounding, training, rollout, model API call,
package installation or original stopped-process resume is performed. Exact
mask/owner/flags and existing depth contracts are checked against Stage 9, plus
mocked SAM2 image/tracking connections, no-readback dispatch guards and gate/raw
statistics. This verifies the bridge and filter, not real SAM2 inference latency.
The previous 47.0→13.4ms CPU→CPU result is a different measurement boundary;
no additional speedup is inferred here.

The overlap change is tested by deterministic thread-order/lifecycle tests,
mocked image/video SAM2 and exact real-CUDA filter comparisons. A saved two-camera
fixture at 128/512/768/1024 verifies all diagnostic fields and hull metadata
bitwise against serial preparation. `verify_preparation.py` measures only CPU
preparation at512 (warmup3/repeats15). Actual SAM2 latency and end-to-end speedup
are not inferred from that CPU-only timing or from the historical7.16ms result.
