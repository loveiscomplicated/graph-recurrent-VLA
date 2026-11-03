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

`raw_cuda` is bool `[2,H,W]`, Peg then Box. The result is a new bool tensor on the
same CUDA device. Inputs are not modified. There is no mask `.cpu()`, `.numpy()`,
`.item()` or dynamic GPU `nonzero` in this filter. Collision hull preparation
remains CPU and small calibration/geometry/sensor data may upload. Optional
`depth_cuda` accepts exactly float64 metric CV-z, including NaN invalid values.
This change does not move FK/Qhull to GPU or claim that all CPU work disappeared.
Diagnostics carry `evaluated_pixels`; high-resolution robot classifications
outside the raw mask union are not computed or presented as full-screen results.

## SAM2 connections

The existing tracker already yields device tensors. Its optional callback now
runs before the existing CPU RGB-D adapter:

```python
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.sam2_bridge import SAM2RobotFilter
for index, clean_masks_cuda in tracker.track(
    state, mask_filter=SAM2RobotFilter(observation_for_frame)
):
    ...
```

`observation_for_frame(index)` returns the exactly joined current camera and FK
surfaces. It must respect the caller's original source-frame/subsampling mapping.
Tracker IDs remain 1/2; missing objects remain missing. A clean mask must not be
passed downstream as if it were the original raw mask for gate statistics.

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
