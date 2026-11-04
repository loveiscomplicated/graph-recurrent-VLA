# Task 10-I: early CUDA ingress and selected-point consumers

Optional prototype backend: `CurrentGeometry(..., perception_backend='cuda_resident')`.
Add `coordinate_frame='world'` or `'ee'` (default) to choose the same common frame
for selected packet, observations, dedup/budget and returned Graph, including
robot nodes and edge directions. View tests project from that frame directly.
The raw sensor/filter/gate boundary retains its existing calibration contract.
Historical measurements below precede this frame choice and are not new timings.
The physical simulator, renderer and FK implementation are unchanged. Existing
`cpu`/`triton_fused` callers retain their existing input/output behavior.

```python
geometry = CurrentGeometry(assets, schema, perception_backend='cuda_resident')
try:
    frame = geometry.stage_inputs(cameras, sensor_data, current_robot_surfaces)
    # RGB uint8 HWC, signed int16 sensor depth and calibration are now on CUDA.
    # The caller must join SAM2 masks to this same episode/time/camera snapshot.
    # raw_masks[name]: CUDA bool [2,H,W], ordered Peg/Box.
    graph, records, gate = geometry.build_staged(frame, raw_masks)
    packed = pack_graphs([graph])  # stays on CUDA, ready for an existing encoder
finally:
    geometry.close()  # close the persistent four-worker Qhull executor
```

`build(cameras,data,surfaces,raw_cuda_masks)` is the convenience equivalent that
stages inside the call. `frame.tensors[name]` exposes `rgb`, `raw_depth`,
`depth64`, `depth32`, `intrinsic`, `world`, `inverse`, `ee` and `world_ee`.
`frame.data_cuda` contains the supplied numeric robot arrays. A per-frame aligned
byte arena uploads CPU raw input once; depth is converted on CUDA with the exact
signed-millimetre lookup already used by the robot filter. Nonpositive sensor
depth remains invalid. Current depth, calibration, poses and hulls are renewed
every call. There is no cache by episode, timestamp or robot pose.

Filter, wrist gate, projection/PCA5, existing Shape-First, selected EE packet,
common-frame observation handling and cross-camera checks keep their arrays on CUDA.
The selected-point check now shares `persistent_geometry.tensor_fusion` with
the default Graph adapter and reuses the staged float64 depth tensors. The
earlier robot-filter/wrist-gate implementations are unchanged by that connection.
`visual.runtime.graph_perception.make_packet_cuda` returns a measured continuous
v2 packet; `graph.vision_input` validates/receives that tensor representation.
It supports full measured feature availability and unimplemented packet edges;
other legacy packet kinds retain the NumPy receiver. The legacy packet contract,
NaN invalidity and source provenance are preserved. `numpy_packet` is an explicit
single packed readback for a legacy consumer, never an implicit receiver step.

After view checking, one packed readback exports only selected arrays. Existing
native 3mm deduplication and stable string-ID FPS act on compact arrays before
Python records are made. Only final representatives get records, with **all**
merged source IDs retained. Existing CPU cKDTree neighbor/tie rules and batched
small-graph gates are reused; final Graph tensors return on CUDA. Current world
Qhull remains CPU and runs in an ordered four-worker map, retaining every hull
coefficient and original link slot. Small trace values and existing ROI bounds
may synchronize with CPU. This is not a zero-synchronization or all-GPU pipeline.

The scoped small-graph consumer context changes process-local functions, like
the preceding experiment contexts. Do not run concurrent builders in one Python
process. CPU snapshots and a staged frame must not be reused for a different
physical timestep.

`sam2_bridge.image_masks_cuda` additionally accepts resident uint8 HWC RGB via
`set_image_cuda`, reusing the installed SAM2 Resize/Normalize and embedding code.
This optional private-API bridge is checked with installed preprocessing and a
fake backbone, **not a full SAM2 inference equivalence test**. NumPy RGB retains
the original public `set_image` path. Saved-mask pipeline measurements below do
not time or validate actual SAM2 inference with the tensor RGB adapter.

Run on the existing CUDA host/environment, with `TASK10I_REPO` pointing to the
source snapshot and `TASK10I_ROOT` to the frozen sensor/mask dataset:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled python -m unittest \
  graph.experiments.cuda_ingress_10i_20261103_v1.test_contracts -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled python -m \
  graph.experiments.cuda_ingress_10i_20261103_v1.benchmark \
  --output /unique/new/output --scenes 10i_s1_0924_t080 --sizes 512 --repeats 15
```

Benchmark inputs are **CPU observations/current FK surfaces + saved raw CUDA
masks ready**, output is **CUDA Graph + CPU records/gate**. It includes RGB/raw
depth/calibration staging, Qhull, all internal transfers, gate, PCA5/SF, dedup,
budget, Graph generation and validation. Baseline uses the current triton_fused
builder, existing GPU mask boundary and full Stage 9 context. Both outputs are
put on CUDA before the synchronized wall timer stops. SAM2, grounding, separate
FK, simulator step/render, file I/O and action inference are excluded. This is a
different metric from the historical 47.0/13.4/7.16ms **filter-only** results.

Warmup is three pairs, then 15 alternating-order pairs, Torch threads4 and
BLAS threads1. CLI `--threads` controls Torch, while the hull executor currently
remains4 workers. `--scenes all --sizes 128 512 768 1024 --repeats 0` performs
saved-input correctness only. It compares masks through source selection,
complete packets, all records/provenance, gates, Graph tensors and metadata;
discrete fields are exact and floats use field-specific bounds. All source/input
identities and raw repetitions are retained with `ExperimentLogger`, W&B disabled.
No rollout, training, publishing, package install or paused-experiment resume.
