# Task 10-I — resident GPU per-object final128 FPS

`graph.persistent_geometry.tensor_records.budget_tensor(records,count=128)`
mirrors ordinary CPU `budget(records,count)`, with the TensorRecords input/output
on its explicit CPU or CUDA device. No padding and no structural anchors. Each
short cloud retains original row order; longer clouds use the existing full
persistent-ID STRING seed/tie order, with float64 distances and no FP contraction.
One fused Triton launch per nonempty oversized object performs all FPS iterations.
Torch gathers features and full ragged merged-source provenance on device.

GPU string keys encode UTF-8 camera prefix plus decimal int64 pixel into positive
seven-byte limbs. Stable GPU lexical sorting preserves numeric-string differences,
prefix cameras, Unicode and all nonnegative signed-int64 pixels. Episode/time are
common metadata. Only static camera-name bytes/powers are cached, never points.
Dynamic object/provenance compaction and an aggregate validation can synchronize;
there is no per-FPS-iteration readback or point-array CPU conversion.

Default triton_fused `CurrentGeometry` uses GPU dedup → `budget_tensor` → explicit
`numpy_records` → existing CPU Graph construction. No CPU final-FPS call occurs.
The stage result remains GPU; final Graph assembly is still a CPU consumer.
CPU perception and the separate cuda_resident prototype keep their existing paths.
Robot filtering/gate are not modified or benchmarked here.

Four paired endpoints,20 warmups/100 samples, rotating/reversing mode/frame order:

- `cpu`: ready CPU records → unchanged NumPy final128 FPS.
- `gpu`: ready GPU records → string ranks/FPS/feature+source gather → synchronized GPU result.
- `gpu_to_cpu`: endpoint2 plus actual post-FPS CPU compatibility materialization.
- `legacy_from_gpu`: ready GPU records → old212-point CPU materialization → CPU FPS.

Same512 real hand input, Peg212→128, Box0, empty cloned base. Inputs and ordered
selection must match the saved preceding baseline NPZ exactly. Every record field
and source is checked, plus actual FK/public Graph in EE and World against the
frozen pre-port geometry. Preparation, robot filter/gate and edges are untimed.
This is not a two-independent-real-view or nonempty-Box latency measurement.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --assets /path/to/fk/assets --schema /path/to/normalizer.json \
  --reference-source /path/to/frozen/reference \
  --baseline-inputs /path/to/baseline_fps_inputs.npz \
  --output /path/to/new/output --warmups 20 --repeats 100
```

Shared ExperimentLogger, W&B disabled, no publishing. CUDA tests execute on
Vast58752018; CPU Torch device tests execute locally too.

Full [report](../../../docs/visual-graph-integration/10i_final_fps_cuda_58752018_20261103.md).
