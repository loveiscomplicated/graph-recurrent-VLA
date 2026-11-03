# Task 10-I — existing per-object final128 FPS baseline

Measures unchanged `graph.plan_a_geometry.core.budget(records,128)` with default
`hybrid=False`, including per-object filtering, frame checks, XYZ/ID preparation,
stable persistent-ID seed/ties, iterative NumPy FPS and selected-record gathering.
The post-dedup CPU records are ready before timing. GPU readback, graph creation
and its subsequent ordering, filtering/gate and all upstream work are excluded.

Same real512 hand input as the preceding measurements: Peg212 post-dedup points
→128, Box0→0, empty base cloned from hand calibration/depth. This is not a
two-independent-real-camera or nonempty-Box performance measurement.
Both EE and World:20 warmups,100 samples, alternating frame order, threads1,
normal garbage collection, previous output cleanup outside the next timed call.

The current default CUDA pipeline prepares and deduplicates actual inputs outside
timing. Complete selected records/provenance match public `CurrentGeometry.build`
with actual FK and prepared masks/gate. Input records must remain unchanged.
Where the robot-preparation interface supports backend selection, the runner
selects CPU and replaces `system.prepare` with the prepared condition: no robot
classification or filtering is executed. Perception itself remains triton_fused.
`fps_inputs.npz` preserves post-dedup XYZ, persistent IDs, objects, structural flags
and expected ordered source rows for a later equivalent GPU implementation.
No runtime implementation is changed here.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --assets /path/to/fk/assets --schema /path/to/normalizer.json \
  --output /path/to/new/output --warmups 20 --repeats 100
```

Uses the existing Vast58752018 environment and shared ExperimentLogger,
W&B disabled, no publishing.

Full [report](../../../docs/visual-graph-integration/10i_final_fps_timing_58752018_20261103.md).
