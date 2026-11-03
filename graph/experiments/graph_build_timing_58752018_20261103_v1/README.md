# Task 10-I — unchanged CPU edge/Graph construction baseline

CPU post-FPS128-point records, robot geometry, calibration and raw depth are
ready. Measures `PlanA.graph` plus `CurrentGeometry._finish`, including input/frame
checks, stable node ordering (no iterative final FPS), robot positions/features,
surface edges and their witness/free-space tests, robot-object/Peg-Box pairs,
edge attributes, CPU tensor construction, schema/provenance/final checks.
GPU-to-CPU return, upstream work, FK, robot filter/gate, model packing/upload and
inference are excluded. Runtime implementation is unchanged.

Primary endpoint:20 warmups,100 samples, alternating EE/World order, no hooks.
Separate diagnostic endpoint:5 warmups,20 samples, scoped timing wrappers around
the SAME production calls and original CameraObservation.depth getter. No cache
or shortcut is introduced. Surface/free-space/depth timings are nested: do not
add them. Other assembly is a per-sample residual inside build_graph. Diagnostic
overhead does not contaminate the primary latency.

Same512 real hand input as preceding tasks, Peg128 and Box0, empty base cloned
from hand depth/calibration. Graph depth tests still inspect both camera objects.
This is not a two-independent-real-view or nonempty-Box latency measurement.
Actual CUDA pipeline prepares records outside timing; public CurrentGeometry
with actual FK and prepared masks/gate validates all Graph tensors/packet and
record/provenance equivalence. Explicit CPU robot backend with prepare replaced
avoids executing the separate robot-filter task.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --assets /path/to/fk/assets --schema /path/to/normalizer.json \
  --output /path/to/new/output --warmups 20 --repeats 100 \
  --diagnostic-warmups 5 --diagnostic-repeats 20
```

Runs on existing Vast58752018 environment with shared ExperimentLogger,
W&B disabled, no publishing. Raw Graph tensors/metadata and timing/call arrays
are preserved for subsequent equivalent optimization.

Full [report](../../../docs/visual-graph-integration/10i_graph_build_timing_58752018_20261103.md).
