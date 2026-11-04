# Task 10-I — existing cross-view consistency timing

Measurement only. The unchanged public CPU NumPy `consistent_views` is called
once for base observations and once for hand observations, exactly as in
`PlanA.represent`. Both calls together are one timed sample. World/EE runs
alternate after20 warmups, with100 samples per frame.

The preceding real512 hand input is reused. **Only one actual sensor is available**:
base uses a clone of its calibration/depth and empty masks/observations, as in the
preceding packet/Graph verification. Hand has1,105 Peg points and0 Box points.
This is not two independent real camera views or a full nonempty two-view latency.

Includes existing camera synchronization/calibration validation, projection,
full depth-property conversion, depth/mask comparisons, audit and output subset
copies. Excludes file IO, CUDA preparation, sampling, D2H, packet/observation
creation, per-point records, deduplication, Graph, robot filter and gate.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --output /path/to/new/output --warmups 20 --repeats 100
```

Run on the existing VastAI58752018 environment, using the frozen source from
the preceding coordinate-frame implementation. CUDA is used only to prepare
observations outside timing. Shared ExperimentLogger, W&B disabled, no publish.
All returned fields and source order are checked exactly; the empty-other-mask
case must retain all input observations. No runtime algorithm changes are made.

Full [report](../../../docs/visual-graph-integration/10i_view_consistency_timing_20261103.md).
