# Task 10-I — existing point records and 3mm deduplication baseline

Measures the unchanged default PlanA SF CPU consumer. CPU observations and
aligned structural flags are ready after `numpy_observations`; timing includes
structural pixel sets, `current_records`, structural marking, camera concatenation,
and `deduplicate(..., .003)`. It excludes readback, the final128-point FPS,
edges, unused Legacy/Hybrid diagnostics, and robot filtering/gate.

Primary combined timing has no intermediate clocks. Separate diagnostic samples
split record construction and deduplication; their medians need not sum to the
combined median. Both configurable frames are measured with20 warmups and100
samples, alternating frame order. Normal Python garbage collection stays enabled;
previous sample outputs are released outside the next timed call.

Input is the same real512×512 hand frame, with an empty base view cloned from
hand calibration/depth. It is not a two-real-view or nonempty-Box benchmark.
Production CUDA packet/view/readback preparation runs outside timing. Every
output record field, audit, representative order and merged provenance is
compared against public `PlanA.represent(..., ('SF',))`. Inputs must stay unchanged.
No runtime implementation is modified.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source /workspace/venvs/task10i-stage4a-cu130/bin/python \
  benchmark.py --inputs /path/to/inputs --output /path/to/new/output \
  --warmups 20 --repeats 100
```

Shared `ExperimentLogger`, local logging, W&B disabled; no publishing.

Full [report](../../../docs/visual-graph-integration/10i_records_dedup_timing_58752018_20261103.md).
