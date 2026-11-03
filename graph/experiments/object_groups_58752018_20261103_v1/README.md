# Task 10-I — per-object packet groups

Correctness verification only; no latency measurements. `verify.py` captures
the public packet and all World observation fields on the real512 hand input.
Run first against the frozen pre-change source, then against the updated source
with `--reference BASELINE/capture.pkl`. Only the new `object_slices` field may
differ; all prior fields/dtypes/shapes/NaNs and every observation must match
exactly. The reference pickle is exclusively the runner's own trusted capture.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python verify.py --inputs /path/to/inputs \
  --output /path/to/new/output --reference /path/to/baseline/capture.pkl
```

Requires the already configured CUDA environment on VastAI58752018. The shared
ExperimentLogger retains identity; W&B is disabled and nothing is published.
Tests in `visual.runtime.test.test_object_groups` additionally cover both objects,
one missing object, both absent, overlap/invalid depth, legacy interleaved rows,
corrupt ranges and CUDA tensor packet validation. Robot filtering is excluded.

Full [report](../../../docs/visual-graph-integration/10i_object_groups_20261103.md).
