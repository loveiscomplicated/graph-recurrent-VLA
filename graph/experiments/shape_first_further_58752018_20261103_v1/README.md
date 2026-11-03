# Task 10-I — further Shape-First optimization

`benchmark.py` compares the previous GPU-output implementation to the current
one on instance58752018. It reuses the preceding reference loader and stage
measurement helpers. Isolated remote snapshots may copy that unchanged runner
beside this file as `refinement_helpers.py`; otherwise it imports the repository
module normally. The reference's adjacent Triton kernel file is loaded into an
independent namespace, so current kernels cannot replace its baseline behavior.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m graph.experiments.shape_first_further_58752018_20261103_v1.benchmark \
  --inputs /path/to/inputs \
  --reference /path/to/preserved/visual/runtime/cuda_sampling.py \
  --output /path/to/new/run --paired --repeats 100 --stage-repeats 30
```

Before/after variants alternate order after20 warmups. CUDA synchronizes at
both wall-time boundaries; the endpoint is GPU sampling output, not enqueue
completion. First-call times are separate. Full diagnostic candidate and
representative arrays are compared exactly outside timing. Canonical-kernel
profiling captures its real invocation arguments and replays that kernel alone;
its overhead-isolated result is not an additive portion of the frame budget.

Projection/PCA, robot filtering/gate, file IO and final Graph assembly are
excluded. The shared `ExperimentLogger` records local UUIDs, W&B is disabled,
and nothing is published. The runner refuses to start while another CUDA
compute process is present and never stops another job.

See the full [report](../../../docs/visual-graph-integration/10i_shape_first_further_58752018_20261103.md).
