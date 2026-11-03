# Task 10-I — Shape-First refinement on Vast58752018

`benchmark.py` loads byte-preserved pre-change Python **and Triton** modules as
the baseline. Current/reference GPU-output variants alternate order, with20
warmups and100 synchronized wall measurements each. Correctness checks compare
all selected fields plus complete diagnostic candidate/representative arrays.
The input is the same real512 hand-camera case from the preceding experiments.
The runner refuses to start while another CUDA compute process is present.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m graph.experiments.shape_first_refinement_58752018_20261103_v1.benchmark \
  --inputs /path/to/inputs \
  --reference /path/to/preserved/visual/runtime/cuda_sampling.py \
  --output /path/to/new/output --paired --stage-repeats 30
```

The reference's adjacent `cuda_sampling_kernels.py` must also be preserved.
Without `--paired`, only the baseline is measured; this was done before edits.
The final runner records first-call latency separately from warm replay.
Isolated stage measurements lose overlap and must not be added as a frame
budget. FPS seed/repeat microbenchmarks in the baseline reproduce its original
128×32/8-warp layout; optimized FPS timing calls its real graph-backed operator.

`tune_seed.py` sweeps exact FP64 tilings. `--pruned` also tests conservative
AABB pruning, asserts every minimum distance equals the old kernel, and counts
evaluated point/seed pairs outside timing. The counter is disabled in normal
execution. The shared `ExperimentLogger` records each execution, W&B is disabled,
and publishing is not performed.

Scope: compact GPU features ready → full sampling output ready on GPU. No
projection/PCA, robot filtering, gate, grounding, final Graph assembly or policy
evaluation is timed. Read the full
[report](../../../docs/visual-graph-integration/10i_shape_first_refinement_58752018_20261103.md).
