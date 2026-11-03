# Task 10-I: masked backprojection at native 512

This experiment compares the byte-preserved original NumPy geometry builder
with `visual.runtime.pca.build_geometry(frame, mask_only=True)`. It uses one
saved 512×512 hand-camera depth frame and its actual SAM2 mask. No inference,
simulator, robot filtering, training or publishing runs here.

The variant calculates X/Y only at valid pixels in the union of the raw object
masks. It retains background Z for existing depth-boundary features and retains
the full raster allocation for the current PCA/feature consumers. Outside the
masks X/Y are NaN. Object XYZ, source indices and feature semantics are checked
against the original builder.

`graph_latency_10i_20261103_v2.dense.extract(..., mask_only=True)` forwards the
option to this builder for an explicit GPU experiment. Both APIs default to
`mask_only=False`: the local CPU benchmark was slower with the new option, and
the modified CUDA feature path has not been executed.

The original builder was timed and snapshotted before editing it. Reproduce the
subsequent paired comparison from the repository root with a fresh output path:

```bash
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m \
  graph.experiments.masked_backprojection_512_20261103_v1.benchmark \
  --input-root outputs/task10i/masked_backprojection_512_20261103_v1 \
  --output outputs/task10i/masked_backprojection_512_20261103_v1/new_paired_run \
  --warmups 50 --repeats 1000
```

The input root contains the preserved pre-edit source, actual input hashes,
normalized input, original point arrays and pre-edit timing receipt. Paired
timing includes validation, mask copies, allocation and optional object-point
gathering. Input loading and depth-mm conversion are outside the timer, as are
PCA/features, sampling, Graph assembly and all robot-filter logic. Host arrays
and output shapes stay comparable. Each pair alternates variant order. Input
and source hashes, correctness checks and logger/file writes stay outside the
timer; GC remains enabled. The shared ExperimentLogger records local UUIDs with
W&B disabled. No artifact publisher is called.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  visual.runtime.test.test_masked_backprojection \
  visual.runtime.test.test_pca_optimized -v
```

See the full report at
`docs/visual-graph-integration/10i_masked_backprojection_512_20261103.md`.
