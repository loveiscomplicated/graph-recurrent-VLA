# Task 10-I: Vast58752018 CPU full vs CUDA masked projection

Independent, opt-in projection experiment. The requested instance is enforced by
`CONTAINER_ID == 58752018`. The original NumPy builder is imported from the
byte-preserved snapshot taken before the preceding CPU experiment. No production
Graph builder or robot filter is changed.

`masked_torch.project(depth, masks, intrinsic)` takes Torch tensors on the same
device. It recalculates valid-depth and mask overlap, compacts source pixels,
and projects only selected pixels. It returns float32 camera XYZ `[N,3]` and
int64 source pixels/object IDs `[N]`, ordered by object then raster index. There
is no dense `[H,W,3]` XYZ allocation, pixel-list cache, sampling, downsampling,
mesh, GT or robot-filter input. Full 2D depth and mask tensors remain inputs.

Float64 arithmetic followed by float32 output matches the existing coordinate
policy. Calibration checks, finite-output checks, dynamic `nonzero` and its host
synchronization costs are included in the timed operation. This prototype uses
ordinary eager Torch, without compilation, custom CUDA or Triton kernels.

The benchmark distinguishes:

- CPU depth/masks/K → original full XYZ → compact CPU points and source IDs.
- GPU depth/masks/K already resident → compact GPU points and source IDs.
- Pageable CPU inputs → upload → identical CUDA operation → compact CPU outputs.

Every CUDA wall measurement waits for completion. File reads, mm conversion,
normal estimation, sampling, Graph building, robot filtering, simulator and
model inference are excluded. The additional CUDA-event span is a diagnostic
elapsed span; dynamic compaction and host decisions mean it is not a sum of
pure kernel times.

On the verified instance, use the already installed cu130 environment and frozen
repo dependencies; do not install packages or overwrite an old output:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/workspace/task10i_predicted_state_gru_20261102/source \
  WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/venvs/task10i-stage4a-cu130/bin/python \
  /workspace/task10i_masked_backprojection_cuda_58752018_20261103_v1/code/benchmark.py \
  --root /workspace/task10i_masked_backprojection_cuda_58752018_20261103_v1 \
  --output new_unique_output --warmups 100 --repeats 1000
```

The root's `inputs/` contains the exact prior 512×512 input and original source;
`upload_manifest.json` pins their hashes. The benchmark refuses to start while
another GPU compute process is listed. It never stops other jobs. Variant order
rotates/reverses across repetitions; input/source hashes and verification stay
outside timing. Each run uses the shared `ExperimentLogger`, W&B disabled, and
does not call the artifact publisher.

Tests run locally on Torch CPU and on the target host with actual CUDA:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  graph.experiments.backprojection_cuda_58752018_20261103_v1.test_projection -v
```

The same test file can run directly as `python code/test_projection.py -v` on
the remote host. The full report belongs in
`docs/visual-graph-integration/10i_backprojection_cuda_58752018_20261103.md`.
