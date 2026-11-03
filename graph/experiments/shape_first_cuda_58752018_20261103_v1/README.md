# Task 10-I — GPU sampling information and Shape-First

`benchmark.py` enforces instance 58752018 and consumes the preceding real 512
hand-camera `hand_camera.npz`/`masks.npz` inputs. Robot filtering, grounding,
projection/PCA and final CPU packet/Graph assembly are outside timing.

Start: compact CUDA points and fused features are ready and synchronized.
End: selection is finished and its outputs are CPU-accessible. The original
variant downloads all features and executes CPU evidence/Shape-First; the GPU
variant keeps pixel/point decisions on CUDA and downloads selected rows only.

```bash
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m graph.experiments.shape_first_cuda_58752018_20261103_v1.benchmark \
  --inputs /path/to/inputs --output /path/to/new/run \
  --variant paired --warmups 20 --repeats 100 --phase-repeats 20
```

The runner refuses to start if another CUDA compute process is present. It
never stops another job. CUDA completion is synchronized at primary wall-time
boundaries; variants alternate order. Phase breakdowns are a separate diagnostic
pass and their medians must not be added to reconstruct the primary median.
JIT/warmup, file IO and parity assertions are excluded. A baseline-only run was
completed before implementing the GPU path; its original runner/source is
preserved separately. The shared `ExperimentLogger` records new UUIDs, W&B is
disabled, and nothing is published.

GPU contracts:

```bash
python -m unittest visual.runtime.test.test_cuda_sampling \
  visual.runtime.test.test_graph_perception visual.runtime.test.test_pca_optimized -v
```

The full [report](../../../docs/visual-graph-integration/10i_shape_first_cuda_58752018_20261103.md)
records the hardware, selected input support, before/after results, source
hashes and final Graph parity limits. This does not evaluate policy success.
