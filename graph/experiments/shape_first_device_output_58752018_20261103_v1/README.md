# Task 10-I — sampling output device

`sample_cuda(...)` now returns CUDA tensors by default, including source IDs,
features and selection provenance. `device='cpu'` explicitly requests CPU
tensors. This changes the sampler's terminal device; computation remains CUDA.
The existing NumPy Graph consumer explicitly requests CPU as a compatibility
adapter. Final Graph assembly was excluded by the user's scope clarification.

`benchmark.py` compares the preserved preceding CPU/NumPy-return sampler,
default CUDA output, and explicit CPU tensor output. All use the same ready
CUDA points/features from the real 512 hand-camera input. CUDA synchronizes at
both wall-time boundaries. Order rotates/reverses, with 20 warmups and 100
measurements per variant. Output downloads for parity/NPZ writing are outside
timing for CUDA output. Projection/PCA and Graph assembly are excluded.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m graph.experiments.shape_first_device_output_58752018_20261103_v1.benchmark \
  --inputs /path/to/inputs \
  --reference /path/to/preserved/cuda_sampling.py \
  --output /path/to/new/output --warmups 20 --repeats 100
```

The runner asserts instance58752018, refuses to start alongside another CUDA
compute process, reuses the shared `ExperimentLogger`, and publishes nothing.
Its reference module is the byte-preserved sampler from the preceding task,
not a reimplementation of the old return behavior.

Full [report](../../../docs/visual-graph-integration/10i_shape_first_device_output_58752018_20261103.md).
