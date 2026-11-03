# Task 10-I — Torch CUDA view consistency

`persistent_geometry.tensor_fusion` implements the existing depth-supported
opposite-object rule on Torch tensors. Default triton_fused `CurrentGeometry`
uses CUDA selected packets and keeps observations resident until the explicit
legacy CPU record boundary.
Robot filtering and the earlier wrist gate are unchanged.

`benchmark.py` pairs the unchanged NumPy reference and three CUDA endpoints:

1. Resident GPU observation/depth/mask inputs → synchronized GPU results.
2. GPU observations + CPU sensors/masks → prepare/upload → GPU results.
3. Endpoint2 plus the actual explicit selected-result CPU compatibility return.

The NumPy endpoint is CPU observations/sensors → CPU results. All observation
preparation is outside timing; CUDA inputs are already on GPU, as in the new
default packet path. Two source-camera calls are one sample. Twenty warmups,
100 samples each, rotating/reversing execution order, OMP/BLAS/MKL/Torch threads1.
Every CUDA endpoint synchronizes at completion.

Uses the preceding real512 hand input (1,105 Peg points,0 Box) with a cloned
empty base view. This is not a two-real-camera latency. Every returned field,
audit and source order is compared exactly. Untimed real FK/public Graph checks
compare new default against the former packet/NumPy-checker path in both frames.
Masks/gate are prepared inputs; no robot filter/gate is executed for those checks.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --assets /path/to/fk/assets --schema /path/to/normalizer.json \
  --output /path/to/new/output --warmups 20 --repeats 100
```

Run on the existing VastAI58752018 CUDA environment. Shared ExperimentLogger,
W&B disabled, no publishing. Tests cover signed-depth values, strict4mm and
adjacent floats, overlap, missing/invalid depth, outside/behind-camera points,
World/EE references, empty groups, resident arrays and actual default routing.

Full [report](../../../docs/visual-graph-integration/10i_view_consistency_cuda_20261103.md).
