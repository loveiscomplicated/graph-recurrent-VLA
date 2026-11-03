# Task 10-I — selectable World / current-EE geometry

Correctness only; no latency measurement. `verify.py` checks both common frames
on the real512 hand-camera case, including source/calibration projection, CUDA
and NumPy packet agreement, direct packet→observation retention, and public
`CurrentGeometry` records/Graph with actual recorded FK assets. It prepares an
empty cloned base support because the recorded input has one real hand view.
Robot filtering and gate are bypassed with identical prepared masks.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python verify.py --inputs /path/to/inputs \
  --assets /path/to/fk/assets --schema /path/to/normalizer.json \
  --output /path/to/new/output
```

Run on the existing CUDA environment of VastAI58752018. The shared
ExperimentLogger supplies identity, W&B is disabled and nothing is published.
`visual.runtime.test.test_coordinate_frames` additionally checks nonidentity
rotation/translation, distinct camera views, normal/edge-vector transformation,
dedup/FPS invariance away from cutoffs/ties, corrupt or mixed frames and stale
EE reference rejection. It covers the resident CUDA tail with filter/gate mocked.

Full [report](../../../docs/visual-graph-integration/10i_coordinate_frames_20261103.md).
