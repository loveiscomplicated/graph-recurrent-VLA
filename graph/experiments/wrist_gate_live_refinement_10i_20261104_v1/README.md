# Task10-I live wrist gate refinement

Optimizes camera preparation/uploads and CPU synchronization in the existing
live `CurrentGeometry._build_prepared -> prepare_tensor_gate` path. The gate
continues to judge World geometry for both EE and World Graph choices.

The runner loads a saved pre-change source snapshot and compares adjacent GPU
before/after calls on the same nonempty 512px, 29,775-point input. Native CUDA
cameras, raw/clean masks and robot-prepared float64 depths are ready before each
timer. Camera validation/calibration upload, gating, full Python trace and C3 mask
construction are included. SAM2, robot filtering, Graph, simulator and correctness
readback are excluded. Every timed result must exactly match the saved accepted
mask hash and every CPU reference trace field, including diameter.

```bash
WANDB_MODE=disabled python -m graph.experiments.wrist_gate_live_refinement_10i_20261104_v1.benchmark \
  --input ORIGINAL_CPU_TIMING_INPUT_DIR \
  --baseline outputs/task10i/wrist_gate_live_refinement_20261104_v1/baseline \
  --output NEW_OUTPUT_DIR --warmups 5 --repeats 30
python -m unittest graph.experiments.wrist_gate_live_refinement_10i_20261104_v1.test_contracts -v
```

The baseline directory contains `tensor_gate.py`, `tensor_fusion.py` and the
source identity in its parent `before_identity.json`. It is an ignored experiment
artifact, not a second production implementation. The runner reuses
`ExperimentLogger` with local artifacts; it does not publish remotely.

## Wrist gate function only

`benchmark_gate_only` times only the current `wrist_gate_tensor` call on the same
nonempty input. Input upload and preparation of the World depth/calibration view
finish before timing. World 3D reconstruction, connectivity, local support,
cross-camera consistency, full Python trace and the accepted CUDA mask are included.
`prepare_tensor_gate`, C3 preparation, Graph-frame views and EE matrices are not
executed. CUDA is synchronized before and after each measured call; correctness
readback happens outside the timer. Every output must match the frozen CPU mask
hash and the entire trace exactly. The source identity JSON maps repository-relative
source paths to SHA-256 and is checked before measurement.

```bash
WANDB_MODE=disabled python -m graph.experiments.wrist_gate_live_refinement_10i_20261104_v1.benchmark_gate_only \
  --input ORIGINAL_CPU_TIMING_INPUT_DIR \
  --identity outputs/task10i/wrist_gate_gpu_only_timing_20261104_v1/source_identity.json \
  --output NEW_OUTPUT_DIR --warmups 5 --repeats 30
```
