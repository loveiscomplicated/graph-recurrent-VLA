# Task10-I compact indices, shared cell sort and fused gate kernels

The live World wrist gate reuses each Boolean compaction index, sorts cell codes
once and fuses World-point reconstruction and integer label propagation. Graph
frame selection, the 171 neighbor offsets, rare-point distance checks and the full
trace contract retain their existing behavior.

The runner loads frozen pre-change gate/kernel files from an isolated package.
Both versions use the same already prepared CUDA inputs and World views. It times
only `wrist_gate_tensor`, alternates before/after order in adjacent pairs and
checks every accepted mask and full trace against the saved CPU reference outside
the timer. Profiler audits run after timing and are excluded from latency stats.
It uses the existing `ExperimentLogger` and performs no artifact publishing.

```bash
WANDB_MODE=disabled python -m graph.experiments.wrist_gate_compact_fused_10i_20261104_v1.benchmark \
  --input ORIGINAL_CPU_TIMING_INPUT_DIR \
  --baseline outputs/task10i/wrist_gate_compact_fused_20261104_v1/baseline \
  --output NEW_OUTPUT_DIR --warmups 5 --repeats 30
python -m unittest graph.experiments.wrist_gate_compact_fused_10i_20261104_v1.test_contracts -v
```

The ignored baseline artifact contains the previous `tensor_gate.py`,
`tensor_gate_kernels.py`, unchanged `tensor_fusion.py` and its parent's
`before_identity.json`. Production callers continue through the existing
`CurrentGeometry._build_prepared -> prepare_tensor_gate -> wrist_gate_tensor` path.
