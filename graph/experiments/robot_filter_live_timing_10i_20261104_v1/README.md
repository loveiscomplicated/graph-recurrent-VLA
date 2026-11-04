# Task10-I live robot filter timing, 2026-11-04

Measurement only. `run.py` compares the byte-frozen historical 7.159 ms filter,
the current filter under the identical input/timing boundary, and the actual
default `CurrentGeometry.build` robot-filter path. It intercepts the clean-mask
handoff before wrist gate/Shape-First/Graph. A separate full-screen classification
case represents robot work required by correction; CPU correction is excluded.

The paired test uses the original saved 512 x 512 scene, two cameras, actual SAM2
masks, three warmups and fifteen timed repetitions. It verifies the original
input fingerprints and exact clean masks outside each synchronized wall timer.
Native GPU camera upload is outside the live timer, matching the current live
input contract; preparation, current snapshots and filtering are inside.

A separate real causal SAM2 video trace uses recorded RGB-derived prompts and
the default native tensor adapter. No detection API, simulator, action, gate or
Graph runs. Its visible filter contribution is measured from same-call wall
spans and is explicitly distinct from standalone filter latency. Video frames
6 through 20 supply fifteen steady samples; frame zero initializes and frames
1 through 5 warm the causal state. The current default FP32/no-autocast behavior
is retained. All fresh clean masks must exactly match the frozen filter applied
to the same fresh raw masks.

Run on the existing CUDA host in an isolated source snapshot:

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m graph.experiments.robot_filter_live_timing_10i_20261104_v1.run \
  --output /absolute/path/to/new-output-directory
```

The existing shared `ExperimentLogger` records a new evaluation identity and
local artifact sidecars. No production change, training or artifact publishing
is performed. Full results belong in
`docs/visual-graph-integration/10i_robot_filter_live_timing_20261104.md`.
