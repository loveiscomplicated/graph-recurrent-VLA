# Task10-I robot-filter-only live coverage audit

Checks the current default `CurrentGeometry` from native CUDA camera inputs to
its raw/clean CUDA-mask handoff. `_build_prepared` is intercepted to exclude the
wrist gate and downstream Graph. No real SAM2, grounding, policy or simulator
execution is performed; saved actual masks stand in for SAM2 device output.

The existing filter/preparation, caller and online-vision contracts are run first.
Two real 512 two-camera cases then verify one shared current-frame preparation,
one batched filter call, default compact32 clipping, reusable calibration rays,
fresh preparation, CUDA masks/depth, bitwise agreement with the direct filter,
no readback/global sync in the filter handoff and recovery after a vision error.
All 65,536 signed-int16 depth values compare the live shared lookup against the
filter's fused converter. A legacy CPU adapter receiving native CUDA RGB is
expected to raise and its exact error is retained as an integration finding.

Run from an isolated source snapshot in the existing remote CUDA environment:

```bash
PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m graph.experiments.robot_filter_live_audit_10i_20261104_v1.run \
  --output /unique/new/output
```

Uses `ExperimentLogger`; no production implementation, stopped experiment,
package, service, checkpoint or external publication is changed. This audit
does not produce a new latency result or establish full live-rollout execution.
