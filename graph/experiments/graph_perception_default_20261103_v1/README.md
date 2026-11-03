# Task 10-I: default Graph perception execution validation

No latency benchmark. `verify.py` enforces Vast instance 58752018 and validates
the saved real 512 hand-camera observation against the old dense fused feature
backend and original CPU Graph packet builder. It records through the shared
`ExperimentLogger`; use `WANDB_MODE=disabled`. No publishing is performed.

```bash
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled \
python -m graph.experiments.graph_perception_default_20261103_v1.verify \
  --inputs /path/to/recorded/inputs \
  --assets /path/to/robot/assets \
  --schema /path/to/normalizer.json \
  --output /path/to/new/output
```

Inputs are `hand_camera.npz`, `robot.npz`, and `masks.npz`. The schema file
contains `schema.fields` for Robot76. The final Graph check uses the real hand
view and empty base support with identical prepared masks/gate, bypassing robot
filtering. It is not a full two-camera rollout or policy-success evaluation.
CUDA synthetic/edge-case contracts also live in
`visual.runtime.test.test_graph_perception`. Full execution evidence is in
[`10i_graph_perception_default_20261103.md`](../../../docs/visual-graph-integration/10i_graph_perception_default_20261103.md).
