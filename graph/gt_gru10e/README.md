# Task 10-E

Read-only checkpoint recovery: `.venv/bin/python -m graph.gt_gru10e.inventory`.
The selected HF revision and every downloaded SHA are retained in
`outputs/task10e/download_inventory.json`. Hub authentication uses the existing
local Hugging Face cache; credentials are never copied into artifacts.

`benchmark.py` must run from an isolated copy of Task10D source with the
archived official `recurrent/` source restored there. It imports that original
LearnedPolicy, encoder, corruption, normalizer, action decoder and TensorPolicy.
It does not replace the checkout's Recurrent package. Example on the existing
server:

```
WANDB_MODE=disabled OMP_NUM_THREADS=1 /venv/main/bin/python -m graph.gt_gru10e.benchmark \
  --root /workspace/task10d_geometry_reach \
  --inputs /workspace/task10e_gt_gru/inputs \
  --out /workspace/task10e_gt_gru/benchmark
```

Use a new output path for each attempt. `--smoke` runs seed0 on the first two
frozen Final cases after official fixed-seed and streaming contract checks.
The 130-action Task10D primary window and 250-action full-task window are kept
separate. Reach is latched before later natural close/manipulation. Targets
are metrics only; inputs are the original current-state 38D features.
