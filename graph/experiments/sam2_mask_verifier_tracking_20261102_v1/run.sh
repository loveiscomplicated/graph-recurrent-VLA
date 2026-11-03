#!/bin/bash
set -euo pipefail
taskroot=/workspace/task10i_stage4a2_sam2_mask_verifier_tracking_20261102_v1
taskpython=/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$taskroot/.execution"
trap '"$taskpython" stop.py' ERR
for stage in prepare static_masks infer capture track evaluate finalize; do
  "$taskpython" "$stage.py" > "$stage.log" 2>&1
done
