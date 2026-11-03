#!/bin/bash
set -euo pipefail
taskroot=/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1
taskpython=/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$taskroot/.execution"
"$taskpython" prepare.py > prepare.log 2>&1
"$taskpython" capture.py > capture.log 2>&1
"$taskpython" dino.py > dino.log 2>&1
"$taskpython" build_requests.py > requests.log 2>&1
"$taskpython" infer.py > infer.log 2>&1
"$taskpython" evaluate.py > evaluate.log 2>&1
"$taskpython" finalize.py > finalize.log 2>&1
