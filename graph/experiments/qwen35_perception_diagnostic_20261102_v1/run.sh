#!/bin/bash
set -eu
task_root=/workspace/task10i_stage4a2_qwen35_perception_diagnostic_20261102_v1
task_python=/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python
cd "$task_root/.execution"
export PYTHONDONTWRITEBYTECODE=1
"$task_python" prepare.py > "$task_root/prepare.log" 2>&1
"$task_python" infer.py > "$task_root/infer.log" 2>&1
"$task_python" finalize.py > "$task_root/finalize.log" 2>&1
